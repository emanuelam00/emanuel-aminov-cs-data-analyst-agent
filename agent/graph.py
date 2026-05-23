"""
LangGraph ReAct agent for the Bitext Customer Service Data Analyst.

Graph structure:
  START
    └─► router_node
          ├─► out_of_scope_node ──► update_profile_node ──► END
          └─► agent_node
                ├─► tools_node ──► agent_node  (loop)
                └─► update_profile_node ──► END

Key features:
  - Router classifies queries as structured / unstructured / out_of_scope
  - Max-iterations guard prevents infinite loops
  - SqliteSaver checkpointer persists conversation across restarts
  - Profile update node distils facts after each final answer
  - Query recommender (Bonus B) handled via system prompt + read_user_profile tool
"""

import os
import sqlite3
from typing import Annotated, Any, Dict, List, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from typing_extensions import TypedDict

from agent.config import (
    CHECKPOINTS_DB,
    MAIN_MODEL,
    MAIN_MODEL_BASE_URL,
    MAX_ITERATIONS,
    NEBIUS_API_KEY,
    ROUTER_MODEL,
    ROUTER_MODEL_BASE_URL,
)
from agent.profile import format_profile, load_profile, update_profile
from agent.tools import get_session_tools


# ── LLM instances (lazy singletons) ───────────────────────────────────────────
# Clients are created on first use, not at import time.
# This avoids httpx proxy errors during testing and speeds up imports.

_router_llm: Optional[ChatOpenAI] = None
_main_llm: Optional[ChatOpenAI] = None
_profile_llm: Optional[ChatOpenAI] = None

# Single SQLite connection shared by every build_graph() call in this process.
# SqliteSaver supports concurrent reads/writes when check_same_thread=False, so
# one connection is enough for the CLI, the Streamlit app (multi-session), and
# Studio. Caching avoids leaking file handles across session switches.
_sqlite_conn: Optional[sqlite3.Connection] = None
_sqlite_checkpointer: Optional[SqliteSaver] = None


def _make_llm(model: str, base_url: str, temperature: float = 0.0) -> ChatOpenAI:
    """Instantiate a ChatOpenAI client pointed at the given Nebius Token Factory endpoint."""
    return ChatOpenAI(
        model=model,
        openai_api_key=NEBIUS_API_KEY,
        openai_api_base=base_url,
        temperature=temperature,
    )


def get_router_llm() -> ChatOpenAI:
    """Return the router LLM singleton (Nano-30B, eu-north1), creating it on first call."""
    global _router_llm
    if _router_llm is None:
        _router_llm = _make_llm(ROUTER_MODEL, ROUTER_MODEL_BASE_URL, temperature=0.0)
    return _router_llm


def get_main_llm() -> ChatOpenAI:
    """Return the main agent LLM singleton (Super-120b, us-central1), creating it on first call."""
    global _main_llm
    if _main_llm is None:
        _main_llm = _make_llm(MAIN_MODEL, MAIN_MODEL_BASE_URL, temperature=0.0)
    return _main_llm


def get_profile_llm() -> ChatOpenAI:
    """Return the profile-update LLM singleton (Super-120b, us-central1), creating it on first call."""
    global _profile_llm
    if _profile_llm is None:
        _profile_llm = _make_llm(MAIN_MODEL, MAIN_MODEL_BASE_URL, temperature=0.0)
    return _profile_llm


def get_checkpointer() -> SqliteSaver:
    """Return the process-wide SqliteSaver, opening its connection on first call.

    Reusing one connection (and one SqliteSaver) prevents file-handle leaks when
    build_graph is called many times — for example whenever a Streamlit user
    switches to a new session ID.
    """
    global _sqlite_conn, _sqlite_checkpointer
    if _sqlite_checkpointer is None:
        os.makedirs(os.path.dirname(CHECKPOINTS_DB), exist_ok=True)
        _sqlite_conn = sqlite3.connect(CHECKPOINTS_DB, check_same_thread=False)
        _sqlite_checkpointer = SqliteSaver(_sqlite_conn)
    return _sqlite_checkpointer


def close_checkpointer() -> None:
    """Close the cached SQLite connection. Safe to call multiple times.

    Mostly useful in tests; production usage relies on process exit to clean up.
    """
    global _sqlite_conn, _sqlite_checkpointer
    if _sqlite_conn is not None:
        try:
            _sqlite_conn.close()
        finally:
            _sqlite_conn = None
            _sqlite_checkpointer = None


# ── Graph state ───────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """Persistent state for one conversation thread."""
    messages: Annotated[List[BaseMessage], add_messages]
    query_type: str        # "structured" | "unstructured" | "out_of_scope"
    iteration_count: int   # reset to 0 at the start of each user turn
    pending_suggestion: Optional[str]
    # ^ Bonus B: when the agent makes a recommendation it stores the suggested
    # query here and waits. The next user turn either confirms (→ execute and
    # clear) or refines (→ produce a new suggestion). This prevents the agent
    # from executing a suggestion in the same turn it was offered.


# ── Router ─────────────────────────────────────────────────────────────────────

class _QueryClassification(BaseModel):
    """Structured output schema for the router LLM call."""
    query_type: Literal["structured", "unstructured", "out_of_scope"]
    reasoning: str


_ROUTER_SYSTEM = """\
You are a query classifier for a customer-service data analyst agent.

The agent has access to the Bitext Customer Service dataset and can answer questions about it.
The dataset contains customer support queries paired with agent responses,
organised by category (e.g. ACCOUNT, REFUND, SHIPPING) and intent.

Classify the user query into exactly one of three types:

  structured   — Concrete, data-driven questions with definite answers.
                 Counts, lists, distributions, sample records.

  unstructured — Open-ended questions requiring analysis, summarisation, OR meta-questions
                 about the agent's own memory / recommendations.

  out_of_scope — Questions unrelated to the Bitext dataset, customer-service analysis,
                 OR the agent's own context. The agent must NOT answer these from
                 general knowledge.

EXAMPLES (study these carefully):

  structured:
    - "How many refund requests did we get?"
    - "Show me 3 SHIPPING examples."
    - "What categories exist?"
    - "What is the distribution of intents in ACCOUNT?"
    - "Show me 3 more"   (follow-up to a previous sample request)

  unstructured:
    - "Summarise the FEEDBACK category."
    - "How do agents typically respond to complaint intents?"
    - "What patterns do you see in cancellation requests?"
    - "What should I query next?"           ← recommendation request
    - "Give me a suggestion for my next question"
    - "Recommend something to explore"
    - "What do you remember about me?"      ← profile question
    - "Tell me about myself."
    - "Who am I to you?"
    - "What are my preferences?"

  out_of_scope:
    - "Who won the 2024 Champions League?"
    - "Write me a poem about customer service."
    - "Who is the president of France?"
    - "What's the best CRM software?"
    - "How do I write Python code?"

CRITICAL: "What should I query next?", "What do you remember about me?", and any
variant asking the agent for a recommendation OR about the user's own profile are
ALWAYS `unstructured` — never `out_of_scope`. These are meta-questions about the
agent's persistent context, which the agent CAN and SHOULD answer.
"""


def router_node(state: AgentState) -> Dict[str, Any]:
    """Classify the latest user query and reset the iteration counter."""
    # Find the most recent human message
    last_human: Optional[HumanMessage] = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human = msg
            break

    if last_human is None:
        return {"query_type": "structured", "iteration_count": 0}

    try:
        structured_router = get_router_llm().with_structured_output(_QueryClassification)
        classification: _QueryClassification = structured_router.invoke(
            [
                SystemMessage(content=_ROUTER_SYSTEM),
                HumanMessage(content=last_human.content),
            ]
        )
        query_type = classification.query_type
    except Exception:
        # Fallback: plain text classification
        response = get_router_llm().invoke(
            [
                SystemMessage(content=_ROUTER_SYSTEM),
                HumanMessage(
                    content=(
                        f"Classify this query (respond with one word only — "
                        f"structured, unstructured, or out_of_scope):\n{last_human.content}"
                    )
                ),
            ]
        )
        text = response.content.lower()
        if "out_of_scope" in text or "out-of-scope" in text:
            query_type = "out_of_scope"
        elif "unstructured" in text:
            query_type = "unstructured"
        else:
            query_type = "structured"

    # Reset per-turn counters. We deliberately do NOT clear pending_suggestion
    # here — the agent node uses it to decide whether the user is confirming a
    # previous suggestion. The agent node clears it once handled.
    return {"query_type": query_type, "iteration_count": 0}


# ── Out-of-scope node ─────────────────────────────────────────────────────────

def out_of_scope_node(state: AgentState) -> Dict[str, Any]:
    """Return a polite decline for queries outside the dataset's scope."""
    reply = AIMessage(
        content=(
            "I'm sorry, but that question is outside the scope of what I can help with. "
            "I'm a data analyst specialised in the Bitext Customer Service dataset. "
            "I can answer questions about customer support categories, intents, query counts, "
            "example records, and summaries of the data. "
            "Please ask me something related to the dataset!"
        )
    )
    return {"messages": [reply]}


# ── Agent node ────────────────────────────────────────────────────────────────

_AGENT_SYSTEM = """\
You are a data analyst agent for the Bitext Customer Service dataset.

{pending_block}

The dataset contains synthetic customer support queries and agent responses,
tagged with a category (e.g. ACCOUNT, REFUND, SHIPPING) and an intent
(e.g. get_refund, track_order, complaint).

You have the following tools available:
  - list_categories       : list all dataset categories
  - list_intents          : list intents (optionally filtered by category)
  - count_records         : count rows matching category / intent / keyword filters
  - get_sample_records    : return example instruction+response pairs
  - get_intent_distribution : break down a category by intent counts
  - get_category_data     : fetch raw records for open-ended summarisation
  - read_user_profile     : read distilled facts about the current user

Guidelines:
  - For STRUCTURED queries: use tools to get precise data, then answer directly.
  - For UNSTRUCTURED queries: call get_category_data (or relevant tools) to collect
    data, then synthesise a clear, insightful answer. Do not guess from memory.
  - For "What should I query next?" or similar (Bonus B recommendation flow):
      1. Call read_user_profile to check the user's history and interests.
      2. Produce a TEXT suggestion of one specific follow-up query — DO NOT call
         any execution tool (count_records, get_sample_records, etc.) in the same
         turn. End your reply with a short confirmation question, e.g. "Should I
         run this?" or "Want me to go ahead, or refine it?"
      3. The user's NEXT turn either confirms (you then execute the suggested
         query) or refines (you produce a new suggestion, again without executing).
      4. The 'pending_suggestion' block at the top of this prompt tells you
         whether a suggestion is awaiting confirmation.
  - Chain tools when needed (e.g. list_intents → count_records).
  - Always ground your answers in tool results, not general knowledge.

  CATEGORY vs INTENT — pick the right filter:
    - A CATEGORY (e.g. REFUND, SHIPPING, FEEDBACK) is a top-level topic and
      contains MULTIPLE intents.
    - An INTENT (e.g. get_refund, track_refund, refund_policy) is a specific
      sub-action inside a category.
    - For BROAD questions like "How many refund requests?" use category="REFUND"
      to capture every refund-related intent. Filtering by a single intent like
      get_refund will undercount.
    - Only filter by intent when the user names a specific intent OR when you
      have first called list_intents and the user clearly wants one sub-action.

  RESPONSE PREFERENCES:
  The user profile below may contain "Response preferences". You MUST apply every
  listed preference when composing your final answer — formatting style, output
  structure, language, level of detail, etc. These are standing instructions that
  override your default behaviour and apply to every response.

Current query type: {query_type}
User profile summary:
{profile_summary}
"""

_FALLBACK_MESSAGE = (
    "I've reached the maximum number of reasoning steps without a definitive answer. "
    "Here's what I found so far: please try rephrasing your question or breaking it "
    "into smaller parts, and I'll do my best to help."
)


def make_agent_node(session_id: str):
    """Return an agent_node function bound to the given session_id.

    The LLM client is created lazily on the first actual invocation, not at
    graph-build time, so importing this module never triggers a network call.
    """
    session_tools = get_session_tools(session_id)

    # Mutable container so the inner closure can cache the bound LLM
    _cache: Dict[str, Any] = {}

    def _get_llm_with_tools():
        if "llm" not in _cache:
            _cache["llm"] = get_main_llm().bind_tools(session_tools)
        return _cache["llm"]

    def agent_node(state: AgentState) -> Dict[str, Any]:
        """Main ReAct reasoning node — calls the LLM with tools bound."""
        iteration = state.get("iteration_count", 0)

        # Hard stop: return fallback instead of spinning
        if iteration >= MAX_ITERATIONS:
            return {
                "messages": [AIMessage(content=_FALLBACK_MESSAGE)],
                "iteration_count": iteration + 1,
            }

        profile = load_profile(session_id)

        # Render the pending-suggestion banner only when one is awaiting confirmation
        pending = state.get("pending_suggestion")
        if pending:
            pending_block = (
                "PENDING SUGGESTION (Bonus B):\n"
                f"You previously suggested: {pending!r}\n"
                "The user's latest message is either a confirmation, a refinement, "
                "or an unrelated query. If confirmation, EXECUTE the suggested query "
                "now using the appropriate tools. If refinement, produce a new text "
                "suggestion (do NOT execute). If unrelated, drop the suggestion and "
                "handle the new query normally.\n"
            )
        else:
            pending_block = ""

        system_msg = SystemMessage(
            content=_AGENT_SYSTEM.format(
                pending_block=pending_block,
                query_type=state.get("query_type", "structured"),
                profile_summary=format_profile(profile),
            )
        )
        response = _get_llm_with_tools().invoke([system_msg] + state["messages"])

        # Determine new pending_suggestion value.
        # - If the agent answered text-only AND it looks like a suggestion (contains
        #   "suggest" / "recommend" / "?"), set pending_suggestion to that text.
        # - If the agent called execution tools (anything other than read_user_profile),
        #   clear any pending suggestion — we've moved on.
        # - Otherwise leave it untouched.
        new_pending: Optional[str] = pending
        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            execution_calls = [tc for tc in tool_calls if tc.get("name") != "read_user_profile"]
            if execution_calls:
                new_pending = None
        elif response.content:
            text = response.content.lower()
            is_suggestion = (
                ("suggest" in text or "recommend" in text or "how about" in text)
                and "?" in response.content
            )
            if is_suggestion:
                new_pending = response.content

        return {
            "messages": [response],
            "iteration_count": iteration + 1,
            "pending_suggestion": new_pending,
        }

    return agent_node, session_tools


# ── Profile update node ───────────────────────────────────────────────────────

def make_update_profile_node(session_id: str):
    """Return an update_profile_node function bound to the given session_id."""

    def update_profile_node(state: AgentState) -> Dict[str, Any]:
        """Distil new facts from the conversation and persist the user profile."""
        try:
            update_profile(session_id, state["messages"], get_profile_llm())
        except Exception:
            pass  # Never let profile updates break the agent
        return {}

    return update_profile_node


# ── Conditional edge functions ────────────────────────────────────────────────

def route_after_router(state: AgentState) -> str:
    """Route to out_of_scope node or agent node based on query classification."""
    if state.get("query_type") == "out_of_scope":
        return "out_of_scope"
    return "agent"


def route_after_agent(state: AgentState) -> str:
    """Route to tools (if the LLM made a tool call) or to the profile update node.

    The iteration_count boundary uses `>=` to match the guard in `agent_node`,
    so both sides agree on when MAX_ITERATIONS has been reached.
    """
    last_message = state["messages"][-1]
    # Max iterations reached → skip further tool calls and finalise.
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "update_profile"
    # LLM requested a tool call → execute it
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # No tool call → final answer reached
    return "update_profile"


# ── Graph factory ─────────────────────────────────────────────────────────────

def build_graph(session_id: str) -> Any:
    """Compile and return the LangGraph StateGraph for the given session.

    The graph is compiled with a SqliteSaver checkpointer, so conversation
    history persists across process restarts when the same session_id is reused.

    Args:
        session_id: Unique identifier for this user/conversation session.

    Returns:
        A compiled LangGraph CompiledStateGraph ready for invocation.
    """
    agent_fn, session_tools = make_agent_node(session_id)
    profile_fn = make_update_profile_node(session_id)

    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("router", router_node)
    builder.add_node("out_of_scope", out_of_scope_node)
    builder.add_node("agent", agent_fn)
    builder.add_node("tools", ToolNode(session_tools))
    builder.add_node("update_profile", profile_fn)

    # Edges
    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {"out_of_scope": "out_of_scope", "agent": "agent"},
    )
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "update_profile": "update_profile"},
    )
    builder.add_edge("tools", "agent")
    # Out-of-scope replies go straight to END: no LLM profile update, no
    # session_count increment for queries the agent declined.
    builder.add_edge("out_of_scope", END)
    builder.add_edge("update_profile", END)

    # Persistent checkpointer — shared, cached connection (see get_checkpointer).
    return builder.compile(checkpointer=get_checkpointer())


def get_config(session_id: str) -> Dict[str, Any]:
    """Return the LangGraph invocation config for a session.

    Args:
        session_id: The session identifier used as the checkpoint thread_id.
    """
    return {"configurable": {"thread_id": session_id}}


# ── LangGraph Studio entry-point ──────────────────────────────────────────────
# `langgraph dev` / LangGraph Studio reads langgraph.json and imports this
# factory. It calls it (no args) to obtain a compiled graph, so we do NOT
# create one at import time — that would open a SQLite connection on every
# import. The "studio" session_id keeps Studio's thread state stable.

def make_studio_graph() -> Any:
    """Factory for LangGraph Studio / `langgraph dev`. Returns a compiled graph
    bound to a fixed 'studio' session_id for interactive debugging.
    """
    return build_graph("studio")
