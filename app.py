"""
Streamlit chat UI — Bonus A.

Run:
    streamlit run app.py

Features:
  - Chat interface with full message history
  - Collapsible "Reasoning steps" expander showing every tool call + result
  - Sidebar session ID input for creating or resuming named conversations
  - Session state holds the display history; actual memory lives in SQLite
"""

import json
import uuid
from typing import Any, Dict, List

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import build_graph, get_config

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CS Data Analyst Agent",
    page_icon="📊",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 CS Data Analyst")
    st.markdown("Powered by **Nemotron-3-Super-120b**")
    st.divider()

    st.subheader("Session")
    session_input = st.text_input(
        "Session ID",
        value=st.session_state.get("session_id", ""),
        placeholder="e.g. alice, demo-session, …",
        help="Use the same ID to resume a previous conversation after a restart.",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Load session", use_container_width=True):
            sid = session_input.strip() or str(uuid.uuid4())[:8]
            if sid != st.session_state.get("session_id"):
                st.session_state["session_id"] = sid
                st.session_state["display_history"] = []
                st.session_state["graph"] = None
                st.rerun()
    with col2:
        if st.button("✦ New session", use_container_width=True):
            st.session_state["session_id"] = str(uuid.uuid4())[:8]
            st.session_state["display_history"] = []
            st.session_state["graph"] = None
            st.rerun()

    if "session_id" in st.session_state:
        st.info(f"Active: **{st.session_state['session_id']}**")

    # Clear-chat clears only the on-screen history. Persistent SQLite memory
    # for the session is unaffected — to wipe that, use `python main.py --reset`.
    if st.button("🗑 Clear chat (display only)", use_container_width=True):
        st.session_state["display_history"] = []
        st.rerun()

    st.divider()
    st.markdown(
        "**Sample queries**\n"
        "- What categories exist?\n"
        "- How many refund requests?\n"
        "- Show me 5 SHIPPING examples\n"
        "- Summarise the FEEDBACK category\n"
        "- What should I query next?\n"
        "- What do you remember about me?"
    )

# ── Session state initialisation ──────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())[:8]

if "display_history" not in st.session_state:
    st.session_state["display_history"] = []

if "graph" not in st.session_state or st.session_state["graph"] is None:
    st.session_state["graph"] = build_graph(st.session_state["session_id"])

# ── Display message history ───────────────────────────────────────────────────

st.title("Customer Service Data Analyst")
st.caption(f"Session: `{st.session_state['session_id']}`")

for entry in st.session_state["display_history"]:
    role = entry["role"]
    with st.chat_message(role):
        # Reasoning steps (tool calls + results) in a collapsible block
        if entry.get("reasoning_steps"):
            with st.expander("🔧 Reasoning steps", expanded=False):
                for step in entry["reasoning_steps"]:
                    if step["type"] == "tool_call":
                        st.markdown(
                            f"**▶ Tool call:** `{step['name']}`  \n"
                            f"```json\n{json.dumps(step['args'], indent=2)}\n```"
                        )
                    elif step["type"] == "observation":
                        preview = step["content"][:500]
                        if len(step["content"]) > 500:
                            preview += "\n…(truncated)"
                        st.markdown(
                            f"**◀ Result:** `{step['name']}`  \n"
                            f"```\n{preview}\n```"
                        )
        if entry.get("content"):
            st.markdown(entry["content"])

# ── Chat input ────────────────────────────────────────────────────────────────

user_input = st.chat_input("Ask something about the Bitext dataset…")

if user_input:
    # Show user message immediately
    st.session_state["display_history"].append(
        {"role": "user", "content": user_input, "reasoning_steps": []}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run the agent and collect the streamed events
    graph = st.session_state["graph"]
    config = get_config(st.session_state["session_id"])

    reasoning_steps: List[Dict[str, Any]] = []
    final_answer: str = ""

    with st.chat_message("assistant"):
        # Live reasoning panel: expanded while the agent works, collapses to a
        # neat summary once the answer is rendered.
        live_reasoning = st.expander("🔧 Reasoning (live)", expanded=True)
        answer_placeholder = st.empty()

        # Stream agent events
        events = graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="updates",
        )

        def _render_step(step: Dict[str, Any]) -> None:
            """Render one tool call or observation into the live panel."""
            if step["type"] == "tool_call":
                live_reasoning.markdown(
                    f"**▶ Tool call:** `{step['name']}`  \n"
                    f"```json\n{json.dumps(step['args'], indent=2)}\n```"
                )
            elif step["type"] == "observation":
                preview = step["content"][:500]
                if len(step["content"]) > 500:
                    preview += "\n…(truncated)"
                live_reasoning.markdown(
                    f"**◀ Result:** `{step['name']}`  \n"
                    f"```\n{preview}\n```"
                )

        with st.spinner("Thinking…"):
            for event in events:
                for node_name, node_output in event.items():
                    if node_output is None:
                        continue
                    messages = node_output.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for tc in msg.tool_calls:
                                step = {
                                    "type": "tool_call",
                                    "name": tc["name"],
                                    "args": tc.get("args", {}),
                                }
                                reasoning_steps.append(step)
                                _render_step(step)
                        elif isinstance(msg, ToolMessage):
                            step = {
                                "type": "observation",
                                "name": msg.name,
                                "content": str(msg.content),
                            }
                            reasoning_steps.append(step)
                            _render_step(step)
                        elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                            final_answer = msg.content

        # Render final answer
        if not final_answer:
            final_answer = "(No answer produced.)"
        answer_placeholder.markdown(final_answer)

    # Persist to display history
    st.session_state["display_history"].append({
        "role": "assistant",
        "content": final_answer,
        "reasoning_steps": reasoning_steps,
    })
