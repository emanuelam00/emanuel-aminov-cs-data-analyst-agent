# Implementation Plan — Customer Service Data Analyst Agent

## Overview

A LangGraph ReAct agent that answers structured and unstructured questions about the
Bitext Customer Service dataset, with persistent conversation memory, a per-user profile,
a FastMCP server, a CLI, a Streamlit UI (Bonus A), and a query recommender (Bonus B).

---

## Model Choice

Two-model split using the Nebius Token Factory models available:

| Role | Model | Reason |
|------|-------|--------|
| **Main agent** | `nvidia/Nemotron-3-Super-120b-a12b` | 120B hybrid MoE, explicitly described as "optimized for efficient multi-agent AI and complex reasoning tasks". 127 Tok/s — fast enough for interactive use. Handles tool calls, multi-step reasoning, and open-ended summarisation well. |
| **Router** | `nvidia/Nemotron-3-Nano-30B-A3B` | Compact MoE, 60 Tok/s, $0.06/$0.24 per 1M tokens. Query classification is a single structured-output call — a nano model handles it fine and costs ~5× less than the Super model. |

The Ultra-253B is overkill at 25 Tok/s and 2× the cost; too slow for a conversational agent.
The Nano-Omni is intriguing ("agentic AI") but its nano size makes multi-step tool chains unreliable.

Both models are on Nebius Token Factory and share the same OpenAI-compatible base URL,
so only one `NEBIUS_API_KEY` is needed.

---

## Python Version

**Use Python 3.12 (conda environment).**

- `langgraph >= 0.2` and `langchain >= 0.3` are developed and tested primarily on 3.11+.
  SqliteSaver in particular uses `sqlite3` features that are more stable on 3.11+.
- Python 3.12 has noticeably better error messages and ~5 % faster CPython interpreter.
- The code will be written to be compatible with 3.9+ (no `match` statements, no `X | Y`
  union syntax — we use `Optional[X]` and `Union[X, Y]` from `typing` throughout), but
  **3.12 is the recommended and tested runtime**.

Setup command:
```bash
conda create -n cs-agent python=3.12 -y
conda activate cs-agent
pip install -r requirements.txt
```

---

## Repository Layout

```
customer-service-agent/
│
├── agent/
│   ├── __init__.py
│   ├── config.py        # Constants: model IDs, paths, MAX_ITERATIONS
│   ├── tools.py         # All @tool functions with Pydantic input schemas
│   ├── graph.py         # LangGraph StateGraph (router + ReAct + memory)
│   └── profile.py       # Per-user JSON profile read/write
│
├── data/
│   └── bitext_dataset.csv   # Downloaded by download_data.py
│
├── checkpoints/
│   └── memory.db            # SQLite conversation checkpoints (auto-created)
│
├── profiles/
│   └── {session_id}.json    # Per-user profile files (auto-created)
│
├── download_data.py     # One-time dataset setup script
├── mcp_server.py        # FastMCP server (Task 3)
├── main.py              # CLI entry-point (Task 1 + Task 2)
├── app.py               # Streamlit chat UI (Bonus A)
│
├── requirements.txt
├── .env.example
├── PLAN.md              # ← this file
└── README.md
```

---

## Component-by-Component Plan

### 1. `agent/config.py`

Single source of truth for all tunable values:
- `NEBIUS_API_KEY` (single key, used for both models)
- `MAIN_MODEL = "nvidia/nemotron-3-super-120b-a12b"` with `MAIN_MODEL_BASE_URL` (us-central1 endpoint)
- `ROUTER_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"` with `ROUTER_MODEL_BASE_URL` (default eu-north1 endpoint)
- `MAX_ITERATIONS = 12`
- Path constants: `DATASET_PATH`, `CHECKPOINTS_DB`, `PROFILES_DIR`

---

### 2. `agent/tools.py` — Data Analysis Tools

Seven tools, each with a `BaseModel` Pydantic input schema and a typed return value.

| Tool | Input | Output | Purpose |
|------|-------|--------|---------|
| `list_categories` | — | `list[str]` | All unique categories |
| `list_intents` | `category?` | `list[str]` | Intents, optionally filtered |
| `count_records` | `category?, intent?, keyword?` | `int` | Row count with filters |
| `get_sample_records` | `n, category?, intent?, keyword?` | `list[dict]` | Sample instruction+response pairs |
| `get_intent_distribution` | `category` | `dict[str,int]` | Counts per intent within a category |
| `get_category_data` | `category, max_rows?=50` | `list[dict]` | Bulk data for summarisation |
| `read_user_profile` | — | `dict` | Current user's persisted profile |

A module-level singleton loads the CSV once and reuses the DataFrame for all calls.

---

### 3. `agent/profile.py` — User Profile

Stores a JSON file at `profiles/{session_id}.json` with:
```json
{
  "name": null,
  "frequent_topics": [],
  "preferred_categories": [],
  "notes": [],
  "session_count": 0
}
```

Two public functions:
- `load_profile(session_id) -> dict` — reads or creates the file
- `update_profile(session_id, messages) -> None` — calls the LLM with the last N messages
  and a structured prompt to extract/update facts; writes the result back to disk

`update_profile` is called after every agent turn that produces a final answer.

---

### 4. `agent/graph.py` — LangGraph StateGraph

#### State

```python
class AgentState(TypedDict):
    messages:        Annotated[list, add_messages]
    query_type:      str          # "structured" | "unstructured" | "out_of_scope"
    iteration_count: int
    session_id:      str
```

#### Nodes

| Node | Role |
|------|------|
| `router_node` | Calls the LLM with structured output to classify the query |
| `out_of_scope_node` | Appends a polite decline message; no tool calls |
| `agent_node` | Main ReAct LLM node; injects system prompt + user profile; increments `iteration_count` |
| `tools_node` | `ToolNode(tools)` — executes whichever tool the LLM called |
| `update_profile_node` | Calls `update_profile()` asynchronously after final answer |

#### Edges

```
START
  └─► router_node
        ├─► out_of_scope_node ──► update_profile_node ──► END
        └─► agent_node
              ├─► tools_node ──► agent_node   (loop while tool calls exist)
              └─► update_profile_node ──► END  (when no tool call OR iteration_count ≥ MAX)
```

#### Max-iterations guard

Inside `agent_node`: if `iteration_count >= MAX_ITERATIONS`, skip the LLM call and
append a hard-coded `AIMessage` fallback, then route to `update_profile_node → END`.

#### Memory

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect(CHECKPOINTS_DB, check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph = builder.compile(checkpointer=checkpointer)
```

Each session is invoked with `config = {"configurable": {"thread_id": session_id}}`.
Restarting the process and passing the same `session_id` fully restores the conversation.

---

### 5. `mcp_server.py` — FastMCP Server (Task 3)

```python
from fastmcp import FastMCP
mcp = FastMCP("Customer Service Analyst")
```

Exposes four tools directly (no LangGraph dependency, pure data functions):
`list_categories`, `count_records`, `get_sample_records`, `get_intent_distribution`.

Run: `python mcp_server.py` (stdio transport, compatible with Claude Desktop and MCP CLI).

---

### 6. `main.py` — CLI

```
python main.py [--session SESSION_ID]
```

- If `--session` is omitted, generates a UUID as the session ID.
- Drops into an interactive `input()` loop.
- Streams the graph and prints each `AIMessage` (including tool calls) and each
  `ToolMessage` (observation) as they arrive, so the reasoning chain is visible.
- Handles `KeyboardInterrupt` / `exit` / `quit` gracefully.
- Displays the active session ID on start so the user can resume it later.

---

### 7. `app.py` — Streamlit UI (Bonus A)

- `st.sidebar` contains a text input for Session ID and a "New Session" button.
- Main area is a chat interface using `st.chat_message`.
- Tool calls are shown in `st.expander("🔧 Reasoning steps")` blocks — collapsed by
  default but inspectable.
- Session state (`st.session_state`) holds the active session ID and the display
  history; actual memory lives in SQLite.

---

### 8. Query Recommender (Bonus B)

Handled inside the existing ReAct loop — no extra graph node required.

When the agent detects a recommendation request (router classifies it as `unstructured`,
and the system prompt instructs the agent to handle "what should I query next?" phrases),
the agent:
1. Calls `read_user_profile()` to get the user's history and interests.
2. Produces a suggestion as a plain `AIMessage` (no tool execution).
3. Waits. The conversation history persists via SQLite.
4. On the user's next turn (confirm / refine), the agent either executes the query or
   adjusts the suggestion.

No special state flag needed — the LLM handles the confirmation dialogue naturally via
conversation context.

---

## Grading Checklist

| Criterion | Where implemented | Points |
|-----------|-------------------|--------|
| Query router | `graph.py` → `router_node` | 15 |
| Tools with Pydantic schemas | `tools.py` | 15 |
| Multi-step reasoning | ReAct loop in `graph.py` | 10 |
| CLI with reasoning output | `main.py` | 5 |
| Max iterations fallback | `agent_node` guard | 5 |
| Conversation memory (SQLite) | `graph.py` + SqliteSaver | 20 |
| User profile | `profile.py` + `update_profile_node` | 10 |
| MCP server (≥3 tools) | `mcp_server.py` | 20 |
| Streamlit UI | `app.py` | +10 |
| Query recommender | system prompt + `read_user_profile` tool | +10 |
| **Total** | | **120** |
