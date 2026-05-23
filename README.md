# Customer Service Data Analyst Agent

A LangGraph ReAct agent that answers structured and open-ended questions about the
**Bitext Customer Service** dataset, with persistent conversation memory, a per-user
profile, a FastMCP server, a CLI, and a Streamlit UI.

---

## Setup (< 5 minutes)

### Prerequisites — conda (recommended)

The setup steps below use **conda** to create an isolated Python 3.12 environment.
If you don't have conda installed, grab it with one of these:

> **This step is not mandatory.** The code is compatible with Python 3.9+ — if you already
> have a suitable Python environment you can skip the `conda` setup in section 1 and go straight to
> `pip install -r requirements.txt` after cloning. Using Python 3.11 or 3.12 is still recommended for
> the most stable experience with `langgraph`'s SqliteSaver.

```bash
# macOS (Homebrew)
brew install miniconda

# macOS / Linux (official installer)
curl -fsSLo miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-$(uname)-$(uname -m).sh
bash miniconda.sh
```

### 1. Clone and create the environment

```bash
git clone https://github.com/emanuelam00/emanuel-aminov-cs-data-analyst-agent.git
cd emanuel-aminov-cs-data-analyst-agent

conda create -n cs-agent python=3.12 -y
conda activate cs-agent
pip install -r requirements.txt
```

### 2. Set your Nebius API key

```bash
cp .env.example .env
# Edit .env and paste your NEBIUS_API_KEY
```

Get your key at [tokenfactory.nebius.com](https://tokenfactory.nebius.com/).

### 3. Download the dataset

```bash
python download_data.py
```

This downloads the Bitext Customer Service dataset from HuggingFace and saves it to
`data/bitext_dataset.csv`. If HuggingFace is unreachable, a synthetic fallback dataset
with the same schema is generated automatically.

---

## Running the CLI

```bash
python main.py                        # auto-generated session ID
python main.py --session alice        # named session (persists across restarts)
```

The CLI prints the full reasoning chain for each turn:

```
You: How many records does the most common SHIPPING intent have?

────────────────────────────────────────────────────
  REASONING
  ▶ TOOL CALL  get_intent_distribution({"category": "SHIPPING"})
  ◀ RESULT     get_intent_distribution: {"track_order": 312, "delivery_options": 198, ...}
  ▶ TOOL CALL  count_records({"intent": "track_order"})
  ◀ RESULT     count_records: 312
────────────────────────────────────────────────────

● ANSWER
The most common SHIPPING intent is `track_order` with 312 records.
```

(Counts above are illustrative — actual numbers depend on the loaded dataset.)

**Resuming a session after restart:**

```bash
python main.py --session alice   # picks up exactly where you left off
```

**Managing sessions:**

```bash
python main.py --list-sessions          # show every saved session
python main.py --reset alice            # delete alice's history + profile
python main.py --reset __all__          # wipe every session (clean slate)
```

---

## Debugging with LangGraph Studio

A `langgraph.json` is included, so you can open the graph visually:

```bash
pip install -U "langgraph-cli[inmem]"
langgraph dev --allow-blocking
```

> **Why `--allow-blocking`?** The SQLite checkpointer opens its connection
> synchronously during graph construction. `langgraph dev` wraps your code in
> `blockbuster` to catch sync I/O inside its async server; the flag disables
> that check. It only affects the dev server — the CLI, Streamlit, and MCP
> server are all sync and unaffected.

Studio loads the `agent` graph from `agent/graph.py:make_studio_graph` (a fixed
`studio` session_id is used so checkpoints have a stable thread). The local
server runs at `http://127.0.0.1:2024`; `langgraph dev` will open a browser to
the Studio UI hosted at smith.langchain.com — a free LangSmith account is
required to access that page, but no API key is needed and all execution stays
local.

---

## Running the Streamlit UI (Bonus A)

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). Enter a Session ID in the sidebar to
create or resume a conversation. Each agent response includes a collapsible
"🔧 Reasoning steps" panel showing every tool call and result.

---

## Running the MCP Server (Task 3)

### Start the server

```bash
python mcp_server.py
```

The server uses stdio transport (compatible with Claude Desktop and the MCP CLI).

### Inspect tools with the MCP CLI

```bash
# Install the MCP dev tools if needed
pip install 'mcp[cli]'

# Launch the inspector UI
mcp dev mcp_server.py
```

### Connect from Claude Desktop

Add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "cs-analyst": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_server.py"]
    }
  }
}
```

### Call a tool from a Python client

Save this as `mcp_client_example.py` and run `python mcp_client_example.py`:

```python
"""Minimal MCP client that connects to mcp_server.py over stdio and calls list_categories."""
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. List every tool the server exposes
            tools = await session.list_tools()
            print("Tools:", [t.name for t in tools.tools])

            # 2. Call list_categories (no arguments)
            result = await session.call_tool("list_categories", {})
            print("Categories:", result.content)

            # 3. Call count_records with a filter
            result = await session.call_tool(
                "count_records", {"category": "REFUND"}
            )
            print("REFUND record count:", result.content)


if __name__ == "__main__":
    asyncio.run(main())
```

Expected output (truncated):

```
Tools: ['list_categories', 'count_records', 'get_sample_records', 'get_intent_distribution', 'list_intents']
Categories: [TextContent(text='["ACCOUNT", "CANCELLATION_FEE", ...]')]
REFUND record count: [TextContent(text='7')]
```

---

## Architecture Overview

### Model Choice

| Role | Model | Reason |
|------|-------|--------|
| **Main agent** | `Nemotron-3-Super-120b-a12b` | 120B hybrid MoE, explicitly optimised for "multi-agent AI and complex reasoning tasks" on Nebius. 127 Tok/s — fast enough for interactive use. Handles tool chaining and open-ended summarisation well. |
| **Router** | `Nemotron-3-Nano-30B-A3B` | Compact MoE, 60 Tok/s, ~5× cheaper. Query classification is a single structured-output call — a nano model handles it reliably and keeps costs low. |

Both models are available on [Nebius Token Factory](https://tokenfactory.nebius.com/).
Each model has its own regional endpoint (configured in `agent/config.py`):

| Model | Endpoint |
|-------|----------|
| `Nemotron-3-Super-120b-a12b` | `https://api.tokenfactory.us-central1.nebius.com/v1/` |
| `Nemotron-3-Nano-30B-A3B` | `https://api.tokenfactory.nebius.com/v1/` |

### LangGraph Graph

```
START
  └─► router_node              (Nano-30B classifies: structured / unstructured / out_of_scope)
        ├─► out_of_scope ──► END                 (polite decline, no profile update)
        └─► agent_node          (Super-120b + tools bound, ReAct loop)
              ├─► tools_node ──► agent_node      (loop until no tool call)
              └─► update_profile_node ──► END
```

- **Max iterations:** Hard cap at 12 (`agent_node` and `route_after_agent` both check
  `>= MAX_ITERATIONS`). If reached, a graceful fallback message is returned.
- **Checkpointer:** `SqliteSaver` at `checkpoints/memory.db`. A single connection is
  cached for the lifetime of the process (`get_checkpointer()` in `graph.py`) and
  shared across every `build_graph` call, so no handles leak when Streamlit users
  switch sessions.
- **Profile update node:** Runs after every final answer **from the agent path**
  (out-of-scope skips it). Calls the LLM to extract new facts (name, topics,
  preferences) and writes them to `profiles/{session_id}.json`.
- **`pending_suggestion` state field (Bonus B):** When the agent issues a
  recommendation it stores the suggested query in this field instead of executing
  it. The next user turn confirms or refines; the agent only executes once a
  non-profile tool call is made (which clears the field). This makes the
  "suggest → confirm → execute" flow enforced by graph state, not just prompt
  discipline.
- **LangGraph Studio entry-point:** `make_studio_graph()` in `agent/graph.py` is
  the lazy factory referenced by `langgraph.json` — Studio gets a compiled graph
  bound to a fixed `studio` session_id without paying the cost at import time.

### Tools Defined

| Tool | Description |
|------|-------------|
| `list_categories` | All unique top-level categories |
| `list_intents` | All intents, optionally filtered by category |
| `count_records` | Row count with category / intent / keyword filters |
| `get_sample_records` | N example instruction+response pairs |
| `get_intent_distribution` | Intent breakdown within a category |
| `get_category_data` | Bulk records for open-ended summarisation |
| `read_user_profile` | Current user's persisted profile (for query recommender) |

### Memory

| Layer | Implementation | Persists across restart? |
|-------|---------------|--------------------------|
| Conversation history | LangGraph `SqliteSaver` checkpointer | ✅ Yes |
| User profile | JSON file at `profiles/{session_id}.json` | ✅ Yes |

The profile includes a `preferences` list (e.g. "respond in Hebrew", "always use
tables"). These are treated as standing instructions and applied to every reply.
If a preference becomes stale, wipe it with:

```bash
python main.py --reset <session_id>     # delete that session's history + profile
python main.py --reset __all__          # nuclear option: wipe everything
```

### Query Recommender (Bonus B)

Implemented inside the existing ReAct loop — no extra graph node, but with explicit
state enforcement so the agent cannot accidentally execute a suggestion in the same
turn:

1. **Router** classifies "What should I query next?" / "recommend something" /
   "what do you remember about me?" etc. as `unstructured` (few-shot examples in
   the router prompt make this reliable).
2. **Agent node** calls `read_user_profile` to read distilled facts about the user,
   then replies with a TEXT suggestion ending in a confirmation question.
3. The agent node detects the suggestion ("suggest" / "recommend" / "how about" +
   `?`) and writes it to the `pending_suggestion` state field. No execution tool
   is called this turn.
4. **Next turn:** the system prompt's PENDING SUGGESTION banner tells the agent
   what was suggested. If the user confirms, the agent runs the appropriate
   execution tool and the field is cleared automatically (because any non-profile
   tool call clears `pending_suggestion`). If the user refines, a new text
   suggestion is produced and the cycle repeats.

So the multi-turn "suggest → confirm → execute" dance is enforced by graph state,
not just prompt discipline.

---

## Project Structure

```
.
├── agent/
│   ├── __init__.py
│   ├── config.py            # Model IDs, paths, MAX_ITERATIONS
│   ├── tools.py             # 6 dataset tools + session-bound read_user_profile factory
│   ├── graph.py             # LangGraph StateGraph, cached checkpointer, Studio entry-point
│   └── profile.py           # Per-user JSON profile load/update
├── data/
│   └── bitext_dataset.csv   # Downloaded by download_data.py
├── checkpoints/
│   └── memory.db            # SQLite conversation history (auto-created)
├── profiles/                # Per-user profile JSON files (auto-created)
├── download_data.py         # Dataset setup script (with synthetic fallback)
├── mcp_server.py            # FastMCP server (5 tools)
├── main.py                  # CLI (--session / --list-sessions / --reset)
├── app.py                   # Streamlit UI (Bonus A)
├── langgraph.json           # LangGraph Studio config — points at make_studio_graph
├── requirements.txt
├── .env.example
└── README.md
```

> A `PLAN.md` is also present — that's the initial design notebook kept for
> reference. It's not authoritative; this README is.

---

## Example Queries to Test

| Query | Type | What it tests |
|-------|------|---------------|
| "What categories exist in the dataset?" | Structured (single tool) | `list_categories` |
| "How many refund requests did we get?" | Structured (single tool) | `count_records(category="REFUND")` — broad question → category filter, not intent |
| "Show me 5 examples of the SHIPPING category." | Structured (single tool) | `get_sample_records` with category filter |
| "What is the distribution of intents in the ACCOUNT category?" | Structured (single tool) | `get_intent_distribution` |
| "Show me examples of people wanting their money back." | Structured (single tool) | `get_sample_records` with keyword |
| **"How many records does the largest REFUND intent have?"** | **Structured (multi-step)** | **`get_intent_distribution` → `count_records(intent=<top>)` — chained tools** |
| **"Show me an example from the most common SHIPPING intent."** | **Structured (multi-step)** | **`get_intent_distribution` → `get_sample_records(intent=<top>, n=1)`** |
| "Show me 3 more" (after a sample) | Structured follow-up | `get_sample_records` with `offset=3` — exercises episodic memory |
| "Summarise how agents respond to complaint intents." | Unstructured | `get_category_data` → LLM synthesis (random-sampled when >max_rows) |
| "What should I query next?" | Unstructured (Bonus B) | `read_user_profile` → suggestion, then waits for confirmation (state-enforced via `pending_suggestion`) |
| "Yes, do it." (after a suggestion) | Unstructured (Bonus B) | Pending suggestion is executed; field cleared |
| "What do you remember about me?" | Unstructured | `read_user_profile` |
| "What's the best CRM software for handling complaints?" | Out-of-scope | Router → polite decline (no profile update, no LLM general knowledge) |
| "Who is the president of France?" | Out-of-scope | Router → polite decline |
