"""
CLI entry-point for the Customer Service Data Analyst Agent.

Usage:
    python main.py                        # auto-generates a session ID
    python main.py --session my_session   # resume or start a named session

The agent drops into an interactive loop. Type your question and press Enter.
Type 'exit', 'quit', or press Ctrl-C to leave.

Each turn prints the full reasoning chain:
  [TOOL CALL]   — tool name and arguments
  [OBSERVATION] — tool result
  [ANSWER]      — the final answer
"""

import argparse
import json
import os
import sqlite3
import sys
import uuid
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.config import CHECKPOINTS_DB, PROFILES_DIR
from agent.graph import build_graph, get_config


# ── ANSI colour helpers ───────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
GREEN  = "\033[32m"
DIM    = "\033[2m"
RED    = "\033[31m"


def _colour(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


# ── Streaming output helpers ──────────────────────────────────────────────────

def _print_tool_call(tool_name: str, args: Dict[str, Any]) -> None:
    """Print a formatted tool-call line."""
    args_str = json.dumps(args, ensure_ascii=False)
    print(f"  {_colour('▶ TOOL CALL', CYAN)}  {_colour(tool_name, BOLD)}({DIM}{args_str}{RESET})")


def _print_observation(tool_name: str, content: str) -> None:
    """Print a formatted tool-result line (truncated for readability)."""
    preview = content[:300].replace("\n", " ")
    if len(content) > 300:
        preview += " …"
    print(f"  {_colour('◀ RESULT', YELLOW)}    {DIM}{tool_name}:{RESET} {preview}")


def _print_answer(content: str) -> None:
    """Print the final agent answer."""
    print(f"\n{_colour('● ANSWER', GREEN)}\n{content}\n")


def _print_reasoning_header() -> None:
    print(f"\n{DIM}{'─' * 60}{RESET}")
    print(f"  {_colour('REASONING', BOLD)}")


def _print_separator() -> None:
    print(f"{DIM}{'─' * 60}{RESET}\n")


# ── Single-turn invocation with streamed reasoning ────────────────────────────

def run_turn(graph: Any, user_input: str, config: Dict[str, Any]) -> None:
    """Send one user message to the graph and print the reasoning + answer.

    Args:
        graph:      Compiled LangGraph StateGraph.
        user_input: The user's text query.
        config:     LangGraph invocation config (contains thread_id).
    """
    _print_reasoning_header()

    events = graph.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="updates",
    )

    final_answer: str = ""

    for event in events:
        for node_name, node_output in event.items():
            if node_output is None:
                continue
            messages = node_output.get("messages", [])

            for msg in messages:
                # ── Tool call made by the LLM ──────────────────────────────
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        _print_tool_call(tc["name"], tc.get("args", {}))

                # ── Tool result returned by ToolNode ───────────────────────
                elif isinstance(msg, ToolMessage):
                    _print_observation(msg.name, str(msg.content))

                # ── Final text answer ──────────────────────────────────────
                elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    final_answer = msg.content

    _print_separator()
    if final_answer:
        _print_answer(final_answer)
    else:
        _print_answer("(No final answer produced.)")


# ── Interactive conversation loop ─────────────────────────────────────────────

def interactive_loop(session_id: str) -> None:
    """Drop into an interactive conversation with the agent.

    Args:
        session_id: Session identifier — determines which SQLite checkpoint to use.
    """
    print(f"\n{_colour('Customer Service Data Analyst Agent', BOLD)}")
    print(f"Session: {_colour(session_id, CYAN)}")
    print(f"Agent:   Nemotron-3-Super-120b-a12b  (us-central1)")
    print(f"Router:  Nemotron-3-Nano-30B-A3B     (eu-north1)")
    print(f"Type your question and press Enter. Type {_colour('exit', RED)} to quit.\n")

    graph = build_graph(session_id)
    config = get_config(session_id)

    while True:
        try:
            user_input = input(_colour("You: ", BOLD)).strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{_colour('Goodbye!', DIM)}")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "q", "bye"}:
            print(_colour("Goodbye!", DIM))
            sys.exit(0)

        try:
            run_turn(graph, user_input, config)
        except Exception as exc:
            print(f"{_colour('ERROR', RED)}: {exc}")
            print("Please try again.\n")


# ── Session management helpers ────────────────────────────────────────────────

def list_sessions() -> List[str]:
    """Return every thread_id (session) stored in the SQLite checkpoint DB.

    Reads directly from the LangGraph checkpoints table so it does not need to
    instantiate the graph (no API key required).
    """
    if not os.path.exists(CHECKPOINTS_DB):
        return []
    try:
        conn = sqlite3.connect(CHECKPOINTS_DB)
        cur = conn.cursor()
        # LangGraph SqliteSaver stores rows in `checkpoints` keyed by thread_id.
        cur.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id")
        return [row[0] for row in cur.fetchall()]
    except sqlite3.Error:
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass


def reset_session(session_id: str) -> None:
    """Delete a session's checkpoints AND its profile file.

    Args:
        session_id: The session to remove. Use "__all__" to wipe everything.
    """
    # 1. Checkpoint rows
    if os.path.exists(CHECKPOINTS_DB):
        conn = sqlite3.connect(CHECKPOINTS_DB)
        try:
            cur = conn.cursor()
            if session_id == "__all__":
                for table in ("checkpoints", "writes"):
                    try:
                        cur.execute(f"DELETE FROM {table}")
                    except sqlite3.OperationalError:
                        pass
            else:
                for table in ("checkpoints", "writes"):
                    try:
                        cur.execute(f"DELETE FROM {table} WHERE thread_id = ?", (session_id,))
                    except sqlite3.OperationalError:
                        pass
            conn.commit()
        finally:
            conn.close()

    # 2. Profile file
    if session_id == "__all__":
        if os.path.isdir(PROFILES_DIR):
            for fname in os.listdir(PROFILES_DIR):
                if fname.endswith(".json"):
                    os.remove(os.path.join(PROFILES_DIR, fname))
    else:
        path = os.path.join(PROFILES_DIR, f"{session_id}.json")
        if os.path.exists(path):
            os.remove(path)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and start the interactive agent loop."""
    parser = argparse.ArgumentParser(
        description="Customer Service Data Analyst Agent — interactive CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py\n"
            "  python main.py --session alice\n"
            "  python main.py --session alice          # resume alice's conversation\n"
            "  python main.py --list-sessions          # show every saved session\n"
            "  python main.py --reset alice            # delete alice's history + profile\n"
            "  python main.py --reset __all__          # wipe every session\n"
        ),
    )
    parser.add_argument(
        "--session",
        metavar="SESSION_ID",
        default=None,
        help=(
            "Session identifier for conversation memory. "
            "Reuse the same ID after a restart to restore the conversation. "
            "Defaults to a new random UUID."
        ),
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List every saved session (thread_id) in the checkpoint DB, then exit.",
    )
    parser.add_argument(
        "--reset",
        metavar="SESSION_ID",
        default=None,
        help=(
            "Delete a session's checkpoints and profile file, then exit. "
            "Pass '__all__' to wipe every session."
        ),
    )
    args = parser.parse_args()

    if args.list_sessions:
        sessions = list_sessions()
        if not sessions:
            print("(no sessions stored)")
        else:
            print(f"Stored sessions ({len(sessions)}):")
            for sid in sessions:
                print(f"  - {sid}")
        sys.exit(0)

    if args.reset is not None:
        reset_session(args.reset)
        if args.reset == "__all__":
            print(_colour("All sessions wiped.", GREEN))
        else:
            print(_colour(f"Session '{args.reset}' wiped.", GREEN))
        sys.exit(0)

    session_id = args.session or str(uuid.uuid4())[:8]
    interactive_loop(session_id)


if __name__ == "__main__":
    main()
