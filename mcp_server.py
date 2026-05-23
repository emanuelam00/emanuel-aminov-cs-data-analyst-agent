"""
FastMCP server — exposes Bitext dataset tools over the MCP protocol.

Run:
    python mcp_server.py

Connect a client (e.g. Claude Desktop or the MCP CLI inspector):
    mcp dev mcp_server.py

Or call a tool directly with the MCP CLI:
    echo '{}' | mcp run mcp_server.py list_categories
"""

from typing import Dict, List, Optional

from fastmcp import FastMCP

# Import the pure data functions (no LangGraph dependency needed here)
from agent.tools import (
    get_dataframe,
    get_intent_distribution as _get_intent_distribution,
    get_sample_records as _get_sample_records,
)

mcp = FastMCP(
    name="Customer Service Analyst",
    instructions=(
        "This server exposes tools for analysing the Bitext Customer Service dataset. "
        "The dataset contains synthetic customer support queries and agent responses "
        "tagged with category (e.g. ACCOUNT, REFUND, SHIPPING) and intent. "
        "Use these tools to answer data-driven questions about the dataset."
    ),
)


# ── Tool 1: list_categories ───────────────────────────────────────────────────

@mcp.tool()
def list_categories() -> List[str]:
    """Return a sorted list of all unique top-level categories in the dataset.

    Categories include: ACCOUNT, CANCELLATION_FEE, CONTACT, DELIVERY, FEEDBACK,
    INVOICE, NEWSLETTER, ORDER, PAYMENT, REFUND, SHIPPING, and others.
    """
    df = get_dataframe()
    return sorted(df["category"].unique().tolist())


# ── Tool 2: count_records ─────────────────────────────────────────────────────

@mcp.tool()
def count_records(
    category: Optional[str] = None,
    intent: Optional[str] = None,
    keyword: Optional[str] = None,
) -> int:
    """Count how many records match the given filters.

    All filters are optional and combined with AND logic.

    Args:
        category: Filter by category name (case-insensitive, e.g. "REFUND").
        intent:   Filter by intent name (case-insensitive, e.g. "get_refund").
        keyword:  Substring to search for in the customer instruction field.

    Returns:
        Integer count of matching records.
    """
    df = get_dataframe()
    if category:
        df = df[df["category"] == category.upper().strip()]
    if intent:
        df = df[df["intent"] == intent.lower().strip()]
    if keyword:
        df = df[df["instruction"].str.contains(keyword, case=False, na=False)]
    return int(len(df))


# ── Tool 3: get_sample_records ────────────────────────────────────────────────

@mcp.tool()
def get_sample_records(
    n: int = 5,
    category: Optional[str] = None,
    intent: Optional[str] = None,
    keyword: Optional[str] = None,
    offset: int = 0,
) -> List[Dict]:
    """Return up to N example records (instruction + response pairs) from the dataset.

    Each record contains: category, intent, instruction (customer query),
    and response (agent reply). Pass `offset` to page past records already shown.

    Args:
        n:        Number of records to return (default 5, max 20).
        category: Optional category filter (case-insensitive).
        intent:   Optional intent filter (case-insensitive).
        keyword:  Optional keyword to search in the instruction text.
        offset:   Number of matching records to skip first (default 0).

    Returns:
        List of dicts, each with keys: category, intent, instruction, response.
    """
    n = max(1, min(n, 20))
    offset = max(0, offset)
    df = get_dataframe()
    if category:
        df = df[df["category"] == category.upper().strip()]
    if intent:
        df = df[df["intent"] == intent.lower().strip()]
    if keyword:
        df = df[df["instruction"].str.contains(keyword, case=False, na=False)]
    cols = ["category", "intent", "instruction", "response"]
    return df.iloc[offset : offset + n][cols].to_dict(orient="records")


# ── Tool 4: get_intent_distribution ──────────────────────────────────────────

@mcp.tool()
def get_intent_distribution(category: str) -> Dict[str, int]:
    """Return the count of records for each intent within a specific category.

    Args:
        category: The category to analyse (required, e.g. "ACCOUNT", "REFUND").

    Returns:
        Dict mapping intent name to record count, sorted by count descending.
    """
    df = get_dataframe()
    filtered = df[df["category"] == category.upper().strip()]
    if filtered.empty:
        return {}
    raw: Dict[str, int] = filtered["intent"].value_counts().to_dict()
    return raw


# ── Tool 5: list_intents ──────────────────────────────────────────────────────

@mcp.tool()
def list_intents(category: Optional[str] = None) -> List[str]:
    """Return a sorted list of all intent names, optionally filtered by category.

    Args:
        category: Optional category to filter intents by.

    Returns:
        Sorted list of intent name strings.
    """
    df = get_dataframe()
    if category:
        df = df[df["category"] == category.upper().strip()]
    return sorted(df["intent"].unique().tolist())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
