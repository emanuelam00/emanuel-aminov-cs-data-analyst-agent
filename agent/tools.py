"""
Data analysis tools for the Bitext Customer Service dataset.

Each tool has:
  - A descriptive name and docstring (the LLM's guide to when/how to use it)
  - A Pydantic BaseModel as args_schema
  - A typed return value

The DataFrame is loaded once at module level and reused across all calls.
"""

import os
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agent.config import DATASET_PATH


# ── Dataset singleton ────────────────────────────────────────────────────────

_dataframe: Optional[pd.DataFrame] = None


def get_dataframe() -> pd.DataFrame:
    """Load the Bitext CSV once and cache it for subsequent calls."""
    global _dataframe
    if _dataframe is None:
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(
                f"Dataset not found at '{DATASET_PATH}'. "
                "Run 'python download_data.py' first."
            )
        _dataframe = pd.read_csv(DATASET_PATH)
        # Normalise category and intent to uppercase for consistent filtering
        _dataframe["category"] = _dataframe["category"].str.upper().str.strip()
        _dataframe["intent"] = _dataframe["intent"].str.lower().str.strip()
    return _dataframe


# ── Pydantic input schemas ───────────────────────────────────────────────────

class ListCategoriesInput(BaseModel):
    """Input schema for list_categories — no parameters required."""
    pass


class ListIntentsInput(BaseModel):
    """Input schema for list_intents."""
    category: Optional[str] = Field(
        default=None,
        description=(
            "Optional category name to filter intents by (e.g. 'REFUND', 'ACCOUNT'). "
            "If omitted, all intents across the entire dataset are returned."
        ),
    )


class CountRecordsInput(BaseModel):
    """Input schema for count_records."""
    category: Optional[str] = Field(
        default=None,
        description="Filter by category name (case-insensitive, e.g. 'REFUND').",
    )
    intent: Optional[str] = Field(
        default=None,
        description="Filter by intent name (case-insensitive, e.g. 'get_refund').",
    )
    keyword: Optional[str] = Field(
        default=None,
        description=(
            "Filter rows whose 'instruction' field contains this keyword "
            "(case-insensitive substring match)."
        ),
    )


class GetSampleRecordsInput(BaseModel):
    """Input schema for get_sample_records."""
    n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of example records to return (1–20).",
    )
    category: Optional[str] = Field(
        default=None,
        description="Filter by category name (case-insensitive).",
    )
    intent: Optional[str] = Field(
        default=None,
        description="Filter by intent name (case-insensitive).",
    )
    keyword: Optional[str] = Field(
        default=None,
        description=(
            "Filter rows whose 'instruction' field contains this keyword. "
            "Use this when the user describes a topic without naming the exact intent "
            "(e.g. 'people wanting their money back' → keyword='refund')."
        ),
    )
    offset: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of matching records to skip before returning the next N. "
            "Use this for follow-ups like 'show me N more' — pass the count of "
            "examples already shown so the user sees fresh records."
        ),
    )


class GetIntentDistributionInput(BaseModel):
    """Input schema for get_intent_distribution."""
    category: str = Field(
        description="The category to analyse (e.g. 'ACCOUNT', 'REFUND'). Required."
    )


class GetCategoryDataInput(BaseModel):
    """Input schema for get_category_data."""
    category: str = Field(
        description="The category to fetch data from for summarisation."
    )
    max_rows: int = Field(
        default=30,
        ge=5,
        le=60,
        description="Maximum number of rows to include (default 30, max 60).",
    )


class ReadUserProfileInput(BaseModel):
    """Input schema for read_user_profile — no parameters required."""
    pass


# ── Tools ────────────────────────────────────────────────────────────────────

@tool("list_categories", args_schema=ListCategoriesInput)
def list_categories() -> List[str]:
    """Return a sorted list of all unique top-level categories in the Bitext
    customer service dataset.

    Use this when the user asks:
      - "What categories exist?"
      - "What topics does the dataset cover?"
      - "What kinds of customer queries are there?"
    """
    df = get_dataframe()
    return sorted(df["category"].unique().tolist())


@tool("list_intents", args_schema=ListIntentsInput)
def list_intents(category: Optional[str] = None) -> List[str]:
    """Return a sorted list of intent names in the dataset, optionally filtered
    to a single category.

    Use this when the user asks what intents or sub-topics exist, for example:
      - "What intents are in the REFUND category?"
      - "List all intents."
      - "What sub-topics fall under SHIPPING?"

    Args:
        category: Optional category name to narrow results.
    """
    df = get_dataframe()
    if category:
        df = df[df["category"] == category.upper().strip()]
    return sorted(df["intent"].unique().tolist())


@tool("count_records", args_schema=CountRecordsInput)
def count_records(
    category: Optional[str] = None,
    intent: Optional[str] = None,
    keyword: Optional[str] = None,
) -> int:
    """Count how many records in the dataset match the given filters.

    Use this when the user asks for a number or quantity, for example:
      - "How many refund requests did we get?"   → category="REFUND"
      - "How many complaints are there?"          → category="FEEDBACK" or intent="complaint"
      - "Count of SHIPPING records."              → category="SHIPPING"

    Filter-choice guidance (IMPORTANT):
      - For BROAD topical questions ("how many refund requests", "how many
        complaints"), prefer `category=` — it covers every related intent.
        Using a single `intent=` will undercount because most categories contain
        several intents (e.g. REFUND has get_refund, track_refund, refund_policy…).
      - Use `intent=` only when the user names a specific intent or you have
        confirmed the intent from a prior tool call.
      - Use `keyword=` for fuzzy phrases the user describes in natural language
        (e.g. "people wanting their money back" → keyword="refund").

    Filters are combined with AND logic (all must match).

    Args:
        category: Optional category filter (case-insensitive).
        intent:   Optional intent filter (case-insensitive).
        keyword:  Optional keyword to search in the 'instruction' column.
    """
    df = get_dataframe()
    if category:
        df = df[df["category"] == category.upper().strip()]
    if intent:
        df = df[df["intent"] == intent.lower().strip()]
    if keyword:
        df = df[df["instruction"].str.contains(keyword, case=False, na=False)]
    return int(len(df))


@tool("get_sample_records", args_schema=GetSampleRecordsInput)
def get_sample_records(
    n: int = 5,
    category: Optional[str] = None,
    intent: Optional[str] = None,
    keyword: Optional[str] = None,
    offset: int = 0,
) -> List[Dict]:
    """Return up to N example records (instruction + response pairs) from the dataset.

    Use this when the user wants to see examples, such as:
      - "Show me 5 examples from the SHIPPING category."
      - "Give me examples of people wanting their money back."
      - "What do refund requests look like?"

    For follow-up requests like "show me N more", pass `offset` equal to the number
    of records already shown so the user sees a fresh batch (no repeats).

    Records include: category, intent, instruction (customer query), response (agent reply).

    Args:
        n:        Number of examples to return (default 5, max 20).
        category: Optional category filter.
        intent:   Optional intent filter.
        keyword:  Optional keyword search in the 'instruction' field.
        offset:   Number of matching records to skip first (default 0).
    """
    df = get_dataframe()
    if category:
        df = df[df["category"] == category.upper().strip()]
    if intent:
        df = df[df["intent"] == intent.lower().strip()]
    if keyword:
        df = df[df["instruction"].str.contains(keyword, case=False, na=False)]
    cols = ["category", "intent", "instruction", "response"]
    # Apply offset then take N — supports "show me N more" follow-ups.
    sample = df.iloc[offset : offset + n][cols].to_dict(orient="records")
    return sample


@tool("get_intent_distribution", args_schema=GetIntentDistributionInput)
def get_intent_distribution(category: str) -> Dict[str, int]:
    """Return the count of records for each intent within a category.

    Use this when the user wants a breakdown or distribution, for example:
      - "What is the distribution of intents in the ACCOUNT category?"
      - "How many records per intent in REFUND?"
      - "Break down the ORDER category by intent."

    Args:
        category: The category to analyse (required).
    """
    df = get_dataframe()
    filtered = df[df["category"] == category.upper().strip()]
    if filtered.empty:
        return {}
    distribution: Dict[str, int] = (
        filtered["intent"].value_counts().to_dict()
    )
    return distribution


@tool("get_category_data", args_schema=GetCategoryDataInput)
def get_category_data(category: str, max_rows: int = 30) -> str:
    """Fetch a representative sample of records from a category as formatted text,
    suitable for open-ended analysis or summarisation.

    Use this for unstructured / open-ended questions such as:
      - "Summarise the FEEDBACK category."
      - "How do agents typically respond to cancellation requests?"
      - "What patterns exist in complaint handling?"

    Sampling: when the category contains more than `max_rows` records, a random
    sample (seed=42 for reproducibility) is returned rather than the first N rows.
    This gives a more representative slice across intents. If the category is
    truncated, the returned text starts with a notice the LLM can pass on to the
    user so they know the summary is based on a sample.

    Returns a text block with intent, customer query, and agent response for each
    record, separated by '---'.

    Args:
        category: Category to fetch data from (required).
        max_rows: Maximum rows to include (default 30, max 60).
    """
    df = get_dataframe()
    filtered = df[df["category"] == category.upper().strip()]
    if filtered.empty:
        return f"No records found for category '{category}'."

    total = len(filtered)
    truncated = total > max_rows
    if truncated:
        # Random sample for representativeness; fixed seed keeps results stable
        # across repeated calls so the LLM doesn't see different data each turn.
        sample = filtered.sample(n=max_rows, random_state=42)
    else:
        sample = filtered

    lines: List[str] = []
    if truncated:
        lines.append(
            f"[NOTE: {total} total records in '{category.upper()}'. "
            f"Showing a random sample of {max_rows} for summarisation.]"
        )
    for _, row in sample.iterrows():
        lines.append(
            f"Intent: {row['intent']}\n"
            f"Customer: {row['instruction']}\n"
            f"Agent: {row['response']}"
        )
    return "\n\n---\n\n".join(lines)


# ── Session-bound tool factory ────────────────────────────────────────────────
# read_user_profile needs the session_id at call time. We expose a single
# factory that builds a session-bound version of the tool — graph.py uses this
# instead of defining its own closure. There is no module-level stub.

def make_read_user_profile_tool(session_id: str) -> Any:
    """Return a `read_user_profile` LangChain tool bound to the given session_id.

    The returned tool reads the persisted profile for `session_id` and returns
    a formatted string. Used by graph.py when assembling the per-session tool
    list for `ToolNode`.
    """
    # Local import avoids a circular dependency at module-load time.
    from agent.profile import format_profile, load_profile

    @tool("read_user_profile", args_schema=ReadUserProfileInput)
    def read_user_profile() -> str:
        """Return the current user's persisted profile as a formatted string.

        Use this when:
          - The user asks "What do you remember about me?"
          - You need context about the user's interests to recommend a next query (Bonus B).
          - The user asks for personalised suggestions.

        The profile contains distilled facts such as the user's name, frequently
        explored topics, and response preferences.
        """
        return format_profile(load_profile(session_id))

    return read_user_profile


def get_session_tools(session_id: str) -> List[Any]:
    """Return the full tool list bound to a specific session.

    Includes all dataset tools plus a session-aware `read_user_profile`.
    Use this everywhere the agent needs its tools (graph.py, future MCP
    integrations that want per-user profile access, etc.).
    """
    return [
        list_categories,
        list_intents,
        count_records,
        get_sample_records,
        get_intent_distribution,
        get_category_data,
        make_read_user_profile_tool(session_id),
    ]


# ── Dataset-only tool registry (no session needed) ────────────────────────────
# Useful for the MCP server and any caller that should not see user profiles.

DATASET_TOOLS = [
    list_categories,
    list_intents,
    count_records,
    get_sample_records,
    get_intent_distribution,
    get_category_data,
]
