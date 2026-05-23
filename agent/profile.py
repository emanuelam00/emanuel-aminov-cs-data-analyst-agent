"""
Per-user profile persistence.

Each user (identified by session_id) gets a JSON file at profiles/{session_id}.json
containing distilled, human-readable facts about them — NOT a replay of messages.

Public API:
  load_profile(session_id)             -> dict
  update_profile(session_id, messages, llm) -> None
"""

import json
import os
from typing import Any, Dict, List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel

from agent.config import PROFILES_DIR


# ── Default profile schema ───────────────────────────────────────────────────

DEFAULT_PROFILE: Dict[str, Any] = {
    "name": None,
    "frequent_topics": [],
    "preferred_categories": [],
    "preferences": [],       # response-style preferences, e.g. "prefers table format"
    "notes": [],
    "session_count": 0,
}


# ── File helpers ─────────────────────────────────────────────────────────────

def _profile_path(session_id: str) -> str:
    """Return the absolute path for a session's profile file."""
    os.makedirs(PROFILES_DIR, exist_ok=True)
    return os.path.join(PROFILES_DIR, f"{session_id}.json")


def load_profile(session_id: str) -> Dict[str, Any]:
    """Load the profile for session_id, creating a default one if absent.

    Args:
        session_id: The conversation / user session identifier.

    Returns:
        A dict with keys: name, frequent_topics, preferred_categories,
        preferences, notes, session_count.
    """
    path = _profile_path(session_id)
    if not os.path.exists(path):
        return dict(DEFAULT_PROFILE)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Backfill any keys added after the profile was first created
        for key, default_val in DEFAULT_PROFILE.items():
            data.setdefault(key, default_val)
        return data
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULT_PROFILE)


def _save_profile(session_id: str, profile: Dict[str, Any]) -> None:
    """Persist the profile dict to disk."""
    path = _profile_path(session_id)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2, ensure_ascii=False)


# ── LLM-driven profile update ─────────────────────────────────────────────────

_UPDATE_PROMPT = """\
You are a memory assistant. Your job is to maintain a concise user profile.

CURRENT PROFILE (JSON):
{current_profile}

RECENT CONVERSATION (last few exchanges):
{recent_messages}

Based ONLY on the conversation above, update the profile if new information is present.
Rules:
- "name": set if the user explicitly states their name.
- "frequent_topics": list of dataset topics the user has asked about (e.g. "REFUND", "SHIPPING").
  Add new topics; keep existing ones. Keep the list unique and sorted.
- "preferred_categories": categories the user returns to repeatedly (appears 2+ times in history).
- "preferences": list of response-style or formatting preferences the user has expressed.
  Look for ANY instruction about HOW the agent should respond, such as:
    "show results as a table", "keep answers brief", "always include examples",
    "use bullet points", "respond in Hebrew", "give me the raw numbers too", etc.
  Capture the preference as a short imperative phrase (e.g. "present data in table format").
  Add new preferences; keep existing ones. Maximum 10 items.
- "notes": brief factual observations about the user's interests or goals (1 sentence each).
  Add new notes; do not repeat existing ones. Maximum 5 notes total.
- "session_count": do NOT change — it is managed separately.

Return ONLY a valid JSON object with exactly these keys:
  name, frequent_topics, preferred_categories, preferences, notes, session_count

If nothing new is learned, return the current profile unchanged.
"""


def update_profile(
    session_id: str,
    messages: List[BaseMessage],
    llm: BaseChatModel,
) -> None:
    """Analyse the most recent conversation turn and persist any new user facts.

    This is a best-effort operation — if the LLM call or JSON parse fails, the
    existing profile is left untouched.

    Args:
        session_id: The session / user identifier.
        messages:   Full conversation message list from graph state.
        llm:        The chat model to use for fact extraction.
    """
    current = load_profile(session_id)

    # Increment session count on every final-answer turn
    current["session_count"] = current.get("session_count", 0) + 1

    # Build a text snippet of the last 6 messages (3 turns) for context
    recent = messages[-6:] if len(messages) > 6 else messages
    recent_text_parts: List[str] = []
    for msg in recent:
        if isinstance(msg, HumanMessage):
            recent_text_parts.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage) and msg.content:
            recent_text_parts.append(f"Agent: {msg.content}")
    recent_text = "\n".join(recent_text_parts)

    if not recent_text.strip():
        _save_profile(session_id, current)
        return

    prompt = _UPDATE_PROMPT.format(
        current_profile=json.dumps(current, indent=2),
        recent_messages=recent_text,
    )

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.splitlines()
            content = "\n".join(
                line for line in lines
                if not line.startswith("```")
            )

        updated = json.loads(content)

        # Enforce schema — never let the LLM add unexpected keys
        safe: Dict[str, Any] = {}
        for key, default in DEFAULT_PROFILE.items():
            safe[key] = updated.get(key, current.get(key, default))

        # Preserve session_count from our own increment (don't let LLM overwrite)
        safe["session_count"] = current["session_count"]

        _save_profile(session_id, safe)
    except Exception:
        # Silent fallback: at least persist the session_count increment
        _save_profile(session_id, current)


def format_profile(profile: Dict[str, Any]) -> str:
    """Return a human-readable string representation of the profile.

    Args:
        profile: Profile dict as returned by load_profile().
    """
    lines: List[str] = []
    if profile.get("name"):
        lines.append(f"Name: {profile['name']}")
    if profile.get("frequent_topics"):
        lines.append(f"Frequent topics: {', '.join(profile['frequent_topics'])}")
    if profile.get("preferred_categories"):
        lines.append(f"Preferred categories: {', '.join(profile['preferred_categories'])}")
    if profile.get("preferences"):
        lines.append("Response preferences (MUST be applied to every answer):")
        for pref in profile["preferences"]:
            lines.append(f"  - {pref}")
    if profile.get("notes"):
        lines.append("Notes:")
        for note in profile["notes"]:
            lines.append(f"  - {note}")
    lines.append(f"Sessions: {profile.get('session_count', 0)}")
    return "\n".join(lines) if lines else "No profile information yet."
