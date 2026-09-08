from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

TOOL_FAILURE_CODES = frozenset(
    {
        "oauth_token_required",
        "unsupported_nested_interaction",
        "missing_delegated_output",
    }
)

# Pseudo-tools that drive the run rather than retrieve evidence. They flow
# through ``add_tool_result`` like real tools, so anything reasoning about
# retrieved evidence has to exclude them: "re-running" one of these ends the
# run or re-contacts the user instead of re-fetching a value.
CONTROL_TOOL_NAMES = frozenset(
    {
        "final_answer",
        "send_message",
        "ask_user_question",
    }
)

# Placeholder outputs the execution layers substitute when a run produced no
# text of its own. They are recognized, not produced, by the delegation
# classifier: a child that returns one of these completed without answering.
NO_OUTPUT_PLACEHOLDER = "No output provided"
NO_RESPONSE_PLACEHOLDER = "No response generated"


def normalize_tool_failure_code(value: Any) -> str | None:
    """Return an exact public tool failure code when it is allowlisted."""

    return value if type(value) is str and value in TOOL_FAILURE_CODES else None


def is_oauth_token_required_code(value: Any) -> bool:
    """Return True iff value is the exact "oauth_token_required" plain string.

    A str subclass is a trust-boundary input, not an allowlisted code.
    """
    return type(value) is str and value == "oauth_token_required"


@dataclass(frozen=True)
class ClassifiedToolFailure:
    """Sentinel an OAuth token resolver returns when it cannot refresh a token.

    Its only current consumer is the delegated-authorization retry path in
    the MCP adapter, which treats any instance as an OAuth failure regardless
    of the carried code. ``failure_code`` is validated against the exact
    ``"oauth_token_required"`` literal, not the wider runtime allowlist: a
    new public failure code becomes surfaceable by the runtime without
    becoming carriable by this type.
    """

    failure_code: str

    def __post_init__(self) -> None:
        if not is_oauth_token_required_code(self.failure_code):
            raise ValueError("invalid tool failure code")


def tool_result_succeeded(result: Any) -> bool:
    """Classify the supported structured tool-result failure shapes."""

    if not isinstance(result, dict):
        return True
    if result.get("success") is False or result.get("is_error") is True:
        return False
    status = result.get("status")
    return not (isinstance(status, str) and status.lower() == "error")


FINAL_ANSWER_KEYS = ("response", "answer", "output", "content", "message")


def assistant_message_key(result: dict[str, Any]) -> str | None:
    """Select one canonical answer without treating other strings as aliases."""

    for key in FINAL_ANSWER_KEYS:
        value = result.get(key)
        if isinstance(value, str) and value:
            return key
    return None


def extract_assistant_message(result: dict[str, Any]) -> str | None:
    """Return the assistant-facing output from a normalized pattern result."""
    key = assistant_message_key(result)
    return unwrap_final_answer_content(str(result[key])) if key is not None else None


def set_assistant_message(result: dict[str, Any], content: str) -> None:
    """Replace the canonical answer and exact aliases, preserving other fields."""
    key = assistant_message_key(result)
    if key is None:
        return
    original = result[key]
    for candidate in FINAL_ANSWER_KEYS:
        if result.get(candidate) == original:
            result[candidate] = content


def unwrap_final_answer_content(content: str) -> str:
    """Unwrap textual legacy final_answer JSON into display-ready content."""
    parsed = _parse_json_like_text(content)
    if not isinstance(parsed, dict):
        return content

    action = str(parsed.get("action") or "").strip()
    if action == "final_answer":
        return _stringify_answer_value(
            parsed.get("action_input")
            if "action_input" in parsed
            else parsed.get("answer", content)
        )

    if "final_answer" in parsed:
        return _stringify_answer_value(parsed["final_answer"])

    return content


def _parse_json_like_text(content: str) -> Any | None:
    text = _strip_code_fence(content.strip())
    if not text or not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _strip_code_fence(text: str) -> str:
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```"):
        closing = lines[-1].strip()
        if closing == "```" or closing.startswith("```"):
            return "\n".join(lines[1:-1]).strip()
    return text


def _stringify_answer_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)
