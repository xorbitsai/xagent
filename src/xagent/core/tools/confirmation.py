"""Shared contract for tools that pause an execution to ask the user.

A tool signals "I cannot continue without the user" by returning
``status="waiting_for_user"``. The pattern turns that into an outbound
message, checkpoints, and stops. When the user answers, the pattern hands the
tool a single-use grant through :meth:`ConfirmableTool.authorize_confirmation`
before re-invoking it.

Nothing here is specific to one tool: patterns must detect the capability
rather than special-case a tool name, because tools reach the pattern wrapped
(output filtering, sandboxing), and a name check silently fails on wrappers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

#: Reserved prefix for arguments the runtime injects. A model-supplied value is
#: always stripped before a tool sees it.
RESERVED_TOOL_ARG_PREFIX = "_xagent_"

#: Reserved argument a model must never be able to forge.
TOOL_APPROVAL_ARG = f"{RESERVED_TOOL_ARG_PREFIX}tool_approval"

#: Legacy name for :data:`TOOL_APPROVAL_ARG`, still stripped defensively.
LEGACY_COMPUTER_APPROVAL_ARG = f"{RESERVED_TOOL_ARG_PREFIX}computer_approval"

#: Reserved argument carrying the plan step a tool call belongs to, so
#: concurrent steps can be given separate browser sessions.
STEP_SESSION_ARG = f"{RESERVED_TOOL_ARG_PREFIX}step_id"

WAITING_FOR_USER_STATUS = "waiting_for_user"

#: Confirmation kind that a user answer can authorize. Other kinds (such as a
#: hand-over where the user acts themselves) never produce a grant.
TOOL_ACTION_CONFIRMATION_KIND = "computer_action_confirmation"

_APPROVAL_PHRASES = frozenset(
    {
        "approve",
        "approved",
        "approve this action",
        "yes",
        "y",
        "ok",
        "okay",
        "confirm",
        "confirmed",
        "go ahead",
        "proceed",
        "continue",
        "do it",
        "allow",
        "sure",
        "同意",
        "批准",
        "确认",
        "确定",
        "是",
        "好",
        "好的",
        "可以",
        "继续",
        "允许",
        "执行",
    }
)

_DENIAL_PHRASES = frozenset(
    {
        "deny",
        "denied",
        "deny this action",
        "no",
        "n",
        "cancel",
        "cancelled",
        "canceled",
        "cancel this action",
        "stop",
        "abort",
        "reject",
        "rejected",
        "don't",
        "do not",
        "拒绝",
        "取消",
        "不",
        "不要",
        "不行",
        "否",
        "停止",
        "算了",
    }
)

_TRAILING_PUNCTUATION = " \t\r\n.。!！?？,，、;；:：\"'“”‘’()（）"


def _normalize_phrase(value: str) -> str:
    return value.strip().strip(_TRAILING_PUNCTUATION).casefold()


def _candidate_phrases(response: str) -> list[str]:
    """Whole answers a reply may consist of, once labels are stripped.

    Replies arrive both as a widget value ("computer_action_decision: approve")
    and as free-form chat ("approve"), so each line counts both in full and
    with its field label removed.
    """
    phrases: list[str] = []
    for line in response.splitlines():
        for chunk in (line, line.split(":", 1)[-1]):
            normalized = _normalize_phrase(chunk)
            if normalized and normalized not in phrases:
                phrases.append(normalized)
    return phrases


def _phrase_fragments(phrase: str) -> list[str]:
    parts = [_normalize_phrase(part) for part in phrase.replace("，", ",").split(",")]
    return [part for part in parts if part]


def user_approved_confirmation(response: str) -> bool | None:
    """Interpret a user's answer to a confirmation prompt.

    Returns ``True`` for approval, ``False`` for refusal, and ``None`` when the
    reply carries no clear decision.

    Matching is whole-answer, not keyword-in-prose: scanning for words inside a
    sentence is how "don't approve" turns into an approval. A comma-separated
    reply counts only when every part agrees ("yes, go ahead"), so a mixed
    answer stays undecided. ``None`` must never be treated as approval, and it
    is kept distinct from refusal so callers can say they did not understand
    rather than silently dropping the request.
    """
    phrases = _candidate_phrases(response)
    if any(phrase in _DENIAL_PHRASES for phrase in phrases):
        return False
    if any(phrase in _APPROVAL_PHRASES for phrase in phrases):
        return True

    for phrase in phrases:
        fragments = _phrase_fragments(phrase)
        if len(fragments) < 2:
            continue
        if any(fragment in _DENIAL_PHRASES for fragment in fragments):
            return False
        if all(fragment in _APPROVAL_PHRASES for fragment in fragments):
            return True
    return None


@runtime_checkable
class ConfirmableTool(Protocol):
    """A tool that can accept a trusted grant for one subsequent call."""

    def authorize_confirmation(
        self,
        *,
        confirmation_id: str,
        decision: str,
        session_id: str,
        frame_signature: Mapping[str, Any] | None = None,
    ) -> None: ...


def tool_result_waits_for_user(result: Any) -> bool:
    """Whether a tool result asks the execution to pause for the user."""
    return (
        isinstance(result, dict)
        and str(result.get("status") or "").strip().lower() == WAITING_FOR_USER_STATUS
    )


def confirmation_grant_callable(tool: Any) -> Callable[..., None] | None:
    """Return the tool's confirmation-grant callable, if it exposes one."""
    grant = getattr(tool, "authorize_confirmation", None)
    return grant if callable(grant) else None


def strip_reserved_tool_args(args: dict[str, Any]) -> dict[str, Any]:
    """Drop runtime-reserved arguments a model may have tried to supply.

    Reserved arguments the runtime itself injects are added after this call, so
    stripping the whole prefix here cannot clobber legitimate values.
    """
    for key in [
        key
        for key in args
        if isinstance(key, str) and key.startswith(RESERVED_TOOL_ARG_PREFIX)
    ]:
        args.pop(key, None)
    return args
