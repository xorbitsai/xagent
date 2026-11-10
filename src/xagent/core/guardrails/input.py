"""Host input policy at the boundary where a user turn enters the agent."""

import asyncio
import hashlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from ..user_context import current_user_id
from .output import get_delegated_review_agent_id

BLOCKED_INPUT = (
    "I can't help with that request. Please rephrase it or ask about a supported topic."
)
UNAVAILABLE_INPUT = "I couldn't complete the input safety check. Please try again."
# Keep a hung reviewer inside the default 30s durable reply wait budget.
INPUT_REVIEW_TIMEOUT_SECONDS = 20
logger = logging.getLogger(__name__)
InputReviewSource = Literal[
    "new_run", "reply", "delegated", "builtin", "workforce_builder"
]


@dataclass(frozen=True)
class InputReviewRequest:
    """Text at the agent boundary.

    Hosts must review both the displayed `text` and `execution_text` when
    they differ. The latter can include additional context such as a task's
    appended prompt context. Attachment bytes are outside this hook, and a
    file-only reply with empty execution text does not invoke it.
    """

    execution_id: str
    user_id: int | None
    text: str = field(repr=False)
    execution_text: str = field(repr=False)
    source: InputReviewSource = "new_run"
    delegated_agent_id: int | None = None


@dataclass(frozen=True)
class InputReviewResult:
    """Verdict and prior user-message fingerprints to omit from history replay.

    Fingerprint the text persisted in the transcript (normally request.text).
    Include request.execution_text as well if either representation can be replayed.
    Refusal text is safe to display to the user.
    """

    refusal: str | None = None
    excluded_inputs: frozenset[str] = frozenset()
    unavailable: bool = False


InputReviewHook = Callable[[InputReviewRequest], Awaitable[InputReviewResult]]
_review: InputReviewHook | None = None


def set_input_review_hook(review: InputReviewHook | None) -> None:
    """Install the host text reviewer, or clear it for standalone runs."""
    global _review
    _review = review


def input_review_configured() -> bool:
    return _review is not None


async def review_input(
    execution_id: str,
    text: str,
    *,
    execution_text: str,
    review_source: InputReviewSource = "new_run",
) -> InputReviewResult:
    hook = _review
    if hook is None:
        return InputReviewResult()
    request = InputReviewRequest(
        execution_id,
        current_user_id.get(),
        text,
        execution_text,
        source=review_source,
        delegated_agent_id=get_delegated_review_agent_id(),
    )
    try:
        async with asyncio.timeout(INPUT_REVIEW_TIMEOUT_SECONDS):
            return await hook(request)
    except Exception as exc:
        # Host errors can contain the prompt. Keep exception text out of logs.
        logger.warning("Host input review failed (%s)", type(exc).__name__)
        return InputReviewResult(refusal=UNAVAILABLE_INPUT, unavailable=True)


class InputReviewRejected(ValueError):
    """A user message was denied before it was inserted into the runner."""

    def __init__(self, refusal: str, *, unavailable: bool = False) -> None:
        super().__init__(refusal)
        self.unavailable = unavailable


def input_fingerprint(text: str) -> str:
    # Keep replay exclusions stable across deployment secret rotations.
    return hashlib.sha256(text.strip().encode()).hexdigest()


def filter_input_history(
    messages: list[dict[str, Any]], excluded: frozenset[str]
) -> list[dict[str, Any]]:
    if not excluded:
        return messages
    return [
        message
        for message in messages
        if message.get("role") != "user"
        or input_fingerprint(str(message.get("content") or "")) not in excluded
    ]
