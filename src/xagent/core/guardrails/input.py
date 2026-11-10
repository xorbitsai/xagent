"""Host input policy at the boundary where a user turn enters the agent."""

import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..user_context import current_user_id

BLOCKED_INPUT = (
    "I can’t help with that request. Please rephrase it or ask about a supported topic."
)
UNAVAILABLE_INPUT = "I couldn't complete the input safety check. Please try again."


@dataclass(frozen=True)
class InputReviewRequest:
    execution_id: str
    user_id: int | None
    text: str = field(repr=False)
    execution_text: str = field(repr=False)


@dataclass(frozen=True)
class InputReviewResult:
    refusal: str | None = None
    excluded_inputs: frozenset[str] = frozenset()


InputReviewHook = Callable[[InputReviewRequest], Awaitable[InputReviewResult]]
_review: InputReviewHook | None = None


def set_input_review_hook(review: InputReviewHook | None) -> None:
    global _review
    _review = review


async def review_input(
    execution_id: str,
    text: str,
    *,
    execution_text: str,
) -> InputReviewResult:
    if _review is None:
        return InputReviewResult()
    return await _review(
        InputReviewRequest(execution_id, current_user_id.get(), text, execution_text)
    )


class InputReviewRejected(ValueError):
    """A user message was denied before it was inserted into the runner."""

    @property
    def unavailable(self) -> bool:
        return str(self) == UNAVAILABLE_INPUT


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
