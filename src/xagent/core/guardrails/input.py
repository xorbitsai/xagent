"""Host-provided review of user input before the runtime accepts it.

A withheld message is not rejected: the runner still accepts the turn, but the
model only ever sees a fixed placeholder that says the message was withheld and
why. This keeps every host's message-delivery protocol unchanged and lets the
agent answer naturally, while the original text never reaches a model prompt.

Only the user-authored text is reviewed: a request's display message when it
has one, otherwise its task or message. Host-added enrichment of the execution
text (uploaded-file context, prompt blocks, channel thread history) is not
reviewed; like tool results it belongs to the indirect-input scope.

Hosts that persist their own transcript must apply the same substitution when
they replay it. ``INPUT_WITHHELD_METADATA_KEY`` marks the accepted ``Message``,
and the user-message trace event carries the same key with the reason.
"""

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal, Protocol, get_args

from ..user_context import current_user_id
from .output import OutputReviewRequest

INPUT_REVIEW_TIMEOUT_SECONDS = 60
INPUT_WITHHELD_METADATA_KEY = "input_withheld"

InputWithholdReason = Literal["platform_policy", "agent_policy", "unavailable"]

# Written as instructions to the agent: the placeholder is the whole user turn
# it sees, so it must say what happened and how to answer. It never carries the
# reviewer's rationale, which may quote the withheld text.
_PLACEHOLDERS: dict[InputWithholdReason, str] = {
    "platform_policy": (
        "[The user's message was withheld by the platform's input safety policy "
        "and its content is not available to you. Tell the user the message "
        "could not be processed and suggest rephrasing it.]"
    ),
    "agent_policy": (
        "[The user's message was withheld because it is outside this agent's "
        "permitted scope, and its content is not available to you. Tell the "
        "user the request could not be processed and briefly say what you can "
        "help with.]"
    ),
    "unavailable": (
        "[The user's message could not be checked by the input safety policy, "
        "so its content is not available to you. Tell the user it could not "
        "be processed right now and ask them to try again.]"
    ),
}


@dataclass(frozen=True)
class InputDecision:
    """A review verdict; ``withheld`` is ``None`` when the input is allowed."""

    withheld: InputWithholdReason | None = None


class InputGuardrail(Protocol):
    """Review one user message before the runtime accepts it.

    Only the text is reviewed. Attachments of a withheld message are dropped
    from the model context, but an allowed message's files are not inspected.
    """

    async def check(self, text: str) -> InputDecision: ...


InputReviewResolver = Callable[[OutputReviewRequest], Awaitable[InputGuardrail | None]]
_resolver: InputReviewResolver | None = None
_agent_generated: ContextVar[bool] = ContextVar(
    "input_review_agent_generated", default=False
)


def set_input_review_resolver(resolver: InputReviewResolver | None) -> None:
    """Install a host policy resolver; standalone runs review no input."""
    global _resolver
    _resolver = resolver


@contextmanager
def bind_agent_generated_input() -> Iterator[None]:
    """Mark runs started from inside a pattern as carrying agent-written input.

    Delegated agents and builtin helpers receive text composed by the parent
    agent, not by the user, so they skip input review.
    """
    token = _agent_generated.set(True)
    try:
        yield
    finally:
        _agent_generated.reset(token)


def withheld_input_placeholder(reason: InputWithholdReason) -> str:
    return _PLACEHOLDERS[reason]


def parse_withhold_reason(value: Any) -> InputWithholdReason | None:
    """Read a withhold reason back from a trace, row or fact payload."""
    return value if value in get_args(InputWithholdReason) else None


async def review_user_input(execution_id: str, text: str) -> InputWithholdReason | None:
    """Return why ``text`` must be withheld, or ``None`` to accept it as is.

    Resolver failures propagate so the caller aborts before writing anything,
    matching output review. A failing reviewer withholds the input instead:
    the caller has a policy to enforce but no verdict.
    """
    if _resolver is None or _agent_generated.get() or not text.strip():
        return None
    guardrail = await _resolver(
        OutputReviewRequest(execution_id, current_user_id.get())
    )
    if guardrail is None:
        return None
    try:
        async with asyncio.timeout(INPUT_REVIEW_TIMEOUT_SECONDS):
            decision = await guardrail.check(text)
    except Exception:
        # Reviewer failures may include the candidate in their exception.
        return "unavailable"
    return decision.withheld
