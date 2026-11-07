"""Host-provided review at the runtime's user-facing delivery boundary."""

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal, Protocol

from ..user_context import current_user_id

BLOCKED_OUTPUT = "I can't share that information."
UNAVAILABLE_OUTPUT = "I couldn't complete the output safety check. Please try again."


@dataclass(frozen=True)
class OutputDecision:
    action: Literal["allow", "block", "redact"]
    text: str
    reason: str = ""


class ReviewedOutputStream(Protocol):
    async def feed(self, delta: str) -> str: ...

    def approved_suffix(self, content: str) -> str: ...


class OutputGuardrail(Protocol):
    async def check(self, text: str, *, structured: bool = False) -> OutputDecision: ...

    def open_stream(self) -> ReviewedOutputStream | None: ...


@dataclass(frozen=True)
class OutputReviewRequest:
    execution_id: str
    user_id: int | None
    delegated_agent_id: int | None = None


OutputReviewResolver = Callable[
    [OutputReviewRequest, OutputGuardrail | None],
    Awaitable[OutputGuardrail | None],
]
_resolver: OutputReviewResolver | None = None
_active_guardrail: ContextVar[OutputGuardrail | None] = ContextVar(
    "active_output_guardrail", default=None
)
_delegated_agent_id: ContextVar[int | None] = ContextVar(
    "output_review_delegated_agent_id", default=None
)


class ProtectedInteractionError(RuntimeError):
    """A rejected question must not park the run waiting for a reply."""

    def __init__(self) -> None:
        super().__init__(BLOCKED_OUTPUT)


def set_output_review_resolver(resolver: OutputReviewResolver | None) -> None:
    """Install a host policy resolver; standalone runs inherit no policy."""
    global _resolver
    _resolver = resolver


async def resolve_output_guardrail(execution_id: str) -> OutputGuardrail | None:
    inherited = _active_guardrail.get()
    if _resolver is None:
        return inherited
    request = OutputReviewRequest(
        execution_id=execution_id,
        user_id=current_user_id.get(),
        delegated_agent_id=_delegated_agent_id.get(),
    )
    return await _resolver(request, inherited) or inherited


def output_guardrail_active() -> bool:
    return _active_guardrail.get() is not None


@contextmanager
def bind_output_guardrail(guardrail: OutputGuardrail | None) -> Iterator[None]:
    token = _active_guardrail.set(guardrail)
    try:
        yield
    finally:
        _active_guardrail.reset(token)


@contextmanager
def bind_output_review_agent(agent_id: int) -> Iterator[None]:
    """Carry the authorized target through a delegated run without a Task row."""
    token = _delegated_agent_id.set(agent_id)
    try:
        yield
    finally:
        _delegated_agent_id.reset(token)
