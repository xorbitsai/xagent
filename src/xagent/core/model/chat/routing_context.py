"""Routing metadata for model calls made during an agent execution."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

xinference_routing_key: ContextVar[str | None] = ContextVar(
    "xinference_routing_key", default=None
)
xinference_interactive: ContextVar[bool] = ContextVar(
    "xinference_interactive", default=False
)


@contextmanager
def xinference_interactive_requests(interactive: bool) -> Iterator[None]:
    token = xinference_interactive.set(interactive)
    try:
        yield
    finally:
        xinference_interactive.reset(token)
