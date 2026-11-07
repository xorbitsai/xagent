"""Routing metadata for model calls made during an agent execution."""

from contextvars import ContextVar

xinference_routing_key: ContextVar[str | None] = ContextVar(
    "xinference_routing_key", default=None
)
