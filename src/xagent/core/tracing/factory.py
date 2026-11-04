"""Factory helpers for composing xagent tracers."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from ..agent.trace import TraceHandler, Tracer
from .langfuse import create_langfuse_trace_handler


def create_agent_tracer(
    *,
    handlers: Optional[Iterable[TraceHandler]] = None,
    task_id: Optional[str] = None,
    user_id: Optional[int] = None,
    trace_name: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Tracer:
    """Create a tracer with optional handlers and Langfuse export."""
    tracer = Tracer()

    for handler in handlers or ():
        tracer.add_handler(handler)

    if task_id is None:
        return tracer

    langfuse_handler = create_langfuse_trace_handler(
        task_id=task_id,
        user_id=user_id,
        trace_name=trace_name,
        session_id=session_id,
        tags=tags,
        metadata=metadata,
    )
    if langfuse_handler is not None:
        tracer.add_handler(langfuse_handler)

    return tracer
