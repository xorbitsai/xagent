"""Langfuse integration for xagent trace events."""

from __future__ import annotations

from typing import Any, Optional

from .client import (
    flush_langfuse,
    get_langfuse_client,
    initialize_langfuse,
    reset_langfuse_client,
)
from .handler import LangfuseTraceHandler


def create_langfuse_trace_handler(
    *,
    task_id: str,
    user_id: Optional[int] = None,
    trace_name: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Optional[LangfuseTraceHandler]:
    """Create a Langfuse trace handler when Langfuse is configured."""
    if get_langfuse_client() is None:
        return None

    return LangfuseTraceHandler(
        task_id=task_id,
        user_id=user_id,
        trace_name=trace_name,
        session_id=session_id,
        tags=tags,
        metadata=metadata,
    )


__all__ = [
    "LangfuseTraceHandler",
    "create_langfuse_trace_handler",
    "flush_langfuse",
    "get_langfuse_client",
    "initialize_langfuse",
    "reset_langfuse_client",
]
