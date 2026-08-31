from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ...context_ref import (
    CONTEXT_REFS_KEY,
    ContextReference,
    normalize_context_references,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Message:
    """Unified message format with content-based deduplication."""

    role: str
    content: str
    timestamp: datetime = field(default_factory=_utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    hidden: bool = False
    output_tokens: int | None = None
    context_refs: tuple[ContextReference, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "context_refs",
            normalize_context_references(self.context_refs),
        )

    @classmethod
    def role_system(cls, content: str, **kwargs: Any) -> "Message":
        return cls(role="system", content=content, **kwargs)

    @classmethod
    def role_user(cls, content: str, **kwargs: Any) -> "Message":
        return cls(role="user", content=content, **kwargs)

    @classmethod
    def role_assistant(cls, content: str, **kwargs: Any) -> "Message":
        return cls(role="assistant", content=content, **kwargs)

    @classmethod
    def role_tool(cls, content: str, **kwargs: Any) -> "Message":
        return cls(role="tool", content=content, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.hidden:
            result["hidden"] = True
        if self.context_refs and not self.metadata.get("superseded"):
            result[CONTEXT_REFS_KEY] = [
                reference.durable_dict() for reference in self.context_refs
            ]
        return result

    def __hash__(self) -> int:
        return hash(self._identity_key())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Message):
            return False
        return self._identity_key() == other._identity_key()

    def _identity_key(self) -> tuple[Any, ...]:
        tool_call_ids = tuple(
            tool_call.get("id")
            for tool_call in self.tool_calls or []
            if isinstance(tool_call, dict)
        )
        context_ref_keys = tuple(
            reference.identity_key() for reference in self.context_refs
        )
        return (
            self.role,
            self.content,
            tool_call_ids,
            self.tool_call_id,
            context_ref_keys,
        )

    def context_refs_text(self) -> str:
        if self.metadata.get("superseded"):
            return ""
        return "\n".join(reference.compact_text() for reference in self.context_refs)

    def context_refs_token_estimate(self) -> int:
        if self.metadata.get("superseded"):
            return 0
        return sum(reference.estimated_tokens() for reference in self.context_refs)


@dataclass
class LLMCallRecord:
    """Tracks token usage for a single LLM call.

    ``synthetic_purpose`` marks records of internal, non-conversational
    calls (currently only ``"context_compaction"``): their prompt is not
    the live conversation, so they are skipped when the context-size
    estimate picks its freshness baseline (see
    ``ExecutionContext._get_total_tokens``).
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int
    message_index: int
    prompt_message_count: int | None = None
    prompt_content_chars: int | None = None
    timestamp: datetime = field(default_factory=_utcnow)
    synthetic_purpose: str | None = None
