from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


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
        return (self.role, self.content, tool_call_ids, self.tool_call_id)


@dataclass
class LLMCallRecord:
    """Tracks token usage for a single LLM call."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    message_index: int
    prompt_message_count: int | None = None
    prompt_content_chars: int | None = None
    timestamp: datetime = field(default_factory=_utcnow)
