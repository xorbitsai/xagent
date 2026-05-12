from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .runner import AgentRunner


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Agent:
    """Minimal agent definition for the execution-centric v2 runtime."""

    name: str
    patterns: list[Any]
    tools: list[Any] = field(default_factory=list)
    llm: Any | None = None
    system_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    memory_store: Any | None = None
    memory_similarity_threshold: float | None = None
    skill_manager: Any | None = None
    allowed_skills: list[str] | None = None
    created_at: datetime = field(default_factory=_utcnow)

    def get_runner(self, **kwargs: Any) -> "AgentRunner":
        """Return a runner bound to this agent."""
        from .runner import AgentRunner

        return AgentRunner(agent=self, **kwargs)
