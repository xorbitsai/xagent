from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...skills.manager import SkillManager
    from ..memory import MemoryStore
    from ..model.chat.basic.base import BaseLLM
    from .pattern import AgentPattern
    from .runner import AgentRunner


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Agent:
    """Minimal agent definition for the execution-centric runtime."""

    name: str
    patterns: list["AgentPattern"]
    tools: list[Any] = field(default_factory=list)
    llm: "BaseLLM | None" = None
    compact_llm: "BaseLLM | None" = None
    system_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    memory_store: "MemoryStore | None" = None
    memory_similarity_threshold: float | None = None
    skill_manager: "SkillManager | None" = None
    allowed_skills: list[str] | None = None
    created_at: datetime = field(default_factory=_utcnow)

    def get_runner(self, **kwargs: Any) -> "AgentRunner":
        """Return a runner bound to this agent."""
        from .runner import AgentRunner

        return AgentRunner(agent=self, **kwargs)
