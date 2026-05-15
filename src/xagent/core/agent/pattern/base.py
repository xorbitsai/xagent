from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..context import ExecutionContext


@dataclass
class PatternResult:
    """Standard result envelope returned by v2 patterns."""

    success: bool
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"success": self.success}
        if self.output is not None:
            result["output"] = self.output
        if self.error is not None:
            result["error"] = self.error
        if self.metadata:
            result.update(self.metadata)
        return result


class AgentPattern(ABC):
    """Abstract interface for agent execution patterns."""

    @abstractmethod
    async def run(
        self,
        context: ExecutionContext,
        tools: list[Any],
        llm: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute the pattern against an execution context."""
