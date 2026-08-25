"""Declarative specifications for code-defined built-in agents."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

from ...tools.adapters.vibe import Tool

BuiltinAgentPattern: TypeAlias = Literal[
    "single_call",
    "react",
    "dag_plan_execute",
    "auto",
]

_BUILTIN_EXECUTION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True, slots=True)
class BuiltinAgentRunContext:
    """Execution-scoped dependencies available to built-in agent factories.

    Identity and request inputs are immutable; ``artifacts`` is the deliberate
    mutable output channel shared with execution-scoped tool builders.
    """

    execution_id: str
    request_context: Mapping[str, Any] = field(default_factory=dict)
    tracer: Any | None = None
    workspace_base_dir: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _BUILTIN_EXECUTION_ID_RE.fullmatch(self.execution_id):
            raise ValueError(
                "Built-in agent execution_id must contain only letters, digits, "
                "dots, underscores, or hyphens"
            )
        object.__setattr__(
            self,
            "request_context",
            MappingProxyType(dict(self.request_context)),
        )


BuiltinToolBuilderResult: TypeAlias = Sequence[Tool] | Awaitable[Sequence[Tool]]
BuiltinToolBuilder: TypeAlias = Callable[
    [BuiltinAgentRunContext], BuiltinToolBuilderResult
]

_BUILTIN_AGENT_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_SUPPORTED_PATTERNS = frozenset({"single_call", "react", "dag_plan_execute", "auto"})


@dataclass(frozen=True, slots=True)
class BuiltinAgentSpec:
    """Immutable definition for an internal, code-owned agent.

    Built-in agents are deliberately separate from database-backed agents and
    Agent Builder. Their capabilities are opt-in and disabled by default.
    """

    name: str
    version: str
    system_prompt: str
    pattern: BuiltinAgentPattern = "single_call"
    model_role: str = "general"
    build_tools: BuiltinToolBuilder | None = None
    memory_enabled: bool = False
    skills_enabled: bool = False
    workspace_enabled: bool = False

    def __post_init__(self) -> None:
        if not _BUILTIN_AGENT_NAME_RE.fullmatch(self.name):
            raise ValueError("Built-in agent name must match ^[a-z][a-z0-9_-]*$")
        if not self.version.strip():
            raise ValueError("Built-in agent version must not be empty")
        if not self.system_prompt.strip():
            raise ValueError("Built-in agent system_prompt must not be empty")
        if self.pattern not in _SUPPORTED_PATTERNS:
            raise ValueError(f"Unsupported built-in agent pattern: {self.pattern}")
        if not self.model_role.strip():
            raise ValueError("Built-in agent model_role must not be empty")
