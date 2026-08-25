"""Public API for code-defined built-in agents."""

from .executor import (
    BuiltinAgentCapabilityError,
    BuiltinAgentExecutor,
    BuiltinAgentModelUnavailableError,
    BuiltinModelResolver,
)
from .registry import (
    BUILTIN_AGENT_REGISTRY,
    BuiltinAgentNotFoundError,
    BuiltinAgentRegistrationError,
    BuiltinAgentRegistry,
)
from .spec import (
    BuiltinAgentPattern,
    BuiltinAgentRunContext,
    BuiltinAgentSpec,
    BuiltinToolBuilder,
)

__all__ = [
    "BUILTIN_AGENT_REGISTRY",
    "BuiltinAgentCapabilityError",
    "BuiltinAgentExecutor",
    "BuiltinAgentModelUnavailableError",
    "BuiltinAgentNotFoundError",
    "BuiltinAgentPattern",
    "BuiltinAgentRegistrationError",
    "BuiltinAgentRegistry",
    "BuiltinAgentRunContext",
    "BuiltinAgentSpec",
    "BuiltinModelResolver",
    "BuiltinToolBuilder",
]
