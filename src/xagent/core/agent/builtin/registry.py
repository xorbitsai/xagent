"""Registry for code-defined built-in agent specifications."""

from __future__ import annotations

from collections.abc import Iterable

from .spec import BuiltinAgentSpec


class BuiltinAgentRegistrationError(ValueError):
    """Raised when a built-in agent specification cannot be registered."""


class BuiltinAgentNotFoundError(LookupError):
    """Raised when a requested built-in agent has not been registered."""


class BuiltinAgentRegistry:
    """In-process registry whose entries are never persisted or user-edited."""

    def __init__(self, specs: Iterable[BuiltinAgentSpec] = ()) -> None:
        self._specs: dict[str, BuiltinAgentSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: BuiltinAgentSpec) -> None:
        if spec.name in self._specs:
            raise BuiltinAgentRegistrationError(
                f"Built-in agent '{spec.name}' is already registered"
            )
        self._specs[spec.name] = spec

    def get(self, name: str) -> BuiltinAgentSpec | None:
        return self._specs.get(name)

    def require(self, name: str) -> BuiltinAgentSpec:
        spec = self.get(name)
        if spec is None:
            raise BuiltinAgentNotFoundError(
                f"Built-in agent '{name}' is not registered"
            )
        return spec

    def list_specs(self) -> tuple[BuiltinAgentSpec, ...]:
        return tuple(self._specs.values())

    def __contains__(self, name: object) -> bool:
        return name in self._specs


BUILTIN_AGENT_REGISTRY = BuiltinAgentRegistry()
