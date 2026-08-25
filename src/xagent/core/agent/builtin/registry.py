"""Registry for code-defined built-in agent specifications."""

from __future__ import annotations

from collections.abc import Iterable
from threading import RLock

from .spec import BuiltinAgentSpec


class BuiltinAgentRegistrationError(ValueError):
    """Raised when a built-in agent specification cannot be registered."""


class BuiltinAgentNotFoundError(LookupError):
    """Raised when a requested built-in agent has not been registered."""


class BuiltinAgentRegistry:
    """In-process registry whose entries are never persisted or user-edited."""

    def __init__(self, specs: Iterable[BuiltinAgentSpec] = ()) -> None:
        self._specs: dict[str, BuiltinAgentSpec] = {}
        self._lock = RLock()
        for spec in specs:
            self.register(spec)

    def register(self, spec: BuiltinAgentSpec) -> None:
        if not isinstance(spec, BuiltinAgentSpec):
            raise BuiltinAgentRegistrationError(
                "Built-in agent registry only accepts BuiltinAgentSpec instances"
            )
        with self._lock:
            if spec.name in self._specs:
                raise BuiltinAgentRegistrationError(
                    f"Built-in agent '{spec.name}' is already registered"
                )
            self._specs[spec.name] = spec

    def get(self, name: str) -> BuiltinAgentSpec | None:
        with self._lock:
            return self._specs.get(name)

    def require(self, name: str) -> BuiltinAgentSpec:
        spec = self.get(name)
        if spec is None:
            raise BuiltinAgentNotFoundError(
                f"Built-in agent '{name}' is not registered"
            )
        return spec

    def list_specs(self) -> tuple[BuiltinAgentSpec, ...]:
        with self._lock:
            return tuple(self._specs.values())

    def __contains__(self, name: object) -> bool:
        with self._lock:
            return name in self._specs


BUILTIN_AGENT_REGISTRY = BuiltinAgentRegistry()
