"""Declarative sandbox config metadata and resolution helpers."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Iterable, TypeVar

from ..base import Tool
from ..function import FunctionTool

_T = TypeVar("_T", bound=type)


@dataclass(frozen=True)
class SandboxConfig:
    """Sandbox enablement plus extra runtime requirements for a class."""

    enabled: bool = True
    packages: tuple[str, ...] = ()
    env_vars: tuple[str, ...] = ()
    # Returns the per-call wall-clock limit in seconds, read at call time so
    # config overrides apply. None leaves the tool unbounded.
    exec_timeout: Callable[[], int] | None = None


def sandbox_config(
    *,
    enabled: bool = True,
    packages: Iterable[str] | None = None,
    env_vars: Iterable[str] | None = None,
    exec_timeout: Callable[[], int] | None = None,
) -> Callable[[_T], _T]:
    """Attach sandbox config metadata to a class via decorator."""

    config = SandboxConfig(
        enabled=enabled,
        packages=tuple(packages or ()),
        env_vars=tuple(env_vars or ()),
        exec_timeout=exec_timeout,
    )

    def decorator(cls: _T) -> _T:
        setattr(cls, "__sandbox_config__", config)
        return cls

    return decorator


def get_sandbox_config(instance: Any) -> SandboxConfig | None:
    """Return sandbox config declared on an instance or its class, if present."""
    instance_config = getattr(instance, "__sandbox_config__", None)
    if isinstance(instance_config, SandboxConfig):
        return instance_config

    config = getattr(instance.__class__, "__sandbox_config__", None)
    if not isinstance(config, SandboxConfig):
        return None
    return config


def set_instance_sandbox_config(instance: Any, config: SandboxConfig) -> None:
    """Attach sandbox config to a single instance."""
    setattr(instance, "__sandbox_config__", config)


def extract_bound_method_target(tool: FunctionTool) -> tuple[Any, str] | None:
    """Extract bound-instance method execution target from a FunctionTool."""
    func = tool.func

    if not inspect.ismethod(func):
        return None

    instance = getattr(func, "__self__", None)
    if instance is None:
        return None

    return instance, func.__name__


def resolve_sandbox_config(tool: Tool) -> SandboxConfig | None:
    """Resolve sandbox config for a tool.

    For a direct AbstractBaseTool, reads metadata from the tool's class.
    For a bound-method FunctionTool, reads metadata from the method owner's class.
    Returns None for closures or undecorated classes.
    """
    if isinstance(tool, FunctionTool):
        function_target = extract_bound_method_target(tool)
        if function_target is None:
            return None

        instance, _ = function_target
        return get_sandbox_config(instance)

    return get_sandbox_config(tool)
