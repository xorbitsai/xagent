"""Opt-in trusted host bindings for memory backends.

The registration hook is process initialization state only.  It stores a
resolver function, never a current principal or store.  Each resolution gets
an immutable request snapshot and must return an immutable binding snapshot.
Consumer wiring lives outside this module so an unregistered host keeps the
legacy memory manager, policy, lifecycle, and tool behavior unchanged.
"""

from __future__ import annotations

import inspect
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias

from ...core.memory.access_scope import MemoryAccessScope, bind_memory_access_scope
from ...core.memory.base import MemoryStore
from ...core.memory.provenance import MemoryProvenance
from ...core.memory.scope_columns import strict_user_id
from .db_runtime import run_db_io_cancellation_safe

logger = logging.getLogger(__name__)

PublicMemoryDiagnosticValue: TypeAlias = str | int | float | bool | None


class MemoryBindingResolutionError(RuntimeError):
    """A registered trusted resolver failed to produce a safe binding."""


def _optional_positive_id(value: int | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer or None")
    validated = strict_user_id(value)
    if validated is None or validated <= 0:
        raise ValueError(f"{field_name} must be a positive signed int64 or None")
    return validated


@dataclass(frozen=True)
class MemoryTaskBindingRequest:
    """Detached trusted task projection used to select a turn binding.

    These values must come from persisted server-side task state.  Request
    headers, query parameters, model text, and ambient context are not valid
    sources for this projection.
    """

    task_id: int | None
    user_id: int | None
    agent_id: int | None
    source: str | None
    is_preview: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "task_id",
            _optional_positive_id(self.task_id, field_name="task_id"),
        )
        object.__setattr__(
            self,
            "user_id",
            _optional_positive_id(self.user_id, field_name="user_id"),
        )
        object.__setattr__(
            self,
            "agent_id",
            _optional_positive_id(self.agent_id, field_name="agent_id"),
        )
        if self.source is not None and not isinstance(self.source, str):
            raise TypeError("source must be a string or None")
        if type(self.is_preview) is not bool:
            raise TypeError("is_preview must be a bool")


@dataclass(frozen=True)
class MemoryManagementBindingRequest:
    """Authenticated principal supplied to one management HTTP request."""

    authenticated_user_id: int

    def __post_init__(self) -> None:
        user_id = _optional_positive_id(
            self.authenticated_user_id,
            field_name="authenticated_user_id",
        )
        if user_id is None:  # Kept explicit for type narrowing and mutation tests.
            raise ValueError("authenticated_user_id is required")
        object.__setattr__(self, "authenticated_user_id", user_id)


MemoryBindingRequest: TypeAlias = (
    MemoryTaskBindingRequest | MemoryManagementBindingRequest
)


@dataclass(frozen=True)
class MemoryToolAssemblyRequest:
    """Turn-local inputs for an opt-in host memory-tool factory.

    The factory is called synchronously during pattern setup and must not
    perform blocking I/O. Core copies the returned sequence before adding
    run-local tools and marks the context for memory-usage guidance.

    ``provenance`` is a plain-dict copy of the task's validated
    ``MemoryPolicyDecision`` provenance (``None`` when the policy resolver
    supplied none). A factory
    that builds its own ``store_memory`` tool should forward it so notes carry
    the same provenance as the default tool set.
    """

    memory_store: MemoryStore = field(repr=False)
    task: str
    runtime: Any | None = field(default=None, repr=False)
    context: Any | None = field(default=None, repr=False)
    provenance: MemoryProvenance | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.memory_store, MemoryStore):
            raise TypeError("memory_store must be a MemoryStore")
        if not isinstance(self.task, str):
            raise TypeError("task must be a string")


MemoryToolFactory: TypeAlias = Callable[[MemoryToolAssemblyRequest], Sequence[Any]]


@dataclass(frozen=True)
class MemoryStoreBinding:
    """Immutable snapshot selecting one existing backend for one request/turn.

    ``backend_identity`` and ``route_generation`` are opaque cache-fencing
    values.  They are not public diagnostics or credentials.  A supplied
    ``access_scope`` is bound immediately, so an unsupported composition fails
    before the snapshot can escape the trusted resolver boundary.
    """

    store: MemoryStore = field(repr=False)
    backend_identity: str = field(repr=False)
    route_generation: str = field(repr=False)
    access_scope: MemoryAccessScope | None = None
    public_diagnostics: Mapping[str, PublicMemoryDiagnosticValue] = field(
        default_factory=dict
    )
    memory_tools_factory: MemoryToolFactory | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.store, MemoryStore):
            raise TypeError("memory binding store must be a MemoryStore")
        for field_name, identity_value in (
            ("backend_identity", self.backend_identity),
            ("route_generation", self.route_generation),
        ):
            if not isinstance(identity_value, str) or not identity_value.strip():
                raise TypeError(f"{field_name} must be a non-empty string")
        if self.access_scope is not None and not isinstance(
            self.access_scope, MemoryAccessScope
        ):
            raise TypeError("access_scope must be a MemoryAccessScope or None")
        if not isinstance(self.public_diagnostics, Mapping):
            raise TypeError("public_diagnostics must be a mapping")
        diagnostics = dict(self.public_diagnostics)
        for key, value in diagnostics.items():
            if not isinstance(key, str) or not key.strip():
                raise TypeError("public diagnostic keys must be non-empty strings")
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise TypeError("public diagnostic values must be scalar JSON values")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("public diagnostic floats must be finite")
        if self.memory_tools_factory is not None and not callable(
            self.memory_tools_factory
        ):
            raise TypeError("memory_tools_factory must be callable or None")

        store = self.store
        if self.access_scope is not None:
            store = bind_memory_access_scope(store, self.access_scope)
        object.__setattr__(self, "store", store)
        object.__setattr__(
            self,
            "public_diagnostics",
            MappingProxyType(diagnostics),
        )


MemoryBindingResolver: TypeAlias = Callable[[MemoryBindingRequest], MemoryStoreBinding]

_trusted_memory_binding_resolver: MemoryBindingResolver | None = None


def set_trusted_memory_binding_resolver(
    resolver: MemoryBindingResolver | None,
) -> None:
    """Install or clear the trusted host resolver during process setup."""

    if resolver is not None and not callable(resolver):
        raise TypeError("memory binding resolver must be callable")
    global _trusted_memory_binding_resolver
    _trusted_memory_binding_resolver = resolver


def resolve_trusted_memory_binding(
    request: MemoryBindingRequest,
) -> MemoryStoreBinding | None:
    """Resolve one snapshot; return None only when no hook is registered.

    Once a hook is registered, errors and invalid values raise
    ``MemoryBindingResolutionError``.  Consumers must not turn that failure
    into a legacy/global-store fallback.
    """

    if not isinstance(
        request,
        (MemoryTaskBindingRequest, MemoryManagementBindingRequest),
    ):
        raise TypeError("request must be a trusted memory binding request")
    resolver = _trusted_memory_binding_resolver
    if resolver is None:
        return None
    try:
        binding = resolver(request)
        if inspect.isawaitable(binding):
            if inspect.iscoroutine(binding):
                binding.close()
            raise TypeError("memory binding resolver must be synchronous")
        if not isinstance(binding, MemoryStoreBinding):
            raise TypeError("memory binding resolver returned an invalid binding")
        return binding
    except Exception as exc:
        traceback_frame = exc.__traceback__
        while traceback_frame is not None and traceback_frame.tb_next is not None:
            traceback_frame = traceback_frame.tb_next
        location = (
            "unknown"
            if traceback_frame is None
            else (
                f"{Path(traceback_frame.tb_frame.f_code.co_filename).name}:"
                f"{traceback_frame.tb_frame.f_code.co_name}:"
                f"{traceback_frame.tb_lineno}"
            )
        )
        logger.warning(
            "Trusted memory binding resolver failed (%s at %s)",
            type(exc).__name__,
            location,
        )
        raise MemoryBindingResolutionError(
            "trusted memory binding resolution failed"
        ) from None


async def resolve_trusted_memory_binding_async(
    request: MemoryBindingRequest,
) -> MemoryStoreBinding | None:
    """Resolve a synchronous host callback without blocking the event loop."""

    return await run_db_io_cancellation_safe(
        lambda: resolve_trusted_memory_binding(request)
    )


__all__ = [
    "MemoryBindingRequest",
    "MemoryBindingResolutionError",
    "MemoryBindingResolver",
    "MemoryManagementBindingRequest",
    "MemoryStoreBinding",
    "MemoryTaskBindingRequest",
    "MemoryToolAssemblyRequest",
    "MemoryToolFactory",
    "PublicMemoryDiagnosticValue",
    "resolve_trusted_memory_binding",
    "resolve_trusted_memory_binding_async",
    "set_trusted_memory_binding_resolver",
]
