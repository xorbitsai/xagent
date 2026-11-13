"""Trusted host overrides for task memory eligibility."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ...core.execution_scope import ExecutionScope
from ...core.memory.access_scope import MemoryAccessScope
from ...core.memory.provenance import MemoryProvenance, validate_memory_provenance
from ...core.memory.scope_columns import strict_user_id

logger = logging.getLogger(__name__)

MEMORY_POLICY_RESOLVER_FAILURE_REASON = "resolver_failed"
MEMORY_POLICY_BINDING_FAILURE_REASON = "binding_failed"
MEMORY_BINDING_RESOLUTION_FAILURE_REASON = "binding_resolution_failed"
MEMORY_POLICY_TASK_MISSING_REASON = "task_missing"
MEMORY_POLICY_TASK_LOOKUP_FAILURE_REASON = "task_lookup_failed"


@dataclass(frozen=True)
class MemoryPolicyRequest:
    """Stable task fields supplied to the trusted host resolver."""

    task_id: int | None
    user_id: int | None
    agent_id: int | None
    source: str | None
    is_preview: bool


@dataclass(frozen=True)
class MemoryPolicyDecision:
    """Explicit memory eligibility and availability decision.

    ``memory_provenance`` optionally describes where memories stored during
    this task are learned (e.g. ``{"channel_kind": "dm", "channel_id": "D1"}``).
    The ``store_memory`` tool copies it into each new note's metadata under
    ``"provenance"``; see ``core.memory.provenance``. It is validated and
    frozen on construction: a value outside the ASCII identifier grammar fails
    the resolver (and therefore memory) closed. That grammar is defense in
    depth, not proof a value is not content; hosts must pass identifiers
    only. It is descriptive only and never affects which memories the task
    may read.

    Like ``memory_access_scope``, it requires enabled, available memory:
    hosts must leave ``memory_provenance`` unset (``None``) whenever they
    return ``enabled=False`` or ``available=False``, otherwise the decision
    is rejected and memory fails closed for the task.
    """

    enabled: bool
    available: bool
    reason: str | None = None
    memory_access_scope: MemoryAccessScope | None = None
    memory_provenance: MemoryProvenance | None = None

    def __post_init__(self) -> None:
        if self.memory_provenance is not None:
            object.__setattr__(
                self,
                "memory_provenance",
                validate_memory_provenance(self.memory_provenance),
            )


MemoryPolicyResolver = Callable[[MemoryPolicyRequest], MemoryPolicyDecision]

_trusted_memory_policy_resolver: MemoryPolicyResolver | None = None


def set_trusted_memory_policy_resolver(
    resolver: MemoryPolicyResolver | None,
) -> None:
    """Install or clear the process-wide resolver owned by the trusted host."""

    if resolver is not None and not callable(resolver):
        raise TypeError("memory policy resolver must be callable")
    global _trusted_memory_policy_resolver
    _trusted_memory_policy_resolver = resolver


def resolve_trusted_memory_policy(
    request: MemoryPolicyRequest,
) -> MemoryPolicyDecision | None:
    """Resolve a host override, failing closed on errors or invalid decisions."""

    resolver = _trusted_memory_policy_resolver
    if resolver is None:
        return None

    try:
        decision = resolver(request)
        _validate_decision(decision)
        return decision
    except Exception:
        logger.warning("Trusted memory policy resolver failed")
        return MemoryPolicyDecision(
            enabled=False,
            available=False,
            reason=MEMORY_POLICY_RESOLVER_FAILURE_REASON,
        )


def _validate_decision(decision: object) -> None:
    if not isinstance(decision, MemoryPolicyDecision):
        raise TypeError("memory policy resolver returned an invalid decision type")
    if type(decision.enabled) is not bool or type(decision.available) is not bool:
        raise TypeError("memory policy decision flags must be bool")
    if decision.enabled and not decision.available:
        raise ValueError("unavailable memory cannot be enabled")
    if decision.reason is not None and (
        not isinstance(decision.reason, str) or not decision.reason.strip()
    ):
        raise TypeError("memory policy decision reason must be a non-empty string")
    if not decision.available and decision.reason is None:
        raise ValueError("unavailable memory requires a reason")
    if decision.memory_access_scope is not None and not isinstance(
        decision.memory_access_scope, MemoryAccessScope
    ):
        raise TypeError("memory policy access scope must be a MemoryAccessScope")
    if decision.memory_access_scope is not None and not (
        decision.enabled and decision.available
    ):
        raise ValueError("memory access scope requires enabled, available memory")
    if decision.memory_provenance is not None and not (
        decision.enabled and decision.available
    ):
        raise ValueError("memory provenance requires enabled, available memory")


def memory_retrieval_scope_fingerprint(
    *,
    enabled: bool,
    access_scope: MemoryAccessScope | None,
    ambient_owner_user_id: int | None,
    ambient_execution_scope: ExecutionScope | None,
    backend_identity: str | None = None,
    route_generation: str | None = None,
) -> str | None:
    """Return a cache identity for the current trusted memory namespace.

    This is a deterministic cache digest, not a credential or a secrecy
    boundary.  The input is the complete namespace actually used by the
    store: a host-fixed scope when present, otherwise the ambient owner and
    execution-scope memory fields.  A disabled policy has no readable
    namespace and is represented explicitly by ``None``.
    """

    if not enabled:
        return None
    if (backend_identity is None) != (route_generation is None):
        raise ValueError(
            "backend_identity and route_generation must be supplied together"
        )
    if backend_identity is not None and (
        not isinstance(backend_identity, str) or not backend_identity.strip()
    ):
        raise TypeError("backend_identity must be a non-empty string or None")
    if route_generation is not None and (
        not isinstance(route_generation, str) or not route_generation.strip()
    ):
        raise TypeError("route_generation must be a non-empty string or None")
    owner_user_id: int | None
    if access_scope is not None:
        owner_user_id = access_scope.owner_user_id
        dimensions = dict(access_scope.memory_dimensions)
        strict = True
        kind = "fixed"
    else:
        owner_user_id = strict_user_id(ambient_owner_user_id)
        dimensions = (
            dict(ambient_execution_scope.memory_dimensions)
            if ambient_execution_scope is not None
            else {}
        )
        strict = bool(
            ambient_execution_scope is not None
            and ambient_execution_scope.strict_memory_isolation
        )
        kind = "ambient"
    if owner_user_id is None:
        return None
    payload: dict[str, Any] = {
        "version": 1,
        "kind": kind,
        "owner_user_id": owner_user_id,
        "memory_dimensions": sorted(dimensions.items()),
        "strict_memory_isolation": strict,
    }
    # Preserve the exact legacy digest when no trusted backend binding is in
    # use. Registered hosts add opaque routing fences without exposing either
    # value in execution metadata.
    if backend_identity is not None:
        payload["backend_identity"] = backend_identity
        payload["route_generation"] = route_generation
    serialized = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
