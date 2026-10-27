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
from ...core.memory.scope_columns import strict_user_id

logger = logging.getLogger(__name__)

MEMORY_POLICY_RESOLVER_FAILURE_REASON = "resolver_failed"
MEMORY_POLICY_BINDING_FAILURE_REASON = "binding_failed"
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
    """Explicit memory eligibility and availability decision."""

    enabled: bool
    available: bool
    reason: str | None = None
    memory_access_scope: MemoryAccessScope | None = None


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


def memory_retrieval_scope_fingerprint(
    *,
    enabled: bool,
    access_scope: MemoryAccessScope | None,
    ambient_owner_user_id: int | None,
    ambient_execution_scope: ExecutionScope | None,
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
    serialized = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
