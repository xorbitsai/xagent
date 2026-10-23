"""Trusted host overrides for task memory eligibility."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ...core.execution_scope import ExecutionScope
from ...core.memory.scope_columns import strict_user_id

logger = logging.getLogger(__name__)

MEMORY_POLICY_RESOLVER_FAILURE_REASON = "resolver_failed"


def memory_retrieval_scope_fingerprint(
    *,
    enabled: bool,
    access_scope: MemoryAccessScope | None,
    ambient_owner_user_id: int | None = None,
    ambient_execution_scope: ExecutionScope | None = None,
) -> str:
    """Return a stable, non-reversible identity for automatic retrieval scope."""

    payload: dict[str, object] = {"enabled": enabled}
    if enabled and access_scope is not None:
        payload["owner_user_id"] = access_scope.owner_user_id
        payload["memory_dimensions"] = sorted(access_scope.memory_dimensions.items())
    elif enabled:
        payload["owner_user_id"] = strict_user_id(ambient_owner_user_id)
        payload["memory_dimensions"] = sorted(
            (
                ambient_execution_scope.memory_dimensions
                if ambient_execution_scope
                else {}
            ).items()
        )
        payload["strict_memory_isolation"] = bool(
            ambient_execution_scope and ambient_execution_scope.strict_memory_isolation
        )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(
        b"xagent-memory-retrieval-scope:v1\0" + canonical.encode()
    ).hexdigest()
    return f"sha256:{digest}"


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


@dataclass(frozen=True)
class MemoryAccessScope:
    """Trusted, immutable memory identity independent of task identity.

    ``owner_user_id`` uses the same owner domain as memory metadata's
    ``user_id``: it must be a real platform ``users.id`` whose lifecycle and
    authorization are managed by the trusted host.  Arbitrary external
    principal integers are not supported because they could collide with a
    platform user.  Dimensions are mandatory so an override can never widen
    access to every memory owned by that user.
    """

    owner_user_id: int
    memory_dimensions: Mapping[str, str]

    def __post_init__(self) -> None:
        if type(self.owner_user_id) is not int:
            raise TypeError("memory owner_user_id must be an integer")
        owner_user_id = strict_user_id(self.owner_user_id)
        if owner_user_id is None or owner_user_id <= 0:
            raise ValueError("memory owner_user_id must be a positive signed int64")
        dimensions = dict(self.memory_dimensions)
        if not dimensions:
            raise ValueError("memory access scope requires memory_dimensions")
        validated = ExecutionScope(memory_dimensions=dimensions).memory_dimensions
        object.__setattr__(self, "owner_user_id", owner_user_id)
        object.__setattr__(self, "memory_dimensions", MappingProxyType(dict(validated)))


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
        logger.exception("Trusted memory policy resolver failed")
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
    if decision.memory_access_scope is not None:
        if not isinstance(decision.memory_access_scope, MemoryAccessScope):
            raise TypeError("memory access scope has an invalid type")
        if not decision.enabled or not decision.available:
            raise ValueError(
                "memory access scope requires enabled and available memory"
            )
