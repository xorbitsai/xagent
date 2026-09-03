"""Trusted host overrides for task memory eligibility."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MEMORY_POLICY_RESOLVER_FAILURE_REASON = "resolver_failed"


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
    require_persistence: bool = False
    require_vector_search: bool = False


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
    if (
        type(decision.require_persistence) is not bool
        or type(decision.require_vector_search) is not bool
    ):
        raise TypeError("memory backend requirement flags must be bool")
    if decision.enabled and not decision.available:
        raise ValueError("unavailable memory cannot be enabled")
    if (decision.require_persistence or decision.require_vector_search) and not (
        decision.enabled and decision.available
    ):
        raise ValueError("memory backend requirements need enabled, available memory")
    if decision.reason is not None and (
        not isinstance(decision.reason, str) or not decision.reason.strip()
    ):
        raise TypeError("memory policy decision reason must be a non-empty string")
    if not decision.available and decision.reason is None:
        raise ValueError("unavailable memory requires a reason")
