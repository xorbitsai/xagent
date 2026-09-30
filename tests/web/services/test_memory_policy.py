from __future__ import annotations

import ast
import pathlib
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from xagent.core.agent.service import AgentService
from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.web.memory_lifecycle import (
    MemoryLifecycleState,
    MemoryLifecycleStatus,
    MemoryUnavailableError,
)
from xagent.web.services import agent_service_manager as agent_runtime_service
from xagent.web.services.memory_policy import (
    MEMORY_POLICY_RESOLVER_FAILURE_REASON,
    MemoryPolicyDecision,
    MemoryPolicyRequest,
    set_trusted_memory_policy_resolver,
)


@pytest.fixture(autouse=True)
def clear_memory_policy_resolver() -> Iterator[None]:
    set_trusted_memory_policy_resolver(None)
    yield
    set_trusted_memory_policy_resolver(None)


def _task(*, agent_id: int | None = None, source: str | None = None) -> Any:
    return SimpleNamespace(
        id=17,
        user_id=23,
        agent_id=agent_id,
        source=source,
        agent_config={},
    )


@pytest.mark.parametrize(
    ("agent_config", "agent_id", "expected_enabled", "uses_in_memory"),
    [
        ({"is_preview": True}, None, False, True),
        ({}, 41, False, False),
        ({}, None, True, False),
    ],
    ids=("preview", "published-agent", "ordinary-task"),
)
def test_default_memory_policy_is_unchanged_without_resolver(
    monkeypatch: pytest.MonkeyPatch,
    agent_config: dict[str, Any],
    agent_id: int | None,
    expected_enabled: bool,
    uses_in_memory: bool,
) -> None:
    dynamic_store = Mock(name="dynamic-memory-store")
    get_memory_store = Mock(return_value=dynamic_store)
    monkeypatch.setattr(agent_runtime_service, "get_memory_store", get_memory_store)

    policy = agent_runtime_service.resolve_agent_service_memory_policy(
        task=_task(agent_id=agent_id),
        agent_config=agent_config,
    )

    assert policy.memory_enabled is expected_enabled
    assert policy.memory_available is True
    assert policy.memory_availability_reason is None
    if uses_in_memory:
        assert isinstance(policy.memory, InMemoryMemoryStore)
        get_memory_store.assert_not_called()
    else:
        assert policy.memory is dynamic_store
        get_memory_store.assert_called_once_with()


def test_trusted_resolver_can_enable_preview_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dynamic_store = Mock(name="dynamic-memory-store")
    monkeypatch.setattr(
        agent_runtime_service,
        "get_memory_store",
        Mock(return_value=dynamic_store),
    )
    resolver = Mock(
        return_value=MemoryPolicyDecision(
            enabled=True,
            available=True,
            reason=None,
        )
    )
    set_trusted_memory_policy_resolver(resolver)

    policy = agent_runtime_service.resolve_agent_service_memory_policy(
        task=_task(source="trusted-ingress"),
        agent_config={"is_preview": True},
    )

    assert policy.memory is dynamic_store
    assert policy.memory_enabled is True
    assert policy.memory_available is True
    assert policy.memory_availability_reason is None
    resolver.assert_called_once_with(
        MemoryPolicyRequest(
            task_id=17,
            user_id=23,
            agent_id=None,
            source="trusted-ingress",
            is_preview=True,
        )
    )


def test_trusted_resolver_can_disable_otherwise_enabled_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dynamic_store = Mock(name="dynamic-memory-store")
    monkeypatch.setattr(
        agent_runtime_service,
        "get_memory_store",
        Mock(return_value=dynamic_store),
    )
    set_trusted_memory_policy_resolver(
        lambda _request: MemoryPolicyDecision(
            enabled=False,
            available=True,
            reason="disabled_by_host_policy",
        )
    )

    policy = agent_runtime_service.resolve_agent_service_memory_policy(task=_task())

    assert policy.memory is dynamic_store
    assert policy.memory_enabled is False
    assert policy.memory_available is True
    assert policy.memory_availability_reason == "disabled_by_host_policy"


def test_trusted_resolver_can_report_unavailable_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_memory_store = Mock(name="get-memory-store")
    monkeypatch.setattr(agent_runtime_service, "get_memory_store", get_memory_store)
    set_trusted_memory_policy_resolver(
        lambda _request: MemoryPolicyDecision(
            enabled=False,
            available=False,
            reason="memory_service_unavailable",
        )
    )

    policy = agent_runtime_service.resolve_agent_service_memory_policy(task=_task())

    assert isinstance(policy.memory, InMemoryMemoryStore)
    assert policy.memory_enabled is False
    assert policy.memory_available is False
    assert policy.memory_availability_reason == "memory_service_unavailable"
    get_memory_store.assert_not_called()


def test_resolver_exception_fails_closed_for_preview() -> None:
    def fail(_request: MemoryPolicyRequest) -> MemoryPolicyDecision:
        raise RuntimeError("resolver is unavailable")

    set_trusted_memory_policy_resolver(fail)

    policy = agent_runtime_service.resolve_agent_service_memory_policy(
        task=_task(),
        agent_config={"is_preview": True},
    )

    assert isinstance(policy.memory, InMemoryMemoryStore)
    assert policy.memory_enabled is False
    assert policy.memory_available is False
    assert policy.memory_availability_reason == MEMORY_POLICY_RESOLVER_FAILURE_REASON


@pytest.mark.parametrize(
    "decision",
    [
        None,
        MemoryPolicyDecision(enabled=True, available=False, reason="unavailable"),
        MemoryPolicyDecision(enabled=False, available=False),
        MemoryPolicyDecision(enabled=True, available=True, reason=""),
        MemoryPolicyDecision(enabled=1, available=True),  # type: ignore[arg-type]
    ],
    ids=(
        "wrong-type",
        "enabled-while-unavailable",
        "unavailable-without-reason",
        "empty-reason",
        "non-bool-flag",
    ),
)
def test_invalid_resolver_decision_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    decision: object,
) -> None:
    get_memory_store = Mock(name="get-memory-store")
    monkeypatch.setattr(agent_runtime_service, "get_memory_store", get_memory_store)
    set_trusted_memory_policy_resolver(lambda _request: decision)  # type: ignore[arg-type,return-value]

    policy = agent_runtime_service.resolve_agent_service_memory_policy(task=_task())

    assert isinstance(policy.memory, InMemoryMemoryStore)
    assert policy.memory_enabled is False
    assert policy.memory_available is False
    assert policy.memory_availability_reason == MEMORY_POLICY_RESOLVER_FAILURE_REASON
    get_memory_store.assert_not_called()


def test_resolver_receives_none_for_missing_or_nonprimitive_task_fields() -> None:
    resolver = Mock(
        return_value=MemoryPolicyDecision(
            enabled=False,
            available=True,
            reason="disabled_by_host_policy",
        )
    )
    set_trusted_memory_policy_resolver(resolver)

    agent_runtime_service.resolve_agent_service_memory_policy(
        task=None, agent_config={}
    )

    resolver.assert_called_once_with(
        MemoryPolicyRequest(
            task_id=None,
            user_id=None,
            agent_id=None,
            source=None,
            is_preview=False,
        )
    )


# --------------------------------------------------------------------------
# A fenced memory state reaches the runtime that has to explain it.
#
# docs/deployment.md promises that tasks and chats keep starting while memory
# is fenced off, running with memory disabled and recording the state as their
# memory availability reason. Both AgentService call sites must therefore carry
# memory_available and memory_availability_reason, not just memory_enabled.
# --------------------------------------------------------------------------

FENCED_STATES = [
    MemoryLifecycleState.BLOCKED_REPAIR,
    MemoryLifecycleState.RESTART_REQUIRED,
    MemoryLifecycleState.CREDENTIAL_UNAVAILABLE,
    MemoryLifecycleState.RETRYABLE_UNAVAILABLE,
]


def _fenced_policy(monkeypatch: pytest.MonkeyPatch, state: MemoryLifecycleState):
    def refuse() -> Any:
        raise MemoryUnavailableError(MemoryLifecycleStatus(state))

    monkeypatch.setattr(agent_runtime_service, "get_memory_store", refuse)
    return agent_runtime_service.resolve_agent_service_memory_policy(task=_task())


@pytest.mark.parametrize("state", FENCED_STATES, ids=lambda s: s.value)
def test_fenced_memory_still_starts_a_task_and_records_the_reason(
    monkeypatch: pytest.MonkeyPatch, state: MemoryLifecycleState
) -> None:
    policy = _fenced_policy(monkeypatch, state)

    # The task still starts: an inert store, memory off, and the state on record.
    assert isinstance(policy.memory, InMemoryMemoryStore)
    assert policy.memory_enabled is False
    assert policy.memory_available is False
    assert policy.memory_availability_reason == state.value
    # Every lifecycle state is already published by /api/memory/store-info, so
    # it survives the caller-safety fold unchanged.
    assert policy.public_availability_reason == state.value
    assert policy.execution_metadata() == {
        "memory_available": False,
        "memory_availability_reason": state.value,
    }


@pytest.mark.parametrize("state", FENCED_STATES, ids=lambda s: s.value)
def test_an_agent_service_built_from_a_fenced_policy_reports_the_reason(
    monkeypatch: pytest.MonkeyPatch, state: MemoryLifecycleState
) -> None:
    """What both call sites construct, for a task and for a chat alike."""
    policy = _fenced_policy(monkeypatch, state)

    service = AgentService(
        name="fenced",
        id="fenced-agent",
        tools=[],
        enable_workspace=False,
        memory=policy.memory,
        memory_enabled=policy.memory_enabled,
        memory_available=policy.memory_available,
        memory_availability_reason=policy.public_availability_reason,
        execution_metadata=policy.execution_metadata(),
    )

    status = service.get_status()
    assert status["memory_enabled"] is False
    assert status["memory_available"] is False
    assert status["memory_availability_reason"] == state.value
    assert service.execution_metadata["memory_availability_reason"] == state.value


def test_available_memory_records_no_reason_and_no_extra_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ordinary task's status and trace are unchanged."""
    monkeypatch.setattr(
        agent_runtime_service, "get_memory_store", Mock(return_value=Mock())
    )

    policy = agent_runtime_service.resolve_agent_service_memory_policy(task=_task())

    assert policy.memory_available is True
    assert policy.public_availability_reason is None
    assert policy.execution_metadata() == {}


def test_a_host_resolver_reason_is_not_published_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host-supplied text stays operator-only and is safe before persistence."""
    set_trusted_memory_policy_resolver(
        lambda _request: MemoryPolicyDecision(
            enabled=False,
            available=False,
            reason="pgbouncer-7 in eu-west-1 refused the memory credential",
        )
    )

    policy = agent_runtime_service.resolve_agent_service_memory_policy(task=_task())

    assert policy.memory_available is False
    # Raw on the short-lived policy for operator diagnostics...
    assert policy.memory_availability_reason.startswith("pgbouncer-7")
    # ...folded before checkpoint/trace persistence and for every caller.
    assert (
        policy.execution_metadata()["memory_availability_reason"]
        == agent_runtime_service.GENERIC_MEMORY_AVAILABILITY_REASON
    )
    assert (
        policy.public_availability_reason
        == agent_runtime_service.GENERIC_MEMORY_AVAILABILITY_REASON
    )


def test_resolver_failure_reason_is_also_folded() -> None:
    set_trusted_memory_policy_resolver(Mock(side_effect=RuntimeError("boom")))

    policy = agent_runtime_service.resolve_agent_service_memory_policy(task=_task())

    assert policy.memory_availability_reason == MEMORY_POLICY_RESOLVER_FAILURE_REASON
    assert (
        policy.public_availability_reason
        == agent_runtime_service.GENERIC_MEMORY_AVAILABILITY_REASON
    )


def test_both_agent_service_call_sites_carry_the_memory_policy() -> None:
    """Pins the defect: one call site propagating it is not enough.

    The two constructions live in long async methods whose dependencies make
    an end-to-end build impractical here, so this reads the call sites
    directly rather than claiming coverage the suite does not have.
    """
    tree = ast.parse(
        pathlib.Path(agent_runtime_service.__file__).read_text(encoding="utf-8")
    )
    required = {
        "memory",
        "memory_enabled",
        "memory_available",
        "memory_availability_reason",
        "execution_metadata",
    }
    call_sites = [
        {keyword.arg for keyword in node.keywords}
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "AgentService"
    ]

    assert len(call_sites) == 2, "expected the fresh-agent and reconstruction sites"
    for keywords in call_sites:
        assert required <= keywords
