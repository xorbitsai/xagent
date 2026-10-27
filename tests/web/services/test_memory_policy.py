from __future__ import annotations

import ast
import pathlib
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from xagent.core.agent.context.memory_retrieval import (
    MEMORY_RETRIEVAL_SCOPE_FINGERPRINT_METADATA_KEY,
)
from xagent.core.agent.service import AgentService
from xagent.core.execution_scope import ExecutionScope, scope_fingerprint
from xagent.core.memory.access_scope import MemoryAccessScope
from xagent.core.memory.core import MemoryNote
from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.web.memory_lifecycle import (
    MemoryLifecycleState,
    MemoryLifecycleStatus,
    MemoryUnavailableError,
)
from xagent.web.services import agent_service_manager as agent_runtime_service
from xagent.web.services.memory_policy import (
    MEMORY_POLICY_BINDING_FAILURE_REASON,
    MEMORY_POLICY_RESOLVER_FAILURE_REASON,
    MemoryPolicyDecision,
    MemoryPolicyRequest,
    set_trusted_memory_policy_resolver,
)
from xagent.web.services.task_setup_snapshot import MemoryPolicyTaskFields
from xagent.web.user_isolated_memory import UserIsolatedMemoryStore


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


def test_trusted_resolver_binds_fixed_scope_and_fingerprints_complete_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dynamic_store = UserIsolatedMemoryStore(InMemoryMemoryStore())
    monkeypatch.setattr(
        agent_runtime_service,
        "get_memory_store",
        Mock(return_value=dynamic_store),
    )
    scope = MemoryAccessScope(
        owner_user_id=700,
        memory_dimensions={"tenant": "acme", "principal": "person-1"},
    )
    set_trusted_memory_policy_resolver(
        lambda _request: MemoryPolicyDecision(
            enabled=True,
            available=True,
            memory_access_scope=scope,
        )
    )

    policy = agent_runtime_service.resolve_agent_service_memory_policy(
        task=_task(source="trusted-ingress")
    )
    metadata = policy.execution_metadata(
        ambient_owner_user_id=23,
        ambient_execution_scope=ExecutionScope(
            memory_dimensions={"principal": "ambient"}
        ),
    )

    assert policy.memory is not dynamic_store
    assert policy.memory_access_scope is scope
    fingerprint = metadata[MEMORY_RETRIEVAL_SCOPE_FINGERPRINT_METADATA_KEY]
    assert isinstance(fingerprint, str) and len(fingerprint) == 64
    reordered = MemoryAccessScope(
        owner_user_id=700,
        memory_dimensions={"principal": "person-1", "tenant": "acme"},
    )
    assert fingerprint == agent_runtime_service.memory_retrieval_scope_fingerprint(
        enabled=True,
        access_scope=reordered,
        ambient_owner_user_id=999,
        ambient_execution_scope=None,
    )

    note = MemoryNote(content="remember this preference")
    assert policy.memory.add(note).success
    assert policy.memory.search("preference") == [note]
    updated = MemoryNote(id=note.id, content="updated preference")
    assert policy.memory.update(updated).success
    assert policy.memory.get(note.id).content.content == "updated preference"
    assert policy.memory.delete(note.id).success
    assert not policy.memory.get(note.id).success


def test_scope_binding_failure_disables_memory_without_publishing_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_runtime_service,
        "get_memory_store",
        Mock(return_value=InMemoryMemoryStore()),
    )
    set_trusted_memory_policy_resolver(
        lambda _request: MemoryPolicyDecision(
            enabled=True,
            available=True,
            memory_access_scope=MemoryAccessScope(
                owner_user_id=700,
                memory_dimensions={"principal": "person-1"},
            ),
        )
    )

    policy = agent_runtime_service.resolve_agent_service_memory_policy(task=_task())

    assert policy.memory_enabled is False
    assert policy.memory_available is False
    assert policy.memory_availability_reason == MEMORY_POLICY_BINDING_FAILURE_REASON
    metadata = policy.execution_metadata(
        ambient_owner_user_id=23,
        ambient_execution_scope=None,
    )
    assert metadata[MEMORY_RETRIEVAL_SCOPE_FINGERPRINT_METADATA_KEY] is None
    assert metadata["memory_availability_reason"] == (
        agent_runtime_service.GENERIC_MEMORY_AVAILABILITY_REASON
    )


def test_ambient_fingerprint_covers_owner_dimensions_and_strictness() -> None:
    fingerprints = {
        agent_runtime_service.memory_retrieval_scope_fingerprint(
            enabled=True,
            access_scope=None,
            ambient_owner_user_id=owner,
            ambient_execution_scope=scope,
        )
        for owner, scope in (
            (23, ExecutionScope(memory_dimensions={"principal": "one"})),
            (24, ExecutionScope(memory_dimensions={"principal": "one"})),
            (23, ExecutionScope(memory_dimensions={"principal": "two"})),
            (
                23,
                ExecutionScope(
                    memory_dimensions={"principal": "one"},
                    strict_memory_isolation=True,
                ),
            ),
        )
    }

    assert len(fingerprints) == 4
    assert None not in fingerprints


@pytest.mark.asyncio
async def test_cache_hit_reloads_minimal_eligibility_and_can_disable_then_reenable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistent = UserIsolatedMemoryStore(InMemoryMemoryStore())
    monkeypatch.setattr(
        agent_runtime_service,
        "get_memory_store",
        Mock(return_value=persistent),
    )
    service = AgentService(
        name="cached",
        id="web_task_17",
        task_id="17",
        tools=[],
        enable_workspace=False,
        memory=persistent,
        memory_enabled=True,
        execution_metadata={MEMORY_RETRIEVAL_SCOPE_FINGERPRINT_METADATA_KEY: "old"},
    )
    service._execution_adapter = service._build_execution_adapter()
    manager = agent_runtime_service.AgentServiceManager()
    manager._agents[17] = service
    manager._agent_owner_ids[17] = 23
    manager._agent_scope_fingerprints[17] = scope_fingerprint(None)
    state = {"agent_id": None, "agent_config": {"is_preview": True}}
    reads: list[int] = []

    async def load_policy_fields(
        task_id: int,
        task_owner_user_id: int | None,
        caller_db: Any,
    ) -> MemoryPolicyTaskFields:
        reads.append(task_id)
        assert task_owner_user_id == 23
        assert caller_db is None
        return MemoryPolicyTaskFields(
            id=task_id,
            user_id=23,
            agent_id=state["agent_id"],
            source="slack",
            agent_config=state["agent_config"],
        )

    monkeypatch.setattr(
        agent_runtime_service,
        "_load_memory_policy_task_fields_for_agent",
        load_policy_fields,
    )

    returned = await manager.get_agent_for_task(
        17,
        task_owner_user_id=23,
        resolved_execution_scope=None,
    )
    assert returned is service
    assert service.memory_enabled is False
    assert service._execution_adapter.config.memory_store is None
    assert (
        service.execution_metadata[MEMORY_RETRIEVAL_SCOPE_FINGERPRINT_METADATA_KEY]
        is None
    )

    state["agent_id"] = 41
    state["agent_config"] = {}
    await manager.get_agent_for_task(
        17,
        task_owner_user_id=23,
        resolved_execution_scope=None,
    )
    assert service.memory_enabled is False

    state["agent_id"] = None
    await manager.get_agent_for_task(
        17,
        task_owner_user_id=23,
        resolved_execution_scope=None,
    )
    assert service.memory_enabled is True
    assert service.memory is persistent
    assert service._execution_adapter.config.memory_store is persistent
    assert isinstance(
        service.execution_metadata[MEMORY_RETRIEVAL_SCOPE_FINGERPRINT_METADATA_KEY],
        str,
    )
    assert reads == [17, 17, 17]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        (None, "unavailable"),
        (RuntimeError("database unavailable"), "unavailable"),
    ],
)
async def test_cache_hit_missing_or_failed_task_lookup_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    fields: object,
    reason: str,
) -> None:
    persistent = UserIsolatedMemoryStore(InMemoryMemoryStore())
    service = AgentService(
        name="cached",
        id="web_task_17",
        tools=[],
        enable_workspace=False,
        memory=persistent,
        memory_enabled=True,
    )
    manager = agent_runtime_service.AgentServiceManager()
    manager._agents[17] = service
    manager._agent_owner_ids[17] = 23
    manager._agent_scope_fingerprints[17] = scope_fingerprint(None)

    async def load_policy_fields(*_args: Any, **_kwargs: Any) -> Any:
        if isinstance(fields, BaseException):
            raise fields
        return fields

    monkeypatch.setattr(
        agent_runtime_service,
        "_load_memory_policy_task_fields_for_agent",
        load_policy_fields,
    )

    await manager.get_agent_for_task(
        17,
        task_owner_user_id=23,
        resolved_execution_scope=None,
    )

    assert service.memory_enabled is False
    assert service.memory_available is False
    assert service.memory_availability_reason == reason


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
        MemoryPolicyDecision(
            enabled=True,
            available=True,
            memory_access_scope={"owner_user_id": 7},  # type: ignore[arg-type]
        ),
        MemoryPolicyDecision(
            enabled=False,
            available=True,
            memory_access_scope=MemoryAccessScope(
                owner_user_id=7,
                memory_dimensions={"principal": "person-1"},
            ),
        ),
    ],
    ids=(
        "wrong-type",
        "enabled-while-unavailable",
        "unavailable-without-reason",
        "empty-reason",
        "non-bool-flag",
        "malformed-access-scope",
        "scope-while-disabled",
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
