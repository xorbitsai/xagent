from __future__ import annotations

import ast
import asyncio
import pathlib
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from xagent.core.agent.context.enrichment import (
    MEMORY_RETRIEVAL_SCOPE_FINGERPRINT_METADATA_KEY,
    enrich_context_with_memory,
)
from xagent.core.agent.service import AgentService
from xagent.core.execution_scope import (
    ExecutionScope,
    ExecutionScopeContext,
    get_execution_scope,
)
from xagent.core.memory.core import MemoryNote
from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.core.user_context import current_user_id
from xagent.web.memory_lifecycle import (
    MemoryLifecycleState,
    MemoryLifecycleStatus,
    MemoryUnavailableError,
)
from xagent.web.revocable_memory_store import RevocableMemoryStore
from xagent.web.services import agent_service_manager as agent_runtime_service
from xagent.web.services.memory_policy import (
    MEMORY_POLICY_RESOLVER_FAILURE_REASON,
    MemoryAccessScope,
    MemoryPolicyDecision,
    MemoryPolicyRequest,
    memory_retrieval_scope_fingerprint,
    set_trusted_memory_policy_resolver,
)
from xagent.web.user_isolated_memory import UserContext, UserIsolatedMemoryStore


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


def test_memory_access_scope_is_complete_and_immutable() -> None:
    dimensions = {"principal": "person-1"}
    scope = MemoryAccessScope(owner_user_id=17, memory_dimensions=dimensions)
    dimensions["principal"] = "forged"

    assert scope.owner_user_id == 17
    assert dict(scope.memory_dimensions) == {"principal": "person-1"}
    with pytest.raises(TypeError):
        scope.memory_dimensions["principal"] = "forged"  # type: ignore[index]
    with pytest.raises(ValueError, match="requires memory_dimensions"):
        MemoryAccessScope(owner_user_id=17, memory_dimensions={})
    with pytest.raises(ValueError, match="positive signed int64"):
        MemoryAccessScope(owner_user_id=0, memory_dimensions={"principal": "p"})
    with pytest.raises(TypeError, match="must be an integer"):
        MemoryAccessScope(  # type: ignore[arg-type]
            owner_user_id="17",
            memory_dimensions={"principal": "p"},
        )


def test_trusted_scope_binds_owner_and_dimensions_over_ambient_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Publication:
        generation = 1

        def publication_generation(self) -> int:
            return self.generation

        def published_status(self) -> MemoryLifecycleStatus:
            return MemoryLifecycleStatus(MemoryLifecycleState.RESTART_REQUIRED)

    base = InMemoryMemoryStore()
    publication = Publication()
    published = RevocableMemoryStore(
        UserIsolatedMemoryStore(base), publication, publication.generation
    )
    scope = MemoryAccessScope(
        owner_user_id=17,
        memory_dimensions={"principal": "person-1"},
    )
    monkeypatch.setattr(
        agent_runtime_service, "get_memory_store", Mock(return_value=published)
    )
    set_trusted_memory_policy_resolver(
        lambda _request: MemoryPolicyDecision(
            enabled=True,
            available=True,
            memory_access_scope=scope,
        )
    )

    policy = agent_runtime_service.resolve_agent_service_memory_policy(task=_task())
    with (
        UserContext(999),
        ExecutionScopeContext(
            ExecutionScope(memory_dimensions={"principal": "attacker"})
        ),
    ):
        note = MemoryNote(
            content="fixed",
            metadata={
                "user_id": 999,
                "execution_scope_principal": "attacker",
            },
        )
        assert policy.memory.add(note).success
        assert policy.memory.get(note.id).success
        assert policy.memory.search(
            "fixed",
            filters={
                "metadata": {
                    "user_id": 999,
                    "execution_scope_principal": "attacker",
                }
            },
        ) == [note]

    assert note.metadata["user_id"] == 17
    assert note.metadata["execution_scope_principal"] == "person-1"
    publication.generation += 1
    with pytest.raises(MemoryUnavailableError):
        policy.memory.search("fixed")


def test_unsupported_store_composition_fails_closed(
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
                owner_user_id=17,
                memory_dimensions={"principal": "person-1"},
            ),
        )
    )

    policy = agent_runtime_service.resolve_agent_service_memory_policy(task=_task())

    assert isinstance(policy.memory, InMemoryMemoryStore)
    assert policy.memory_enabled is False
    assert policy.memory_available is False
    assert policy.memory_availability_reason == MEMORY_POLICY_RESOLVER_FAILURE_REASON


@pytest.mark.asyncio
async def test_fixed_scope_enrichment_preserves_ambient_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = InMemoryMemoryStore()
    published = UserIsolatedMemoryStore(base)
    monkeypatch.setattr(
        agent_runtime_service, "get_memory_store", Mock(return_value=published)
    )
    set_trusted_memory_policy_resolver(
        lambda _request: MemoryPolicyDecision(
            enabled=True,
            available=True,
            memory_access_scope=MemoryAccessScope(
                owner_user_id=17,
                memory_dimensions={"principal": "person-1"},
            ),
        )
    )
    policy = agent_runtime_service.resolve_agent_service_memory_policy(task=_task())
    assert policy.memory.add(MemoryNote(content="likes jasmine tea")).success
    ambient_scope = ExecutionScope(memory_dimensions={"principal": "attacker"})
    context = SimpleNamespace(metadata={}, execution_id="task-17")

    with UserContext(999), ExecutionScopeContext(ambient_scope):
        memories = await enrich_context_with_memory(
            context=context,
            query="jasmine",
            category="general",
            memory_store=policy.memory,
        )
        assert current_user_id.get() == 999
        assert get_execution_scope() is ambient_scope

    assert [memory["content"] for memory in memories] == ["likes jasmine tea"]


@pytest.mark.asyncio
async def test_fixed_scope_is_shared_across_concurrent_task_identities() -> None:
    store = UserIsolatedMemoryStore(
        InMemoryMemoryStore(),
        access_scope=MemoryAccessScope(
            owner_user_id=17,
            memory_dimensions={"principal": "person-1"},
        ),
    )
    assert store.add(MemoryNote(content="shared preference")).success

    async def retrieve(task_owner: int, task_principal: str) -> list[dict[str, Any]]:
        ambient = ExecutionScope(memory_dimensions={"principal": task_principal})
        with UserContext(task_owner), ExecutionScopeContext(ambient):
            result = await enrich_context_with_memory(
                context=SimpleNamespace(metadata={}, execution_id=str(task_owner)),
                query="preference",
                category="general",
                memory_store=store,
            )
            assert current_user_id.get() == task_owner
            assert get_execution_scope() is ambient
            return result

    results = await asyncio.gather(retrieve(101, "a"), retrieve(202, "b"))
    assert [[item["content"] for item in result] for result in results] == [
        ["shared preference"],
        ["shared preference"],
    ]


@pytest.mark.asyncio
async def test_fixed_scope_exception_preserves_task_identity() -> None:
    class FailingStore(InMemoryMemoryStore):
        seen_filters: dict[str, Any] | None = None

        def search(self, query: str, **kwargs: Any) -> list[MemoryNote]:
            self.seen_filters = kwargs["filters"]
            raise RuntimeError("search failed")

    base = FailingStore()
    store = UserIsolatedMemoryStore(
        base,
        access_scope=MemoryAccessScope(
            owner_user_id=17,
            memory_dimensions={"principal": "person-1"},
        ),
    )
    ambient = ExecutionScope(memory_dimensions={"principal": "attacker"})
    with UserContext(999), ExecutionScopeContext(ambient):
        assert (
            await enrich_context_with_memory(
                context=SimpleNamespace(metadata={}, execution_id="failing"),
                query="anything",
                category="general",
                memory_store=store,
            )
            == []
        )
        assert current_user_id.get() == 999
        assert get_execution_scope() is ambient

    assert base.seen_filters is not None
    assert base.seen_filters["metadata"] == {
        "user_id": 17,
        "execution_scope_principal": "person-1",
    }


def test_agent_config_cannot_select_a_memory_access_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = UserIsolatedMemoryStore(InMemoryMemoryStore())
    monkeypatch.setattr(
        agent_runtime_service, "get_memory_store", Mock(return_value=store)
    )

    policy = agent_runtime_service.resolve_agent_service_memory_policy(
        task=_task(),
        agent_config={
            "memory_access_scope": {
                "owner_user_id": 17,
                "memory_dimensions": {"principal": "forged"},
            }
        },
    )
    with UserContext(23):
        note = MemoryNote(content="legacy")
        assert policy.memory.add(note).success

    assert note.metadata == {"user_id": 23}


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
            memory_access_scope=object(),  # type: ignore[arg-type]
        ),
        MemoryPolicyDecision(
            enabled=False,
            available=True,
            memory_access_scope=MemoryAccessScope(
                owner_user_id=17,
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
        "invalid-access-scope",
        "disabled-with-access-scope",
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


@pytest.mark.asyncio
async def test_cached_service_revalidates_and_replaces_memory_every_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = UserIsolatedMemoryStore(InMemoryMemoryStore())
    second = UserIsolatedMemoryStore(InMemoryMemoryStore())
    inert = InMemoryMemoryStore()
    service = AgentService(
        name="cached",
        id="cached",
        memory=first,
        tools=[],
        enable_workspace=False,
    )
    service._execution_adapter = service._build_execution_adapter()
    manager = agent_runtime_service.AgentServiceManager()
    manager._agents[17] = service
    policies = iter(
        [
            agent_runtime_service.AgentServiceMemoryPolicy(
                memory=second,
                memory_enabled=True,
                memory_available=True,
            ),
            agent_runtime_service.AgentServiceMemoryPolicy(
                memory=inert,
                memory_enabled=False,
                memory_available=False,
                memory_availability_reason="resolver_failed",
            ),
            agent_runtime_service.AgentServiceMemoryPolicy(
                memory=first,
                memory_enabled=True,
                memory_available=True,
            ),
        ]
    )

    async def resolve(**_: Any) -> Any:
        return next(policies)

    monkeypatch.setattr(
        agent_runtime_service,
        "resolve_agent_service_memory_policy_async",
        resolve,
    )

    await manager._reconcile_cached_agent_memory_policy(
        17,
        task=_task(),
        agent_config={},
        ambient_owner_user_id=23,
        ambient_execution_scope=ExecutionScope(memory_dimensions={"tenant": "a"}),
    )
    assert service.memory is second
    assert service._execution_adapter.config.memory_store is second
    first_scope_fingerprint = service.execution_metadata[
        MEMORY_RETRIEVAL_SCOPE_FINGERPRINT_METADATA_KEY
    ]

    await manager._reconcile_cached_agent_memory_policy(
        17,
        task=_task(),
        agent_config={},
        ambient_owner_user_id=23,
        ambient_execution_scope=ExecutionScope(memory_dimensions={"tenant": "b"}),
    )
    assert service.memory is inert
    assert service._execution_adapter.config.memory_store is None
    assert service.memory_enabled is False
    denied_fingerprint = service.execution_metadata[
        MEMORY_RETRIEVAL_SCOPE_FINGERPRINT_METADATA_KEY
    ]
    assert denied_fingerprint != first_scope_fingerprint

    await manager._reconcile_cached_agent_memory_policy(
        17,
        task=_task(),
        agent_config={},
        ambient_owner_user_id=23,
        ambient_execution_scope=ExecutionScope(memory_dimensions={"tenant": "a"}),
    )
    assert service.memory is first
    assert service._execution_adapter.config.memory_store is first
    assert service.memory_enabled is True
    assert "memory_available" not in service.execution_metadata
    assert (
        service.execution_metadata[MEMORY_RETRIEVAL_SCOPE_FINGERPRINT_METADATA_KEY]
        == first_scope_fingerprint
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent_id", "expected_enabled"),
    [(41, False), (None, True)],
    ids=("agent-backed-default-disabled", "ordinary-task-reenabled"),
)
async def test_no_snapshot_cache_hit_loads_trusted_memory_eligibility(
    monkeypatch: pytest.MonkeyPatch,
    agent_id: int | None,
    expected_enabled: bool,
) -> None:
    persistent = InMemoryMemoryStore()
    service = AgentService(
        name="cached",
        id="cached",
        memory=InMemoryMemoryStore(),
        memory_enabled=False,
        tools=[],
        enable_workspace=False,
    )
    service._execution_adapter = service._build_execution_adapter()
    manager = agent_runtime_service.AgentServiceManager()
    manager._agents[17] = service
    manager._agent_owner_ids[17] = 23
    manager._agent_scope_fingerprints[17] = None
    snapshot = SimpleNamespace(
        task=_task(agent_id=agent_id),
        runtime_user=None,
        agent_config={},
        model_override=None,
    )
    load_snapshot = AsyncMock(return_value=snapshot)
    monkeypatch.setattr(
        agent_runtime_service,
        "_load_task_setup_snapshot_for_agent",
        load_snapshot,
    )
    get_memory_store = Mock(return_value=persistent)
    monkeypatch.setattr(agent_runtime_service, "get_memory_store", get_memory_store)

    returned = await manager.get_agent_for_task(
        17,
        task_owner_user_id=23,
        resolved_execution_scope=None,
    )

    assert returned is service
    load_snapshot.assert_awaited_once_with(17, 23, None)
    assert service.memory_enabled is expected_enabled
    assert (service._execution_adapter.config.memory_store is persistent) is (
        expected_enabled
    )


@pytest.mark.asyncio
async def test_no_snapshot_cache_hit_fails_memory_closed_when_task_disappears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistent = InMemoryMemoryStore()
    service = AgentService(
        name="cached",
        id="cached",
        memory=persistent,
        tools=[],
        enable_workspace=False,
    )
    service._execution_adapter = service._build_execution_adapter()
    manager = agent_runtime_service.AgentServiceManager()
    manager._agents[17] = service
    manager._agent_owner_ids[17] = 23
    manager._agent_scope_fingerprints[17] = None
    monkeypatch.setattr(
        agent_runtime_service,
        "_load_task_setup_snapshot_for_agent",
        AsyncMock(return_value=None),
    )
    get_memory_store = Mock(return_value=persistent)
    monkeypatch.setattr(agent_runtime_service, "get_memory_store", get_memory_store)

    returned = await manager.get_agent_for_task(
        17,
        task_owner_user_id=23,
        resolved_execution_scope=None,
    )

    assert returned is service
    assert service.memory is not persistent
    assert isinstance(service.memory, InMemoryMemoryStore)
    assert service.memory_enabled is False
    assert service.memory_available is False
    assert (
        service.memory_availability_reason
        == agent_runtime_service.GENERIC_MEMORY_AVAILABILITY_REASON
    )
    get_memory_store.assert_not_called()


@pytest.mark.asyncio
async def test_no_snapshot_cache_hit_does_not_reuse_memory_after_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistent = InMemoryMemoryStore()
    service = AgentService(
        name="cached",
        id="cached",
        memory=persistent,
        tools=[],
        enable_workspace=False,
    )
    manager = agent_runtime_service.AgentServiceManager()
    manager._agents[17] = service
    manager._agent_owner_ids[17] = 23
    manager._agent_scope_fingerprints[17] = None
    monkeypatch.setattr(
        agent_runtime_service,
        "_load_task_setup_snapshot_for_agent",
        AsyncMock(side_effect=RuntimeError("snapshot unavailable")),
    )

    with pytest.raises(RuntimeError, match="snapshot unavailable"):
        await manager.get_agent_for_task(
            17,
            task_owner_user_id=23,
            resolved_execution_scope=None,
        )

    assert service.memory is persistent
    assert service.memory_enabled is True


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
        MEMORY_RETRIEVAL_SCOPE_FINGERPRINT_METADATA_KEY: (
            memory_retrieval_scope_fingerprint(enabled=False, access_scope=None)
        ),
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


def test_available_memory_records_only_retrieval_scope_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ordinary task records no availability diagnostic."""
    monkeypatch.setattr(
        agent_runtime_service, "get_memory_store", Mock(return_value=Mock())
    )

    policy = agent_runtime_service.resolve_agent_service_memory_policy(task=_task())

    assert policy.memory_available is True
    assert policy.public_availability_reason is None
    assert policy.execution_metadata() == {
        MEMORY_RETRIEVAL_SCOPE_FINGERPRINT_METADATA_KEY: (
            memory_retrieval_scope_fingerprint(enabled=True, access_scope=None)
        )
    }


def test_memory_retrieval_scope_fingerprint_is_stable_and_opaque() -> None:
    first = MemoryAccessScope(
        owner_user_id=23,
        memory_dimensions={"tenant": "team-a", "channel": "private"},
    )
    reordered = MemoryAccessScope(
        owner_user_id=23,
        memory_dimensions={"channel": "private", "tenant": "team-a"},
    )
    different = MemoryAccessScope(
        owner_user_id=23,
        memory_dimensions={"tenant": "team-b", "channel": "private"},
    )

    first_fingerprint = memory_retrieval_scope_fingerprint(
        enabled=True, access_scope=first
    )
    assert first_fingerprint == memory_retrieval_scope_fingerprint(
        enabled=True, access_scope=reordered
    )
    assert first_fingerprint != memory_retrieval_scope_fingerprint(
        enabled=True, access_scope=different
    )
    assert first_fingerprint.startswith("sha256:")
    assert "team-a" not in first_fingerprint
    ambient = memory_retrieval_scope_fingerprint(
        enabled=True,
        access_scope=None,
        ambient_owner_user_id=23,
        ambient_execution_scope=ExecutionScope(
            memory_dimensions={"tenant": "team-a", "channel": "private"}
        ),
    )
    reordered_ambient = memory_retrieval_scope_fingerprint(
        enabled=True,
        access_scope=None,
        ambient_owner_user_id=23,
        ambient_execution_scope=ExecutionScope(
            memory_dimensions={"channel": "private", "tenant": "team-a"}
        ),
    )
    different_ambient_owner = memory_retrieval_scope_fingerprint(
        enabled=True,
        access_scope=None,
        ambient_owner_user_id=24,
        ambient_execution_scope=ExecutionScope(
            memory_dimensions={"tenant": "team-a", "channel": "private"}
        ),
    )
    different_ambient_dimensions = memory_retrieval_scope_fingerprint(
        enabled=True,
        access_scope=None,
        ambient_owner_user_id=23,
        ambient_execution_scope=ExecutionScope(
            memory_dimensions={"tenant": "team-b", "channel": "private"}
        ),
    )
    strict_ambient = memory_retrieval_scope_fingerprint(
        enabled=True,
        access_scope=None,
        ambient_owner_user_id=23,
        ambient_execution_scope=ExecutionScope(
            memory_dimensions={"tenant": "team-a", "channel": "private"},
            strict_memory_isolation=True,
        ),
    )
    denied = memory_retrieval_scope_fingerprint(enabled=False, access_scope=None)
    assert ambient == reordered_ambient
    assert ambient != different_ambient_owner
    assert ambient != different_ambient_dimensions
    assert ambient != strict_ambient
    assert first_fingerprint != ambient
    assert first_fingerprint != denied
    assert ambient != denied
    with pytest.raises(ValueError, match="not a boolean"):
        memory_retrieval_scope_fingerprint(
            enabled=True,
            access_scope=None,
            ambient_owner_user_id=True,
        )


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
