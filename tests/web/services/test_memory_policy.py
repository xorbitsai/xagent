from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.web.api import chat as chat_api
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
    monkeypatch.setattr(chat_api, "get_memory_store", get_memory_store)

    policy = chat_api.resolve_agent_service_memory_policy(
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
    monkeypatch.setattr(chat_api, "get_memory_store", Mock(return_value=dynamic_store))
    resolver = Mock(
        return_value=MemoryPolicyDecision(
            enabled=True,
            available=True,
            reason=None,
        )
    )
    set_trusted_memory_policy_resolver(resolver)

    policy = chat_api.resolve_agent_service_memory_policy(
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
    monkeypatch.setattr(chat_api, "get_memory_store", Mock(return_value=dynamic_store))
    set_trusted_memory_policy_resolver(
        lambda _request: MemoryPolicyDecision(
            enabled=False,
            available=True,
            reason="disabled_by_host_policy",
        )
    )

    policy = chat_api.resolve_agent_service_memory_policy(task=_task())

    assert policy.memory is dynamic_store
    assert policy.memory_enabled is False
    assert policy.memory_available is True
    assert policy.memory_availability_reason == "disabled_by_host_policy"


def test_trusted_resolver_can_report_unavailable_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_memory_store = Mock(name="get-memory-store")
    monkeypatch.setattr(chat_api, "get_memory_store", get_memory_store)
    set_trusted_memory_policy_resolver(
        lambda _request: MemoryPolicyDecision(
            enabled=False,
            available=False,
            reason="memory_service_unavailable",
        )
    )

    policy = chat_api.resolve_agent_service_memory_policy(task=_task())

    assert isinstance(policy.memory, InMemoryMemoryStore)
    assert policy.memory_enabled is False
    assert policy.memory_available is False
    assert policy.memory_availability_reason == "memory_service_unavailable"
    get_memory_store.assert_not_called()


def test_resolver_exception_fails_closed_for_preview() -> None:
    def fail(_request: MemoryPolicyRequest) -> MemoryPolicyDecision:
        raise RuntimeError("resolver is unavailable")

    set_trusted_memory_policy_resolver(fail)

    policy = chat_api.resolve_agent_service_memory_policy(
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
    monkeypatch.setattr(chat_api, "get_memory_store", get_memory_store)
    set_trusted_memory_policy_resolver(lambda _request: decision)  # type: ignore[arg-type,return-value]

    policy = chat_api.resolve_agent_service_memory_policy(task=_task())

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

    chat_api.resolve_agent_service_memory_policy(task=None, agent_config={})

    resolver.assert_called_once_with(
        MemoryPolicyRequest(
            task_id=None,
            user_id=None,
            agent_id=None,
            source=None,
            is_preview=False,
        )
    )
