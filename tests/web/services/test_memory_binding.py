from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator

import pytest

from xagent.core.memory import MemoryAccessScope
from xagent.core.memory.core import MemoryNote
from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.web.services.memory_binding import (
    MemoryBindingResolutionError,
    MemoryManagementBindingRequest,
    MemoryStoreBinding,
    MemoryTaskBindingRequest,
    MemoryToolAssemblyRequest,
    resolve_trusted_memory_binding,
    resolve_trusted_memory_binding_async,
    set_trusted_memory_binding_resolver,
)
from xagent.web.user_isolated_memory import UserIsolatedMemoryStore


@pytest.fixture(autouse=True)
def clear_memory_binding_resolver() -> Iterator[None]:
    set_trusted_memory_binding_resolver(None)
    yield
    set_trusted_memory_binding_resolver(None)


def _task_request() -> MemoryTaskBindingRequest:
    return MemoryTaskBindingRequest(
        task_id=17,
        user_id=23,
        agent_id=None,
        source="trusted-ingress",
        is_preview=False,
    )


def _binding(
    store: InMemoryMemoryStore | UserIsolatedMemoryStore | None = None,
    *,
    identity: str = "backend-a",
    generation: str = "route-1",
) -> MemoryStoreBinding:
    return MemoryStoreBinding(
        store=store or InMemoryMemoryStore(),
        backend_identity=identity,
        route_generation=generation,
    )


@pytest.mark.parametrize(
    ("request_type", "kwargs"),
    [
        (MemoryManagementBindingRequest, {"authenticated_user_id": True}),
        (MemoryManagementBindingRequest, {"authenticated_user_id": 0}),
        (
            MemoryTaskBindingRequest,
            {
                "task_id": "17",
                "user_id": 23,
                "agent_id": None,
                "source": None,
                "is_preview": False,
            },
        ),
        (
            MemoryTaskBindingRequest,
            {
                "task_id": 17,
                "user_id": 23,
                "agent_id": None,
                "source": None,
                "is_preview": 0,
            },
        ),
    ],
)
def test_binding_requests_fail_closed_on_invalid_trusted_identity(
    request_type: type[object], kwargs: dict[str, object]
) -> None:
    with pytest.raises((TypeError, ValueError)):
        request_type(**kwargs)  # type: ignore[call-arg]


def test_unregistered_resolver_preserves_legacy_selection() -> None:
    assert resolve_trusted_memory_binding(_task_request()) is None


@pytest.mark.asyncio
async def test_async_resolution_runs_sync_callback_off_event_loop() -> None:
    event_loop_thread = threading.get_ident()
    resolver_thread: int | None = None

    def resolver(request: object) -> MemoryStoreBinding:
        nonlocal resolver_thread
        assert isinstance(request, MemoryManagementBindingRequest)
        resolver_thread = threading.get_ident()
        return _binding()

    set_trusted_memory_binding_resolver(resolver)
    result = await resolve_trusted_memory_binding_async(
        MemoryManagementBindingRequest(authenticated_user_id=23)
    )

    assert isinstance(result, MemoryStoreBinding)
    assert resolver_thread is not None
    assert resolver_thread != event_loop_thread


def test_each_request_resolves_an_immutable_binding_snapshot() -> None:
    calls = 0

    def resolver(request: object) -> MemoryStoreBinding:
        nonlocal calls
        assert isinstance(request, MemoryTaskBindingRequest)
        calls += 1
        return _binding(generation=f"route-{calls}")

    set_trusted_memory_binding_resolver(resolver)
    first = resolve_trusted_memory_binding(_task_request())
    second = resolve_trusted_memory_binding(_task_request())

    assert first is not None and second is not None
    assert first is not second
    assert first.route_generation == "route-1"
    assert second.route_generation == "route-2"
    with pytest.raises(AttributeError):
        first.route_generation = "forged"  # type: ignore[misc]


@pytest.mark.parametrize(
    "resolver",
    [
        lambda _request: None,
        lambda _request: object(),
        lambda _request: (_ for _ in ()).throw(RuntimeError("boom")),
    ],
)
def test_registered_resolver_failure_cannot_fall_back_to_global_store(
    resolver: object,
) -> None:
    set_trusted_memory_binding_resolver(resolver)  # type: ignore[arg-type]

    with pytest.raises(MemoryBindingResolutionError, match="resolution failed"):
        resolve_trusted_memory_binding(_task_request())


@pytest.mark.parametrize(
    ("identity", "generation"),
    [("", "route-1"), ("backend-a", ""), (1, "route-1")],
)
def test_binding_requires_opaque_nonempty_cache_fences(
    identity: object, generation: object
) -> None:
    with pytest.raises(TypeError):
        MemoryStoreBinding(
            store=InMemoryMemoryStore(),
            backend_identity=identity,  # type: ignore[arg-type]
            route_generation=generation,  # type: ignore[arg-type]
        )


def test_binding_copies_public_diagnostics_and_rejects_non_public_values() -> None:
    diagnostics = {"backend": "host-managed", "available": True}
    binding = MemoryStoreBinding(
        store=InMemoryMemoryStore(),
        backend_identity="backend-a",
        route_generation="route-1",
        public_diagnostics=diagnostics,
    )
    diagnostics["backend"] = "mutated"

    assert dict(binding.public_diagnostics) == {
        "backend": "host-managed",
        "available": True,
    }
    with pytest.raises(TypeError):
        binding.public_diagnostics["backend"] = "forged"  # type: ignore[index]
    with pytest.raises(TypeError, match="scalar JSON"):
        MemoryStoreBinding(
            store=InMemoryMemoryStore(),
            backend_identity="backend-a",
            route_generation="route-1",
            public_diagnostics={"nested": {"secret": "not-public-contract"}},
        )


def test_binding_applies_scope_before_snapshot_escapes() -> None:
    base = InMemoryMemoryStore()
    unbound = UserIsolatedMemoryStore(base)
    binding = MemoryStoreBinding(
        store=unbound,
        backend_identity="backend-a",
        route_generation="route-1",
        access_scope=MemoryAccessScope(
            owner_user_id=700,
            memory_dimensions={},
            strict_owner_only=True,
        ),
    )
    note = MemoryNote(content="owner-only note")

    assert binding.store is not unbound
    assert binding.store.add(note).success
    stored = base.get(note.id).content
    assert isinstance(stored, MemoryNote)
    assert stored.metadata == {"user_id": 700}


def test_binding_scope_failure_is_a_resolver_failure() -> None:
    def resolver(_request: object) -> MemoryStoreBinding:
        return MemoryStoreBinding(
            store=InMemoryMemoryStore(),
            backend_identity="backend-a",
            route_generation="route-1",
            access_scope=MemoryAccessScope(
                owner_user_id=700,
                memory_dimensions={"principal": "person-1"},
            ),
        )

    set_trusted_memory_binding_resolver(resolver)

    with pytest.raises(MemoryBindingResolutionError):
        resolve_trusted_memory_binding(_task_request())


def test_host_tool_factory_is_snapshot_local_and_opt_in() -> None:
    marker = object()

    def factory(request: MemoryToolAssemblyRequest) -> list[object]:
        assert request.task == "remember safely"
        return [marker]

    binding = MemoryStoreBinding(
        store=InMemoryMemoryStore(),
        backend_identity="backend-a",
        route_generation="route-1",
        memory_tools_factory=factory,
    )
    request = MemoryToolAssemblyRequest(
        memory_store=binding.store,
        task="remember safely",
    )

    assert binding.memory_tools_factory is not None
    assert list(binding.memory_tools_factory(request)) == [marker]
    assert _binding().memory_tools_factory is None


def test_sync_resolver_does_not_accept_awaitable_result() -> None:
    async def async_resolver(_request: object) -> MemoryStoreBinding:
        await asyncio.sleep(0)
        return _binding()

    set_trusted_memory_binding_resolver(async_resolver)  # type: ignore[arg-type]

    with pytest.raises(MemoryBindingResolutionError):
        resolve_trusted_memory_binding(_task_request())
