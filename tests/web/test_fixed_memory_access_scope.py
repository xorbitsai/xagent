"""Fixed memory namespaces that do not change execution identity."""

from __future__ import annotations

import asyncio

import pytest

from xagent.core.agent.context.memory_tool import DeleteMemoryTool
from xagent.core.execution_scope import (
    ExecutionScope,
    ExecutionScopeContext,
    get_execution_scope,
)
from xagent.core.memory import MemoryAccessScope, bind_memory_access_scope
from xagent.core.memory.core import MemoryNote, MemoryResponse
from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.web.memory_lifecycle import (
    MemoryLifecycleState,
    MemoryLifecycleStatus,
    MemoryUnavailableError,
)
from xagent.web.revocable_memory_store import RevocableMemoryStore
from xagent.web.user_isolated_memory import (
    UserContext,
    UserIsolatedMemoryStore,
    current_user_id,
)

SCOPE_A = MemoryAccessScope(
    owner_user_id=700,
    memory_dimensions={"principal": "person-1", "tenant": "acme"},
)
SCOPE_B = MemoryAccessScope(
    owner_user_id=700,
    memory_dimensions={"principal": "person-2", "tenant": "acme"},
)
OTHER_OWNER = MemoryAccessScope(
    owner_user_id=701,
    memory_dimensions={"principal": "person-1", "tenant": "acme"},
)


class _PublicationSource:
    def __init__(self) -> None:
        self.generation = 1
        self.status = MemoryLifecycleStatus(MemoryLifecycleState.READY)

    def publication_generation(self) -> int:
        return self.generation

    def published_status(self) -> MemoryLifecycleStatus:
        return self.status


class _UnconfirmedDeleteStore(InMemoryMemoryStore):
    def delete(self, note_id: str) -> MemoryResponse:
        return MemoryResponse(success=True, memory_id=note_id)


@pytest.mark.parametrize(
    "owner",
    [True, False, 0, -1, 2**63, "700", 700.0],
)
def test_memory_access_scope_rejects_non_platform_owner(owner: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        MemoryAccessScope(  # type: ignore[arg-type]
            owner_user_id=owner,
            memory_dimensions={"principal": "person-1"},
        )


@pytest.mark.parametrize(
    "dimensions",
    [
        [],
        [("principal", "person-1")],
        {},
        {"": "person-1"},
        {"principal": ""},
        {"principal/name": "person-1"},
        {"principal": 1},
    ],
)
def test_memory_access_scope_requires_valid_nonempty_dimensions(
    dimensions: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        MemoryAccessScope(  # type: ignore[arg-type]
            owner_user_id=700,
            memory_dimensions=dimensions,
        )


def test_memory_access_scope_copies_and_freezes_dimensions() -> None:
    source = {"principal": "person-1"}
    scope = MemoryAccessScope(owner_user_id=700, memory_dimensions=source)

    source["principal"] = "person-2"

    assert dict(scope.memory_dimensions) == {"principal": "person-1"}
    with pytest.raises(TypeError):
        scope.memory_dimensions["principal"] = "person-3"  # type: ignore[index]


@pytest.mark.asyncio
async def test_fixed_scope_covers_crud_filters_clear_stats_and_thread_hops() -> None:
    base = InMemoryMemoryStore()
    first = UserIsolatedMemoryStore(base).with_memory_access_scope(SCOPE_A)
    same = UserIsolatedMemoryStore(base).with_memory_access_scope(SCOPE_A)
    other_principal = UserIsolatedMemoryStore(base).with_memory_access_scope(SCOPE_B)
    other_owner = UserIsolatedMemoryStore(base).with_memory_access_scope(OTHER_OWNER)

    caller_filters = {
        "metadata": {
            "user_id": 999,
            "execution_scope_principal": "forged",
            "execution_scope_extra": "forged",
            "kind": "preference",
        }
    }
    with (
        UserContext(1),
        ExecutionScopeContext(
            ExecutionScope(memory_dimensions={"principal": "ambient"})
        ),
    ):
        note = MemoryNote(
            content="shared memory",
            metadata={
                "user_id": 999,
                "execution_scope_principal": "forged",
                "execution_scope_tenant": "forged",
                "execution_scope_extra": "forged",
                "kind": "preference",
            },
        )
        assert first.add(note).success
        assert first.search("shared", filters=caller_filters) == [note]

    assert caller_filters == {
        "metadata": {
            "user_id": 999,
            "execution_scope_principal": "forged",
            "execution_scope_extra": "forged",
            "kind": "preference",
        }
    }
    assert note.metadata == {
        "user_id": 700,
        "execution_scope_principal": "person-1",
        "execution_scope_tenant": "acme",
        "kind": "preference",
    }

    with (
        UserContext(2),
        ExecutionScopeContext(
            ExecutionScope(memory_dimensions={"principal": "other-ambient"})
        ),
    ):
        assert same.get(note.id).success
        assert same.list_all() == [note]
        assert same.get_stats()["total_count"] == 1
        updated = MemoryNote(
            id=note.id,
            content="updated memory",
            metadata={
                "user_id": 999,
                "execution_scope_principal": "forged",
                "execution_scope_tenant": "forged",
                "kind": "preference",
            },
        )
        assert same.update(updated).success
        assert current_user_id.get() == 2
        assert get_execution_scope() == ExecutionScope(
            memory_dimensions={"principal": "other-ambient"}
        )

    threaded_results = await asyncio.to_thread(same.search, "updated")
    assert [result.id for result in threaded_results] == [note.id]
    assert other_principal.search("updated") == []
    assert other_owner.search("updated") == []
    assert not other_principal.get(note.id).success
    assert not other_owner.get(note.id).success
    assert not other_principal.update(
        MemoryNote(id=note.id, content="forged update")
    ).success
    assert not other_owner.delete(note.id).success

    stored = base.get(note.id).content
    assert isinstance(stored, MemoryNote)
    assert stored.metadata == {
        "user_id": 700,
        "execution_scope_principal": "person-1",
        "execution_scope_tenant": "acme",
        "kind": "preference",
    }

    retained = MemoryNote(content="other principal")
    assert other_principal.add(retained).success
    same.clear()
    assert same.list_all() == []
    assert other_principal.get(retained.id).success


@pytest.mark.asyncio
async def test_delete_tool_verifies_with_the_same_fixed_scope() -> None:
    base = InMemoryMemoryStore()
    bound = UserIsolatedMemoryStore(base).with_memory_access_scope(SCOPE_A)
    note = MemoryNote(content="forget me")
    assert bound.add(note).success

    with (
        UserContext(999),
        ExecutionScopeContext(
            ExecutionScope(memory_dimensions={"principal": "ambient"})
        ),
    ):
        result = await DeleteMemoryTool(memory_store=bound).execute(note.id)

    assert result == {
        "success": True,
        "deleted": True,
        "verified": True,
        "memory_id": note.id,
    }
    assert not base.get(note.id).success


@pytest.mark.asyncio
async def test_delete_tool_does_not_claim_unconfirmed_fixed_scope_deletion() -> None:
    base = _UnconfirmedDeleteStore()
    bound = UserIsolatedMemoryStore(base).with_memory_access_scope(SCOPE_A)
    note = MemoryNote(content="retain me")
    assert bound.add(note).success

    result = await DeleteMemoryTool(memory_store=bound).execute(note.id)

    assert result == {
        "success": False,
        "deleted": False,
        "verified": False,
        "error": "Memory deletion could not be verified; the memory still exists.",
    }
    assert base.get(note.id).success


def test_public_binding_replaces_scope_without_exposing_wrapper_internals() -> None:
    base = InMemoryMemoryStore()
    ambient = UserIsolatedMemoryStore(base)
    first = bind_memory_access_scope(ambient, SCOPE_A)
    second = bind_memory_access_scope(first, SCOPE_B)

    note_a = MemoryNote(content="scope a")
    note_b = MemoryNote(content="scope b")
    assert first.add(note_a).success
    assert second.add(note_b).success

    assert [note.id for note in first.list_all()] == [note_a.id]
    assert [note.id for note in second.list_all()] == [note_b.id]


def test_revocable_binding_preserves_publication_generation() -> None:
    source = _PublicationSource()
    publication = RevocableMemoryStore(
        UserIsolatedMemoryStore(InMemoryMemoryStore()),
        source,
        generation=1,
    )
    bound = bind_memory_access_scope(publication, SCOPE_A)
    note = MemoryNote(content="published memory")

    assert bound.add(note).success
    source.status = MemoryLifecycleStatus(MemoryLifecycleState.RESTART_REQUIRED)
    source.generation = 2

    assert publication.revoked
    assert bound.revoked
    with pytest.raises(MemoryUnavailableError):
        bound.get(note.id)


def test_unsupported_binding_compositions_fail_closed() -> None:
    raw = InMemoryMemoryStore()
    with pytest.raises(TypeError, match="does not support"):
        bind_memory_access_scope(raw, SCOPE_A)

    publication = RevocableMemoryStore(raw, _PublicationSource(), generation=1)
    with pytest.raises(TypeError, match="does not support"):
        bind_memory_access_scope(publication, SCOPE_A)

    nested = UserIsolatedMemoryStore(UserIsolatedMemoryStore(raw))
    with pytest.raises(TypeError, match="nested"):
        bind_memory_access_scope(nested, SCOPE_A)

    reversed_publication = UserIsolatedMemoryStore(publication)
    with pytest.raises(TypeError, match="nested"):
        bind_memory_access_scope(reversed_publication, SCOPE_A)
