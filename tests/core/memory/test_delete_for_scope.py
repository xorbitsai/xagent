"""Tests for scoped bulk erasure: ``delete_for_scope``, ``delete_current_scope``
and ``LanceDBMemoryStore.compact``.

``delete_for_scope`` is the control-plane primitive behind "forget everything
in this workspace" (owner + client application) and per-member deletes (owner
+ end user + ...); ``delete_current_scope`` is its agent-facing form, bounded
to exactly what the by-id ``delete`` could reach from the active scope.
``compact`` makes those deletes physical.
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from xagent.core.execution_scope import (
    MEMORY_DIMENSION_METADATA_PREFIX,
    ExecutionScope,
    ExecutionScopeContext,
    InvalidScopeComponentError,
)
from xagent.core.memory.access_scope import MemoryAccessScope
from xagent.core.memory.base import MemoryScopeTooBroadError, MemoryStore
from xagent.core.memory.core import MemoryNote
from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.core.memory.lancedb import LanceDBMemoryStore
from xagent.web.user_isolated_memory import UserContext, UserIsolatedMemoryStore

from .conftest import ConstantEmbedding

P = MEMORY_DIMENSION_METADATA_PREFIX
# One Slack workspace (client application 10) under machine owner 1; member 7
# and member 8 talk to agent 3; member 7 also talks to agent 4.
MEMBER_7 = {"client_application_id": "10", "end_user_id": "7", "agent_id": "3"}


def _note(content: str, owner: int, **dims: str) -> MemoryNote:
    return MemoryNote(
        content=content,
        metadata={
            "user_id": owner,
            **{f"{P}{key}": value for key, value in dims.items()},
        },
    )


def _seed(store: MemoryStore) -> dict[str, str]:
    notes = {
        "member7": _note("member 7 likes tea", 1, **MEMBER_7),
        "member7_agent4": _note(
            "member 7 via agent 4",
            1,
            client_application_id="10",
            end_user_id="7",
            agent_id="4",
        ),
        "member8": _note(
            "member 8 likes coffee",
            1,
            client_application_id="10",
            end_user_id="8",
            agent_id="3",
        ),
        "other_app": _note(
            "same member id, other app",
            1,
            client_application_id="11",
            end_user_id="7",
            agent_id="3",
        ),
        "app_prefix": _note(
            "app id prefix", 1, client_application_id="1", end_user_id="7", agent_id="3"
        ),
        "other_owner": _note("same dims, other owner", 2, **MEMBER_7),
        "owner_plain": _note("creator-direct note", 1),
    }
    ids = {}
    for name, note in notes.items():
        response = store.add(note)
        assert response.success
        ids[name] = response.memory_id
    return ids


def _remaining(store: MemoryStore, ids: dict[str, str]) -> set[str]:
    return {name for name, note_id in ids.items() if store.get(note_id).success}


def _lancedb_store(root: Path) -> LanceDBMemoryStore:
    return LanceDBMemoryStore(
        db_dir=str(root),
        collection_name="memories",
        embedding_model=ConstantEmbedding(8),
    )


@pytest.fixture(params=["lancedb", "in_memory"])
def store(request: Any, tmp_path: Path) -> MemoryStore:
    if request.param == "in_memory":
        return InMemoryMemoryStore()
    return _lancedb_store(tmp_path)


def test_member_delete_touches_only_that_member(store: MemoryStore) -> None:
    ids = _seed(store)

    assert store.delete_for_scope(1, MEMBER_7) == 1

    assert _remaining(store, ids) == set(ids) - {"member7"}
    # Idempotent.
    assert store.delete_for_scope(1, MEMBER_7) == 0


def test_workspace_delete_spares_other_owners_apps_and_plain_notes(
    store: MemoryStore,
) -> None:
    ids = _seed(store)

    assert store.delete_for_scope(1, {"client_application_id": "10"}) == 3

    assert _remaining(store, ids) == {
        "other_app",
        "app_prefix",
        "other_owner",
        "owner_plain",
    }


@pytest.mark.parametrize("dims", [{}, None])
def test_empty_dims_are_rejected(store: MemoryStore, dims: Any) -> None:
    ids = _seed(store)

    with pytest.raises(MemoryScopeTooBroadError):
        store.delete_for_scope(1, dims)

    assert _remaining(store, ids) == set(ids)


@pytest.mark.parametrize("owner", [0, -1, True, "1", 1.0, 2**63])
def test_invalid_owner_is_rejected(store: MemoryStore, owner: Any) -> None:
    with pytest.raises(ValueError):
        store.delete_for_scope(owner, MEMBER_7)


@pytest.mark.parametrize(
    "dims",
    [{"client_application_id": 10}, {"client_application_id": ""}, {"a b": "1"}],
)
def test_invalid_dimension_is_rejected(store: MemoryStore, dims: Any) -> None:
    with pytest.raises(InvalidScopeComponentError):
        store.delete_for_scope(1, dims)


def test_raw_store_refuses_current_scope_delete(store: MemoryStore) -> None:
    with pytest.raises(MemoryScopeTooBroadError):
        store.delete_current_scope()


def test_lancedb_backend_failure_raises(tmp_path: Path, monkeypatch: Any) -> None:
    """An erasure request must never be reported as done when it was not."""
    store = _lancedb_store(tmp_path)

    class ExplodingTable:
        def count_rows(self, *_args: Any) -> int:
            return 1

        def delete(self, *_args: Any) -> None:
            raise RuntimeError("simulated backend failure")

    class StubConnection:
        def open_table(self, _name: str) -> ExplodingTable:
            return ExplodingTable()

    monkeypatch.setattr(
        store._vector_store, "get_raw_connection", lambda: StubConnection()
    )
    with pytest.raises(RuntimeError, match="simulated backend failure"):
        store.delete_for_scope(1, MEMBER_7)


# ---------------------------------------------------------------------------
# UserIsolatedMemoryStore.delete_current_scope
# ---------------------------------------------------------------------------


def test_current_scope_delete_matches_the_active_scope_exactly(
    store: MemoryStore,
) -> None:
    ids = _seed(store)
    isolated = UserIsolatedMemoryStore(store)

    with (
        UserContext(1),
        ExecutionScopeContext(ExecutionScope(memory_dimensions=MEMBER_7)),
    ):
        assert isolated.delete_current_scope() == 1

    assert _remaining(store, ids) == set(ids) - {"member7"}


def test_current_scope_delete_with_fixed_access_scope(store: MemoryStore) -> None:
    ids = _seed(store)
    bound = UserIsolatedMemoryStore(store).with_memory_access_scope(
        MemoryAccessScope(owner_user_id=1, memory_dimensions=MEMBER_7)
    )

    # Ambient identity cannot widen or redirect a fixed binding.
    with (
        UserContext(2),
        ExecutionScopeContext(
            ExecutionScope(memory_dimensions={"client_application_id": "10"})
        ),
    ):
        assert bound.delete_current_scope() == 1

    assert _remaining(store, ids) == set(ids) - {"member7"}


@pytest.mark.parametrize(
    ("user_id", "scope"),
    [
        # Creator-direct website task: strict, dimension-less.
        (1, ExecutionScope(strict_memory_isolation=True)),
        # Unscoped execution: by-id access spans the whole owner.
        (1, None),
        # Scoped but ownerless: by-id access is not owner-gated.
        (None, ExecutionScope(memory_dimensions=MEMBER_7)),
    ],
)
def test_current_scope_delete_refuses_unbounded_scopes(
    store: MemoryStore, user_id: int | None, scope: ExecutionScope | None
) -> None:
    ids = _seed(store)
    isolated = UserIsolatedMemoryStore(store)

    with UserContext(user_id), ExecutionScopeContext(scope):
        with pytest.raises(MemoryScopeTooBroadError):
            isolated.delete_current_scope()

    assert _remaining(store, ids) == set(ids)


def test_strict_owner_only_access_scope_refuses(store: MemoryStore) -> None:
    ids = _seed(store)
    bound = UserIsolatedMemoryStore(store).with_memory_access_scope(
        MemoryAccessScope(owner_user_id=1, memory_dimensions={}, strict_owner_only=True)
    )

    with pytest.raises(MemoryScopeTooBroadError):
        bound.delete_current_scope()

    assert _remaining(store, ids) == set(ids)


def test_isolated_delete_for_scope_is_not_narrowed_by_ambient_scope(
    store: MemoryStore,
) -> None:
    """The control-plane form is explicit: ambient identity neither widens nor
    narrows it."""
    ids = _seed(store)
    isolated = UserIsolatedMemoryStore(store)

    with (
        UserContext(2),
        ExecutionScopeContext(ExecutionScope(memory_dimensions=MEMBER_7)),
    ):
        assert isolated.delete_for_scope(1, {"client_application_id": "10"}) == 3

    assert "other_owner" in _remaining(store, ids)


# ---------------------------------------------------------------------------
# LanceDBMemoryStore.compact
# ---------------------------------------------------------------------------


def _files_containing(root: Path, needle: bytes) -> list[str]:
    hits = []
    for directory, _subdirs, files in os.walk(root):
        for name in files:
            path = os.path.join(directory, name)
            with open(path, "rb") as handle:
                if needle in handle.read():
                    hits.append(path)
    return hits


def test_compact_physically_removes_deleted_rows(tmp_path: Path) -> None:
    store = _lancedb_store(tmp_path)
    secret = "member-seven-secret-preference"
    for index in range(40):
        assert store.add(
            _note(f"member 8 note {index}", 1, **{**MEMBER_7, "end_user_id": "8"})
        ).success
    assert store.add(_note(secret, 1, **MEMBER_7)).success
    # Merge everything into one large fragment first: that is the layout in
    # which a plain optimize() never materializes a 1-in-41 delete.
    store.compact(cleanup_older_than=timedelta(0))
    assert _files_containing(tmp_path, secret.encode())

    assert store.delete_for_scope(1, MEMBER_7) == 1
    # Soft-deleted: the text is still on disk until compaction.
    assert _files_containing(tmp_path, secret.encode())

    store.compact(cleanup_older_than=timedelta(0))

    assert _files_containing(tmp_path, secret.encode()) == []
    table = store._vector_store.get_raw_connection().open_table("memories")
    assert len(table.list_versions()) == 1
    # Survivors keep their rows, vectors and scope columns.
    isolated = UserIsolatedMemoryStore(store)
    member_8 = ExecutionScope(memory_dimensions={**MEMBER_7, "end_user_id": "8"})
    with UserContext(1), ExecutionScopeContext(member_8):
        assert len(isolated.search("member 8 note", k=100)) == 40


def test_compact_with_retention_keeps_recent_versions(tmp_path: Path) -> None:
    store = _lancedb_store(tmp_path)
    assert store.add(_note("kept", 1, **MEMBER_7)).success
    table = store._vector_store.get_raw_connection().open_table("memories")
    versions_before = len(table.list_versions())

    store.compact()

    # The default retention prunes nothing this young, so concurrent writers
    # never lose the version they started from.
    assert len(table.list_versions()) > versions_before
