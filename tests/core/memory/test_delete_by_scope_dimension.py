"""Tests for MemoryStore.delete_by_scope_dimension.

Bulk-reaping every note stamped with one execution-scope dimension
(``key=value``) — the primitive a control plane needs to clean up memories
whose dimension no longer maps to a live principal (e.g. a revoked client
application, xagent-saas#91). Covers the LanceDB predicate-pushdown override,
the base class's generic list-and-delete fallback (via InMemoryMemoryStore),
and the UserIsolatedMemoryStore delegation.
"""

from __future__ import annotations

import shutil
import tempfile

import pytest

from xagent.core.execution_scope import MEMORY_DIMENSION_METADATA_PREFIX
from xagent.core.memory.core import MemoryNote, MemoryResponse
from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.core.memory.lancedb import LanceDBMemoryStore
from xagent.web.user_isolated_memory import UserContext, UserIsolatedMemoryStore

from .conftest import ConstantEmbedding

P = MEMORY_DIMENSION_METADATA_PREFIX


@pytest.fixture
def temp_db_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def lancedb_store(temp_db_dir):
    return LanceDBMemoryStore(
        db_dir=temp_db_dir,
        collection_name="test_memories",
        embedding_model=ConstantEmbedding(64),
    )


def _note(content, **dims):
    return MemoryNote(
        content=content,
        metadata={f"{P}{key}": str(value) for key, value in dims.items()},
    )


def _seed(store):
    """Notes for CA 42 (two end users), CA 4 (the prefix of 42), and a
    dimension-less creator-direct note. Returns their ids."""
    ids = {}
    ids["ca42_a"] = store.add(
        _note("ca42 user a", client_application_id=42, end_user_id=7)
    ).memory_id
    ids["ca42_b"] = store.add(
        _note("ca42 user b", client_application_id=42, end_user_id=8)
    ).memory_id
    ids["ca4"] = store.add(
        _note("ca4 user", client_application_id=4, end_user_id=9)
    ).memory_id
    ids["plain"] = store.add(MemoryNote(content="creator-direct note")).memory_id
    return ids


def _assert_reap_semantics(store):
    """The contract shared by every implementation: exact-match bulk delete,
    no prefix collisions, dimension-less notes untouched, idempotent."""
    ids = _seed(store)

    # int value must match the string-stamped dimension (control planes pass
    # integer surrogate keys; stamps are strings).
    response = store.delete_by_scope_dimension("client_application_id", 42)
    assert response.success
    assert response.metadata["deleted_count"] == 2

    assert not store.get(ids["ca42_a"]).success
    assert not store.get(ids["ca42_b"]).success
    # "client_application_id=4" is a prefix of "...=42" — must survive.
    assert store.get(ids["ca4"]).success
    assert store.get(ids["plain"]).success

    # Idempotent: nothing left to reap.
    again = store.delete_by_scope_dimension("client_application_id", 42)
    assert again.success
    assert again.metadata["deleted_count"] == 0


def test_lancedb_pushdown_reap(lancedb_store):
    _assert_reap_semantics(lancedb_store)


def test_lancedb_reap_before_any_write(lancedb_store):
    """Empty store (no note ever written) — reap is a successful no-op."""
    response = lancedb_store.delete_by_scope_dimension("client_application_id", 42)
    assert response.success
    assert response.metadata["deleted_count"] == 0


def test_base_default_reap_via_in_memory():
    _assert_reap_semantics(InMemoryMemoryStore())


def _assert_enumeration_semantics(store):
    """list_scope_dimension_values: distinct stamped values for one dimension,
    other dimensions and dimension-less notes invisible."""
    _seed(store)
    assert store.list_scope_dimension_values("client_application_id") == {"42", "4"}
    assert store.list_scope_dimension_values("end_user_id") == {"7", "8", "9"}
    assert store.list_scope_dimension_values("no_such_dimension") == set()


def test_lancedb_list_scope_dimension_values(lancedb_store):
    _assert_enumeration_semantics(lancedb_store)


def test_lancedb_list_values_before_any_write(lancedb_store):
    assert lancedb_store.list_scope_dimension_values("client_application_id") == set()


def test_base_default_list_values_via_in_memory():
    _assert_enumeration_semantics(InMemoryMemoryStore())


def _assert_special_character_values(store):
    """Values containing ``=`` and a single quote survive both the SQL-literal
    escaping on the delete pushdown and the fixed-length prefix strip on
    enumeration."""
    kept = store.add(_note("equals value", session_ref="a=b")).memory_id
    quoted = store.add(_note("quoted value", session_ref="o'brien")).memory_id

    assert store.list_scope_dimension_values("session_ref") == {"a=b", "o'brien"}

    response = store.delete_by_scope_dimension("session_ref", "o'brien")
    assert response.success
    assert response.metadata["deleted_count"] == 1
    assert not store.get(quoted).success
    assert store.get(kept).success

    response = store.delete_by_scope_dimension("session_ref", "a=b")
    assert response.success
    assert response.metadata["deleted_count"] == 1
    assert store.list_scope_dimension_values("session_ref") == set()


def test_lancedb_special_character_values(lancedb_store):
    _assert_special_character_values(lancedb_store)


def test_base_default_special_character_values():
    _assert_special_character_values(InMemoryMemoryStore())


def test_base_default_reap_reports_partial_failure():
    """Base fallback: one per-note delete failing yields success=False while
    deleted_count still reports the notes that did go (best-effort)."""
    store = InMemoryMemoryStore()
    ids = [
        store.add(_note(f"ca42 note {i}", client_application_id=42)).memory_id
        for i in range(3)
    ]
    stuck_id = ids[1]
    real_delete = store.delete

    def flaky_delete(note_id):
        if note_id == stuck_id:
            return MemoryResponse(success=False, error="simulated backend failure")
        return real_delete(note_id)

    store.delete = flaky_delete

    response = store.delete_by_scope_dimension("client_application_id", 42)
    assert not response.success
    assert response.metadata["deleted_count"] == 2
    assert "1" in response.error


def test_lancedb_reap_failure_on_backend_error(lancedb_store, monkeypatch):
    """LanceDB override: a backend error during the bulk delete surfaces as
    success=False with deleted_count=0 (all-or-nothing), not an exception."""
    _seed(lancedb_store)

    class ExplodingTable:
        def count_rows(self, *args, **kwargs):
            return 2

        def delete(self, *args, **kwargs):
            raise RuntimeError("simulated backend failure")

    class StubConnection:
        def open_table(self, name):
            return ExplodingTable()

    monkeypatch.setattr(
        lancedb_store._vector_store, "get_raw_connection", lambda: StubConnection()
    )

    response = lancedb_store.delete_by_scope_dimension("client_application_id", 42)
    assert not response.success
    assert response.metadata["deleted_count"] == 0
    assert "simulated backend failure" in response.error


def test_user_isolated_store_delegates_ungated():
    """The wrapper reaps regardless of the current user context: the dimension
    predicate itself bounds the deletion, and a maintenance job runs with no
    (or an unrelated) user bound."""
    base = InMemoryMemoryStore()
    store = UserIsolatedMemoryStore(base)

    note = _note("ca42 note", client_application_id=42, end_user_id=7)
    note.metadata["user_id"] = 1
    note_id = base.add(note).memory_id

    with UserContext(99):  # some other principal entirely
        response = store.delete_by_scope_dimension("client_application_id", 42)

    assert response.success
    assert response.metadata["deleted_count"] == 1
    assert not base.get(note_id).success
