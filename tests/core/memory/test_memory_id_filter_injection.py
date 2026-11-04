"""Caller-supplied values cannot rewrite LanceDB filter expressions.

A memory ID, vector ID or scope-dimension value used to be interpolated raw
into a DataFusion predicate (``id = '{note_id}'``), so an ID such as
``x' OR '1'='1`` turned a by-ID lookup/delete into a whole-table one and
sidestepped the per-user ownership check. These tests drive real temporary
LanceDB tables and assert crafted values match nothing and delete nothing,
while values that legitimately contain quotes/backslashes still round-trip.
"""

from __future__ import annotations

import pytest

from xagent.core.execution_scope import MEMORY_DIMENSION_METADATA_PREFIX
from xagent.core.memory.core import MemoryNote
from xagent.core.memory.lancedb import LanceDBMemoryStore
from xagent.core.memory.scope_columns import (
    SCOPE_DIMS_COLUMN,
    USER_ID_COLUMN,
    build_scope_where,
)
from xagent.core.utils.lancedb_sql import sql_and, sql_string_in, sql_string_literal
from xagent.providers.vector_store.lancedb import LanceDBVectorStore
from xagent.web.user_isolated_memory import UserContext, UserIsolatedMemoryStore

from .conftest import ConstantEmbedding

P = MEMORY_DIMENSION_METADATA_PREFIX

MALICIOUS_IDS = [
    "x' OR '1'='1",
    "') OR (1=1",
    "' OR true OR '",
    "x' OR id IS NOT NULL OR '",
    "'",
    "''",
    "\\",
    "\\' OR 1=1 --",
    "x'; DELETE FROM test_memories; --",
    # LIKE wildcards are ordinary characters in an equality literal.
    "%",
    "_",
    "%' OR id LIKE '%",
]


# --- helper ------------------------------------------------------------------


def test_sql_string_literal_doubles_single_quotes_only():
    assert sql_string_literal("abc") == "'abc'"
    assert sql_string_literal("x' OR '1'='1") == "'x'' OR ''1''=''1'"
    # Backslash is an ordinary character in DataFusion string literals.
    assert sql_string_literal("a\\b") == "'a\\b'"


def test_sql_string_literal_rejects_non_strings():
    with pytest.raises(TypeError):
        sql_string_literal(1)  # type: ignore[arg-type]


def test_sql_string_in_quotes_every_value_and_matches_nothing_when_empty():
    assert sql_string_in("id", ["a", "b'c"]) == "id IN ('a', 'b''c')"
    assert sql_string_in("id", []) == "false"


def test_sql_and_parenthesises_every_term():
    assert sql_and([]) is None
    assert sql_and([None, ""]) is None
    assert sql_and(["a = 1"]) == "a = 1"
    assert sql_and(["a = 1", "b = 2 OR c = 3"]) == "(a = 1) AND (b = 2 OR c = 3)"


def test_build_scope_where_quotes_dimension_values_and_groups_owner_term():
    where_sql, _ = build_scope_where(
        {"metadata": {"user_id": 1, f"{P}tenant": "x') OR (1=1"}}
    )
    assert where_sql == (
        f"({USER_ID_COLUMN} = 1) AND "
        f"(array_contains({SCOPE_DIMS_COLUMN}, 'tenant=x'') OR (1=1'))"
    )


# --- memory store --------------------------------------------------------------


@pytest.fixture
def base_store(tmp_path):
    return LanceDBMemoryStore(
        db_dir=str(tmp_path / "memory"),
        collection_name="test_memories",
        embedding_model=ConstantEmbedding(16),
    )


@pytest.fixture
def two_users(base_store):
    """User 1 and user 2 each own one note. User 1's note is inserted first so
    an unescaped ``... OR '1'='1`` lookup would return it and pass user 1's
    ownership check before the delete fans out to the whole table."""
    isolated = UserIsolatedMemoryStore(base_store)
    with UserContext(1):
        own_id = isolated.add(MemoryNote(content="user one note")).memory_id
    with UserContext(2):
        victim_id = isolated.add(MemoryNote(content="user two note")).memory_id
    assert own_id and victim_id
    return isolated, own_id, victim_id


def _all_ids(store: LanceDBMemoryStore) -> set[str]:
    return {note.id for note in store.list_all()}


@pytest.mark.parametrize("crafted_id", MALICIOUS_IDS)
def test_crafted_id_does_not_match_on_get(base_store, two_users, crafted_id):
    response = base_store.get(crafted_id)
    assert not response.success
    assert response.content is None


@pytest.mark.parametrize("crafted_id", MALICIOUS_IDS)
def test_crafted_id_deletes_nothing_in_base_store(base_store, two_users, crafted_id):
    _, own_id, victim_id = two_users
    base_store.delete(crafted_id)
    assert _all_ids(base_store) == {own_id, victim_id}


@pytest.mark.parametrize("crafted_id", MALICIOUS_IDS)
def test_crafted_id_cannot_reach_another_users_note(base_store, two_users, crafted_id):
    isolated, own_id, victim_id = two_users
    with UserContext(1):
        assert not isolated.delete(crafted_id).success
        assert not isolated.update(
            MemoryNote(id=crafted_id, content="overwritten")
        ).success
    assert _all_ids(base_store) == {own_id, victim_id}
    victim = base_store.get(victim_id)
    assert victim.success and victim.content.content == "user two note"


@pytest.mark.parametrize(
    "odd_id",
    ["it's", "a\\b", "'; --", "q''q", "50%_off", "id with spaces", "a/b", "%", "_"],
)
def test_ids_with_quotes_and_backslashes_round_trip_exactly(base_store, odd_id):
    other = base_store.add(MemoryNote(content="bystander")).memory_id
    assert base_store.add(MemoryNote(id=odd_id, content="odd")).memory_id == odd_id

    fetched = base_store.get(odd_id)
    assert fetched.success and fetched.content.id == odd_id

    assert base_store.delete(odd_id).success
    assert not base_store.get(odd_id).success
    assert _all_ids(base_store) == {other}


def test_crafted_scope_dimension_cannot_widen_scoped_search(base_store):
    base_store.add(
        MemoryNote(content="tenant a", metadata={"user_id": 1, f"{P}tenant": "a"})
    )
    base_store.add(
        MemoryNote(content="tenant b", metadata={"user_id": 2, f"{P}tenant": "b"})
    )
    results = base_store.search(
        "tenant",
        k=10,
        filters={"metadata": {"user_id": 1, f"{P}tenant": "x') OR (1=1"}},
    )
    assert results == []


# --- vector store ------------------------------------------------------------


@pytest.mark.parametrize("crafted_id", MALICIOUS_IDS)
def test_vector_store_delete_is_exact_match(tmp_path, crafted_id):
    store = LanceDBVectorStore(str(tmp_path / "vectors"), "vectors")
    ids = store.add_vectors(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], ids=["keep-1", "keep-2"]
    )

    assert store.delete_vectors([crafted_id]) is True

    remaining = {row["id"] for row in store.search_vectors([1.0, 0.0, 0.0], 10)}
    assert remaining == set(ids)


def test_vector_store_delete_removes_only_listed_ids(tmp_path):
    store = LanceDBVectorStore(str(tmp_path / "vectors"), "vectors")
    store.add_vectors(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ids=["a'b", "c\\d", "keep"],
    )

    assert store.delete_vectors(["a'b", "c\\d"]) is True

    remaining = {row["id"] for row in store.search_vectors([1.0, 0.0, 0.0], 10)}
    assert remaining == {"keep"}
