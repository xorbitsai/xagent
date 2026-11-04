"""Regression tests for #842: user_id isolation through
``UserIsolatedMemoryStore(InMemoryMemoryStore())`` — the default production
path when no embedding model is configured.

``UserIsolatedMemoryStore._add_user_filter`` nests the isolation filters under
``filters["metadata"]``. Before the fix, ``InMemoryMemoryStore.search()``
compared that nested dict flat (never matched → always empty) and
``list_all()`` ignored it entirely (fail-open → cross-user exposure).
"""

import pytest

from xagent.core.memory.core import MemoryNote
from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.web.user_isolated_memory import UserContext, UserIsolatedMemoryStore


@pytest.fixture
def store() -> UserIsolatedMemoryStore:
    return UserIsolatedMemoryStore(InMemoryMemoryStore())


@pytest.fixture
def seeded_store(store: UserIsolatedMemoryStore) -> UserIsolatedMemoryStore:
    with UserContext(1):
        store.add(MemoryNote(id="a1", content="project deadline friday"))
        store.add(MemoryNote(id="a2", content="project kickoff notes"))
    with UserContext(2):
        store.add(MemoryNote(id="b1", content="project deadline monday"))
    return store


class TestUserIsolationOnInMemoryPath:
    def test_search_returns_only_own_notes(self, seeded_store):
        with UserContext(1):
            results = seeded_store.search("project", k=10)
        assert {n.id for n in results} == {"a1", "a2"}

        with UserContext(2):
            results = seeded_store.search("project", k=10)
        assert {n.id for n in results} == {"b1"}

    def test_list_all_returns_only_own_notes(self, seeded_store):
        with UserContext(1):
            results = seeded_store.list_all()
        assert {n.id for n in results} == {"a1", "a2"}

        with UserContext(2):
            results = seeded_store.list_all()
        assert {n.id for n in results} == {"b1"}

    def test_list_all_with_caller_filters_keeps_isolation(self, seeded_store):
        # A caller-supplied nested metadata filter must be merged with, not
        # replace, the user_id isolation filter.
        with UserContext(1):
            seeded_store.add(
                MemoryNote(id="a3", content="tagged", metadata={"source": "chat"})
            )
        with UserContext(2):
            seeded_store.add(
                MemoryNote(id="b2", content="tagged", metadata={"source": "chat"})
            )
        with UserContext(1):
            results = seeded_store.list_all(filters={"metadata": {"source": "chat"}})
        assert {n.id for n in results} == {"a3"}

    def test_clear_only_removes_own_notes(self, seeded_store):
        # clear() deletes via list_all(); fail-open list_all made user 1's
        # clear wipe every user's notes.
        with UserContext(1):
            seeded_store.clear()
            assert seeded_store.list_all() == []
        with UserContext(2):
            assert {n.id for n in seeded_store.list_all()} == {"b1"}

    def test_no_user_context_sees_everything(self, seeded_store):
        # Without a user context no isolation filter is added — unchanged
        # pre-existing behavior for non-web callers.
        results = seeded_store.list_all()
        assert {n.id for n in results} == {"a1", "a2", "b1"}
