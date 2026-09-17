"""Slice 4 of #757: scope-aware memory isolation.

``UserIsolatedMemoryStore`` stamps the active scope's ``memory_dimensions``
onto note metadata on add and filters on them on search, alongside the
existing ``user_id`` handling. Default visibility is one-way; strict
isolation excludes scope-stamped notes from dimension-less searches. That
exclusion is carried in the filters as ``__scope_exclusive__`` and applied by
the store (#822 slice 003), not post-filtered in the wrapper.

The fake base store mirrors the LanceDB filter semantics: nested
``filters["metadata"]`` entries applied as string-equality checks, plus the
``__scope_exclusive__`` directive.
"""

from typing import Any, List, Optional

import pytest

from tests.shared.execution_scope import register_scope_resolver
from xagent.core.execution_scope import (
    ExecutionScope,
    ExecutionScopeContext,
    turn_execution_scope,
)
from xagent.core.memory.base import MemoryStore
from xagent.core.memory.core import MemoryNote, MemoryResponse
from xagent.core.memory.scope_columns import (
    SCOPE_EXCLUSIVE_FILTER_KEY,
    encode_scope_dims,
)
from xagent.web.user_isolated_memory import UserContext, UserIsolatedMemoryStore

SCOPE_A = ExecutionScope(memory_dimensions={"tenant": "a"})
SCOPE_B = ExecutionScope(memory_dimensions={"tenant": "b"})


class FakeBaseStore(MemoryStore):
    """LanceDB-flavored fake: nested metadata filters, string equality."""

    def __init__(self) -> None:
        self.notes: dict[str, MemoryNote] = {}

    def add(self, note: MemoryNote) -> MemoryResponse:
        self.notes[note.id] = note
        return MemoryResponse(success=True, memory_id=note.id)

    def get(self, note_id: str) -> MemoryResponse:
        note = self.notes.get(note_id)
        if note is None:
            return MemoryResponse(success=False, error="not found")
        return MemoryResponse(success=True, memory_id=note_id, content=note)

    def update(self, note: MemoryNote) -> MemoryResponse:
        self.notes[note.id] = note
        return MemoryResponse(success=True, memory_id=note.id)

    def delete(self, note_id: str) -> MemoryResponse:
        self.notes.pop(note_id, None)
        return MemoryResponse(success=True, memory_id=note_id)

    def _matches(self, note: MemoryNote, filters: Optional[dict]) -> bool:
        filters = filters or {}
        # Mirror the store's __scope_exclusive__ handling: strict dimension-less
        # searches exclude any scope-stamped note.
        if filters.get(SCOPE_EXCLUSIVE_FILTER_KEY) and encode_scope_dims(note.metadata):
            return False
        metadata_filters = filters.get("metadata", {})
        return all(
            str(note.metadata.get(key, "")) == str(value)
            for key, value in metadata_filters.items()
        )

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[MemoryNote]:
        return [
            note
            for note in self.notes.values()
            if query.lower() in str(note.content).lower()
            and self._matches(note, filters)
        ][:k]

    def clear(self) -> None:
        self.notes.clear()

    def list_all(self, filters: Optional[dict[str, Any]] = None) -> List[MemoryNote]:
        return [note for note in self.notes.values() if self._matches(note, filters)]

    def get_stats(self) -> dict[str, Any]:
        return {"total_count": len(self.notes), "memory_store_type": "fake"}


@pytest.fixture
def store() -> UserIsolatedMemoryStore:
    return UserIsolatedMemoryStore(FakeBaseStore())


def _add(store: UserIsolatedMemoryStore, content: str) -> MemoryNote:
    note = MemoryNote(content=content)
    assert store.add(note).success
    return note


def _seed_user_notes(store: UserIsolatedMemoryStore) -> None:
    """One unscoped note plus one note per scope, all for user 1."""
    with UserContext(1):
        _add(store, "note unscoped")
        with ExecutionScopeContext(SCOPE_A):
            _add(store, "note tenant a")
        with ExecutionScopeContext(SCOPE_B):
            _add(store, "note tenant b")


class TestScopedAdd:
    def test_add_stamps_prefixed_dimensions(self, store):
        with UserContext(1), ExecutionScopeContext(SCOPE_A):
            note = _add(store, "hello")
        assert note.metadata["user_id"] == 1
        assert note.metadata["execution_scope_tenant"] == "a"

    def test_unscoped_add_is_byte_identical(self, store):
        with UserContext(1):
            note = _add(store, "hello")
        assert note.metadata == {"user_id": 1}

    def test_dimensionless_scope_adds_nothing(self, store):
        """Fields are independent: a scope without memory_dimensions must
        not change note metadata."""
        with (
            UserContext(1),
            ExecutionScopeContext(ExecutionScope(sandbox_key_suffix="t-a")),
        ):
            note = _add(store, "hello")
        assert note.metadata == {"user_id": 1}


class TestOneWayVisibility:
    def test_scoped_searches_are_isolated(self, store):
        _seed_user_notes(store)
        with UserContext(1), ExecutionScopeContext(SCOPE_A):
            results = store.search("note")
        assert [n.content for n in results] == ["note tenant a"]

    def test_differently_scoped_searches_are_disjoint(self, store):
        _seed_user_notes(store)
        with UserContext(1):
            with ExecutionScopeContext(SCOPE_A):
                seen_a = {n.id for n in store.search("note")}
            with ExecutionScopeContext(SCOPE_B):
                seen_b = {n.id for n in store.search("note")}
        assert seen_a and seen_b
        assert seen_a.isdisjoint(seen_b)

    def test_unscoped_search_sees_everything_under_the_user(self, store):
        _seed_user_notes(store)
        with UserContext(1):
            results = store.search("note")
        assert {str(n.content) for n in results} == {
            "note unscoped",
            "note tenant a",
            "note tenant b",
        }

    def test_other_users_notes_stay_invisible(self, store):
        _seed_user_notes(store)
        with UserContext(2), ExecutionScopeContext(SCOPE_A):
            assert store.search("note") == []
        with UserContext(2):
            assert store.search("note") == []


class TestStrictIsolation:
    def test_strict_excludes_scope_stamped_notes(self, store):
        _seed_user_notes(store)
        strict = ExecutionScope(strict_memory_isolation=True)
        with UserContext(1), ExecutionScopeContext(strict):
            results = store.search("note")
        assert [str(n.content) for n in results] == ["note unscoped"]

    def test_strict_flag_works_with_otherwise_empty_scope(self, store):
        """The flag is consumed even when every other field is empty."""
        _seed_user_notes(store)
        strict = ExecutionScope(strict_memory_isolation=True)
        assert strict.memory_dimensions == {} and strict.sandbox_key_suffix is None
        with UserContext(1), ExecutionScopeContext(strict):
            results = store.list_all()
        assert [str(n.content) for n in results] == ["note unscoped"]

    def test_without_strict_default_stays_one_way(self, store):
        _seed_user_notes(store)
        relaxed = ExecutionScope(strict_memory_isolation=False)
        with UserContext(1), ExecutionScopeContext(relaxed):
            results = store.search("note")
        assert len(results) == 3

    def test_strict_scoped_search_still_filters_to_its_dimensions(self, store):
        """Strict + dimensions: the dimension filter already isolates; the
        post-filter must not empty the results."""
        _seed_user_notes(store)
        strict_a = ExecutionScope(
            memory_dimensions={"tenant": "a"}, strict_memory_isolation=True
        )
        with UserContext(1), ExecutionScopeContext(strict_a):
            results = store.search("note")
        assert [str(n.content) for n in results] == ["note tenant a"]


class TestCallerFiltersNotMutated:
    """``_add_user_filter`` must work on copies: a caller-supplied ``filters``
    dict (and its nested ``metadata`` dict) is never mutated in place, so the
    same object can be reused across scopes."""

    def test_search_leaves_caller_filters_untouched(self, store):
        _seed_user_notes(store)
        original = {"metadata": {"foo": "bar"}}
        strict = ExecutionScope(strict_memory_isolation=True)
        with UserContext(1), ExecutionScopeContext(strict):
            store.search("note", filters=original)
        # No injected user_id / scope-exclusive directive leaked back out.
        assert original == {"metadata": {"foo": "bar"}}

    def test_same_filters_dict_reused_across_scopes(self, store):
        _seed_user_notes(store)
        shared = {"metadata": {}}
        with UserContext(1):
            with ExecutionScopeContext(SCOPE_A):
                seen_a = {n.id for n in store.search("note", filters=shared)}
            with ExecutionScopeContext(SCOPE_B):
                seen_b = {n.id for n in store.search("note", filters=shared)}
        # A leaked SCOPE_A dimension would empty (or cross-contaminate) the
        # SCOPE_B search; disjoint results prove no state carried over.
        assert seen_a and seen_b
        assert seen_a.isdisjoint(seen_b)
        assert shared == {"metadata": {}}


class TestResolverPathAndNestedReactivation:
    def test_turn_scope_reaches_memory_via_resolver(self, store):
        """The resolver path: memory operations inside a turn carry the
        resolved scope without any explicit plumbing."""
        register_scope_resolver(lambda task_id: SCOPE_A)
        with UserContext(1):
            with turn_execution_scope(42):
                stamped = _add(store, "from the turn")
            unscoped_view = store.search("from the turn")
        assert stamped.metadata["execution_scope_tenant"] == "a"
        assert len(unscoped_view) == 1  # one-way: unscoped sees it

    def test_nested_snapshot_reactivation_lands_in_parent_scope(self, store):
        """The nested-agent pattern: a snapshot captured at tool
        construction is re-activated around the nested execution (no
        re-resolution), so nested memory writes land in the parent's
        scope even when the ambient context is empty."""
        with UserContext(1), ExecutionScopeContext(SCOPE_A):
            snapshot = SCOPE_A  # what AgentTool captures at construction

        # Nested execution later, ambient context unscoped (e.g. a fresh
        # context after restart-like conditions):
        with UserContext(1), ExecutionScopeContext(snapshot):
            nested_note = _add(store, "nested write")

        assert nested_note.metadata["execution_scope_tenant"] == "a"
        with UserContext(1), ExecutionScopeContext(SCOPE_A):
            assert {n.id for n in store.search("nested")} == {nested_note.id}
        with UserContext(1), ExecutionScopeContext(SCOPE_B):
            assert store.search("nested") == []


_DENIED = "Memory note not found or access denied"


class TestScopedByIdAccess:
    """get/update/delete enforce the active scope's dimensions, mirroring
    search/list_all — a scoped caller cannot read, modify, or delete another
    scope's note by id even under a shared user_id (#80 review point 4). A
    mismatch is indistinguishable from a genuine miss."""

    @staticmethod
    def _note_in(store, scope, content):
        with UserContext(1), ExecutionScopeContext(scope):
            return _add(store, content)

    def test_get_denied_across_dimensions(self, store):
        note_b = self._note_in(store, SCOPE_B, "b note")
        with UserContext(1), ExecutionScopeContext(SCOPE_A):
            resp = store.get(note_b.id)
        assert resp.success is False
        assert resp.error == _DENIED

    def test_get_allowed_for_own_dimensions(self, store):
        note_a = self._note_in(store, SCOPE_A, "a note")
        with UserContext(1), ExecutionScopeContext(SCOPE_A):
            resp = store.get(note_a.id)
        assert resp.success is True
        assert resp.content.id == note_a.id

    def test_update_denied_across_dimensions_leaves_note_intact(self, store):
        note_b = self._note_in(store, SCOPE_B, "b note")
        edited = note_b.model_copy(deep=True)
        edited.content = "hijacked"
        with UserContext(1), ExecutionScopeContext(SCOPE_A):
            resp = store.update(edited)
        assert resp.success is False
        assert resp.error == _DENIED
        assert store._base_store.notes[note_b.id].content == "b note"

    def test_delete_denied_across_dimensions_leaves_note_intact(self, store):
        note_b = self._note_in(store, SCOPE_B, "b note")
        with UserContext(1), ExecutionScopeContext(SCOPE_A):
            resp = store.delete(note_b.id)
        assert resp.success is False
        assert note_b.id in store._base_store.notes

    def test_update_and_delete_allowed_for_own_dimensions(self, store):
        note_a = self._note_in(store, SCOPE_A, "a note")
        edited = note_a.model_copy(deep=True)
        edited.content = "edited a"
        with UserContext(1), ExecutionScopeContext(SCOPE_A):
            assert store.update(edited).success is True
            assert store.delete(note_a.id).success is True
        assert note_a.id not in store._base_store.notes

    def test_strict_dimensionless_cannot_touch_scoped_note_by_id(self, store):
        note_a = self._note_in(store, SCOPE_A, "a note")
        strict = ExecutionScope(strict_memory_isolation=True)
        with UserContext(1), ExecutionScopeContext(strict):
            assert store.get(note_a.id).success is False
            assert store.delete(note_a.id).success is False
        assert note_a.id in store._base_store.notes

    def test_nonstrict_dimensionless_keeps_one_way_by_id_access(self, store):
        """Default one-way visibility: a dimension-less non-strict caller can
        still reach a scoped note by id (matches unscoped search seeing
        everything under the user)."""
        note_a = self._note_in(store, SCOPE_A, "a note")
        with UserContext(1):
            assert store.get(note_a.id).success is True
            assert store.delete(note_a.id).success is True
        assert note_a.id not in store._base_store.notes

    def test_unscoped_by_id_behavior_unchanged(self, store):
        """No user context and no scope: the original no-check by-id path is
        preserved (get/update/delete all succeed)."""
        note = MemoryNote(content="free")
        store.add(note)
        assert store.get(note.id).success is True
        edited = note.model_copy(deep=True)
        edited.content = "free edited"
        assert store.update(edited).success is True
        assert store._base_store.notes[note.id].content == "free edited"
        assert store.delete(note.id).success is True
        assert note.id not in store._base_store.notes
