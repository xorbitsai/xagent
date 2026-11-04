"""Test cases for InMemory memory store implementation."""

from datetime import datetime, timedelta, timezone

import pytest

from xagent.core.execution_scope import MEMORY_DIMENSION_METADATA_PREFIX
from xagent.core.memory.core import MemoryNote
from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.core.memory.scope_columns import SCOPE_EXCLUSIVE_FILTER_KEY


@pytest.fixture
def memory_store():
    """Create an InMemory memory store for testing."""
    return InMemoryMemoryStore()


@pytest.fixture
def sample_memory_note():
    """Create a sample memory note for testing."""
    return MemoryNote(
        id="test_123",
        content="Test memory content",
        keywords=["test", "sample"],
        tags=["experiment"],
        category="general",
        metadata={"source": "test", "priority": 1},
        timestamp=datetime.now(),
        mime_type="text/plain",
    )


class TestInMemoryMemoryStoreBasic:
    """Test cases for basic InMemory memory store operations."""

    def test_add_and_get_memory(self, memory_store, sample_memory_note):
        """Test successful memory addition and retrieval."""
        # Add memory
        result = memory_store.add(sample_memory_note)
        assert result.success is True
        assert result.memory_id == "test_123"

        # Get memory
        get_result = memory_store.get("test_123")
        assert get_result.success is True
        retrieved_note = get_result.content
        assert isinstance(retrieved_note, MemoryNote)
        assert retrieved_note.content == "Test memory content"
        assert retrieved_note.keywords == ["test", "sample"]
        assert retrieved_note.category == "general"

    def test_get_memory_not_found(self, memory_store):
        """Test memory retrieval for non-existent ID."""
        result = memory_store.get("non_existent")
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_update_memory(self, memory_store, sample_memory_note):
        """Test successful memory update."""
        # Add memory first
        memory_store.add(sample_memory_note)

        # Create updated memory
        updated_note = MemoryNote(
            id="test_123",
            content="Updated content",
            keywords=["updated"],
            tags=["modified"],
            category="general",
            metadata={"source": "test", "priority": 2},
            timestamp=sample_memory_note.timestamp,
            mime_type="text/plain",
        )

        result = memory_store.update(updated_note)
        assert result.success is True

        # Verify update was applied
        get_result = memory_store.get("test_123")
        assert get_result.success is True
        assert get_result.content.content == "Updated content"
        assert get_result.content.keywords == ["updated"]

    def test_update_memory_not_found(self, memory_store):
        """Test memory update for non-existent ID."""
        note = MemoryNote(id="non_existent", content="Test", category="general")

        result = memory_store.update(note)
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_delete_memory(self, memory_store, sample_memory_note):
        """Test successful memory deletion."""
        # Add memory first
        memory_store.add(sample_memory_note)

        # Delete memory
        result = memory_store.delete("test_123")
        assert result.success is True

        # Verify deletion was applied
        get_result = memory_store.get("test_123")
        assert get_result.success is False

    def test_delete_memory_not_found(self, memory_store):
        """Test memory deletion for non-existent ID."""
        result = memory_store.delete("non_existent")
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_clear_all_memories(self, memory_store):
        """Test clearing all memories."""
        # Add test memories
        memory_store.add(MemoryNote(content="Memory 1", category="general"))
        memory_store.add(MemoryNote(content="Memory 2", category="system"))

        # Clear all memories
        memory_store.clear()

        # Verify all memories are gone
        search_results = memory_store.search("Memory", k=10)
        assert len(search_results) == 0

        list_results = memory_store.list_all()
        assert len(list_results) == 0


class TestInMemoryMemoryStoreListAll:
    """Test cases for InMemory memory store list_all functionality."""

    def test_list_all_memories(self, memory_store):
        """Test listing all memories."""
        # Add test memories
        memories = [
            MemoryNote(content="Memory 1", category="general"),
            MemoryNote(content="Memory 2", category="system"),
            MemoryNote(content="Memory 3", category="general"),
        ]

        for memory in memories:
            memory_store.add(memory)

        # List all memories
        results = memory_store.list_all()
        assert len(results) >= 3
        assert all(isinstance(r, MemoryNote) for r in results)

    def test_list_all_with_category_filter(self, memory_store):
        """Test listing memories with category filter."""
        # Add test memories
        memories = [
            MemoryNote(content="General memory 1", category="general"),
            MemoryNote(content="General memory 2", category="general"),
            MemoryNote(content="System memory", category="system"),
        ]

        for memory in memories:
            memory_store.add(memory)

        # Test category filter
        general_results = memory_store.list_all(filters={"category": "general"})
        assert len(general_results) >= 2
        assert all(r.category == "general" for r in general_results)

        system_results = memory_store.list_all(filters={"category": "system"})
        assert len(system_results) >= 1
        assert all(r.category == "system" for r in system_results)

    def test_list_all_with_date_filters(self, memory_store):
        """Test listing memories with date range filters."""
        now = datetime.now()
        old_time = now - timedelta(days=1)
        future_time = now + timedelta(days=1)

        # Add memories with different timestamps
        memories = [
            MemoryNote(content="Old memory", category="test", timestamp=old_time),
            MemoryNote(content="Recent memory", category="test", timestamp=now),
            MemoryNote(content="Future memory", category="test", timestamp=future_time),
        ]

        for memory in memories:
            memory_store.add(memory)

        # Test date_from filter
        results = memory_store.list_all(filters={"date_from": now})
        assert len(results) >= 2  # Recent and future
        assert all(r.timestamp >= now for r in results)

        # Test date_to filter
        results = memory_store.list_all(filters={"date_to": now})
        assert len(results) >= 2  # Old and recent
        assert all(r.timestamp <= now for r in results)

    def test_list_all_with_tag_filters(self, memory_store):
        """Test listing memories with tag filters."""
        # Add memories with different tags
        memories = [
            MemoryNote(
                content="Memory with tag1", category="test", tags=["tag1", "common"]
            ),
            MemoryNote(
                content="Memory with tag2", category="test", tags=["tag2", "common"]
            ),
            MemoryNote(
                content="Memory with both tags",
                category="test",
                tags=["tag1", "tag2", "common"],
            ),
        ]

        for memory in memories:
            memory_store.add(memory)

        # Test single tag filter
        results = memory_store.list_all(filters={"tags": ["tag1"]})
        assert len(results) >= 2
        assert all("tag1" in r.tags for r in results)

        # Test multiple tag filter (AND logic)
        results = memory_store.list_all(filters={"tags": ["tag1", "tag2"]})
        assert len(results) >= 1
        assert all("tag1" in r.tags and "tag2" in r.tags for r in results)

    def test_list_all_with_keyword_filters(self, memory_store):
        """Test listing memories with keyword filters."""
        # Add memories with different keywords
        memories = [
            MemoryNote(
                content="Memory with keyword1",
                category="test",
                keywords=["keyword1", "common"],
            ),
            MemoryNote(
                content="Memory with keyword2",
                category="test",
                keywords=["keyword2", "common"],
            ),
            MemoryNote(
                content="Memory with both keywords",
                category="test",
                keywords=["keyword1", "keyword2", "common"],
            ),
        ]

        for memory in memories:
            memory_store.add(memory)

        # Test single keyword filter
        results = memory_store.list_all(filters={"keywords": ["keyword1"]})
        assert len(results) >= 2
        assert all("keyword1" in r.keywords for r in results)

        # Test multiple keyword filter (AND logic)
        results = memory_store.list_all(filters={"keywords": ["keyword1", "keyword2"]})
        assert len(results) >= 1
        assert all(
            "keyword1" in r.keywords and "keyword2" in r.keywords for r in results
        )


class TestInMemoryMemoryStoreStats:
    """Test cases for InMemory memory store statistics."""

    def test_get_stats(self, memory_store):
        """Test getting memory store statistics."""
        # Add test memories
        memories = [
            MemoryNote(
                content="General memory 1", category="general", tags=["important"]
            ),
            MemoryNote(content="General memory 2", category="general", tags=["normal"]),
            MemoryNote(
                content="System memory", category="system", tags=["system", "config"]
            ),
        ]

        for memory in memories:
            memory_store.add(memory)

        # Get stats
        stats = memory_store.get_stats()

        assert stats["total_count"] >= 3
        assert stats["category_counts"]["general"] >= 2
        assert stats["category_counts"]["system"] >= 1
        assert stats["tag_counts"]["important"] >= 1
        assert stats["tag_counts"]["normal"] >= 1
        assert stats["tag_counts"]["system"] >= 1
        assert stats["tag_counts"]["config"] >= 1
        assert stats["memory_store_type"] == "in_memory"

    def test_get_stats_empty_store(self, memory_store):
        """Test getting stats from empty store."""
        stats = memory_store.get_stats()

        assert stats["total_count"] == 0
        assert stats["category_counts"] == {}
        assert stats["tag_counts"] == {}
        assert stats["memory_store_type"] == "in_memory"


def test_add_and_get(memory_store):
    """Test basic add and get functionality (legacy test)."""
    note = MemoryNote(content="Test memory", metadata={"user": "alice"})
    response = memory_store.add(note)
    assert response.success
    assert response.memory_id is not None

    retrieved = memory_store.get(response.memory_id)
    assert retrieved.success
    assert isinstance(retrieved.content, MemoryNote)
    assert retrieved.content.content == "Test memory"


def test_update(memory_store):
    """Test basic update functionality (legacy test)."""
    note = MemoryNote(content="Original", metadata={"version": 1})
    response = memory_store.add(note)

    updated_note = MemoryNote(
        id=response.memory_id, content="Updated", metadata={"version": 2}
    )
    update_response = memory_store.update(updated_note)
    assert update_response.success

    get_response = memory_store.get(response.memory_id)
    assert isinstance(get_response.content, MemoryNote)
    assert get_response.content.content == "Updated"
    assert get_response.content.metadata["version"] == 2


def test_delete(memory_store):
    """Test basic delete functionality (legacy test)."""
    note = MemoryNote(content="To be deleted")
    response = memory_store.add(note)

    delete_response = memory_store.delete(response.memory_id)
    assert delete_response.success

    get_response = memory_store.get(response.memory_id)
    assert not get_response.success


def test_search(memory_store):
    """Test basic search functionality (legacy test)."""
    memory_store.add(MemoryNote(content="cats are great"))
    memory_store.add(MemoryNote(content="dogs are loyal"))

    results = memory_store.search("cats")
    assert any("cat" in note.content.lower() for note in results)


def test_clear(memory_store):
    """Test basic clear functionality (legacy test)."""
    memory_store.add(MemoryNote(content="temporary note"))
    memory_store.clear()

    results = memory_store.search("temporary")
    assert len(results) == 0


class TestInMemoryMemoryStoreNestedMetadataFilters:
    """#842: nested ``filters["metadata"]`` dicts (the shape
    ``UserIsolatedMemoryStore._add_user_filter`` emits) must be interpreted
    with the same string-coerced equality semantics as the LanceDB store
    (now the shared ``MemoryStore._matches_metadata_filters``, #916) —
    previously search() never matched them and list_all() silently ignored
    them (fail-open)."""

    @pytest.fixture(autouse=True)
    def seed(self, memory_store):
        memory_store.add(
            MemoryNote(
                id="alice",
                content="shared note text",
                metadata={"user_id": 1, "source": "chat"},
            )
        )
        memory_store.add(
            MemoryNote(
                id="bob",
                content="shared note text",
                metadata={"user_id": 2, "source": "chat"},
            )
        )

    def test_search_matches_nested_metadata_filter(self, memory_store):
        results = memory_store.search(
            "shared", k=10, filters={"metadata": {"user_id": 1}}
        )
        assert [n.id for n in results] == ["alice"]

    def test_search_nested_metadata_filter_no_match(self, memory_store):
        results = memory_store.search(
            "shared", k=10, filters={"metadata": {"user_id": 3}}
        )
        assert results == []

    def test_list_all_enforces_nested_metadata_filter(self, memory_store):
        results = memory_store.list_all(filters={"metadata": {"user_id": 2}})
        assert [n.id for n in results] == ["bob"]

    def test_list_all_nested_metadata_filter_no_match_is_empty(self, memory_store):
        # Previously fail-open: an unmatched nested filter returned every note.
        results = memory_store.list_all(filters={"metadata": {"user_id": 3}})
        assert results == []

    def test_nested_metadata_filter_multiple_keys(self, memory_store):
        assert [
            n.id
            for n in memory_store.list_all(
                filters={"metadata": {"user_id": 1, "source": "chat"}}
            )
        ] == ["alice"]
        assert (
            memory_store.list_all(
                filters={"metadata": {"user_id": 1, "source": "email"}}
            )
            == []
        )

    def test_nested_metadata_filter_string_coerced_equality(self, memory_store):
        # LanceDB compares str(metadata value) == str(filter value); the
        # in-memory store must match a string filter against an int value.
        results = memory_store.list_all(filters={"metadata": {"user_id": "1"}})
        assert [n.id for n in results] == ["alice"]

    def test_list_all_enforces_flat_metadata_filter(self, memory_store):
        # Flat metadata keys (outside the known category/date/tags/keywords
        # set) were previously ignored by list_all() — the same fail-open
        # shape as the nested case.
        results = memory_store.list_all(filters={"source": "chat", "user_id": 1})
        assert [n.id for n in results] == ["alice"]

        assert memory_store.list_all(filters={"source": "email"}) == []

    def test_search_flat_metadata_filter_string_coerced(self, memory_store):
        # Flat keys use the same string-coerced equality as LanceDB: a string
        # filter value matches an int metadata value.
        results = memory_store.search("shared", k=10, filters={"user_id": "2"})
        assert [n.id for n in results] == ["bob"]

    def test_nested_metadata_filter_combines_with_category(self, memory_store):
        memory_store.add(
            MemoryNote(
                id="alice-system",
                content="shared note text",
                category="system",
                metadata={"user_id": 1},
            )
        )
        results = memory_store.list_all(
            filters={"category": "system", "metadata": {"user_id": 1}}
        )
        assert [n.id for n in results] == ["alice-system"]

    def test_nested_metadata_filter_combines_with_flat_key(self, memory_store):
        # Both dispatch paths (nested dict + flat "other" key) applied in one
        # call — AND semantics across the two.
        results = memory_store.list_all(
            filters={"metadata": {"user_id": 1}, "source": "chat"}
        )
        assert [n.id for n in results] == ["alice"]

        assert (
            memory_store.list_all(
                filters={"metadata": {"user_id": 1}, "source": "email"}
            )
            == []
        )

    def test_non_dict_nested_metadata_filter_matches_nothing(self, memory_store):
        # A malformed filters["metadata"] value must not crash — it can't
        # match anything.
        for bad in (None, "abc", 42):
            assert memory_store.search("shared", k=10, filters={"metadata": bad}) == []
            assert memory_store.list_all(filters={"metadata": bad}) == []

    def test_flat_key_combines_with_scope_exclusive(self, memory_store):
        prefix = MEMORY_DIMENSION_METADATA_PREFIX
        memory_store.add(
            MemoryNote(
                id="alice-scoped",
                content="shared note text",
                metadata={"user_id": 1, "source": "chat", f"{prefix}tenant": "acme"},
            )
        )
        # user_id excludes bob (flat), the directive excludes alice-scoped.
        results = memory_store.list_all(
            filters={"user_id": 1, "source": "chat", SCOPE_EXCLUSIVE_FILTER_KEY: True}
        )
        assert [n.id for n in results] == ["alice"]


class TestSearchListOnlyFilterParity:
    """search() and list_all() share one filter dispatch: tags / keywords /
    date_from / date_to must behave identically on both. Previously search()
    let these keys fall through to flat metadata equality — they live on
    note.tags/note.keywords/note.timestamp, never note.metadata, so any
    query combining text search with one of them silently returned empty
    (reachable via the web memory list route, which calls search() whenever
    a text query is present)."""

    @pytest.fixture(autouse=True)
    def seed(self, memory_store):
        now = datetime.now()
        memory_store.add(
            MemoryNote(
                id="old-work",
                content="quarterly report draft",
                tags=["work"],
                keywords=["report"],
                timestamp=now - timedelta(days=2),
            )
        )
        memory_store.add(
            MemoryNote(
                id="new-home",
                content="grocery report list",
                tags=["home"],
                keywords=["groceries"],
                timestamp=now,
            )
        )
        self.now = now

    def test_search_with_tags_filter(self, memory_store):
        results = memory_store.search("report", k=10, filters={"tags": ["work"]})
        assert [n.id for n in results] == ["old-work"]

    def test_search_with_keywords_filter(self, memory_store):
        results = memory_store.search(
            "report", k=10, filters={"keywords": ["groceries"]}
        )
        assert [n.id for n in results] == ["new-home"]

    def test_search_with_date_filters(self, memory_store):
        cutoff = self.now - timedelta(days=1)
        assert [
            n.id
            for n in memory_store.search("report", k=10, filters={"date_from": cutoff})
        ] == ["new-home"]
        assert [
            n.id
            for n in memory_store.search("report", k=10, filters={"date_to": cutoff})
        ] == ["old-work"]

    def test_timezone_aware_date_filters(self, memory_store):
        # A tz-aware filter (FastAPI parses ISO date query params with an
        # offset into aware datetimes) must compare against the naive stored
        # timestamps instead of raising TypeError.
        cutoff_utc = (self.now - timedelta(days=1)).astimezone(timezone.utc)
        assert [
            n.id
            for n in memory_store.search(
                "report", k=10, filters={"date_from": cutoff_utc}
            )
        ] == ["new-home"]
        assert [
            n.id for n in memory_store.list_all(filters={"date_from": cutoff_utc})
        ] == ["new-home"]

    def test_timezone_aware_stored_timestamp_with_naive_filter(self, memory_store):
        # The other direction: an aware stored timestamp must compare against
        # a naive filter value.
        memory_store.add(
            MemoryNote(
                id="aware-note",
                content="weekly report summary",
                tags=["work"],
                timestamp=datetime.now(timezone.utc),
            )
        )
        results = memory_store.search(
            "report", k=10, filters={"date_from": self.now - timedelta(days=1)}
        )
        assert {n.id for n in results} == {"new-home", "aware-note"}

    def test_search_and_list_all_agree(self, memory_store):
        filters = {"tags": ["work"], "keywords": ["report"]}
        search_ids = {
            n.id for n in memory_store.search("report", k=10, filters=filters)
        }
        list_ids = {n.id for n in memory_store.list_all(filters=filters)}
        assert search_ids == list_ids == {"old-work"}

    def test_non_iterable_tags_keywords_match_nothing(self, memory_store):
        # Malformed (non-iterable) tags/keywords values can't match anything —
        # same no-match policy as a non-dict filters["metadata"], no TypeError
        # on either method.
        for key in ("tags", "keywords"):
            assert memory_store.search("report", k=10, filters={key: 5}) == []
            assert memory_store.list_all(filters={key: 5}) == []

    def test_empty_tags_keywords_match_everything(self, memory_store):
        # An empty list means "no tags/keywords required" — vacuously matches
        # every note, distinct from the malformed no-match case above.
        for key in ("tags", "keywords"):
            assert {
                n.id for n in memory_store.search("report", k=10, filters={key: []})
            } == {"old-work", "new-home"}
            assert {n.id for n in memory_store.list_all(filters={key: []})} == {
                "old-work",
                "new-home",
            }

    def test_string_tags_keywords_treated_as_single_item(self, memory_store):
        # A bare string is one required item, not a per-character iteration.
        assert [
            n.id for n in memory_store.search("report", k=10, filters={"tags": "work"})
        ] == ["old-work"]
        assert [
            n.id for n in memory_store.list_all(filters={"keywords": "groceries"})
        ] == ["new-home"]

    def test_explicit_none_date_bounds_mean_no_bound(self, memory_store):
        # #916: an explicit None date_from/date_to means "no bound" — same as
        # an absent key — instead of raising TypeError inside the date
        # comparison (this store has no outer exception guard, so the
        # TypeError used to propagate uncaught).
        for filters in (
            {"date_from": None},
            {"date_to": None},
            {"date_from": None, "date_to": None},
        ):
            assert {
                n.id for n in memory_store.search("report", k=10, filters=filters)
            } == {"old-work", "new-home"}
            assert {n.id for n in memory_store.list_all(filters=filters)} == {
                "old-work",
                "new-home",
            }

    def test_none_date_bound_combines_with_real_bound(self, memory_store):
        # A None bound on one side must not disable the real bound on the
        # other side — both directions.
        cutoff = self.now - timedelta(days=1)
        assert [
            n.id
            for n in memory_store.search(
                "report", k=10, filters={"date_from": cutoff, "date_to": None}
            )
        ] == ["new-home"]
        assert [
            n.id
            for n in memory_store.search(
                "report", k=10, filters={"date_from": None, "date_to": cutoff}
            )
        ] == ["old-work"]

    def test_category_filter_is_string_coerced(self, memory_store):
        # #916: the deliberate cross-store decision — category comparison is
        # string-coerced on both stores, consistent with the metadata-equality
        # policy of the shared dispatch (previously this store compared
        # exactly while LanceDB coerced).
        memory_store.add(
            MemoryNote(
                id="cat-42",
                content="numeric category report",
                category="42",
            )
        )
        assert [
            n.id for n in memory_store.search("report", k=10, filters={"category": 42})
        ] == ["cat-42"]
        assert [n.id for n in memory_store.list_all(filters={"category": 42})] == [
            "cat-42"
        ]


def test_scope_exclusive_filters_scoped_notes(memory_store):
    """#822: the `__scope_exclusive__` directive (strict dimension-less
    isolation) excludes any scope-stamped note on the real InMemoryMemoryStore,
    via the shared `_matches_filters` dispatch, on both search() and
    list_all()."""
    prefix = MEMORY_DIMENSION_METADATA_PREFIX
    memory_store.add(MemoryNote(id="u", content="hello unscoped", metadata={}))
    memory_store.add(
        MemoryNote(id="s", content="hello scoped", metadata={f"{prefix}tenant": "acme"})
    )

    got = memory_store.search("hello", k=10, filters={SCOPE_EXCLUSIVE_FILTER_KEY: True})
    assert {n.content for n in got} == {"hello unscoped"}

    got = memory_store.list_all(filters={SCOPE_EXCLUSIVE_FILTER_KEY: True})
    assert {n.content for n in got} == {"hello unscoped"}

    # Without the directive, both notes are visible.
    assert len(memory_store.search("hello", k=10)) == 2
