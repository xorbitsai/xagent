"""Tests for LanceDB-backed storage implementations."""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
    LanceDBMainPointerStore,
    LanceDBMetadataStore,
    LanceDBPromptTemplateStore,
    LanceDBVectorIndexStore,
)


def create_mock_arrow_table(data_list: List[Dict[str, Any]]) -> Mock:
    """Create a mock Arrow table that supports to_pylist() and len()."""
    mock_table = Mock()
    mock_table.to_pylist = Mock(return_value=data_list)
    mock_table.__len__ = Mock(return_value=len(data_list))
    # Support iteration for 'for row in result' patterns
    mock_table.__iter__ = Mock(return_value=iter(data_list))
    return mock_table


@pytest.fixture(autouse=True)
def mock_schema_manager_user_id_migration() -> None:
    """Disable schema-manager user_id migration side effects in unit tests."""
    with patch(
        "xagent.core.tools.core.RAG_tools.LanceDB.schema_manager._migrate_table_user_id_to_int64",
        return_value=None,
    ):
        yield


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_metadata_store_save_collection_config(mock_get_connection: Mock) -> None:
    """Metadata store should save collection config correctly."""
    from types import SimpleNamespace

    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    mock_table = Mock()
    # Mock schema as iterable for _ensure_schema_fields
    mock_table.schema = [SimpleNamespace(name="collection")]
    mock_conn.open_table.return_value = mock_table

    store = LanceDBMetadataStore()
    asyncio.run(
        store.save_collection_config(
            collection="test_collection",
            config_json='{"parse_method": "default"}',
            user_id=1,
        )
    )

    # Verify table.delete was called to remove existing config
    mock_table.delete.assert_called_once()

    # Verify table.add was called with new config
    mock_table.add.assert_called_once()
    added_data = mock_table.add.call_args[0][0]
    assert added_data[0]["collection"] == "test_collection"
    assert added_data[0]["config_json"] == '{"parse_method": "default"}'
    assert added_data[0]["user_id"] == 1


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_metadata_store_get_collection_config_success(
    mock_get_connection: Mock,
) -> None:
    """Metadata store should retrieve collection config correctly."""
    from types import SimpleNamespace

    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    mock_table = Mock()
    # Mock schema as iterable for _ensure_schema_fields
    mock_table.schema = [SimpleNamespace(name="collection")]
    mock_conn.open_table.return_value = mock_table

    # Mock Arrow table with result[0]["config_json"].as_py() access pattern
    mock_scalar = Mock()
    mock_scalar.as_py = Mock(return_value='{"parse_method": "default"}')

    mock_config_col = Mock()
    mock_config_col.__getitem__ = Mock(return_value=mock_scalar)

    mock_result = Mock()
    mock_result.__len__ = Mock(return_value=1)
    mock_result.__getitem__ = Mock(
        side_effect=lambda key: mock_config_col if key == "config_json" else Mock()
    )

    mock_table.search.return_value.where.return_value.to_arrow.return_value = (
        mock_result
    )

    store = LanceDBMetadataStore()
    config = asyncio.run(
        store.get_collection_config(collection="test_collection", user_id=1)
    )

    assert config == '{"parse_method": "default"}'


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_metadata_store_get_collection_config_not_found(
    mock_get_connection: Mock,
) -> None:
    """Metadata store should return None when config not found."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    mock_table = Mock()
    mock_table.schema = Mock(names=[])
    mock_conn.open_table.return_value = mock_table
    mock_result = Mock()
    mock_result.__len__ = Mock(return_value=0)
    mock_table.search.return_value.where.return_value.to_arrow.return_value = (
        mock_result
    )

    store = LanceDBMetadataStore()
    config = asyncio.run(
        store.get_collection_config(collection="test_collection", user_id=1)
    )

    assert config is None


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_metadata_store_get_collection_config_admin_picks_newest(
    mock_get_connection: Mock,
) -> None:
    """When is_admin, multiple tenant rows should resolve to latest updated_at."""
    import pyarrow as pa

    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    mock_table = Mock()
    mock_table.schema = Mock(names=[])
    mock_conn.open_table.return_value = mock_table

    older = datetime(2020, 1, 1)
    newer = datetime(2021, 6, 1)
    tbl = pa.table(
        {
            "collection": ["test_collection", "test_collection"],
            "config_json": [
                '{"parse_method": "default"}',
                '{"parse_method": "deepdoc"}',
            ],
            "updated_at": [older, newer],
            "user_id": [1, 2],
        }
    )
    mock_table.search.return_value.where.return_value.to_arrow.return_value = tbl

    store = LanceDBMetadataStore()
    config = asyncio.run(
        store.get_collection_config(
            collection="test_collection", user_id=0, is_admin=True
        )
    )

    assert config == '{"parse_method": "deepdoc"}'


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_metadata_store_get_collection_success(mock_get_connection: Mock) -> None:
    """Metadata store should deserialize collection metadata correctly."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    mock_table = Mock()
    mock_table.schema = Mock(names=[])
    mock_conn.open_table.return_value = mock_table

    # Use helper to create mock Arrow table
    mock_data = {
        "name": "test_collection",
        "schema_version": "1.0.0",
        "embedding_model_id": "text-embedding-v4",
        "embedding_dimension": 1024,
        "documents": 2,
        "processed_documents": 2,
        "parses": 2,
        "chunks": 8,
        "embeddings": 8,
        "document_names": '["a.pdf","b.pdf"]',
        "collection_locked": False,
        "allow_mixed_parse_methods": False,
        "skip_config_validation": False,
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "last_accessed_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "extra_metadata": "{}",
    }

    mock_table.search.return_value.where.return_value.to_arrow.return_value = (
        create_mock_arrow_table([mock_data])
    )

    store = LanceDBMetadataStore()
    collection = asyncio.run(store.get_collection("test_collection"))
    assert collection.name == "test_collection"
    assert collection.documents == 2
    assert collection.document_names == ["a.pdf", "b.pdf"]


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.UserPermissions.get_user_filter"
)
@patch("xagent.core.tools.core.RAG_tools.storage.lancedb_stores.query_to_list")
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_vector_store_list_document_records_filters_and_maps(
    mock_get_connection: Mock,
    mock_query_to_list: Mock,
    mock_user_filter: Mock,
) -> None:
    """Vector store should apply combined filter and map to DocumentRecord."""
    from types import SimpleNamespace

    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    mock_user_filter.return_value = "user_id == 1"
    mock_table = Mock()
    # Mock schema as iterable for _ensure_schema_fields
    mock_table.schema = [SimpleNamespace(name="doc_id")]
    mock_conn.open_table.return_value = mock_table
    mock_query_to_list.return_value = [
        {"doc_id": "doc-1", "source_path": "/tmp/a.pdf"},
        {"doc_id": "doc-2", "source_path": None},
    ]

    store = LanceDBVectorIndexStore()
    records = store.list_document_records(
        collection_name="kb1",
        user_id=1,
        is_admin=False,
        max_results=50,
    )

    assert [r.doc_id for r in records] == ["doc-1", "doc-2"]
    assert records[0].source_path == "/tmp/a.pdf"
    mock_table.search.return_value.where.assert_called_once()


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_vector_store_rename_collection_data_updates_expected_tables(
    mock_get_connection: Mock,
) -> None:
    """Rename should update core and embeddings tables only."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    table_names = [
        "documents",
        "parses",
        "chunks",
        "embeddings_text_embedding_v4",
        "collection_metadata",
    ]
    mock_conn.table_names.return_value = table_names
    mock_conn.list_tables.return_value = table_names
    mock_table = Mock()
    mock_table.schema = Mock(names=[])
    mock_conn.open_table.return_value = mock_table

    store = LanceDBVectorIndexStore()
    warnings = store.rename_collection_data("old_name", "new_name")

    assert warnings == []
    # 4 target tables should be updated; control-plane table excluded.
    assert mock_table.update.call_count == 4


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_delete_collection_data_raises_when_any_table_delete_fails(
    mock_get_connection: Mock,
) -> None:
    """Collection deletes should preserve partial cleanup counts and warnings."""

    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_conn.table_names.return_value = [
        "documents",
        "parses",
        "chunks",
        "embeddings_text_embedding_v4",
    ]
    mock_conn.list_tables.return_value = [
        "documents",
        "parses",
        "chunks",
        "embeddings_text_embedding_v4",
    ]

    def _build_table(
        *, before: int, after: int = 0, delete_side_effect: Exception | None = None
    ):
        table = Mock()
        schema = Mock(names=["collection", "doc_id", "user_id", "file_id"])
        field = Mock()
        field.type = "int64"
        schema.field.return_value = field
        table.schema = schema
        if delete_side_effect is None:
            table.count_rows.side_effect = [before, after]
        else:
            table.count_rows.return_value = before
            table.delete.side_effect = delete_side_effect
        return table

    documents_table = _build_table(before=1)
    parses_table = _build_table(
        before=1, delete_side_effect=RuntimeError("parse delete failed")
    )
    chunks_table = _build_table(before=0)
    embeddings_table = _build_table(before=2)
    tables = {
        "documents": documents_table,
        "parses": parses_table,
        "chunks": chunks_table,
        "embeddings_text_embedding_v4": embeddings_table,
    }
    mock_conn.open_table.side_effect = lambda name: tables[name]

    store = LanceDBVectorIndexStore()
    warnings: List[str] = []

    deleted_counts = store.delete_collection_data(
        "demo", user_id=1, is_admin=False, warnings_out=warnings
    )

    assert deleted_counts == {"documents": 1, "embeddings_text_embedding_v4": 2}
    assert warnings == ["Failed to delete from 'parses': parse delete failed"]
    documents_table.delete.assert_called_once()
    parses_table.delete.assert_called_once()
    embeddings_table.delete.assert_called_once()


# --- Upsert Fallback Tests ---


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_upsert_embeddings_merge_insert_success(mock_get_connection: Mock) -> None:
    """Test successful merge_insert upsert."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_conn.list_tables.return_value = []

    mock_table = Mock()
    mock_table.schema = Mock(names=["collection", "doc_id", "chunk_id", "vector"])
    mock_conn.open_table.return_value = mock_table

    # Mock merge_insert chain
    mock_merge_insert = Mock()
    mock_when_matched = Mock()
    mock_when_not_matched = Mock()
    mock_table.merge_insert.return_value = mock_merge_insert
    mock_merge_insert.when_matched_update_all.return_value = mock_when_matched
    mock_when_matched.when_not_matched_insert_all.return_value = mock_when_not_matched
    mock_when_not_matched.execute.return_value = None

    store = LanceDBVectorIndexStore()

    records = [
        {
            "collection": "test_col",
            "doc_id": "doc1",
            "chunk_id": "chunk1",
            "vector": [0.1, 0.2],
            "text": "test",
        }
    ]

    store.upsert_embeddings("text_embedding_v4", records)

    # Verify merge_insert was called
    mock_table.merge_insert.assert_called_once_with(
        ["collection", "doc_id", "chunk_id"]
    )
    mock_merge_insert.when_matched_update_all.assert_called_once()
    mock_when_matched.when_not_matched_insert_all.assert_called_once()
    mock_when_not_matched.execute.assert_called_once()

    # Verify add was NOT called (no fallback needed)
    mock_table.add.assert_not_called()


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_upsert_embeddings_merge_insert_fallback_to_add(
    mock_get_connection: Mock,
) -> None:
    """Test fallback to add() when merge_insert fails with recoverable error."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_conn.list_tables.return_value = []

    mock_table = Mock()
    mock_table.schema = Mock(names=["collection", "doc_id", "chunk_id", "vector"])
    mock_conn.open_table.return_value = mock_table

    # Mock merge_insert chain that fails
    mock_merge_insert = Mock()
    mock_when_matched = Mock()
    mock_when_not_matched = Mock()
    mock_table.merge_insert.return_value = mock_merge_insert
    mock_merge_insert.when_matched_update_all.return_value = mock_when_matched
    mock_when_matched.when_not_matched_insert_all.return_value = mock_when_not_matched
    # merge_insert fails with recoverable error (e.g., network issue)
    mock_when_not_matched.execute.side_effect = Exception("Temporary network error")

    # Mock add() to succeed
    mock_table.add.return_value = None

    store = LanceDBVectorIndexStore()

    records = [
        {
            "collection": "test_col",
            "doc_id": "doc1",
            "chunk_id": "chunk1",
            "vector": [0.1, 0.2],
            "text": "test",
        }
    ]

    store.upsert_embeddings("text_embedding_v4", records)

    # Verify merge_insert was attempted
    mock_table.merge_insert.assert_called_once()

    # Verify fallback to add() was used
    mock_table.add.assert_called_once()


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_upsert_embeddings_non_recoverable_error_no_fallback(
    mock_get_connection: Mock,
) -> None:
    """Test that non-recoverable errors (schema, type mismatch) do not fallback."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_conn.list_tables.return_value = []

    mock_table = Mock()
    mock_table.schema = Mock(names=["collection", "doc_id", "chunk_id", "vector"])
    mock_conn.open_table.return_value = mock_table

    # Mock merge_insert chain that fails with non-recoverable error
    mock_merge_insert = Mock()
    mock_when_matched = Mock()
    mock_when_not_matched = Mock()
    mock_table.merge_insert.return_value = mock_merge_insert
    mock_merge_insert.when_matched_update_all.return_value = mock_when_matched
    mock_when_matched.when_not_matched_insert_all.return_value = mock_when_not_matched
    # Schema error - should NOT fallback
    mock_when_not_matched.execute.side_effect = ValueError("Schema mismatch")

    store = LanceDBVectorIndexStore()

    records = [
        {
            "collection": "test_col",
            "doc_id": "doc1",
            "chunk_id": "chunk1",
            "vector": [0.1, 0.2],
            "text": "test",
        }
    ]

    # Should raise ValueError without fallback
    with pytest.raises(ValueError, match="Schema mismatch"):
        store.upsert_embeddings("text_embedding_v4", records)

    # Verify merge_insert was attempted
    mock_table.merge_insert.assert_called_once()

    # Verify add() was NOT called (no fallback for non-recoverable errors)
    mock_table.add.assert_not_called()


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_upsert_embeddings_both_methods_fail(mock_get_connection: Mock) -> None:
    """Test that error is raised when both merge_insert and add() fail."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_conn.list_tables.return_value = []

    mock_table = Mock()
    mock_table.schema = Mock(names=["collection", "doc_id", "chunk_id", "vector"])
    mock_conn.open_table.return_value = mock_table

    # Mock merge_insert chain that fails
    mock_merge_insert = Mock()
    mock_when_matched = Mock()
    mock_when_not_matched = Mock()
    mock_table.merge_insert.return_value = mock_merge_insert
    mock_merge_insert.when_matched_update_all.return_value = mock_when_matched
    mock_when_matched.when_not_matched_insert_all.return_value = mock_when_not_matched
    mock_when_not_matched.execute.side_effect = Exception("merge_insert failed")

    # Mock add() to also fail
    mock_table.add.side_effect = Exception("add() also failed")

    store = LanceDBVectorIndexStore()

    records = [
        {
            "collection": "test_col",
            "doc_id": "doc1",
            "chunk_id": "chunk1",
            "vector": [0.1, 0.2],
            "text": "test",
        }
    ]

    # Should raise when both methods fail
    with pytest.raises(Exception, match="add.*also failed"):
        store.upsert_embeddings("text_embedding_v4", records)

    # Verify both methods were attempted
    mock_table.merge_insert.assert_called_once()
    mock_table.add.assert_called_once()


# ============================================================================
# Index Management Tests (Phase 1A Part 2)
# ============================================================================


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_should_reindex_immediate_reindex_enabled(
    mock_get_connection: Mock,
) -> None:
    """Test should_reindex returns True when immediate reindex is enabled."""
    from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy

    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    # Mock index stats
    mock_stats = Mock()
    mock_stats.num_indexed_rows = 1000
    mock_stats.num_unindexed_rows = 100
    mock_table.index_stats.return_value = mock_stats

    store = LanceDBVectorIndexStore()

    policy = IndexPolicy(
        reindex_batch_size=1000,
        enable_immediate_reindex=True,
        enable_smart_reindex=False,
    )

    result = store.should_reindex("embeddings_test", total_upserted=10, policy=policy)

    assert result is True  # immediate reindex enabled


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_should_reindex_batch_threshold(
    mock_get_connection: Mock,
) -> None:
    """Test should_reindex returns True when batch size threshold reached."""
    from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy

    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    store = LanceDBVectorIndexStore()

    policy = IndexPolicy(
        reindex_batch_size=100,
        enable_immediate_reindex=False,
        enable_smart_reindex=False,
    )

    # Total upserted >= batch_size
    result = store.should_reindex("embeddings_test", total_upserted=100, policy=policy)
    assert result is True

    # Below threshold
    result = store.should_reindex("embeddings_test", total_upserted=99, policy=policy)
    assert result is False


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_should_reindex_smart_reindex(
    mock_get_connection: Mock,
) -> None:
    """Test should_reindex with smart reindex enabled."""
    from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy

    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    # Mock index stats with high unindexed ratio
    mock_stats = Mock()
    mock_stats.num_indexed_rows = 100
    mock_stats.num_unindexed_rows = 60  # 60% unindexed
    mock_table.index_stats.return_value = mock_stats

    store = LanceDBVectorIndexStore()

    policy = IndexPolicy(
        reindex_batch_size=10000,
        enable_immediate_reindex=False,
        enable_smart_reindex=True,
        reindex_unindexed_ratio_threshold=0.5,  # 50% threshold
    )

    # High unindexed ratio should trigger reindex
    result = store.should_reindex("embeddings_test", total_upserted=10, policy=policy)
    assert result is True


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_trigger_reindex_success(mock_get_connection: Mock) -> None:
    """Test trigger_reindex calls table.optimize()."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    store = LanceDBVectorIndexStore()

    result = store.trigger_reindex("embeddings_test")

    assert result is True
    mock_table.optimize.assert_called_once()


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_trigger_reindex_failure(mock_get_connection: Mock) -> None:
    """Test trigger_reindex returns False on exception."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table
    mock_table.optimize.side_effect = Exception("Optimize failed")

    store = LanceDBVectorIndexStore()

    result = store.trigger_reindex("embeddings_test")

    assert result is False


@pytest.mark.asyncio
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
async def test_should_reindex_async_delegates_to_sync(
    mock_get_connection: Mock,
) -> None:
    """Test async version delegates to sync implementation."""
    from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy

    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    # Mock index stats with high unindexed ratio (60%)
    mock_stats = Mock()
    mock_stats.num_indexed_rows = 100
    mock_stats.num_unindexed_rows = 60  # 60% unindexed, exceeds 50% threshold
    mock_table.index_stats.return_value = mock_stats

    store = LanceDBVectorIndexStore()

    policy = IndexPolicy(
        reindex_batch_size=10000,
        enable_immediate_reindex=False,
        enable_smart_reindex=True,
        reindex_unindexed_ratio_threshold=0.5,
    )

    # Async version should delegate to sync
    result = await store.should_reindex_async(
        "embeddings_test", total_upserted=10, policy=policy
    )
    assert result is True  # Smart reindex triggers due to high unindexed ratio


@pytest.mark.asyncio
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
async def test_trigger_reindex_async_delegates_to_sync(
    mock_get_connection: Mock,
) -> None:
    """Test async trigger_reindex delegates to sync implementation."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    store = LanceDBVectorIndexStore()

    # Async version should delegate to sync
    result = await store.trigger_reindex_async("embeddings_test")
    assert result is True
    mock_table.optimize.assert_called_once()


# ============================================================================
# PromptTemplateStore Tests (Phase 1A Part 3)
# ============================================================================


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_prompt_template_store_save_and_get(mock_get_connection: Mock) -> None:
    """Test saving and retrieving a prompt template."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    # Mock empty result for existing check
    mock_result = Mock()
    mock_result.__len__ = Mock(return_value=0)
    mock_table.search.return_value.where.return_value.to_arrow.return_value = (
        mock_result
    )

    store = LanceDBPromptTemplateStore()

    # Save template
    template_id = store.save_prompt_template(
        name="test_template",
        template="Test prompt content",
        user_id=1,
    )

    assert template_id is not None
    mock_table.add.assert_called_once()

    # Mock get result
    row_data = {
        "id": template_id,
        "name": "test_template",
        "template": "Test prompt content",
        "version": 1,
        "is_latest": True,
        "metadata": "",
        "user_id": 1,
        "created_at": None,
        "updated_at": None,
    }
    mock_table.search.return_value.where.return_value.to_arrow.return_value = (
        create_mock_arrow_table([row_data])
    )

    # Get template
    template = store.get_prompt_template(template_id, user_id=1)
    assert template is not None
    assert template["name"] == "test_template"


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_prompt_template_store_get_latest(mock_get_connection: Mock) -> None:
    """Test getting the latest version of a template by name."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    # Mock result
    row_data = {
        "id": "test-id",
        "name": "test_template",
        "template": "Latest content",
        "version": 2,
        "is_latest": True,
        "metadata": "",
        "user_id": 1,
        "created_at": None,
        "updated_at": None,
    }
    mock_table.search.return_value.where.return_value.to_arrow.return_value = (
        create_mock_arrow_table([row_data])
    )

    store = LanceDBPromptTemplateStore()

    template = store.get_latest_prompt_template("test_template", user_id=1)
    assert template is not None
    assert template["version"] == 2
    assert template["template"] == "Latest content"


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_prompt_template_store_delete(mock_get_connection: Mock) -> None:
    """Test deleting a prompt template."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    # Mock existing template
    mock_row = {"is_latest": True, "name": "test-template"}
    mock_result = create_mock_arrow_table([mock_row])

    # Mock remaining versions after delete (empty for this test)
    mock_result_empty = create_mock_arrow_table([])

    mock_table.search.return_value.where.return_value.to_arrow.side_effect = [
        mock_result,
        mock_result_empty,
    ]

    store = LanceDBPromptTemplateStore()

    result = store.delete_prompt_template("test-id", user_id=1)
    assert result is True
    mock_table.delete.assert_called_once()


# ============================================================================
# MainPointerStore Tests (Phase 1A Part 3)
# ============================================================================


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_main_pointer_store_set_and_get(mock_get_connection: Mock) -> None:
    """Test setting and getting a main pointer."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    # Mock no existing pointer
    mock_result = Mock()
    mock_result.__len__ = Mock(return_value=0)
    mock_table.search.return_value.where.return_value.to_arrow.return_value = (
        mock_result
    )

    store = LanceDBMainPointerStore()

    # Set pointer
    store.set_main_pointer(
        collection="test_collection",
        doc_id="test_doc",
        step_type="parse",
        semantic_id="parse-123",
        technical_id="hash-456",
    )

    # Verify merge_insert was called
    mock_table.merge_insert.assert_called_once()

    # Mock get result
    mock_row = {
        "collection": "test_collection",
        "doc_id": "test_doc",
        "step_type": "parse",
        "model_tag": "",
        "semantic_id": "parse-123",
        "technical_id": "hash-456",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "operator": "unknown",
    }

    mock_table.search.return_value.where.return_value.to_arrow.return_value = (
        create_mock_arrow_table([mock_row])
    )

    # Get pointer
    pointer = store.get_main_pointer("test_collection", "test_doc", "parse")
    assert pointer is not None
    assert pointer["semantic_id"] == "parse-123"
    assert pointer["technical_id"] == "hash-456"


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_main_pointer_store_user_id_warning(mock_get_connection: Mock, caplog) -> None:
    """Test that user_id parameter triggers a warning."""
    import logging

    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    # Mock no existing pointer
    mock_result = Mock()
    mock_result.__len__ = Mock(return_value=0)
    mock_table.search.return_value.where.return_value.to_arrow.return_value = (
        mock_result
    )

    store = LanceDBMainPointerStore()

    # Set pointer with user_id (should log warning)
    with caplog.at_level(logging.WARNING):
        store.set_main_pointer(
            collection="test_collection",
            doc_id="test_doc",
            step_type="parse",
            semantic_id="parse-123",
            technical_id="hash-456",
            user_id=1,
        )

    # Verify warning was logged
    assert any(
        "user_id parameter provided" in record.message
        for record in caplog.records
        if record.levelname == "WARNING"
    )


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_main_pointer_store_list(mock_get_connection: Mock) -> None:
    """Test listing main pointers."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    # Mock count_rows > 0
    mock_table.search.return_value.where.return_value.count_rows.return_value = 1

    # Mock result
    mock_row_data = {
        "collection": "test_collection",
        "doc_id": "test_doc",
        "step_type": "parse",
        "model_tag": "",
        "semantic_id": "parse-123",
        "technical_id": "hash-456",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "operator": "unknown",
    }

    mock_table.search.return_value.where.return_value.limit.return_value.to_arrow.return_value = create_mock_arrow_table(
        [mock_row_data]
    )

    store = LanceDBMainPointerStore()

    pointers = store.list_main_pointers("test_collection")
    assert len(pointers) == 1
    assert pointers[0]["semantic_id"] == "parse-123"


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_main_pointer_store_delete(mock_get_connection: Mock) -> None:
    """Test deleting a main pointer."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    # Mock existing pointer
    mock_result = Mock()
    mock_result.__len__ = Mock(return_value=1)
    mock_table.search.return_value.where.return_value.to_arrow.return_value = (
        mock_result
    )

    store = LanceDBMainPointerStore()

    result = store.delete_main_pointer("test_collection", "test_doc", "parse")
    assert result is True
    mock_table.delete.assert_called_once()


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_main_pointer_store_delete_not_found(mock_get_connection: Mock) -> None:
    """Test deleting a non-existent pointer returns False."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    # Mock no existing pointer
    mock_result = Mock()
    mock_result.__len__ = Mock(return_value=0)
    mock_table.search.return_value.where.return_value.to_arrow.return_value = (
        mock_result
    )

    store = LanceDBMainPointerStore()

    result = store.delete_main_pointer("test_collection", "test_doc", "parse")
    assert result is False
    mock_table.delete.assert_not_called()


# =============================================================================
# Async Method Tests (Phase 1A Coverage Improvement)
# =============================================================================


@pytest.mark.asyncio
@patch("lancedb.connect_async", new_callable=AsyncMock)
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
async def test_search_vectors_async_basic(
    mock_get_connection: Mock, mock_connect_async: AsyncMock
) -> None:
    """Test basic async vector search."""
    import pyarrow as pa

    mock_conn = Mock()
    mock_conn.uri = "test_uri"
    mock_get_connection.return_value = mock_conn

    # Mock async connection
    mock_async_conn = Mock()
    mock_connect_async.return_value = mock_async_conn

    # Mock Arrow table with results
    data = {
        "doc_id": ["doc1", "doc2"],
        "score": [0.95, 0.87],
        "vector": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    }
    arrow_table = pa.Table.from_pydict(data)

    # Mock table and vector search
    mock_table = Mock()
    mock_table.schema = Mock(names=[])
    mock_async_conn.open_table = AsyncMock(return_value=mock_table)

    # Mock vector search - chain needs to return mock objects
    mock_search = Mock()
    mock_search.limit.return_value = mock_search
    mock_search.where = Mock(return_value=mock_search)

    # to_arrow needs to be a coroutine that returns the arrow table
    async def mock_to_arrow():
        return arrow_table

    mock_search.to_arrow = mock_to_arrow

    mock_table.search = Mock(return_value=mock_search)

    store = LanceDBVectorIndexStore()

    # Create a query vector
    query_vector = [0.1, 0.2, 0.3]

    results = await store.search_vectors_async(
        table_name="embeddings_test",
        query_vector=query_vector,
        top_k=5,
    )

    assert len(results) == 2
    assert results[0]["doc_id"] == "doc1"
    assert results[0]["score"] == 0.95


@pytest.mark.asyncio
@patch("lancedb.connect_async", new_callable=AsyncMock)
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
async def test_search_fts_async_basic(
    mock_get_connection: Mock, mock_connect_async: AsyncMock
) -> None:
    """Test basic async FTS search."""
    import pyarrow as pa

    mock_conn = Mock()
    mock_conn.uri = "test_uri"
    mock_get_connection.return_value = mock_conn

    # Mock async connection
    mock_async_conn = Mock()
    mock_connect_async.return_value = mock_async_conn

    # Mock Arrow table with FTS results
    data = {
        "doc_id": ["doc1", "doc2"],
        "text": ["hello world", "test content"],
        "score": [0.9, 0.8],
    }
    arrow_table = pa.Table.from_pydict(data)

    # Mock table and FTS search
    mock_table = Mock()
    mock_table.schema = Mock(names=[])
    mock_async_conn.open_table = AsyncMock(return_value=mock_table)

    # Mock search to return our table
    mock_search = Mock()
    mock_search.limit.return_value = mock_search
    mock_search.where = Mock(return_value=mock_search)

    async def mock_to_arrow():
        return arrow_table

    mock_search.to_arrow = mock_to_arrow

    mock_table.search = Mock(return_value=mock_search)

    store = LanceDBVectorIndexStore()

    results = await store.search_fts_async(
        table_name="chunks",
        query_text="hello",
        top_k=5,
    )

    assert len(results) == 2
    assert results[0]["doc_id"] == "doc1"


@pytest.mark.asyncio
@patch("lancedb.connect_async", new_callable=AsyncMock)
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
async def test_iter_batches_async_basic(
    mock_get_connection: Mock, mock_connect_async: AsyncMock
) -> None:
    """Test async batch iteration."""
    import pyarrow as pa

    mock_conn = Mock()
    mock_conn.uri = "test_uri"
    mock_get_connection.return_value = mock_conn

    # Mock async connection
    mock_async_conn = Mock()
    mock_connect_async.return_value = mock_async_conn

    # Mock table and to_batches
    mock_table = Mock()
    mock_async_conn.open_table = AsyncMock(return_value=mock_table)

    # Create mock batches
    batch1_schema = pa.schema([("doc_id", pa.string()), ("text", pa.string())])
    batch1_data = {"doc_id": ["doc1"], "text": ["text1"]}
    batch1 = pa.RecordBatch.from_pydict(batch1_data, schema=batch1_schema)

    # Mock to_batches as async generator
    async def mock_to_batches(**kwargs):
        yield batch1

    mock_table.to_batches = mock_to_batches

    store = LanceDBVectorIndexStore()

    batches = []
    async for batch in store.iter_batches_async(
        table_name="chunks",
        batch_size=100,
    ):
        batches.append(batch)

    assert len(batches) == 1
    assert batches[0].num_rows == 1


@pytest.mark.asyncio
@patch("lancedb.connect_async", new_callable=AsyncMock)
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
async def test_count_rows_async_basic(
    mock_get_connection: Mock, mock_connect_async: AsyncMock
) -> None:
    """Test async row counting."""
    mock_conn = Mock()
    mock_conn.uri = "test_uri"
    mock_get_connection.return_value = mock_conn

    # Mock async connection
    mock_async_conn = Mock()
    mock_connect_async.return_value = mock_async_conn

    # Mock table and count_rows
    mock_table = Mock()
    mock_async_conn.open_table = AsyncMock(return_value=mock_table)
    mock_table.count_rows = AsyncMock(return_value=100)

    store = LanceDBVectorIndexStore()

    count = await store.count_rows_async(table_name="chunks")

    assert count == 100
    mock_table.count_rows.assert_awaited_once()


@pytest.mark.asyncio
@patch("lancedb.connect_async", new_callable=AsyncMock)
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
async def test_upsert_documents_async(
    mock_get_connection: Mock, mock_connect_async: AsyncMock
) -> None:
    """Test async document upsert."""
    mock_conn = Mock()
    mock_conn.uri = "test_uri"
    mock_get_connection.return_value = mock_conn

    # Mock sync connection for ensure_documents_table
    mock_conn.open_table.return_value = Mock()

    # Mock async connection
    mock_async_conn = Mock()
    mock_connect_async.return_value = mock_async_conn

    # Mock table and merge_insert
    mock_table = Mock()
    mock_async_conn.open_table = AsyncMock(return_value=mock_table)

    # Mock merge_insert chain
    mock_merge_builder = Mock()
    mock_merge_builder.when_matched_update_all = Mock(return_value=mock_merge_builder)
    mock_merge_builder.when_not_matched_insert_all = Mock(
        return_value=mock_merge_builder
    )

    async def mock_execute(records):
        return None

    mock_merge_builder.execute = mock_execute

    mock_table.merge_insert = Mock(return_value=mock_merge_builder)

    store = LanceDBVectorIndexStore()

    records = [
        {"doc_id": "doc1", "source_path": "/tmp/test.pdf"},
        {"doc_id": "doc2", "source_path": "/tmp/test2.pdf"},
    ]

    await store.upsert_documents_async(records)

    # Verify merge_insert was called
    mock_table.merge_insert.assert_called_once()


@pytest.mark.asyncio
@patch("lancedb.connect_async", new_callable=AsyncMock)
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
async def test_upsert_chunks_async(
    mock_get_connection: Mock, mock_connect_async: AsyncMock
) -> None:
    """Test async chunk upsert."""
    mock_conn = Mock()
    mock_conn.uri = "test_uri"
    mock_get_connection.return_value = mock_conn
    mock_conn.list_tables.return_value = []

    # Mock sync connection for ensure_chunks_table
    sync_table = Mock()
    sync_table.schema = Mock(names=["collection", "doc_id", "chunk_id"])
    mock_conn.open_table.return_value = sync_table

    # Mock async connection
    mock_async_conn = Mock()
    mock_connect_async.return_value = mock_async_conn

    # Mock table and merge_insert
    mock_table = Mock()
    mock_async_conn.open_table = AsyncMock(return_value=mock_table)

    # Mock merge_insert chain
    mock_merge_builder = Mock()
    mock_merge_builder.when_matched_update_all = Mock(return_value=mock_merge_builder)
    mock_merge_builder.when_not_matched_insert_all = Mock(
        return_value=mock_merge_builder
    )

    async def mock_execute(records):
        return None

    mock_merge_builder.execute = mock_execute

    mock_table.merge_insert = Mock(return_value=mock_merge_builder)

    store = LanceDBVectorIndexStore()

    records = [
        {"chunk_id": "chunk1", "text": "test content 1"},
        {"chunk_id": "chunk2", "text": "test content 2"},
    ]

    await store.upsert_chunks_async(records)

    # Verify merge_insert was called
    mock_table.merge_insert.assert_called_once()


@pytest.mark.asyncio
@patch("lancedb.connect_async", new_callable=AsyncMock)
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
async def test_upsert_embeddings_async(
    mock_get_connection: Mock, mock_connect_async: AsyncMock
) -> None:
    """Test async embedding upsert."""
    mock_conn = Mock()
    mock_conn.uri = "test_uri"
    mock_get_connection.return_value = mock_conn
    mock_conn.list_tables.return_value = []

    # Mock sync connection for ensure_embeddings_table
    sync_table = Mock()
    sync_table.schema = Mock(names=["collection", "doc_id", "chunk_id", "vector"])
    mock_conn.open_table.return_value = sync_table

    # Mock async connection
    mock_async_conn = Mock()
    mock_connect_async.return_value = mock_async_conn

    # Mock table and merge_insert
    mock_table = Mock()
    mock_async_conn.open_table = AsyncMock(return_value=mock_table)

    # Mock merge_insert chain
    mock_merge_builder = Mock()
    mock_merge_builder.when_matched_update_all = Mock(return_value=mock_merge_builder)
    mock_merge_builder.when_not_matched_insert_all = Mock(
        return_value=mock_merge_builder
    )

    async def mock_execute(records):
        return None

    mock_merge_builder.execute = mock_execute

    mock_table.merge_insert = Mock(return_value=mock_merge_builder)

    store = LanceDBVectorIndexStore()

    records = [
        {"chunk_id": "chunk1", "vector": [0.1, 0.2, 0.3]},
        {"chunk_id": "chunk2", "vector": [0.4, 0.5, 0.6]},
    ]

    await store.upsert_embeddings_async("bge_large", records)

    # Verify merge_insert was called
    mock_table.merge_insert.assert_called_once()


# ============================================================================
# Core Sync Upsert Method Tests
# ============================================================================


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_upsert_documents_basic(mock_get_connection: Mock) -> None:
    """Test basic document upsert."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    # Mock table and merge_insert
    mock_table = Mock()
    mock_table.schema = Mock(names=[])
    mock_conn.open_table.return_value = mock_table

    mock_merge = Mock()
    mock_merge.when_matched_update_all = Mock(return_value=mock_merge)
    mock_merge.when_not_matched_insert_all = Mock(return_value=mock_merge)
    mock_merge.execute = Mock(return_value=None)
    mock_table.merge_insert = Mock(return_value=mock_merge)

    store = LanceDBVectorIndexStore()

    records = [
        {"doc_id": "doc1", "source_path": "/tmp/test.pdf"},
        {"doc_id": "doc2", "source_path": "/tmp/test2.pdf"},
    ]

    store.upsert_documents(records)

    # Verify merge_insert was called with correct keys
    mock_table.merge_insert.assert_called_once_with(["collection", "doc_id"])
    mock_merge.execute.assert_called_once_with(records)


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_upsert_documents_empty(mock_get_connection: Mock) -> None:
    """Test document upsert with empty records returns early."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    store = LanceDBVectorIndexStore()

    # Should return early without opening table
    store.upsert_documents([])

    # Verify table was never opened
    mock_conn.open_table.assert_not_called()


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_upsert_parses_basic(mock_get_connection: Mock) -> None:
    """Test basic parse upsert."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    # Mock table and merge_insert
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    mock_merge = Mock()
    mock_merge.when_matched_update_all = Mock(return_value=mock_merge)
    mock_merge.when_not_matched_insert_all = Mock(return_value=mock_merge)
    mock_merge.execute = Mock(return_value=None)
    mock_table.merge_insert = Mock(return_value=mock_merge)

    store = LanceDBVectorIndexStore()

    records = [
        {"doc_id": "doc1", "parse_hash": "hash1", "parse_status": "success"},
        {"doc_id": "doc2", "parse_hash": "hash2", "parse_status": "success"},
    ]

    store.upsert_parses(records)

    # Verify merge_insert was called with correct keys
    mock_table.merge_insert.assert_called_once_with(
        ["collection", "doc_id", "parse_hash"]
    )
    mock_merge.execute.assert_called_once_with(records)


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_upsert_chunks_basic(mock_get_connection: Mock) -> None:
    """Test basic chunk upsert."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_conn.list_tables.return_value = []

    # Mock table and merge_insert
    mock_table = Mock()
    mock_table.schema = Mock(names=["collection", "doc_id", "chunk_id"])
    mock_conn.open_table.return_value = mock_table

    mock_merge = Mock()
    mock_merge.when_matched_update_all = Mock(return_value=mock_merge)
    mock_merge.when_not_matched_insert_all = Mock(return_value=mock_merge)
    mock_merge.execute = Mock(return_value=None)
    mock_table.merge_insert = Mock(return_value=mock_merge)

    store = LanceDBVectorIndexStore()

    records = [
        {
            "chunk_id": "chunk1",
            "doc_id": "doc1",
            "parse_hash": "hash1",
            "text": "test content 1",
        },
        {
            "chunk_id": "chunk2",
            "doc_id": "doc1",
            "parse_hash": "hash1",
            "text": "test content 2",
        },
    ]

    store.upsert_chunks(records)

    # Verify merge_insert was called with correct keys
    mock_table.merge_insert.assert_called_once_with(
        ["collection", "doc_id", "parse_hash", "chunk_id"]
    )
    mock_merge.execute.assert_called_once_with(records)


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.asyncio
@patch("lancedb.connect_async", new_callable=AsyncMock)
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
async def test_search_vectors_async_table_not_found(
    mock_get_connection: Mock, mock_connect_async: AsyncMock
) -> None:
    """Test async vector search handles missing table gracefully."""
    mock_conn = Mock()
    mock_conn.uri = "test_uri"
    mock_get_connection.return_value = mock_conn

    # Mock async connection
    mock_async_conn = Mock()
    mock_connect_async.return_value = mock_async_conn

    # Mock open_table to raise exception
    mock_async_conn.open_table = AsyncMock(side_effect=Exception("Table not found"))

    store = LanceDBVectorIndexStore()

    query_vector = [0.1, 0.2, 0.3]
    results = await store.search_vectors_async(
        table_name="nonexistent_table",
        query_vector=query_vector,
        top_k=5,
    )

    # Should return empty list on error
    assert results == []


@pytest.mark.asyncio
@patch("lancedb.connect_async", new_callable=AsyncMock)
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
async def test_search_vectors_async_search_failure(
    mock_get_connection: Mock, mock_connect_async: AsyncMock
) -> None:
    """Test async vector search handles search failure gracefully."""

    mock_conn = Mock()
    mock_conn.uri = "test_uri"
    mock_get_connection.return_value = mock_conn

    # Mock async connection
    mock_async_conn = Mock()
    mock_connect_async.return_value = mock_async_conn

    # Mock table
    mock_table = Mock()
    mock_async_conn.open_table = AsyncMock(return_value=mock_table)

    # Mock search that fails
    mock_search = Mock()
    mock_search.limit.return_value = mock_search
    mock_search.where = Mock(return_value=mock_search)

    async def mock_to_arrow():
        raise Exception("Search failed")

    mock_search.to_arrow = mock_to_arrow

    mock_table.search = Mock(return_value=mock_search)

    store = LanceDBVectorIndexStore()

    query_vector = [0.1, 0.2, 0.3]
    results = await store.search_vectors_async(
        table_name="embeddings_test",
        query_vector=query_vector,
        top_k=5,
    )

    # Should return empty list on search error
    assert results == []


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_upsert_documents_with_invalid_data(mock_get_connection: Mock) -> None:
    """Test document upsert handles invalid data gracefully."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    # Mock table and merge_insert that raises exception
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    mock_merge = Mock()
    mock_merge.when_matched_update_all = Mock(return_value=mock_merge)
    mock_merge.when_not_matched_insert_all = Mock(return_value=mock_merge)
    mock_merge.execute = Mock(side_effect=Exception("Invalid data"))
    mock_table.merge_insert = Mock(return_value=mock_merge)

    store = LanceDBVectorIndexStore()

    records = [{"doc_id": "doc1", "invalid_field": "value"}]

    # Should raise exception on invalid data
    with pytest.raises(Exception, match="Invalid data"):
        store.upsert_documents(records)


@pytest.mark.asyncio
@patch("lancedb.connect_async", new_callable=AsyncMock)
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
async def test_iter_batches_async_invalid_columns(
    mock_get_connection: Mock, mock_connect_async: AsyncMock
) -> None:
    """Test async iter_batches handles invalid columns gracefully."""
    mock_conn = Mock()
    mock_conn.uri = "test_uri"
    mock_get_connection.return_value = mock_conn

    # Mock async connection
    mock_async_conn = Mock()
    mock_connect_async.return_value = mock_async_conn

    # Mock table
    mock_table = Mock()
    mock_async_conn.open_table = AsyncMock(return_value=mock_table)

    # Mock to_batches generator that raises exception
    async def mock_to_batches(**kwargs):
        raise Exception("Invalid columns")

    # Make to_batches return an async generator that raises
    def make_to_batches():
        async def inner(**kwargs):
            raise Exception("Invalid columns")

        return inner()

    mock_table.to_batches = make_to_batches()

    store = LanceDBVectorIndexStore()

    # Should handle exception gracefully and not yield any batches
    batches = []
    async for batch in store.iter_batches_async(
        table_name="chunks",
        batch_size=100,
        columns=["nonexistent_column"],
    ):
        batches.append(batch)

    # Should get no batches due to error
    assert len(batches) == 0


@pytest.mark.asyncio
@patch("lancedb.connect_async", new_callable=AsyncMock)
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
async def test_count_rows_async_table_not_found(
    mock_get_connection: Mock, mock_connect_async: AsyncMock
) -> None:
    """Test async count_rows handles missing table gracefully."""
    mock_conn = Mock()
    mock_conn.uri = "test_uri"
    mock_get_connection.return_value = mock_conn

    # Mock async connection
    mock_async_conn = Mock()
    mock_connect_async.return_value = mock_async_conn

    # Mock open_table to raise exception
    mock_async_conn.open_table = AsyncMock(side_effect=Exception("Table not found"))

    store = LanceDBVectorIndexStore()

    count = await store.count_rows_async(table_name="nonexistent_table")

    # Should return 0 on error
    assert count == 0


# --- get_vector_dimension Tests (Issue #14) ---


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_get_vector_dimension_success(mock_get_connection: Mock) -> None:
    """Test get_vector_dimension returns correct dimension from schema."""
    from types import SimpleNamespace

    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    # Mock table with fixed-size vector field
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    # Mock schema with vector field having list_size
    mock_vector_type = SimpleNamespace(list_size=1536)
    mock_vector_field = SimpleNamespace(type=mock_vector_type)
    mock_schema = Mock()
    mock_schema.field.return_value = mock_vector_field
    mock_table.schema = mock_schema

    store = LanceDBVectorIndexStore()
    dimension = store.get_vector_dimension("embeddings_test_model")

    assert dimension == 1536
    mock_schema.field.assert_called_once_with("vector")


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_get_vector_dimension_table_not_found(mock_get_connection: Mock) -> None:
    """Test get_vector_dimension returns None when table not found."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    # Mock open_table to raise exception
    mock_conn.open_table.side_effect = Exception("Table not found")

    store = LanceDBVectorIndexStore()
    dimension = store.get_vector_dimension("nonexistent_table")

    assert dimension is None


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_get_vector_dimension_variable_length(mock_get_connection: Mock) -> None:
    """Test get_vector_dimension returns None for variable-length vectors."""
    from types import SimpleNamespace

    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    # Mock table with variable-length vector field (no list_size)
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    # Mock schema with vector field lacking list_size attribute
    mock_vector_type = SimpleNamespace()  # No list_size
    mock_vector_field = SimpleNamespace(type=mock_vector_type)
    mock_schema = Mock()
    mock_schema.field.return_value = mock_vector_field
    mock_table.schema = mock_schema

    store = LanceDBVectorIndexStore()
    dimension = store.get_vector_dimension("embeddings_variable")

    assert dimension is None


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_get_vector_dimension_no_vector_field(mock_get_connection: Mock) -> None:
    """Test get_vector_dimension returns None when vector field missing."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    # Mock table without vector field
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    mock_schema = Mock()
    mock_schema.field.side_effect = Exception("Field 'vector' not found")
    mock_table.schema = mock_schema

    store = LanceDBVectorIndexStore()
    dimension = store.get_vector_dimension("embeddings_no_vector")

    assert dimension is None


# --- list_table_names Tests (Issue #14) ---


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_list_table_names_success(mock_get_connection: Mock) -> None:
    """Test list_table_names returns correct table names."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    # New compatibility path prefers list_tables.
    mock_conn.list_tables.return_value = ["documents", "chunks", "embeddings_test"]

    store = LanceDBVectorIndexStore()
    names = store.list_table_names()

    assert names == ["documents", "chunks", "embeddings_test"]
    mock_conn.list_tables.assert_called_once()


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_list_table_names_connection_error(mock_get_connection: Mock) -> None:
    """Test list_table_names returns empty list on error."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    # Mock table_names to raise exception
    mock_conn.table_names.side_effect = Exception("Connection error")

    store = LanceDBVectorIndexStore()
    names = store.list_table_names()

    assert names == []


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_list_table_names_no_table_names_attr(mock_get_connection: Mock) -> None:
    """Test list_table_names returns empty list when connection lacks table_names."""
    # Mock connection without table_names attribute
    mock_conn = Mock(spec=[])  # Empty spec means no attributes
    mock_get_connection.return_value = mock_conn

    store = LanceDBVectorIndexStore()
    names = store.list_table_names()

    assert names == []


# --- get_vector_dimension_async Tests (Issue #14) ---


@pytest.mark.asyncio
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
async def test_get_vector_dimension_async_delegates_to_sync(
    mock_get_connection: Mock,
) -> None:
    """Test async version delegates to sync implementation."""
    from types import SimpleNamespace

    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    # Mock table with fixed-size vector field
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    mock_vector_type = SimpleNamespace(list_size=768)
    mock_vector_field = SimpleNamespace(type=mock_vector_type)
    mock_schema = Mock()
    mock_schema.field.return_value = mock_vector_field
    mock_table.schema = mock_schema

    store = LanceDBVectorIndexStore()
    dimension = await store.get_vector_dimension_async("embeddings_async_test")

    assert dimension == 768


# --- _get_table cache Tests ---


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_get_table_cache_miss_calls_open_table(mock_get_connection: Mock) -> None:
    """_get_table should call open_table on cache miss."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    store = LanceDBVectorIndexStore()
    table = store._get_table("documents")

    assert table is mock_table
    mock_conn.open_table.assert_called_once_with("documents")


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_get_table_cache_hit_skips_open_table(mock_get_connection: Mock) -> None:
    """_get_table should not call open_table on cache hit."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    store = LanceDBVectorIndexStore()
    first = store._get_table("documents")
    second = store._get_table("documents")

    assert first is second is mock_table
    mock_conn.open_table.assert_called_once()


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_get_table_cache_multiple_tables(mock_get_connection: Mock) -> None:
    """_get_table should cache multiple different tables independently."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_docs = Mock()
    mock_parses = Mock()
    mock_conn.open_table.side_effect = [mock_docs, mock_parses]

    store = LanceDBVectorIndexStore()
    docs = store._get_table("documents")
    parses = store._get_table("parses")

    assert docs is mock_docs
    assert parses is mock_parses
    assert mock_conn.open_table.call_count == 2
    assert store._get_table("documents") is mock_docs  # still cached
    assert mock_conn.open_table.call_count == 2  # no additional call


# --- invalidate_table_cache Tests ---


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_invalidate_cache_all_clears_everything(
    mock_get_connection: Mock,
) -> None:
    """invalidate_table_cache() should clear the entire cache when no arg given."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_conn.open_table.side_effect = [Mock(), Mock(), Mock()]

    store = LanceDBVectorIndexStore()
    store._get_table("documents")
    store._get_table("parses")
    assert len(store._table_cache) == 2

    store.invalidate_table_cache()
    assert len(store._table_cache) == 0

    # Subsequent access re-opens
    store._get_table("documents")
    assert mock_conn.open_table.call_count == 3  # 2 initial + 1 re-open


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_invalidate_cache_by_name_only_removes_one(
    mock_get_connection: Mock,
) -> None:
    """invalidate_table_cache('name') should only remove that entry."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_conn.open_table.side_effect = [Mock(), Mock(), Mock()]

    store = LanceDBVectorIndexStore()
    store._get_table("documents")
    store._get_table("parses")
    assert len(store._table_cache) == 2

    store.invalidate_table_cache("documents")
    assert len(store._table_cache) == 1
    assert "documents" not in store._table_cache
    assert "parses" in store._table_cache

    # Re-access evicted table
    store._get_table("documents")
    assert mock_conn.open_table.call_count == 3


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_invalidate_cache_unknown_name_noop(mock_get_connection: Mock) -> None:
    """invalidate_table_cache('unknown') should not raise or affect cache."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_conn.open_table.return_value = Mock()

    store = LanceDBVectorIndexStore()
    store._get_table("documents")
    assert len(store._table_cache) == 1

    store.invalidate_table_cache("nonexistent")
    assert len(store._table_cache) == 1


# --- LRU Eviction Tests ---


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_lru_eviction_at_maxsize(mock_get_connection: Mock) -> None:
    """Cache should evict oldest entry when exceeding _TABLE_CACHE_MAXSIZE (64)."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_conn.open_table.return_value = Mock()

    store = LanceDBVectorIndexStore()
    maxsize = store._TABLE_CACHE_MAXSIZE

    # Fill cache to exactly maxsize
    for i in range(maxsize):
        store._get_table(f"table_{i}")
    assert len(store._table_cache) == maxsize

    # Insert one more — oldest should be evicted
    store._get_table("overflow_table")
    assert len(store._table_cache) == maxsize
    assert "table_0" not in store._table_cache
    assert "overflow_table" in store._table_cache


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_lru_access_refreshes_position(mock_get_connection: Mock) -> None:
    """Accessing a cached table should move it to the end (most-recently-used)."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_conn.open_table.return_value = Mock()

    store = LanceDBVectorIndexStore()
    maxsize = store._TABLE_CACHE_MAXSIZE

    # Fill cache, with table_0 first
    for i in range(maxsize):
        store._get_table(f"table_{i}")

    # Access table_0 — should move to MRU end
    store._get_table("table_0")

    # Insert one more — table_1 (now the oldest) should be evicted, not table_0
    store._get_table("overflow_table")
    assert len(store._table_cache) == maxsize
    assert "table_0" in store._table_cache  # still alive (was refreshed)
    assert "table_1" not in store._table_cache  # became oldest and evicted


# --- _count_collections_fast / aggregate_collection_stats Tests ---


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_aggregate_collection_stats_basic(mock_get_connection: Mock) -> None:
    """aggregate_collection_stats should return per-collection counts."""
    import pyarrow as pa

    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    # Build Arrow tables for documents, parses, chunks
    docs_tbl = pa.table({"collection": ["col_a", "col_a", "col_b"]})
    parses_tbl = pa.table({"collection": ["col_a"]})
    chunks_tbl = pa.table({"collection": ["col_a", "col_a", "col_b", "col_b"]})

    # Mock open_table to return different Arrow tables based on table name
    def mock_search(table_name):
        mock_result = Mock()
        # Set up search().where().select().limit().to_arrow() chain
        mock_chain = Mock()
        if table_name == "documents":
            mock_chain.to_arrow.return_value = docs_tbl
        elif table_name == "parses":
            mock_chain.to_arrow.return_value = parses_tbl
        elif table_name == "chunks":
            mock_chain.to_arrow.return_value = chunks_tbl
        else:
            mock_chain.to_arrow.return_value = pa.table({"collection": []})
        mock_result.search.return_value = mock_chain
        return mock_result

    mock_table = Mock()
    mock_table.search = Mock()
    mock_conn.open_table.return_value = mock_table

    # Patch the search().where().select().limit().to_arrow() chain for each table
    def build_chains(table_name):
        if table_name == "documents":
            tbl = docs_tbl
        elif table_name == "parses":
            tbl = parses_tbl
        elif table_name == "chunks":
            tbl = chunks_tbl
        else:
            tbl = pa.table({"collection": []})
        chain = Mock()
        chain.select.return_value = chain
        chain.where.return_value = chain
        chain.limit.return_value = chain
        chain.to_arrow.return_value = tbl
        return chain

    chains = {}
    for name in ["documents", "parses", "chunks"]:
        chains[name] = build_chains(name)

    # The table is cached; the _get_table returns the mock_table,
    # and the code calls mock_table.search() to start the chain
    mock_table.search.side_effect = (
        lambda: chains.get(
            # Figure out which table name from the cache — use a side effect approach
            # Since _get_table caches by name, we need a smarter mock
        )
    )

    # Update: simplify — just use MagicMock with per-table chains
    mock_conn.open_table.side_effect = lambda name: _make_mock_for(name)

    def _make_mock_for(name):
        t = Mock()
        t.search.return_value = chains[name]
        return t

    store = LanceDBVectorIndexStore()
    # Prime cache with mock tables
    for name in ["documents", "parses", "chunks"]:
        store._table_cache[name] = _make_mock_for(name)

    # Mock list_table_names to return no extra embeddings tables
    store.list_table_names = Mock(return_value=[])

    stats = store.aggregate_collection_stats(user_id=None, is_admin=True)

    assert "col_a" in stats
    assert "col_b" in stats
    assert stats["col_a"]["documents"] == 2
    assert stats["col_a"]["parses"] == 1
    assert stats["col_a"]["chunks"] == 2
    assert stats["col_b"]["documents"] == 1
    assert stats["col_b"]["parses"] == 0
    assert stats["col_b"]["chunks"] == 2


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_count_collections_fast_with_user_filter(mock_get_connection: Mock) -> None:
    """_count_collections_fast should apply user_id filter when not admin."""
    import pyarrow as pa

    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    tbl = pa.table(
        {
            "collection": ["col_a", "col_a", "col_b"],
            "user_id": [1, 2, 3],
        }
    )

    chain = Mock()
    chain.select.return_value = chain
    chain.where.return_value = chain
    chain.limit.return_value = chain
    chain.to_arrow.return_value = tbl

    mock_table = Mock()
    mock_table.search.return_value = chain
    mock_conn.open_table.return_value = mock_table

    store = LanceDBVectorIndexStore()
    store._table_cache["documents"] = mock_table

    stats: dict = {}
    store._count_collections_fast(
        "documents", "documents", stats, user_id=1, is_admin=False
    )

    # Verify that the where filter was applied
    chain.where.assert_called_once()


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_count_collections_fast_admin_no_filter(mock_get_connection: Mock) -> None:
    """_count_collections_fast should not apply user filter when is_admin=True."""
    import pyarrow as pa

    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    tbl = pa.table(
        {
            "collection": ["col_a", "col_a", "col_b"],
            "user_id": [1, 2, 3],
        }
    )

    chain = Mock()
    chain.select.return_value = chain
    chain.where.return_value = chain
    chain.limit.return_value = chain
    chain.to_arrow.return_value = tbl

    mock_table = Mock()
    mock_table.search.return_value = chain
    mock_conn.open_table.return_value = mock_table

    store = LanceDBVectorIndexStore()
    store._table_cache["documents"] = mock_table

    stats: dict = {}
    store._count_collections_fast(
        "documents", "documents", stats, user_id=None, is_admin=True
    )

    # For admin, the where filter should NOT be called (get_user_filter returns "")
    chain.where.assert_not_called()


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_count_collections_fast_empty_table(mock_get_connection: Mock) -> None:
    """_count_collections_fast should handle empty Arrow table gracefully."""
    import pyarrow as pa

    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    empty_tbl = pa.table({"collection": pa.array([], type=pa.string())})

    chain = Mock()
    chain.select.return_value = chain
    chain.where.return_value = chain
    chain.limit.return_value = chain
    chain.to_arrow.return_value = empty_tbl

    mock_table = Mock()
    mock_table.search.return_value = chain
    mock_conn.open_table.return_value = mock_table

    store = LanceDBVectorIndexStore()
    store._table_cache["documents"] = mock_table

    stats: dict = {}
    store._count_collections_fast(
        "documents", "documents", stats, user_id=None, is_admin=True
    )

    # Empty table should produce no stats entries
    assert stats == {}


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_count_collections_fast_error_graceful(mock_get_connection: Mock) -> None:
    """_count_collections_fast should not raise on error, just log debug."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    mock_table = Mock()
    mock_table.search.side_effect = Exception("LanceDB read error")
    mock_conn.open_table.return_value = mock_table

    store = LanceDBVectorIndexStore()
    store._table_cache["documents"] = mock_table

    stats: dict = {}
    # Should not raise
    store._count_collections_fast(
        "documents", "documents", stats, user_id=None, is_admin=True
    )
    assert stats == {}
