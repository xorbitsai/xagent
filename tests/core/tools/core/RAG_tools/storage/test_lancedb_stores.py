"""Tests for LanceDB-backed storage implementations."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import lancedb
import pandas as pd
import pyarrow as pa
import pytest

from xagent.core.tools.core.RAG_tools.core.config import DEFAULT_INDEX_POLICY
from xagent.core.tools.core.RAG_tools.core.exceptions import DatabaseOperationError
from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    _evaluate_filter_expression,
)
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    ensure_embeddings_table,
)
from xagent.core.tools.core.RAG_tools.storage.contracts import (
    FilterCondition,
    FilterExpression,
    FilterOperator,
)
from xagent.core.tools.core.RAG_tools.storage.factory import StorageFactory
from xagent.core.tools.core.RAG_tools.storage.lancedb_filter_utils import (
    translate_condition,
    translate_filter_expression,
)
from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
    LanceDBIngestionStatusStore,
    LanceDBMainPointerStore,
    LanceDBMetadataStore,
    LanceDBPromptTemplateStore,
    LanceDBVectorIndexStore,
    _stale_version_count,
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
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.LanceDBMetadataStore.ensure_collection_metadata_table",
    new_callable=AsyncMock,
)
@patch(
    "xagent.core.tools.core.RAG_tools.LanceDB.schema_manager.ensure_collection_config_table"
)
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_metadata_store_rename_collection_updates_tables(
    mock_get_connection: Mock,
    _mock_ensure_config: Mock,
    _mock_ensure_meta: AsyncMock,
) -> None:
    """rename_collection should update collection_config and collection_metadata."""
    from types import SimpleNamespace

    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    mock_config = Mock()
    mock_config.schema = [SimpleNamespace(name="collection")]
    mock_meta = Mock()
    mock_meta.schema = [SimpleNamespace(name="name")]

    def _open(name: str) -> Mock:
        if name == "collection_config":
            return mock_config
        if name == "collection_metadata":
            return mock_meta
        raise AssertionError(name)

    mock_conn.open_table.side_effect = _open

    store = LanceDBMetadataStore()
    asyncio.run(
        store.rename_collection(
            "old_col",
            "new_col",
            user_id=1,
            is_admin=True,
        )
    )

    mock_config.update.assert_called_once()
    cfg_where, cfg_updates = mock_config.update.call_args[0]
    assert "old_col" in cfg_where
    assert cfg_updates["collection"] == "new_col"

    mock_meta.update.assert_called_once()
    meta_where, meta_updates = mock_meta.update.call_args[0]
    assert "old_col" in meta_where
    assert meta_updates["name"] == "new_col"


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.LanceDBMetadataStore.ensure_collection_metadata_table",
    new_callable=AsyncMock,
)
@patch(
    "xagent.core.tools.core.RAG_tools.LanceDB.schema_manager.ensure_collection_config_table"
)
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_metadata_store_rename_collection_tenant_scoped_config_only(
    mock_get_connection: Mock,
    _mock_ensure_config: Mock,
    _mock_ensure_meta: AsyncMock,
) -> None:
    """Tenant rename should update only the caller's config row."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    mock_config = Mock()
    mock_meta = Mock()

    def _open(name: str) -> Mock:
        if name == "collection_config":
            return mock_config
        if name == "collection_metadata":
            return mock_meta
        raise AssertionError(name)

    mock_conn.open_table.side_effect = _open

    store = LanceDBMetadataStore()
    asyncio.run(
        store.rename_collection(
            "old_col",
            "new_col",
            user_id=42,
            is_admin=False,
        )
    )

    mock_config.update.assert_called_once()
    cfg_where, cfg_updates = mock_config.update.call_args[0]
    assert "old_col" in cfg_where
    assert "user_id = 42" in cfg_where
    assert cfg_updates["collection"] == "new_col"
    mock_meta.delete.assert_called_once()
    assert "old_col" in mock_meta.delete.call_args.args[0]
    mock_meta.update.assert_not_called()


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
def test_metadata_store_get_collection_config_read_error_raises(
    mock_get_connection: Mock,
) -> None:
    """Read errors should not be conflated with a missing collection config."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    mock_table = Mock()
    mock_table.schema = Mock(names=[])
    mock_conn.open_table.return_value = mock_table
    mock_table.search.return_value.where.return_value.to_arrow.side_effect = (
        RuntimeError("read failed")
    )

    store = LanceDBMetadataStore()
    with pytest.raises(RuntimeError, match="read failed"):
        asyncio.run(
            store.get_collection_config(collection="test_collection", user_id=1)
        )


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


@patch("xagent.core.tools.core.RAG_tools.storage.lancedb_stores.query_to_list")
@patch(
    "xagent.core.tools.core.RAG_tools.LanceDB.schema_manager.ensure_collection_config_table"
)
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_metadata_store_list_collection_config_owner_ids(
    mock_get_connection: Mock,
    _mock_ensure_config: Mock,
    mock_query_to_list: Mock,
) -> None:
    """Metadata store should own stale collection_config owner discovery."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table
    mock_query_to_list.return_value = [
        {"user_id": 101},
        {"user_id": "202"},
        {"user_id": None},
        {"user_id": "bad"},
    ]

    store = LanceDBMetadataStore()

    assert store.list_collection_config_owner_ids("FAQ") == {101, 202}
    mock_conn.open_table.assert_called_once_with("collection_config")
    mock_query_to_list.assert_called_once()
    mock_table.search.return_value.where.assert_called_once()


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
        {"doc_id": "doc-1", "source_path": "/tmp/a.pdf", "user_id": 1},
        {"doc_id": "doc-2", "source_path": None, "user_id": 1},
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
    assert records[0].user_id == 1
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
    warnings = store.rename_collection_data(
        "old_name",
        "new_name",
        user_id=None,
        is_admin=True,
    )

    assert warnings == []
    # 4 target tables should be updated; control-plane table excluded.
    assert mock_table.update.call_count == 4


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.UserPermissions.get_user_filter"
)
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_ingestion_status_store_rename_collection_status_is_tenant_scoped(
    mock_get_connection: Mock,
    mock_user_filter: Mock,
) -> None:
    """Non-admin status rename should include both collection and user filters."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table
    mock_user_filter.return_value = "user_id == 101"

    store = LanceDBIngestionStatusStore()
    warnings = store.rename_collection_status(
        old_name="old",
        new_name="new",
        user_id=101,
        is_admin=False,
    )

    assert warnings == []
    where_expr = mock_table.update.call_args.args[0]
    assert "collection == 'old'" in where_expr
    assert "user_id == 101" in where_expr
    assert mock_table.update.call_args.args[1]["collection"] == "new"


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.UserPermissions.get_user_filter"
)
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_ingestion_status_store_rename_collection_status_admin_is_global(
    mock_get_connection: Mock,
    mock_user_filter: Mock,
) -> None:
    """Admin status rename should update every owner for the collection."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table
    mock_user_filter.return_value = None

    store = LanceDBIngestionStatusStore()
    warnings = store.rename_collection_status(
        old_name="old",
        new_name="new",
        user_id=999,
        is_admin=True,
    )

    assert warnings == []
    where_expr = mock_table.update.call_args.args[0]
    assert where_expr == "collection == 'old'"
    assert mock_table.update.call_args.args[1]["collection"] == "new"


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_vector_store_rename_collection_data_tenant_scoped(
    mock_get_connection: Mock,
) -> None:
    """Tenant rename should include the user filter in each table update."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    table_names = ["documents", "parses", "chunks", "embeddings_text_embedding_v4"]
    mock_conn.table_names.return_value = table_names
    mock_conn.list_tables.return_value = table_names
    mock_table = Mock()
    mock_table.schema = Mock(names=[])
    mock_conn.open_table.return_value = mock_table

    store = LanceDBVectorIndexStore()
    warnings = store.rename_collection_data(
        "old_name",
        "new_name",
        user_id=42,
        is_admin=False,
    )

    assert warnings == []
    assert mock_table.update.call_count == 4
    for call_args in mock_table.update.call_args_list:
        where_expr = call_args.args[0]
        assert "old_name" in where_expr
        assert "user_id" in where_expr
        assert "42" in where_expr


@patch.object(LanceDBVectorIndexStore, "cascade_delete")
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_delete_collection_data_delegates_to_store_cascade_delete(
    mock_get_conn: Mock,
    mock_cascade: Mock,
) -> None:
    """delete_collection_data should route through self.cascade_delete."""
    mock_get_conn.return_value = Mock()
    mock_cascade.return_value = {"documents": 1, "parses": 1}
    store = LanceDBVectorIndexStore()
    warnings: List[str] = []

    result = store.delete_collection_data(
        "demo", user_id=1, is_admin=False, warnings_out=warnings
    )

    mock_cascade.assert_called_once()
    kw = mock_cascade.call_args.kwargs
    assert kw["target"] == "collection"
    assert kw["collection"] == "demo"
    assert kw["user_id"] == 1
    assert kw["is_admin"] is False
    assert kw["preview_only"] is False
    assert kw["confirm"] is True
    assert "conn" not in kw  # contract method owns its connection
    assert result == {"documents": 1, "parses": 1}
    assert warnings == []


@patch.object(LanceDBVectorIndexStore, "cascade_delete")
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_delete_document_data_delegates_to_store_cascade_delete(
    mock_get_conn: Mock,
    mock_cascade: Mock,
) -> None:
    """delete_document_data should route through self.cascade_delete."""
    mock_get_conn.return_value = Mock()
    mock_cascade.return_value = {"documents": 1}
    store = LanceDBVectorIndexStore()

    result = store.delete_document_data("demo", "d1", user_id=2, is_admin=True)

    mock_cascade.assert_called_once()
    kw = mock_cascade.call_args.kwargs
    assert kw["target"] == "document"
    assert kw["collection"] == "demo"
    assert kw["doc_id"] == "d1"
    assert kw["user_id"] == 2
    assert kw["is_admin"] is True
    assert kw["preview_only"] is False
    assert kw["confirm"] is True
    assert "conn" not in kw
    assert result == {"documents": 1}


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores._vis_cascade_delete_documents"
)
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_delete_documents_data_uses_vis_batch_driver(
    mock_get_conn: Mock,
    mock_batch: Mock,
) -> None:
    """delete_documents_data should route batches through _vis_cascade_delete_documents."""
    mock_conn = Mock()
    mock_get_conn.return_value = mock_conn
    mock_batch.return_value = {"documents": 2, "chunks": 4}

    store = LanceDBVectorIndexStore()
    warnings: List[str] = []

    result = store.delete_documents_data(
        "demo",
        ["doc-2", "doc-1", "doc-1"],
        user_id=7,
        is_admin=False,
        warnings_out=warnings,
    )

    mock_batch.assert_called_once()
    args, kw = mock_batch.call_args
    assert args[0] is mock_conn  # conn is first positional arg
    assert kw["collection"] == "demo"
    assert kw["doc_ids"] == ["doc-1", "doc-2"]  # normalized + deduped + sorted
    assert kw["user_id"] == 7
    assert kw["is_admin"] is False
    assert kw["preview_only"] is False
    assert kw["confirm"] is True
    assert result == {"documents": 2, "chunks": 4}
    assert warnings == []


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores._vis_cascade_delete_documents"
)
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_delete_documents_data_partial_failure_raises_with_details(
    mock_get_conn: Mock,
    mock_batch: Mock,
) -> None:
    """A later batch failure should preserve prior batch progress in the error details."""
    mock_get_conn.return_value = Mock()
    mock_batch.side_effect = [
        {"documents": 100, "chunks": 200},
        RuntimeError("batch failed"),
    ]

    store = LanceDBVectorIndexStore()
    store.invalidate_table_cache = Mock()  # type: ignore[method-assign]
    warnings: List[str] = []
    doc_ids = [f"doc-{idx:03d}" for idx in range(101)]

    with pytest.raises(DatabaseOperationError) as exc_info:
        store.delete_documents_data(
            "demo",
            doc_ids,
            user_id=7,
            is_admin=False,
            warnings_out=warnings,
        )

    assert mock_batch.call_count == 2
    store.invalidate_table_cache.assert_called_once()
    assert warnings == ["Failed to delete document batch 2: batch failed"]
    assert exc_info.value.details["deleted_counts"] == {"documents": 100, "chunks": 200}
    assert exc_info.value.details["deleted_doc_ids"] == doc_ids[:100]
    assert exc_info.value.details["failed_batch_index"] == 2


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
        filters=FilterCondition(
            field="doc_id", operator=FilterOperator.EQ, value="doc1"
        ),
        user_id=7,
        is_admin=False,
    )

    assert len(results) == 2
    assert results[0]["doc_id"] == "doc1"
    assert results[0]["score"] == 0.95
    filter_expr = mock_search.where.call_args.args[0]
    assert "doc_id == 'doc1'" in filter_expr
    assert "user_id == 7" in filter_expr


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
        filters=FilterCondition(
            field="doc_id", operator=FilterOperator.EQ, value="doc1"
        ),
        user_id=7,
        is_admin=False,
    )

    assert len(results) == 2
    assert results[0]["doc_id"] == "doc1"
    filter_expr = mock_search.where.call_args.args[0]
    assert "doc_id == 'doc1'" in filter_expr
    assert "user_id == 7" in filter_expr


@pytest.mark.asyncio
async def test_async_by_model_wrappers_preserve_user_scope() -> None:
    store = LanceDBVectorIndexStore()
    store.open_embeddings_table = Mock(return_value=(Mock(), "embeddings_model-x"))
    store.search_vectors_async = AsyncMock(return_value=[])
    store.search_fts_async = AsyncMock(return_value=[])

    await store.search_vectors_by_model_async(
        model_tag="model-x",
        query_vector=[0.1],
        top_k=5,
        user_id=7,
        is_admin=False,
    )
    await store.search_fts_by_model_async(
        model_tag="model-x",
        query_text="needle",
        top_k=5,
        user_id=7,
        is_admin=False,
    )

    vector_kwargs = store.search_vectors_async.call_args.kwargs
    fts_kwargs = store.search_fts_async.call_args.kwargs
    assert vector_kwargs["user_id"] == fts_kwargs["user_id"] == 7
    assert vector_kwargs["is_admin"] is fts_kwargs["is_admin"] is False


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

    # Mock table and async query batch reader
    mock_table = Mock()
    mock_async_conn.open_table = AsyncMock(return_value=mock_table)

    # Create mock batches
    batch1_schema = pa.schema([("doc_id", pa.string()), ("text", pa.string())])
    batch1_data = {"doc_id": ["doc1"], "text": ["text1"]}
    batch1 = pa.RecordBatch.from_pydict(batch1_data, schema=batch1_schema)

    mock_query = Mock()
    mock_query.where.return_value = mock_query
    mock_query.select.return_value = mock_query

    async def mock_to_batches(**kwargs):
        return _read_batches()

    async def _read_batches():
        yield batch1

    mock_query.to_batches = mock_to_batches
    mock_table.query = Mock(return_value=mock_query)

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
@pytest.mark.parametrize(
    ("user_id", "is_admin", "expected_rows"),
    [
        (7, False, [{"doc_id": "d1", "text": "one"}]),
        (8, False, [{"doc_id": "d2", "text": "two"}]),
        (
            None,
            True,
            [
                {"doc_id": "d1", "text": "one"},
                {"doc_id": "d2", "text": "two"},
            ],
        ),
        (None, False, []),
    ],
)
async def test_iter_batches_async_uses_locked_lancedb_query_api(
    tmp_path,
    user_id: int | None,
    is_admin: bool,
    expected_rows: list[dict[str, str]],
) -> None:
    """Test async batch iteration against the locked LanceDB API and schema."""
    conn = lancedb.connect(str(tmp_path / "lancedb"))
    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("parse_hash", pa.string()),
            pa.field("text", pa.large_string()),
            pa.field("metadata", pa.string()),
            pa.field("user_id", pa.int64()),
        ]
    )
    table = conn.create_table("chunks", schema=schema)
    table.add(
        [
            {
                "collection": "docs",
                "doc_id": "d1",
                "chunk_id": "c1",
                "parse_hash": "h1",
                "text": "one",
                "metadata": None,
                "user_id": 7,
            },
            {
                "collection": "docs",
                "doc_id": "d2",
                "chunk_id": "c2",
                "parse_hash": "h2",
                "text": "two",
                "metadata": None,
                "user_id": 8,
            },
        ]
    )

    store = LanceDBVectorIndexStore()
    store._conn = conn
    rows = []
    async for batch in store.iter_batches_async(
        table_name="chunks",
        columns=["doc_id", "text"],
        batch_size=1,
        user_id=user_id,
        is_admin=is_admin,
    ):
        rows.extend(batch.to_pylist())

    assert rows == expected_rows


@pytest.mark.parametrize(
    "needle",
    ["100%_real", "file_name", "O'Reilly\\docs"],
)
def test_contains_filter_matches_literals_in_locked_lancedb(
    tmp_path, needle: str
) -> None:
    """Ensure backend CONTAINS uses the same literal semantics as fallback."""
    conn = lancedb.connect(str(tmp_path / "lancedb"))
    table = conn.create_table(
        "rows",
        schema=pa.schema(
            [pa.field("id", pa.string()), pa.field("text", pa.large_string())]
        ),
    )
    table.add(
        [
            {
                "id": "literal",
                "text": "100%_real file_name O'Reilly\\docs",
            },
            {
                "id": "wildcard-lookalike",
                "text": "100XXreal fileXname OXReilly/docs",
            },
        ]
    )
    backend_filter = translate_condition(
        FilterCondition("text", FilterOperator.CONTAINS, needle)
    )

    rows = table.search().where(backend_filter).select(["id"]).to_list()

    assert rows == [{"id": "literal"}]


@pytest.mark.parametrize(
    ("expression", "expected_doc_ids"),
    [
        (FilterCondition("doc_id", FilterOperator.EQ, "d1"), ["d1"]),
        (
            FilterCondition("chunk_hash", FilterOperator.EQ, "h'1\\path"),
            ["d1"],
        ),
        (FilterCondition("doc_id", FilterOperator.NE, "d1"), ["d2", "d3"]),
        (FilterCondition("user_id", FilterOperator.GT, 7), ["d2", "d3"]),
        (FilterCondition("user_id", FilterOperator.GTE, 8), ["d2", "d3"]),
        (FilterCondition("user_id", FilterOperator.LT, 9), ["d1", "d2"]),
        (FilterCondition("user_id", FilterOperator.LTE, 8), ["d1", "d2"]),
        (
            FilterCondition("doc_id", FilterOperator.IN, ["d1", "d3"]),
            ["d1", "d3"],
        ),
        (FilterCondition("text", FilterOperator.CONTAINS, "%_"), ["d1"]),
        (FilterCondition("metadata", FilterOperator.IS_NULL, None), ["d1"]),
        (
            FilterCondition("metadata", FilterOperator.IS_NOT_NULL, None),
            ["d2", "d3"],
        ),
        (
            [
                FilterCondition("doc_id", FilterOperator.EQ, "d1"),
                FilterCondition("doc_id", FilterOperator.EQ, "d3"),
            ],
            ["d1", "d3"],
        ),
    ],
)
def test_filter_expression_backend_and_fallback_parity_on_embeddings_schema(
    tmp_path,
    expression: FilterExpression,
    expected_doc_ids: list[str],
) -> None:
    """Compare backend and fallback results using the persisted embeddings schema."""
    conn = lancedb.connect(str(tmp_path / "lancedb"))
    ensure_embeddings_table(conn, "test", vector_dim=2)
    table = conn.open_table("embeddings_test")
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    records = [
        {
            "collection": "docs",
            "doc_id": "d1",
            "chunk_id": "c1",
            "parse_hash": "p1",
            "model": "test",
            "vector": [0.1, 0.2],
            "vector_dimension": 2,
            "text": "literal %_ value",
            "chunk_hash": "h'1\\path",
            "created_at": now,
            "metadata": None,
            "user_id": 7,
        },
        {
            "collection": "docs",
            "doc_id": "d2",
            "chunk_id": "c2",
            "parse_hash": "p2",
            "model": "test",
            "vector": [0.2, 0.3],
            "vector_dimension": 2,
            "text": "plain value",
            "chunk_hash": "h2",
            "created_at": now,
            "metadata": "{}",
            "user_id": 8,
        },
        {
            "collection": "docs",
            "doc_id": "d3",
            "chunk_id": "c3",
            "parse_hash": "p3",
            "model": "test",
            "vector": [0.3, 0.4],
            "vector_dimension": 2,
            "text": "another value",
            "chunk_hash": "h3",
            "created_at": now,
            "metadata": '{"section": "intro"}',
            "user_id": 9,
        },
    ]
    table.add(records)

    backend_rows = (
        table.search()
        .where(translate_filter_expression(expression))
        .select(["doc_id"])
        .to_list()
    )
    batch_df = pd.DataFrame(records)
    fallback_mask = _evaluate_filter_expression(batch_df, expression)
    fallback_doc_ids = batch_df.loc[fallback_mask, "doc_id"].tolist()

    assert sorted(row["doc_id"] for row in backend_rows) == expected_doc_ids
    assert fallback_doc_ids == expected_doc_ids


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
    """Test async vector search propagates a missing-table error."""
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
    with pytest.raises(Exception, match="Table not found"):
        await store.search_vectors_async(
            table_name="nonexistent_table",
            query_vector=query_vector,
            top_k=5,
        )


@pytest.mark.asyncio
@patch("lancedb.connect_async", new_callable=AsyncMock)
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
async def test_search_vectors_async_search_failure(
    mock_get_connection: Mock, mock_connect_async: AsyncMock
) -> None:
    """Test async vector search propagates backend failures."""

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
    with pytest.raises(Exception, match="Search failed"):
        await store.search_vectors_async(
            table_name="embeddings_test",
            query_vector=query_vector,
            top_k=5,
        )


@pytest.mark.asyncio
@patch("lancedb.connect_async", new_callable=AsyncMock)
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
async def test_search_fts_async_propagates_search_failure(
    mock_get_connection: Mock, mock_connect_async: AsyncMock
) -> None:
    """Test async FTS search does not report a backend failure as no matches."""
    mock_conn = Mock()
    mock_conn.uri = "test_uri"
    mock_get_connection.return_value = mock_conn

    mock_async_conn = Mock()
    mock_connect_async.return_value = mock_async_conn
    mock_table = Mock()
    mock_async_conn.open_table = AsyncMock(return_value=mock_table)

    mock_search = Mock()
    mock_search.limit.return_value = mock_search

    async def mock_to_arrow():
        raise Exception("FTS search failed")

    mock_search.to_arrow = mock_to_arrow
    mock_table.search.return_value = mock_search

    store = LanceDBVectorIndexStore()

    with pytest.raises(Exception, match="FTS search failed"):
        await store.search_fts_async(
            table_name="embeddings_test",
            query_text="needle",
            top_k=5,
            is_admin=True,
        )


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
async def test_iter_batches_async_rejects_unknown_columns(tmp_path) -> None:
    """Test the locked LanceDB API rejects unknown projected columns."""
    conn = lancedb.connect(str(tmp_path / "lancedb"))
    table = conn.create_table(
        "chunks",
        schema=pa.schema(
            [
                pa.field("doc_id", pa.string()),
                pa.field("text", pa.large_string()),
                pa.field("user_id", pa.int64()),
            ]
        ),
    )
    table.add([{"doc_id": "d1", "text": "one", "user_id": 7}])
    store = LanceDBVectorIndexStore()
    store._conn = conn

    batches = store.iter_batches_async(
        table_name="chunks",
        batch_size=100,
        columns=["does_not_exist"],
        is_admin=True,
    )
    with pytest.raises(RuntimeError, match="No field named does_not_exist"):
        await anext(batches)


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


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_get_table_cache_bypass_reopens_without_storing(
    mock_get_connection: Mock,
) -> None:
    """_get_table(use_cache=False) should always open a fresh uncached handle."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    first = Mock()
    second = Mock()
    cached = Mock()
    mock_conn.open_table.side_effect = [first, second, cached]

    store = LanceDBVectorIndexStore()

    assert store._get_table("documents", use_cache=False) is first
    assert store._get_table("documents", use_cache=False) is second
    assert "documents" not in store._table_cache

    assert store._get_table("documents") is cached
    assert store._get_table("documents") is cached
    assert mock_conn.open_table.call_count == 3


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
    mock_table.search.side_effect = lambda: chains.get(
        # Figure out which table name from the cache — use a side effect approach
        # Since _get_table caches by name, we need a smarter mock
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


def test_count_collections_fast_bypasses_table_cache() -> None:
    """_count_collections_fast should fresh-open tables used by KB listings."""
    import pyarrow as pa

    tbl = pa.table({"collection": ["col_a"]})

    chain = Mock()
    chain.select.return_value = chain
    chain.where.return_value = chain
    chain.limit.return_value = chain
    chain.to_arrow.return_value = tbl

    mock_table = Mock()
    mock_table.search.return_value = chain

    store = LanceDBVectorIndexStore()
    store._get_table = Mock(return_value=mock_table)  # type: ignore[method-assign]

    stats: dict = {}
    store._count_collections_fast(
        "documents", "documents", stats, user_id=None, is_admin=True
    )

    store._get_table.assert_called_once_with("documents", use_cache=False)
    assert stats["col_a"]["documents"] == 1


def test_iter_batches_bypasses_table_cache() -> None:
    """iter_batches should fresh-open tables so KB list reads do not use stale handles."""
    import pyarrow as pa

    batch = pa.record_batch([pa.array(["col_a"])], names=["collection"])

    mock_table = Mock()
    mock_table.to_batches.return_value = [batch]

    store = LanceDBVectorIndexStore()
    store._get_table = Mock(return_value=mock_table)  # type: ignore[method-assign]

    batches = list(store.iter_batches("custom_table", is_admin=True))

    store._get_table.assert_called_once_with("custom_table", use_cache=False)
    assert len(batches) == 1
    assert batches[0].num_rows == 1


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


# ---------------------------------------------------------------------------
# _vis_build_* predicate-builder tests
# ---------------------------------------------------------------------------


def _mk_table_with_columns(columns: list):
    """Mock LanceDB table whose schema.names == columns."""
    table = Mock()
    schema = Mock()
    schema.names = columns
    table.schema = schema
    table.count_rows.return_value = 1
    table.delete = Mock()
    return table


from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (  # noqa: E402
    _vis_build_collection_filter,
    _vis_build_document_filter,
    _vis_build_documents_filter,
    _vis_doc_ids_filter,
)


def test_vis_build_collection_filter_non_admin_with_user_id_column():
    conn = Mock()
    conn.open_table.return_value = _mk_table_with_columns(
        ["collection", "doc_id", "user_id"]
    )
    filt = _vis_build_collection_filter(
        conn=conn, table_name="documents", collection="c1", user_id=7, is_admin=False
    )
    assert "collection == 'c1'" in filt
    assert "user_id == 7" in filt


def test_vis_build_collection_filter_legacy_schema_omits_user_id():
    conn = Mock()
    conn.open_table.return_value = _mk_table_with_columns(["collection", "doc_id"])
    filt = _vis_build_collection_filter(
        conn=conn,
        table_name="documents",
        collection="c_legacy",
        user_id=11,
        is_admin=False,
    )
    assert "collection == 'c_legacy'" in filt
    assert "user_id" not in filt


def test_vis_build_document_filter_scopes_collection_and_doc():
    conn = Mock()
    conn.open_table.return_value = _mk_table_with_columns(
        ["collection", "doc_id", "user_id"]
    )
    filt = _vis_build_document_filter(
        conn=conn,
        table_name="documents",
        collection="c1",
        doc_id="d1",
        user_id=9,
        is_admin=False,
    )
    assert "collection == 'c1'" in filt
    assert "doc_id == 'd1'" in filt
    assert "user_id == 9" in filt


def test_vis_doc_ids_filter_single_and_multi():
    assert _vis_doc_ids_filter(["d1"]) == "doc_id == 'd1'"
    assert _vis_doc_ids_filter(["d1", "d2"]) == "doc_id IN ('d1', 'd2')"


def test_vis_build_documents_filter_unauthenticated_non_admin_fails_closed():
    conn = Mock()
    filt = _vis_build_documents_filter(
        conn=conn,
        table_name="documents",
        collection="c1",
        doc_ids=["d1", "d2"],
        user_id=None,
        is_admin=False,
    )
    assert "collection == 'c1'" in filt
    assert "doc_id IN ('d1', 'd2')" in filt
    # no_access_filter is appended (no tenant rows visible)
    assert "AND (" in filt


# ---------------------------------------------------------------------------
# cascade_delete tests
# ---------------------------------------------------------------------------


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_store_cascade_delete_collection_applies_user_filter(mock_get_conn, mocker):
    for n in (
        "ensure_documents_table",
        "ensure_parses_table",
        "ensure_chunks_table",
        "ensure_main_pointers_table",
        "ensure_ingestion_runs_table",
    ):
        mocker.patch(
            f"xagent.core.tools.core.RAG_tools.LanceDB.schema_manager.{n}",
            return_value=None,
        )
    conn = Mock()
    conn.table_names.return_value = ["documents"]
    conn.list_tables.return_value = ["documents"]
    table = _mk_table_with_columns(["collection", "doc_id", "user_id"])
    conn.open_table.return_value = table
    mock_get_conn.return_value = conn

    store = LanceDBVectorIndexStore()
    store.cascade_delete(
        target="collection",
        collection="c1",
        user_id=7,
        is_admin=False,
        preview_only=False,
        confirm=True,
    )

    assert table.delete.call_count >= 1
    # Check the filter used contained collection and user_id scope
    filt = table.delete.call_args_list[0][0][0]
    assert "collection == 'c1'" in filt
    assert "user_id" in filt


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_store_cascade_delete_document_scopes_doc_id(mock_get_conn, mocker):
    for n in (
        "ensure_documents_table",
        "ensure_parses_table",
        "ensure_chunks_table",
        "ensure_main_pointers_table",
        "ensure_ingestion_runs_table",
    ):
        mocker.patch(
            f"xagent.core.tools.core.RAG_tools.LanceDB.schema_manager.{n}",
            return_value=None,
        )
    conn = Mock()
    conn.table_names.return_value = ["documents"]
    conn.list_tables.return_value = ["documents"]
    table = _mk_table_with_columns(["collection", "doc_id", "user_id"])
    conn.open_table.return_value = table
    mock_get_conn.return_value = conn

    store = LanceDBVectorIndexStore()
    store.cascade_delete(
        target="document",
        collection="c1",
        doc_id="d1",
        user_id=9,
        is_admin=False,
        preview_only=False,
        confirm=True,
    )

    filt = table.delete.call_args_list[0][0][0]
    assert "collection == 'c1'" in filt
    assert "doc_id == 'd1'" in filt


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_store_cascade_delete_preview_does_not_delete(mock_get_conn, mocker):
    for n in (
        "ensure_documents_table",
        "ensure_parses_table",
        "ensure_chunks_table",
        "ensure_main_pointers_table",
        "ensure_ingestion_runs_table",
    ):
        mocker.patch(
            f"xagent.core.tools.core.RAG_tools.LanceDB.schema_manager.{n}",
            return_value=None,
        )
    conn = Mock()
    conn.table_names.return_value = ["documents"]
    conn.list_tables.return_value = ["documents"]
    table = _mk_table_with_columns(["collection", "doc_id"])
    conn.open_table.return_value = table
    mock_get_conn.return_value = conn

    store = LanceDBVectorIndexStore()
    spy = mocker.spy(store, "invalidate_table_cache")
    store.cascade_delete(
        target="collection",
        collection="c1",
        user_id=None,
        is_admin=True,
        preview_only=True,
        confirm=False,
    )
    assert table.delete.call_count == 0  # preview: plan only, no delete
    assert spy.call_count == 0  # preview: no cache invalidation


def test_store_cascade_delete_document_requires_doc_id():
    from xagent.core.tools.core.RAG_tools.core.exceptions import CascadeCleanupError

    store = LanceDBVectorIndexStore()
    with pytest.raises(CascadeCleanupError):
        store.cascade_delete(
            target="document",
            collection="c1",
            user_id=None,
            is_admin=True,
            preview_only=True,
            confirm=False,
        )


def test_vis_cascade_delete_documents_batched_predicates(mocker):
    from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
        _vis_cascade_delete_documents,
    )

    for n in (
        "ensure_documents_table",
        "ensure_parses_table",
        "ensure_chunks_table",
        "ensure_main_pointers_table",
        "ensure_ingestion_runs_table",
    ):
        mocker.patch(
            f"xagent.core.tools.core.RAG_tools.LanceDB.schema_manager.{n}",
            return_value=None,
        )
    conn = Mock()
    conn.table_names.return_value = ["documents"]
    conn.list_tables.return_value = [
        "documents"
    ]  # _vis_get_table_names uses list_tables
    table = _mk_table_with_columns(["collection", "doc_id", "user_id"])
    conn.open_table.return_value = table

    _vis_cascade_delete_documents(
        conn,
        collection="c1",
        doc_ids=["d2", "d1", "d1"],  # out of order, dupe
        user_id=7,
        is_admin=False,
        preview_only=False,
        confirm=True,
    )
    assert table.delete.call_count >= 1
    filt = table.delete.call_args_list[0][0][0]
    assert "collection == 'c1'" in filt
    # doc_ids normalized to sorted deduped list: ["d1", "d2"]
    assert "doc_id IN ('d1', 'd2')" in filt


def test_vis_cascade_delete_documents_unauthenticated_non_admin_returns_empty():
    from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
        _vis_cascade_delete_documents,
    )

    conn = Mock()
    result = _vis_cascade_delete_documents(
        conn,
        collection="c1",
        doc_ids=["d1"],
        user_id=None,
        is_admin=False,
        preview_only=False,
        confirm=True,
    )
    assert result == {}
    conn.open_table.assert_not_called()


# ============================================================================
# Compaction Tests (xorbitsai/xagent#1140)
# ============================================================================


def _mock_table_with_fragments(count: int, versions: int = 1) -> Mock:
    table = Mock()
    table.stats.return_value = {"fragment_stats": {"num_fragments": count}}
    table.list_versions.return_value = _versions(stale=0, fresh=versions)
    return table


def _versions(*, stale: int, fresh: int) -> List[Dict[str, Any]]:
    """Version entries shaped like LanceDB's, split either side of a 7-day window.

    LanceDB reports naive timestamps in local time, so these are local too.
    """
    from datetime import timedelta

    now = datetime.now()
    old = [
        {"version": i, "timestamp": now - timedelta(days=30, seconds=i)}
        for i in range(stale)
    ]
    new = [
        {"version": stale + i, "timestamp": now - timedelta(seconds=i)}
        for i in range(fresh)
    ]
    return old + new


def _mock_conn_with_tables(tmp_dir: Any = None, **tables: Mock) -> Mock:
    """Connection whose table listing matches the tables it can open."""
    conn = Mock()
    conn.uri = str(tmp_dir) if tmp_dir else "memory://not-a-real-path"
    conn.list_tables.return_value = list(tables)
    conn.open_table.side_effect = lambda name, *a, **k: tables[name]
    return conn


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_should_compact_triggers_on_fragmentation(mock_get_connection: Mock) -> None:
    """Fragment count at or above the threshold marks a table for compaction."""
    from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy

    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_conn.open_table.return_value = _mock_table_with_fragments(100)

    store = LanceDBVectorIndexStore()

    assert store.should_compact("collection_metadata", IndexPolicy()) is True


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_should_compact_skips_below_fragment_threshold(
    mock_get_connection: Mock,
) -> None:
    """A table below the fragment threshold is left alone."""
    from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy

    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_conn.open_table.return_value = _mock_table_with_fragments(99)

    store = LanceDBVectorIndexStore()

    assert store.should_compact("collection_metadata", IndexPolicy()) is False


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_should_compact_ignores_index_staleness(mock_get_connection: Mock) -> None:
    """A barely-fragmented table is never compacted just because its index is stale.

    ``optimize()`` bundles compaction with an index rebuild, so a shared
    predicate would rebuild a large unrelated index on every ingest.
    """
    from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy

    table = _mock_table_with_fragments(2)
    stats = Mock()
    stats.num_indexed_rows = 1000
    stats.num_unindexed_rows = 50000  # blows past both smart-reindex thresholds
    table.index_stats.return_value = stats
    mock_get_connection.return_value = _mock_conn_with_tables(embeddings_big=table)

    store = LanceDBVectorIndexStore()
    policy = IndexPolicy()

    # should_reindex still fires on staleness; compaction must not follow it.
    assert store.should_reindex("embeddings_big", 0, policy) is True
    assert store.should_compact("embeddings_big", policy) is False
    assert store.compact_tables(["embeddings_big"], policy) == []
    table.optimize.assert_not_called()


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_should_compact_returns_false_when_stats_unsupported(
    mock_get_connection: Mock,
) -> None:
    """Older lancedb without ``stats()`` degrades to "do nothing", never raises."""
    from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import _fragment_count

    table = Mock()
    table.stats.side_effect = Exception("unsupported")

    mock_conn = Mock()
    mock_conn.open_table.return_value = table
    mock_get_connection.return_value = mock_conn

    assert _fragment_count(table) == 0
    assert LanceDBVectorIndexStore().should_compact("documents") is False


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_trigger_reindex_keeps_recent_versions(mock_get_connection: Mock) -> None:
    """Version pruning always leaves a non-zero safety margin for live readers."""
    from datetime import timedelta

    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_table = Mock()
    mock_conn.open_table.return_value = mock_table

    assert LanceDBVectorIndexStore().trigger_reindex("documents") is True
    kwargs = mock_table.optimize.call_args.kwargs
    assert kwargs["cleanup_older_than"] == timedelta(days=7)
    assert kwargs["cleanup_older_than"] > timedelta(0)


def test_index_policy_rejects_zero_version_retention() -> None:
    """A zero retention window would delete every version but the latest."""
    from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy

    with pytest.raises(ValueError, match="version_retention_days must be positive"):
        IndexPolicy(version_retention_days=0)
    with pytest.raises(ValueError, match="version_retention_days must be positive"):
        IndexPolicy(version_retention_days=-1)


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_compact_tables_only_touches_fragmented_tables(
    mock_get_connection: Mock,
) -> None:
    """Only the named tables are inspected, and only fragmented ones optimized."""
    from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy

    fragmented = _mock_table_with_fragments(500)
    healthy = _mock_table_with_fragments(3)
    mock_get_connection.return_value = _mock_conn_with_tables(
        documents=fragmented, chunks=healthy
    )

    store = LanceDBVectorIndexStore()
    compacted = store.compact_tables(["documents", "chunks"], IndexPolicy())

    assert compacted == ["documents"]
    fragmented.optimize.assert_called_once()
    healthy.optimize.assert_not_called()


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_compact_tables_never_opens_unrelated_tables(
    mock_get_connection: Mock,
) -> None:
    """Only the caller's tables are opened, however many the database holds.

    Listing names is a single cheap metadata call and is expected; opening every
    table in the database is the cost this API exists to avoid.
    """
    from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy

    mock_conn = _mock_conn_with_tables(
        documents=_mock_table_with_fragments(500),
        chunks=_mock_table_with_fragments(500),
        parses=_mock_table_with_fragments(500),
    )
    mock_get_connection.return_value = mock_conn

    store = LanceDBVectorIndexStore()
    assert store.compact_tables(["documents"], IndexPolicy()) == ["documents"]

    # Exact call list, not a set: it also pins how many times each table is
    # opened (should_compact then trigger_reindex), so an extra open shows up.
    assert [c.args[0] for c in mock_conn.open_table.call_args_list] == [
        "documents",
        "documents",
    ]


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_compact_tables_skips_names_that_do_not_exist(
    mock_get_connection: Mock,
) -> None:
    """A missing table is skipped without opening anything.

    Callers probe both spellings of the embeddings table name, so for any
    vendor-prefixed model id one candidate is guaranteed not to exist.
    """
    from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy

    mock_conn = _mock_conn_with_tables(documents=_mock_table_with_fragments(500))
    mock_get_connection.return_value = mock_conn

    store = LanceDBVectorIndexStore()
    compacted = store.compact_tables(
        ["embeddings_BAAI_bge_large_zh_v1_5", "documents"], IndexPolicy()
    )

    assert compacted == ["documents"]
    opened = {c.args[0] for c in mock_conn.open_table.call_args_list}
    assert "embeddings_BAAI_bge_large_zh_v1_5" not in opened


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_compact_tables_swallows_optimize_failure(mock_get_connection: Mock) -> None:
    """A failing optimize is reported, not raised: maintenance is not the flow."""
    from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy

    broken = _mock_table_with_fragments(500)
    broken.optimize.side_effect = RuntimeError("disk full")
    mock_get_connection.return_value = _mock_conn_with_tables(documents=broken)

    assert LanceDBVectorIndexStore().compact_tables(["documents"], IndexPolicy()) == []
    # The point of the test: optimize really was attempted and really failed.
    broken.optimize.assert_called_once()


def test_ingestion_compaction_hook_never_raises() -> None:
    """Compaction failure must not break a successful ingestion."""
    from xagent.core.tools.core.RAG_tools.pipelines.document_ingestion import (
        _compact_storage_if_needed,
    )

    with patch.object(
        StorageFactory,
        "get_vector_index_store",
        side_effect=RuntimeError("store unavailable"),
    ):
        _compact_storage_if_needed("embedding-default")


def test_ingestion_compaction_hook_scopes_to_written_tables() -> None:
    """The hook compacts exactly the tables an ingestion writes, nothing else."""
    from xagent.core.tools.core.RAG_tools.pipelines.document_ingestion import (
        _compact_storage_if_needed,
    )

    store = Mock()
    store.compact_tables.return_value = []
    with patch.object(StorageFactory, "get_vector_index_store", return_value=store):
        _compact_storage_if_needed("text-embedding-v4")

    names = store.compact_tables.call_args.args[0]
    assert set(names) == {
        "documents",
        "parses",
        "chunks",
        "collection_config",
        "collection_metadata",
        "ingestion_runs",
        "embeddings_text_embedding_v4",
    }
    assert len(names) == len(set(names))  # idempotent tag: no duplicate probe


def test_ingestion_compaction_hook_covers_vendor_prefixed_model_ids() -> None:
    """A vendor-prefixed id must reach the table the write path really creates.

    ``to_model_tag`` is not idempotent for ids containing a slash, and the write
    path applies it twice (CollectionHandle then upsert_embeddings), so probing
    only the single-applied spelling silently misses the embeddings table.
    """
    from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import to_model_tag
    from xagent.core.tools.core.RAG_tools.pipelines.document_ingestion import (
        _compact_storage_if_needed,
    )

    model_id = "BAAI/bge-large-zh-v1.5"
    # Guard the premise: if to_model_tag ever becomes idempotent this test is moot.
    assert to_model_tag(model_id) != to_model_tag(to_model_tag(model_id))

    store = Mock()
    store.compact_tables.return_value = []
    with patch.object(StorageFactory, "get_vector_index_store", return_value=store):
        _compact_storage_if_needed(model_id)

    names = store.compact_tables.call_args.args[0]
    assert set(names) == {
        "documents",
        "parses",
        "chunks",
        "collection_config",
        "collection_metadata",
        "ingestion_runs",
        "embeddings_BAAI_bge_large_zh_v1_5",  # single-applied (legacy spelling)
        "embeddings_baai_bge_large_zh_v1_5",  # double-applied (what writes create)
    }


def test_compaction_collapses_fragments_and_retains_recent_versions(
    tmp_path: Any,
) -> None:
    """End-to-end against real LanceDB: files collapse, recent versions survive."""
    import lancedb
    import pyarrow as pa

    from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy

    db = lancedb.connect(str(tmp_path))
    schema = pa.schema([pa.field("name", pa.string())])
    table = db.create_table("documents", schema=schema)
    for i in range(12):
        table.add([{"name": f"c{i}"}])

    store = LanceDBVectorIndexStore()
    policy = IndexPolicy(compact_fragment_threshold=10)
    with patch.object(store, "_get_connection", return_value=db):
        # Append-driven table: one data file per write, so fragments are what
        # schedule it. (Update-driven tables are covered separately below.)
        assert store.should_reindex("documents", 0, policy) is False
        assert store.should_compact("documents", policy) is True
        assert store.compact_tables(["documents"], policy) == ["documents"]

    handle = db.open_table("documents")
    assert handle.stats()["fragment_stats"]["num_fragments"] == 1
    assert len(handle.search().to_arrow()) == 12
    # 7-day retention: nothing written during this test may be pruned.
    assert len(handle.list_versions()) > 1


def test_compaction_prunes_versions_outside_the_retention_window(
    tmp_path: Any,
) -> None:
    """A short retention window really does delete old versions on disk."""
    from datetime import timedelta

    import lancedb
    import pyarrow as pa

    db = lancedb.connect(str(tmp_path))
    schema = pa.schema([pa.field("name", pa.string())])
    table = db.create_table("documents", schema=schema)
    for i in range(12):
        table.add([{"name": f"d{i}"}])
    before = len(db.open_table("documents").list_versions())
    assert before > 10

    store = LanceDBVectorIndexStore()
    with patch.object(store, "_get_connection", return_value=db):
        assert (
            store.trigger_reindex(
                "documents", cleanup_older_than=timedelta(microseconds=1)
            )
            is True
        )

    handle = db.open_table("documents")
    assert len(handle.list_versions()) < before
    assert len(handle.search().to_arrow()) == 12  # data survives the pruning


def test_compaction_preserves_live_rows_of_a_delete_add_table(
    tmp_path: Any,
) -> None:
    """``ingestion_runs`` upserts by delete+add; compaction must keep the winners.

    That pattern is the fastest source of fragmentation (two versions per run)
    and the one where a wrong compaction would be most visible: superseded rows
    must stay gone and the surviving status must be the last one written.
    """
    import lancedb
    import pyarrow as pa

    from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy

    db = lancedb.connect(str(tmp_path))
    schema = pa.schema(
        [pa.field("doc_id", pa.string()), pa.field("status", pa.string())]
    )
    table = db.create_table("ingestion_runs", schema=schema)
    for i in range(10):
        table.delete(f"doc_id = 'd{i}'")
        table.add([{"doc_id": f"d{i}", "status": "running"}])
    # Re-run every doc: each is deleted then re-added with a terminal status.
    for i in range(10):
        table.delete(f"doc_id = 'd{i}'")
        table.add([{"doc_id": f"d{i}", "status": "success"}])

    store = LanceDBVectorIndexStore()
    policy = IndexPolicy(compact_fragment_threshold=10)
    with patch.object(store, "_get_connection", return_value=db):
        assert store.should_compact("ingestion_runs", policy) is True
        assert store.compact_tables(["ingestion_runs"], policy) == ["ingestion_runs"]

    rows = db.open_table("ingestion_runs").search().to_arrow().to_pylist()
    assert {r["doc_id"] for r in rows} == {f"d{i}" for i in range(10)}
    assert all(r["status"] == "success" for r in rows)  # no tombstone resurrection
    assert len(rows) == 10  # no duplicates from the superseded versions


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_trigger_reindex_drops_the_cached_handle(mock_get_connection: Mock) -> None:
    """Pruning removes the versions a cached handle points at, so it must go."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn
    mock_conn.open_table.return_value = Mock()

    store = LanceDBVectorIndexStore()
    store._get_table("documents")
    assert "documents" in store._table_cache

    assert store.trigger_reindex("documents") is True
    assert "documents" not in store._table_cache


def test_compaction_finds_the_embeddings_table_the_write_path_creates(
    tmp_path: Any,
) -> None:
    """End-to-end: the real write path's table name is one the hook actually probes.

    Exercises the genuine double application of ``to_model_tag`` -- the caller
    (CollectionHandle) tags the model id, then ``upsert_embeddings`` tags it
    again -- and asserts the ingest hook asks for that name letter for letter.

    The comparison is on the probed strings, not on whether ``open_table``
    happens to succeed: a case-insensitive filesystem (macOS) resolves the
    wrongly-cased name to the same directory, so an open-based assertion would
    pass locally and still miss the table on a case-sensitive production box.
    """
    import lancedb

    from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import to_model_tag
    from xagent.core.tools.core.RAG_tools.pipelines.document_ingestion import (
        _compact_storage_if_needed,
    )

    model_id = "BAAI/bge-large-zh-v1.5"
    db = lancedb.connect(str(tmp_path))
    store = LanceDBVectorIndexStore()
    probed: List[str] = []

    with patch.object(store, "_get_connection", return_value=db):
        # CollectionHandle._upsert_model_embeddings tags once, then passes on.
        model_tag = to_model_tag(model_id)
        for i in range(12):
            store.upsert_embeddings(
                model_tag,
                [
                    {
                        "collection": "demo",
                        "doc_id": "doc-1",
                        "chunk_id": f"chunk-{i}",
                        "parse_hash": "hash-1",
                        "model": model_id,
                        "vector": [0.1, 0.2],
                        "text": f"text-{i}",
                        "chunk_hash": f"chunk-hash-{i}",
                        "metadata": "{}",
                    }
                ],
            )

        created = [n for n in db.table_names() if n.startswith("embeddings_")]
        assert created == ["embeddings_baai_bge_large_zh_v1_5"]

        from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy

        real_compact_tables = store.compact_tables

        def _record(names: Any, policy: Any = None) -> List[str]:
            probed.extend(names)
            return real_compact_tables(names, policy)

        with patch.object(store, "compact_tables", _record):
            with patch.object(
                StorageFactory, "get_vector_index_store", return_value=store
            ):
                with patch(
                    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores."
                    "DEFAULT_INDEX_POLICY",
                    IndexPolicy(compact_fragment_threshold=10),
                ):
                    _compact_storage_if_needed(model_id)

        stats = db.open_table(created[0]).stats()

    # The name the write path really created must be asked for verbatim.
    assert created[0] in probed
    assert stats["fragment_stats"]["num_fragments"] == 1  # and really got compacted


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_compact_tables_with_empty_list_does_nothing(
    mock_get_connection: Mock,
) -> None:
    """Empty input is a normal boundary: no work, no connection, no error."""
    mock_conn = Mock()
    mock_get_connection.return_value = mock_conn

    assert LanceDBVectorIndexStore().compact_tables([]) == []
    mock_conn.open_table.assert_not_called()
    mock_conn.list_tables.assert_not_called()  # short-circuits before any I/O


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_trigger_reindex_returns_false_for_unopenable_table(
    mock_get_connection: Mock,
) -> None:
    """A table that cannot even be opened reports False rather than raising.

    Distinct from the covered case where an opened table's optimize() fails:
    this is the open_table branch, which the probe-both-spellings scheme hits
    routinely, since one of the two candidate names usually does not exist.
    """
    mock_conn = Mock()
    mock_conn.open_table.side_effect = FileNotFoundError("no such table")
    mock_get_connection.return_value = mock_conn

    assert LanceDBVectorIndexStore().trigger_reindex("embeddings_nope") is False


def test_vector_index_store_contract_defaults_to_no_compaction() -> None:
    """A backend that does not implement compaction inherits safe no-ops.

    Calls the base-class bodies unbound: VectorIndexStore has 42 abstract
    methods, so a stub subclass would be 40 lines of noise to test two.
    """
    from xagent.core.tools.core.RAG_tools.storage.contracts import VectorIndexStore

    backend = Mock(spec=VectorIndexStore)

    assert VectorIndexStore.should_compact(backend, "documents") is False
    assert VectorIndexStore.compact_tables(backend, ["documents"]) == []


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_maintenance_failures_never_log_at_error_level(
    mock_get_connection: Mock,
    caplog: Any,
) -> None:
    """Compaction is best-effort, so its failures stay at WARNING, never ERROR.

    One unopenable table must not shout ERROR on every upload for every
    collection, forever.
    """
    import logging

    # Listed, so compact_tables reaches the body, but unopenable once there.
    mock_conn = Mock()
    mock_conn.uri = "memory://not-a-real-path"
    mock_conn.list_tables.return_value = ["gone"]
    mock_conn.open_table.side_effect = FileNotFoundError("no such table")
    mock_get_connection.return_value = mock_conn

    store = LanceDBVectorIndexStore()
    with caplog.at_level(
        logging.DEBUG, logger="xagent.core.tools.core.RAG_tools.storage.lancedb_stores"
    ):
        assert store.should_reindex("gone", 0, DEFAULT_INDEX_POLICY) is False
        assert store.should_compact("gone") is False
        assert store.trigger_reindex("gone") is False
        assert store.compact_tables(["gone"]) == []

    assert mock_conn.open_table.called  # the failing branch was really reached
    assert [r.levelno for r in caplog.records if r.levelno >= logging.ERROR] == []
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_should_compact_catches_merge_insert_version_growth(tmp_path: Any) -> None:
    """The collection_metadata case: real merge_insert upserts, no new fragments.

    This is the table the whole fix exists for (229 MB in production). It is
    only ever written through ``merge_insert(["name"])``, which rewrites the
    matched rows instead of appending, so its fragment count stays pinned near
    the row count and NEVER reaches the fragment threshold, however long it
    degrades. Version history is what grows without bound and holds the disk.

    Driven by real merge_insert rather than a mocked fragment count on purpose:
    a mock said this table was fragmented, which is exactly why the gap was
    invisible for four review rounds.
    """
    from datetime import datetime, timezone

    import lancedb
    import pyarrow as pa

    from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy

    db = lancedb.connect(str(tmp_path))
    schema = pa.schema(
        [
            pa.field("name", pa.string()),
            pa.field("description", pa.string()),
            pa.field("updated_at", pa.timestamp("us")),
        ]
    )
    rows = 20
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    table = db.create_table("collection_metadata", schema=schema)
    table.add(
        [{"name": f"c{i}", "description": "d", "updated_at": now} for i in range(rows)]
    )

    # Same upsert shape as MetadataStore.save_collection.
    for i in range(120):
        table.merge_insert(
            "name"
        ).when_matched_update_all().when_not_matched_insert_all().execute(
            [
                {
                    "name": f"c{i % rows}",
                    "description": f"rev-{i}",
                    "updated_at": now,
                }
            ]
        )

    handle = db.open_table("collection_metadata")
    fragments = handle.stats()["fragment_stats"]["num_fragments"]
    assert handle.stats()["num_rows"] == rows  # updates, never appends

    # Fixed expectations, not values read back from the same API the code under
    # test uses: a self-derived threshold would pass even if counting were wrong.
    assert len(handle.list_versions()) >= 120  # history grows one per upsert
    assert fragments <= rows  # fragments do not
    assert fragments < IndexPolicy().compact_fragment_threshold

    store = LanceDBVectorIndexStore()
    with patch.object(store, "_get_connection", return_value=db):
        # Everything here was written seconds ago, so nothing is reclaimable
        # yet and neither criterion fires -- including under the shipped default.
        assert store.should_compact("collection_metadata", IndexPolicy()) is False
        assert store.should_compact("collection_metadata") is False

        # Age the whole history past the retention window: now the stale-version
        # criterion is the only thing that can catch this table.
        cutoff = datetime.now() + timedelta(minutes=1)
        assert _stale_version_count(handle, cutoff) >= 120
        policy = IndexPolicy(compact_stale_version_threshold=100)
        with patch(
            "xagent.core.tools.core.RAG_tools.storage.lancedb_stores"
            "._stale_version_count",
            lambda table, _cutoff: _stale_version_count(table, cutoff),
        ):
            assert store.should_compact("collection_metadata", policy) is True
            assert store.compact_tables(["collection_metadata"], policy) == [
                "collection_metadata"
            ]

    after = db.open_table("collection_metadata")
    assert len(after.search().to_arrow()) == rows  # data intact


def test_index_policy_rejects_non_positive_compaction_thresholds() -> None:
    """A zero threshold would make every table qualify on every ingestion."""
    from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy

    for field in ("compact_fragment_threshold", "compact_stale_version_threshold"):
        for bad in (0, -1):
            with pytest.raises(ValueError, match=f"{field} must be positive"):
                IndexPolicy(**{field: bad})


def test_compaction_lock_keeps_concurrent_workers_off_one_table(
    tmp_path: Any,
) -> None:
    """A table already being compacted elsewhere is skipped, not rewritten twice.

    Every loser of LanceDB's commit race has already paid for a full table
    rewrite by the time the commit is rejected. Uses a real subprocess holding
    a real lock, so it asserts cross-process exclusion -- the thing production
    needs -- rather than the in-process behaviour of one locking primitive.
    """
    import subprocess
    import sys
    import textwrap
    import time

    import lancedb
    import pyarrow as pa

    from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy

    db = lancedb.connect(str(tmp_path))
    table = db.create_table("documents", schema=pa.schema([pa.field("n", pa.string())]))
    for i in range(12):
        table.add([{"n": str(i)}])

    store = LanceDBVectorIndexStore()
    policy = IndexPolicy(compact_fragment_threshold=10)
    ready = tmp_path / "held.flag"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            # Holds the lock through the real helper, so the test cannot drift
            # out of step with how lock files are named.
            textwrap.dedent(f"""
                import time
                from unittest.mock import Mock
                from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
                    _compaction_lock,
                )
                conn = Mock()
                conn.uri = {str(tmp_path)!r}
                with _compaction_lock(conn, "documents") as held:
                    assert held is True
                    open({str(ready)!r}, "w").close()
                    time.sleep(30)
            """),
        ]
    )
    try:
        for _ in range(200):
            if ready.exists():
                break
            time.sleep(0.05)
        assert ready.exists(), "lock holder subprocess never started"

        with patch.object(store, "_get_connection", return_value=db):
            assert store.compact_tables(["documents"], policy) == []
    finally:
        holder.kill()
        holder.wait()

    # With the other process gone, the very same call goes through.
    with patch.object(store, "_get_connection", return_value=db):
        assert store.compact_tables(["documents"], policy) == ["documents"]


def test_compaction_lock_is_per_table_not_per_database(tmp_path: Any) -> None:
    """Holding one table's lock must not stop another table being compacted.

    A database-wide lock would let a busy collection starve every other table.
    """
    import lancedb
    import pyarrow as pa

    from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy
    from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
        _compaction_lock,
    )

    db = lancedb.connect(str(tmp_path))
    for name in ("documents", "chunks"):
        t = db.create_table(name, schema=pa.schema([pa.field("n", pa.string())]))
        for i in range(12):
            t.add([{"n": str(i)}])

    store = LanceDBVectorIndexStore()
    policy = IndexPolicy(compact_fragment_threshold=10)

    with _compaction_lock(db, "documents") as held:
        assert held is True
        with patch.object(store, "_get_connection", return_value=db):
            assert store.compact_tables(["chunks"], policy) == ["chunks"]


def test_compaction_lock_does_not_swallow_body_exceptions(tmp_path: Any) -> None:
    """The lock must re-raise the caller's error, with its own type and message.

    A context manager that catches around its own ``yield`` turns any failure
    inside the ``with`` into "generator didn't stop after throw()" and loses the
    original traceback.
    """
    from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
        _compaction_lock,
    )

    conn = Mock()
    conn.uri = str(tmp_path)

    with pytest.raises(ValueError, match="real failure from trigger_reindex"):
        with _compaction_lock(conn, "documents"):
            raise ValueError("real failure from trigger_reindex")

    # And the lock is takeable again afterwards, so the failure did not wedge it.
    # (Only re-acquirability is asserted here; filelock also releases on __del__,
    # so this cannot prove the explicit release ran.)
    with _compaction_lock(conn, "documents") as acquired:
        assert acquired is True


def test_compaction_still_runs_when_the_uri_cannot_be_locked(tmp_path: Any) -> None:
    """A remote URI has no lock file, and compaction must still do its work."""
    import lancedb
    import pyarrow as pa

    from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy

    db = lancedb.connect(str(tmp_path))
    table = db.create_table("documents", schema=pa.schema([pa.field("n", pa.string())]))
    for i in range(12):
        table.add([{"n": str(i)}])

    store = LanceDBVectorIndexStore()
    policy = IndexPolicy(compact_fragment_threshold=10)

    # Same database, but reported under a URI that cannot host a lock file.
    unlockable = Mock(wraps=db)
    unlockable.uri = "s3://bucket/lancedb"
    unlockable.list_tables.return_value = ["documents"]
    unlockable.open_table.side_effect = db.open_table

    with patch.object(store, "_get_connection", return_value=unlockable):
        assert store.compact_tables(["documents"], policy) == ["documents"]

    assert db.open_table("documents").stats()["fragment_stats"]["num_fragments"] == 1


def test_compaction_clears_the_stale_version_backlog(tmp_path: Any) -> None:
    """After compacting, the version criterion falls back to zero.

    This is the anti-ratchet guarantee. If the predicate counted *all* versions
    instead of reclaimable ones it would stay above the threshold forever once
    crossed -- compaction cannot delete versions still inside the retention
    window, and each run commits one more -- so every later ingestion would
    rewrite every table, index rebuild included. Measuring only what a
    compaction can actually remove makes that state unreachable.
    """
    from datetime import timedelta

    import lancedb
    import pyarrow as pa

    db = lancedb.connect(str(tmp_path))
    table = db.create_table("documents", schema=pa.schema([pa.field("n", pa.string())]))
    for i in range(120):
        table.add([{"n": str(i)}])

    handle = db.open_table("documents")
    # Treat the whole history as past the retention window.
    cutoff = datetime.now() + timedelta(minutes=1)
    before = _stale_version_count(handle, cutoff)
    assert before >= 120

    store = LanceDBVectorIndexStore()
    with patch.object(store, "_get_connection", return_value=db):
        assert store.trigger_reindex(
            "documents", cleanup_older_than=timedelta(microseconds=1)
        )

    after = _stale_version_count(db.open_table("documents"), cutoff)
    assert after < before
    assert after <= 2  # only what compaction itself just committed
    assert len(db.open_table("documents").search().to_arrow()) == 120


def test_should_compact_ignores_versions_inside_the_retention_window() -> None:
    """Versions too young to delete must never trigger a compaction.

    Counting them is what produces the ratchet: they cannot be removed, so the
    count only grows and the predicate latches on.
    """
    from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy

    table = Mock()
    table.stats.return_value = {"fragment_stats": {"num_fragments": 2}}
    table.list_versions.return_value = _versions(stale=0, fresh=5000)

    store = LanceDBVectorIndexStore()
    with patch.object(
        store, "_get_connection", return_value=_mock_conn_with_tables(t=table)
    ):
        assert store.should_compact("t", IndexPolicy()) is False


def test_should_compact_counts_only_reclaimable_versions() -> None:
    """The threshold applies to versions older than the retention window."""
    from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy

    table = Mock()
    table.stats.return_value = {"fragment_stats": {"num_fragments": 2}}
    table.list_versions.return_value = _versions(stale=150, fresh=900)

    store = LanceDBVectorIndexStore()
    with patch.object(
        store, "_get_connection", return_value=_mock_conn_with_tables(t=table)
    ):
        assert (
            store.should_compact("t", IndexPolicy(compact_stale_version_threshold=100))
            is True
        )
        # 150 stale is below this threshold even though 1050 versions exist.
        assert (
            store.should_compact("t", IndexPolicy(compact_stale_version_threshold=200))
            is False
        )


def test_should_compact_probes_fragments_first_and_only_once() -> None:
    """Pins the operand order of the ``or`` and bounds the probes per call.

    ``or`` short-circuits only when the left side is already True, so which
    operand comes first *is* the cost model: swap them and every ingestion pays
    the version scan even when fragments alone settle the answer -- ~2.2 ms
    against ~124 ms at 1000 retained versions. A doubled probe is the same bill
    twice. Neither shows up as a failure anywhere else in the suite.
    """
    from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy

    policy = IndexPolicy(compact_fragment_threshold=10)
    calls: List[str] = []

    def _recording_table(fragments: int) -> Mock:
        table = Mock()

        def stats() -> Dict[str, Any]:
            calls.append("stats")
            return {"fragment_stats": {"num_fragments": fragments}}

        def list_versions() -> List[Dict[str, Any]]:
            calls.append("list_versions")
            return _versions(stale=0, fresh=3)

        table.stats.side_effect = stats
        table.list_versions.side_effect = list_versions
        return table

    store = LanceDBVectorIndexStore()

    # Fragmented: settled by the cheap probe, so the scan is never reached.
    with patch.object(
        store,
        "_get_connection",
        return_value=_mock_conn_with_tables(t=_recording_table(50)),
    ):
        assert store.should_compact("t", policy) is True
    assert calls == ["stats"]

    # Healthy: both run, each exactly once, and fragments still go first.
    calls.clear()
    with patch.object(
        store,
        "_get_connection",
        return_value=_mock_conn_with_tables(t=_recording_table(2)),
    ):
        assert store.should_compact("t", policy) is False
    assert calls == ["stats", "list_versions"]


def test_compact_tables_never_raises_from_the_connection() -> None:
    """The "logged, never raised" promise has to cover opening the database.

    ``_get_connection()`` used to sit outside every try in ``compact_tables``,
    so an unreachable database raised straight out of a method documented as
    best-effort; only the caller's own blanket except hid it.
    """
    store = LanceDBVectorIndexStore()
    with patch.object(store, "_get_connection", side_effect=RuntimeError("no db")):
        assert store.compact_tables(["documents"]) == []


def test_compaction_lock_never_raises_from_release(tmp_path: Any) -> None:
    """A failing unlock is maintenance noise, not the caller's exception."""
    from filelock import FileLock

    from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import _compaction_lock

    conn = _mock_conn_with_tables(tmp_path)
    with patch.object(FileLock, "release", side_effect=RuntimeError("lock gone")):
        with _compaction_lock(conn, "documents") as acquired:
            assert acquired is True  # really locked, so release really runs


def test_stale_version_count_returns_zero_when_unsupported() -> None:
    """An older lancedb without list_versions degrades to "do nothing"."""
    table = Mock()
    table.list_versions.side_effect = Exception("unsupported")

    assert _stale_version_count(table, datetime.now()) == 0


@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_compact_tables_warns_when_no_candidate_is_listed(
    mock_get_connection: Mock,
    caplog: Any,
) -> None:
    """Compaction silently switching itself off must leave a trace.

    list_table_names() returns [] instead of raising for any listing shape it
    does not recognise, so a lancedb API change would disable compaction
    permanently with no other signal.
    """
    import logging

    mock_conn = Mock()
    mock_conn.uri = "memory://not-a-real-path"
    mock_conn.list_tables.return_value = []
    mock_get_connection.return_value = mock_conn

    store = LanceDBVectorIndexStore()
    with caplog.at_level(
        logging.DEBUG, logger="xagent.core.tools.core.RAG_tools.storage.lancedb_stores"
    ):
        assert store.compact_tables(["documents", "chunks"]) == []

    assert any(
        r.levelno == logging.WARNING and "documents" in r.getMessage()
        for r in caplog.records
    )


def test_stale_version_count_judges_aware_timestamps_by_instant() -> None:
    """An aware timestamp is judged by the instant it names, not its wall clock.

    LanceDB returns naive local timestamps today; this branch is what keeps that
    assumption from silently inverting if it ever returns aware ones. Reading a
    far zone's wall clock instead shifts a version by most of a day, in either
    direction depending on the offset.
    """
    local_now = datetime.now()
    cutoff = local_now - timedelta(hours=6)

    # A zone at least 12 hours from local, on whichever side stays legal.
    local_offset = local_now.astimezone().utcoffset() or timedelta(0)
    shift = (
        timedelta(hours=-12) if local_offset >= timedelta(0) else timedelta(hours=12)
    )
    far = timezone(local_offset + shift)

    def _aware(hours_ago: int) -> datetime:
        return (local_now - timedelta(hours=hours_ago)).astimezone().astimezone(far)

    table = Mock()
    # One version each side of the cutoff: any constant misreading of the offset
    # moves both by the same amount and so flips exactly one of them.
    table.list_versions.return_value = [
        {"version": 1, "timestamp": _aware(1)},  # inside the window
        {"version": 2, "timestamp": _aware(11)},  # outside it
    ]

    assert _stale_version_count(table, cutoff) == 1


def test_should_compact_measures_the_retention_window_in_local_time() -> None:
    """The cutoff comes from local time, matching LanceDB's naive timestamps.

    Deriving it from utcnow() shifts the whole window by the UTC offset. West of
    Greenwich that marks versions still inside the window as reclaimable, and
    compaction then deletes versions live readers are holding.

    Necessarily a no-op on a machine running at UTC, where the two clocks agree.
    """
    from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy

    policy = IndexPolicy(compact_stale_version_threshold=1, version_retention_days=7)
    boundary = datetime.now() - timedelta(days=policy.version_retention_days)

    table = Mock()
    table.stats.return_value = {"fragment_stats": {"num_fragments": 1}}
    # Straddle the true cutoff by a minute, so any clock skew moves both across.
    table.list_versions.return_value = [
        {"version": 1, "timestamp": boundary - timedelta(minutes=1)},
        {"version": 2, "timestamp": boundary + timedelta(minutes=1)},
    ]

    store = LanceDBVectorIndexStore()
    with patch.object(
        store, "_get_connection", return_value=_mock_conn_with_tables(t=table)
    ):
        assert store.should_compact("t", policy) is True


def test_should_compact_fires_at_exactly_the_stale_version_threshold() -> None:
    """The comparison is >=, so the threshold value itself must trigger."""
    from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy

    table = Mock()
    table.stats.return_value = {"fragment_stats": {"num_fragments": 2}}
    table.list_versions.return_value = _versions(stale=100, fresh=5)

    store = LanceDBVectorIndexStore()
    with patch.object(
        store, "_get_connection", return_value=_mock_conn_with_tables(t=table)
    ):
        assert (
            store.should_compact("t", IndexPolicy(compact_stale_version_threshold=100))
            is True
        )
        assert (
            store.should_compact("t", IndexPolicy(compact_stale_version_threshold=101))
            is False
        )
