from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import (
    EmbeddingReadResponse,
    EmbeddingWriteResponse,
)
from xagent.core.tools.core.RAG_tools.kb import (
    KBCoordinator,
    KBVectorStorageCleanupResult,
    KBVectorStorageCompatibilityFacade,
)
from xagent.core.tools.core.RAG_tools.kb.cleanup_filters import (
    normalize_cleanup_chunk_ids,
)
from xagent.core.tools.core.RAG_tools.utils.user_scope import user_scope_context
from xagent.core.tools.core.RAG_tools.vector_storage import vector_manager


def _cleanup_shim(conn: MagicMock) -> MagicMock:
    """Shim whose handle path reaches *conn* (coordinator -> handle route)."""
    storage_shim = MagicMock()
    storage_shim.get_vector_store_raw_connection.return_value = conn
    storage_shim.get_vector_index_store.return_value.get_raw_connection.return_value = (
        conn
    )
    metadata_store = MagicMock()
    metadata_store.get_collection = AsyncMock(return_value=None)
    storage_shim.get_metadata_store.return_value = metadata_store
    return storage_shim


def test_public_vector_storage_functions_route_through_facade(monkeypatch):
    facade = MagicMock()
    read_response = EmbeddingReadResponse(chunks=[], total_count=0, pending_count=0)
    write_response = EmbeddingWriteResponse(
        upsert_count=0,
        deleted_stale_count=0,
        index_status="skipped",
    )
    facade.read_chunks_for_embedding.return_value = read_response
    facade.write_vectors_to_db.return_value = write_response

    monkeypatch.setattr(
        vector_manager,
        "_get_vector_storage_compatibility_facade",
        lambda: facade,
    )

    vector_manager.validate_query_vector([1.0])
    assert (
        vector_manager.read_chunks_for_embedding(
            collection="c",
            doc_id="d",
            parse_hash="p",
            model="m",
            filters={"section": "s"},
            user_id=7,
            is_admin=False,
        )
        is read_response
    )
    assert (
        vector_manager.write_vectors_to_db(
            collection="c",
            embeddings=[],
            create_index=False,
            user_id=7,
        )
        is write_response
    )

    facade.validate_query_vector.assert_called_once_with(
        [1.0],
        model_tag=None,
        conn=None,
        user_id=None,
        is_admin=False,
    )
    facade.read_chunks_for_embedding.assert_called_once_with(
        collection="c",
        doc_id="d",
        parse_hash="p",
        model="m",
        filters={"section": "s"},
        user_id=7,
        is_admin=False,
    )
    facade.write_vectors_to_db.assert_called_once_with(
        collection="c",
        embeddings=[],
        create_index=False,
        user_id=7,
    )


def test_vector_storage_facade_binds_storage_shim_for_read_chunks():
    # An injected shim (no coordinator) must keep embedding reads bound to that
    # shim's stores: the facade opens a handle through a shim-backed coordinator
    # and the handle reads from the shim's vector store (#510 re-route).
    vector_store = MagicMock()
    vector_store.count_rows_or_zero.return_value = 0
    metadata_store = MagicMock()
    metadata_store.get_collection = AsyncMock(return_value=None)
    storage_shim = MagicMock()
    storage_shim.get_vector_index_store.return_value = vector_store
    storage_shim.get_metadata_store.return_value = metadata_store
    facade = KBVectorStorageCompatibilityFacade(storage_shim=storage_shim)

    result = facade.read_chunks_for_embedding(
        collection="c",
        doc_id="d",
        parse_hash="p",
        model="m",
        user_id=7,
        is_admin=False,
    )

    assert result == EmbeddingReadResponse(chunks=[], total_count=0, pending_count=0)
    vector_store.count_rows_or_zero.assert_called_once_with(
        table_name="chunks",
        filters={"collection": "c", "doc_id": "d", "parse_hash": "p"},
        user_id=7,
        is_admin=False,
    )
    vector_store.iter_batches.assert_not_called()


def test_coordinator_exposes_vector_storage_facade():
    coordinator = KBCoordinator()

    assert isinstance(
        coordinator.vector_storage_compatibility,
        KBVectorStorageCompatibilityFacade,
    )
    assert coordinator.vector_storage is coordinator.vector_storage_compatibility


def test_normalize_cleanup_chunk_ids_skips_none_and_empty_values():
    result = normalize_cleanup_chunk_ids(["ch2", None, "", "ch1", "ch1"])

    assert result == ("ch1", "ch2")


def test_cleanup_vectors_for_chunks_uses_model_tag_table_and_reports_counts(
    monkeypatch,
):
    table = _table_with_user_id(count=2)
    conn = MagicMock()
    conn.list_tables.return_value = ["embeddings_model_a", "embeddings_model_b"]
    conn.open_table.return_value = table
    storage_shim = _cleanup_shim(conn)
    facade = KBVectorStorageCompatibilityFacade(storage_shim=storage_shim)

    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.kb.cleanup_filters."
        "build_lancedb_filter_expression",
        lambda filters, **kwargs: " AND ".join(
            f"{key} == '{value}'" for key, value in filters.items()
        ),
    )

    result = facade.cleanup_vectors_for_chunks(
        collection="c",
        doc_id="d",
        chunk_ids=["ch1", "ch1"],
        model_tag="model_a",
        user_id=7,
        is_admin=False,
        preview_only=False,
        confirm=True,
    )

    assert result == KBVectorStorageCleanupResult(
        collection="c",
        status="complete",
        deleted_count=2,
        table_counts={"embeddings_model_a": 2},
        model_tag="model_a",
        preview_only=False,
    )
    assert conn.open_table.call_args_list == [
        call("embeddings_model_a"),
        call("embeddings_model_a"),
    ]
    table.count_rows.assert_called_once_with(
        "collection == 'c' AND doc_id == 'd' AND chunk_id == 'ch1' AND user_id == 7"
    )
    table.delete.assert_called_once_with(
        "collection == 'c' AND doc_id == 'd' AND chunk_id == 'ch1' AND user_id == 7"
    )


def test_cleanup_vectors_for_chunks_batches_chunk_ids_with_in_filter(monkeypatch):
    table = _table_with_user_id(count=4)
    conn = MagicMock()
    conn.list_tables.return_value = ["embeddings_model_a"]
    conn.open_table.return_value = table
    storage_shim = _cleanup_shim(conn)
    facade = KBVectorStorageCompatibilityFacade(storage_shim=storage_shim)

    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.kb.cleanup_filters."
        "build_lancedb_filter_expression",
        lambda filters, **kwargs: " AND ".join(
            f"{key} == '{value}'" for key, value in filters.items()
        ),
    )

    result = facade.cleanup_vectors_for_chunks(
        collection="c",
        doc_id="d",
        chunk_ids=["ch2", "ch1", "ch1"],
        model_tag="model_a",
        user_id=7,
        is_admin=False,
        preview_only=False,
        confirm=True,
    )

    assert result.deleted_count == 4
    assert result.table_counts == {"embeddings_model_a": 4}
    expected_filter = (
        "collection == 'c' AND doc_id == 'd' "
        "AND chunk_id IN ('ch1', 'ch2') AND user_id == 7"
    )
    table.count_rows.assert_called_once_with(expected_filter)
    table.delete.assert_called_once_with(expected_filter)


def test_cleanup_vectors_for_operation_uses_request_user_scope(monkeypatch):
    table = _table_with_user_id(count=1)
    conn = MagicMock()
    conn.list_tables.return_value = ["embeddings_model_a"]
    conn.open_table.return_value = table
    storage_shim = _cleanup_shim(conn)
    facade = KBVectorStorageCompatibilityFacade(storage_shim=storage_shim)

    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.kb.cleanup_filters."
        "build_lancedb_filter_expression",
        lambda filters, **kwargs: " AND ".join(
            f"{key} == '{value}'" for key, value in filters.items()
        ),
    )

    with user_scope_context(user_id=42, is_admin=False):
        result = facade.cleanup_vectors_for_document(
            collection="c",
            doc_id="d",
            preview_only=True,
            confirm=False,
        )

    assert result.status == "planned"
    table.count_rows.assert_called_once_with(
        "collection == 'c' AND doc_id == 'd' AND user_id == 42"
    )


def test_cleanup_vectors_for_operation_without_user_scope_fails_closed(monkeypatch):
    table = _table_with_user_id(count=0)
    conn = MagicMock()
    conn.list_tables.return_value = ["embeddings_model_a"]
    conn.open_table.return_value = table
    storage_shim = _cleanup_shim(conn)
    facade = KBVectorStorageCompatibilityFacade(storage_shim=storage_shim)

    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.kb.cleanup_filters."
        "build_lancedb_filter_expression",
        lambda filters, **kwargs: " AND ".join(
            f"{key} == '{value}'" for key, value in filters.items()
        ),
    )

    facade.cleanup_vectors_for_document(
        collection="c",
        doc_id="d",
        user_id=None,
        is_admin=False,
        preview_only=True,
        confirm=False,
    )

    filter_expr = table.count_rows.call_args.args[0]
    assert "collection == 'c'" in filter_expr
    assert "doc_id == 'd'" in filter_expr
    assert "user_id IS NULL" in filter_expr
    assert "user_id IS NOT NULL" in filter_expr


def test_cleanup_vectors_for_operation_rejects_chunk_ids_without_doc_id():
    facade = KBVectorStorageCompatibilityFacade(storage_shim=MagicMock())

    with pytest.raises(ValueError, match="doc_id is required"):
        facade.cleanup_vectors_for_operation(
            collection="c",
            chunk_ids=["ch1"],
            model_tag="model_a",
        )


def test_cleanup_vectors_for_operation_preview_does_not_delete(monkeypatch):
    table = _table_with_user_id(count=3)
    conn = MagicMock()
    conn.list_tables.return_value = ["embeddings_model_a"]
    conn.open_table.return_value = table
    storage_shim = _cleanup_shim(conn)
    facade = KBVectorStorageCompatibilityFacade(storage_shim=storage_shim)

    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.kb.cleanup_filters."
        "build_lancedb_filter_expression",
        lambda filters, **kwargs: "base_filter",
    )

    result = facade.cleanup_vectors_for_operation(
        collection="c",
        doc_id="d",
        preview_only=True,
        confirm=True,
    )

    assert result.status == "planned"
    assert result.deleted_count == 3
    assert result.table_counts == {"embeddings_model_a": 3}
    table.delete.assert_not_called()


def test_cleanup_vectors_for_operation_reports_partial_cleanup_failure(monkeypatch):
    table = _table_with_user_id(count=0)
    table.count_rows.side_effect = RuntimeError("count failed")
    conn = MagicMock()
    conn.list_tables.return_value = ["embeddings_model_a"]
    conn.open_table.return_value = table
    storage_shim = _cleanup_shim(conn)
    facade = KBVectorStorageCompatibilityFacade(storage_shim=storage_shim)

    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.kb.cleanup_filters."
        "build_lancedb_filter_expression",
        lambda filters, **kwargs: "base_filter",
    )

    result = facade.cleanup_vectors_for_operation(
        collection="c",
        doc_id="d",
        preview_only=False,
        confirm=True,
    )

    assert result.status == "incomplete"
    assert result.deleted_count == 0
    assert result.table_counts == {}
    assert result.side_effects_may_remain is True
    assert result.warnings == ("embeddings_model_a: count failed",)


def _table_with_user_id(count: int) -> MagicMock:
    table = MagicMock()
    schema = MagicMock()
    schema.names = ["collection", "doc_id", "chunk_id", "user_id"]
    user_id_field = MagicMock()
    user_id_field.type = "int64"
    schema.field.return_value = user_id_field
    table.schema = schema
    table.count_rows.return_value = count
    return table


def test_cleanup_facade_delegates_to_coordinator_and_never_opens_raw_connection(
    monkeypatch,
) -> None:
    """#515 thinness: the facade normalizes and delegates; the raw LanceDB
    connection is opened by the handle, never by the facade."""
    from xagent.core.tools.core.RAG_tools.storage import factory as storage_factory

    def _forbidden() -> None:
        raise AssertionError("facade must not open a raw vector connection")

    monkeypatch.setattr(storage_factory, "get_vector_store_raw_connection", _forbidden)

    coordinator = MagicMock()
    expected = KBVectorStorageCleanupResult(collection="c", status="planned")
    coordinator.cleanup_vectors_for_operation_sync.return_value = expected
    facade = KBVectorStorageCompatibilityFacade(coordinator=coordinator)

    result = facade.cleanup_vectors_for_operation(
        collection="c",
        doc_id="d",
        model_tag="model_a",
        user_id=7,
        is_admin=False,
        preview_only=True,
        confirm=False,
    )

    assert result is expected
    coordinator.cleanup_vectors_for_operation_sync.assert_called_once_with(
        "c",
        doc_id="d",
        parse_hash=None,
        chunk_ids=(),
        model_tag="model_a",
        user_id=7,
        is_admin=False,
        preview_only=True,
        confirm=False,
    )


def test_vector_storage_facade_binds_storage_shim_for_write_vectors():
    # #514-deferred embedding-WRITE routing proof: an injected shim (no
    # coordinator) must land writes on that shim's vector store through the
    # handle write path (handle.write_embeddings), never a real backend.
    from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

    vector_store = MagicMock()
    metadata_store = MagicMock()
    metadata_store.get_collection = AsyncMock(return_value=None)
    storage_shim = MagicMock()
    storage_shim.get_vector_index_store.return_value = vector_store
    storage_shim.get_metadata_store.return_value = metadata_store
    facade = KBVectorStorageCompatibilityFacade(storage_shim=storage_shim)

    embedding = ChunkEmbeddingData(
        doc_id="d",
        chunk_id="ch1",
        parse_hash="p",
        model="m",
        vector=[0.1, 0.2],
        text="t",
        chunk_hash="h",
    )
    result = facade.write_vectors_to_db("c", [embedding], create_index=False, user_id=7)

    assert result.upsert_count == 1
    vector_store.upsert_embeddings.assert_called_once()
    _model_tag, records = vector_store.upsert_embeddings.call_args.args
    assert records[0]["collection"] == "c"
    assert records[0]["doc_id"] == "d"
    assert records[0]["user_id"] == 7
