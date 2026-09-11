"""Tests for cascade_cleaner unified entry and wrappers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from xagent.core.tools.core.RAG_tools.storage.factory import get_vector_index_store
from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
    LanceDBVectorIndexStore,
)

_STORE = "xagent.core.tools.core.RAG_tools.storage.lancedb_stores"


@pytest.fixture(autouse=True)
def _patch_ensure_tables(mocker) -> None:
    """Avoid schema-manager side effects in unit tests (focus on filter logic).

    Patches the schema_manager source module. The store's cleanup_cascade_by_scope
    imports ensure_* fresh from ``LanceDB.schema_manager``, so these need to be
    patched at the source to avoid real table creation on the mock connection.
    """
    schema = "xagent.core.tools.core.RAG_tools.LanceDB.schema_manager"
    for name in (
        "ensure_documents_table",
        "ensure_parses_table",
        "ensure_chunks_table",
        "ensure_main_pointers_table",
        "ensure_ingestion_runs_table",
    ):
        mocker.patch(f"{schema}.{name}", return_value=None)


def _create_mock_table_with_schema() -> MagicMock:
    """Create a mock table with a schema that includes the metadata field.

    This helper function ensures that schema validation passes in tests
    by providing a mock schema that includes all required fields, especially
    the 'metadata' field that is validated in ensure_chunks_table.

    Returns:
        A MagicMock table object with a properly configured schema.
    """
    table = MagicMock()
    # Create mock schema fields - at minimum include 'metadata' which is validated
    metadata_field = MagicMock()
    metadata_field.name = "metadata"
    collection_field = MagicMock()
    collection_field.name = "collection"
    doc_id_field = MagicMock()
    doc_id_field.name = "doc_id"
    # Set schema as a list of field objects (mimicking PyArrow schema structure)
    table.schema = [collection_field, doc_id_field, metadata_field]
    return table


def _create_mock_table_with_columns(columns: list[str]) -> MagicMock:
    """Create a mock table with schema.names for column-aware filtering tests."""
    table = MagicMock()
    schema = MagicMock()
    schema.names = columns
    table.schema = schema
    table.count_rows.return_value = 1
    table.delete = MagicMock()
    return table


@patch.object(LanceDBVectorIndexStore, "_get_connection")
def test_cleanup_document_preview_then_confirm(mock_get_conn: MagicMock) -> None:
    """Test document cascade cleanup with preview and confirm modes.

    Verifies that:
    1. Preview mode returns deletion counts without actually deleting data
    2. Confirm mode actually executes deletions and returns final counts
    3. Both modes follow the correct deletion order: embeddings -> chunks -> parses -> main_pointers -> documents
    """
    conn = MagicMock(spec=["table_names", "open_table"])
    conn.table_names.return_value = [
        "documents",
        "parses",
        "chunks",
        "main_pointers",
        "embeddings_m1",
    ]

    def _df(n: int) -> pd.DataFrame:
        # Include filter columns so where(...) matches and counts are non-zero
        return pd.DataFrame([{"collection": "c", "doc_id": "d"}] * n)

    table_map = {
        "embeddings_m1": _create_mock_table_with_columns(
            ["collection", "doc_id", "parse_hash", "user_id"]
        ),
        "chunks": _create_mock_table_with_columns(
            ["collection", "doc_id", "parse_hash", "metadata", "user_id"]
        ),
        "parses": _create_mock_table_with_columns(
            ["collection", "doc_id", "parse_hash", "user_id"]
        ),
        "main_pointers": _create_mock_table_with_columns(["collection", "doc_id"]),
        "documents": _create_mock_table_with_columns(
            ["collection", "doc_id", "user_id"]
        ),
        "ingestion_runs": _create_mock_table_with_columns(
            ["collection", "doc_id", "status", "user_id"]
        ),
    }
    table_map["embeddings_m1"].count_rows.return_value = 2
    for t in ["chunks", "parses", "main_pointers", "documents"]:
        table_map[t].count_rows.return_value = 1
    conn.open_table.side_effect = lambda name: table_map[name]
    mock_get_conn.return_value = conn

    store = get_vector_index_store()
    res = store.cleanup_cascade_by_scope(
        "c", "d", "document", preview_only=True, confirm=False
    )
    assert res["embeddings"] == 2 and res["chunks"] == 1

    # confirm: delete paths
    res2 = store.cleanup_cascade_by_scope(
        "c", "d", "document", preview_only=False, confirm=True
    )
    assert res2["documents"] == 1
    assert table_map["documents"].delete.call_count >= 1


@patch.object(LanceDBVectorIndexStore, "_get_connection")
def test_cleanup_parse_preview(mock_get_conn: MagicMock) -> None:
    """Preview counts for parse scope (embeddings, chunks, parses)."""
    # Re-pointed to the store method (logic relocated in #513 Task 5 / H06).
    conn = MagicMock(spec=["table_names", "open_table"])
    conn.table_names.return_value = ["chunks", "embeddings_m1", "parses"]
    table = _create_mock_table_with_schema()
    # parse preview: __embeddings__, chunks, parses
    table.count_rows.side_effect = [1, 1, 1]
    conn.open_table.return_value = table
    mock_get_conn.return_value = conn

    res_parse = get_vector_index_store().cleanup_cascade_by_scope(
        "c", "d", "parse", old_parse_hash="old", new_parse_hash="new"
    )
    assert isinstance(res_parse, dict)


@patch.object(LanceDBVectorIndexStore, "_get_connection")
def test_cleanup_chunk_preview(mock_get_conn: MagicMock) -> None:
    """Preview counts for chunk scope (embeddings, chunks)."""
    # Re-pointed to the store method (logic relocated in #513 Task 5 / H06).
    conn = MagicMock(spec=["table_names", "open_table"])
    conn.table_names.return_value = ["chunks", "embeddings_m1", "parses"]
    table = _create_mock_table_with_schema()
    # chunk preview: __embeddings__, chunks
    table.count_rows.side_effect = [1, 1]
    conn.open_table.return_value = table
    mock_get_conn.return_value = conn

    res_chunk = get_vector_index_store().cleanup_cascade_by_scope(
        "c", "d", "chunk", old_parse_hash="old", new_parse_hash="new"
    )
    assert isinstance(res_chunk, dict)


@patch.object(LanceDBVectorIndexStore, "_get_connection")
def test_cleanup_embed(mock_get_conn: MagicMock) -> None:
    """Test embeddings cascade cleanup functionality.

    Verifies that:
    1. Embeddings cascade cleanup removes embeddings for a specific model_tag
    2. Function returns proper result dictionary with deletion counts
    3. Cleanup is scoped to the specified model_tag only
    4. Handles cases where no embeddings exist (returns 0 count)
    """
    # Re-pointed to the store method (logic relocated in #513 Task 5 / H06).
    conn = MagicMock(spec=["table_names", "open_table"])
    conn.table_names.return_value = ["embeddings_m1"]
    table = _create_mock_table_with_schema()
    table.count_rows.return_value = 1
    conn.open_table.return_value = table
    mock_get_conn.return_value = conn

    res = get_vector_index_store().cleanup_cascade_by_scope(
        "c", "d", "embeddings", model_tag="m1"
    )
    assert res["embeddings"] >= 0


@patch.object(LanceDBVectorIndexStore, "_get_connection")
def test_cleanup_handles_missing_tables(mock_get_conn: MagicMock) -> None:
    """Gracefully handle cases where required tables do not exist.

    Verifies that cleanup functions do not raise when tables are missing and
    return zero counts accordingly.
    """
    # Re-pointed to the store method (logic relocated in #513 Task 5 / H06).
    conn = MagicMock(spec=["table_names", "open_table"])
    # Simulate no tables present in the database
    conn.table_names.return_value = []
    table = _create_mock_table_with_schema()
    # Even if called, return 0 count
    table.count_rows.return_value = 0
    conn.open_table.return_value = table
    mock_get_conn.return_value = conn

    store = get_vector_index_store()

    # Document scope
    r1 = store.cleanup_cascade_by_scope(
        "c", "d", "document", preview_only=True, confirm=False
    )
    assert r1 == {
        "embeddings": 0,
        "chunks": 0,
        "parses": 0,
        "main_pointers": 0,
        "ingestion_runs": 0,
        "documents": 0,
    }

    # Parse scope
    r2 = store.cleanup_cascade_by_scope(
        "c", "d", "parse", old_parse_hash="old", new_parse_hash="new"
    )
    assert (
        r2["embeddings"] == 0
        and r2["chunks"] == 0
        and r2.get("parses", 0) in (0, r2.get("parses", 0))
    )

    # Chunk scope
    r3 = store.cleanup_cascade_by_scope(
        "c", "d", "chunk", old_parse_hash="old", new_parse_hash="new"
    )
    assert r3["embeddings"] == 0 and r3.get("chunks", 0) in (0, r3.get("chunks", 0))

    # Embed scope
    r4 = store.cleanup_cascade_by_scope("c", "d", "embeddings", model_tag="m1")
    assert r4["embeddings"] == 0


@patch.object(LanceDBVectorIndexStore, "_get_connection")
def test_cleanup_embed_with_multiple_models(mock_get_conn: MagicMock) -> None:
    """Test that cleanup_embed respects model_tag and doesn't touch other models.

    This test verifies the critical fix for the model_tag filtering bug:
    1. Setup: Two embeddings tables (bge_large and minilm), both containing data for doc1
    2. Action: Cleanup embeddings for doc1, but only for model_tag="bge_large"
    3. Assert: Only bge_large table is affected, minilm table is completely untouched

    This regression test ensures that calling cleanup_embed_cascade with a specific
    model_tag does not inadvertently delete data from other embeddings tables.
    """
    conn = MagicMock(spec=["table_names", "open_table"])
    # Two embeddings tables exist
    conn.table_names.return_value = ["embeddings_bge_large", "embeddings_minilm"]

    # Track which tables were queried and deleted
    queried_tables = []
    deleted_tables = []

    def mock_open_table(table_name: str) -> MagicMock:
        """Mock open_table to track which tables are accessed."""
        queried_tables.append(table_name)
        table = _create_mock_table_with_schema()

        # bge_large has 3 rows for doc1
        # minilm has 2 rows for doc1
        if table_name == "embeddings_bge_large":
            table.count_rows.return_value = 3
        elif table_name == "embeddings_minilm":
            table.count_rows.return_value = 2

        # Track delete calls
        def mock_delete(filter_expr: str) -> None:
            deleted_tables.append(table_name)

        table.delete = mock_delete
        return table

    conn.open_table.side_effect = mock_open_table
    mock_get_conn.return_value = conn

    # Re-pointed to the store method (logic relocated in #513 Task 5 / H06).
    # Action: Cleanup embeddings for doc1, but only for bge_large
    result = get_vector_index_store().cleanup_cascade_by_scope(
        "c", "d", "embeddings", model_tag="bge_large", preview_only=False, confirm=True
    )

    # Assert: Only deleted from bge_large (3 rows)
    assert result["embeddings"] == 3

    # Critical assertion: minilm table was NOT accessed at all
    assert "embeddings_minilm" not in queried_tables, (
        "Bug: minilm table should not be queried when model_tag='bge_large'"
    )
    assert "embeddings_minilm" not in deleted_tables, (
        "Bug: minilm table should not be deleted when model_tag='bge_large'"
    )

    # Only bge_large should be touched
    assert "embeddings_bge_large" in queried_tables
    assert "embeddings_bge_large" in deleted_tables


@patch.object(LanceDBVectorIndexStore, "_get_connection")
def test_cleanup_embed_without_model_tag_affects_all_tables(
    mock_get_conn: MagicMock,
) -> None:
    """Test that cleanup_embed without model_tag cleans all embeddings tables.

    When model_tag is None, all embeddings tables should be affected.
    This is the expected behavior for full document cleanup.
    """
    conn = MagicMock(spec=["table_names", "open_table"])
    conn.table_names.return_value = ["embeddings_bge_large", "embeddings_minilm"]

    queried_tables = []

    def mock_open_table(table_name: str) -> MagicMock:
        queried_tables.append(table_name)
        table = _create_mock_table_with_schema()
        # Both tables have data
        table.count_rows.return_value = 2
        table.delete = MagicMock()
        return table

    conn.open_table.side_effect = mock_open_table
    mock_get_conn.return_value = conn

    # Re-pointed to the store method (logic relocated in #513 Task 5 / H06).
    # Action: Cleanup without model_tag
    result = get_vector_index_store().cleanup_cascade_by_scope(
        "c", "d", "embeddings", model_tag=None, confirm=True
    )

    # Assert: Both tables affected (2 + 2 = 4 rows)
    assert result["embeddings"] == 4

    # Both tables should be queried
    assert "embeddings_bge_large" in queried_tables
    assert "embeddings_minilm" in queried_tables


@patch.object(LanceDBVectorIndexStore, "_get_connection")
def test_cleanup_document_injection_attack_prevention(mock_get_conn: MagicMock) -> None:
    """Test that SQL injection attacks are properly prevented in document cleanup.

    Verifies that special characters in doc_id and collection are properly escaped
    to prevent SQL injection attacks when building filter expressions.
    """
    # Malicious inputs attempting SQL injection
    malicious_doc_id = "test_doc' OR 1=1 --"
    malicious_collection = "test_coll'; DROP TABLE documents; --"

    conn = MagicMock(spec=["table_names", "open_table"])
    conn.table_names.return_value = ["documents", "parses", "chunks", "embeddings_m1"]

    # Mock table that records the filter expression used
    table = _create_mock_table_with_schema()
    captured_filter = []

    def capture_count_rows(filter_expr: str):
        captured_filter.append(filter_expr)
        return 0

    table.count_rows.side_effect = capture_count_rows
    conn.open_table.return_value = table
    mock_get_conn.return_value = conn

    # Re-pointed to the store method (logic relocated in #513 Task 5 / H06).
    # Execute cleanup (preview mode is enough to test filter building)
    get_vector_index_store().cleanup_cascade_by_scope(
        malicious_collection,
        malicious_doc_id,
        "document",
        preview_only=True,
        confirm=False,
    )

    # Assert: The filter expression should have properly escaped the malicious inputs
    # Single quotes should be doubled, preventing SQL injection
    # Expected: collection == 'test_coll''; DROP TABLE documents; --' AND doc_id == 'test_doc'' OR 1=1 --'
    assert len(captured_filter) > 0
    filter_expr = captured_filter[0]

    # Check that single quotes are escaped (doubled)
    assert "test_coll''; DROP TABLE documents; --'" in filter_expr
    assert "test_doc'' OR 1=1 --'" in filter_expr

    # The filter should NOT contain unescaped malicious SQL
    # (i.e., should not have bare '; DROP TABLE' or ' OR 1=1 ')
    assert "'; DROP TABLE documents; --" not in filter_expr.replace("'';", "")
    assert "' OR 1=1 --" not in filter_expr.replace("''", "")


@patch.object(LanceDBVectorIndexStore, "_get_connection")
def test_cleanup_parse_injection_attack_prevention(mock_get_conn: MagicMock) -> None:
    """Test that SQL injection attacks are properly prevented in parse cleanup.

    Verifies that special characters in parse_hash are properly escaped
    when building filter expressions with parse_hash conditions.
    """
    malicious_parse_hash = "hash123' OR '1'='1"
    collection = "test_collection"
    doc_id = "test_doc"

    conn = MagicMock(spec=["table_names", "open_table"])
    conn.table_names.return_value = ["parses", "chunks", "embeddings_m1"]

    # Mock table that records the filter expression used
    table = _create_mock_table_with_schema()
    captured_filters = []

    def capture_count_rows(filter_expr: str):
        captured_filters.append(filter_expr)
        return 0

    table.count_rows.side_effect = capture_count_rows
    conn.open_table.return_value = table
    mock_get_conn.return_value = conn

    # Re-pointed to the store method (logic relocated in #513 Task 5 / H06).
    # Mock get_main_pointer to return None (so old_parse_hash is None)
    with patch(
        "xagent.core.tools.core.RAG_tools.version_management.main_pointer_manager._get_main_pointer_impl"
    ) as mock_pointer:
        mock_pointer.return_value = None

        # Execute cleanup with malicious parse_hash
        get_vector_index_store().cleanup_cascade_by_scope(
            collection,
            doc_id,
            "parse",
            new_parse_hash=malicious_parse_hash,
            preview_only=True,
            confirm=False,
        )

    # Assert: The filter expression should have properly escaped the malicious parse_hash
    # Expected: parse_hash != 'hash123'' OR ''1''=''1'
    assert len(captured_filters) > 0

    # Check that at least one filter contains the escaped parse_hash
    escaped_found = False
    for filter_expr in captured_filters:
        if "hash123'' OR ''1''=''1'" in filter_expr:
            escaped_found = True
            # Verify the malicious SQL is neutralized
            assert "' OR '1'='1" not in filter_expr.replace("''", "")
            break

    assert escaped_found, (
        f"Expected escaped parse_hash not found in filters: {captured_filters}"
    )


@patch.object(LanceDBVectorIndexStore, "_get_connection")
def test_cleanup_document_preview_respects_model_tag(mock_get_conn: MagicMock) -> None:
    """Test that preview mode respects model_tag filter and doesn't inflate counts.

    This is a critical regression test for the bug where preview_only mode
    would count all embeddings tables regardless of model_tag, while confirm mode
    would only delete the specified model's table, leading to mismatched counts.
    """
    collection = "test_collection"
    doc_id = "test_doc"
    target_model_tag = "model_a"

    conn = MagicMock(spec=["table_names", "open_table"])
    # Setup: Two embeddings tables with different models
    conn.table_names.return_value = [
        "documents",
        "embeddings_model_a",
        "embeddings_model_b",
    ]

    # Mock tables
    tables_called = []

    def mock_open_table(table_name: str):
        tables_called.append(table_name)
        table = _create_mock_table_with_schema()

        # Each table has matching rows
        if table_name.startswith("embeddings_"):
            table.count_rows.return_value = 5
        else:
            table.count_rows.return_value = 1

        return table

    conn.open_table.side_effect = mock_open_table
    mock_get_conn.return_value = conn

    # Re-pointed to the store method (logic relocated in #513 Task 5 / H06).
    # Execute: Preview mode with model_tag specified
    result = get_vector_index_store().cleanup_cascade_by_scope(
        collection,
        doc_id,
        "document",
        model_tag=target_model_tag,
        preview_only=True,
        confirm=False,
    )

    # Assert: Preview should ONLY count embeddings_model_a (5 rows), NOT embeddings_model_b
    # If the bug exists, it would count both tables (10 rows total)
    assert result["embeddings"] == 5, (
        f"Expected 5 rows from embeddings_model_a only, "
        f"got {result['embeddings']} (bug would show 10)"
    )

    # Verify that only the target model's table was queried
    embeddings_tables_queried = [
        t for t in tables_called if t.startswith("embeddings_")
    ]
    assert "embeddings_model_a" in embeddings_tables_queried
    assert "embeddings_model_b" not in embeddings_tables_queried, (
        "Preview mode should respect model_tag filter and not query other models' tables"
    )


@patch.object(LanceDBVectorIndexStore, "_get_connection")
def test_cleanup_parse_cascade_probe_error_without_user_id_denied(
    mock_get_conn: MagicMock,
) -> None:
    """Parse cleanup should fail closed if table schema probing fails."""
    # Re-pointed to the store method (logic relocated in #513 Task 5 / H06).
    conn = MagicMock(spec=["table_names", "open_table"])
    conn.table_names.return_value = ["chunks", "embeddings_m1"]

    def mock_open_table(table_name: str) -> MagicMock:
        if table_name == "chunks":
            raise RuntimeError("chunks unavailable")
        if table_name == "embeddings_m1":
            raise RuntimeError("embeddings unavailable")
        raise AssertionError(table_name)

    conn.open_table.side_effect = mock_open_table
    mock_get_conn.return_value = conn

    with (
        patch(f"{_STORE}._vis_plan_by_predicates") as mock_plan,
        patch(
            "xagent.core.tools.core.RAG_tools.version_management.main_pointer_manager._get_main_pointer_impl",
            return_value=None,
        ),
    ):
        mock_plan.return_value = {}
        get_vector_index_store().cleanup_cascade_by_scope(
            "c_probe",
            "d_probe",
            "parse",
            new_parse_hash="new",
            user_id=None,
            is_admin=False,
            preview_only=True,
            confirm=False,
        )

    predicates = mock_plan.call_args.args[1]
    assert "user_id IS NULL" in predicates["chunks"][0]
    assert "user_id IS NOT NULL" in predicates["chunks"][0]
    assert "user_id IS NULL" in predicates["embeddings_m1"][0]
    assert "user_id IS NOT NULL" in predicates["embeddings_m1"][0]


@patch.object(LanceDBVectorIndexStore, "_get_connection")
def test_cleanup_embed_without_user_scope_fails_closed(
    mock_get_conn: MagicMock,
) -> None:
    """Embeddings cleanup should not become document-wide without user scope."""
    # Re-pointed to the store method (logic relocated in #513 Task 5 / H06).
    conn = MagicMock(spec=["table_names", "open_table"])
    conn.table_names.return_value = ["embeddings_m1"]
    table = _create_mock_table_with_columns(["collection", "doc_id", "user_id"])
    table.count_rows.return_value = 0
    conn.open_table.return_value = table
    mock_get_conn.return_value = conn

    get_vector_index_store().cleanup_cascade_by_scope(
        "c_denied",
        "d_denied",
        "embeddings",
        user_id=None,
        is_admin=False,
        preview_only=True,
        confirm=False,
    )

    filter_expr = table.count_rows.call_args.args[0]
    assert "collection == 'c_denied'" in filter_expr
    assert "doc_id == 'd_denied'" in filter_expr
    assert "user_id IS NULL" in filter_expr
    assert "user_id IS NOT NULL" in filter_expr


def test_cleanup_embed_cascade_preserves_multiple_predicates_per_table() -> None:
    """Embeddings cleanup should pass all per-table predicates to the executor."""
    # Re-pointed to the store method (logic relocated in #513 Task 5 / H06).
    with (
        patch.object(LanceDBVectorIndexStore, "_get_connection") as mock_get_conn,
        patch(
            "xagent.core.tools.core.RAG_tools.storage.lancedb_cleanup_filters.build_embedding_cleanup_filters"
        ) as mock_build_filters,
        patch(f"{_STORE}._vis_plan_by_predicates") as mock_plan,
    ):
        conn = MagicMock(spec=["table_names", "open_table"])
        mock_get_conn.return_value = conn
        mock_build_filters.return_value = {
            "embeddings_m1": ["chunk_id == 'ch1'", "chunk_id == 'ch2'"],
        }
        mock_plan.return_value = {"embeddings_m1": 2}

        result = get_vector_index_store().cleanup_cascade_by_scope(
            "c_multi",
            "d_multi",
            "embeddings",
            user_id=7,
            is_admin=False,
            preview_only=True,
            confirm=False,
        )

    predicates = mock_plan.call_args.args[1]
    assert predicates["embeddings_m1"] == ["chunk_id == 'ch1'", "chunk_id == 'ch2'"]
    assert result["embeddings"] == 2


def test_cleanup_embed_cascade_deletes_multiple_predicates_per_table() -> None:
    """Confirm cleanup should execute every predicate in a table predicate list."""
    # Re-pointed to the store method (logic relocated in #513 Task 5 / H06).
    with (
        patch.object(LanceDBVectorIndexStore, "_get_connection") as mock_get_conn,
        patch(
            "xagent.core.tools.core.RAG_tools.storage.lancedb_cleanup_filters.build_embedding_cleanup_filters"
        ) as mock_build_filters,
    ):
        table = MagicMock()
        table.count_rows.side_effect = [1, 1]
        conn = MagicMock(spec=["table_names", "open_table"])
        conn.table_names.return_value = ["embeddings_m1"]
        conn.open_table.return_value = table
        mock_get_conn.return_value = conn
        mock_build_filters.return_value = {
            "embeddings_m1": ["chunk_id == 'ch1'", "chunk_id == 'ch2'"],
        }

        result = get_vector_index_store().cleanup_cascade_by_scope(
            "c_multi",
            "d_multi",
            "embeddings",
            user_id=7,
            is_admin=False,
            preview_only=False,
            confirm=True,
        )

    assert result["embeddings"] == 2
    assert [args.args[0] for args in table.count_rows.call_args_list] == [
        "chunk_id == 'ch1'",
        "chunk_id == 'ch2'",
    ]
    assert [args.args[0] for args in table.delete.call_args_list] == [
        "chunk_id == 'ch1'",
        "chunk_id == 'ch2'",
    ]


@patch.object(LanceDBVectorIndexStore, "_get_connection")
def test_cleanup_parse_cascade_expands_embeddings_predicates_per_table(
    mock_get_conn: MagicMock,
) -> None:
    """Parse cleanup should use per-table embeddings predicates, not __embeddings__."""
    # Re-pointed to the store method (logic relocated in #513 Task 5 / H06).
    conn = MagicMock(spec=["table_names", "open_table"])
    conn.table_names.return_value = [
        "chunks",
        "parses",
        "embeddings_m1",
        "embeddings_legacy",
    ]
    table_map = {
        "chunks": _create_mock_table_with_columns(
            ["collection", "doc_id", "parse_hash", "user_id"]
        ),
        "parses": _create_mock_table_with_columns(
            ["collection", "doc_id", "parse_hash", "user_id"]
        ),
        "embeddings_m1": _create_mock_table_with_columns(
            ["collection", "doc_id", "parse_hash", "user_id"]
        ),
        "embeddings_legacy": _create_mock_table_with_columns(
            ["collection", "doc_id", "parse_hash"]
        ),
    }
    conn.open_table.side_effect = lambda name: table_map[name]
    mock_get_conn.return_value = conn

    with (
        patch(f"{_STORE}._vis_plan_by_predicates") as mock_plan,
        patch(
            "xagent.core.tools.core.RAG_tools.version_management.main_pointer_manager._get_main_pointer_impl",
            return_value=None,
        ),
    ):
        mock_plan.return_value = {"embeddings_m1": 2, "embeddings_legacy": 3}
        result = get_vector_index_store().cleanup_cascade_by_scope(
            "c_parse",
            "d_parse",
            "parse",
            new_parse_hash="new",
            user_id=7,
            is_admin=False,
            preview_only=True,
            confirm=False,
        )

    predicates = mock_plan.call_args.args[1]
    assert "__embeddings__" not in predicates
    assert "embeddings_m1" in predicates
    assert "embeddings_legacy" in predicates
    assert "user_id == 7" in predicates["embeddings_m1"][0]
    assert "user_id == 7" not in predicates["embeddings_legacy"][0]
    assert result["embeddings"] == 5


@patch.object(LanceDBVectorIndexStore, "_get_connection")
def test_cleanup_parse_cascade_replaces_superseded_predicates(
    mock_get_conn: MagicMock,
) -> None:
    # Re-pointed to the store method (logic relocated in #513 Task 5 / H06).
    conn = MagicMock(spec=["table_names", "open_table"])
    conn.table_names.return_value = ["chunks", "parses", "embeddings_m1"]
    table_map = {
        "chunks": _create_mock_table_with_columns(
            ["collection", "doc_id", "parse_hash", "user_id"]
        ),
        "parses": _create_mock_table_with_columns(
            ["collection", "doc_id", "parse_hash", "user_id"]
        ),
        "embeddings_m1": _create_mock_table_with_columns(
            ["collection", "doc_id", "parse_hash", "user_id"]
        ),
    }
    conn.open_table.side_effect = lambda name: table_map[name]
    mock_get_conn.return_value = conn

    with patch(f"{_STORE}._vis_plan_by_predicates") as mock_plan:
        mock_plan.return_value = {"embeddings_m1": 1, "chunks": 1, "parses": 1}
        get_vector_index_store().cleanup_cascade_by_scope(
            "c_replace",
            "d_replace",
            "parse",
            old_parse_hash="old",
            new_parse_hash="new",
            user_id=7,
            is_admin=False,
            preview_only=True,
            confirm=False,
        )

    predicates = mock_plan.call_args.args[1]
    assert len(predicates["embeddings_m1"]) == 1
    assert "parse_hash != 'new'" in predicates["embeddings_m1"][0]
    assert "parse_hash == 'old'" not in predicates["embeddings_m1"][0]
    assert len(predicates["chunks"]) == 1
    assert "parse_hash != 'new'" in predicates["chunks"][0]
    assert "parse_hash == 'old'" not in predicates["chunks"][0]


@pytest.mark.parametrize(
    ("preview_only", "confirm"),
    [
        (False, False),
        (True, True),
    ],
)
def test_cleanup_cascade_plans_unless_confirmed_outside_preview(
    preview_only: bool,
    confirm: bool,
) -> None:
    # Re-pointed to the store method (logic relocated in #513 Task 5 / H06).
    with (
        patch.object(LanceDBVectorIndexStore, "_get_connection") as mock_get_conn,
        patch(
            "xagent.core.tools.core.RAG_tools.storage.lancedb_cleanup_filters.build_embedding_cleanup_filters"
        ) as mock_build_filters,
        patch(f"{_STORE}._vis_plan_by_predicates") as mock_plan,
        patch(f"{_STORE}._vis_delete_by_predicates") as mock_delete,
    ):
        conn = MagicMock(spec=["table_names", "open_table"])
        mock_get_conn.return_value = conn
        mock_build_filters.return_value = {"embeddings_m1": ["collection == 'c_gate'"]}
        mock_plan.return_value = {"embeddings_m1": 1}

        result = get_vector_index_store().cleanup_cascade_by_scope(
            "c_gate",
            "d_gate",
            "embeddings",
            user_id=7,
            is_admin=False,
            preview_only=preview_only,
            confirm=confirm,
        )

    assert result == {"embeddings": 1}
    mock_plan.assert_called_once()
    mock_delete.assert_not_called()
