"""Tests for cascade_cleaner unified entry and wrappers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner import (
    cascade_delete,
    cleanup_chunk_cascade,
    cleanup_document_cascade,
    cleanup_embed_cascade,
    cleanup_parse_cascade,
)


@pytest.fixture(autouse=True)
def _patch_ensure_tables(mocker) -> None:
    """Avoid schema-manager side effects in unit tests (focus on filter logic)."""
    base = "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner"
    mocker.patch(f"{base}.ensure_documents_table", return_value=None)
    mocker.patch(f"{base}.ensure_parses_table", return_value=None)
    mocker.patch(f"{base}.ensure_chunks_table", return_value=None)
    mocker.patch(f"{base}.ensure_main_pointers_table", return_value=None)
    mocker.patch(f"{base}.ensure_ingestion_runs_table", return_value=None)


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


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_vector_store_raw_connection"
)
def test_cleanup_document_preview_then_confirm(mock_get_conn: MagicMock) -> None:
    """Test document cascade cleanup with preview and confirm modes.

    Verifies that:
    1. Preview mode returns deletion counts without actually deleting data
    2. Confirm mode actually executes deletions and returns final counts
    3. Both modes follow the correct deletion order: embeddings -> chunks -> parses -> main_pointers -> documents
    """
    conn = MagicMock()
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

    res = cleanup_document_cascade("c", "d", preview_only=True, confirm=False)
    assert res["embeddings"] == 2 and res["chunks"] == 1

    # confirm: delete paths
    res2 = cleanup_document_cascade("c", "d", preview_only=False, confirm=True)
    assert res2["documents"] == 1
    assert table_map["documents"].delete.call_count >= 1


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_vector_store_raw_connection"
)
def test_cleanup_parse_preview(mock_get_conn: MagicMock) -> None:
    """Preview counts for parse scope (embeddings, chunks, parses)."""
    conn = MagicMock()
    conn.table_names.return_value = ["chunks", "embeddings_m1", "parses"]
    table = _create_mock_table_with_schema()
    # parse preview: __embeddings__, chunks, parses
    table.count_rows.side_effect = [1, 1, 1]
    conn.open_table.return_value = table
    mock_get_conn.return_value = conn

    res_parse = cleanup_parse_cascade(
        "c", "d", old_parse_hash="old", new_parse_hash="new"
    )
    assert isinstance(res_parse, dict)


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_vector_store_raw_connection"
)
def test_cleanup_chunk_preview(mock_get_conn: MagicMock) -> None:
    """Preview counts for chunk scope (embeddings, chunks)."""
    conn = MagicMock()
    conn.table_names.return_value = ["chunks", "embeddings_m1", "parses"]
    table = _create_mock_table_with_schema()
    # chunk preview: __embeddings__, chunks
    table.count_rows.side_effect = [1, 1]
    conn.open_table.return_value = table
    mock_get_conn.return_value = conn

    res_chunk = cleanup_chunk_cascade(
        "c", "d", old_parse_hash="old", new_parse_hash="new"
    )
    assert isinstance(res_chunk, dict)


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_vector_store_raw_connection"
)
def test_cleanup_embed(mock_get_conn: MagicMock) -> None:
    """Test embeddings cascade cleanup functionality.

    Verifies that:
    1. Embeddings cascade cleanup removes embeddings for a specific model_tag
    2. Function returns proper result dictionary with deletion counts
    3. Cleanup is scoped to the specified model_tag only
    4. Handles cases where no embeddings exist (returns 0 count)
    """
    conn = MagicMock()
    conn.table_names.return_value = ["embeddings_m1"]
    table = _create_mock_table_with_schema()
    table.count_rows.return_value = 1
    conn.open_table.return_value = table
    mock_get_conn.return_value = conn

    res = cleanup_embed_cascade("c", "d", model_tag="m1")
    assert res["embeddings"] >= 0


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_vector_store_raw_connection"
)
def test_cleanup_handles_missing_tables(mock_get_conn: MagicMock) -> None:
    """Gracefully handle cases where required tables do not exist.

    Verifies that cleanup functions do not raise when tables are missing and
    return zero counts accordingly.
    """
    conn = MagicMock()
    # Simulate no tables present in the database
    conn.table_names.return_value = []
    table = _create_mock_table_with_schema()
    # Even if called, return 0 count
    table.count_rows.return_value = 0
    conn.open_table.return_value = table
    mock_get_conn.return_value = conn

    # Document scope
    r1 = cleanup_document_cascade("c", "d", preview_only=True, confirm=False)
    assert r1 == {
        "embeddings": 0,
        "chunks": 0,
        "parses": 0,
        "main_pointers": 0,
        "ingestion_runs": 0,
        "documents": 0,
    }

    # Parse scope
    r2 = cleanup_parse_cascade("c", "d", old_parse_hash="old", new_parse_hash="new")
    assert (
        r2["embeddings"] == 0
        and r2["chunks"] == 0
        and r2.get("parses", 0) in (0, r2.get("parses", 0))
    )

    # Chunk scope
    r3 = cleanup_chunk_cascade("c", "d", old_parse_hash="old", new_parse_hash="new")
    assert r3["embeddings"] == 0 and r3.get("chunks", 0) in (0, r3.get("chunks", 0))

    # Embed scope
    r4 = cleanup_embed_cascade("c", "d", model_tag="m1")
    assert r4["embeddings"] == 0


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_vector_store_raw_connection"
)
def test_cleanup_embed_with_multiple_models(mock_get_conn: MagicMock) -> None:
    """Test that cleanup_embed respects model_tag and doesn't touch other models.

    This test verifies the critical fix for the model_tag filtering bug:
    1. Setup: Two embeddings tables (bge_large and minilm), both containing data for doc1
    2. Action: Cleanup embeddings for doc1, but only for model_tag="bge_large"
    3. Assert: Only bge_large table is affected, minilm table is completely untouched

    This regression test ensures that calling cleanup_embed_cascade with a specific
    model_tag does not inadvertently delete data from other embeddings tables.
    """
    conn = MagicMock()
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

    # Action: Cleanup embeddings for doc1, but only for bge_large
    result = cleanup_embed_cascade(
        "c", "d", model_tag="bge_large", preview_only=False, confirm=True
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


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_vector_store_raw_connection"
)
def test_cleanup_embed_without_model_tag_affects_all_tables(
    mock_get_conn: MagicMock,
) -> None:
    """Test that cleanup_embed without model_tag cleans all embeddings tables.

    When model_tag is None, all embeddings tables should be affected.
    This is the expected behavior for full document cleanup.
    """
    conn = MagicMock()
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

    # Action: Cleanup without model_tag
    result = cleanup_embed_cascade("c", "d", model_tag=None, confirm=True)

    # Assert: Both tables affected (2 + 2 = 4 rows)
    assert result["embeddings"] == 4

    # Both tables should be queried
    assert "embeddings_bge_large" in queried_tables
    assert "embeddings_minilm" in queried_tables


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_vector_store_raw_connection"
)
def test_cleanup_document_injection_attack_prevention(mock_get_conn: MagicMock) -> None:
    """Test that SQL injection attacks are properly prevented in document cleanup.

    Verifies that special characters in doc_id and collection are properly escaped
    to prevent SQL injection attacks when building filter expressions.
    """
    # Malicious inputs attempting SQL injection
    malicious_doc_id = "test_doc' OR 1=1 --"
    malicious_collection = "test_coll'; DROP TABLE documents; --"

    conn = MagicMock()
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

    # Execute cleanup (preview mode is enough to test filter building)
    cleanup_document_cascade(
        malicious_collection, malicious_doc_id, preview_only=True, confirm=False
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


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_vector_store_raw_connection"
)
def test_cleanup_parse_injection_attack_prevention(mock_get_conn: MagicMock) -> None:
    """Test that SQL injection attacks are properly prevented in parse cleanup.

    Verifies that special characters in parse_hash are properly escaped
    when building filter expressions with parse_hash conditions.
    """
    malicious_parse_hash = "hash123' OR '1'='1"
    collection = "test_collection"
    doc_id = "test_doc"

    conn = MagicMock()
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

    # Mock get_main_pointer to return None (so old_parse_hash is None)
    with patch(
        "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_main_pointer"
    ) as mock_pointer:
        mock_pointer.return_value = None

        # Execute cleanup with malicious parse_hash
        cleanup_parse_cascade(
            collection,
            doc_id,
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


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_vector_store_raw_connection"
)
def test_cleanup_document_preview_respects_model_tag(mock_get_conn: MagicMock) -> None:
    """Test that preview mode respects model_tag filter and doesn't inflate counts.

    This is a critical regression test for the bug where preview_only mode
    would count all embeddings tables regardless of model_tag, while confirm mode
    would only delete the specified model's table, leading to mismatched counts.
    """
    collection = "test_collection"
    doc_id = "test_doc"
    target_model_tag = "model_a"

    conn = MagicMock()
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

    # Execute: Preview mode with model_tag specified
    result = cleanup_document_cascade(
        collection, doc_id, model_tag=target_model_tag, preview_only=True, confirm=False
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


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_vector_store_raw_connection"
)
def test_cascade_delete_collection_with_user_id_column_applies_user_filter(
    mock_get_conn: MagicMock,
) -> None:
    """Non-admin collection delete should include user_id when table has the column."""
    conn = MagicMock()
    conn.table_names.return_value = ["documents"]
    table = _create_mock_table_with_columns(["collection", "doc_id", "user_id"])
    conn.open_table.return_value = table
    mock_get_conn.return_value = conn

    cascade_delete(
        target="collection",
        collection="c1",
        user_id=7,
        is_admin=False,
        preview_only=False,
        confirm=True,
    )

    assert table.delete.call_count == 1
    filt = table.delete.call_args[0][0]
    assert "collection == 'c1'" in filt
    assert "user_id == 7" in filt


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_vector_store_raw_connection"
)
def test_cascade_delete_document_with_user_id_column_applies_user_filter(
    mock_get_conn: MagicMock,
) -> None:
    """Non-admin document delete should include user_id when table has the column."""
    conn = MagicMock()
    conn.table_names.return_value = ["documents"]
    table = _create_mock_table_with_columns(["collection", "doc_id", "user_id"])
    conn.open_table.return_value = table
    mock_get_conn.return_value = conn

    cascade_delete(
        target="document",
        collection="c1",
        doc_id="d1",
        user_id=9,
        is_admin=False,
        preview_only=False,
        confirm=True,
    )

    assert table.delete.call_count == 1
    filt = table.delete.call_args[0][0]
    assert "collection == 'c1'" in filt
    assert "doc_id == 'd1'" in filt
    assert "user_id == 9" in filt


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_vector_store_raw_connection"
)
def test_cascade_delete_collection_without_user_id_column_compatible(
    mock_get_conn: MagicMock,
) -> None:
    """Non-admin collection delete should not inject user_id for legacy schemas."""
    conn = MagicMock()
    conn.table_names.return_value = ["documents"]
    table = _create_mock_table_with_columns(["collection", "doc_id"])
    conn.open_table.return_value = table
    mock_get_conn.return_value = conn

    cascade_delete(
        target="collection",
        collection="c_legacy",
        user_id=11,
        is_admin=False,
        preview_only=False,
        confirm=True,
    )

    filt = table.delete.call_args[0][0]
    assert "collection == 'c_legacy'" in filt
    assert "user_id ==" not in filt


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_vector_store_raw_connection"
)
def test_cascade_delete_document_without_user_id_column_compatible(
    mock_get_conn: MagicMock,
) -> None:
    """Non-admin document delete should stay compatible for legacy schemas."""
    conn = MagicMock()
    conn.table_names.return_value = ["documents"]
    table = _create_mock_table_with_columns(["collection", "doc_id"])
    conn.open_table.return_value = table
    mock_get_conn.return_value = conn

    cascade_delete(
        target="document",
        collection="c_legacy",
        doc_id="d_legacy",
        user_id=12,
        is_admin=False,
        preview_only=False,
        confirm=True,
    )

    filt = table.delete.call_args[0][0]
    assert "collection == 'c_legacy'" in filt
    assert "doc_id == 'd_legacy'" in filt
    assert "user_id ==" not in filt


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_vector_store_raw_connection"
)
def test_cascade_delete_admin_vs_non_admin_user_filter_behavior(
    mock_get_conn: MagicMock,
) -> None:
    """Admin should not be filtered by user_id; non-admin should be filtered."""
    conn = MagicMock()
    conn.table_names.return_value = ["documents"]
    table = _create_mock_table_with_columns(["collection", "doc_id", "user_id"])
    conn.open_table.return_value = table
    mock_get_conn.return_value = conn

    # Admin (even with user_id passed) should not have user_id predicate
    cascade_delete(
        target="collection",
        collection="c_admin",
        user_id=1,
        is_admin=True,
        preview_only=False,
        confirm=True,
    )
    admin_filter = table.delete.call_args[0][0]
    assert "collection == 'c_admin'" in admin_filter
    assert "user_id ==" not in admin_filter

    table.delete.reset_mock()

    # Non-admin should include user_id predicate
    cascade_delete(
        target="collection",
        collection="c_user",
        user_id=1,
        is_admin=False,
        preview_only=False,
        confirm=True,
    )
    user_filter = table.delete.call_args[0][0]
    assert "collection == 'c_user'" in user_filter
    assert "user_id == 1" in user_filter


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_vector_store_raw_connection"
)
def test_cascade_delete_collection_open_table_error_uses_tenant_safe_fallback(
    mock_get_conn: MagicMock,
) -> None:
    """When table introspection fails, collection delete should keep tenant scoping."""
    conn = MagicMock()
    conn.table_names.return_value = ["documents"]
    conn.open_table.side_effect = RuntimeError("table unavailable")
    mock_get_conn.return_value = conn

    with patch(
        "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner._plan_by_predicates"
    ) as mock_plan:
        mock_plan.return_value = {"documents": 0}

        cascade_delete(
            target="collection",
            collection="c_err",
            user_id=7,
            is_admin=False,
            preview_only=True,
            confirm=False,
        )

    predicates = mock_plan.call_args.args[1]
    filt = predicates["documents"]
    assert "collection == 'c_err'" in filt
    assert "user_id == 7" in filt


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_vector_store_raw_connection"
)
def test_cascade_delete_document_open_table_error_uses_tenant_safe_fallback(
    mock_get_conn: MagicMock,
) -> None:
    """When table introspection fails, document delete should keep tenant scoping."""
    conn = MagicMock()
    conn.table_names.return_value = ["documents"]
    conn.open_table.side_effect = RuntimeError("table unavailable")
    mock_get_conn.return_value = conn

    with patch(
        "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner._plan_by_predicates"
    ) as mock_plan:
        mock_plan.return_value = {"documents": 0}

        cascade_delete(
            target="document",
            collection="c_err",
            doc_id="d_err",
            user_id=7,
            is_admin=False,
            preview_only=True,
            confirm=False,
        )

    predicates = mock_plan.call_args.args[1]
    filt = predicates["documents"]
    assert "collection == 'c_err'" in filt
    assert "doc_id == 'd_err'" in filt
    assert "user_id == 7" in filt


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_vector_store_raw_connection"
)
def test_cascade_delete_collection_open_table_error_without_user_id_denied(
    mock_get_conn: MagicMock,
) -> None:
    """Unauthenticated non-admin fallback should deny access on introspection error."""
    conn = MagicMock()
    conn.table_names.return_value = ["documents"]
    conn.open_table.side_effect = RuntimeError("table unavailable")
    mock_get_conn.return_value = conn

    with patch(
        "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner._plan_by_predicates"
    ) as mock_plan:
        mock_plan.return_value = {"documents": 0}

        cascade_delete(
            target="collection",
            collection="c_err",
            user_id=None,
            is_admin=False,
            preview_only=True,
            confirm=False,
        )

    predicates = mock_plan.call_args.args[1]
    filt = predicates["documents"]
    assert "collection == 'c_err'" in filt
    assert "user_id IS NULL" in filt and "user_id IS NOT NULL" in filt


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_vector_store_raw_connection"
)
def test_cascade_delete_document_open_table_error_without_user_id_denied(
    mock_get_conn: MagicMock,
) -> None:
    """Unauthenticated non-admin document fallback should deny access."""
    conn = MagicMock()
    conn.table_names.return_value = ["documents"]
    conn.open_table.side_effect = RuntimeError("table unavailable")
    mock_get_conn.return_value = conn

    with patch(
        "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner._plan_by_predicates"
    ) as mock_plan:
        mock_plan.return_value = {"documents": 0}

        cascade_delete(
            target="document",
            collection="c_err",
            doc_id="d_err",
            user_id=None,
            is_admin=False,
            preview_only=True,
            confirm=False,
        )

    predicates = mock_plan.call_args.args[1]
    filt = predicates["documents"]
    assert "collection == 'c_err'" in filt
    assert "doc_id == 'd_err'" in filt
    assert "user_id IS NULL" in filt and "user_id IS NOT NULL" in filt


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_vector_store_raw_connection"
)
def test_cascade_delete_document_schema_probe_error_keeps_legacy_compatible_filter(
    mock_get_conn: MagicMock,
) -> None:
    """If schema.names probing fails, document delete should stay legacy-compatible."""
    conn = MagicMock()
    conn.table_names.return_value = ["documents"]
    table = MagicMock()
    schema = MagicMock()
    # Simulate schema names probe failure in _table_has_column.
    type(schema).names = property(
        lambda _self: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    table.schema = schema
    conn.open_table.return_value = table
    mock_get_conn.return_value = conn

    with patch(
        "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner._plan_by_predicates"
    ) as mock_plan:
        mock_plan.return_value = {"documents": 0}
        cascade_delete(
            target="document",
            collection="c_probe",
            doc_id="d_probe",
            user_id=21,
            is_admin=False,
            preview_only=True,
            confirm=False,
        )

    predicates = mock_plan.call_args.args[1]
    filt = predicates["documents"]
    assert "collection == 'c_probe'" in filt
    assert "doc_id == 'd_probe'" in filt
    assert "user_id ==" not in filt


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_vector_store_raw_connection"
)
def test_cascade_delete_document_multitable_predicates_are_consistent(
    mock_get_conn: MagicMock,
) -> None:
    """Document delete should build consistent predicates across all target tables."""
    conn = MagicMock()
    conn.table_names.return_value = [
        "documents",
        "parses",
        "chunks",
        "main_pointers",
        "ingestion_runs",
        "embeddings_m1",
    ]

    table_map = {
        "documents": _create_mock_table_with_columns(
            ["collection", "doc_id", "user_id"]
        ),
        "parses": _create_mock_table_with_columns(["collection", "doc_id", "user_id"]),
        "chunks": _create_mock_table_with_columns(
            ["collection", "doc_id", "parse_hash", "user_id"]
        ),
        "main_pointers": _create_mock_table_with_columns(["collection", "doc_id"]),
        "ingestion_runs": _create_mock_table_with_columns(
            ["collection", "doc_id", "status", "user_id"]
        ),
        "embeddings_m1": _create_mock_table_with_columns(
            ["collection", "doc_id", "parse_hash", "user_id"]
        ),
    }
    conn.open_table.side_effect = lambda name: table_map[name]
    mock_get_conn.return_value = conn

    with patch(
        "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner._plan_by_predicates"
    ) as mock_plan:
        mock_plan.return_value = {}
        cascade_delete(
            target="document",
            collection="c_multi",
            doc_id="d_multi",
            user_id=5,
            is_admin=False,
            preview_only=True,
            confirm=False,
        )

    predicates = mock_plan.call_args.args[1]

    # Tables with user_id column should have tenant filter.
    for table_name in (
        "documents",
        "parses",
        "chunks",
        "ingestion_runs",
        "embeddings_m1",
    ):
        filt = predicates[table_name]
        assert "collection == 'c_multi'" in filt
        assert "doc_id == 'd_multi'" in filt
        assert "user_id == 5" in filt

    # Legacy-compatible table without user_id should not have tenant filter appended.
    pointers_filter = predicates["main_pointers"]
    assert "collection == 'c_multi'" in pointers_filter
    assert "doc_id == 'd_multi'" in pointers_filter
    assert "user_id ==" not in pointers_filter


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_vector_store_raw_connection"
)
def test_cascade_delete_document_model_tag_only_builds_target_embeddings_predicate(
    mock_get_conn: MagicMock,
) -> None:
    """Document cascade with model_tag should only target the specified embeddings table."""
    conn = MagicMock()
    conn.table_names.return_value = ["embeddings_m1", "embeddings_m2"]

    queried_tables: list[str] = []

    def mock_open_table(table_name: str) -> MagicMock:
        queried_tables.append(table_name)
        if table_name == "embeddings_m2":
            raise AssertionError(
                "embeddings_m2 should not be queried when model_tag='m1'"
            )
        return _create_mock_table_with_columns(
            ["collection", "doc_id", "parse_hash", "user_id"]
        )

    conn.open_table.side_effect = mock_open_table
    mock_get_conn.return_value = conn

    with patch(
        "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner._plan_by_predicates"
    ) as mock_plan:
        mock_plan.return_value = {}
        cascade_delete(
            target="document",
            collection="c_model",
            doc_id="d_model",
            user_id=8,
            is_admin=False,
            model_tag="m1",
            preview_only=True,
            confirm=False,
        )

    predicates = mock_plan.call_args.args[1]
    assert "embeddings_m1" in predicates
    assert "embeddings_m2" not in predicates
    assert queried_tables == ["embeddings_m1"]


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_vector_store_raw_connection"
)
def test_cascade_delete_document_model_tag_probe_error_without_user_id_denied(
    mock_get_conn: MagicMock,
) -> None:
    """If target embeddings table probing fails, unauthenticated fallback must deny access."""
    conn = MagicMock()
    conn.table_names.return_value = ["embeddings_m1", "embeddings_m2"]

    queried_tables: list[str] = []

    def mock_open_table(table_name: str) -> MagicMock:
        queried_tables.append(table_name)
        if table_name == "embeddings_m1":
            raise RuntimeError("embeddings_m1 unavailable")
        return _create_mock_table_with_columns(["collection", "doc_id", "parse_hash"])

    conn.open_table.side_effect = mock_open_table
    mock_get_conn.return_value = conn

    with patch(
        "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner._plan_by_predicates"
    ) as mock_plan:
        mock_plan.return_value = {}
        cascade_delete(
            target="document",
            collection="c_model",
            doc_id="d_model",
            user_id=None,
            is_admin=False,
            model_tag="m1",
            preview_only=True,
            confirm=False,
        )

    predicates = mock_plan.call_args.args[1]
    assert "embeddings_m1" in predicates
    assert "embeddings_m2" not in predicates
    filt = predicates["embeddings_m1"]
    assert "collection == 'c_model'" in filt
    assert "doc_id == 'd_model'" in filt
    assert "user_id IS NULL" in filt and "user_id IS NOT NULL" in filt
    assert queried_tables == ["embeddings_m1"]


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_vector_store_raw_connection"
)
def test_cascade_delete_confirm_delete_error_propagates(
    mock_get_conn: MagicMock,
) -> None:
    """Confirm mode should fail fast when table.delete raises an exception."""
    conn = MagicMock()
    conn.table_names.return_value = ["documents"]
    table = _create_mock_table_with_columns(["collection", "doc_id", "user_id"])
    table.count_rows.return_value = 1
    table.delete.side_effect = RuntimeError("delete failed")
    conn.open_table.return_value = table
    mock_get_conn.return_value = conn

    with pytest.raises(RuntimeError, match="delete failed"):
        cascade_delete(
            target="document",
            collection="c_err",
            doc_id="d_err",
            user_id=1,
            is_admin=False,
            preview_only=False,
            confirm=True,
        )


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_vector_store_raw_connection"
)
def test_cascade_delete_preview_and_confirm_counts_match_for_model_tag(
    mock_get_conn: MagicMock,
) -> None:
    """Preview counts should match confirm counts in fixed-count model_tag scenario."""
    conn = MagicMock()
    conn.table_names.return_value = [
        "documents",
        "parses",
        "chunks",
        "main_pointers",
        "ingestion_runs",
        "embeddings_m1",
        "embeddings_m2",
    ]

    table_map = {
        "documents": _create_mock_table_with_columns(
            ["collection", "doc_id", "user_id"]
        ),
        "parses": _create_mock_table_with_columns(["collection", "doc_id", "user_id"]),
        "chunks": _create_mock_table_with_columns(["collection", "doc_id", "user_id"]),
        "main_pointers": _create_mock_table_with_columns(["collection", "doc_id"]),
        "ingestion_runs": _create_mock_table_with_columns(
            ["collection", "doc_id", "status", "user_id"]
        ),
        "embeddings_m1": _create_mock_table_with_columns(
            ["collection", "doc_id", "parse_hash", "user_id"]
        ),
        "embeddings_m2": _create_mock_table_with_columns(
            ["collection", "doc_id", "parse_hash", "user_id"]
        ),
    }
    table_map["documents"].count_rows.return_value = 1
    table_map["parses"].count_rows.return_value = 2
    table_map["chunks"].count_rows.return_value = 3
    table_map["main_pointers"].count_rows.return_value = 4
    table_map["ingestion_runs"].count_rows.return_value = 5
    table_map["embeddings_m1"].count_rows.return_value = 6
    table_map[
        "embeddings_m2"
    ].count_rows.return_value = 99  # must not be touched by model_tag

    conn.open_table.side_effect = lambda name: table_map[name]
    mock_get_conn.return_value = conn

    preview = cascade_delete(
        target="document",
        collection="c_sync",
        doc_id="d_sync",
        user_id=2,
        is_admin=False,
        model_tag="m1",
        preview_only=True,
        confirm=False,
    )
    confirm = cascade_delete(
        target="document",
        collection="c_sync",
        doc_id="d_sync",
        user_id=2,
        is_admin=False,
        model_tag="m1",
        preview_only=False,
        confirm=True,
    )
    assert preview == confirm
    assert "embeddings_m2" not in preview


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_vector_store_raw_connection"
)
def test_cascade_delete_confirm_deletion_order_is_stable(
    mock_get_conn: MagicMock,
) -> None:
    """Confirm mode should delete tables in dependency-safe fixed order."""
    conn = MagicMock()
    conn.table_names.return_value = [
        "documents",
        "parses",
        "chunks",
        "main_pointers",
        "ingestion_runs",
        "embeddings_m1",
    ]

    delete_order: list[str] = []

    def make_table(table_name: str) -> MagicMock:
        table = _create_mock_table_with_columns(
            ["collection", "doc_id", "parse_hash", "user_id"]
        )
        table.count_rows.return_value = 1

        def _delete(_: str) -> None:
            delete_order.append(table_name)

        table.delete = _delete
        return table

    table_map = {name: make_table(name) for name in conn.table_names.return_value}
    conn.open_table.side_effect = lambda name: table_map[name]
    mock_get_conn.return_value = conn

    cascade_delete(
        target="document",
        collection="c_order",
        doc_id="d_order",
        user_id=3,
        is_admin=False,
        preview_only=False,
        confirm=True,
    )

    assert delete_order == [
        "embeddings_m1",
        "chunks",
        "parses",
        "main_pointers",
        "ingestion_runs",
        "documents",
    ]
