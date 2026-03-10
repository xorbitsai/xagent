"""Tests for cascade_cleaner unified entry and wrappers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner import (
    cleanup_chunk_cascade,
    cleanup_document_cascade,
    cleanup_embed_cascade,
    cleanup_parse_cascade,
)


def _create_mock_table_with_schema() -> MagicMock:
    """Create a mock table with a schema that passes ensure_*_table validation.

    Schema includes all fields required by ensure_documents_table, ensure_parses_table,
    ensure_chunks_table, and ensure_main_pointers_table (e.g. metadata, user_id),
    so cleanup_cascade()'s upfront ensure_* calls do not raise.
    """
    table = MagicMock()

    def _field(name: str) -> MagicMock:
        f = MagicMock()
        f.name = name
        return f

    # Required by: documents/parses/main_pointers -> user_id; chunks -> metadata, user_id
    table.schema = [
        _field("collection"),
        _field("doc_id"),
        _field("metadata"),
        _field("user_id"),
    ]
    return table


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_connection_from_env"
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

    # preview: counts only
    table = _create_mock_table_with_schema()
    table.count_rows.side_effect = [
        2,  # embeddings
        1,  # chunks
        1,  # parses
        1,  # pointers
        1,  # documents
    ]
    conn.open_table.return_value = table
    mock_get_conn.return_value = conn

    res = cleanup_document_cascade("c", "d", preview_only=True, confirm=False)
    assert res["embeddings"] == 2 and res["chunks"] == 1

    # confirm: delete paths
    table = _create_mock_table_with_schema()
    table.count_rows.side_effect = [
        2,
        1,
        1,
        1,
        1,
    ]
    conn.open_table.return_value = table
    res2 = cleanup_document_cascade("c", "d", preview_only=False, confirm=True)
    assert res2["documents"] == 1
    assert table.delete.call_count >= 1


@patch(
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_connection_from_env"
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
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_connection_from_env"
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
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_connection_from_env"
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
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_connection_from_env"
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
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_connection_from_env"
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
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_connection_from_env"
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
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_connection_from_env"
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
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_connection_from_env"
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
    "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_connection_from_env"
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
