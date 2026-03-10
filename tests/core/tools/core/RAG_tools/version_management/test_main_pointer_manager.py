"""Tests for main_pointer_manager functions.

These tests mock the LanceDB connection returned by get_connection_from_env
to validate basic CRUD behaviors without touching real storage.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from xagent.core.tools.core.RAG_tools.version_management.main_pointer_manager import (
    delete_main_pointer,
    get_main_pointer,
    list_main_pointers,
    set_main_pointer,
)

# Schema field names expected by ensure_main_pointers_table (must include user_id)
MAIN_POINTERS_SCHEMA_NAMES = [
    "collection",
    "doc_id",
    "step_type",
    "model_tag",
    "semantic_id",
    "technical_id",
    "created_at",
    "updated_at",
    "operator",
    "user_id",
]


def _mock_main_pointers_schema() -> MagicMock:
    """Return a mock schema for main_pointers table (includes user_id for ensure_main_pointers_table)."""
    schema = MagicMock()
    schema.names = list(MAIN_POINTERS_SCHEMA_NAMES)
    schema.__iter__ = lambda self: iter([SimpleNamespace(name=n) for n in self.names])
    return schema


class TestMainPointerManager:
    """Test cases for main_pointer_manager functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_env = os.environ.get("LANCEDB_DIR")
        os.environ["LANCEDB_DIR"] = self.temp_dir

    def teardown_method(self):
        """Clean up test fixtures."""
        # Restore original environment
        if self.original_env is not None:
            os.environ["LANCEDB_DIR"] = self.original_env
        elif "LANCEDB_DIR" in os.environ:
            del os.environ["LANCEDB_DIR"]

        # Clean up temp directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch(
        "xagent.core.tools.core.RAG_tools.version_management.main_pointer_manager.get_connection_from_env"
    )
    def test_get_main_pointer_not_found(
        self,
        mock_get_conn: MagicMock,
    ) -> None:
        conn = MagicMock()
        table = MagicMock()
        docs_table = MagicMock()
        table.schema = _mock_main_pointers_schema()
        table.search.return_value.where.return_value.to_pandas.return_value = (
            pd.DataFrame()
        )
        conn.open_table.side_effect = (
            lambda name: docs_table if name == "documents" else table
        )
        conn.table_names.return_value = ["main_pointers"]
        mock_get_conn.return_value = conn

        assert get_main_pointer("c", "d", "parse") is None

    @patch(
        "xagent.core.tools.core.RAG_tools.version_management.main_pointer_manager.get_connection_from_env"
    )
    def test_set_and_get_main_pointer_roundtrip(
        self,
        mock_get_conn: MagicMock,
    ) -> None:
        conn = MagicMock()
        table = MagicMock()
        docs_table = MagicMock()
        table.schema = _mock_main_pointers_schema()

        # Mock merge_insert chain
        mock_merge = MagicMock()
        table.merge_insert.return_value = mock_merge
        mock_merge.when_matched_update_all.return_value = mock_merge
        mock_merge.when_not_matched_insert_all.return_value = mock_merge

        row_df = pd.DataFrame(
            [
                {
                    "collection": "c",
                    "doc_id": "d",
                    "step_type": "parse",
                    "model_tag": "",
                    "semantic_id": "parse_x",
                    "technical_id": "abc",
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                    "operator": "tester",
                }
            ]
        )

        empty_df = pd.DataFrame()
        # set_main_pointer: may probe existence first (empty), then get_main_pointer reads row
        table.search.return_value.where.return_value.to_pandas.side_effect = [
            empty_df,
            row_df,
            row_df,
        ]
        conn.open_table.side_effect = (
            lambda name: docs_table if name == "documents" else table
        )
        conn.table_names.return_value = ["main_pointers"]
        mock_get_conn.return_value = conn

        # set should use merge_insert
        set_main_pointer(
            self.temp_dir,
            "c",
            "d",
            "parse",
            semantic_id="parse_x",
            technical_id="abc",
            operator="tester",
        )
        table.merge_insert.assert_called_once()
        mock_merge.execute.assert_called_once()

        # get should return the row
        result = get_main_pointer("c", "d", "parse")
        assert result is not None and result["technical_id"] == "abc"
        assert result["model_tag"] == ""

    @patch(
        "xagent.core.tools.core.RAG_tools.version_management.main_pointer_manager.get_connection_from_env"
    )
    @patch(
        "xagent.core.tools.core.RAG_tools.utils.user_permissions.UserPermissions.get_user_filter"
    )
    def test_list_and_delete_main_pointers(
        self,
        mock_get_user_filter: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        conn = MagicMock()
        table = MagicMock()
        docs_table = MagicMock()
        mock_get_user_filter.return_value = None
        df = pd.DataFrame(
            [
                {
                    "collection": "c",
                    "doc_id": "d",
                    "step_type": "parse",
                    "model_tag": None,
                    "semantic_id": "parse_x",
                    "technical_id": "abc",
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                    "operator": "tester",
                }
            ]
        )
        table.search.return_value.where.return_value.to_pandas.return_value = df
        table.search.return_value.where.return_value.count_rows.return_value = 1
        conn.open_table.side_effect = (
            lambda name: docs_table if name == "documents" else table
        )
        table.schema = _mock_main_pointers_schema()
        conn.table_names.return_value = ["main_pointers"]
        mock_get_conn.return_value = conn

        rows = list_main_pointers("c", doc_id="d")
        assert len(rows) == 1
        row = rows[0]
        assert row["model_tag"] == ""  # Normalized in list_main_pointers

        deleted = delete_main_pointer("c", "d", "parse")
        assert deleted is True
        table.delete.assert_called_once()

        # Verify delete filter expression includes NULL check (backward compatibility)
        call_args = table.delete.call_args
        filter_used = call_args[0][0] if call_args[0] else call_args[1].get("where")
        assert filter_used is not None
        assert "model_tag IS NULL" in filter_used

    @patch(
        "xagent.core.tools.core.RAG_tools.version_management.main_pointer_manager.get_connection_from_env"
    )
    def test_get_main_pointer_backward_compatibility(
        self,
        mock_get_conn: MagicMock,
    ) -> None:
        """Test that get_main_pointer can find records with NULL model_tag."""
        conn = MagicMock()
        table = MagicMock()
        docs_table = MagicMock()

        # Row with NULL model_tag
        df = pd.DataFrame(
            [
                {
                    "collection": "c",
                    "doc_id": "d",
                    "step_type": "parse",
                    "model_tag": None,
                    "semantic_id": "parse_x",
                    "technical_id": "abc",
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                    "operator": "tester",
                }
            ]
        )

        captured_filters = []

        def capture_where(filter_expr):
            captured_filters.append(filter_expr)
            mock_res = MagicMock()
            mock_res.to_pandas.return_value = df
            return mock_res

        table.search.return_value.where.side_effect = capture_where
        conn.open_table.side_effect = (
            lambda name: docs_table if name == "documents" else table
        )
        conn.table_names.return_value = ["main_pointers"]
        mock_get_conn.return_value = conn

        result = get_main_pointer("c", "d", "parse", model_tag=None)

        assert result is not None
        assert result["model_tag"] == ""  # Normalized to "" in result

        # Verify filter expression includes NULL check
        assert "(model_tag == '' OR model_tag IS NULL)" in captured_filters[0]

    @patch(
        "xagent.core.tools.core.RAG_tools.version_management.main_pointer_manager.get_connection_from_env"
    )
    def test_get_main_pointer_injection_attack_prevention(
        self,
        mock_get_conn: MagicMock,
    ) -> None:
        conn = MagicMock()
        conn.table_names.return_value = ["main_pointers"]
        table = MagicMock()
        docs_table = MagicMock()
        captured_filter = []

        def capture_where(filter_expr: str):
            captured_filter.append(filter_expr)
            mock_result = MagicMock()
            mock_result.to_pandas.return_value = pd.DataFrame()
            return mock_result

        table.schema = _mock_main_pointers_schema()
        table.search.return_value.where.side_effect = capture_where
        conn.open_table.side_effect = (
            lambda name: docs_table if name == "documents" else table
        )
        mock_get_conn.return_value = conn

        get_main_pointer(
            "coll'; DROP TABLE main_pointers; --",
            "doc' OR '1'='1",
            "parse' OR 'a'='a",
            model_tag="model'; DELETE FROM main_pointers; --",
        )

        filter_expr = captured_filter[0]
        assert "coll''; DROP TABLE main_pointers; --'" in filter_expr
        assert "doc'' OR ''1''=''1'" in filter_expr
        assert "parse'' OR ''a''=''a'" in filter_expr
        assert "model''; DELETE FROM main_pointers; --'" in filter_expr

    @patch(
        "xagent.core.tools.core.RAG_tools.version_management.main_pointer_manager.get_connection_from_env"
    )
    def test_set_main_pointer_preserves_created_at(
        self, mock_get_conn: MagicMock
    ) -> None:
        """Test that set_main_pointer preserves the original created_at timestamp on update."""
        conn = MagicMock()
        table = MagicMock()
        mock_merge = MagicMock()
        table.merge_insert.return_value = mock_merge
        mock_merge.when_matched_update_all.return_value = mock_merge
        mock_merge.when_not_matched_insert_all.return_value = mock_merge

        # Simulate existing record with an old timestamp
        old_time = pd.Timestamp("2023-01-01 12:00:00", tz="UTC")
        existing_df = pd.DataFrame(
            [
                {
                    "collection": "c",
                    "doc_id": "d",
                    "step_type": "parse",
                    "model_tag": "",
                    "semantic_id": "old_semantic",
                    "technical_id": "old_tech",
                    "created_at": old_time,
                    "updated_at": old_time,
                    "operator": "old_op",
                }
            ]
        )

        # Configure search to return the existing record
        table.search.return_value.where.return_value.to_pandas.return_value = (
            existing_df
        )
        conn.open_table.return_value = table
        conn.table_names.return_value = ["main_pointers"]
        mock_get_conn.return_value = conn

        set_main_pointer(
            self.temp_dir,
            "c",
            "d",
            "parse",
            semantic_id="new_semantic",
            technical_id="new_tech",
            operator="new_op",
        )

        # Check the DataFrame passed to execute
        mock_merge.execute.assert_called_once()
        call_args = mock_merge.execute.call_args
        df_passed = call_args[0][0]

        # Verify created_at matches the OLD time, not current time
        assert pd.Timestamp(df_passed.iloc[0]["created_at"]) == old_time
        # Verify other fields are updated
        assert df_passed.iloc[0]["semantic_id"] == "new_semantic"
        assert df_passed.iloc[0]["technical_id"] == "new_tech"

    @patch(
        "xagent.core.tools.core.RAG_tools.version_management.main_pointer_manager.get_connection_from_env"
    )
    def test_set_main_pointer_new_record_created_at(
        self, mock_get_conn: MagicMock
    ) -> None:
        """Test that set_main_pointer sets new created_at for new records."""
        conn = MagicMock()
        table = MagicMock()
        mock_merge = MagicMock()
        table.merge_insert.return_value = mock_merge
        mock_merge.when_matched_update_all.return_value = mock_merge
        mock_merge.when_not_matched_insert_all.return_value = mock_merge

        # Simulate NO existing record
        table.search.return_value.where.return_value.to_pandas.return_value = (
            pd.DataFrame()
        )
        table.schema = _mock_main_pointers_schema()
        conn.open_table.return_value = table
        conn.table_names.return_value = ["main_pointers"]
        mock_get_conn.return_value = conn

        before = pd.Timestamp.now(tz="UTC")
        set_main_pointer(
            self.temp_dir,
            "c",
            "d",
            "parse",
            semantic_id="new_semantic",
            technical_id="new_tech",
        )
        after = pd.Timestamp.now(tz="UTC")

        # Check the DataFrame passed to execute
        mock_merge.execute.assert_called_once()
        call_args = mock_merge.execute.call_args
        df_passed = call_args[0][0]

        created_at = pd.Timestamp(df_passed.iloc[0]["created_at"])
        # created_at should be roughly now (between before and after)
        assert before <= created_at <= after

    @patch(
        "xagent.core.tools.core.RAG_tools.version_management.main_pointer_manager.get_connection_from_env"
    )
    def test_set_main_pointer_normalizes_null_model_tag(
        self, mock_get_conn: MagicMock
    ) -> None:
        """Test that set_main_pointer attempts to update NULL model_tag to empty string."""
        conn = MagicMock()
        table = MagicMock()
        docs_table = MagicMock()
        mock_merge = MagicMock()
        table.merge_insert.return_value = mock_merge
        mock_merge.when_matched_update_all.return_value = mock_merge
        mock_merge.when_not_matched_insert_all.return_value = mock_merge

        # Simulate existing record with NULL model_tag
        existing_df = pd.DataFrame(
            [
                {
                    "collection": "c",
                    "doc_id": "d",
                    "step_type": "parse",
                    "model_tag": None,  # Legacy data
                    "semantic_id": "x",
                    "technical_id": "y",
                    "created_at": pd.Timestamp.now(),
                    "updated_at": pd.Timestamp.now(),
                    "operator": "op",
                }
            ]
        )

        # Configure search to return the existing NULL-tag record
        table.search.return_value.where.return_value.to_pandas.return_value = (
            existing_df
        )
        conn.open_table.side_effect = (
            lambda name: docs_table if name == "documents" else table
        )
        conn.table_names.return_value = ["main_pointers"]
        mock_get_conn.return_value = conn

        set_main_pointer(
            self.temp_dir,
            "c",
            "d",
            "parse",
            semantic_id="new_x",
            technical_id="new_y",
            # No model_tag provided, so it defaults to None -> normalized to ""
        )

        # Verify that update() was called to fix the NULL tag
        table.update.assert_called_once()
        call_args = table.update.call_args
        # Check that we are updating to empty string
        assert call_args[1]["values"] == {"model_tag": ""}
        # Check that we are targeting NULL records
        assert "model_tag IS NULL" in call_args[1]["where"]

    @patch(
        "xagent.core.tools.core.RAG_tools.version_management.main_pointer_manager.get_connection_from_env"
    )
    def test_set_main_pointer_always_attempts_normalization(
        self, mock_get_conn: MagicMock
    ) -> None:
        """Test that set_main_pointer safely attempts normalization whenever using empty model_tag."""
        conn = MagicMock()
        table = MagicMock()
        docs_table = MagicMock()
        mock_merge = MagicMock()
        table.merge_insert.return_value = mock_merge
        mock_merge.when_matched_update_all.return_value = mock_merge
        mock_merge.when_not_matched_insert_all.return_value = mock_merge

        # Simulate existing record with already NORMALIZED model_tag ("")
        existing_df = pd.DataFrame(
            [
                {
                    "collection": "c",
                    "doc_id": "d",
                    "step_type": "parse",
                    "model_tag": "",
                    "semantic_id": "x",
                    "technical_id": "y",
                    "created_at": pd.Timestamp.now(),
                    "updated_at": pd.Timestamp.now(),
                    "operator": "op",
                }
            ]
        )

        table.search.return_value.where.return_value.to_pandas.return_value = (
            existing_df
        )
        conn.open_table.side_effect = (
            lambda name: docs_table if name == "documents" else table
        )
        conn.table_names.return_value = ["main_pointers"]
        mock_get_conn.return_value = conn

        set_main_pointer(
            self.temp_dir,
            "c",
            "d",
            "parse",
            semantic_id="new_x",
            technical_id="new_y",
        )

        # Verify that update() IS called (it's a safe idempotent call)
        table.update.assert_called_once()
        # Merge insert should still proceed
        table.merge_insert.assert_called_once()
