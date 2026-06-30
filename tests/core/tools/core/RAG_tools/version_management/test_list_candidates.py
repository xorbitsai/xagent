"""Tests for list_candidates unified getter path."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import VersionManagementError
from xagent.core.tools.core.RAG_tools.core.schemas import StepType
from xagent.core.tools.core.RAG_tools.version_management.list_candidates import (
    list_candidates,
)


class TestListCandidates:
    """Test cases for list_candidates function."""

    def _patch_get_connection_from_env(self, mock_conn):
        """Patch the LanceDB connection at the store level (thread-safe).

        The call path is:
          list_candidates (public)
          → facade.list_candidates
          → coordinator.list_candidates_sync
          → _run_in_separate_loop  ← NEW THREAD
          → handle.list_candidates
          → store.list_version_candidate_rows
          → _vis_query_table (uses store._get_connection())

        ``unittest.mock.patch`` context managers do NOT propagate across threads,
        so patching ``get_connection_from_env`` at the provider level is
        ineffective here.  Instead we patch
        ``LanceDBVectorIndexStore._get_connection`` **at the class level** —
        class-attribute patches are visible in any thread that uses the class.

        The legacy ``get_vector_store_raw_connection`` patch is kept so that the
        fallback path (coordinator is None → ``_list_candidates_impl``) still
        works for tests that exercise error handling before any DB call.
        """
        from contextlib import ExitStack
        from unittest.mock import patch

        from xagent.core.tools.core.RAG_tools.storage import factory as _fac
        from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
            LanceDBVectorIndexStore,
        )

        # Reset singletons so a fresh store instance is created for this test.
        _fac.StorageFactory.get_factory().reset_all()

        stack = ExitStack()
        # Patch _get_connection at the CLASS level — propagates to all instances
        # in all threads (class dict lookup; not instance-local).
        stack.enter_context(
            patch.object(
                LanceDBVectorIndexStore,
                "_get_connection",
                return_value=mock_conn,
            )
        )
        # Keep legacy path patched for fallback / error-handling tests.
        import importlib

        list_candidates_module = importlib.import_module(
            "xagent.core.tools.core.RAG_tools.version_management.list_candidates"
        )
        stack.enter_context(
            patch.object(
                list_candidates_module,
                "get_vector_store_raw_connection",
                return_value=mock_conn,
            )
        )
        return stack

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

    def test_invalid_step_type(self):
        """Test that function raises error for invalid step_type."""
        mock_conn = MagicMock(spec=["table_names", "open_table"])

        with self._patch_get_connection_from_env(mock_conn):
            with pytest.raises(
                VersionManagementError,
                match="Invalid step_type string: 'invalid_step'.*Expected one of: 'parse', 'chunk', 'embed'",
            ):
                list_candidates("test_collection", "test_doc", "invalid_step")  # type: ignore

    def test_parse_candidates_empty(self):
        """Test list_candidates returns empty list when no parse candidates exist."""
        mock_conn = MagicMock(spec=["table_names", "open_table"])
        mock_conn.table_names.return_value = []

        with self._patch_get_connection_from_env(mock_conn):
            result = list_candidates("test_collection", "test_doc", StepType.PARSE)

            assert result["candidates"] == []
            assert result["total_count"] == 0
            assert result["returned_count"] == 0
            assert result["step_type"] == "parse"

    def test_parse_candidates_with_data(self):
        """Test list_candidates returns parse candidates when data exists."""
        mock_conn = MagicMock(spec=["table_names", "open_table"])
        mock_conn.table_names.return_value = ["parses"]
        mock_table = MagicMock()

        # Mock pandas result
        now = datetime.now()
        mock_data = pd.DataFrame(
            [
                {
                    "collection": "test_collection",
                    "doc_id": "test_doc",
                    "parse_hash": "hash1",
                    "parse_method": "unstructured",
                    "parser": "local:UnstructuredParser@v1",
                    "created_at": now + timedelta(milliseconds=1),
                },
                {
                    "collection": "test_collection",
                    "doc_id": "test_doc",
                    "parse_hash": "hash2",
                    "parse_method": "pypdf",
                    "parser": "local:PyPDFParser@v1",
                    "created_at": now,
                },
            ]
        )
        # Mock three-tier fallback: to_arrow() fails, fallback to to_pandas()
        mock_where = mock_table.search.return_value.where.return_value
        mock_where.to_arrow.side_effect = AttributeError("to_arrow not available")
        mock_where.to_list.side_effect = AttributeError("to_list not available")
        mock_where.to_pandas.return_value = mock_data
        mock_conn.open_table.return_value = mock_table

        with self._patch_get_connection_from_env(mock_conn):
            result = list_candidates("test_collection", "test_doc", StepType.PARSE)

            assert len(result["candidates"]) == 2
            assert result["total_count"] == 2
            assert result["returned_count"] == 2
            assert result["step_type"] == "parse"

            # Check first candidate (should be hash1 as it's newer)
            candidate1 = result["candidates"][0]
            assert candidate1["technical_id"] == "hash1"
            assert candidate1["state"] == "candidate"
            assert "parse_unstructured" in candidate1["semantic_id"]

    def test_chunk_candidates_with_data(self):
        """Test list_candidates returns chunk candidates when data exists."""
        mock_conn = MagicMock(spec=["table_names", "open_table"])
        mock_conn.table_names.return_value = ["chunks"]
        mock_table = MagicMock()

        # Mock pandas result
        base = datetime.now()
        mock_data = pd.DataFrame(
            [
                {
                    "collection": "test_collection",
                    "doc_id": "test_doc",
                    "parse_hash": "parse_hash1",
                    "chunk_id": "chunk1",
                    "text": "This is chunk 1",
                    "created_at": base + timedelta(milliseconds=2),
                },
                {
                    "collection": "test_collection",
                    "doc_id": "test_doc",
                    "parse_hash": "parse_hash1",
                    "chunk_id": "chunk2",
                    "text": "This is chunk 2",
                    "created_at": base + timedelta(milliseconds=1),
                },
                {
                    "collection": "test_collection",
                    "doc_id": "test_doc",
                    "parse_hash": "parse_hash2",
                    "chunk_id": "chunk3",
                    "text": "This is chunk 3",
                    "created_at": base,
                },
            ]
        )
        # Mock three-tier fallback: to_arrow() fails, fallback to to_pandas()
        mock_where = mock_table.search.return_value.where.return_value
        mock_where.to_arrow.side_effect = AttributeError("to_arrow not available")
        mock_where.to_list.side_effect = AttributeError("to_list not available")
        mock_where.to_pandas.return_value = mock_data
        mock_conn.open_table.return_value = mock_table

        with self._patch_get_connection_from_env(mock_conn):
            result = list_candidates("test_collection", "test_doc", StepType.CHUNK)

            assert len(result["candidates"]) == 2  # Grouped by parse_hash
            assert result["total_count"] == 2
            assert result["returned_count"] == 2
            assert result["step_type"] == "chunk"

            # Check first candidate (should be parse_hash1 as it has the newest chunk)
            candidate1 = result["candidates"][0]
            assert candidate1["technical_id"] == "parse_hash1"
            assert candidate1["stats"]["chunks_count"] == 2

    def test_chunk_candidates_none_text(self):
        """Regression for #709/#708: a chunk row whose ``text`` value is None
        must not crash. The pre-#708 ``len(row.get("text", ""))`` raised
        ``TypeError: object of type 'NoneType' has no len()`` (get() only
        substitutes the default when the key is *absent*, not when it is None).
        Also verifies avg_length treats the None row as length 0.
        """
        mock_conn = MagicMock(spec=["table_names", "open_table"])
        mock_conn.table_names.return_value = ["chunks"]

        mock_table = MagicMock()
        base = datetime.now()
        mock_data = pd.DataFrame(
            [
                {
                    "collection": "test_collection",
                    "doc_id": "test_doc",
                    "parse_hash": "parse_hash1",
                    "chunk_id": "chunk1",
                    "text": "0123456789",  # len 10
                    "created_at": base + timedelta(milliseconds=2),
                },
                {
                    "collection": "test_collection",
                    "doc_id": "test_doc",
                    "parse_hash": "parse_hash1",
                    "chunk_id": "chunk2",
                    "text": None,  # len treated as 0 -> avg (10+0)/2 = 5
                    "created_at": base + timedelta(milliseconds=1),
                },
                {
                    "collection": "test_collection",
                    "doc_id": "test_doc",
                    "parse_hash": "parse_hash2",
                    "chunk_id": "chunk3",
                    "text": "0123456789",  # len 10 -> avg 10
                    "created_at": base,
                },
            ]
        )
        mock_where = mock_table.search.return_value.where.return_value
        mock_where.to_arrow.side_effect = AttributeError("to_arrow not available")
        mock_where.to_list.side_effect = AttributeError("to_list not available")
        mock_where.to_pandas.return_value = mock_data
        mock_conn.open_table.return_value = mock_table

        with self._patch_get_connection_from_env(mock_conn):
            result = list_candidates("test_collection", "test_doc", StepType.CHUNK)

        assert len(result["candidates"]) == 2  # grouped by parse_hash
        by_hash = {c["technical_id"]: c for c in result["candidates"]}
        assert by_hash["parse_hash1"]["stats"]["chunks_count"] == 2
        assert by_hash["parse_hash1"]["stats"]["avg_length"] == 5
        assert by_hash["parse_hash2"]["stats"]["chunks_count"] == 1
        assert by_hash["parse_hash2"]["stats"]["avg_length"] == 10

    def test_embed_candidates_with_data(self):
        """Test list_candidates returns embed candidates when data exists."""
        mock_conn = MagicMock(spec=["table_names", "open_table"])
        mock_conn.table_names.return_value = [
            "embeddings_bge_large",
        ]

        # Mock embeddings table
        mock_table = MagicMock()
        mock_data = pd.DataFrame(
            [
                {
                    "collection": "test_collection",
                    "doc_id": "test_doc",
                    "model": "BAAI/bge-large-zh-v1.5",
                    "parse_hash": "parse_hash1",
                    "vector": [0.1, 0.2, 0.3],
                    "created_at": datetime.now(),
                },
                {
                    "collection": "test_collection",
                    "doc_id": "test_doc",
                    "model": "BAAI/bge-large-zh-v1.5",
                    "parse_hash": "parse_hash2",
                    "vector": [0.4, 0.5, 0.6],
                    "created_at": datetime.now(),
                },
            ]
        )
        # Mock three-tier fallback: to_arrow() fails, fallback to to_pandas()
        mock_where = mock_table.search.return_value.where.return_value
        mock_where.to_arrow.side_effect = AttributeError("to_arrow not available")
        mock_where.to_list.side_effect = AttributeError("to_list not available")
        mock_where.to_pandas.return_value = mock_data
        mock_conn.open_table.return_value = mock_table

        with self._patch_get_connection_from_env(mock_conn):
            result = list_candidates(
                "test_collection", "test_doc", StepType.EMBED, model_tag="bge_large"
            )

            assert len(result["candidates"]) == 2  # Two different parse_hash versions
            assert result["total_count"] == 2
            assert result["returned_count"] == 2
            assert result["step_type"] == "embed"

            # Check that we have both parse_hash versions
            technical_ids = [c["technical_id"] for c in result["candidates"]]
            assert "parse_hash1" in technical_ids
            assert "parse_hash2" in technical_ids
            # Check stats for both candidates
            for candidate in result["candidates"]:
                assert (
                    candidate["stats"]["upsert_count"] == 1
                )  # Each parse_hash has 1 row
                assert candidate["stats"]["vector_dim"] == 3

    def test_embed_candidates_numpy_vector(self):
        """Regression for #709/#708: a numpy-ndarray ``vector`` column must not
        crash candidate listing. The pre-#708 ``if vector:`` check raised
        ``ValueError: truth value of an array ... is ambiguous`` on np.ndarray.
        """
        mock_conn = MagicMock(spec=["table_names", "open_table"])
        mock_conn.table_names.return_value = ["embeddings_bge_large"]

        mock_table = MagicMock()
        mock_data = pd.DataFrame(
            [
                {
                    "collection": "test_collection",
                    "doc_id": "test_doc",
                    "model": "BAAI/bge-large-zh-v1.5",
                    "parse_hash": "parse_hash1",
                    "vector": np.array([0.1, 0.2, 0.3]),  # numpy, not list
                    "created_at": datetime.now(),
                },
                {
                    "collection": "test_collection",
                    "doc_id": "test_doc",
                    "model": "BAAI/bge-large-zh-v1.5",
                    "parse_hash": "parse_hash2",
                    "vector": np.array([0.4, 0.5, 0.6]),  # numpy, not list
                    "created_at": datetime.now(),
                },
            ]
        )
        # Force the to_pandas fallback (where vector columns come back as
        # np.ndarray); to_arrow/to_list raise as in the sibling tests.
        mock_where = mock_table.search.return_value.where.return_value
        mock_where.to_arrow.side_effect = AttributeError("to_arrow not available")
        mock_where.to_list.side_effect = AttributeError("to_list not available")
        mock_where.to_pandas.return_value = mock_data
        mock_conn.open_table.return_value = mock_table

        with self._patch_get_connection_from_env(mock_conn):
            result = list_candidates(
                "test_collection", "test_doc", StepType.EMBED, model_tag="bge_large"
            )

        assert len(result["candidates"]) == 2
        for candidate in result["candidates"]:
            assert candidate["stats"]["vector_dim"] == 3

    def test_state_filter(self):
        """Test that state filter works correctly."""
        mock_conn = MagicMock(spec=["table_names", "open_table"])
        mock_conn.table_names.return_value = ["parses"]
        mock_table = MagicMock()

        # Mock pandas result
        mock_data = pd.DataFrame(
            [
                {
                    "collection": "test_collection",
                    "doc_id": "test_doc",
                    "parse_hash": "hash1",
                    "parse_method": "unstructured",
                    "parser": "local:UnstructuredParser@v1",
                    "created_at": datetime.now(),
                }
            ]
        )
        # Mock three-tier fallback: to_arrow() fails, fallback to to_pandas()
        mock_where = mock_table.search.return_value.where.return_value
        mock_where.to_arrow.side_effect = AttributeError("to_arrow not available")
        mock_where.to_list.side_effect = AttributeError("to_list not available")
        mock_where.to_pandas.return_value = mock_data
        mock_conn.open_table.return_value = mock_table

        with self._patch_get_connection_from_env(mock_conn):
            result = list_candidates(
                "test_collection", "test_doc", StepType.PARSE, state="candidate"
            )

            assert len(result["candidates"]) == 1
            assert result["total_count"] == 1
            assert result["returned_count"] == 1
            assert result["filters"]["state"] == "candidate"

    def test_limit_filter(self):
        """Test that limit filter works correctly."""
        mock_conn = MagicMock(spec=["table_names", "open_table"])
        mock_conn.table_names.return_value = ["parses"]
        mock_table = MagicMock()

        # Mock pandas result with multiple entries
        mock_data = pd.DataFrame(
            [
                {
                    "collection": "test_collection",
                    "doc_id": "test_doc",
                    "parse_hash": f"hash{i}",
                    "parse_method": "unstructured",
                    "parser": "local:UnstructuredParser@v1",
                    "created_at": datetime.now(),
                }
                for i in range(5)
            ]
        )
        # Mock three-tier fallback: to_arrow() fails, fallback to to_pandas()
        mock_where = mock_table.search.return_value.where.return_value
        mock_where.to_arrow.side_effect = AttributeError("to_arrow not available")
        mock_where.to_list.side_effect = AttributeError("to_list not available")
        mock_where.to_pandas.return_value = mock_data
        mock_conn.open_table.return_value = mock_table

        with self._patch_get_connection_from_env(mock_conn):
            result = list_candidates(
                "test_collection", "test_doc", StepType.PARSE, limit=3
            )

            assert len(result["candidates"]) == 3
            assert result["total_count"] == 5  # Total before limit
            assert result["returned_count"] == 3  # Actually returned after limit
            assert result["filters"]["limit"] == 3

    def test_model_tag_filter_for_embeddings(self):
        """Test that model_tag filter works for embed step_type."""
        mock_conn = MagicMock(spec=["table_names", "open_table"])
        mock_conn.table_names.return_value = [
            "embeddings_bge_large",
            "embeddings_minilm",
        ]

        # Mock embeddings table
        mock_table = MagicMock()
        mock_data = pd.DataFrame(
            [
                {
                    "collection": "test_collection",
                    "doc_id": "test_doc",
                    "model": "BAAI/bge-large-zh-v1.5",
                    "parse_hash": "parse_hash1",
                    "vector": [0.1, 0.2, 0.3],
                    "created_at": datetime.now(),
                }
            ]
        )
        # Mock three-tier fallback: to_arrow() fails, fallback to to_pandas()
        mock_where = mock_table.search.return_value.where.return_value
        mock_where.to_arrow.side_effect = AttributeError("to_arrow not available")
        mock_where.to_list.side_effect = AttributeError("to_list not available")
        mock_where.to_pandas.return_value = mock_data
        mock_conn.open_table.return_value = mock_table

        with self._patch_get_connection_from_env(mock_conn):
            result = list_candidates(
                "test_collection", "test_doc", StepType.EMBED, model_tag="bge_large"
            )

            assert len(result["candidates"]) == 1
            assert result["total_count"] == 1
            assert result["returned_count"] == 1
            assert result["model_tag"] == "bge_large"

    def test_sort_before_limit(self):
        """Test that sorting happens before limit to get correct top-N results."""
        mock_conn = MagicMock(spec=["table_names", "open_table"])
        mock_conn.table_names.return_value = ["parses"]
        mock_table = MagicMock()

        # Create test data with specific timestamps
        base_time = datetime(2024, 1, 1)
        mock_data = pd.DataFrame(
            [
                {
                    "collection": "test_collection",
                    "doc_id": "test_doc",
                    "parse_hash": "hash_oldest",
                    "parse_method": "unstructured",
                    "parser": "local:UnstructuredParser@v1",
                    "created_at": base_time,  # Oldest
                },
                {
                    "collection": "test_collection",
                    "doc_id": "test_doc",
                    "parse_hash": "hash_middle",
                    "parse_method": "unstructured",
                    "parser": "local:UnstructuredParser@v1",
                    "created_at": base_time + timedelta(days=5),
                },
                {
                    "collection": "test_collection",
                    "doc_id": "test_doc",
                    "parse_hash": "hash_newer",
                    "parse_method": "unstructured",
                    "parser": "local:UnstructuredParser@v1",
                    "created_at": base_time + timedelta(days=7),
                },
                {
                    "collection": "test_collection",
                    "doc_id": "test_doc",
                    "parse_hash": "hash_newest",
                    "parse_method": "unstructured",
                    "parser": "local:UnstructuredParser@v1",
                    "created_at": base_time + timedelta(days=10),  # Newest
                },
                {
                    "collection": "test_collection",
                    "doc_id": "test_doc",
                    "parse_hash": "hash_second_newest",
                    "parse_method": "unstructured",
                    "parser": "local:UnstructuredParser@v1",
                    "created_at": base_time + timedelta(days=8),
                },
            ]
        )
        # Mock three-tier fallback: to_arrow() fails, fallback to to_pandas()
        mock_where = mock_table.search.return_value.where.return_value
        mock_where.to_arrow.side_effect = AttributeError("to_arrow not available")
        mock_where.to_list.side_effect = AttributeError("to_list not available")
        mock_where.to_pandas.return_value = mock_data
        mock_conn.open_table.return_value = mock_table

        with self._patch_get_connection_from_env(mock_conn):
            # Request top 3 with descending order
            result = list_candidates(
                "test_collection",
                "test_doc",
                StepType.PARSE,
                limit=3,
                order_by="created_at desc",
            )

            assert len(result["candidates"]) == 3
            assert result["total_count"] == 5  # Total before limit
            assert result["returned_count"] == 3  # After limit

            # Verify we got the NEWEST 3 items (not just the first 3 from DB)
            technical_ids = [c["technical_id"] for c in result["candidates"]]
            assert technical_ids[0] == "hash_newest"  # Most recent
            assert technical_ids[1] == "hash_second_newest"  # Second most recent
            assert technical_ids[2] == "hash_newer"  # Third most recent

            # These should NOT be in the result
            assert "hash_middle" not in technical_ids
            assert "hash_oldest" not in technical_ids

            # Verify timestamps are in descending order
            timestamps = [c["created_at"] for c in result["candidates"]]
            assert timestamps[0] > timestamps[1] > timestamps[2]

    def test_sql_injection_protection(self):
        """Test that list_candidates protects against SQL injection."""
        mock_conn = MagicMock(spec=["table_names", "open_table"])
        mock_conn.table_names.return_value = ["parses"]
        mock_table = MagicMock()

        # Mock empty pandas result for the malicious query
        mock_table.search.return_value.where.return_value.to_pandas.return_value = (
            pd.DataFrame()
        )
        mock_conn.open_table.return_value = mock_table

        malicious_doc_id = "test_doc' OR 1=1 --"
        collection_name = "test_collection"

        with self._patch_get_connection_from_env(mock_conn):
            result = list_candidates(collection_name, malicious_doc_id, StepType.PARSE)

            # Assert that the where clause was called with the correctly escaped string.
            # New path (_vis_query_table) uses skip_user_filter=True (is_admin=True), so
            # no user-access filter is appended — the expression is purely the two field
            # conditions joined by AND.
            expected_where_clause = (
                f"(collection == '{collection_name}')"
                f" AND (doc_id == 'test_doc'' OR 1=1 --')"
            )

            mock_table.search.assert_called_once()
            mock_table.search.return_value.where.assert_called_once_with(
                expected_where_clause
            )

            assert result["candidates"] == []
            assert result["total_count"] == 0
            assert result["returned_count"] == 0
            assert result["step_type"] == "parse"
