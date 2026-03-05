"""Tests for search_sparse functionality.

This module tests the sparse (FTS) search implementation:
- search_sparse main function
- Integration with LanceDB and index management
"""

import importlib
from typing import List
from unittest.mock import Mock, patch

import pandas as pd

from xagent.core.tools.core.RAG_tools.core.schemas import (
    SearchFallbackAction,
    SearchResult,
    SearchWarning,
    SparseSearchResponse,
)

search_sparse_module = importlib.import_module(
    "xagent.core.tools.core.RAG_tools.retrieval.search_sparse"
)


class TestSearchSparse:
    """Test search_sparse main function."""

    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_connection_from_env"
    )
    @patch("xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_index_manager")
    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.build_lancedb_filter_expression"
    )
    def test_search_sparse_success_no_filters(
        self,
        mock_build_filter: Mock,
        mock_get_index_manager: Mock,
        mock_get_conn: Mock,
    ) -> None:
        """Test successful sparse search with collection filter only (KB isolation)."""
        # Mock connection and table
        mock_conn = Mock()
        mock_table = Mock()
        mock_table.name = "embeddings_test_model"  # Set the table name
        mock_get_conn.return_value = mock_conn
        mock_conn.open_table.return_value = mock_table  # Ensure open_table succeeds

        # Mock index manager
        mock_index_manager = Mock()
        mock_index_manager.check_and_create_index.return_value = (
            "index_ready",
            "Index ready",
        )
        mock_index_manager.get_fts_index_status.return_value = True
        mock_get_index_manager.return_value = mock_index_manager

        # Collection filter is always applied for KB isolation (Issue #72)
        mock_build_filter.return_value = "collection == 'test_col'"

        # Mock search results; chain: search() -> limit() -> where() -> to_pandas()
        mock_results_df = pd.DataFrame(
            [
                {
                    "doc_id": "doc1",
                    "chunk_id": "chunk1",
                    "text": "test content one",
                    "_score": 0.9,
                    "parse_hash": "hash1",
                    "created_at": pd.Timestamp.now(),
                }
            ]
        )
        mock_search = Mock()
        mock_limit = Mock()
        mock_where = Mock()
        mock_table.search.return_value = mock_search
        mock_search.limit.return_value = mock_limit
        mock_limit.where.return_value = mock_where
        mock_where.to_pandas.return_value = mock_results_df

        response = search_sparse_module.search_sparse(
            collection="test_col",
            model_tag="test_model",
            query_text="content",
            top_k=1,
            user_id=None,
            is_admin=True,
        )

        assert isinstance(response, SparseSearchResponse)
        assert response.status == "success"
        assert response.total_count == 1
        assert response.fts_enabled is True
        assert len(response.results) == 1
        assert response.results[0].doc_id == "doc1"
        assert response.results[0].text == "test content one"
        # Score is normalized from TF-IDF to similarity score (0-1 range)
        assert abs(response.results[0].score - 0.4736842105263158) < 1e-10
        assert not response.warnings

        # Verify calls: collection filter must be applied for KB isolation
        mock_get_conn.assert_called_once()
        mock_conn.open_table.assert_called_once_with("embeddings_test_model")
        mock_get_index_manager.assert_called_once()
        mock_build_filter.assert_called_once_with({"collection": "test_col"})
        mock_table.search.assert_called_once_with("content", query_type="fts")
        mock_search.limit.assert_called_once_with(1)
        mock_limit.where.assert_called_once()
        where_arg = mock_limit.where.call_args[0][0]
        assert "collection" in where_arg.lower() or "test_col" in where_arg

    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_connection_from_env"
    )
    @patch("xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_index_manager")
    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.build_lancedb_filter_expression"
    )
    def test_search_sparse_with_filters(
        self, mock_build_filter: Mock, mock_get_index_manager: Mock, mock_get_conn: Mock
    ) -> None:
        """Test sparse search with filters."""
        with patch.object(
            search_sparse_module, "_substring_fallback", return_value=[]
        ) as mock_fallback:
            # Mock connection and table
            mock_conn = Mock()
            mock_table = Mock()
            mock_table.name = "embeddings_test_model"  # Set the table name
            mock_get_conn.return_value = mock_conn
            mock_conn.open_table.return_value = mock_table

            mock_index_manager = Mock()
            mock_index_manager.check_and_create_index.return_value = (
                "index_ready",
                "Index ready",
            )
            mock_index_manager.get_fts_index_status.return_value = True
            mock_get_index_manager.return_value = mock_index_manager

            mock_results_df = pd.DataFrame([])
            mock_search = Mock()
            mock_limit = Mock()
            mock_where = Mock()

            mock_table.search.return_value = mock_search
            mock_search.limit.return_value = mock_limit
            mock_limit.where.return_value = mock_where
            mock_where.to_pandas.return_value = mock_results_df

            filters = {"doc_id": "filtered_doc", "collection": "test_col"}
            expected_filter_clause = (
                "doc_id = 'filtered_doc' AND collection = 'test_col'"
            )
            # Collection filter first, then custom filters (Issue #72)
            mock_build_filter.side_effect = [
                "collection == 'test_col'",
                expected_filter_clause,
            ]

            response = search_sparse_module.search_sparse(
                collection="test_col",
                model_tag="test_model",
                query_text="filtered content",
                top_k=5,
                filters=filters,
                user_id=None,
                is_admin=True,
            )

            assert response.status == "success"
            assert response.total_count == 0
            assert len(response.results) == 0
            assert response.warnings == []

            mock_fallback.assert_called_once()
            mock_get_conn.assert_called_once()
            mock_conn.open_table.assert_called_once_with("embeddings_test_model")
            mock_get_index_manager.assert_called_once()
            mock_index_manager.get_fts_index_status.assert_called_once_with(mock_table)
            mock_build_filter.assert_any_call({"collection": "test_col"})
            mock_build_filter.assert_any_call(filters)
            mock_table.search.assert_called_once_with(
                "filtered content", query_type="fts"
            )
            mock_search.limit.assert_called_once_with(5)
            mock_limit.where.assert_called_once()
            where_arg = mock_limit.where.call_args[0][0]
            assert expected_filter_clause in where_arg
            mock_where.to_pandas.assert_called_once()

    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_connection_from_env"
    )
    @patch("xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_index_manager")
    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.build_lancedb_filter_expression"
    )
    def test_search_sparse_applies_collection_filter(
        self,
        mock_build_filter: Mock,
        mock_get_index_manager: Mock,
        mock_get_conn: Mock,
    ) -> None:
        """Test that search_sparse always applies collection filter for KB isolation (Issue #72)."""
        with patch.object(search_sparse_module, "_substring_fallback", return_value=[]):
            mock_conn = Mock()
            mock_table = Mock()
            mock_get_conn.return_value = mock_conn
            mock_conn.open_table.return_value = mock_table
            mock_index_manager = Mock()
            mock_index_manager.check_and_create_index.return_value = (
                "index_ready",
                "Index ready",
            )
            mock_index_manager.get_fts_index_status.return_value = True
            mock_get_index_manager.return_value = mock_index_manager
            mock_build_filter.return_value = "collection == 'my_kb'"
            mock_search = Mock()
            mock_limit = Mock()
            mock_where = Mock()
            mock_table.search.return_value = mock_search
            mock_search.limit.return_value = mock_limit
            mock_limit.where.return_value = mock_where
            mock_where.to_pandas.return_value = pd.DataFrame()

            search_sparse_module.search_sparse(
                collection="my_kb",
                model_tag="test_model",
                query_text="query",
                top_k=5,
                user_id=None,
                is_admin=True,
            )

            mock_build_filter.assert_called_once_with({"collection": "my_kb"})
            mock_limit.where.assert_called_once()

    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_connection_from_env"
    )
    @patch("xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_index_manager")
    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.build_lancedb_filter_expression"
    )
    def test_search_sparse_fts_index_missing(
        self,
        mock_build_filter: Mock,
        mock_get_index_manager: Mock,
        mock_get_conn: Mock,
    ) -> None:
        """Test sparse search when FTS index is missing."""
        with patch.object(search_sparse_module, "_substring_fallback", return_value=[]):
            mock_conn = Mock()
            mock_table = Mock()
            mock_get_conn.return_value = mock_conn
            mock_conn.open_table.return_value = mock_table

            mock_index_manager = Mock()
            mock_index_manager.check_and_create_index.return_value = (
                "index_ready",
                "Index ready",
            )
            mock_index_manager.get_fts_index_status.return_value = False
            mock_get_index_manager.return_value = mock_index_manager

            mock_build_filter.return_value = "collection == 'test_col'"
            mock_search = Mock()
            mock_limit = Mock()
            mock_where = Mock()
            mock_table.search.return_value = mock_search
            mock_search.limit.return_value = mock_limit
            mock_limit.where.return_value = mock_where
            mock_where.to_pandas.return_value = pd.DataFrame()

            response = search_sparse_module.search_sparse(
                collection="test_col",
                model_tag="test_model",
                query_text="query",
                top_k=1,
                user_id=None,
                is_admin=True,
            )

            assert response.status == "success"
            assert response.fts_enabled is False
            assert any(w.code == "FTS_INDEX_MISSING" for w in response.warnings)

            mock_get_conn.assert_called_once()
            mock_conn.open_table.assert_called_once_with("embeddings_test_model")
            mock_get_index_manager.assert_called_once()
            mock_index_manager.get_fts_index_status.assert_called_once_with(mock_table)
            mock_table.search.assert_called_once_with("query", query_type="fts")
            mock_search.limit.assert_called_once_with(1)

    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_connection_from_env"
    )
    @patch("xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_index_manager")
    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.build_lancedb_filter_expression"
    )
    def test_search_sparse_readonly_mode(
        self,
        mock_build_filter: Mock,
        mock_get_index_manager: Mock,
        mock_get_conn: Mock,
    ) -> None:
        """Test sparse search in readonly mode."""
        with patch.object(search_sparse_module, "_substring_fallback", return_value=[]):
            mock_conn = Mock()
            mock_table = Mock()
            mock_get_conn.return_value = mock_conn
            mock_conn.open_table.return_value = mock_table

            mock_index_manager = Mock()
            mock_index_manager.check_and_create_index.return_value = (
                "readonly",
                "Readonly mode",
            )
            mock_index_manager.get_fts_index_status.return_value = False
            mock_get_index_manager.return_value = mock_index_manager

            mock_build_filter.return_value = "collection == 'test_col'"
            mock_search = Mock()
            mock_limit = Mock()
            mock_where = Mock()
            mock_table.search.return_value = mock_search
            mock_search.limit.return_value = mock_limit
            mock_limit.where.return_value = mock_where
            mock_where.to_pandas.return_value = pd.DataFrame()

            response = search_sparse_module.search_sparse(
                collection="test_col",
                model_tag="test_model",
                query_text="query",
                top_k=1,
                readonly=True,
                user_id=None,
                is_admin=True,
            )

            assert response.status == "success"
            assert response.fts_enabled is False
            assert any(w.code == "READONLY_MODE" for w in response.warnings)

            mock_get_conn.assert_called_once()
            mock_conn.open_table.assert_called_once_with("embeddings_test_model")
            mock_get_index_manager.assert_called_once()
            mock_index_manager.get_fts_index_status.assert_called_once_with(mock_table)
            mock_table.search.assert_called_once_with("query", query_type="fts")
            mock_search.limit.assert_called_once_with(1)

    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_connection_from_env"
    )
    def test_search_sparse_database_error(self, mock_get_conn: Mock) -> None:
        """Test error handling during database operation."""
        mock_conn = Mock()
        mock_get_conn.return_value = mock_conn
        # Simulate open_table failure
        db_exception_message = "DB connection failed"
        mock_conn.open_table.side_effect = Exception(db_exception_message)

        response = search_sparse_module.search_sparse(
            collection="test_col",
            model_tag="test_model",
            query_text="query",
            top_k=1,
        )

        assert response.status == "failed"
        assert response.total_count == 0
        assert len(response.results) == 0
        assert len(response.warnings) == 1
        assert response.warnings[0].code == "FTS_SEARCH_FAILED"
        # Check for the wrapped error message
        assert (
            f"An unexpected error occurred during sparse search: {db_exception_message}"
            in response.warnings[0].message
        )

        # Verify calls
        mock_get_conn.assert_called_once()
        mock_conn.open_table.assert_called_once_with("embeddings_test_model")

    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_connection_from_env"
    )
    @patch("xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_index_manager")
    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.build_lancedb_filter_expression"
    )
    def test_search_sparse_empty_results(
        self,
        mock_build_filter: Mock,
        mock_get_index_manager: Mock,
        mock_get_conn: Mock,
    ) -> None:
        """Test sparse search returning no results."""
        with patch.object(search_sparse_module, "_substring_fallback", return_value=[]):
            mock_conn = Mock()
            mock_table = Mock()
            mock_get_conn.return_value = mock_conn
            mock_conn.open_table.return_value = mock_table

            mock_index_manager = Mock()
            mock_index_manager.check_and_create_index.return_value = (
                "index_ready",
                "Index ready",
            )
            mock_index_manager.get_fts_index_status.return_value = True
            mock_get_index_manager.return_value = mock_index_manager
            mock_build_filter.return_value = "collection == 'test_col'"
            mock_search = Mock()
            mock_limit = Mock()
            mock_where = Mock()
            mock_table.search.return_value = mock_search
            mock_search.limit.return_value = mock_limit
            mock_limit.where.return_value = mock_where
            mock_where.to_pandas.return_value = pd.DataFrame()

            response = search_sparse_module.search_sparse(
                collection="test_col",
                model_tag="test_model",
                query_text="no matches",
                top_k=5,
                user_id=None,
                is_admin=True,
            )

            assert response.status == "success"
            assert response.total_count == 0
            assert len(response.results) == 0
            assert response.warnings == []

            mock_get_conn.assert_called_once()
            mock_conn.open_table.assert_called_once_with("embeddings_test_model")
            mock_get_index_manager.assert_called_once()
            mock_table.search.assert_called_once_with("no matches", query_type="fts")
            mock_search.limit.assert_called_once_with(5)

    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_connection_from_env"
    )
    @patch("xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_index_manager")
    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.build_lancedb_filter_expression"
    )
    def test_search_sparse_triggers_fallback_with_results(
        self,
        mock_build_filter: Mock,
        mock_get_index_manager: Mock,
        mock_get_conn: Mock,
    ) -> None:
        """Ensure fallback populates results and emits an FTS warning."""

        def _fake_fallback(**kwargs: object) -> List[SearchResult]:
            current_warnings: List[SearchWarning] = kwargs["current_warnings"]  # type: ignore[assignment]
            current_warnings.append(
                SearchWarning(
                    code="FTS_FALLBACK",
                    message="Fallback executed",
                    fallback_action=SearchFallbackAction.PARTIAL_RESULTS,
                    affected_models=["test_model"],
                )
            )
            return [
                SearchResult(
                    doc_id="doc-fallback",
                    chunk_id="chunk-fallback",
                    text="matched text",
                    score=1.0,
                    parse_hash="hash",
                    model_tag="test_model",
                    created_at=pd.Timestamp.now(),
                )
            ]

        mock_conn = Mock()
        mock_table = Mock()
        mock_table.name = "embeddings_test_model"  # Set the table name
        mock_get_conn.return_value = mock_conn
        mock_conn.open_table.return_value = mock_table

        mock_index_manager = Mock()
        mock_index_manager.check_and_create_index.return_value = (
            "index_ready",
            "Index ready",
        )
        mock_index_manager.get_fts_index_status.return_value = True
        mock_get_index_manager.return_value = mock_index_manager
        mock_build_filter.return_value = "collection == 'test_col'"
        mock_search = Mock()
        mock_limit = Mock()
        mock_where = Mock()
        mock_table.search.return_value = mock_search
        mock_search.limit.return_value = mock_limit
        mock_limit.where.return_value = mock_where
        mock_where.to_pandas.return_value = pd.DataFrame()

        with patch.object(
            search_sparse_module, "_substring_fallback", side_effect=_fake_fallback
        ):
            response = search_sparse_module.search_sparse(
                collection="test_col",
                model_tag="test_model",
                query_text="fallback",
                top_k=3,
                user_id=None,
                is_admin=True,
            )

        assert response.status == "success"
        assert response.total_count == 1
        assert response.results[0].doc_id == "doc-fallback"
        assert any(w.code == "FTS_FALLBACK" for w in response.warnings)

    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_connection_from_env"
    )
    @patch("xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_index_manager")
    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.build_lancedb_filter_expression"
    )
    def test_search_sparse_score_clamping(
        self,
        mock_build_filter: Mock,
        mock_get_index_manager: Mock,
        mock_get_conn: Mock,
    ) -> None:
        """Test that sparse search scores are properly clamped to [0, 1] range."""
        # Mock connection and table
        mock_conn = Mock()
        mock_table = Mock()
        mock_table.name = "embeddings_test_model"
        mock_get_conn.return_value = mock_conn
        mock_conn.open_table.return_value = mock_table

        # Mock index manager
        mock_index_manager = Mock()
        mock_index_manager.check_and_create_index.return_value = (
            "index_ready",
            "Index ready",
        )
        mock_index_manager.get_fts_index_status.return_value = True
        mock_get_index_manager.return_value = mock_index_manager
        mock_build_filter.return_value = "collection == 'test_col'"
        mock_search = Mock()
        mock_limit = Mock()
        mock_where = Mock()
        mock_table.search.return_value = mock_search
        mock_search.limit.return_value = mock_limit
        mock_limit.where.return_value = mock_where

        # Create test data with a very high _score that would result in score > 1
        test_data = pd.DataFrame(
            {
                "doc_id": ["doc1"],
                "chunk_id": ["chunk1"],
                "text": ["test text"],
                "parse_hash": ["hash1"],
                "created_at": [pd.Timestamp.now()],
                "metadata": ['{"key": "value"}'],
                "_score": [100.0],  # score = 100/101 ≈ 0.99
            }
        )
        mock_where.to_pandas.return_value = test_data

        response = search_sparse_module.search_sparse(
            collection="test_col",
            model_tag="test_model",
            query_text="test",
            top_k=10,
            user_id=None,
            is_admin=True,
        )

        assert response.status == "success"
        assert len(response.results) == 1
        # Verify score is properly clamped and within [0, 1]
        assert 0.0 <= response.results[0].score <= 1.0
        # For _score = 100, expected score = 100 / (1 + 100) = 100/101 ≈ 0.9901
        expected_score = 100.0 / (1.0 + 100.0)
        assert abs(response.results[0].score - expected_score) < 0.0001

    def test_search_sparse_fts_fallback_warning_content(self) -> None:
        """Test that FTS_FALLBACK warning has correct content and fallback_action."""
        # Test the warning creation directly by calling _substring_fallback
        from xagent.core.tools.core.RAG_tools.retrieval.search_sparse import (
            _substring_fallback,
        )

        warnings: List[SearchWarning] = []

        # Mock table with some matching results to trigger the warning
        mock_table = Mock()
        mock_batch = Mock()
        mock_batch.to_pandas.return_value = pd.DataFrame(
            {
                "collection": ["test_col"],
                "doc_id": ["doc1"],
                "chunk_id": ["chunk1"],
                "text": ["test query content"],
                "parse_hash": ["hash1"],
                "created_at": [pd.Timestamp.now()],
                "metadata": ['{"key": "value"}'],
            }
        )
        mock_table.to_batches.return_value = [mock_batch]

        results = _substring_fallback(
            table=mock_table,
            collection="test_col",
            query_text="test query",
            model_tag="test_model",
            top_k=5,
            filters=None,
            current_warnings=warnings,
        )

        # Verify results were found and warning was added
        assert len(results) > 0
        assert len(warnings) == 1
        warning = warnings[0]

        assert warning.code == "FTS_FALLBACK"
        assert warning.fallback_action == SearchFallbackAction.BRUTE_FORCE
        assert warning.affected_models == ["test_model"]

        # Verify detailed message content
        assert "Full-text index returned no matches" in warning.message
        assert "used substring search fallback" in warning.message
        assert "Check FTS tokenizer configuration" in warning.message
        assert "update LanceDB to ensure proper tokenisation" in warning.message
