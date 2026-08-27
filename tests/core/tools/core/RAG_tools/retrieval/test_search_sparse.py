"""Tests for search_sparse functionality.

This module tests the sparse (FTS) search implementation:
- search_sparse main function
- Integration with LanceDB and index management
"""

import importlib
from typing import List
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import DocumentValidationError
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

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"collection": "", "model_tag": "model", "query_text": "query", "top_k": 1},
            {"collection": "docs", "model_tag": "", "query_text": "query", "top_k": 1},
            {"collection": "docs", "model_tag": "model", "query_text": "", "top_k": 1},
            {
                "collection": "docs",
                "model_tag": "model",
                "query_text": "query",
                "top_k": 0,
            },
        ],
    )
    def test_search_sparse_validates_public_inputs(self, kwargs: dict) -> None:
        with pytest.raises(DocumentValidationError):
            search_sparse_module.search_sparse(**kwargs)

    @pytest.mark.asyncio
    async def test_search_sparse_async_validates_public_inputs(self) -> None:
        with pytest.raises(DocumentValidationError):
            await search_sparse_module.search_sparse_async(
                collection="docs",
                model_tag="model",
                query_text="",
                top_k=1,
            )

    def test_search_sparse_success_no_filters(self, make_handle, routed_facade) -> None:
        """Test successful sparse search with collection filter only (KB isolation).

        Re-pointed (#511): the public ``search_sparse`` runs the real handle FTS
        logic against a mock ``vector_index_store``. The store-call chain
        (open_embeddings_table -> create_index -> table.search(..fts).limit().where()
        -> to_pandas) is identical to the legacy free-function path, so the
        behavioral assertions are unchanged.
        """
        from xagent.core.tools.core.RAG_tools.core.schemas import IndexResult

        handle, mock_vector_store, _ = make_handle(collection="test_col")

        mock_table = Mock()
        mock_table.name = "embeddings_test_model"
        mock_table.list_indices.return_value = [
            Mock(index_type="FTS", columns=["text"])
        ]

        mock_vector_store.create_index.return_value = IndexResult(
            status="index_ready", advice=None, fts_enabled=True
        )
        mock_vector_store.build_filter_expression.return_value = (
            "collection == 'test_col'"
        )
        mock_vector_store.open_embeddings_table.return_value = (
            mock_table,
            "embeddings_test_model",
        )

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

        with routed_facade(search_sparse_module, handle):
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
        mock_vector_store.open_embeddings_table.assert_called_once_with("test_model")
        mock_vector_store.build_filter_expression.assert_called_once()
        mock_table.search.assert_called_once_with("content", query_type="fts")
        mock_search.limit.assert_called_once_with(1)
        mock_limit.where.assert_called_once()
        where_arg = mock_limit.where.call_args[0][0]
        assert "collection" in where_arg.lower() or "test_col" in where_arg

    def test_search_sparse_with_filters(self, make_handle, routed_facade) -> None:
        """Test sparse search with filters."""
        from xagent.core.tools.core.RAG_tools.core.schemas import IndexResult
        from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
            LanceDBCollectionHandle,
        )

        handle, mock_vector_store, _ = make_handle(collection="test_col")

        mock_table = Mock()
        mock_table.name = "embeddings_test_model"
        mock_table.list_indices.return_value = [
            Mock(index_type="FTS", columns=["text"])
        ]

        mock_vector_store.create_index.return_value = IndexResult(
            status="index_ready", advice=None, fts_enabled=True
        )
        mock_vector_store.build_filter_expression.return_value = (
            "doc_id = 'filtered_doc' AND collection = 'test_col'"
        )
        mock_vector_store.open_embeddings_table.return_value = (
            mock_table,
            "embeddings_test_model",
        )

        mock_results_df = pd.DataFrame([])
        mock_search = Mock()
        mock_limit = Mock()
        mock_where = Mock()
        mock_table.search.return_value = mock_search
        mock_search.limit.return_value = mock_limit
        mock_limit.where.return_value = mock_where
        mock_where.to_pandas.return_value = mock_results_df

        filters = {"doc_id": "filtered_doc", "collection": "test_col"}

        with patch.object(
            LanceDBCollectionHandle, "_substring_fallback", return_value=[]
        ) as mock_fallback:
            with routed_facade(search_sparse_module, handle):
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
        mock_vector_store.open_embeddings_table.assert_called_once_with("test_model")
        mock_vector_store.build_filter_expression.assert_called()
        mock_table.search.assert_called_once_with("filtered content", query_type="fts")
        mock_search.limit.assert_called_once_with(5)
        mock_limit.where.assert_called_once()
        mock_where.to_pandas.assert_called_once()

    def test_search_sparse_applies_collection_filter(
        self, make_handle, routed_facade
    ) -> None:
        """Test that search_sparse always applies collection filter for KB isolation (Issue #72)."""
        from xagent.core.tools.core.RAG_tools.core.schemas import IndexResult
        from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
            LanceDBCollectionHandle,
        )

        handle, mock_vector_store, _ = make_handle(collection="my_kb")

        mock_table = Mock()
        mock_table.list_indices.return_value = [
            Mock(index_type="FTS", columns=["text"])
        ]

        mock_vector_store.create_index.return_value = IndexResult(
            status="index_ready", advice=None, fts_enabled=True
        )
        mock_vector_store.build_filter_expression.return_value = "collection == 'my_kb'"
        mock_vector_store.open_embeddings_table.return_value = (
            mock_table,
            "embeddings_test_model",
        )

        mock_search = Mock()
        mock_limit = Mock()
        mock_where = Mock()
        mock_table.search.return_value = mock_search
        mock_search.limit.return_value = mock_limit
        mock_limit.where.return_value = mock_where
        mock_where.to_pandas.return_value = pd.DataFrame()

        with patch.object(
            LanceDBCollectionHandle, "_substring_fallback", return_value=[]
        ):
            with routed_facade(search_sparse_module, handle):
                search_sparse_module.search_sparse(
                    collection="my_kb",
                    model_tag="test_model",
                    query_text="query",
                    top_k=5,
                    user_id=None,
                    is_admin=True,
                )

        mock_vector_store.build_filter_expression.assert_called_once()
        mock_limit.where.assert_called_once()

    def test_search_sparse_fts_index_missing(self, make_handle, routed_facade) -> None:
        """Test sparse search when FTS index is missing."""
        from xagent.core.tools.core.RAG_tools.core.schemas import IndexResult
        from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
            LanceDBCollectionHandle,
        )

        handle, mock_vector_store, _ = make_handle(collection="test_col")

        mock_table = Mock()
        mock_table.list_indices.return_value = []

        mock_vector_store.create_index.return_value = IndexResult(
            status="index_ready",
            advice=None,
            fts_enabled=False,  # FTS not enabled
        )
        mock_vector_store.build_filter_expression.return_value = (
            "collection == 'test_col'"
        )
        mock_vector_store.open_embeddings_table.return_value = (
            mock_table,
            "embeddings_test_model",
        )

        mock_search = Mock()
        mock_limit = Mock()
        mock_where = Mock()
        mock_table.search.return_value = mock_search
        mock_search.limit.return_value = mock_limit
        mock_limit.where.return_value = mock_where
        mock_where.to_pandas.return_value = pd.DataFrame()

        with patch.object(
            LanceDBCollectionHandle, "_substring_fallback", return_value=[]
        ):
            with routed_facade(search_sparse_module, handle):
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

        mock_vector_store.open_embeddings_table.assert_called_once_with("test_model")
        mock_table.search.assert_called_once_with("query", query_type="fts")
        mock_search.limit.assert_called_once_with(1)

    def test_search_sparse_readonly_mode(self, make_handle, routed_facade) -> None:
        """Test sparse search in readonly mode."""
        from xagent.core.tools.core.RAG_tools.core.schemas import IndexResult
        from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
            LanceDBCollectionHandle,
        )

        handle, mock_vector_store, _ = make_handle(collection="test_col")

        mock_table = Mock()
        mock_table.list_indices.return_value = [
            Mock(index_type="FTS", columns=["text"])
        ]

        mock_vector_store.create_index.return_value = IndexResult(
            status="index_ready", advice=None, fts_enabled=True
        )
        mock_vector_store.build_filter_expression.return_value = (
            "collection == 'test_col'"
        )
        mock_vector_store.open_embeddings_table.return_value = (
            mock_table,
            "embeddings_test_model",
        )

        mock_search = Mock()
        mock_limit = Mock()
        mock_where = Mock()
        mock_table.search.return_value = mock_search
        mock_search.limit.return_value = mock_limit
        mock_limit.where.return_value = mock_where
        mock_where.to_pandas.return_value = pd.DataFrame()

        with patch.object(
            LanceDBCollectionHandle, "_substring_fallback", return_value=[]
        ):
            with routed_facade(search_sparse_module, handle):
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
        # FTS should be enabled since the table has the index
        assert response.fts_enabled is True
        assert any(w.code == "READONLY_MODE" for w in response.warnings)

        mock_vector_store.open_embeddings_table.assert_called_once_with("test_model")
        mock_table.search.assert_called_once_with("query", query_type="fts")
        mock_search.limit.assert_called_once_with(1)

    @patch(
        "xagent.core.tools.core.RAG_tools.utils.model_resolver.resolve_embedding_adapter"
    )
    def test_search_sparse_database_error(
        self, mock_resolve: Mock, make_handle, routed_facade
    ) -> None:
        """Test error handling during database operation."""
        handle, mock_vector_store, _ = make_handle(collection="test_col")
        db_exception_message = "DB connection failed"
        mock_vector_store.open_embeddings_table.side_effect = Exception(
            db_exception_message
        )

        mock_cfg = Mock()
        mock_cfg.model_name = "legacy_model"
        mock_resolve.return_value = (mock_cfg, object())

        with routed_facade(search_sparse_module, handle):
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

        # Verify calls - open_embeddings_table is called once (handles fallback internally)
        assert mock_vector_store.open_embeddings_table.call_count == 1
        mock_vector_store.open_embeddings_table.assert_called_once_with("test_model")

    def test_search_sparse_empty_results(self, make_handle, routed_facade) -> None:
        """Test sparse search returning no results."""
        from xagent.core.tools.core.RAG_tools.core.schemas import IndexResult
        from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
            LanceDBCollectionHandle,
        )

        handle, mock_vector_store, _ = make_handle(collection="test_col")

        mock_table = Mock()
        mock_table.list_indices.return_value = [
            Mock(index_type="FTS", columns=["text"])
        ]

        mock_vector_store.create_index.return_value = IndexResult(
            status="index_ready", advice=None, fts_enabled=True
        )
        mock_vector_store.build_filter_expression.return_value = (
            "collection == 'test_col'"
        )
        mock_vector_store.open_embeddings_table.return_value = (
            mock_table,
            "embeddings_test_model",
        )

        mock_search = Mock()
        mock_limit = Mock()
        mock_where = Mock()
        mock_table.search.return_value = mock_search
        mock_search.limit.return_value = mock_limit
        mock_limit.where.return_value = mock_where
        mock_where.to_pandas.return_value = pd.DataFrame()

        with patch.object(
            LanceDBCollectionHandle, "_substring_fallback", return_value=[]
        ):
            with routed_facade(search_sparse_module, handle):
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

        mock_vector_store.open_embeddings_table.assert_called_once_with("test_model")
        mock_table.search.assert_called_once_with("no matches", query_type="fts")
        mock_search.limit.assert_called_once_with(5)

    def test_search_sparse_triggers_fallback_with_results(
        self, make_handle, routed_facade
    ) -> None:
        """Ensure fallback populates results and emits an FTS warning."""
        from xagent.core.tools.core.RAG_tools.core.schemas import IndexResult
        from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
            LanceDBCollectionHandle,
        )

        def _fake_fallback(self, **kwargs: object) -> List[SearchResult]:
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

        handle, mock_vector_store, _ = make_handle(collection="test_col")

        mock_table = Mock()
        mock_table.list_indices.return_value = [
            Mock(index_type="FTS", columns=["text"])
        ]

        mock_vector_store.create_index.return_value = IndexResult(
            status="index_ready", advice=None, fts_enabled=True
        )
        mock_vector_store.build_filter_expression.return_value = (
            "collection == 'test_col'"
        )
        mock_vector_store.open_embeddings_table.return_value = (
            mock_table,
            "embeddings_test_model",
        )

        mock_search = Mock()
        mock_limit = Mock()
        mock_where = Mock()
        mock_table.search.return_value = mock_search
        mock_search.limit.return_value = mock_limit
        mock_limit.where.return_value = mock_where
        mock_where.to_pandas.return_value = pd.DataFrame()

        with patch.object(
            LanceDBCollectionHandle,
            "_substring_fallback",
            autospec=True,
            side_effect=_fake_fallback,
        ):
            with routed_facade(search_sparse_module, handle):
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

    def test_search_sparse_score_clamping(self, make_handle, routed_facade) -> None:
        """Test that sparse search scores are properly clamped to [0, 1] range."""
        from xagent.core.tools.core.RAG_tools.core.schemas import IndexResult

        handle, mock_vector_store, _ = make_handle(collection="test_col")

        mock_table = Mock()
        mock_table.list_indices.return_value = [
            Mock(index_type="FTS", columns=["text"])
        ]

        mock_vector_store.create_index.return_value = IndexResult(
            status="index_ready", advice=None, fts_enabled=True
        )
        mock_vector_store.build_filter_expression.return_value = (
            "collection == 'test_col'"
        )
        mock_vector_store.open_embeddings_table.return_value = (
            mock_table,
            "embeddings_test_model",
        )

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

        with routed_facade(search_sparse_module, handle):
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
        """Test that FTS_FALLBACK warning has correct content and fallback_action.

        The _substring_fallback logic now lives on LanceDBCollectionHandle.
        Call it via the handle instance to verify warning message content.
        """
        from xagent.core.tools.core.RAG_tools.core.schemas import SearchWarning
        from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
            LanceDBCollectionHandle,
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

        # Create a handle instance to call the method
        handle = LanceDBCollectionHandle.__new__(LanceDBCollectionHandle)
        results = handle._substring_fallback(
            table=mock_table,
            collection="test_col",
            query_text="test query",
            model_tag="test_model",
            top_k=5,
            filters=None,
            current_warnings=warnings,
            is_admin=True,
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
