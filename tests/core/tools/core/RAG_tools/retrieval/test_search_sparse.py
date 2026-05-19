"""Tests for search_sparse functionality.

This module tests the sparse (FTS) search implementation:
- search_sparse main function
- Integration with VectorIndexStore (no raw LanceDB table.search in retrieval)
"""

import importlib
from typing import Any, List
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import (
    IndexResult,
    SearchFallbackAction,
    SearchResult,
    SearchWarning,
    SparseSearchResponse,
)

search_sparse_module = importlib.import_module(
    "xagent.core.tools.core.RAG_tools.retrieval.search_sparse"
)


def _mock_vector_store_for_sparse(
    *,
    fts_enabled: bool = True,
    fts_rows: List[dict[str, Any]],
) -> Mock:
    """Build a VectorIndexStore mock with create_index + search_fts_by_model wired."""
    mock_vector_store = Mock()
    mock_vector_store.create_index.return_value = IndexResult(
        status="index_ready",
        advice=None,
        fts_enabled=fts_enabled,
    )
    mock_vector_store.search_fts_by_model.return_value = fts_rows
    return mock_vector_store


class TestSearchSparse:
    """Test search_sparse main function."""

    def test_search_sparse_success_no_filters(self) -> None:
        """Test successful sparse search with collection filter only (KB isolation)."""
        fts_rows = [
            {
                "doc_id": "doc1",
                "chunk_id": "chunk1",
                "text": "test content one",
                "_score": 0.9,
                "parse_hash": "hash1",
                "created_at": pd.Timestamp.now(),
            }
        ]
        mock_vector_store = _mock_vector_store_for_sparse(fts_rows=fts_rows)

        with patch(
            "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_vector_index_store"
        ) as mock_get_vector_store:
            mock_get_vector_store.return_value = mock_vector_store

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
            assert abs(response.results[0].score - 0.4736842105263158) < 1e-10
            assert not response.warnings

            mock_vector_store.search_fts_by_model.assert_called_once()
            call_kw = mock_vector_store.search_fts_by_model.call_args.kwargs
            assert call_kw["model_tag"] == "test_model"
            assert call_kw["query_text"] == "content"
            assert call_kw["top_k"] == 1
            assert call_kw["user_id"] is None
            assert call_kw["is_admin"] is True
            assert call_kw["filters"] is not None

    def test_search_sparse_with_filters(self) -> None:
        """Test sparse search with filters."""
        with patch.object(
            search_sparse_module, "_substring_fallback", return_value=[]
        ) as mock_fallback:
            mock_vector_store = _mock_vector_store_for_sparse(fts_rows=[])

            with patch(
                "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_vector_index_store"
            ) as mock_get_vector_store:
                mock_get_vector_store.return_value = mock_vector_store

                filters = {"doc_id": "filtered_doc", "collection": "test_col"}

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
            mock_vector_store.search_fts_by_model.assert_called_once()
            assert mock_vector_store.search_fts_by_model.call_args.kwargs["top_k"] == 5

    def test_search_sparse_applies_collection_filter(self) -> None:
        """Test that search_sparse always applies collection filter for KB isolation (Issue #72)."""
        with patch.object(search_sparse_module, "_substring_fallback", return_value=[]):
            mock_vector_store = _mock_vector_store_for_sparse(fts_rows=[])

            with patch(
                "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_vector_index_store"
            ) as mock_get_vector_store:
                mock_get_vector_store.return_value = mock_vector_store

                search_sparse_module.search_sparse(
                    collection="my_kb",
                    model_tag="test_model",
                    query_text="query",
                    top_k=5,
                    user_id=None,
                    is_admin=True,
                )

            call_kw = mock_vector_store.search_fts_by_model.call_args.kwargs
            assert call_kw["filters"] is not None

    def test_search_sparse_fts_index_missing(self) -> None:
        """Test sparse search when FTS index is missing."""
        with patch.object(search_sparse_module, "_substring_fallback", return_value=[]):
            mock_vector_store = _mock_vector_store_for_sparse(
                fts_enabled=False, fts_rows=[]
            )

            with patch(
                "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_vector_index_store"
            ) as mock_get_vector_store:
                mock_get_vector_store.return_value = mock_vector_store

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
            mock_vector_store.search_fts_by_model.assert_called_once()

    def test_search_sparse_readonly_mode(self) -> None:
        """Test sparse search in readonly mode."""
        with patch.object(search_sparse_module, "_substring_fallback", return_value=[]):
            mock_vector_store = _mock_vector_store_for_sparse(fts_rows=[])

            with patch(
                "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_vector_index_store"
            ) as mock_get_vector_store:
                mock_get_vector_store.return_value = mock_vector_store

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
            assert response.fts_enabled is True
            assert any(w.code == "READONLY_MODE" for w in response.warnings)
            mock_vector_store.search_fts_by_model.assert_called_once()

    @patch(
        "xagent.core.tools.core.RAG_tools.utils.model_resolver.resolve_embedding_adapter"
    )
    def test_search_sparse_database_error(self, mock_resolve: Mock) -> None:
        """Test error handling during database operation."""
        mock_vector_store = Mock()
        db_exception_message = "DB connection failed"
        mock_vector_store.create_index.return_value = IndexResult(
            status="index_ready",
            advice=None,
            fts_enabled=True,
        )
        mock_vector_store.search_fts_by_model.side_effect = Exception(
            db_exception_message
        )

        mock_cfg = Mock()
        mock_cfg.model_name = "legacy_model"
        mock_resolve.return_value = (mock_cfg, object())

        with patch(
            "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_vector_index_store"
        ) as mock_get_vector_store:
            mock_get_vector_store.return_value = mock_vector_store

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
        assert (
            f"An unexpected error occurred during sparse search: {db_exception_message}"
            in response.warnings[0].message
        )

        mock_vector_store.search_fts_by_model.assert_called_once()

    def test_search_sparse_empty_results(self) -> None:
        """Test sparse search returning no results."""
        with patch.object(search_sparse_module, "_substring_fallback", return_value=[]):
            mock_vector_store = _mock_vector_store_for_sparse(fts_rows=[])

            with patch(
                "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_vector_index_store"
            ) as mock_get_vector_store:
                mock_get_vector_store.return_value = mock_vector_store

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
            mock_vector_store.search_fts_by_model.assert_called_once()

    def test_search_sparse_triggers_fallback_with_results(self) -> None:
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

        mock_vector_store = _mock_vector_store_for_sparse(fts_rows=[])

        with patch.object(
            search_sparse_module, "_substring_fallback", side_effect=_fake_fallback
        ):
            with patch(
                "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_vector_index_store"
            ) as mock_get_vector_store:
                mock_get_vector_store.return_value = mock_vector_store

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

    def test_search_sparse_score_clamping(self) -> None:
        """Test that sparse search scores are properly clamped to [0, 1] range."""
        fts_rows = [
            {
                "doc_id": "doc1",
                "chunk_id": "chunk1",
                "text": "test text",
                "parse_hash": "hash1",
                "created_at": pd.Timestamp.now(),
                "metadata": '{"key": "value"}',
                "_score": 100.0,
            }
        ]
        mock_vector_store = _mock_vector_store_for_sparse(fts_rows=fts_rows)

        with patch(
            "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_vector_index_store"
        ) as mock_get_vector_store:
            mock_get_vector_store.return_value = mock_vector_store

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
        assert 0.0 <= response.results[0].score <= 1.0
        expected_score = 100.0 / (1.0 + 100.0)
        assert abs(response.results[0].score - expected_score) < 0.0001

    def test_search_sparse_fts_fallback_warning_content(self) -> None:
        """Test that FTS_FALLBACK warning has correct content and fallback_action."""
        from xagent.core.tools.core.RAG_tools.retrieval.search_sparse import (
            _substring_fallback,
        )

        warnings: List[SearchWarning] = []

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

        mock_vector_store = Mock()
        mock_vector_store.open_embeddings_table.return_value = (Mock(), "embeddings_x")
        mock_vector_store.iter_batches.return_value = [mock_batch]

        with patch(
            "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_vector_index_store",
            return_value=mock_vector_store,
        ):
            results = _substring_fallback(
                model_tag="test_model",
                collection="test_col",
                query_text="test query",
                top_k=5,
                filters=None,
                current_warnings=warnings,
            )

        assert len(results) > 0
        assert len(warnings) == 1
        warning = warnings[0]

        assert warning.code == "FTS_FALLBACK"
        assert warning.fallback_action == SearchFallbackAction.BRUTE_FORCE
        assert warning.affected_models == ["test_model"]

        assert "Full-text index returned no matches" in warning.message
        assert "used substring search fallback" in warning.message
        assert "Check FTS tokenizer configuration" in warning.message
        assert "update LanceDB to ensure proper tokenisation" in warning.message

    def test_substring_fallback_pins_route_collection_over_caller_override(
        self,
    ) -> None:
        """Route collection must not be replaced by caller filters with the same key."""
        from xagent.core.tools.core.RAG_tools.retrieval.search_sparse import (
            _substring_fallback,
        )

        mock_batch = Mock()
        mock_batch.to_pandas.return_value = pd.DataFrame(
            {
                "doc_id": ["doc1"],
                "chunk_id": ["chunk1"],
                "text": ["needle"],
                "parse_hash": ["hash1"],
                "created_at": [pd.Timestamp.now()],
                "metadata": [None],
            }
        )

        mock_vector_store = Mock()
        mock_vector_store.open_embeddings_table.return_value = (Mock(), "embeddings_x")
        mock_vector_store.iter_batches.return_value = [mock_batch]

        with patch(
            "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_vector_index_store",
            return_value=mock_vector_store,
        ):
            _substring_fallback(
                model_tag="test_model",
                collection="route_collection",
                query_text="needle",
                top_k=5,
                filters={"collection": "evil_override", "doc_id": "d1"},
                current_warnings=[],
            )

        iter_kwargs = mock_vector_store.iter_batches.call_args.kwargs
        assert iter_kwargs["filters"]["collection"] == "route_collection"
        assert iter_kwargs["filters"]["doc_id"] == "d1"


@pytest.mark.asyncio
class TestSearchSparseAsync:
    """Test async sparse search paths."""

    async def test_search_sparse_async_success_forwards_user_scope(self) -> None:
        """Async sparse search should forward user scope to async store call."""
        mock_vector_store = Mock()
        mock_vector_store.create_index.return_value = IndexResult(
            status="index_ready",
            advice=None,
            fts_enabled=True,
        )
        mock_vector_store.search_fts_by_model_async = AsyncMock(
            return_value=[
                {
                    "doc_id": "doc1",
                    "chunk_id": "chunk1",
                    "text": "hello world",
                    "_score": 0.8,
                    "parse_hash": "h1",
                    "created_at": pd.Timestamp.now(),
                }
            ]
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_vector_index_store",
            return_value=mock_vector_store,
        ):
            response = await search_sparse_module.search_sparse_async(
                collection="kb1",
                model_tag="m1",
                query_text="hello",
                top_k=3,
                user_id=9,
                is_admin=False,
            )

        assert response.status == "success"
        assert response.total_count == 1
        mock_vector_store.search_fts_by_model_async.assert_called_once_with(
            model_tag="m1",
            query_text="hello",
            top_k=3,
            filters=mock_vector_store.search_fts_by_model_async.call_args.kwargs[
                "filters"
            ],
            text_column_name="text",
            user_id=9,
            is_admin=False,
        )

    async def test_search_sparse_async_triggers_fallback(self) -> None:
        """Async sparse should use async fallback when FTS returns empty."""
        mock_vector_store = Mock()
        mock_vector_store.create_index.return_value = IndexResult(
            status="index_ready",
            advice=None,
            fts_enabled=True,
        )
        mock_vector_store.search_fts_by_model_async = AsyncMock(return_value=[])

        fallback_result = [
            SearchResult(
                doc_id="doc-fallback",
                chunk_id="chunk-fallback",
                text="fallback text",
                score=1.0,
                parse_hash="h2",
                model_tag="m1",
                created_at=pd.Timestamp.now(),
            )
        ]

        with (
            patch(
                "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_vector_index_store",
                return_value=mock_vector_store,
            ),
            patch.object(
                search_sparse_module,
                "_substring_fallback_async",
                return_value=fallback_result,
            ) as mock_fallback,
        ):
            response = await search_sparse_module.search_sparse_async(
                collection="kb1",
                model_tag="m1",
                query_text="miss",
                top_k=2,
                user_id=7,
                is_admin=False,
            )

        assert response.status == "success"
        assert response.total_count == 1
        assert response.results[0].doc_id == "doc-fallback"
        mock_fallback.assert_called_once()

    async def test_search_sparse_async_error_returns_failed_response(self) -> None:
        """Async sparse should return failed response when store call errors."""
        mock_vector_store = Mock()
        mock_vector_store.create_index.return_value = IndexResult(
            status="index_ready",
            advice=None,
            fts_enabled=True,
        )
        mock_vector_store.search_fts_by_model_async = AsyncMock(
            side_effect=Exception("boom")
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.retrieval.search_sparse.get_vector_index_store",
            return_value=mock_vector_store,
        ):
            response = await search_sparse_module.search_sparse_async(
                collection="kb1",
                model_tag="m1",
                query_text="q",
                top_k=1,
            )

        assert response.status == "failed"
        assert response.total_count == 0
        assert any(w.code == "FTS_SEARCH_FAILED" for w in response.warnings)
