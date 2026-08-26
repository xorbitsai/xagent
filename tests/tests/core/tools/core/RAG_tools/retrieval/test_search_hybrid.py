import datetime
from typing import Generator, Tuple
from unittest.mock import Mock

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import DocumentValidationError
from xagent.core.tools.core.RAG_tools.core.exceptions import VectorValidationError
from xagent.core.tools.core.RAG_tools.core.schemas import (
    DenseSearchResponse,
    FusionConfig,
    FusionStrategy,
    HybridSearchResponse,
    IndexStatus,
    SearchFallbackAction,
    SearchResult,
    SearchWarning,
    SparseSearchResponse,
)
from xagent.core.tools.core.RAG_tools.retrieval.search_hybrid import (
    _linear_fusion,
    _rrf_fusion,
    search_hybrid,
)


class TestFusionFunctions:
    """Tests for internal fusion helper functions."""

    def test_rrf_fusion_basic(self) -> None:
        """Test RRF fusion with basic scenario."""
        # Setup sample data
        results1 = [
            SearchResult(
                doc_id="d1",
                chunk_id="c1",
                text="t1",
                score=0.9,
                parse_hash="p1",
                model_tag="m1",
                created_at=datetime.datetime.now(),
            ),
            SearchResult(
                doc_id="d2",
                chunk_id="c2",
                text="t2",
                score=0.8,
                parse_hash="p2",
                model_tag="m1",
                created_at=datetime.datetime.now(),
            ),
        ]
        results2 = [
            SearchResult(
                doc_id="d2",
                chunk_id="c2",
                text="t2",
                score=0.95,
                parse_hash="p2",
                model_tag="m1",
                created_at=datetime.datetime.now(),
            ),
            SearchResult(
                doc_id="d3",
                chunk_id="c3",
                text="t3",
                score=0.7,
                parse_hash="p3",
                model_tag="m1",
                created_at=datetime.datetime.now(),
            ),
        ]

        rank_lists = [results1, results2]
        fused = _rrf_fusion(rank_lists, k=1)  # Using k=1 for simpler score calculation

        # Expected RRF scores:
        # d1-c1: 1/(1+1) = 0.5 (from results1)
        # d2-c2: 1/(1+2) + 1/(1+1) = 0.333 + 0.5 = 0.833 (from results1 & results2)
        # d3-c3: 1/(1+2) = 0.333 (from results2)

        assert len(fused) == 3
        # d2-c2 should be highest ranked
        assert fused[0].doc_id == "d2"
        assert abs(fused[0].score - 0.833) < 0.001
        # d1-c1 next
        assert fused[1].doc_id == "d1"
        assert abs(fused[1].score - 0.5) < 0.001
        # d3-c3 last
        assert fused[2].doc_id == "d3"
        assert abs(fused[2].score - 0.333) < 0.001

    def test_rrf_fusion_empty_lists(self) -> None:
        """Test RRF fusion with empty input lists."""
        fused = _rrf_fusion([], k=10)
        assert len(fused) == 0

        fused = _rrf_fusion([[]], k=10)
        assert len(fused) == 0

        fused = _rrf_fusion([[], []], k=10)
        assert len(fused) == 0

    def test_linear_fusion_basic(self) -> None:
        """Test linear fusion with basic scenario and no normalization."""
        dense_results = [
            SearchResult(
                doc_id="d1",
                chunk_id="c1",
                text="t1",
                score=0.8,
                parse_hash="p1",
                model_tag="m1",
                created_at=datetime.datetime.now(),
            ),
            SearchResult(
                doc_id="d2",
                chunk_id="c2",
                text="t2",
                score=0.6,
                parse_hash="p2",
                model_tag="m1",
                created_at=datetime.datetime.now(),
            ),
        ]
        sparse_results = [
            SearchResult(
                doc_id="d2",
                chunk_id="c2",
                text="t2",
                score=0.7,
                parse_hash="p2",
                model_tag="m1",
                created_at=datetime.datetime.now(),
            ),
            SearchResult(
                doc_id="d3",
                chunk_id="c3",
                text="t3",
                score=0.5,
                parse_hash="p3",
                model_tag="m1",
                created_at=datetime.datetime.now(),
            ),
        ]

        fused = _linear_fusion(
            dense_results,
            sparse_results,
            dense_weight=0.5,
            sparse_weight=0.5,
            normalize_scores=False,
        )

        # Expected scores before final normalization:
        # d1-c1: 0.8 * 0.5 = 0.4
        # d2-c2: (0.6 * 0.5) + (0.7 * 0.5) = 0.3 + 0.35 = 0.65
        # d3-c3: 0.5 * 0.5 = 0.25
        # After final normalization (Min-Max to [0,1]):
        # min_score = 0.25, max_score = 0.65, range = 0.4
        # d2-c2: (0.65 - 0.25) / 0.4 = 1.0
        # d1-c1: (0.4 - 0.25) / 0.4 = 0.375
        # d3-c3: (0.25 - 0.25) / 0.4 = 0.0

        assert len(fused) == 3
        assert fused[0].doc_id == "d2"
        assert abs(fused[0].score - 1.0) < 0.001
        assert fused[1].doc_id == "d1"
        assert abs(fused[1].score - 0.375) < 0.001
        assert fused[2].doc_id == "d3"
        assert abs(fused[2].score - 0.0) < 0.001

    def test_linear_fusion_with_normalization(self) -> None:
        """Test linear fusion with Min-Max normalization."""
        dense_results = [
            SearchResult(
                doc_id="d1",
                chunk_id="c1",
                text="t1",
                score=0.8,
                parse_hash="p1",
                model_tag="m1",
                created_at=datetime.datetime.now(),
            ),
            SearchResult(
                doc_id="d2",
                chunk_id="c2",
                text="t2",
                score=0.6,
                parse_hash="p2",
                model_tag="m1",
                created_at=datetime.datetime.now(),
            ),
        ]
        sparse_results = [
            SearchResult(
                doc_id="d2",
                chunk_id="c2",
                text="t2",
                score=0.7,
                parse_hash="p2",
                model_tag="m1",
                created_at=datetime.datetime.now(),
            ),
            SearchResult(
                doc_id="d3",
                chunk_id="c3",
                text="t3",
                score=0.5,
                parse_hash="p3",
                model_tag="m1",
                created_at=datetime.datetime.now(),
            ),
        ]

        # Normalized dense scores: [ (0.8-0.6)/(0.8-0.6)=1.0, (0.6-0.6)/(0.8-0.6)=0.0 ] -> [1.0, 0.0]
        # Normalized sparse scores: [ (0.7-0.5)/(0.7-0.5)=1.0, (0.5-0.5)/(0.7-0.5)=0.0 ] -> [1.0, 0.0]

        fused = _linear_fusion(
            dense_results,
            sparse_results,
            dense_weight=0.5,
            sparse_weight=0.5,
            normalize_scores=True,
        )

        # Expected scores before final normalization:
        # d1-c1: 1.0 * 0.5 = 0.5
        # d2-c2: (0.0 * 0.5) + (1.0 * 0.5) = 0.0 + 0.5 = 0.5
        # d3-c3: 0.0 * 0.5 = 0.0
        # After final normalization (Min-Max to [0,1]):
        # d1-c1 and d2-c2: 0.5/0.5 = 1.0 (both have same score)
        # d3-c3: 0.0/0.5 = 0.0

        assert len(fused) == 3
        # d1 and d2 have the same score (both 1.0 after normalization)
        doc_ids = [fused[0].doc_id, fused[1].doc_id]
        assert set(doc_ids) == {"d1", "d2"}
        assert abs(fused[0].score - 1.0) < 0.001
        assert abs(fused[1].score - 1.0) < 0.001
        assert fused[2].doc_id == "d3"
        assert abs(fused[2].score - 0.0) < 0.001

    def test_linear_fusion_empty_lists(self) -> None:
        """Test linear fusion with empty input lists."""
        fused = _linear_fusion(
            [], [], dense_weight=0.5, sparse_weight=0.5, normalize_scores=True
        )
        assert len(fused) == 0

        fused = _linear_fusion(
            [
                SearchResult(
                    doc_id="d1",
                    chunk_id="c1",
                    text="t1",
                    score=0.8,
                    parse_hash="p1",
                    model_tag="m1",
                    created_at=datetime.datetime.now(),
                )
            ],
            [],
            dense_weight=0.5,
            sparse_weight=0.5,
            normalize_scores=True,
        )
        assert len(fused) == 1
        assert fused[0].doc_id == "d1"


class TestSearchHybrid:
    """Tests for search_hybrid main function."""

    def _patch_search_hybrid_module(self):
        """Helper method to import and patch search_hybrid module.

        Resolves ambiguity when module name and function name are the same.
        """
        import importlib

        search_hybrid_module = importlib.import_module(
            "xagent.core.tools.core.RAG_tools.retrieval.search_hybrid"
        )
        return search_hybrid_module

    def _create_mock_search_results(self, count: int, base_id: str = "d") -> list:
        """Helper to create mock search results for testing.

        Args:
            count: Number of results to create
            base_id: Base identifier for doc_id, chunk_id, etc.

        Returns:
            List of SearchResult objects
        """
        results = []
        for i in range(count):
            results.append(
                SearchResult(
                    doc_id=f"{base_id}{i + 1}",
                    chunk_id=f"c{i + 1}",
                    text=f"t{i + 1}",
                    score=0.9 - i * 0.01,
                    parse_hash=f"p{i + 1}",
                    model_tag="m1",
                    created_at=datetime.datetime.now(),
                )
            )
        return results

    @pytest.mark.parametrize(
        "collection,model_tag,top_k,query_text,query_vector",
        [
            ("", "model", 1, "hello", [0.1]),
            ("collection", "", 1, "hello", [0.1]),
            ("collection", "model", 0, "hello", [0.1]),
            ("collection", "model", 2000, "hello", [0.1]),
            ("collection", "model", 1, "", [0.1]),
        ],
    )
    def test_search_hybrid_input_validation(
        self,
        collection: str,
        model_tag: str,
        top_k: int,
        query_text: str,
        query_vector: list[float],
    ) -> None:
        with pytest.raises(DocumentValidationError):
            search_hybrid(
                collection=collection,
                model_tag=model_tag,
                query_text=query_text,
                query_vector=query_vector,
                top_k=top_k,
            )

    def test_search_hybrid_query_vector_validation(self) -> None:
        with pytest.raises(VectorValidationError):
            search_hybrid(
                collection="collection",
                model_tag="model",
                query_text="hello",
                query_vector=[],
                top_k=1,
            )

    @pytest.fixture
    def mock_sub_searches(
        self, make_handle, routed_facade
    ) -> Generator[Tuple[Mock, Mock], None, None]:
        """Mock the handle's dense/sparse sub-searches and route the public call.

        Re-pointed (#511): the public ``search_hybrid`` now runs the real handle
        ``search_hybrid`` (fusion logic identical to before — it reuses the same
        ``_rrf_fusion``/``_linear_fusion`` free functions), which calls
        ``handle.search_dense``/``handle.search_sparse``. We patch those handle
        methods and route the public function to that handle. The handle invokes
        them positionally ``(model_tag, query_vector|query_text, top_k=..., ...)``
        with NO ``collection`` kwarg (the handle owns its collection).
        """
        search_hybrid_module = self._patch_search_hybrid_module()
        handle, _store, _ = make_handle(collection="test_col")

        mock_dense = Mock()
        mock_sparse = Mock()
        # Bind through the instance so calls drop the implicit ``self``.
        object.__setattr__(handle, "search_dense", mock_dense)
        object.__setattr__(handle, "search_sparse", mock_sparse)

        with routed_facade(search_hybrid_module, handle):
            yield mock_dense, mock_sparse

    def test_hybrid_search_rrf_strategy(
        self, mock_sub_searches: Tuple[Mock, Mock]
    ) -> None:
        """Test hybrid search with RRF fusion strategy."""
        mock_dense, mock_sparse = mock_sub_searches

        # Mock dense search response
        dense_results = [
            SearchResult(
                doc_id="d1",
                chunk_id="c1",
                text="t1",
                score=0.9,
                parse_hash="p1",
                model_tag="m1",
                created_at=datetime.datetime.now(),
            ),
            SearchResult(
                doc_id="d2",
                chunk_id="c2",
                text="t2",
                score=0.8,
                parse_hash="p2",
                model_tag="m1",
                created_at=datetime.datetime.now(),
            ),
        ]
        mock_dense.return_value = DenseSearchResponse(
            results=dense_results,
            total_count=len(dense_results),
            index_status=IndexStatus.INDEX_READY,
            index_advice="Ready",
            warnings=[],
        )

        # Mock sparse search response
        sparse_results = [
            SearchResult(
                doc_id="d2",
                chunk_id="c2",
                text="t2",
                score=0.7,
                parse_hash="p2",
                model_tag="m1",
                created_at=datetime.datetime.now(),
            ),
            SearchResult(
                doc_id="d3",
                chunk_id="c3",
                text="t3",
                score=0.6,
                parse_hash="p3",
                model_tag="m1",
                created_at=datetime.datetime.now(),
            ),
        ]
        mock_sparse.return_value = SparseSearchResponse(
            results=sparse_results,
            total_count=len(sparse_results),
            fts_enabled=True,
            query_text="query",
            warnings=[],
        )

        fusion_config = FusionConfig(strategy=FusionStrategy.RRF, rrf_k=1)

        response = search_hybrid(
            collection="test_col",
            model_tag="test_model",
            query_text="hybrid query",
            query_vector=[0.1, 0.2, 0.3],
            top_k=2,
            fusion_config=fusion_config,
        )

        assert isinstance(response, HybridSearchResponse)
        assert response.status == "success"
        assert len(response.results) == 2
        assert response.total_count == 2
        assert response.fusion_config.strategy == FusionStrategy.RRF
        assert response.index_status == IndexStatus.INDEX_READY
        assert response.index_advice is not None
        assert "Ready" in response.index_advice

        # Verify RRF logic (same as _rrf_fusion_basic test)
        assert response.results[0].doc_id == "d2"
        assert abs(response.results[0].score - 0.833) < 0.001
        assert response.results[1].doc_id == "d1"
        assert abs(response.results[1].score - 0.5) < 0.001

        mock_dense.assert_called_once_with(
            "test_model",
            [0.1, 0.2, 0.3],
            top_k=4,  # top_k * 2
            filters=None,
            readonly=False,
            nprobes=None,
            refine_factor=None,
            user_id=None,
            is_admin=False,
        )
        mock_sparse.assert_called_once_with(
            "test_model",
            "hybrid query",
            top_k=4,  # top_k * 2
            filters=None,
            readonly=False,
            user_id=None,
            is_admin=False,
        )

    def test_hybrid_search_linear_strategy(
        self, mock_sub_searches: Tuple[Mock, Mock]
    ) -> None:
        """Test hybrid search with linear fusion strategy."""
        mock_dense, mock_sparse = mock_sub_searches

        # Mock dense search response
        dense_results = [
            SearchResult(
                doc_id="d1",
                chunk_id="c1",
                text="t1",
                score=0.8,
                parse_hash="p1",
                model_tag="m1",
                created_at=datetime.datetime.now(),
            ),
            SearchResult(
                doc_id="d2",
                chunk_id="c2",
                text="t2",
                score=0.6,
                parse_hash="p2",
                model_tag="m1",
                created_at=datetime.datetime.now(),
            ),
        ]
        mock_dense.return_value = DenseSearchResponse(
            results=dense_results,
            total_count=len(dense_results),
            index_status=IndexStatus.INDEX_READY,
            index_advice="Ready",
            warnings=[],
        )

        # Mock sparse search response
        sparse_results = [
            SearchResult(
                doc_id="d2",
                chunk_id="c2",
                text="t2",
                score=0.7,
                parse_hash="p2",
                model_tag="m1",
                created_at=datetime.datetime.now(),
            ),
            SearchResult(
                doc_id="d3",
                chunk_id="c3",
                text="t3",
                score=0.5,
                parse_hash="p3",
                model_tag="m1",
                created_at=datetime.datetime.now(),
            ),
        ]
        mock_sparse.return_value = SparseSearchResponse(
            results=sparse_results,
            total_count=len(sparse_results),
            fts_enabled=True,
            query_text="query",
            warnings=[],
        )

        fusion_config = FusionConfig(
            strategy=FusionStrategy.LINEAR,
            dense_weight=0.6,
            sparse_weight=0.4,
            normalize_scores=False,
        )

        response = search_hybrid(
            collection="test_col",
            model_tag="test_model",
            query_text="hybrid query",
            query_vector=[0.1, 0.2, 0.3],
            top_k=3,
            fusion_config=fusion_config,
        )

        assert isinstance(response, HybridSearchResponse)
        assert response.status == "success"
        assert len(response.results) == 3
        assert response.total_count == 3
        assert response.fusion_config.strategy == FusionStrategy.LINEAR

        # Verify linear fusion logic with final normalization
        # Before final normalization:
        # d1-c1: 0.8 * 0.6 = 0.48
        # d2-c2: (0.6 * 0.6) + (0.7 * 0.4) = 0.36 + 0.28 = 0.64
        # d3-c3: 0.5 * 0.4 = 0.2
        # After final normalization (Min-Max to [0,1]):
        # min_score = 0.2, max_score = 0.64, range = 0.44
        # d2-c2: (0.64 - 0.2) / 0.44 = 1.0
        # d1-c1: (0.48 - 0.2) / 0.44 ≈ 0.636
        # d3-c3: (0.2 - 0.2) / 0.44 = 0.0

        assert response.results[0].doc_id == "d2"
        assert abs(response.results[0].score - 1.0) < 0.001
        assert response.results[1].doc_id == "d1"
        assert abs(response.results[1].score - ((0.48 - 0.2) / (0.64 - 0.2))) < 0.001
        assert response.results[2].doc_id == "d3"
        assert abs(response.results[2].score - 0.0) < 0.001

    def test_hybrid_search_with_filters_and_warnings(
        self, mock_sub_searches: Tuple[Mock, Mock]
    ) -> None:
        """Test hybrid search with filters and warning propagation."""
        mock_dense, mock_sparse = mock_sub_searches

        dense_warning = SearchWarning(
            code="DENSE_WARN",
            message="Dense degraded",
            fallback_action=SearchFallbackAction.PARTIAL_RESULTS,
            affected_models=["m1"],
        )
        sparse_warning = SearchWarning(
            code="SPARSE_WARN",
            message="Sparse degraded",
            fallback_action=SearchFallbackAction.PARTIAL_RESULTS,
            affected_models=["m1"],
        )

        dense_results = [
            SearchResult(
                doc_id="d1",
                chunk_id="c1",
                text="t1",
                score=0.9,
                parse_hash="p1",
                model_tag="m1",
                created_at=datetime.datetime.now(),
            )
        ]
        sparse_results = [
            SearchResult(
                doc_id="d2",
                chunk_id="c2",
                text="t2",
                score=0.8,
                parse_hash="p2",
                model_tag="m1",
                created_at=datetime.datetime.now(),
            )
        ]

        mock_dense.return_value = DenseSearchResponse(
            results=dense_results,
            total_count=len(dense_results),
            index_status=IndexStatus.NO_INDEX,
            index_advice="Build index",
            warnings=[dense_warning],
        )
        mock_sparse.return_value = SparseSearchResponse(
            results=sparse_results,
            total_count=len(sparse_results),
            fts_enabled=False,
            query_text="query",
            warnings=[sparse_warning],
        )

        filters = {"collection": "filtered_col"}
        fusion_config = FusionConfig(strategy=FusionStrategy.RRF)

        response = search_hybrid(
            collection="test_col",
            model_tag="test_model",
            query_text="filtered query",
            query_vector=[0.1, 0.2, 0.3],
            top_k=1,
            filters=filters,
            fusion_config=fusion_config,
        )

        assert response.status == "partial_success"
        assert len(response.warnings) == 2
        assert any(w.code == "DENSE_WARN" for w in response.warnings)
        assert any(w.code == "SPARSE_WARN" for w in response.warnings)
        assert response.index_status == IndexStatus.NO_INDEX
        assert response.index_advice is not None
        assert "Build index" in response.index_advice

        mock_dense.assert_called_once_with(
            "test_model",
            [0.1, 0.2, 0.3],
            top_k=2,  # top_k * 2
            filters=filters,
            readonly=False,
            nprobes=None,
            refine_factor=None,
            user_id=None,
            is_admin=False,
        )
        mock_sparse.assert_called_once_with(
            "test_model",
            "filtered query",
            top_k=2,  # top_k * 2
            filters=filters,
            readonly=False,
            user_id=None,
            is_admin=False,
        )

    def test_hybrid_search_empty_results(
        self, mock_sub_searches: Tuple[Mock, Mock]
    ) -> None:
        """Test hybrid search when sub-searches return empty results."""
        mock_dense, mock_sparse = mock_sub_searches

        mock_dense.return_value = DenseSearchResponse(
            results=[],
            total_count=0,
            index_status=IndexStatus.NO_INDEX,
            index_advice="No index",
            warnings=[],
        )
        mock_sparse.return_value = SparseSearchResponse(
            results=[],
            total_count=0,
            fts_enabled=False,
            query_text="empty",
            warnings=[],
        )

        response = search_hybrid(
            collection="test_col",
            model_tag="test_model",
            query_text="empty query",
            query_vector=[0.1, 0.2, 0.3],
            top_k=5,
        )

        assert response.status == "success"
        assert len(response.results) == 0
        assert response.total_count == 0
        assert not response.warnings

    def test_hybrid_search_default_fusion_config(
        self, mock_sub_searches: Tuple[Mock, Mock]
    ) -> None:
        """Test hybrid search uses default FusionConfig when none is provided."""
        mock_dense, mock_sparse = mock_sub_searches

        dense_results = [
            SearchResult(
                doc_id="d1",
                chunk_id="c1",
                text="t1",
                score=0.9,
                parse_hash="p1",
                model_tag="m1",
                created_at=datetime.datetime.now(),
            )
        ]
        sparse_results = [
            SearchResult(
                doc_id="d2",
                chunk_id="c2",
                text="t2",
                score=0.8,
                parse_hash="p2",
                model_tag="m1",
                created_at=datetime.datetime.now(),
            )
        ]

        mock_dense.return_value = DenseSearchResponse(
            results=dense_results,
            total_count=len(dense_results),
            index_status=IndexStatus.INDEX_READY,
            index_advice="Ready",
            warnings=[],
        )
        mock_sparse.return_value = SparseSearchResponse(
            results=sparse_results,
            total_count=len(sparse_results),
            fts_enabled=True,
            query_text="query",
            warnings=[],
        )

        response = search_hybrid(
            collection="test_col",
            model_tag="test_model",
            query_text="default config",
            query_vector=[0.1, 0.2, 0.3],
            top_k=1,
        )

        assert response.fusion_config.strategy == FusionStrategy.RRF
        assert response.fusion_config.rrf_k == 60  # Default RRF k value
        assert response.status == "success"  # No warnings, so success
        assert len(response.results) == 1
        assert response.total_count == 1
        assert response.index_status == IndexStatus.INDEX_READY

    def test_hybrid_search_readonly_mode(
        self, mock_sub_searches: Tuple[Mock, Mock]
    ) -> None:
        """Test readonly mode propagation in hybrid search."""
        mock_dense, mock_sparse = mock_sub_searches

        dense_results = [
            SearchResult(
                doc_id="d1",
                chunk_id="c1",
                text="t1",
                score=0.9,
                parse_hash="p1",
                model_tag="m1",
                created_at=datetime.datetime.now(),
            )
        ]
        sparse_results = [
            SearchResult(
                doc_id="d2",
                chunk_id="c2",
                text="t2",
                score=0.8,
                parse_hash="p2",
                model_tag="m1",
                created_at=datetime.datetime.now(),
            )
        ]

        mock_dense.return_value = DenseSearchResponse(
            results=dense_results,
            total_count=len(dense_results),
            index_status=IndexStatus.READONLY,
            index_advice="Readonly mode enabled",
            warnings=[
                SearchWarning(
                    code="READONLY_MODE",
                    message="RO",
                    fallback_action=SearchFallbackAction.REBUILD_INDEX,
                    affected_models=["m1"],
                )
            ],
        )
        mock_sparse.return_value = SparseSearchResponse(
            results=sparse_results,
            total_count=len(sparse_results),
            fts_enabled=True,
            query_text="query",
            warnings=[
                SearchWarning(
                    code="READONLY_MODE",
                    message="RO",
                    fallback_action=SearchFallbackAction.REBUILD_INDEX,
                    affected_models=["m1"],
                )
            ],
        )

        response = search_hybrid(
            collection="test_col",
            model_tag="test_model",
            query_text="readonly query",
            query_vector=[0.1, 0.2, 0.3],
            top_k=1,
            readonly=True,
        )

        assert response.status == "partial_success"
        assert response.index_status == IndexStatus.READONLY
        assert any(w.code == "READONLY_MODE" for w in response.warnings)

        mock_dense.assert_called_once_with(
            "test_model",
            [0.1, 0.2, 0.3],
            top_k=2,
            filters=None,
            readonly=True,  # Verify readonly is propagated
            nprobes=None,
            refine_factor=None,
            user_id=None,
            is_admin=False,
        )
        mock_sparse.assert_called_once_with(
            "test_model",
            "readonly query",
            top_k=2,
            filters=None,
            readonly=True,  # Verify readonly is propagated
            user_id=None,
            is_admin=False,
        )

    def test_hybrid_search_top_k_limit(
        self, mock_sub_searches: Tuple[Mock, Mock]
    ) -> None:
        """Test top_k limit is applied after fusion."""
        mock_dense, mock_sparse = mock_sub_searches

        # Create dense results with decreasing scores
        dense_results = []
        for i in range(5):
            dense_results.append(
                SearchResult(
                    doc_id=f"d{i}",
                    chunk_id=f"c{i}",
                    text=f"t{i}",
                    score=0.9 - i * 0.01,
                    parse_hash="p1",
                    model_tag="m1",
                    created_at=datetime.datetime.now(),
                )
            )

        # Create sparse results with different doc_ids
        sparse_results = []
        for i in range(5):
            sparse_results.append(
                SearchResult(
                    doc_id=f"d{i + 5}",
                    chunk_id=f"c{i + 5}",
                    text=f"t{i + 5}",
                    score=0.8 - i * 0.01,
                    parse_hash="p2",
                    model_tag="m1",
                    created_at=datetime.datetime.now(),
                )
            )

        mock_dense.return_value = DenseSearchResponse(
            results=dense_results,
            total_count=len(dense_results),
            index_status=IndexStatus.INDEX_READY,
            index_advice="Ready",
            warnings=[],
        )
        mock_sparse.return_value = SparseSearchResponse(
            results=sparse_results,
            total_count=len(sparse_results),
            fts_enabled=True,
            query_text="query",
            warnings=[],
        )

        response = search_hybrid(
            collection="test_col",
            model_tag="test_model",
            query_text="top_k test",
            query_vector=[0.1, 0.2, 0.3],
            top_k=3,  # Request only 3 results
        )

        assert response.total_count == 3
        assert len(response.results) == 3
        # Ensure results are sorted by score and truncated
        assert response.results[0].score >= response.results[1].score
        assert response.results[1].score >= response.results[2].score

    # Removed test_hybrid_search_unsupported_strategy_fallback:
    # Pydantic V2's strict enum validation prevents creating FusionConfig with invalid strategy.
    # This is the desired behavior - invalid strategies are caught at input validation layer.

    def test_linear_fusion_score_normalization(self) -> None:
        """Test that linear fusion normalizes final scores to [0, 1] range."""
        from xagent.core.tools.core.RAG_tools.retrieval.search_hybrid import (
            _linear_fusion,
        )

        # Create test data that would result in scores > 1.0 before normalization
        dense_results = [
            SearchResult(
                doc_id="doc1",
                chunk_id="chunk1",
                text="dense content 1",
                score=0.9,
                parse_hash="hash1",
                model_tag="model1",
                created_at=datetime.datetime.now(),
            ),
            SearchResult(
                doc_id="doc2",
                chunk_id="chunk2",
                text="dense content 2",
                score=0.6,
                parse_hash="hash2",
                model_tag="model1",
                created_at=datetime.datetime.now(),
            ),
        ]

        sparse_results = [
            SearchResult(
                doc_id="doc1",  # Same doc as in dense - should get combined score
                chunk_id="chunk1",
                text="sparse content 1",
                score=0.8,
                parse_hash="hash1",
                model_tag="model1",
                created_at=datetime.datetime.now(),
            ),
            SearchResult(
                doc_id="doc3",
                chunk_id="chunk3",
                text="sparse content 3",
                score=0.5,
                parse_hash="hash3",
                model_tag="model1",
                created_at=datetime.datetime.now(),
            ),
        ]

        # Use weights that would cause combined scores > 1.0
        # doc1: 0.7 * 0.9 + 0.8 * 0.8 = 0.63 + 0.64 = 1.27
        # doc2: 0.7 * 0.6 = 0.42
        # doc3: 0.8 * 0.5 = 0.4
        fused_results = _linear_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            dense_weight=0.7,
            sparse_weight=0.8,
            normalize_scores=True,
        )

        # Verify that all scores are in [0, 1] range after normalization
        for result in fused_results:
            assert 0.0 <= result.score <= 1.0, (
                f"Score {result.score} is not in [0, 1] range"
            )

        # Verify that the highest score is 1.0 (after normalization)
        max_score = max(r.score for r in fused_results)
        assert abs(max_score - 1.0) < 0.001, (
            f"Max score should be 1.0 after normalization, got {max_score}"
        )

        # Verify results are sorted by score in descending order
        scores = [r.score for r in fused_results]
        assert scores == sorted(scores, reverse=True), (
            "Results should be sorted by score descending"
        )
