"""Tests for the collection handle dense search lifecycle (#511).

The handle owns collection-scoped dense search mechanics: capability guard,
index creation, filter building, score conversion, and result assembly.
Search provider calls (vector store) are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import (
    DenseSearchResponse,
    IndexStatus,
    SparseSearchResponse,
)
from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    LanceDBCollectionHandle,
)
from xagent.core.tools.core.RAG_tools.storage.contracts import (
    FilterCondition,
    FilterOperator,
)


def _make_handle(*, supports_search: bool = True):
    """Create a LanceDBCollectionHandle with mocked context/store/capabilities.

    LanceDBCollectionHandle is a frozen dataclass, so we use object.__setattr__
    to inject mocks into the underlying ``context`` field.
    """
    handle = LanceDBCollectionHandle.__new__(LanceDBCollectionHandle)
    ctx = MagicMock()
    ctx.collection = "col1"
    # frozen dataclass — must use object.__setattr__
    object.__setattr__(handle, "context", ctx)

    store = MagicMock()
    ctx.vector_index_store = store

    caps = MagicMock()
    caps.supports_search = supports_search
    ctx.capabilities = caps

    return handle, ctx, store, caps


def _index_result(status="index_ready", advice=None):
    obj = MagicMock()
    obj.status = status
    obj.advice = advice
    obj.fts_enabled = True
    return obj


def test_search_dense_success_score_and_filters():
    handle, ctx, store, _ = _make_handle()
    store.create_index.return_value = _index_result()
    store.search_vectors_by_model.return_value = [
        {
            "doc_id": "d1",
            "chunk_id": "c1",
            "text": "t",
            "parse_hash": "h",
            "created_at": "2026-01-01",
            "metadata": None,
            "_distance": 1.0,
        },
    ]
    resp = handle.search_dense(
        "model-x", [0.1, 0.2], top_k=5, user_id=7, is_admin=False
    )
    assert isinstance(resp, DenseSearchResponse)
    assert resp.status == "success"
    assert resp.total_count == 1
    assert resp.results[0].score == pytest.approx(0.5)  # 1/(1+1.0)
    # collection filter + user scope reached the store
    kwargs = store.search_vectors_by_model.call_args.kwargs
    assert kwargs["model_tag"] == "model-x"
    assert kwargs["user_id"] == 7 and kwargs["is_admin"] is False


def test_search_dense_preserves_prebuilt_filter_expression():
    handle, _, store, _ = _make_handle()
    store.create_index.return_value = _index_result()
    store.search_vectors_by_model.return_value = [
        {
            "doc_id": "d1",
            "chunk_id": "c1",
            "text": "t",
            "parse_hash": "h",
            "created_at": "2026-01-01",
            "metadata": None,
            "_distance": 0.0,
        },
    ]

    custom_filter = (
        FilterCondition(
            field="doc_id",
            operator=FilterOperator.EQ,
            value="d1",
        ),
        FilterCondition(
            field="chunk_id",
            operator=FilterOperator.EQ,
            value="c1",
        ),
    )

    resp = handle.search_dense(
        "model-x",
        [0.1, 0.2],
        top_k=5,
        filters=custom_filter,
        user_id=7,
        is_admin=False,
    )

    assert isinstance(resp, DenseSearchResponse)
    assert resp.status == "success"
    passed_filters = store.search_vectors_by_model.call_args.kwargs["filters"]
    assert isinstance(passed_filters, tuple)
    assert len(passed_filters) == 2
    assert passed_filters[1] == custom_filter


@pytest.mark.parametrize(
    "custom_filter",
    [
        FilterCondition(
            field="doc_id",
            operator=FilterOperator.EQ,
            value="d1",
        ),
        [
            FilterCondition(
                field="doc_id",
                operator=FilterOperator.EQ,
                value="d1",
            ),
            FilterCondition(
                field="chunk_id",
                operator=FilterOperator.EQ,
                value="c1",
            ),
        ],
    ],
)
def test_search_dense_preserves_single_and_list_filter_expressions(custom_filter):
    handle, _, store, _ = _make_handle()
    store.create_index.return_value = _index_result()
    store.search_vectors_by_model.return_value = [
        {
            "doc_id": "d1",
            "chunk_id": "c1",
            "text": "t",
            "parse_hash": "h",
            "created_at": "2026-01-01",
            "metadata": None,
            "_distance": 0.0,
        },
    ]

    resp = handle.search_dense(
        "model-x",
        [0.1, 0.2],
        top_k=5,
        filters=custom_filter,
        user_id=7,
        is_admin=False,
    )

    assert resp.status == "success"
    passed_filters = store.search_vectors_by_model.call_args.kwargs["filters"]
    assert isinstance(passed_filters, tuple)
    assert len(passed_filters) == 2
    assert passed_filters[1] == custom_filter


def test_search_dense_failure_returns_failed_response():
    handle, _, store, _ = _make_handle()
    store.create_index.side_effect = RuntimeError("boom")
    resp = handle.search_dense("model-x", [0.1])
    assert resp.status == "failed"
    assert resp.results == [] and resp.total_count == 0
    assert resp.index_status == IndexStatus.NO_INDEX
    assert any(w.code == "DENSE_SEARCH_FAILED" for w in resp.warnings)


def test_search_dense_capability_unsupported():
    handle, _, store, _ = _make_handle(supports_search=False)
    resp = handle.search_dense("model-x", [0.1])
    assert resp.status == "failed"
    assert any(w.code == "SEARCH_NOT_SUPPORTED" for w in resp.warnings)
    store.create_index.assert_not_called()  # guard is before any store access


@pytest.mark.asyncio
async def test_search_dense_async_capability_unsupported():
    handle, _, store, _ = _make_handle(supports_search=False)
    resp = await handle.search_dense_async("model-x", [0.1])
    assert resp.status == "failed"
    assert any(w.code == "SEARCH_NOT_SUPPORTED" for w in resp.warnings)


@pytest.mark.asyncio
async def test_search_dense_async_success():
    handle, ctx, store, _ = _make_handle()
    store.create_index.return_value = _index_result()
    store.search_vectors_by_model_async = AsyncMock(
        return_value=[
            {
                "doc_id": "d1",
                "chunk_id": "c1",
                "text": "t",
                "parse_hash": "h",
                "created_at": "2026-01-01",
                "metadata": None,
                "_distance": 1.0,
            },
        ]
    )
    resp = await handle.search_dense_async(
        "model-x", [0.1, 0.2], top_k=5, user_id=7, is_admin=False
    )
    assert isinstance(resp, DenseSearchResponse)
    assert resp.status == "success"
    assert resp.total_count == 1
    assert resp.results[0].score == pytest.approx(0.5)  # 1/(1+1.0)
    kwargs = store.search_vectors_by_model_async.call_args.kwargs
    assert kwargs["model_tag"] == "model-x"
    assert kwargs["user_id"] == 7 and kwargs["is_admin"] is False


def test_search_sparse_substring_fallback_honors_operator_filters():
    from xagent.core.tools.core.RAG_tools.core.schemas import IndexResult

    handle, ctx, mock_vector_store, _ = _make_handle()
    ctx.collection = "test_col"

    mock_table = MagicMock()
    mock_table.list_indices.return_value = [MagicMock(index_type="FTS", columns=["text"])]

    mock_vector_store.create_index.return_value = IndexResult(
        status="index_ready", advice=None, fts_enabled=True
    )
    mock_vector_store.build_filter_expression.return_value = "collection == 'test_col'"
    mock_vector_store.open_embeddings_table.return_value = (
        mock_table,
        "embeddings_test_model",
    )

    mock_search = MagicMock()
    mock_limit = MagicMock()
    mock_where = MagicMock()
    mock_table.search.return_value = mock_search
    mock_search.limit.return_value = mock_limit
    mock_limit.where.return_value = mock_where
    mock_where.to_pandas.return_value = pd.DataFrame()

    batch_df = pd.DataFrame(
        [
            {
                "collection": "test_col",
                "doc_id": "doc1",
                "chunk_id": "chunk1",
                "text": "hello world",
                "parse_hash": "hash1",
                "created_at": "2026-01-01",
                "metadata": None,
                "page_number": 1,
            },
            {
                "collection": "test_col",
                "doc_id": "doc2",
                "chunk_id": "chunk2",
                "text": "hello again",
                "parse_hash": "hash2",
                "created_at": "2026-01-02",
                "metadata": None,
                "page_number": 3,
            },
        ]
    )
    mock_batch = MagicMock()
    mock_batch.to_pandas.return_value = batch_df
    mock_table.to_batches.return_value = [mock_batch]

    filters = {"page_number": {"operator": "gte", "value": 2}}

    response = handle.search_sparse(
        "test_model",
        "hello",
        top_k=5,
        filters=filters,
        user_id=None,
        is_admin=True,
    )

    assert response.status == "success"
    assert len(response.results) == 1
    assert response.results[0].doc_id == "doc2"


@pytest.mark.asyncio
async def test_search_sparse_async_substring_fallback_honors_operator_filters():
    handle, ctx, mock_vector_store, _ = _make_handle()
    ctx.collection = "test_col"

    mock_vector_store.create_index.return_value = _index_result()
    mock_vector_store.search_fts_by_model_async = AsyncMock(return_value=[])
    mock_vector_store.open_embeddings_table.return_value = (
        MagicMock(),
        "embeddings_test_model",
    )

    batch_df = pd.DataFrame(
        [
            {
                "collection": "test_col",
                "doc_id": "doc1",
                "chunk_id": "chunk1",
                "text": "hello world",
                "parse_hash": "hash1",
                "created_at": "2026-01-01",
                "metadata": None,
                "page_number": 1,
            },
            {
                "collection": "test_col",
                "doc_id": "doc2",
                "chunk_id": "chunk2",
                "text": "hello again",
                "parse_hash": "hash2",
                "created_at": "2026-01-02",
                "metadata": None,
                "page_number": 3,
            },
        ]
    )
    mock_batch = MagicMock()
    mock_batch.to_pandas.return_value = batch_df

    async def iter_batches_async(**kwargs):
        yield mock_batch

    mock_vector_store.iter_batches_async = iter_batches_async

    with patch(
        "xagent.core.tools.core.RAG_tools.kb.collection_handle._safe_close_table"
    ):
        response = await handle.search_sparse_async(
            "test_model",
            "hello",
            top_k=5,
            filters={"page_number": {"operator": "gte", "value": 2}},
            user_id=None,
            is_admin=True,
        )

    assert response.status == "success"
    assert len(response.results) == 1
    assert response.results[0].doc_id == "doc2"


# ---------------------------------------------------------------------------
# Sparse search tests
# ---------------------------------------------------------------------------


def test_search_sparse_capability_unsupported():
    handle, _, store, _ = _make_handle(supports_search=False)
    resp = handle.search_sparse("model-x", "hello", top_k=3)
    assert isinstance(resp, SparseSearchResponse)
    assert resp.status == "failed"
    assert any(w.code == "SEARCH_NOT_SUPPORTED" for w in resp.warnings)
    store.open_embeddings_table.assert_not_called()


def test_search_sparse_fts_hit_scores_normalized():
    handle, _, store, _ = _make_handle()
    store.open_embeddings_table.return_value = (MagicMock(), "embeddings_model-x")
    store.create_index.return_value = _index_result()
    # Return None from build_filter_expression so the .where() branch is skipped
    # and the FTS result chain is: search().limit().to_pandas()
    store.build_filter_expression.return_value = None
    fts_table = store.open_embeddings_table.return_value[0]
    rows = pd.DataFrame(
        [
            {
                "doc_id": "d1",
                "chunk_id": "c1",
                "text": "hello",
                "parse_hash": "h",
                "created_at": "2026",
                "metadata": None,
                "_score": 3.0,
            }
        ]
    )
    fts_table.search.return_value.limit.return_value.to_pandas.return_value = rows
    resp = handle.search_sparse("model-x", "hello", top_k=3)
    assert resp.status == "success"
    assert resp.fts_enabled is True
    assert resp.results[0].score == pytest.approx(0.75)  # 3/(1+3)


# ---------------------------------------------------------------------------
# Hybrid search tests
# ---------------------------------------------------------------------------


def test_search_hybrid_capability_unsupported():
    from xagent.core.tools.core.RAG_tools.core.schemas import HybridSearchResponse

    handle, _, _, _ = _make_handle(supports_search=False)
    resp = handle.search_hybrid("model-x", "q", [0.1], top_k=5)
    assert isinstance(resp, HybridSearchResponse)
    assert resp.status == "failed"
    assert any(w.code == "SEARCH_NOT_SUPPORTED" for w in resp.warnings)
    assert resp.dense_count == 0 and resp.sparse_count == 0


def test_search_hybrid_fetches_double_top_k_and_fuses(monkeypatch):
    from xagent.core.tools.core.RAG_tools.core.schemas import (
        DenseSearchResponse,
        IndexStatus,
        SearchResult,
        SparseSearchResponse,
    )

    handle, _, _, _ = _make_handle()
    dense = DenseSearchResponse(
        results=[
            SearchResult(
                doc_id="d",
                chunk_id="c",
                text="t",
                score=0.9,
                parse_hash="h",
                model_tag="model-x",
                created_at="2026",
                metadata=None,
            )
        ],
        total_count=1,
        status="success",
        warnings=[],
        index_status=IndexStatus.INDEX_READY,
        index_advice=None,
        idempotency_key=None,
        fallback_info=None,
        nprobes=None,
        refine_factor=None,
    )
    sparse = SparseSearchResponse(
        results=[],
        total_count=0,
        status="success",
        warnings=[],
        fts_enabled=True,
        query_text="q",
    )
    captured = {}

    def fake_dense(self, model_tag, query_vector, *, top_k, **kw):
        captured["dense_top_k"] = top_k
        return dense

    def fake_sparse(self, model_tag, query_text, *, top_k, **kw):
        captured["sparse_top_k"] = top_k
        return sparse

    monkeypatch.setattr(type(handle), "search_dense", fake_dense)
    monkeypatch.setattr(type(handle), "search_sparse", fake_sparse)
    resp = handle.search_hybrid("model-x", "q", [0.1], top_k=5)
    assert captured["dense_top_k"] == 10 and captured["sparse_top_k"] == 10  # top_k*2
    assert resp.status in ("success", "partial_success")
    assert resp.dense_count == 1 and resp.sparse_count == 0
    # The fused output contains the dense result with the original score/rank
    # attached: vector_* from dense, fts_* unset because sparse missed.
    assert len(resp.results) == 1
    fused = resp.results[0]
    assert fused.doc_id == "d" and fused.chunk_id == "c"
    assert fused.vector_score == pytest.approx(0.9)  # dense original score
    assert fused.vector_rank == 1
    assert fused.fts_score is None and fused.fts_rank is None


def test_search_hybrid_linear_fusion_attaches_scores(monkeypatch):
    from xagent.core.tools.core.RAG_tools.core.schemas import (
        DenseSearchResponse,
        FusionConfig,
        FusionStrategy,
        IndexStatus,
        SearchResult,
        SparseSearchResponse,
    )

    handle, _, _, _ = _make_handle()
    dense = DenseSearchResponse(
        results=[
            SearchResult(
                doc_id="d",
                chunk_id="c",
                text="t",
                score=0.9,
                parse_hash="h",
                model_tag="model-x",
                created_at="2026",
                metadata=None,
            )
        ],
        total_count=1,
        status="success",
        warnings=[],
        index_status=IndexStatus.INDEX_READY,
        index_advice=None,
        idempotency_key=None,
        fallback_info=None,
        nprobes=None,
        refine_factor=None,
    )
    sparse = SparseSearchResponse(
        results=[
            SearchResult(
                doc_id="d2",
                chunk_id="c2",
                text="t2",
                score=0.7,
                parse_hash="h",
                model_tag="model-x",
                created_at="2026",
                metadata=None,
            )
        ],
        total_count=1,
        status="success",
        warnings=[],
        fts_enabled=True,
        query_text="q",
    )

    def fake_dense(self, model_tag, query_vector, *, top_k, **kw):
        return dense

    def fake_sparse(self, model_tag, query_text, *, top_k, **kw):
        return sparse

    monkeypatch.setattr(type(handle), "search_dense", fake_dense)
    monkeypatch.setattr(type(handle), "search_sparse", fake_sparse)
    cfg = FusionConfig(
        strategy=FusionStrategy.LINEAR,
        dense_weight=0.6,
        sparse_weight=0.4,
        normalize_scores=True,
    )
    resp = handle.search_hybrid("model-x", "q", [0.1], top_k=5, fusion_config=cfg)
    assert resp.status in ("success", "partial_success")
    assert resp.dense_count == 1 and resp.sparse_count == 1
    assert resp.fusion_config.strategy == FusionStrategy.LINEAR
    # Linear fusion combines both lists; both unique results survive.
    assert len(resp.results) == 2
    by_doc = {r.doc_id: r for r in resp.results}
    # Dense-only doc carries its dense score/rank, no fts attachment.
    assert by_doc["d"].vector_score == pytest.approx(0.9)
    assert by_doc["d"].vector_rank == 1
    assert by_doc["d"].fts_score is None and by_doc["d"].fts_rank is None
    # Sparse-only doc carries its fts score/rank, no vector attachment.
    assert by_doc["d2"].fts_score == pytest.approx(0.7)
    assert by_doc["d2"].fts_rank == 1
    assert by_doc["d2"].vector_score is None and by_doc["d2"].vector_rank is None


@pytest.mark.asyncio
async def test_search_hybrid_async_capability_unsupported():
    from xagent.core.tools.core.RAG_tools.core.schemas import HybridSearchResponse

    handle, _, _, _ = _make_handle(supports_search=False)
    resp = await handle.search_hybrid_async("model-x", "q", [0.1], top_k=5)
    assert isinstance(resp, HybridSearchResponse)
    assert resp.status == "failed"
    assert any(w.code == "SEARCH_NOT_SUPPORTED" for w in resp.warnings)
    assert resp.dense_count == 0 and resp.sparse_count == 0


@pytest.mark.asyncio
async def test_search_hybrid_async_fetches_double_top_k_and_fuses(monkeypatch):
    from xagent.core.tools.core.RAG_tools.core.schemas import (
        DenseSearchResponse,
        IndexStatus,
        SearchResult,
        SparseSearchResponse,
    )

    handle, _, _, _ = _make_handle()
    dense = DenseSearchResponse(
        results=[
            SearchResult(
                doc_id="d",
                chunk_id="c",
                text="t",
                score=0.9,
                parse_hash="h",
                model_tag="model-x",
                created_at="2026",
                metadata=None,
            )
        ],
        total_count=1,
        status="success",
        warnings=[],
        index_status=IndexStatus.INDEX_READY,
        index_advice=None,
        idempotency_key=None,
        fallback_info=None,
        nprobes=None,
        refine_factor=None,
    )
    sparse = SparseSearchResponse(
        results=[],
        total_count=0,
        status="success",
        warnings=[],
        fts_enabled=True,
        query_text="q",
    )
    captured = {}

    async def fake_dense_async(self, model_tag, query_vector, *, top_k, **kw):
        captured["dense_top_k"] = top_k
        return dense

    async def fake_sparse_async(self, model_tag, query_text, *, top_k, **kw):
        captured["sparse_top_k"] = top_k
        return sparse

    monkeypatch.setattr(type(handle), "search_dense_async", fake_dense_async)
    monkeypatch.setattr(type(handle), "search_sparse_async", fake_sparse_async)
    resp = await handle.search_hybrid_async("model-x", "q", [0.1], top_k=5)
    assert captured["dense_top_k"] == 10 and captured["sparse_top_k"] == 10  # top_k*2
    assert resp.status in ("success", "partial_success")
    assert resp.dense_count == 1 and resp.sparse_count == 0
    # _fuse_hybrid is shared; verify it attaches vector_score/rank via the async path too.
    assert len(resp.results) == 1
    fused = resp.results[0]
    assert fused.doc_id == "d" and fused.vector_score == pytest.approx(0.9)
    assert fused.vector_rank == 1
    assert fused.fts_score is None and fused.fts_rank is None


# ---------------------------------------------------------------------------
# Cross-mode capability-degradation tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda h: h.search_dense("m", [0.1], top_k=3),
        lambda h: h.search_sparse("m", "q", top_k=3),
        lambda h: h.search_hybrid("m", "q", [0.1], top_k=3),
    ],
)
def test_all_modes_degrade_when_search_unsupported(call):
    handle, _, _, _ = _make_handle(supports_search=False)
    resp = call(handle)
    assert resp.status == "failed"
    assert any(w.code == "SEARCH_NOT_SUPPORTED" for w in resp.warnings)


# ---------------------------------------------------------------------------
# Rollback-scope filter test
# ---------------------------------------------------------------------------


def test_dense_passes_collection_and_user_scope_to_store():
    # Rollback-incomplete remnants must still be filtered: assert the store call
    # always carries the collection filter + user scope so a stray row in another
    # collection / another user cannot leak.
    handle, ctx, store, _ = _make_handle()
    store.create_index.return_value = _index_result()
    store.search_vectors_by_model.return_value = []
    handle.search_dense("m", [0.1], top_k=3, user_id=42, is_admin=False)
    kwargs = store.search_vectors_by_model.call_args.kwargs
    assert kwargs["user_id"] == 42 and kwargs["is_admin"] is False
    # collection filter present in the FilterExpression
    assert kwargs["filters"] is not None


@pytest.mark.asyncio
async def test_search_hybrid_async_linear_fusion(monkeypatch):
    """LINEAR fusion path exercised via the async hybrid method."""
    from xagent.core.tools.core.RAG_tools.core.schemas import (
        DenseSearchResponse,
        FusionConfig,
        FusionStrategy,
        IndexStatus,
        SearchResult,
        SparseSearchResponse,
    )

    handle, _, _, _ = _make_handle()
    dense = DenseSearchResponse(
        results=[
            SearchResult(
                doc_id="da",
                chunk_id="ca",
                text="ta",
                score=0.8,
                parse_hash="ha",
                model_tag="model-x",
                created_at="2026",
                metadata=None,
            )
        ],
        total_count=1,
        status="success",
        warnings=[],
        index_status=IndexStatus.INDEX_READY,
        index_advice=None,
        idempotency_key=None,
        fallback_info=None,
        nprobes=None,
        refine_factor=None,
    )
    sparse = SparseSearchResponse(
        results=[
            SearchResult(
                doc_id="db",
                chunk_id="cb",
                text="tb",
                score=0.6,
                parse_hash="hb",
                model_tag="model-x",
                created_at="2026",
                metadata=None,
            )
        ],
        total_count=1,
        status="success",
        warnings=[],
        fts_enabled=True,
        query_text="q",
    )

    async def fake_dense_async(self, model_tag, query_vector, *, top_k, **kw):
        return dense

    async def fake_sparse_async(self, model_tag, query_text, *, top_k, **kw):
        return sparse

    monkeypatch.setattr(type(handle), "search_dense_async", fake_dense_async)
    monkeypatch.setattr(type(handle), "search_sparse_async", fake_sparse_async)
    cfg = FusionConfig(
        strategy=FusionStrategy.LINEAR,
        dense_weight=0.5,
        sparse_weight=0.5,
        normalize_scores=True,
    )
    resp = await handle.search_hybrid_async(
        "model-x", "q", [0.1], top_k=5, fusion_config=cfg
    )
    assert resp.status in ("success", "partial_success")
    assert resp.fusion_config.strategy == FusionStrategy.LINEAR
    assert resp.dense_count == 1 and resp.sparse_count == 1
    assert len(resp.results) == 2
    by_doc = {r.doc_id: r for r in resp.results}
    assert by_doc["da"].vector_rank == 1 and by_doc["da"].fts_score is None
    assert by_doc["db"].fts_rank == 1 and by_doc["db"].vector_score is None


# ---------------------------------------------------------------------------
# Issue #72: Ported from test_retrieval_helper_compatibility.py
# Exact collection-filter equality assertions moved from facade engine tests.
# ---------------------------------------------------------------------------


def _filter_conditions(expr):
    """Extract (field, operator, value) triples from a FilterExpression tree."""
    if expr is None:
        return []
    if isinstance(expr, (tuple, list)):
        conditions = []
        for item in expr:
            conditions.extend(_filter_conditions(item))
        return conditions
    operator = getattr(expr, "operator", None)
    return [
        (
            getattr(expr, "field"),
            getattr(operator, "value", operator),
            getattr(expr, "value"),
        )
    ]


def test_dense_collection_filter_equality_issue_72():
    """search_dense always applies an exact-match collection filter (Issue #72).

    Ported from test_retrieval_facade_preserves_sync_tuple_filter_scope_and_conversion.
    The handle must build FilterCondition(field="collection", operator=EQ, value=collection)
    and pass it to store.search_vectors_by_model — not just any non-None filter.
    """
    handle, ctx, store, _ = _make_handle()
    ctx.collection = "docs"
    store.create_index.return_value = _index_result()
    store.search_vectors_by_model.return_value = []
    handle.search_dense("model-a", [0.5, 0.25], top_k=5, user_id=7, is_admin=False)
    kwargs = store.search_vectors_by_model.call_args.kwargs
    conditions = _filter_conditions(kwargs["filters"])
    # The first condition MUST be the collection equality filter for Issue #72.
    assert ("collection", "eq", "docs") in conditions, (
        "Issue #72: collection equality filter missing from store call"
    )


@pytest.mark.asyncio
async def test_dense_async_collection_filter_equality_issue_72():
    """search_dense_async always applies an exact-match collection filter (Issue #72)."""
    from unittest.mock import AsyncMock

    handle, ctx, store, _ = _make_handle()
    ctx.collection = "docs"
    store.create_index.return_value = _index_result()
    store.search_vectors_by_model_async = AsyncMock(return_value=[])
    await handle.search_dense_async(
        "model-a", [0.5], top_k=3, user_id=None, is_admin=True
    )
    kwargs = store.search_vectors_by_model_async.call_args.kwargs
    conditions = _filter_conditions(kwargs["filters"])
    assert ("collection", "eq", "docs") in conditions, (
        "Issue #72: collection equality filter missing from async store call"
    )


def test_dense_invalid_filter_returns_failed_response():
    """search_dense returns a failed response for unknown filter operators.

    Ported from test_retrieval_facade_preserves_invalid_legacy_filter_errors.
    The handle catches parse errors and returns a structured failed response.
    """
    handle, ctx, store, _ = _make_handle()
    ctx.collection = "docs"
    store.create_index.return_value = _index_result()
    # Unknown operator causes parse failure; handle returns failed response
    resp = handle.search_dense(
        "model-a",
        [0.5],
        top_k=5,
        filters={"page_number": {"operator": "between", "value": [1, 3]}},
        user_id=7,
        is_admin=False,
    )
    assert resp.status == "failed"
    assert any(w.code == "DENSE_SEARCH_FAILED" for w in resp.warnings)
    assert any("between" in w.message for w in resp.warnings)
