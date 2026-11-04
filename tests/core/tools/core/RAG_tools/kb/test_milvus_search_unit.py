"""Milvus search without a server: filter translation, score and LIKE mappings, and
how the handle drives the client (scope, concurrency, fallback, failures)."""

from __future__ import annotations

import math
import threading
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import (
    DatabaseOperationError,
    DocumentValidationError,
)
from xagent.core.tools.core.RAG_tools.core.schemas import (
    FusionConfig,
    FusionStrategy,
    IndexStatus,
    SearchFallbackAction,
)
from xagent.core.tools.core.RAG_tools.kb import collection_handle
from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    KBCollectionHandle,
    MilvusCollectionHandle,
    milvus_collection_name,
)
from xagent.core.tools.core.RAG_tools.kb.milvus_search import (
    caller_filter,
    dense_score,
    keyword_score,
    like_pattern,
    to_result,
)
from xagent.core.tools.core.RAG_tools.kb.models import (
    KBAccessMode,
    KBBackendCapabilities,
    KBCollectionContext,
    KBStorageBackend,
    KBUserScope,
)
from xagent.providers.vector_store.milvus import MilvusConnectionManager

MODEL = "BAAI/bge-m3"
SCOPE = 'kb_id in ["kb-1", "kb-2"] and visible == true'


@pytest.mark.parametrize(
    ("filters", "expr", "params"),
    [
        ({"doc_id": "d1"}, "(doc_id == {p0})", {"p0": "d1"}),
        (
            {"chunk_id": {"operator": "ne", "value": "c1"}},
            "(chunk_id != {p0})",
            {"p0": "c1"},
        ),
        (
            {"metadata.page": {"operator": "gt", "value": 2}},
            '(metadata["page"] > {p0})',
            {"p0": 2},
        ),
        (
            {"metadata.page": {"operator": "gte", "value": 2}},
            '(metadata["page"] >= {p0})',
            {"p0": 2},
        ),
        (
            {"metadata.score": {"operator": "lt", "value": 0.5}},
            '(metadata["score"] < {p0})',
            {"p0": 0.5},
        ),
        (
            {"metadata.page": {"operator": "lte", "value": 9}},
            '(metadata["page"] <= {p0})',
            {"p0": 9},
        ),
        ({"metadata.draft": False}, '(metadata["draft"] == {p0})', {"p0": False}),
        (
            {"metadata.a.b-c": "x"},
            '(metadata["a"]["b-c"] == {p0})',
            {"p0": "x"},
        ),
        (
            {"doc_id": {"operator": "in", "value": ["d1", "d2"]}},
            "(doc_id in {p0})",
            {"p0": ["d1", "d2"]},
        ),
        (
            {"text": {"operator": "contains", "value": "kiwi"}},
            "(text like {p0})",
            {"p0": "%kiwi%"},
        ),
        (
            {"metadata.页码": 1},
            '(metadata["页码"] == {p0})',
            {"p0": 1},
        ),
    ],
)
def test_caller_filter_translates_each_operator_to_a_template(
    filters: dict[str, Any], expr: str, params: dict[str, Any]
) -> None:
    assert caller_filter(filters) == (expr, params)


def test_caller_filter_ands_conditions_in_order_with_distinct_parameters() -> None:
    expr, params = caller_filter(
        {"doc_id": "d1", "metadata.page": {"operator": "gte", "value": 2}}
    )

    assert expr == '(doc_id == {p0}) and (metadata["page"] >= {p1})'
    assert params == {"p0": "d1", "p1": 2}


@pytest.mark.parametrize("filters", [None, {}])
def test_caller_filter_without_conditions_adds_nothing(
    filters: dict[str, Any] | None,
) -> None:
    assert caller_filter(filters) == ("", {})


@pytest.mark.parametrize(
    "filters",
    [
        {"created_at": 1},
        {"collection": "kb"},
        {"user_id": 1},
        {"kb_id": "other"},
        {"visible": False},
        {"metadata": "x"},
        {"metadata.": "x"},
        {"metadata..a": "x"},
        {'metadata.a"] or ["b': "x"},
        {"metadata.a b": "x"},
        {"doc_id; drop": "x"},
        {"doc_id": None},
        {"doc_id": {"nested": 1}},
        {"doc_id": {"operator": "in", "value": ["d", None]}},
        {"doc_id": {"operator": "in", "value": [["d"]]}},
        {"text": {"operator": "contains", "value": "50%"}},
        {"text": {"operator": "contains", "value": "a_b"}},
        {"text": {"operator": "contains", "value": "C:\\Users"}},
        {"text": {"operator": "contains", "value": 5}},
        {"doc_id": {"operator": "regex", "value": "x"}},
    ],
)
def test_caller_filter_raises_for_what_it_cannot_translate(
    filters: dict[str, Any],
) -> None:
    with pytest.raises(DocumentValidationError):
        caller_filter(filters)


def test_caller_filter_rejects_a_filter_that_is_not_a_dict() -> None:
    with pytest.raises(DocumentValidationError, match="must be a dict"):
        caller_filter([("doc_id", "d1")])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "vector", [(1.0, 0.0), (0.6, 0.8), (0.0, 1.0), (-0.6, 0.8), (-1.0, 0.0)]
)
def test_dense_score_matches_lancedb_for_unit_vectors(
    vector: tuple[float, float],
) -> None:
    query = (1.0, 0.0)
    cosine = vector[0] * query[0] + vector[1] * query[1]
    squared_l2 = sum((a - b) ** 2 for a, b in zip(vector, query))

    assert dense_score(cosine) == pytest.approx(1.0 / (1.0 + squared_l2))


def test_dense_score_ranks_the_nearer_vector_higher_and_stays_in_the_unit_interval() -> (
    None
):
    cosines = [1.0000001, 1.0, 0.99, 0.5, 0.0, -0.5, -1.0, -1.0000001]
    scores = [dense_score(cosine) for cosine in cosines]

    assert scores == sorted(scores, reverse=True)
    assert all(0.0 < score <= 1.0 for score in scores)
    assert scores[0] == scores[1] == 1.0
    assert scores[-1] == scores[-2] == pytest.approx(0.2)


def test_keyword_score_squashes_bm25_like_the_lancedb_fts_score() -> None:
    assert keyword_score(0.0) == 0.0
    assert keyword_score(1.0) == 0.5
    assert keyword_score(3.0) == pytest.approx(0.75)
    assert keyword_score(-0.2) == 0.0
    scores = [keyword_score(raw) for raw in (0.1, 1.0, 10.0, 1e6)]
    assert scores == sorted(scores) and all(0.0 <= score < 1.0 for score in scores)


@pytest.mark.parametrize(
    ("term", "pattern"),
    [
        ("kiwi", "%kiwi%"),
        ("向量 检索", "%向量 检索%"),
        ("50%", "%50_%"),
        ("%50", "%_50%"),
        ("%", "%_%"),
        ("a_b", "%a_b%"),
        ("C:\\Users", "%C:_Users%"),
        ("\\%", "%__%"),
        ("\\_", "%__%"),
        ('say "hi"', '%say "hi"%'),
        ("it's", "%it's%"),
    ],
)
def test_like_pattern_turns_special_characters_into_wildcards(
    term: str, pattern: str
) -> None:
    assert like_pattern(term) == pattern


def test_a_search_row_becomes_a_result_with_a_naive_utc_timestamp() -> None:
    entity = {
        "doc_id": "d",
        "chunk_id": "c",
        "text": "t",
        "parse_hash": "p",
        "created_at": 1_700_000_000,
        "metadata": {"page": 1},
    }

    result = to_result(entity, 0.5, MODEL)

    assert (result.doc_id, result.chunk_id, result.text, result.score) == (
        "d",
        "c",
        "t",
        0.5,
    )
    assert (result.parse_hash, result.model_tag, result.metadata) == (
        "p",
        MODEL,
        {"page": 1},
    )
    assert result.created_at == datetime(2023, 11, 14, 22, 13, 20)
    assert result.created_at.tzinfo is None


class NotLoaded(Exception):
    code = 101


class FakeClient:
    """Answers search and query with canned rows and records every call."""

    def __init__(
        self,
        dense: list[tuple[str, float]] | None = None,
        sparse: list[tuple[str, float]] | None = None,
        rows: list[dict[str, Any]] | None = None,
        pages: list[list[dict[str, Any]]] | None = None,
        queries_before_failure: int | None = None,
    ) -> None:
        self.dense = dense or []
        self.sparse = sparse or []
        self.rows = rows or []
        self.pages = pages
        self.queries_before_failure = queries_before_failure
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.barrier: threading.Barrier | None = None
        self.error: Exception | None = None

    @staticmethod
    def entity(chunk_id: str, text: str | None = None) -> dict[str, Any]:
        return {
            "doc_id": f"doc-{chunk_id}",
            "chunk_id": chunk_id,
            "text": text or f"text {chunk_id}",
            "parse_hash": "ph",
            "created_at": 1_700_000_000,
            "metadata": {"page": 1},
        }

    def search(self, name: str, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.calls.append(("search", {"name": name, **kwargs}))
        if self.barrier is not None:
            self.barrier.wait()
        if self.error is not None:
            raise self.error
        canned = self.dense if kwargs["anns_field"] == "dense" else self.sparse
        return [
            [
                {"id": chunk_id, "distance": distance, "entity": self.entity(chunk_id)}
                for chunk_id, distance in canned[: kwargs["limit"]]
            ]
        ]

    def query(self, name: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(("query", {"name": name, **kwargs}))
        if self.error is not None:
            raise self.error
        queries = sum(method == "query" for method, _ in self.calls)
        if self.queries_before_failure is not None and (
            queries > self.queries_before_failure
        ):
            raise RuntimeError("query failed")
        if self.pages is not None:
            return self.pages[queries - 1]
        offset = kwargs["offset"]
        return self.rows[offset : offset + kwargs["limit"]]


def client_row(chunk_id: str = "r", text: str = "kiwi") -> dict[str, Any]:
    return FakeClient.entity(chunk_id, text)


def _handle(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch, kb_ids: list[str] | None
) -> MilvusCollectionHandle:
    context = KBCollectionContext(
        collection="kb",
        user_scope=KBUserScope(user_id=1, is_admin=False),
        access_mode=KBAccessMode.READ,
        allow_create=False,
        hide_missing=True,
        metadata_store=MagicMock(),
        vector_index_store=MagicMock(),
        ingestion_status_store=MagicMock(),
        main_pointer_store=MagicMock(),
        backend=KBStorageBackend.MILVUS,
        capabilities=KBBackendCapabilities.milvus(),
    )
    connections = MagicMock(spec=MilvusConnectionManager)
    connections.get_shared_client_from_env.return_value = client
    monkeypatch.setattr(
        MilvusCollectionHandle,
        "_kb_ids",
        lambda self, user_id, is_admin: ["kb-1", "kb-2"] if kb_ids is None else kb_ids,
    )
    return MilvusCollectionHandle(
        context, ledger=MagicMock(spec=KBCollectionHandle), connections=connections
    )


def _all_routes(handle: MilvusCollectionHandle, **scope: Any) -> list[Any]:
    return [
        handle.search_dense(MODEL, [1.0, 0.0], top_k=3, **scope),
        handle.search_sparse(MODEL, "kiwi", top_k=3, **scope),
        handle.search_hybrid(MODEL, "kiwi", [1.0, 0.0], top_k=3, **scope),
    ]


def test_every_route_applies_the_same_scope_and_caller_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient(dense=[("a", 1.0)], sparse=[("a", 1.0)])
    handle = _handle(client, monkeypatch, None)

    _all_routes(handle, filters={"doc_id": "d1"}, user_id=1, is_admin=False)

    calls = [kwargs for method, kwargs in client.calls if method == "search"]
    assert len(calls) == 4
    assert {
        (call["filter"], tuple(call["filter_params"].items())) for call in calls
    } == {(f"{SCOPE} and (doc_id == {{p0}})", (("p0", "d1"),))}
    assert {call["name"] for call in calls} == {milvus_collection_name(MODEL)}
    assert {call["anns_field"] for call in calls} == {"dense", "sparse"}
    assert handle.ledger.mock_calls == []


def test_the_collection_comes_from_the_model_id_not_its_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient(dense=[("a", 1.0)])
    handle = _handle(client, monkeypatch, None)

    handle.search_dense(" BAAI/bge-m3 ", [1.0, 0.0], top_k=1)
    handle.search_dense("BAAI_bge_m3", [1.0, 0.0], top_k=1)

    assert [kwargs["name"] for _, kwargs in client.calls] == [
        milvus_collection_name(MODEL),
        milvus_collection_name("BAAI_bge_m3"),
    ]
    assert milvus_collection_name("BAAI_bge_m3") != milvus_collection_name(MODEL)


def test_a_caller_who_reads_no_kb_id_reaches_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient(dense=[("a", 1.0)], sparse=[("a", 1.0)], rows=[client_row()])
    handle = _handle(client, monkeypatch, [])

    dense, sparse, hybrid = _all_routes(handle, user_id=None, is_admin=False)

    assert client.calls == []
    for response in (dense, sparse, hybrid):
        assert (response.status, response.results, response.warnings) == (
            "success",
            [],
            [],
        )
        assert response.total_count == 0


@pytest.mark.parametrize("kb_ids", [None, []])
def test_an_untranslatable_filter_raises_before_any_call_for_every_caller(
    monkeypatch: pytest.MonkeyPatch, kb_ids: list[str] | None
) -> None:
    client = FakeClient(dense=[("a", 1.0)], sparse=[("a", 1.0)])
    handle = _handle(client, monkeypatch, kb_ids)

    for search in (
        lambda: handle.search_dense(MODEL, [1.0], top_k=1, filters={"created_at": 1}),
        lambda: handle.search_sparse(MODEL, "kiwi", top_k=1, filters={"kb_id": "x"}),
        lambda: handle.search_hybrid(
            MODEL, "kiwi", [1.0], top_k=1, filters={"collection": "x"}
        ),
    ):
        with pytest.raises(DocumentValidationError):
            search()
    assert client.calls == []


def test_scores_and_shapes_follow_the_lancedb_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient(dense=[("a", 1.0), ("b", 0.6)], sparse=[("a", 3.0)])
    handle = _handle(client, monkeypatch, None)

    dense, sparse, _ = _all_routes(handle, nprobes=7, refine_factor=2)

    assert [r.chunk_id for r in dense.results] == ["a", "b"]
    assert [r.score for r in dense.results] == pytest.approx([1.0, 1.0 / 1.8])
    assert (dense.status, dense.total_count, dense.index_status) == (
        "success",
        2,
        IndexStatus.INDEX_READY,
    )
    assert (dense.nprobes, dense.refine_factor, dense.warnings) == (7, 2, [])
    assert [r.score for r in sparse.results] == pytest.approx([0.75])
    assert (sparse.status, sparse.fts_enabled, sparse.query_text) == (
        "success",
        True,
        "kiwi",
    )
    first = dense.results[0]
    assert (first.doc_id, first.text, first.parse_hash, first.model_tag) == (
        "doc-a",
        "text a",
        "ph",
        MODEL,
    )
    assert first.metadata == {"page": 1}


@pytest.mark.parametrize("strategy", [FusionStrategy.RRF, FusionStrategy.LINEAR])
def test_hybrid_fuses_two_times_top_k_per_route_and_keeps_route_scores(
    monkeypatch: pytest.MonkeyPatch, strategy: FusionStrategy
) -> None:
    dense = [("a", 1.0), ("b", 0.5), ("c", 0.1), ("d", 0.0), ("e", -0.1)]
    sparse = [("b", 2.0), ("c", 1.0), ("f", 0.5), ("g", 0.2), ("h", 0.1)]
    client = FakeClient(dense=dense, sparse=sparse)
    handle = _handle(client, monkeypatch, None)
    config = FusionConfig(strategy=strategy)

    response = handle.search_hybrid(
        MODEL, "kiwi", [1.0, 0.0], top_k=2, fusion_config=config
    )

    limits = {kw["anns_field"]: kw["limit"] for _, kw in client.calls}
    assert limits == {"dense": 4, "sparse": 4}
    assert (response.dense_count, response.sparse_count) == (4, 4)
    assert response.fusion_config == config
    assert response.index_status is IndexStatus.INDEX_READY
    assert len(response.results) == response.total_count == 2
    dense_ranks = {c: rank for rank, (c, _) in enumerate(dense[:4], start=1)}
    sparse_ranks = {c: rank for rank, (c, _) in enumerate(sparse[:4], start=1)}
    for result in response.results:
        chunk = result.chunk_id
        assert result.vector_rank == dense_ranks.get(chunk)
        assert result.fts_rank == sparse_ranks.get(chunk)
        assert (result.vector_score is None) == (chunk not in dense_ranks)
        assert (result.fts_score is None) == (chunk not in sparse_ranks)
    if strategy is FusionStrategy.RRF:
        assert [r.chunk_id for r in response.results] == ["b", "c"]
        top = response.results[0]
        assert top.score == pytest.approx(1 / 62 + 1 / 61)
        assert top.vector_score == pytest.approx(0.5)
        assert top.fts_score == pytest.approx(2.0 / 3.0)


def test_hybrid_fuses_with_the_module_function_the_lancedb_handle_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fuse(*args: Any, **kwargs: Any) -> str:
        seen.append((args, kwargs))
        return "fused"

    monkeypatch.setattr(collection_handle, "_fuse_hybrid", fuse)
    client = FakeClient(dense=[("a", 1.0)], sparse=[("a", 1.0)])
    handle = _handle(client, monkeypatch, None)

    result = handle.search_hybrid(MODEL, "kiwi", [1.0, 0.0], top_k=2)

    assert result == "fused"
    ((args, kwargs),) = seen
    assert args[:2] == (MODEL, "kiwi")
    assert [r.chunk_id for r in args[2].results] == ["a"]
    assert [r.chunk_id for r in args[3].results] == ["a"]
    assert kwargs["top_k"] == 2 and kwargs["fusion_config"] == FusionConfig()


def test_hybrid_sends_the_dense_and_keyword_queries_at_the_same_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient(dense=[("a", 1.0)], sparse=[("a", 1.0)])
    client.barrier = threading.Barrier(2, timeout=5)
    handle = _handle(client, monkeypatch, None)

    response = handle.search_hybrid(MODEL, "kiwi", [1.0, 0.0], top_k=1)

    assert response.status == "success"
    assert response.warnings == []
    assert [r.chunk_id for r in response.results] == ["a"]


def test_an_unloaded_collection_reads_as_empty_on_every_route(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client = FakeClient(dense=[("a", 1.0)], sparse=[("a", 1.0)])
    client.error = NotLoaded("collection not loaded")
    handle = _handle(client, monkeypatch, None)

    with caplog.at_level("WARNING"):
        dense, sparse, hybrid = _all_routes(handle, user_id=1, is_admin=False)

    for response in (dense, sparse, hybrid):
        assert (response.status, response.results, response.warnings) == (
            "success",
            [],
            [],
        )
    assert "not loaded; treated as empty" in caplog.text


class CollectionNotFound(Exception):
    code = 100


@pytest.mark.parametrize("error", [RuntimeError("boom"), CollectionNotFound("gone")])
def test_a_failing_route_becomes_a_failed_response_not_a_raise(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    client = FakeClient(dense=[("a", 1.0)], sparse=[("a", 1.0)])
    client.error = error
    handle = _handle(client, monkeypatch, None)

    dense, sparse, hybrid = _all_routes(handle, nprobes=3)

    assert (dense.status, dense.results, dense.index_status) == (
        "failed",
        [],
        IndexStatus.NO_INDEX,
    )
    assert dense.nprobes == 3
    assert [(w.code, w.fallback_action, w.affected_models) for w in dense.warnings] == [
        ("DENSE_SEARCH_FAILED", SearchFallbackAction.PARTIAL_RESULTS, [MODEL])
    ]
    assert str(error) in dense.warnings[0].message
    assert (sparse.status, sparse.results) == ("failed", [])
    assert [w.code for w in sparse.warnings] == ["FTS_SEARCH_FAILED"]
    assert (hybrid.status, hybrid.results) == ("partial_success", [])
    assert [w.code for w in hybrid.warnings] == [
        "DENSE_SEARCH_FAILED",
        "FTS_SEARCH_FAILED",
    ]


def test_a_keyword_miss_falls_back_to_like_and_keeps_only_real_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(collection_handle, "_MILVUS_FALLBACK_PAGE", 2)
    rows = [client_row(f"x{i}", "aXb") for i in range(4)]
    rows.insert(2, client_row("w1", "say a_b now"))
    rows.append(client_row("w2", "a_b"))
    client = FakeClient(rows=rows)
    handle = _handle(client, monkeypatch, None)

    response = handle.search_sparse(MODEL, "a_b", top_k=2)

    assert [r.chunk_id for r in response.results] == ["w1", "w2"]
    assert [r.score for r in response.results] == [1.0, 1.0]
    assert [(w.code, w.fallback_action) for w in response.warnings] == [
        ("FTS_FALLBACK", SearchFallbackAction.BRUTE_FORCE)
    ]
    assert response.status == "success"
    queries = [kw for method, kw in client.calls if method == "query"]
    assert [(q["offset"], q["limit"]) for q in queries] == [(0, 2), (2, 2), (4, 2)]
    assert {q["filter"] for q in queries} == {f"{SCOPE} and text like {{pattern}}"}
    assert {q["filter_params"]["pattern"] for q in queries} == {"%a_b%"}
    assert {tuple(q["output_fields"]) for q in queries} == {
        ("doc_id", "chunk_id", "text", "parse_hash", "created_at", "metadata")
    }


def test_the_fallback_ends_at_a_short_page_and_without_matches_adds_no_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(collection_handle, "_MILVUS_FALLBACK_PAGE", 2)
    client = FakeClient(rows=[client_row(f"x{i}", "aXb") for i in range(3)])
    handle = _handle(client, monkeypatch, None)

    response = handle.search_sparse(MODEL, "a_b", top_k=2)

    assert (response.status, response.results, response.warnings) == (
        "success",
        [],
        [],
    )
    queries = [kw for method, kw in client.calls if method == "query"]
    assert [(q["offset"], q["limit"]) for q in queries] == [(0, 2), (2, 2)]


def test_the_fallback_returns_top_k_rows_when_the_last_page_overshoots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [client_row(f"x{i}", "aXb") for i in range(2)]
    rows += [client_row(f"w{i}", "a_b") for i in range(1, 5)]
    handle = _handle(FakeClient(rows=rows), monkeypatch, None)

    response = handle.search_sparse(MODEL, "a_b", top_k=3)

    assert [r.chunk_id for r in response.results] == ["w1", "w2", "w3"]
    assert response.total_count == 3


def test_the_fallback_keeps_a_row_once_when_pages_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(collection_handle, "_MILVUS_FALLBACK_PAGE", 2)
    hit, other = client_row("w1", "a_b"), client_row("w2", "a_b")
    pages = [[hit, client_row("x0", "aXb")], [hit, other]]
    handle = _handle(FakeClient(pages=pages), monkeypatch, None)

    response = handle.search_sparse(MODEL, "a_b", top_k=2)

    assert [r.chunk_id for r in response.results] == ["w1", "w2"]


def test_a_ledger_failure_while_reading_kb_ids_is_raised_not_answered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient(dense=[("a", 1.0)], sparse=[("a", 1.0)])
    handle = _handle(client, monkeypatch, None)

    def fail(self: MilvusCollectionHandle, user_id: Any, is_admin: Any) -> list[str]:
        raise DatabaseOperationError("ledger unavailable")

    monkeypatch.setattr(MilvusCollectionHandle, "_kb_ids", fail)

    for search in (
        lambda: handle.search_dense(MODEL, [1.0], top_k=1),
        lambda: handle.search_sparse(MODEL, "kiwi", top_k=1),
        lambda: handle.search_hybrid(MODEL, "kiwi", [1.0], top_k=1),
    ):
        with pytest.raises(DatabaseOperationError, match="ledger unavailable"):
            search()
    assert client.calls == []


@pytest.mark.parametrize(
    ("top_k", "windows"),
    [
        (10, [(0, 1_000), (1_000, 1_000), (2_000, 1_000)]),
        (1_000, [(0, 1_000), (1_000, 1_000), (2_000, 1_000)]),
        (2_000, [(0, 2_000), (2_000, 2_000)]),
        (20_000, [(0, 16_384)]),
    ],
)
def test_the_fallback_page_size_is_at_least_a_thousand_and_within_the_window(
    monkeypatch: pytest.MonkeyPatch, top_k: int, windows: list[tuple[int, int]]
) -> None:
    client = FakeClient(rows=[client_row(f"x{i}", "aXb") for i in range(2_500)])
    handle = _handle(client, monkeypatch, None)

    handle.search_sparse(MODEL, "a_b", top_k=top_k)

    queries = [kw for method, kw in client.calls if method == "query"]
    assert [(q["offset"], q["limit"]) for q in queries] == windows


def test_a_failing_fallback_logs_and_returns_nothing_like_lancedb(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client = FakeClient(rows=[client_row("w1", "a_b")], queries_before_failure=0)
    handle = _handle(client, monkeypatch, None)

    with caplog.at_level("ERROR"):
        response = handle.search_sparse(MODEL, "a_b", top_k=3)

    assert (response.status, response.results, response.warnings) == (
        "success",
        [],
        [],
    )
    assert [r.levelname for r in caplog.records] == ["ERROR"]
    assert "Substring fallback failed: query failed" in caplog.text


def test_a_fallback_that_fails_on_a_later_page_returns_nothing_not_a_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(collection_handle, "_MILVUS_FALLBACK_PAGE", 2)
    rows = [client_row("w1", "a_b"), client_row("x0", "aXb"), client_row("w2", "a_b")]
    client = FakeClient(rows=rows, queries_before_failure=1)
    handle = _handle(client, monkeypatch, None)

    response = handle.search_sparse(MODEL, "a_b", top_k=3)

    assert (response.status, response.results, response.warnings) == (
        "success",
        [],
        [],
    )
    assert [m for m, _ in client.calls].count("query") == 2


def test_keyword_hits_skip_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient(sparse=[("a", 1.0)], rows=[client_row()])
    handle = _handle(client, monkeypatch, None)

    response = handle.search_sparse(MODEL, "kiwi", top_k=3)

    assert [r.chunk_id for r in response.results] == ["a"]
    assert response.warnings == []
    assert [method for method, _ in client.calls] == ["search"]


def test_the_fallback_never_reads_past_the_query_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient(rows=[client_row(f"x{i}", "aXb") for i in range(20_000)])
    handle = _handle(client, monkeypatch, None)

    handle.search_sparse(MODEL, "a_b", top_k=5_000)

    windows = [(kw["offset"], kw["limit"]) for m, kw in client.calls if m == "query"]
    assert windows == [(0, 5_000), (5_000, 5_000), (10_000, 5_000), (15_000, 1_384)]
    assert all(offset + limit <= 16_384 for offset, limit in windows)


def test_a_query_vector_of_numpy_floats_is_sent_as_plain_floats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    client = FakeClient(dense=[("a", 1.0)])
    handle = _handle(client, monkeypatch, None)

    handle.search_dense(MODEL, list(np.array([1.0, 0.5], dtype=np.float32)), top_k=1)

    (sent,) = client.calls[0][1]["data"]
    assert [type(x) for x in sent] == [float, float]
    assert math.isclose(sent[1], 0.5)
