"""Milvus search, on the server at ``MILVUS_URI``.

Rows come from the real write and commit path; dense, keyword and hybrid search
must return what LanceDB returns for the same data, inside the caller's scope.
"""

from __future__ import annotations

import math
import os
import random
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import DocumentValidationError
from xagent.core.tools.core.RAG_tools.core.schemas import (
    ChunkEmbeddingData,
    FusionConfig,
    FusionStrategy,
    IndexStatus,
    SearchFallbackAction,
)
from xagent.core.tools.core.RAG_tools.kb import collection_handle
from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    KBCollectionHandle,
    KBHandleProvider,
    MilvusCollectionHandle,
    milvus_collection_name,
)
from xagent.core.tools.core.RAG_tools.kb.kb_ids import get_or_create_kb_id
from xagent.core.tools.core.RAG_tools.kb.models import (
    KBAccessMode,
    KBBackendCapabilities,
    KBCollectionContext,
    KBStorageBackend,
    KBUserScope,
)
from xagent.core.tools.core.RAG_tools.storage.factory import (
    get_ingestion_status_store,
    get_main_pointer_store,
    get_metadata_store,
    get_vector_index_store,
)
from xagent.providers.vector_store.milvus import MilvusConnectionManager

pytestmark = pytest.mark.milvus

PARSE = "ph-1"
COLLECTION = "kb"
MODES = ("dense", "sparse", "hybrid")
ONE = [1.0, 0.0, 0.0]


def unit(degrees: float) -> list[float]:
    radians = math.radians(degrees)
    return [math.cos(radians), math.sin(radians), 0.0]


class Spy:
    """Forwards every call to a Milvus client and records the searches."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        target = getattr(self.client, name)

        def call(*args: Any, **kwargs: Any) -> Any:
            if name in ("search", "query"):
                self.calls.append((name, kwargs))
            return target(*args, **kwargs)

        return call


@pytest.fixture
def client() -> Any:
    from pymilvus import MilvusClient

    return MilvusClient(uri=os.environ["MILVUS_URI"])


@pytest.fixture
def model(client: Any) -> Iterator[str]:
    model = f"search-{uuid.uuid4().hex[:12]}"
    yield model
    client.drop_collection(milvus_collection_name(model))


@pytest.fixture(autouse=True)
def milvus_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAGENT_VECTOR_BACKEND", "milvus")


@pytest.fixture
def spy(client: Any, monkeypatch: pytest.MonkeyPatch) -> Spy:
    spy = Spy(client)
    monkeypatch.setattr(
        MilvusConnectionManager, "get_shared_client_from_env", lambda _self: spy
    )
    return spy


def _open(
    collection: str = COLLECTION, backend: KBStorageBackend = KBStorageBackend.MILVUS
) -> KBCollectionHandle:
    return KBHandleProvider().open(
        KBCollectionContext(
            collection=collection,
            user_scope=KBUserScope(user_id=None, is_admin=True),
            access_mode=KBAccessMode.WRITE,
            allow_create=True,
            hide_missing=True,
            metadata_store=get_metadata_store(),
            vector_index_store=get_vector_index_store(),
            ingestion_status_store=get_ingestion_status_store(),
            main_pointer_store=get_main_pointer_store(),
            backend=backend,
            capabilities=(
                KBBackendCapabilities.milvus()
                if backend is KBStorageBackend.MILVUS
                else KBBackendCapabilities.lancedb()
            ),
        )
    )


Chunk = tuple[str, str, list[float], dict[str, Any]]


def _chunk(
    chunk_id: str,
    text: str,
    vector: list[float] = ONE,
    metadata: dict[str, Any] | None = None,
) -> Chunk:
    return chunk_id, text, vector, metadata or {}


def _ingest(
    handle: KBCollectionHandle,
    model: str,
    doc_id: str,
    chunks: list[Chunk],
    *,
    user_id: int | None,
    commit: bool = True,
) -> None:
    now = datetime.now(timezone.utc)
    handle.write_chunks(
        doc_id,
        PARSE,
        "cfg",
        {},
        [
            {
                "chunk_id": chunk_id,
                "index": index,
                "text": text,
                "created_at": now,
                "metadata": metadata,
            }
            for index, (chunk_id, text, _, metadata) in enumerate(chunks)
        ],
        user_id=user_id,
    )
    vectors = {chunk_id: vector for chunk_id, _, vector, _ in chunks}
    admin = user_id is None
    pending = handle.read_chunks_needing_embedding(
        doc_id, PARSE, model, user_id=user_id, is_admin=admin
    ).chunks
    handle.write_embeddings(
        [
            ChunkEmbeddingData(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                parse_hash=chunk.parse_hash,
                model=model,
                vector=vectors[chunk.chunk_id],
                text=chunk.text,
                chunk_hash=chunk.chunk_hash,
                metadata=chunk.metadata,
            )
            for chunk in pending
        ],
        user_id=user_id,
    )
    if commit:
        handle.commit_embeddings(doc_id, PARSE, model, user_id=user_id, is_admin=admin)
    else:
        # A Strong read, so the invisible rows are served before the test searches.
        rest = handle.read_chunks_needing_embedding(
            doc_id, PARSE, model, user_id=user_id, is_admin=admin
        )
        assert rest.pending_count == 0


def _search(
    handle: KBCollectionHandle,
    model: str,
    mode: str,
    *,
    query: str = "kiwi",
    vector: list[float] = ONE,
    top_k: int = 10,
    **scope: Any,
) -> Any:
    scope = {"user_id": None, "is_admin": True, **scope}
    if mode == "dense":
        return handle.search_dense(model, vector, top_k=top_k, **scope)
    if mode == "sparse":
        return handle.search_sparse(model, query, top_k=top_k, **scope)
    return handle.search_hybrid(model, query, vector, top_k=top_k, **scope)


def _docs(response: Any) -> set[str]:
    assert response.status == "success", response.warnings
    return {result.doc_id for result in response.results}


def test_dense_scores_and_order_match_lancedb_for_the_same_vectors(model: str) -> None:
    chunks = [
        _chunk(f"c{angle}", f"kiwi {angle}", unit(angle))
        for angle in (0, 20, 45, 80, 120, 170)
    ]
    milvus, lance = _open("kb-m"), _open("kb-l", KBStorageBackend.LANCEDB)
    _ingest(milvus, model, "d", chunks, user_id=1)
    _ingest(lance, model, "d", chunks, user_id=1)

    wanted = lance.search_dense(model, unit(30), top_k=6, user_id=1, is_admin=False)
    got = milvus.search_dense(model, unit(30), top_k=6, user_id=1, is_admin=False)

    assert [r.chunk_id for r in got.results] == [r.chunk_id for r in wanted.results]
    assert [r.chunk_id for r in got.results] == [
        "c20",
        "c45",
        "c0",
        "c80",
        "c120",
        "c170",
    ]
    assert [r.score for r in got.results] == pytest.approx(
        [r.score for r in wanted.results], abs=1e-5
    )
    assert all(0.0 < r.score <= 1.0 for r in got.results)
    assert (got.status, got.total_count, got.index_status) == (
        "success",
        6,
        IndexStatus.INDEX_READY,
    )
    top, lance_top = got.results[0], wanted.results[0]
    assert (top.doc_id, top.text, top.parse_hash, top.model_tag) == (
        lance_top.doc_id,
        lance_top.text,
        lance_top.parse_hash,
        model,
    )
    assert top.created_at.tzinfo is None
    assert abs(top.created_at - lance_top.created_at).total_seconds() < 60


def test_keyword_scores_are_the_squashed_bm25_score_in_lancedb_order(
    client: Any, model: str
) -> None:
    chunks = [
        _chunk("k1", "kiwi kiwi kiwi apple"),
        _chunk("k2", "kiwi banana split cherry"),
        _chunk("k3", "plum cherry"),
    ]
    milvus, lance = _open("kb-m"), _open("kb-l", KBStorageBackend.LANCEDB)
    _ingest(milvus, model, "d", chunks, user_id=1)
    _ingest(lance, model, "d", chunks, user_id=1)

    got = milvus.search_sparse(model, "kiwi", top_k=5, user_id=1, is_admin=False)
    wanted = lance.search_sparse(model, "kiwi", top_k=5, user_id=1, is_admin=False)

    assert [r.chunk_id for r in got.results] == ["k1", "k2"]
    assert [r.chunk_id for r in wanted.results] == ["k1", "k2"]
    raw = client.search(
        milvus_collection_name(model),
        data=["kiwi"],
        anns_field="sparse",
        limit=5,
        search_params={"metric_type": "BM25"},
        filter="visible == true",
        output_fields=["chunk_id"],
        consistency_level="Strong",
    )[0]
    assert [hit["chunk_id"] for hit in raw] == ["k1", "k2"]
    assert [r.score for r in got.results] == pytest.approx(
        [hit["distance"] / (1 + hit["distance"]) for hit in raw]
    )
    assert all(0.0 < r.score < 1.0 for r in got.results)
    assert (got.status, got.fts_enabled, got.warnings) == ("success", True, [])
    assert (wanted.status, wanted.fts_enabled) == ("success", True)


@pytest.mark.parametrize("strategy", [FusionStrategy.RRF, FusionStrategy.LINEAR])
def test_hybrid_results_are_the_fusion_of_the_two_routes_with_route_scores(
    model: str, strategy: FusionStrategy
) -> None:
    handle = _open()
    _ingest(
        handle,
        model,
        "d",
        [
            _chunk("k1", "kiwi kiwi kiwi apple", unit(60)),
            _chunk("k2", "kiwi banana split cherry", unit(10)),
            _chunk("k3", "plum cherry", unit(0)),
            _chunk("k4", "kiwi grape", unit(90)),
        ],
        user_id=1,
    )
    config = FusionConfig(strategy=strategy)
    scope = {"user_id": 1, "is_admin": False}

    dense = handle.search_dense(model, unit(5), top_k=4, **scope)
    sparse = handle.search_sparse(model, "kiwi", top_k=4, **scope)
    hybrid = handle.search_hybrid(
        model, "kiwi", unit(5), top_k=2, fusion_config=config, **scope
    )

    expected = collection_handle._fuse_hybrid(
        model, "kiwi", dense, sparse, top_k=2, fusion_config=config
    )
    assert hybrid.results == expected.results
    assert (hybrid.dense_count, hybrid.sparse_count) == (4, 3)
    assert (hybrid.status, hybrid.warnings) == ("success", [])
    assert hybrid.fusion_config == config
    dense_by_id = {
        r.chunk_id: (rank, r.score) for rank, r in enumerate(dense.results, 1)
    }
    sparse_by_id = {
        r.chunk_id: (rank, r.score) for rank, r in enumerate(sparse.results, 1)
    }
    for result in hybrid.results:
        assert (result.vector_rank, result.vector_score) == dense_by_id[result.chunk_id]
        assert (result.fts_rank, result.fts_score) == sparse_by_id[result.chunk_id]
    if strategy is FusionStrategy.RRF:
        by_id = {r.chunk_id: r for r in hybrid.results}
        both = [c for c in dense_by_id if c in sparse_by_id]
        assert both and set(by_id) <= set(dense_by_id)
        for chunk_id, result in by_id.items():
            rank_sum = 1 / (60 + dense_by_id[chunk_id][0])
            if chunk_id in sparse_by_id:
                rank_sum += 1 / (60 + sparse_by_id[chunk_id][0])
            assert result.score == pytest.approx(rank_sum)


@pytest.fixture
def scoped(model: str) -> dict[str, KBCollectionHandle]:
    """Owners 1, 2 and no owner write ``kb``; owner 1 also writes ``other``."""
    kb, other = _open(COLLECTION), _open("other")
    _ingest(kb, model, "d1", [_chunk("d1-c0", "kiwi one")], user_id=1)
    _ingest(kb, model, "d2", [_chunk("d2-c0", "kiwi two")], user_id=2)
    _ingest(kb, model, "d0", [_chunk("d0-c0", "kiwi legacy")], user_id=None)
    _ingest(other, model, "dx", [_chunk("dx-c0", "kiwi other")], user_id=1)
    _ingest(
        kb,
        model,
        "pending",
        [_chunk("pending-c0", "kiwi pending")],
        user_id=1,
        commit=False,
    )
    return {COLLECTION: kb, "other": other}


def test_every_path_returns_only_committed_rows_of_the_callers_kb_ids(
    scoped: dict[str, KBCollectionHandle], model: str
) -> None:
    kb, other = scoped[COLLECTION], scoped["other"]
    everyone = {"d0", "d1", "d2"}
    cases = [
        (kb, {"user_id": 1, "is_admin": False}, {"d1"}),
        (kb, {"user_id": 2, "is_admin": False}, {"d2"}),
        (kb, {"user_id": 3, "is_admin": False}, set()),
        (kb, {"user_id": None, "is_admin": False}, set()),
        (kb, {"user_id": None, "is_admin": True}, everyone),
        (kb, {"user_id": 1, "is_admin": True}, everyone),
        (other, {"user_id": 1, "is_admin": False}, {"dx"}),
        (other, {"user_id": 2, "is_admin": False}, set()),
        (other, {"user_id": None, "is_admin": True}, {"dx"}),
    ]

    for mode in MODES:
        for handle, scope, expected in cases:
            found = _docs(_search(handle, model, mode, **scope))
            assert found == expected, (mode, scope)


def test_a_row_is_searchable_once_its_ingest_commits_and_not_before(
    scoped: dict[str, KBCollectionHandle], model: str
) -> None:
    kb = scoped[COLLECTION]
    mine = {"user_id": 1, "is_admin": False}
    before = {m: _docs(_search(kb, model, m, **mine)) for m in MODES}

    kb.commit_embeddings("pending", PARSE, model, user_id=1)

    after = {m: _docs(_search(kb, model, m, **mine)) for m in MODES}
    assert before == dict.fromkeys(MODES, {"d1"})
    assert after == dict.fromkeys(MODES, {"d1", "pending"})


def test_rows_of_a_kb_id_the_ledger_does_not_list_never_match(
    client: Any, scoped: dict[str, KBCollectionHandle], model: str
) -> None:
    name = milvus_collection_name(model)
    client.upsert(
        name,
        [
            {
                "chunk_id": "foreign-c0",
                "kb_id": uuid.uuid4().hex,
                "user_id": 1,
                "doc_id": "foreign",
                "parse_hash": PARSE,
                "config_hash": "",
                "text": "kiwi foreign",
                "dense": ONE,
                "visible": True,
                "created_at": 1,
                "metadata": {},
            }
        ],
    )
    client.query(name, filter='chunk_id != ""', consistency_level="Strong", limit=1)

    for mode in MODES:
        for scope in ({"user_id": 1, "is_admin": False}, {"is_admin": True}):
            docs = _docs(_search(scoped[COLLECTION], model, mode, **scope))
            assert docs and "foreign" not in docs, (mode, scope)


def test_an_admin_query_by_name_uses_the_kb_ids_of_every_owner(
    spy: Spy, scoped: dict[str, KBCollectionHandle], model: str
) -> None:
    conn = get_vector_index_store().get_raw_connection()
    kb_ids = {
        owner: get_or_create_kb_id(conn, COLLECTION, owner) for owner in (1, 2, None)
    }

    for mode in MODES:
        _docs(_search(scoped[COLLECTION], model, mode))

    searches = [kw for method, kw in spy.calls if method == "search"]
    assert len(searches) == 4
    for call in searches:
        assert call["filter"].endswith("and visible == true")
        for kb_id in kb_ids.values():
            assert f'"{kb_id}"' in call["filter"]
        assert get_or_create_kb_id(conn, "other", 1) not in call["filter"]


def test_a_team_collection_is_read_through_the_storage_owners_kb_id(
    model: str,
) -> None:
    team = _open("team")
    _ingest(team, model, "owner-doc", [_chunk("o-c0", "kiwi owner")], user_id=7)
    _ingest(team, model, "tenant-doc", [_chunk("t-c0", "kiwi tenant")], user_id=9)

    for mode in MODES:
        assert _docs(_search(team, model, mode, user_id=7, is_admin=False)) == {
            "owner-doc"
        }
        assert _docs(_search(team, model, mode, user_id=9, is_admin=False)) == {
            "tenant-doc"
        }


@pytest.fixture
def filtered(model: str) -> KBCollectionHandle:
    handle = _open()
    _ingest(
        handle,
        model,
        "d1",
        [
            _chunk("a", "kiwi alpha", metadata={"page": 1, "tag": "red"}),
            _chunk("b", "kiwi beta", metadata={"page": 2, "tag": "green"}),
            _chunk("c", "kiwi gamma", metadata={"page": 3, "meta": {"deep": "x"}}),
        ],
        user_id=1,
    )
    _ingest(handle, model, "d2", [_chunk("e", "kiwi delta")], user_id=1)
    _ingest(
        handle, model, "hidden", [_chunk("h", "kiwi hidden")], user_id=1, commit=False
    )
    return handle


FILTER_CASES = [
    ({"doc_id": "d1"}, {"a", "b", "c"}),
    ({"doc_id": {"operator": "ne", "value": "d1"}}, {"e"}),
    ({"doc_id": {"operator": "in", "value": ["d2", "zzz"]}}, {"e"}),
    ({"chunk_id": "b"}, {"b"}),
    ({"text": {"operator": "contains", "value": "alp"}}, {"a"}),
    ({"parse_hash": PARSE}, {"a", "b", "c", "e"}),
    ({"metadata.page": 2}, {"b"}),
    ({"metadata.page": {"operator": "gte", "value": 2}}, {"b", "c"}),
    ({"metadata.tag": {"operator": "in", "value": ["red", "green"]}}, {"a", "b"}),
    ({"metadata.meta.deep": "x"}, {"c"}),
    ({"doc_id": "d1", "metadata.page": {"operator": "lt", "value": 3}}, {"a", "b"}),
    ({"doc_id": "hidden"}, set()),
    ({"doc_id": {"operator": "ne", "value": "nothing"}}, {"a", "b", "c", "e"}),
]


def test_caller_filters_narrow_every_path_and_never_widen_the_scope(
    filtered: KBCollectionHandle, model: str
) -> None:
    for mode in MODES:
        for filters, expected in FILTER_CASES:
            response = _search(
                filtered, model, mode, filters=filters, user_id=1, is_admin=False
            )

            assert response.status == "success", (mode, filters, response.warnings)
            assert {r.chunk_id for r in response.results} == expected, (mode, filters)


@pytest.mark.parametrize("mode", MODES)
def test_an_untranslatable_filter_raises_instead_of_being_ignored(
    model: str, mode: str
) -> None:
    for filters in (
        {"created_at": 1},
        {"kb_id": "x"},
        {"text": {"operator": "contains", "value": "50%"}},
    ):
        with pytest.raises(DocumentValidationError):
            _search(_open(), model, mode, filters=filters)


PUNCTUATION_CORPUS = [
    "50% off today",
    "100%",
    "%start",
    "end%",
    "a_b",
    "aXb",
    "a__b",
    "C:\\Users\\me",
    "C:/Users",
    'say "hi"',
    "it's",
    "x\\_y",
    "x\\%y",
    "double\\\\slash",
    "%_%",
    "a%%b",
    "sale%off",
    "100%\\done",
    "snake_\\case",
    "'%'",
    '"_"',
    "plain words here",
    "全角逗号，与感叹号！！",
]
PUNCTUATION_TERMS = [
    "%",
    "%%",
    "_",
    "__",
    "\\",
    "\\\\",
    "\\%",
    "\\_",
    "_%",
    "%_",
    "%_%",
    "%\\",
    "_\\",
    "'",
    '"',
    "'%",
    '"_',
    "，",
    "！！",
]


@pytest.fixture
def punctuation(model: str) -> KBCollectionHandle:
    handle = _open()
    _ingest(
        handle,
        model,
        "d",
        [_chunk(f"p{i}", text) for i, text in enumerate(PUNCTUATION_CORPUS)],
        user_id=1,
    )
    return handle


def test_terms_the_analyzer_drops_fall_back_to_a_literal_substring_match(
    punctuation: KBCollectionHandle, model: str
) -> None:
    for term in PUNCTUATION_TERMS:
        response = punctuation.search_sparse(model, term, top_k=50, user_id=1)

        wanted = {f"p{i}" for i, text in enumerate(PUNCTUATION_CORPUS) if term in text}
        assert wanted, term
        assert {r.chunk_id for r in response.results} == wanted, term
        assert {r.score for r in response.results} == {1.0}
        assert [(w.code, w.fallback_action) for w in response.warnings] == [
            ("FTS_FALLBACK", SearchFallbackAction.BRUTE_FORCE)
        ], term
        assert response.status == "success"


def test_a_dropped_term_that_no_text_contains_returns_nothing_without_a_warning(
    punctuation: KBCollectionHandle, model: str
) -> None:
    response = punctuation.search_sparse(model, "^^", top_k=5, user_id=1)

    assert (response.status, response.results, response.warnings) == (
        "success",
        [],
        [],
    )


def test_the_fallback_stays_inside_the_callers_scope(
    punctuation: KBCollectionHandle, model: str
) -> None:
    _ingest(punctuation, model, "other", [_chunk("o0", "50% OFF")], user_id=2)

    mine = punctuation.search_sparse(model, "%", top_k=50, user_id=1)
    theirs = punctuation.search_sparse(model, "%", top_k=50, user_id=2)

    assert {r.chunk_id for r in theirs.results} == {"o0"}
    assert "o0" not in {r.chunk_id for r in mine.results}


SUBSTRING_TERMS = [
    "50%",
    "%off",
    "100%",
    "a_b",
    "C:\\Users",
    "C:\\",
    "x\\_y",
    "x\\%y",
    "\\\\slash",
    'say "hi"',
    "it's",
    "off today",
    "words",
]


def test_the_like_pattern_keeps_exactly_the_texts_that_contain_the_term(
    punctuation: KBCollectionHandle, model: str
) -> None:
    assert isinstance(punctuation, MilvusCollectionHandle)
    scope = punctuation._search_scope(None, 1, False)
    assert scope is not None

    for term in SUBSTRING_TERMS:
        found = punctuation._substring_results(model, term, 50, *scope)

        wanted = {f"p{i}" for i, text in enumerate(PUNCTUATION_CORPUS) if term in text}
        assert wanted, term
        assert {r.chunk_id for r in found} == wanted, term
        assert all(term in r.text for r in found)


def test_the_like_fallback_is_case_sensitive_like_lancedb(
    punctuation: KBCollectionHandle, model: str
) -> None:
    assert isinstance(punctuation, MilvusCollectionHandle)
    scope = punctuation._search_scope(None, 1, False)
    assert scope is not None

    for term in ("OFF", "Words", "C:\\users"):
        assert punctuation._substring_results(model, term, 50, *scope) == [], term


def test_the_like_pattern_never_misses_a_match_on_random_special_character_texts(
    client: Any, model: str
) -> None:
    handle = _open()
    assert isinstance(handle, MilvusCollectionHandle)
    _ingest(handle, model, "seed", [_chunk("seed", "seed")], user_id=1)
    name = milvus_collection_name(model)
    kb_id = get_or_create_kb_id(get_vector_index_store().get_raw_connection(), "kb", 1)
    rng = random.Random(7)
    alphabet = ["a", "b", "%", "_", "\\", '"', "'", "中", " ", "5", "\\_", "\\%", "%%"]
    texts = sorted(
        {"".join(rng.choices(alphabet, k=rng.randint(1, 9))) for _ in range(250)}
    )
    rows = [
        {
            "chunk_id": f"r{i}",
            "kb_id": kb_id,
            "user_id": 1,
            "doc_id": "rand",
            "parse_hash": PARSE,
            "config_hash": "",
            "text": text,
            "dense": ONE,
            "visible": True,
            "created_at": 1,
            "metadata": {},
        }
        for i, text in enumerate(texts)
    ]
    for start in range(0, len(rows), 100):
        client.upsert(name, rows[start : start + 100])
    client.query(name, filter='chunk_id != ""', consistency_level="Strong", limit=1)
    terms = sorted(
        {"".join(rng.choices(alphabet, k=rng.randint(1, 4))) for _ in range(60)}
    )
    scope = handle._search_scope(None, 1, False)
    assert scope is not None

    for term in terms:
        found = handle._substring_results(model, term, len(texts) + 1, *scope)
        wanted = {f"r{i}" for i, text in enumerate(texts) if term in text}
        assert {r.chunk_id for r in found} == wanted, term


def test_the_fallback_pages_past_rows_the_pattern_matches_but_the_term_does_not(
    model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(collection_handle, "_MILVUS_FALLBACK_PAGE", 3)
    handle = _open()
    assert isinstance(handle, MilvusCollectionHandle)
    _ingest(
        handle,
        model,
        "d",
        [_chunk(f"a{i:02d}", "aXb") for i in range(40)]
        + [_chunk(f"z{i}", "a_b") for i in range(3)],
        user_id=1,
    )
    scope = handle._search_scope(None, 1, False)
    assert scope is not None

    found = handle._substring_results(model, "a_b", 3, *scope)

    assert {r.chunk_id for r in found} == {"z0", "z1", "z2"}


def test_a_collection_that_is_not_loaded_reads_as_empty_on_every_path(
    client: Any, model: str
) -> None:
    handle = _open()
    _ingest(handle, model, "d", [_chunk("c0", "kiwi one")], user_id=1)
    name = milvus_collection_name(model)
    client.release_collection(name)

    for mode in MODES:
        response = _search(handle, model, mode, user_id=1, is_admin=False)
        assert (response.status, response.results, response.warnings) == (
            "success",
            [],
            [],
        )
    punct = handle.search_sparse(model, "%", top_k=5, user_id=1, is_admin=False)
    assert (punct.status, punct.results, punct.warnings) == ("success", [], [])

    client.load_collection(name)
    for mode in MODES:
        assert _docs(_search(handle, model, mode, user_id=1, is_admin=False)) == {"d"}


def test_a_model_without_a_collection_fails_each_route_with_a_warning(
    model: str,
) -> None:
    handle = _open()
    _ingest(handle, "another-" + model, "d", [_chunk("c0", "kiwi")], user_id=1)

    dense, sparse, hybrid = (
        _search(handle, model, mode, user_id=1, is_admin=False) for mode in MODES
    )

    assert (dense.status, dense.results) == ("failed", [])
    assert [w.code for w in dense.warnings] == ["DENSE_SEARCH_FAILED"]
    assert (sparse.status, sparse.results) == ("failed", [])
    assert [w.code for w in sparse.warnings] == ["FTS_SEARCH_FAILED"]
    assert (hybrid.status, hybrid.results) == ("partial_success", [])
    assert [w.code for w in hybrid.warnings] == [
        "DENSE_SEARCH_FAILED",
        "FTS_SEARCH_FAILED",
    ]


def test_the_models_of_one_kb_are_searched_apart(client: Any, model: str) -> None:
    other = "second-" + model
    try:
        handle = _open()
        _ingest(handle, model, "d1", [_chunk("c1", "kiwi one")], user_id=1)
        _ingest(handle, other, "d2", [_chunk("c2", "kiwi two")], user_id=1)

        for mode in MODES:
            assert _docs(_search(handle, model, mode, user_id=1, is_admin=False)) == {
                "d1"
            }
            assert _docs(_search(handle, other, mode, user_id=1, is_admin=False)) == {
                "d2"
            }
    finally:
        client.drop_collection(milvus_collection_name(other))
