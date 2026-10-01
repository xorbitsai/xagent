"""Sparse search user scope against a real LanceDB embeddings table.

Storage isolation is provided by the autouse ``isolate_rag_storage`` fixture in
``tests/conftest.py``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pytest

from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    LanceDBCollectionHandle,
)
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

MODEL = "scope-model"
COLLECTION = "shared"
ALL_DOCS = {"doc-u1", "doc-u2", "doc-legacy"}


def _handle() -> LanceDBCollectionHandle:
    context = KBCollectionContext(
        collection=COLLECTION,
        user_scope=KBUserScope(user_id=None, is_admin=True),
        access_mode=KBAccessMode.READ,
        allow_create=False,
        hide_missing=True,
        metadata_store=get_metadata_store(),
        vector_index_store=get_vector_index_store(),
        ingestion_status_store=get_ingestion_status_store(),
        main_pointer_store=get_main_pointer_store(),
        backend=KBStorageBackend.LANCEDB,
        capabilities=KBBackendCapabilities.lancedb(),
        collection_info=None,
    )
    return LanceDBCollectionHandle(context)


def _row(doc_id: str) -> dict:
    return {
        "collection": COLLECTION,
        "doc_id": doc_id,
        "chunk_id": f"{doc_id}-c0",
        "parse_hash": "ph",
        "model": MODEL,
        "vector": [0.1, 0.2],
        "vector_dimension": 2,
        "text": f"{doc_id} alphazulu",
        "chunk_hash": f"ch-{doc_id}",
        "created_at": datetime.now(timezone.utc),
        "metadata": "{}",
    }


@pytest.fixture
def shared_collection() -> None:
    # One write per row keeps scan order; doc-u2 last makes a pre-scope top_k cutoff miss it.
    for doc_id, user_id in (("doc-u1", 1), ("doc-legacy", None), ("doc-u2", 2)):
        get_vector_index_store().upsert_embeddings(
            MODEL, [{**_row(doc_id), "user_id": user_id}]
        )


@pytest.mark.parametrize(
    ("user_id", "is_admin", "expected"),
    [
        (1, False, {"doc-u1"}),
        (2, False, {"doc-u2"}),
        (None, False, set()),
        (None, True, ALL_DOCS),
        (1, True, ALL_DOCS),
    ],
)
# "alphazulu" is an indexed token; "phazu" misses FTS and hits the substring scan.
@pytest.mark.parametrize("query", ["alphazulu", "phazu"])
def test_sparse_search_applies_user_scope_on_fts_and_fallback(
    shared_collection: None,
    query: str,
    user_id: int | None,
    is_admin: bool,
    expected: set[str],
) -> None:
    response = _handle().search_sparse(
        MODEL, query, top_k=10, user_id=user_id, is_admin=is_admin
    )

    assert response.status == "success"
    assert {r.doc_id for r in response.results} == expected
    fell_back = any(w.code == "FTS_FALLBACK" for w in response.warnings)
    assert fell_back is (query == "phazu" and bool(expected))


def test_fallback_scope_applies_before_top_k(shared_collection: None) -> None:
    response = _handle().search_sparse(
        MODEL, "phazu", top_k=1, user_id=2, is_admin=False
    )

    assert {r.doc_id for r in response.results} == {"doc-u2"}


def test_fallback_caller_filters_cannot_change_collection(
    shared_collection: None,
) -> None:
    get_vector_index_store().upsert_embeddings(
        MODEL, [{**_row("doc-other"), "collection": "other", "user_id": 1}]
    )

    response = _handle().search_sparse(
        MODEL,
        "phazu",
        top_k=10,
        filters={"collection": "other"},
        user_id=1,
        is_admin=False,
    )

    assert response.results == []


def test_fallback_returns_more_than_ten_in_scope_rows() -> None:
    mine = {f"doc-u1-{i:02d}" for i in range(15)}
    get_vector_index_store().upsert_embeddings(
        MODEL,
        [{**_row(f"doc-u2-{i}"), "user_id": 2} for i in range(5)]
        + [{**_row(doc_id), "user_id": 1} for doc_id in sorted(mine)],
    )

    response = _handle().search_sparse(
        MODEL, "phazu", top_k=50, user_id=1, is_admin=False
    )

    assert {r.doc_id for r in response.results} == mine


def test_admin_fallback_reads_table_without_user_id_column() -> None:
    get_vector_index_store().get_raw_connection().create_table(
        "embeddings_legacy", data=[_row("doc-old")]
    )

    response = _handle().search_sparse(
        "legacy", "phazu", top_k=10, user_id=None, is_admin=True
    )

    assert [r.doc_id for r in response.results] == ["doc-old"]


def test_non_admin_fallback_skips_table_without_user_id_column(
    caplog: pytest.LogCaptureFixture,
) -> None:
    get_vector_index_store().get_raw_connection().create_table(
        "embeddings_legacy", data=[_row("doc-old")]
    )

    with caplog.at_level(logging.ERROR):
        results = _handle()._substring_fallback(
            table_name="embeddings_legacy",
            collection=COLLECTION,
            query_text="phazu",
            model_tag="legacy",
            top_k=10,
            filters=None,
            current_warnings=[],
            user_id=1,
            is_admin=False,
        )

    assert results == []
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
