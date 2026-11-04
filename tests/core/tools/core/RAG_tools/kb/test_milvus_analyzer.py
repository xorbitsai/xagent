"""Milvus keyword search across scripts, on the server at ``MILVUS_URI``.

The analyzer is fixed when the collection is created, so these tests cover the
one ``ensure_milvus_collection`` installs and the one old collections keep.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData
from xagent.core.tools.core.RAG_tools.kb import collection_handle
from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    KBCollectionHandle,
    KBHandleProvider,
    ensure_milvus_collection,
    milvus_collection_name,
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

pytestmark = pytest.mark.milvus

PARSE = "ph-1"
COLLECTION = "kb"
DIM = 3
ONE = [1.0, 0.0, 0.0]


@pytest.fixture
def client() -> Any:
    from pymilvus import MilvusClient

    return MilvusClient(uri=os.environ["MILVUS_URI"])


@pytest.fixture
def model(client: Any) -> Iterator[str]:
    model = f"analyzer-{uuid.uuid4().hex[:12]}"
    yield model
    client.drop_collection(milvus_collection_name(model))


@pytest.fixture(autouse=True)
def milvus_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAGENT_VECTOR_BACKEND", "milvus")


def _open() -> KBCollectionHandle:
    return KBHandleProvider().open(
        KBCollectionContext(
            collection=COLLECTION,
            user_scope=KBUserScope(user_id=None, is_admin=True),
            access_mode=KBAccessMode.WRITE,
            allow_create=True,
            hide_missing=True,
            metadata_store=get_metadata_store(),
            vector_index_store=get_vector_index_store(),
            ingestion_status_store=get_ingestion_status_store(),
            main_pointer_store=get_main_pointer_store(),
            backend=KBStorageBackend.MILVUS,
            capabilities=KBBackendCapabilities.milvus(),
        )
    )


def _ingest(
    handle: KBCollectionHandle, model: str, doc_id: str, chunks: list[tuple[str, str]]
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
                "metadata": {},
            }
            for index, (chunk_id, text) in enumerate(chunks)
        ],
        user_id=1,
    )
    pending = handle.read_chunks_needing_embedding(
        doc_id, PARSE, model, user_id=1, is_admin=False
    ).chunks
    handle.write_embeddings(
        [
            ChunkEmbeddingData(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                parse_hash=chunk.parse_hash,
                model=model,
                vector=ONE,
                text=chunk.text,
                chunk_hash=chunk.chunk_hash,
                metadata={},
            )
            for chunk in pending
        ],
        user_id=1,
    )
    handle.commit_embeddings(doc_id, PARSE, model, user_id=1, is_admin=False)


def _sparse(handle: KBCollectionHandle, model: str, query: str) -> Any:
    return handle.search_sparse(model, query, top_k=5, user_id=1, is_admin=False)


def _found(response: Any) -> set[str]:
    assert response.status == "success", response.warnings
    return {result.chunk_id for result in response.results}


@pytest.mark.parametrize(
    ("text", "queries"),
    [
        ("Milvus 2.6 向量数据库", ("milvus", "MILVUS", "Milvus")),
        ("知识库检索与向量数据库", ("向量数据库", "知识库")),
        ("ＭｉｌｖｕｓＤＢ ２０２６", ("ＭｉｌｖｕｓＤＢ",)),
        ("こんにちは世界", ("こんにちは",)),
        ("안녕하세요 세계", ("안녕하세요",)),
        ("Milvus2.6 release", ("Milvus2.6", "milvus2.6")),
        ("v2.6.25 发布", ("v2.6",)),
        ("version 2.6 notes", ("2.6",)),
    ],
    ids=[
        "ascii-case",
        "chinese-words",
        "full-width",
        "kana",
        "hangul",
        "mixed-alpha",
        "mixed-version",
        "dotted-version",
    ],
)
def test_keyword_search_matches_expected_tokens(
    text: str, queries: tuple[str, ...], model: str
) -> None:
    handle = _open()
    _ingest(handle, model, "d", [("c0", text)])

    for query in queries:
        response = _sparse(handle, model, query)
        assert _found(response) == {"c0"}, query
        assert response.warnings == [], query


def test_han_symbols_jieba_emits_are_kept(model: str) -> None:
    handle = _open()
    _ingest(handle, model, "d", [("c0", "日期⽇期")])

    response = _sparse(handle, model, "⽇")

    assert _found(response) == {"c0"}
    assert response.warnings == []


def test_full_width_text_is_matched_one_character_at_a_time(model: str) -> None:
    """Document character-level matching of full-width text, not a feature to extend."""
    handle = _open()
    _ingest(handle, model, "d", [("c0", "ＡＢＣ"), ("c1", "ＣＢＡ")])

    response = _sparse(handle, model, "ＡＢＣ")

    assert _found(response) == {"c0", "c1"}
    assert response.warnings == []
    assert response.results[0].score == pytest.approx(response.results[1].score)


@pytest.mark.parametrize(
    ("text", "query"),
    [
        ("Привет мир", "мир"),
        ("Müller Café", "ü"),
    ],
    ids=["cyrillic", "accented-latin"],
)
def test_other_scripts_reach_the_substring_fallback(
    text: str, query: str, model: str
) -> None:
    handle = _open()
    _ingest(handle, model, "d", [("c0", text)])

    response = _sparse(handle, model, query)

    assert _found(response) == {"c0"}
    assert [warning.code for warning in response.warnings] == ["FTS_FALLBACK"]


@pytest.mark.parametrize(
    ("text", "query"),
    [
        ("alpha beta", "zzz qqq"),
        ("hello ， world!! 🎉 结束", "!!"),
        ("全角逗号，与感叹号！！", "，！！"),
        ("hello ， world!! 🎉 结束", "🎉"),
    ],
    ids=["whitespace-terms", "ascii-punctuation", "cjk-punctuation", "emoji"],
)
def test_queries_without_indexed_tokens_match_no_bm25_row(
    text: str, query: str, client: Any, model: str
) -> None:
    handle = _open()
    _ingest(handle, model, "d", [("c0", text)])
    name = milvus_collection_name(model)

    hits = client.search(
        name,
        data=[query],
        anns_field="sparse",
        limit=5,
        search_params={"metric_type": "BM25"},
        output_fields=["chunk_id"],
        consistency_level="Strong",
    )[0]

    assert hits == [], query


def test_an_existing_collection_of_the_old_analyzer_is_kept(
    client: Any, model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Patch the analyzer constant for one create, so the old schema is real."""
    with monkeypatch.context() as patched:
        patched.setattr(
            collection_handle, "_MILVUS_ANALYZER_PARAMS", {"type": "chinese"}
        )
        name = ensure_milvus_collection(client, model, DIM)
    assert ensure_milvus_collection(client, model, DIM) == name
    fields = {
        field["name"]: field for field in client.describe_collection(name)["fields"]
    }
    assert json.loads(fields["text"]["params"]["analyzer_params"]) == {
        "type": "chinese"
    }

    handle = _open()
    _ingest(
        handle,
        model,
        "d",
        [("c0", "Milvus 2.6 向量数据库"), ("f0", "ＭｉｌｖｕｓＤＢ ２０２６")],
    )
    lower = _sparse(handle, model, "milvus")
    upper = _sparse(handle, model, "Milvus")
    full = _sparse(handle, model, "ＭｉｌｖｕｓＤＢ")
    mixed = _sparse(handle, model, "Milvus ＭｉｌｖｕｓＤＢ")

    assert (lower.results, lower.warnings) == ([], [])
    assert (_found(upper), upper.warnings) == ({"c0"}, [])
    assert (_found(full), [w.code for w in full.warnings]) == (
        {"f0"},
        ["FTS_FALLBACK"],
    )
    assert (_found(mixed), mixed.warnings) == ({"c0"}, [])
