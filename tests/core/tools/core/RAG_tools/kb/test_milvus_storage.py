"""Milvus collection layout and visible-row counts, on the server at ``MILVUS_URI``."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import VectorValidationError
from xagent.core.tools.core.RAG_tools.core.schemas import RegisterDocumentRequest
from xagent.core.tools.core.RAG_tools.kb import collection_handle
from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    KBCollectionHandle,
    KBHandleProvider,
    ensure_milvus_collection,
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
from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
    embeddings_table_name,
    to_model_tag,
)
from xagent.core.tools.core.RAG_tools.storage.factory import (
    get_ingestion_status_store,
    get_main_pointer_store,
    get_metadata_store,
    get_vector_index_store,
)

pytestmark = pytest.mark.milvus

DIM = 3


@pytest.fixture
def client() -> Any:
    from pymilvus import MilvusClient

    return MilvusClient(uri=os.environ["MILVUS_URI"])


@pytest.fixture
def models(client: Any) -> Iterator[tuple[str, str]]:
    suffix = uuid.uuid4().hex[:12]
    pair = (f"model-a-{suffix}", f"BAAI/Bge-M3-{suffix}")
    yield pair
    for model in pair:
        client.drop_collection(milvus_collection_name(model))


@pytest.fixture
def milvus_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAGENT_VECTOR_BACKEND", "milvus")


def _open(collection: str) -> KBCollectionHandle:
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
            backend=KBStorageBackend.MILVUS,
            capabilities=KBBackendCapabilities.milvus(),
        )
    )


def _row(
    chunk_id: str, kb_id: str, doc_id: str, user_id: int, *, visible: bool
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "kb_id": kb_id,
        "user_id": user_id,
        "doc_id": doc_id,
        "parse_hash": "ph",
        "config_hash": "cfg",
        "text": f"kiwi {chunk_id}",
        "dense": [1.0, 0.0, 0.0],
        "visible": visible,
        "created_at": 1,
        "metadata": {},
    }


def _insert(client: Any, name: str, rows: list[dict[str, Any]]) -> None:
    client.insert(name, rows)
    # A Strong read waits until the rows are served, so later default reads see them.
    client.query(
        name,
        filter='chunk_id != ""',
        output_fields=["count(*)"],
        consistency_level="Strong",
    )


def _ledger_document(
    handle: KBCollectionHandle, doc_id: str, chunks: int, user_id: int, tmp_path: Path
) -> None:
    source = tmp_path / f"{handle.context.collection}-{doc_id}.txt"
    source.write_text(doc_id, encoding="utf-8")
    handle.register_document(
        RegisterDocumentRequest(
            collection=handle.context.collection,
            source_path=str(source),
            doc_id=doc_id,
            user_id=user_id,
        )
    )
    handle.write_chunks(
        doc_id,
        "ph",
        "cfg",
        {},
        [
            {
                "chunk_id": f"{doc_id}-{index}",
                "index": index,
                "text": "kiwi",
                "created_at": datetime.now(timezone.utc),
            }
            for index in range(chunks)
        ],
        user_id=user_id,
    )


def test_the_collection_refuses_rows_without_kb_id_text_or_vector(
    client: Any, models: tuple[str, str]
) -> None:
    name = ensure_milvus_collection(client, models[0], DIM)
    assert ensure_milvus_collection(client, models[0], DIM) == name
    fields = {
        field["name"]: field for field in client.describe_collection(name)["fields"]
    }

    assert [n for n in ("kb_id", "text", "dense") if fields[n].get("nullable")] == []
    assert fields["kb_id"].get("is_partition_key") is True
    assert json.loads(fields["text"]["params"]["analyzer_params"]) == {
        "tokenizer": "jieba",
        "filter": [
            "lowercase",
            {
                "type": "regex",
                "expr": (
                    r"[0-9A-Za-z\p{Han}\p{Hiragana}\p{Katakana}\p{Hangul}"
                    r"\x{FF10}-\x{FF19}\x{FF21}-\x{FF3A}\x{FF41}-\x{FF5A}]"
                ),
            },
        ],
    }
    (bm25,) = client.describe_collection(name)["functions"]
    assert (bm25["input_field_names"], bm25["output_field_names"]) == (
        ["text"],
        ["sparse"],
    )

    from pymilvus import MilvusException

    _insert(client, name, [_row("c0", "kb", "doc", 1, visible=False)])
    flip = [{"chunk_id": "c0", "visible": True}, {"chunk_id": "ghost", "visible": True}]
    with pytest.raises(MilvusException, match="kb_id"):
        client.upsert(name, flip, partial_update=True)
    rows = client.query(
        name,
        filter='chunk_id != ""',
        output_fields=["visible"],
        consistency_level="Strong",
    )
    assert rows == [{"chunk_id": "c0", "visible": False}]


def test_an_existing_collection_of_another_dimension_is_refused(
    client: Any, models: tuple[str, str]
) -> None:
    name = ensure_milvus_collection(client, models[0], DIM)
    client.release_collection(name)

    with pytest.raises(VectorValidationError, match=f"{name}.*{DIM}.*{DIM + 1}"):
        ensure_milvus_collection(client, models[0], DIM + 1)
    assert client.get_load_state(name)["state"].name == "NotLoad"


def test_a_collection_left_unindexed_is_indexed_and_loaded_again(
    client: Any, models: tuple[str, str]
) -> None:
    name = ensure_milvus_collection(client, models[0], DIM)
    client.release_collection(name)
    for field in client.list_indexes(name):
        client.drop_index(name, field)

    assert ensure_milvus_collection(client, models[0], DIM) == name
    assert client.get_load_state(name)["state"].name == "Loaded"
    assert sorted(client.list_indexes(name)) == ["dense", "doc_id", "sparse", "user_id"]


def test_model_ids_that_tag_alike_get_collections_of_their_own(client: Any) -> None:
    suffix = uuid.uuid4().hex[:12]
    trio = (f"BAAI/bge-m3-{suffix}", f"baai-bge-m3-{suffix}", f"baai_bge_m3_{suffix}")
    names = [milvus_collection_name(model) for model in trio]
    try:
        for model in trio:
            ensure_milvus_collection(client, model, DIM)

        assert len(set(names)) == 3
        recorded = [
            client.describe_collection(name)["properties"]["xagent.model_id"]
            for name in names
        ]
        assert recorded == list(trio)
    finally:
        for name in names:
            client.drop_collection(name)


@pytest.mark.parametrize(
    "model", ["智谱/GLM 4.5-embedding", "vendor/" + "long-model-name-" * 30]
)
def test_any_model_id_names_a_collection_milvus_accepts(
    client: Any, model: str
) -> None:
    name = ensure_milvus_collection(client, model, DIM)
    try:
        assert client.get_load_state(name)["state"].name == "Loaded"
    finally:
        client.drop_collection(name)


def test_a_collection_recorded_for_another_model_is_refused(
    client: Any, models: tuple[str, str]
) -> None:
    name = ensure_milvus_collection(client, models[0], DIM)
    client.release_collection(name)

    client.alter_collection_properties(name, {"xagent.model_id": "someone/else"})
    with pytest.raises(VectorValidationError, match=f"{name}.*someone/else"):
        ensure_milvus_collection(client, models[0], DIM)
    client.drop_collection_properties(name, ["xagent.model_id"])
    with pytest.raises(VectorValidationError, match=f"{name}.*model None"):
        ensure_milvus_collection(client, models[0], DIM)
    assert client.get_load_state(name)["state"].name == "NotLoad"


def test_concurrent_first_callers_agree_on_one_collection(
    client: Any, models: tuple[str, str]
) -> None:
    with ThreadPoolExecutor(6) as pool:
        names = set(
            pool.map(
                lambda _: ensure_milvus_collection(client, models[0], DIM), range(6)
            )
        )

    assert names == {milvus_collection_name(models[0])}
    assert client.get_load_state(names.pop())["state"].name == "Loaded"


def test_counts_read_only_visible_rows_of_the_callers_kb_ids(
    client: Any,
    models: tuple[str, str],
    milvus_deployment: None,
    tmp_path: Path,
) -> None:
    kb, other = _open("kb"), _open("other")
    _ledger_document(kb, "doc-1", 3, 1, tmp_path)
    _ledger_document(kb, "doc-2", 1, 2, tmp_path)
    _ledger_document(kb, "doc-3", 1, 1, tmp_path)
    _ledger_document(other, "doc-x", 1, 1, tmp_path)
    conn = get_vector_index_store().get_raw_connection()
    mine, theirs = (get_or_create_kb_id(conn, "kb", owner) for owner in (1, 2))
    elsewhere = get_or_create_kb_id(conn, "other", 1)
    model_a = ensure_milvus_collection(client, models[0], DIM)
    model_b = ensure_milvus_collection(client, models[1], DIM)
    _insert(
        client,
        model_a,
        [
            _row("doc-1-0", mine, "doc-1", 1, visible=True),
            _row("doc-1-1", mine, "doc-1", 1, visible=True),
            _row("doc-1-2", mine, "doc-1", 1, visible=False),
            _row("doc-2-0", theirs, "doc-2", 2, visible=True),
            _row("doc-3-0", mine, "doc-3", 1, visible=True),
            _row("doc-x-0", elsewhere, "doc-x", 1, visible=True),
        ],
    )
    _insert(client, model_b, [_row("doc-1-b", mine, "doc-1", 1, visible=True)])
    a_key, b_key = (embeddings_table_name(to_model_tag(model)) for model in models)

    assert kb.collection_stats(None, True) == {
        "documents": 3,
        "chunks": 5,
        "embeddings": 5,
    }
    assert kb.collection_stats(1, False)["embeddings"] == 4
    assert kb.collection_stats(3, False)["embeddings"] == 0
    assert kb.count_rows_by_document(user_id=None, is_admin=True) == {
        "doc-1": {"chunks": 3, a_key: 2, b_key: 1},
        "doc-2": {"chunks": 1, a_key: 1},
        "doc-3": {"chunks": 1, a_key: 1},
    }
    assert kb.count_rows_by_document(user_id=2, is_admin=False) == {
        "doc-2": {"chunks": 1, a_key: 1}
    }
    assert kb.count_rows_by_document(user_id=1, is_admin=False, doc_id="doc-1") == {
        "doc-1": {"chunks": 3, a_key: 2, b_key: 1}
    }
    assert kb.count_rows_by_document(user_id=2, is_admin=False, doc_id="doc-1") == {}

    provider = KBHandleProvider()
    admin = provider.aggregate_collection_stats(user_id=None, is_admin=True)
    assert {name: row["embeddings"] for name, row in admin.items()} == {
        "kb": 5,
        "other": 1,
    }
    tenant = provider.aggregate_collection_stats(user_id=2, is_admin=False)
    assert {name: row["embeddings"] for name, row in tenant.items()} == {"kb": 1}


def test_a_collection_that_is_not_loaded_counts_as_empty(
    client: Any,
    models: tuple[str, str],
    milvus_deployment: None,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    kb = _open("kb")
    _ledger_document(kb, "doc-1", 2, 1, tmp_path)
    mine = get_or_create_kb_id(get_vector_index_store().get_raw_connection(), "kb", 1)
    loaded = ensure_milvus_collection(client, models[0], DIM)
    released = ensure_milvus_collection(client, models[1], DIM)
    _insert(client, loaded, [_row("doc-1-0", mine, "doc-1", 1, visible=True)])
    _insert(client, released, [_row("doc-1-1", mine, "doc-1", 1, visible=True)])
    client.release_collection(released)
    key = embeddings_table_name(to_model_tag(models[0]))
    expected = {"doc-1": {"chunks": 2, key: 1}}

    assert kb.collection_stats(1, False)["embeddings"] == 1
    assert kb.count_rows_by_document(user_id=1, is_admin=False) == expected
    assert kb.count_rows_by_document(user_id=1, is_admin=False, doc_id="doc-1") == (
        expected
    )
    stats = KBHandleProvider().aggregate_collection_stats(user_id=1, is_admin=False)
    assert stats["kb"]["embeddings"] == 1
    assert released in caplog.text and "not loaded" in caplog.text


def test_collections_without_the_prefix_are_not_counted(
    client: Any,
    models: tuple[str, str],
    milvus_deployment: None,
    tmp_path: Path,
) -> None:
    kb = _open("kb")
    _ledger_document(kb, "doc-1", 1, 1, tmp_path)
    mine = get_or_create_kb_id(get_vector_index_store().get_raw_connection(), "kb", 1)
    ours = ensure_milvus_collection(client, models[0], DIM)
    stray = f"stray_{uuid.uuid4().hex[:12]}"
    client.create_collection(stray, dimension=DIM)
    try:
        _insert(client, ours, [_row("doc-1-0", mine, "doc-1", 1, visible=True)])
        client.insert(
            stray,
            [{"id": 1, "vector": [1.0, 0.0, 0.0], "kb_id": mine, "visible": True}],
        )
        client.query(
            stray, filter="id >= 0", output_fields=["id"], consistency_level="Strong"
        )

        assert kb.collection_stats(1, False)["embeddings"] == 1
        stats = KBHandleProvider().aggregate_collection_stats(user_id=1, is_admin=False)
        assert stats["kb"]["embeddings"] == 1
    finally:
        client.drop_collection(stray)


def test_counts_page_through_every_batch_of_the_query_iterator(
    client: Any,
    models: tuple[str, str],
    milvus_deployment: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(collection_handle, "_MILVUS_QUERY_BATCH", 2)
    kb = _open("kb")
    _ledger_document(kb, "doc-1", 1, 1, tmp_path)
    mine = get_or_create_kb_id(get_vector_index_store().get_raw_connection(), "kb", 1)
    name = ensure_milvus_collection(client, models[0], DIM)
    _insert(
        client,
        name,
        [
            *(_row(f"doc-1-{i}", mine, "doc-1", 1, visible=True) for i in range(5)),
            _row("doc-1-hidden", mine, "doc-1", 1, visible=False),
            _row("doc-2-0", mine, "doc-2", 1, visible=True),
            _row("doc-3-0", mine, "doc-3", 1, visible=True),
            _row("doc-x-0", "elsewhere", "doc-x", 1, visible=True),
        ],
    )
    key = embeddings_table_name(to_model_tag(models[0]))

    assert kb.count_rows_by_document(user_id=1, is_admin=False) == {
        "doc-1": {"chunks": 1, key: 5},
        "doc-2": {key: 1},
        "doc-3": {key: 1},
    }
