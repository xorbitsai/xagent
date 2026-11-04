"""Milvus embedding writes and the commit step, on the server at ``MILVUS_URI``.

Rows are written invisible and become visible only in ``commit_embeddings``;
the diff, the flip and the check read with Strong consistency.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import EmbeddingModelConfig
from xagent.core.tools.core.RAG_tools.core.exceptions import (
    DatabaseOperationError,
    DocumentValidationError,
)
from xagent.core.tools.core.RAG_tools.core.schemas import (
    ChunkEmbeddingData,
    DocumentProcessingStatus,
    IngestionConfig,
)
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
from xagent.core.tools.core.RAG_tools.management import collection_manager
from xagent.core.tools.core.RAG_tools.pipelines import document_ingestion
from xagent.core.tools.core.RAG_tools.storage.contracts import VectorIndexStore
from xagent.core.tools.core.RAG_tools.storage.factory import (
    get_ingestion_status_store,
    get_main_pointer_store,
    get_metadata_store,
    get_vector_index_store,
)
from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
    LanceDBVectorIndexStore,
)
from xagent.providers.vector_store.milvus import MilvusConnectionManager

pytestmark = pytest.mark.milvus

COLLECTION = "kb"
PARSE = "ph-1"
NEXT_PARSE = "ph-2"


class Spy:
    """Forwards every call to a Milvus client and records it."""

    failures = 0
    swallow = False

    def __init__(self, client: Any) -> None:
        self.client, self.calls = client, []

    def __getattr__(self, name: str) -> Any:
        target = getattr(self.client, name)

        def call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args, kwargs))
            return target(*args, **kwargs)

        return call

    def upsert(self, name: str, rows: list[dict[str, Any]], **kwargs: Any) -> Any:
        self.calls.append(("upsert", (name, rows), kwargs))
        if kwargs.get("partial_update") and self.failures:
            self.failures -= 1
            raise RuntimeError("partial update failed")
        if kwargs.get("partial_update") and self.swallow:
            return {}
        return self.client.upsert(name, rows, **kwargs)

    def partial_updates(self) -> list[list[dict[str, Any]]]:
        return [
            args[1]
            for name, args, kwargs in self.calls
            if name == "upsert" and kwargs.get("partial_update")
        ]


@pytest.fixture
def client() -> Any:
    from pymilvus import MilvusClient

    return MilvusClient(uri=os.environ["MILVUS_URI"])


@pytest.fixture
def model(client: Any) -> Iterator[str]:
    model = f"commit-{uuid.uuid4().hex[:12]}"
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
    user_id: int = 1,
    store: VectorIndexStore | None = None,
    backend: KBStorageBackend = KBStorageBackend.MILVUS,
) -> KBCollectionHandle:
    return KBHandleProvider().open(
        KBCollectionContext(
            collection=COLLECTION,
            user_scope=KBUserScope(user_id=user_id, is_admin=False),
            access_mode=KBAccessMode.WRITE,
            allow_create=True,
            hide_missing=True,
            metadata_store=get_metadata_store(),
            vector_index_store=store or get_vector_index_store(),
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


def _kb_id(user_id: int = 1) -> str:
    conn = get_vector_index_store().get_raw_connection()
    return get_or_create_kb_id(conn, COLLECTION, user_id)


def _chunks(
    handle: KBCollectionHandle,
    ids: list[str],
    *,
    doc_id: str = "doc",
    parse_hash: str = PARSE,
    text: str | None = None,
) -> None:
    now = datetime.now(timezone.utc)
    handle.write_chunks(
        doc_id,
        parse_hash,
        "cfg",
        {},
        [
            {
                "chunk_id": chunk_id,
                "index": index,
                "text": text or f"kiwi {chunk_id}",
                "created_at": now,
            }
            for index, chunk_id in enumerate(ids)
        ],
        user_id=1,
    )


def _embed(
    handle: KBCollectionHandle,
    model: str,
    *,
    doc_id: str = "doc",
    parse_hash: str = PARSE,
    limit: int | None = None,
) -> list[str]:
    """Write what the diff reports, as the pipeline does; return those chunk ids."""
    pending = handle.read_chunks_needing_embedding(
        doc_id, parse_hash, model, user_id=1
    ).chunks[:limit]
    handle.write_embeddings(
        [
            ChunkEmbeddingData(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                parse_hash=chunk.parse_hash,
                model=model,
                vector=[1.0, 0.0, 0.0],
                text=chunk.text,
                chunk_hash=chunk.chunk_hash,
                metadata=chunk.metadata,
            )
            for chunk in pending
        ],
        user_id=1,
    )
    return [chunk.chunk_id for chunk in pending]


def _commit(
    handle: KBCollectionHandle,
    model: str,
    *,
    doc_id: str = "doc",
    parse_hash: str = PARSE,
    commit_gate: Any = None,
) -> None:
    handle.commit_embeddings(
        doc_id, parse_hash, model, commit_gate=commit_gate, user_id=1
    )


def _rows(client: Any, model: str) -> dict[str, dict[str, Any]]:
    rows = client.query(
        milvus_collection_name(model),
        filter='chunk_id != ""',
        output_fields=["kb_id", "user_id", "doc_id", "visible", "created_at"],
        consistency_level="Strong",
    )
    return {row.pop("chunk_id"): row for row in rows}


def _visible(client: Any, model: str) -> set[str]:
    return {id_ for id_, row in _rows(client, model).items() if row["visible"]}


def _foreign_row(client: Any, model: str, chunk_id: str, **fields: Any) -> None:
    row = {
        "chunk_id": chunk_id,
        "kb_id": "other-kb",
        "user_id": 2,
        "doc_id": "doc",
        "parse_hash": "ph",
        "config_hash": "",
        "text": "kiwi",
        "dense": [1.0, 0.0, 0.0],
        "visible": True,
        "created_at": 1,
        "metadata": {},
    }
    client.upsert(milvus_collection_name(model), [{**row, **fields}])


def test_written_rows_stay_invisible_until_the_commit(client: Any, model: str) -> None:
    handle = _open()
    _chunks(handle, ["a", "b", "c"])

    assert set(_embed(handle, model)) == {"a", "b", "c"}

    rows = _rows(client, model)
    assert {id_: row["visible"] for id_, row in rows.items()} == dict.fromkeys(
        "abc", False
    )
    assert all(row["created_at"] > 0 for row in rows.values())
    assert {(row["kb_id"], row["user_id"], row["doc_id"]) for row in rows.values()} == {
        (_kb_id(), 1, "doc")
    }
    assert handle.collection_stats(1, False)["embeddings"] == 0

    _commit(handle, model)

    assert _visible(client, model) == {"a", "b", "c"}
    assert handle.collection_stats(1, False)["embeddings"] == 3


def test_an_interrupted_import_shows_nothing_and_resume_embeds_only_the_rest(
    client: Any, model: str
) -> None:
    handle = _open()
    _chunks(handle, ["a", "b", "c", "d"])
    first = _embed(handle, model, limit=2)

    assert _visible(client, model) == set()
    assert handle.collection_stats(1, False)["embeddings"] == 0

    resumed = _embed(handle, model)
    assert len(resumed) == 2 and not set(resumed) & set(first)
    _commit(handle, model)

    assert _visible(client, model) == {"a", "b", "c", "d"}


def test_running_the_same_import_again_changes_nothing(client: Any, model: str) -> None:
    handle = _open()
    _chunks(handle, ["a", "b", "c"])
    _embed(handle, model)
    _commit(handle, model)
    before = _rows(client, model)

    assert _embed(handle, model) == []
    _commit(handle, model)

    assert _rows(client, model) == before


def test_a_row_under_a_stale_kb_id_is_written_again(client: Any, model: str) -> None:
    handle = _open()
    _chunks(handle, ["a", "b"])
    _embed(handle, model)
    _foreign_row(client, model, "b", kb_id="stale", user_id=1)

    assert _embed(handle, model) == ["b"]
    _commit(handle, model)

    rows = _rows(client, model)
    assert {row["kb_id"] for row in rows.values()} == {_kb_id()}
    assert _visible(client, model) == {"a", "b"}


def test_a_reimport_deletes_the_old_batch_and_keeps_reused_rows(
    client: Any, model: str
) -> None:
    handle = _open()
    _chunks(handle, ["a", "b", "c"])
    _embed(handle, model)
    _commit(handle, model)
    reused = _rows(client, model)["b"]
    _foreign_row(client, model, "x")
    _foreign_row(client, model, "y", kb_id=_kb_id(), user_id=1, doc_id="other-doc")
    _chunks(handle, ["b", "d"], parse_hash=NEXT_PARSE)

    assert _embed(handle, model, parse_hash=NEXT_PARSE) == ["d"]
    _commit(handle, model, parse_hash=NEXT_PARSE)

    rows = _rows(client, model)
    assert set(rows) == {"b", "d", "x", "y"}
    assert rows["b"] == reused
    assert all(row["visible"] for row in rows.values())


def test_a_superseded_run_commits_and_deletes_nothing(client: Any, model: str) -> None:
    handle = _open()
    _chunks(handle, ["a"])
    _embed(handle, model)
    _commit(handle, model)
    _chunks(handle, ["z"], parse_hash=NEXT_PARSE)
    _embed(handle, model, parse_hash=NEXT_PARSE)

    def superseded() -> None:
        raise RuntimeError("superseded")

    with pytest.raises(RuntimeError, match="superseded"):
        _commit(handle, model, parse_hash=NEXT_PARSE, commit_gate=superseded)

    assert {id_: row["visible"] for id_, row in _rows(client, model).items()} == {
        "a": True,
        "z": False,
    }


def test_a_commit_that_finds_a_missing_row_flips_nothing_and_keeps_the_old_batch(
    client: Any, model: str
) -> None:
    handle = _open()
    _chunks(handle, ["a"])
    _embed(handle, model)
    _commit(handle, model)
    _chunks(handle, ["x", "y", "z"], parse_hash=NEXT_PARSE)
    written = _embed(handle, model, parse_hash=NEXT_PARSE, limit=2)
    (missing,) = {"x", "y", "z"} - set(written)

    with pytest.raises(
        DatabaseOperationError,
        match=rf"1 of 3 chunks of doc have no row under this kb_id, so nothing was "
        rf"committed \(for example \['{missing}'\]\)",
    ):
        _commit(handle, model, parse_hash=NEXT_PARSE)

    assert _visible(client, model) == {"a"}

    _embed(handle, model, parse_hash=NEXT_PARSE)
    _commit(handle, model, parse_hash=NEXT_PARSE)
    assert _visible(client, model) == {"x", "y", "z"}


@pytest.mark.parametrize(("failures", "committed"), [(1, True), (2, False)])
def test_a_failed_flip_is_queried_and_retried_once(
    client: Any, model: str, spy: Spy, failures: int, committed: bool
) -> None:
    handle = _open()
    _chunks(handle, ["a", "b"])
    _embed(handle, model)
    spy.failures = failures

    if committed:
        _commit(handle, model)
    else:
        with pytest.raises(DatabaseOperationError, match="partial update failed"):
            _commit(handle, model)

    assert len(spy.partial_updates()) == 2
    assert _visible(client, model) == ({"a", "b"} if committed else set())


def test_a_flip_that_changes_nothing_fails_the_commit(
    client: Any, model: str, spy: Spy
) -> None:
    handle = _open()
    _chunks(handle, ["a", "b"])
    _embed(handle, model)
    spy.swallow = True

    with pytest.raises(
        DatabaseOperationError,
        match=r"2 of 2 chunks of doc are missing or invisible in Milvus after the "
        r"commit \(for example \['a', 'b'\]\)",
    ):
        _commit(handle, model)

    assert _visible(client, model) == set()


def test_a_run_superseded_after_the_flip_deletes_nothing(
    client: Any, model: str
) -> None:
    handle = _open()
    _chunks(handle, ["a"])
    _embed(handle, model)
    _commit(handle, model)
    _chunks(handle, ["z"], parse_hash=NEXT_PARSE)
    _embed(handle, model, parse_hash=NEXT_PARSE)
    calls = []

    def superseded_after_the_flip() -> None:
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("superseded")

    with pytest.raises(RuntimeError, match="superseded"):
        _commit(
            handle,
            model,
            parse_hash=NEXT_PARSE,
            commit_gate=superseded_after_the_flip,
        )

    assert _visible(client, model) == {"a", "z"}


@pytest.mark.parametrize(
    ("broken", "message"),
    [
        ("_get_table", "Read 0 of 3 ledger chunks"),
        ("build_filter_expression", "Cannot count the ledger chunks of doc"),
    ],
)
def test_a_ledger_read_that_fails_deletes_nothing_and_reports_no_pending_chunks(
    client: Any, model: str, monkeypatch: pytest.MonkeyPatch, broken: str, message: str
) -> None:
    handle = _open()
    _chunks(handle, ["a", "b", "c"])
    _embed(handle, model)
    _commit(handle, model)

    def unreadable(*_: Any, **__: Any) -> Any:
        raise OSError("table unreadable")

    monkeypatch.setattr(
        get_vector_index_store(),
        broken,
        unreadable if broken == "_get_table" else lambda *_, **__: "(((",
    )

    with pytest.raises(DatabaseOperationError, match=message):
        _commit(handle, model)
    with pytest.raises(DatabaseOperationError, match=message):
        handle.read_chunks_needing_embedding("doc", PARSE, model, user_id=1)

    monkeypatch.undo()
    assert _visible(client, model) == {"a", "b", "c"}


def test_a_commit_counts_the_ledger_on_a_fresh_table_not_a_stale_cached_one(
    client: Any, model: str
) -> None:
    handle = _open()
    _chunks(handle, ["a", "b"])
    _embed(handle, model)
    other = _open(store=LanceDBVectorIndexStore())
    _chunks(other, ["c"])
    other.write_embeddings(
        [
            ChunkEmbeddingData(
                doc_id="doc",
                chunk_id="c",
                parse_hash=PARSE,
                model=model,
                vector=[1.0, 0.0, 0.0],
                text="kiwi c",
                chunk_hash="h",
            )
        ],
        user_id=1,
    )

    _commit(handle, model)

    assert _visible(client, model) == {"a", "b", "c"}


def test_a_ledger_read_that_stops_early_deletes_nothing(
    client: Any, model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    handle = _open()
    _chunks(handle, ["a", "b", "c"])
    _embed(handle, model)
    _commit(handle, model)
    store = get_vector_index_store()
    read = store.iter_batches

    def first_row_only(**kwargs: Any) -> Iterator[Any]:
        for batch in read(**kwargs):
            yield batch.slice(0, 1)
            return

    monkeypatch.setattr(store, "iter_batches", first_row_only)

    with pytest.raises(DatabaseOperationError, match="Read 1 of 3 ledger chunks"):
        _commit(handle, model)
    with pytest.raises(DatabaseOperationError, match="Read 1 of 3 ledger chunks"):
        handle.read_chunks_needing_embedding("doc", PARSE, model, user_id=1)

    monkeypatch.undo()
    assert _visible(client, model) == {"a", "b", "c"}


def test_a_chunk_written_between_the_count_and_the_read_is_not_a_short_read(
    client: Any, model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    handle, writer = _open(), _open()
    _chunks(handle, ["a", "b"])
    store = get_vector_index_store()
    read = store.iter_batches
    written: list[int] = []

    def read_after_a_write(**kwargs: Any) -> Iterator[Any]:
        if not written:
            written.append(1)
            _chunks(writer, ["a", "b", "c"])
        return read(**kwargs)

    monkeypatch.setattr(store, "iter_batches", read_after_a_write)

    pending = handle.read_chunks_needing_embedding("doc", PARSE, model, user_id=1)

    assert written == [1]
    assert {chunk.chunk_id for chunk in pending.chunks} == {"a", "b", "c"}


@pytest.mark.parametrize(
    "broken", [("list_tables", "table_names"), ("open_table",)], ids=["list", "open"]
)
def test_a_ledger_table_that_cannot_be_listed_or_opened_is_not_a_missing_table(
    client: Any, model: str, monkeypatch: pytest.MonkeyPatch, broken: tuple[str, ...]
) -> None:
    handle = _open()
    _chunks(handle, ["a", "b", "c"])
    _embed(handle, model)
    _commit(handle, model)
    store = get_vector_index_store()
    connection = store.get_raw_connection()

    class Unreadable:
        def __getattr__(self, name: str) -> Any:
            if name not in broken:
                return getattr(connection, name)

            def fail(*_: Any, **__: Any) -> Any:
                raise OSError(f"{name} unreadable")

            return fail

    monkeypatch.setattr(store, "get_raw_connection", lambda: Unreadable())

    with pytest.raises(DatabaseOperationError, match="Cannot count the ledger chunks"):
        _commit(handle, model)
    with pytest.raises(DatabaseOperationError, match="Cannot count the ledger chunks"):
        handle.read_chunks_needing_embedding("doc", PARSE, model, user_id=1)

    monkeypatch.undo()
    assert _visible(client, model) == {"a", "b", "c"}


def test_a_document_is_flipped_in_one_call(client: Any, model: str, spy: Spy) -> None:
    handle = _open()
    ids = [f"c{index:04d}" for index in range(1200)]
    _chunks(handle, ids)
    _embed(handle, model)

    _commit(handle, model)

    (update,) = spy.partial_updates()
    assert sorted(row["chunk_id"] for row in update) == ids
    assert len(_visible(client, model)) == 1200


def test_every_milvus_call_is_strong_and_scoped_to_kb_id_and_doc_id(
    client: Any, model: str, spy: Spy
) -> None:
    handle = _open()
    _chunks(handle, ["a", "b"])
    _embed(handle, model)
    _commit(handle, model)
    _chunks(handle, ["c"], parse_hash=NEXT_PARSE)
    _embed(handle, model, parse_hash=NEXT_PARSE)
    _commit(handle, model, parse_hash=NEXT_PARSE)

    kb_id = _kb_id()
    seen = {name for name, _, _ in spy.calls}
    assert {"query", "upsert", "delete"} <= seen and "insert" not in seen
    for name, args, kwargs in spy.calls:
        if name == "query":
            assert kwargs["consistency_level"] == "Strong"
        if name in ("query", "delete"):
            assert "kb_id ==" in kwargs["filter"] and "doc_id ==" in kwargs["filter"]
            assert kwargs["filter_params"]["kb_id"] == kb_id
            assert kwargs["filter_params"]["doc_id"] == "doc"
        if name == "upsert" and not kwargs.get("partial_update"):
            assert {(row["kb_id"], row["doc_id"]) for row in args[1]} == {
                (kb_id, "doc")
            }


@pytest.mark.parametrize(
    ("text", "accepted"),
    [
        ("x" * 65_535, True),
        ("x" * 65_536, False),
        ("é" * 32_767 + "x", True),
        ("é" * 32_768, False),
    ],
    ids=["ascii-at-limit", "ascii-over", "bytes-at-limit", "bytes-over"],
)
def test_text_over_the_milvus_limit_fails_before_anything_is_embedded(
    client: Any, model: str, text: str, accepted: bool
) -> None:
    handle = _open()
    _chunks(handle, ["a"], text=text)

    if accepted:
        _embed(handle, model)
        _commit(handle, model)
        assert _visible(client, model) == {"a"}
    else:
        with pytest.raises(DocumentValidationError, match="lower the chunk size"):
            _embed(handle, model)


def test_reading_pending_chunks_logs_no_missing_lancedb_table_warning(
    model: str, caplog: pytest.LogCaptureFixture
) -> None:
    handle = _open()
    _chunks(handle, ["a"])

    assert _embed(handle, model) == ["a"]
    assert "Failed to query existing embeddings" not in caplog.text


def test_chunks_that_lancedb_already_embedded_stay_pending_until_milvus_holds_them(
    client: Any, model: str
) -> None:
    handle = _open()
    _chunks(handle, ["a", "b", "c"])
    held = _embed(handle, model, limit=1)
    _open(backend=KBStorageBackend.LANCEDB).write_embeddings(
        [
            ChunkEmbeddingData(
                doc_id="doc",
                chunk_id=chunk_id,
                parse_hash=PARSE,
                model=model,
                vector=[1.0, 0.0, 0.0],
                text=f"kiwi {chunk_id}",
                chunk_hash="h",
            )
            for chunk_id in "abc"
        ],
        user_id=1,
    )

    pending = handle.read_chunks_needing_embedding("doc", PARSE, model, user_id=1)

    assert {chunk.chunk_id for chunk in pending.chunks} == set("abc") - set(held)
    assert (pending.total_count, pending.pending_count) == (3, 2)

    _embed(handle, model)
    _commit(handle, model)
    assert _visible(client, model) == {"a", "b", "c"}


def test_a_created_but_unloaded_collection_is_read_as_empty_and_loaded_by_the_write(
    client: Any, model: str
) -> None:
    name = milvus_collection_name(model)
    ensure_milvus_collection(client, model, 3)
    client.release_collection(name)
    handle = _open()
    _chunks(handle, ["a", "b"])

    assert set(_embed(handle, model)) == {"a", "b"}

    assert client.get_load_state(name)["state"].name == "Loaded"
    _commit(handle, model)
    assert _visible(client, model) == {"a", "b"}


def test_the_commit_does_not_read_an_unloaded_collection_as_empty(
    client: Any, model: str
) -> None:
    handle = _open()
    _chunks(handle, ["a"])
    _embed(handle, model)
    client.release_collection(milvus_collection_name(model))

    with pytest.raises(Exception, match="not loaded") as raised:
        _commit(handle, model)

    assert getattr(raised.value, "code", None) == 101


def test_a_first_document_with_no_chunks_has_nothing_pending(model: str) -> None:
    pending = _open().read_chunks_needing_embedding("doc", PARSE, model, user_id=1)

    assert (pending.chunks, pending.total_count, pending.pending_count) == ([], 0, 0)


def test_the_commit_error_names_at_most_five_missing_chunks(
    client: Any, model: str
) -> None:
    handle = _open()
    ids = [f"c{index}" for index in range(8)]
    _chunks(handle, ids)
    written = _embed(handle, model, limit=2)
    missing = sorted(set(ids) - set(written))

    with pytest.raises(DatabaseOperationError, match="6 of 8 chunks of doc") as raised:
        _commit(handle, model)

    assert str(missing[:5]) in str(raised.value)
    assert missing[5] not in str(raised.value)


class _Embedder(BaseEmbedding):
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.fail_first = False

    def encode(  # type: ignore[override]
        self, text: Any, dimension: int | None = None, instruct: str | None = None
    ) -> Any:
        if self.fail_first:
            self.fail_first = False
            raise RuntimeError("provider down")
        batch = [text] if isinstance(text, str) else list(text)
        self.texts += batch
        vectors = [[1.0, 0.0, float(len(item))] for item in batch]
        return vectors[0] if isinstance(text, str) else vectors

    def get_dimension(self) -> int:
        return 3

    @property
    def abilities(self) -> list[str]:
        return ["embedding"]


class _Import:
    """Runs the real ingestion pipeline on one file with a stub embedder."""

    def __init__(self, source: Path, model: str, embedder: _Embedder) -> None:
        self.source, self.model, self.embedder = source, model, embedder

    def run(self, config: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        return document_ingestion.process_document(
            COLLECTION,
            str(self.source),
            config=IngestionConfig(
                embedding_model_id=self.model,
                chunk_size=200,
                chunk_overlap=0,
                embedding_batch_size=1,
                **(config or {}),
            ),
            user_id=1,
            is_admin=False,
            **kwargs,
        )


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model: str) -> _Import:
    config = EmbeddingModelConfig(
        id=model, model_name="stub", model_provider="stub", dimension=3
    )
    embedder = _Embedder()
    monkeypatch.setattr(
        document_ingestion, "_resolve_embedding_adapter", lambda _: (config, embedder)
    )
    monkeypatch.setattr(
        collection_manager,
        "resolve_embedding_adapter",
        lambda *_, **__: (config, embedder),
    )
    source = tmp_path / "doc.txt"
    source.write_text(
        "\n\n".join(f"paragraph {index} " * 30 for index in range(4)), encoding="utf-8"
    )
    return _Import(source, model, embedder)


def _status(doc_id: str) -> str:
    (row,) = _open().load_ingestion_status(doc_id=doc_id, user_id=1)
    return str(row["status"])


def test_an_import_becomes_visible_when_it_completes_and_rerunning_it_changes_nothing(
    client: Any, pipeline: _Import
) -> None:
    result = pipeline.run()

    assert result.status == "success" and result.chunk_count > 1
    rows = _rows(client, pipeline.model)
    assert len(rows) == result.chunk_count and _visible(client, pipeline.model) == set(
        rows
    )
    embedded = len(pipeline.embedder.texts)

    again = pipeline.run()

    assert again.status == "success" and again.embedding_count == 0
    assert len(pipeline.embedder.texts) == embedded
    assert _rows(client, pipeline.model) == rows


def test_an_import_that_stops_midway_is_invisible_and_resumes_with_the_rest(
    client: Any, pipeline: _Import, monkeypatch: pytest.MonkeyPatch
) -> None:
    write = document_ingestion.write_vectors_to_db
    writes: list[int] = []

    def stop_after_two(**kwargs: Any) -> Any:
        if len(writes) == 2:
            raise RuntimeError("stopped")
        writes.append(1)
        return write(**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(document_ingestion, "write_vectors_to_db", stop_after_two)
        failed = pipeline.run()

    assert failed.status == "partial"
    assert len(_rows(client, pipeline.model)) == 2
    assert _visible(client, pipeline.model) == set()
    assert _status(failed.doc_id) == DocumentProcessingStatus.FAILED.value
    first_run = len(pipeline.embedder.texts)

    result = pipeline.run()

    assert result.status == "success"
    assert len(pipeline.embedder.texts) - first_run == result.chunk_count - 2
    assert len(_visible(client, pipeline.model)) == result.chunk_count


def test_a_run_with_nothing_pending_still_commits(
    client: Any, pipeline: _Import, monkeypatch: pytest.MonkeyPatch
) -> None:
    def crash(**_: Any) -> None:
        raise RuntimeError("crashed before the commit")

    with monkeypatch.context() as patch:
        patch.setattr(document_ingestion, "commit_vectors_to_db", crash)
        failed = pipeline.run()
    assert failed.status == "partial"
    assert _visible(client, pipeline.model) == set()
    embedded = len(pipeline.embedder.texts)

    result = pipeline.run()

    assert result.status == "success" and result.embedding_count == 0
    assert len(pipeline.embedder.texts) == embedded
    assert len(_visible(client, pipeline.model)) == result.chunk_count


def test_a_superseded_import_commits_nothing(
    client: Any, pipeline: _Import, monkeypatch: pytest.MonkeyPatch
) -> None:
    written = []
    write = document_ingestion.write_vectors_to_db

    def record(**kwargs: Any) -> Any:
        written.append(1)
        return write(**kwargs)

    def gate() -> None:
        if written:
            raise RuntimeError("superseded")

    monkeypatch.setattr(document_ingestion, "write_vectors_to_db", record)

    result = pipeline.run(commit_gate=gate)

    assert result.status == "partial" and "superseded" in result.message
    assert _rows(client, pipeline.model) and not _visible(client, pipeline.model)


def test_a_row_lost_before_the_check_fails_the_import_and_a_rerun_refills_it(
    client: Any, pipeline: _Import, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = document_ingestion.commit_vectors_to_db

    def lose_a_row(**kwargs: Any) -> None:
        lost = next(iter(_rows(client, pipeline.model)))
        client.delete(
            milvus_collection_name(pipeline.model),
            filter="chunk_id == {id}",
            filter_params={"id": lost},
        )
        commit(**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(document_ingestion, "commit_vectors_to_db", lose_a_row)
        failed = pipeline.run()
    assert failed.status == "partial" and "have no row" in failed.message
    assert _status(failed.doc_id) == DocumentProcessingStatus.FAILED.value
    embedded = len(pipeline.embedder.texts)

    result = pipeline.run()

    assert result.status == "success"
    assert len(pipeline.embedder.texts) - embedded == 1
    assert len(_visible(client, pipeline.model)) == result.chunk_count


def test_a_chunk_that_fails_to_embed_fails_the_import_instead_of_leaving_a_gap(
    client: Any, pipeline: _Import
) -> None:
    pipeline.embedder.fail_first = True

    failed = pipeline.run({"embedding_use_async": True, "max_retries": 1})

    assert failed.status == "partial" and "1 of" in failed.message
    assert _rows(client, pipeline.model) and not _visible(client, pipeline.model)
    embedded = len(pipeline.embedder.texts)

    result = pipeline.run({"embedding_use_async": True, "max_retries": 1})

    assert result.status == "success"
    assert len(pipeline.embedder.texts) - embedded == 1
    assert len(_visible(client, pipeline.model)) == result.chunk_count


def test_the_batches_of_one_import_share_one_collection_and_kb_id_lookup(
    client: Any, pipeline: _Import, monkeypatch: pytest.MonkeyPatch
) -> None:
    ensured: list[int] = []
    kb_ids: list[int] = []
    writes: list[int] = []
    before_commit: list[int] = []
    ensure = collection_handle.ensure_milvus_collection
    get_kb_id = collection_handle.get_or_create_kb_id
    write = document_ingestion.write_vectors_to_db
    commit = document_ingestion.commit_vectors_to_db
    monkeypatch.setattr(
        collection_handle,
        "ensure_milvus_collection",
        lambda *args: ensured.append(1) or ensure(*args),
    )
    monkeypatch.setattr(
        collection_handle,
        "get_or_create_kb_id",
        lambda *args: kb_ids.append(1) or get_kb_id(*args),
    )
    monkeypatch.setattr(
        document_ingestion,
        "write_vectors_to_db",
        lambda **kwargs: writes.append(1) or write(**kwargs),
    )
    monkeypatch.setattr(
        document_ingestion,
        "commit_vectors_to_db",
        lambda **kwargs: before_commit.append(len(kb_ids)) or commit(**kwargs),
    )

    result = pipeline.run()

    assert result.status == "success" and len(writes) == result.chunk_count > 2
    assert len(ensured) == 1
    assert before_commit == [1] and len(kb_ids) == 2
    assert len(_visible(client, pipeline.model)) == result.chunk_count
