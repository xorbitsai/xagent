"""Embed: copy what the target lacks, and only for eligible collections."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pytest

from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import EmbeddingModelConfig
from xagent.migrations.embedding import embed as mod
from xagent.migrations.embedding import scope

TARGET = EmbeddingModelConfig(id="new-model", model_name="new-model", dimension=2)
SOURCE_MODEL = "BAAI/old-model"


def _config(batch_size: int = 2, async_mode: bool = False) -> Any:
    from xagent.core.tools.core.RAG_tools.core.schemas import IngestionConfig

    return IngestionConfig(
        embedding_model_id=TARGET.id,
        embedding_batch_size=batch_size,
        embedding_use_async=async_mode,
        embedding_concurrent=2,
    )


class _Adapter(BaseEmbedding):
    """Deterministic vectors; counts calls so a dry run can prove none happened."""

    def __init__(self, fail_on: Optional[List[str]] = None) -> None:
        self.calls: List[Any] = []
        self.fail_on = fail_on or []

    def encode(self, text: Any, dimension: Any = None, instruct: Any = None) -> Any:
        self.calls.append(text)
        texts = [text] if isinstance(text, str) else list(text)
        for item in texts:
            if item in self.fail_on:
                raise RuntimeError(f"provider refused {item!r}")
        vectors = [[0.25, 0.75] for _ in texts]
        return vectors[0] if isinstance(text, str) else vectors

    def get_dimension(self) -> Optional[int]:
        return 2

    @property
    def abilities(self) -> List[str]:
        return ["embedding"]


def _save_collection(
    name: str = "kb",
    model_id: Optional[str] = SOURCE_MODEL,
    dimension: Optional[int] = 2,
) -> None:
    from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
    from xagent.core.tools.core.RAG_tools.management import collection_manager as cm

    asyncio.run(
        cm.collection_manager.save_collection(
            CollectionInfo(
                name=name,
                embedding_model_id=model_id,
                embedding_dimension=dimension,
                embeddings=4,
            )
        )
    )


_SCHEMA = pa.schema(
    [
        ("collection", pa.string()),
        ("doc_id", pa.string()),
        ("chunk_id", pa.string()),
        ("parse_hash", pa.string()),
        ("user_id", pa.int64()),
        ("text", pa.string()),
        ("chunk_hash", pa.string()),
        ("metadata", pa.string()),
        ("model", pa.string()),
        ("created_at", pa.timestamp("us")),
        ("vector", pa.list_(pa.float32(), 2)),
    ]
)


def collection_model(collection: str) -> Optional[str]:
    return SOURCE_MODEL if collection == "kb" else collection


def _row(
    chunk_id: str,
    *,
    doc_id: Optional[str] = "d1",
    text: Optional[str] = None,
    user_id: Optional[int] = 7,
    collection: str = "kb",
) -> Dict[str, Any]:
    return {
        "collection": collection,
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "parse_hash": "p",
        "user_id": user_id,
        "text": text if text is not None else f"text {chunk_id}",
        "chunk_hash": f"h-{chunk_id}",
        "metadata": None,
        "model": collection_model(collection),
        "created_at": datetime(2026, 1, 1),
        "vector": [0.5, 0.5],
    }


def _table(name: str, rows: List[Dict[str, Any]], schema: Any = _SCHEMA) -> None:
    conn = scope.connection()
    if not rows:
        conn.create_table(name, schema=schema)
        return
    conn.create_table(name, data=pa.Table.from_pylist(rows, schema=schema))


def _seed_target(chunk_ids: List[str]) -> None:
    """Create the target table the way the write path does, with these chunks."""
    from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData
    from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
        write_vectors_to_db,
    )

    write_vectors_to_db(
        collection="kb",
        embeddings=[
            ChunkEmbeddingData(
                doc_id="d1",
                chunk_id=chunk_id,
                parse_hash="p",
                model=TARGET.id,
                vector=[0.25, 0.75],
                text=f"held {chunk_id}",
                chunk_hash=f"h-{chunk_id}",
                metadata=None,
            )
            for chunk_id in chunk_ids
        ],
        create_index=False,
        user_id=7,
    )


def _rows_in(name: str) -> List[Dict[str, Any]]:
    conn = scope.connection()
    if name not in scope.table_names(conn):
        return []
    return conn.open_table(name).search().where("true").to_arrow().to_pylist()


def _source(rows: List[Dict[str, Any]]) -> None:
    _table(scope.source_table(SOURCE_MODEL), rows)


@pytest.fixture(autouse=True)
def _no_legacy_spelling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scope, "_legacy_spelling", lambda model_id: None)


@pytest.fixture(autouse=True)
def _indexes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Index building is covered by its own tests; keep the copy tests focused."""

    class _Result:
        status = "success"
        advice = ""
        fts_enabled = True

    class _Store:
        def __init__(self) -> None:
            self.calls: List[str] = []

        def create_index(self, model_id: str) -> Any:
            self.calls.append(model_id)
            return _Result()

    store = _Store()
    monkeypatch.setattr(mod, "_index_store", lambda: store)
    return store


def _run(
    adapter: Optional[_Adapter] = None,
    *,
    dry_run: bool = False,
    batch_size: int = 2,
    async_mode: bool = False,
) -> mod.CollectionReport:
    return mod.embed_collection(
        "kb",
        model_id=TARGET.id,
        embedding_config=TARGET,
        embedding_adapter=adapter or _Adapter(),
        cfg=_config(batch_size=batch_size, async_mode=async_mode),
        dry_run=dry_run,
    )


# --- copy ---


def test_pending_chunks_are_copied_into_the_target_table() -> None:
    _save_collection()
    _source([_row("c1"), _row("c2", doc_id="d2")])
    adapter = _Adapter()

    report = _run(adapter)

    assert report.failures == []
    assert (report.documents_embedded, report.chunks_embedded) == (2, 2)
    copied = _rows_in(scope.source_table(TARGET.id))
    assert {row["chunk_id"] for row in copied} == {"c1", "c2"}
    assert {row["user_id"] for row in copied} == {7}
    assert len(adapter.calls) == 2


def test_a_chunk_the_target_holds_is_not_embedded_again() -> None:
    _save_collection()
    _source([_row("c1"), _row("c2", doc_id="d2")])
    _seed_target(["c1"])
    adapter = _Adapter()

    report = _run(adapter)

    assert (report.documents_embedded, report.documents_unchanged) == (1, 1)
    assert report.chunks_embedded == 1
    assert adapter.calls == [["text c2"]]


def test_a_rerun_writes_nothing() -> None:
    _save_collection()
    _source([_row("c1")])
    _seed_target(["c1"])
    adapter = _Adapter()

    report = _run(adapter)

    assert report.chunks_embedded == 0
    assert adapter.calls == []


def test_a_dry_run_calls_no_provider_and_writes_nothing() -> None:
    _save_collection()
    _source([_row("c1")])
    adapter = _Adapter()

    report = _run(adapter, dry_run=True)

    assert report.chunks_pending == 1
    assert adapter.calls == []
    assert _rows_in(scope.source_table(TARGET.id)) == []


def test_excluded_rows_are_counted_once_and_not_written() -> None:
    _save_collection()
    _source(
        [
            _row("c1"),
            _row("c2", doc_id=None),
            _row("c3", user_id=None),
            _row("c4", text=""),
            # A chunkless row beside a usable sibling of its document: it must
            # be counted at the document read, not again when documents are listed.
            _row("", doc_id="d9"),
            _row("c5", doc_id="d9"),
        ]
    )

    report = _run()

    assert report.skipped == {
        scope.SKIP_NO_DOC_ID: 2,
        scope.SKIP_NO_OWNER: 1,
        scope.SKIP_NO_TEXT: 1,
    }
    assert report.chunks_embedded == 2
    scoped = scope.scope_collection("kb")
    assert scoped.skipped == report.skipped


def test_a_document_is_settled_once_across_many_chunks() -> None:
    _save_collection()
    _source([_row(f"c{index:03d}", doc_id=f"d{index:03d}") for index in range(60)])
    report = _run(_Adapter(), batch_size=7)

    assert (report.documents_total, report.documents_embedded) == (60, 60)
    assert report.chunks_embedded == 60


def test_documents_split_between_held_and_pending_are_counted_once() -> None:
    _save_collection()
    rows = [_row("c1"), _row("c2")]
    _source(rows + [_row("c3", doc_id="d2")])
    _seed_target(["c1"])

    report = _run()

    assert (report.documents_total, report.documents_embedded) == (2, 2)
    assert report.chunks_embedded == 2


# --- eligibility ---


def test_an_unbound_collection_is_skipped_and_not_written() -> None:
    _save_collection(model_id=None, dimension=None)
    _source([_row("c1")])

    report = _run(_Adapter())

    assert report.collection_skip == scope.SKIP_UNBOUND
    assert _rows_in(scope.source_table(TARGET.id)) == []


def test_a_collection_with_rows_elsewhere_is_skipped() -> None:
    _save_collection()
    _table("embeddings_BAAI_old_model", [_row("c1")])

    report = _run(_Adapter())

    assert report.collection_skip == scope.SKIP_ROWS_IN_OTHER_TABLE
    assert _rows_in(scope.source_table(TARGET.id)) == []


def test_a_collection_with_a_missing_column_is_skipped() -> None:
    _save_collection()
    narrow = pa.schema(
        [(name, _SCHEMA.field(name).type) for name in ("collection", "doc_id")]
    )
    _table(scope.source_table(SOURCE_MODEL), [_row("c1")], schema=narrow)

    assert _run(_Adapter()).collection_skip == scope.SKIP_MISSING_COLUMN


# --- failures ---


def test_one_document_failing_does_not_stop_the_collection() -> None:
    _save_collection()
    _source([_row("c1"), _row("c2", doc_id="d2")])

    report = _run(_Adapter(fail_on=["text c1"]))

    assert [failure.doc_id for failure in report.failures] == ["d1"]
    assert report.chunks_embedded == 1


def test_consecutive_failures_abort_the_run() -> None:
    _save_collection()
    _source([_row(f"c{index}", doc_id=f"d{index}") for index in range(8)])

    report = _run(
        _Adapter(fail_on=[f"text c{index}" for index in range(8)]), batch_size=1
    )

    assert report.aborted
    assert len(report.failures) == mod.MAX_CONSECUTIVE_DOCUMENT_FAILURES


def test_a_chunk_that_does_not_land_is_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_collection()
    _source([_row("c1"), _row("c2", doc_id="d2")])

    def _drop_one(**kwargs: Any) -> None:
        return None

    monkeypatch.setattr(mod, "write_vectors_to_db", _drop_one)

    report = _run()

    assert [failure.doc_id for failure in report.failures] == ["d1", "d2"]
    assert "missing from the target model" in report.failures[0].error
    assert report.chunks_embedded == 0


def test_a_provider_vector_of_the_wrong_width_is_a_failure() -> None:
    _save_collection()
    _source([_row("c1")])

    class _Wide(_Adapter):
        def encode(self, text: Any, dimension: Any = None, instruct: Any = None) -> Any:
            self.calls.append(text)
            return [[0.5, 0.5, 0.5]]

    report = _run(_Wide())

    assert report.failures and "dimension" in report.failures[0].error


# --- indexes ---


def _real_index_store(monkeypatch: pytest.MonkeyPatch, result: Any) -> List[str]:
    calls: List[str] = []

    class _Store:
        def create_index(self, model_id: str) -> Any:
            calls.append(model_id)
            return result

    monkeypatch.setattr(mod, "_index_store", lambda: _Store())
    return calls


class _IndexResult:
    def __init__(self, status: str, fts: bool) -> None:
        self.status = status
        self.fts_enabled = fts
        self.advice = "advice"


def test_a_populated_target_without_fts_fails_even_when_nothing_was_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_collection()
    _source([_row("c1")])
    _seed_target(["c1"])
    calls = _real_index_store(monkeypatch, _IndexResult("success", fts=False))

    report = _run(_Adapter())

    assert calls == [TARGET.id]
    assert [failure.error for failure in report.failures] == [
        "target table has no full-text index after the copy"
    ]


def test_no_target_table_needs_no_index(monkeypatch: pytest.MonkeyPatch) -> None:
    _save_collection()
    _source([])
    calls = _real_index_store(monkeypatch, _IndexResult("failed", fts=False))

    report = _run(_Adapter())

    assert calls == []
    assert report.failures == []


def test_a_failed_index_build_fails_the_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_collection()
    _source([_row("c1")])
    _real_index_store(monkeypatch, _IndexResult("index_corrupted", fts=True))

    report = _run(_Adapter())

    assert "index_corrupted" in report.failures[0].error


# --- run level ---


def test_one_collection_crashing_does_not_stop_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "resolve_collection_names", lambda **kwargs: ["a", "b"])
    monkeypatch.setattr(mod, "connection", lambda: object())
    monkeypatch.setattr(
        mod,
        "resolve_target_model",
        lambda model_id: (TARGET, _Adapter()),
    )
    real = mod.embed_collection

    def _one_crashes(name: str, **kwargs: Any) -> Any:
        if name == "a":
            raise RuntimeError("boom")
        return real(name, **kwargs)

    monkeypatch.setattr(mod, "embed_collection", _one_crashes)

    report = mod.embed_collections(TARGET.id)

    assert [item.name for item in report.collections] == ["a", "b"]
    assert report.failed
    assert not report.aborted


def test_an_unresolvable_target_never_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "resolve_collection_names", lambda **kwargs: ["kb"])

    def _unknown(model_id: str) -> Any:
        raise scope.MigrationCliError("no such model")

    monkeypatch.setattr(mod, "resolve_target_model", _unknown)

    assert mod.main(["--model", "nope"]) == 2


def test_the_exit_code_is_zero_when_something_was_copied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_collection()
    _source([_row("c1")])
    monkeypatch.setattr(mod, "resolve_collection_names", lambda **kwargs: ["kb"])
    monkeypatch.setattr(
        mod, "resolve_target_model", lambda model_id: (TARGET, _Adapter())
    )

    assert mod.main(["--model", TARGET.id]) == 0


def test_every_collection_skipped_is_a_failure_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_collection(model_id=None, dimension=None)
    _source([_row("c1")])
    monkeypatch.setattr(mod, "resolve_collection_names", lambda **kwargs: ["kb"])
    monkeypatch.setattr(
        mod, "resolve_target_model", lambda model_id: (TARGET, _Adapter())
    )

    assert mod.main(["--model", TARGET.id]) == 1


def test_one_limiter_paces_every_collection() -> None:
    limiter = mod.RateLimiter(rate=0.0)
    paced = mod.PacedEmbedding(_Adapter(), limiter)
    assert paced.get_dimension() == 2
    assert paced.abilities == ["embedding"]


def test_the_configured_batch_size_is_used() -> None:
    _save_collection()
    _source([_row(f"c{index}", doc_id="d1") for index in range(5)])
    adapter = _Adapter()

    _run(adapter, batch_size=2)

    assert [len(call) for call in adapter.calls] == [2, 2, 1]


def test_the_async_path_encodes_one_text_per_call() -> None:
    _save_collection()
    _source([_row("c1"), _row("c2")])
    adapter = _Adapter()

    report = _run(adapter, async_mode=True)

    assert report.failures == []
    assert adapter.calls == ["text c1", "text c2"]


def test_a_target_that_writes_the_source_table_is_refused_before_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
        written_embeddings_table_name,
    )

    # Distinct ids, one physical table: the copy would overwrite its own source.
    assert written_embeddings_table_name("old-model") == written_embeddings_table_name(
        "old_model"
    )
    _save_collection(model_id="old-model", dimension=2)
    _table(scope.source_table("old-model"), [_row("c1")])
    adapter = _Adapter()
    same = EmbeddingModelConfig(id="old_model", model_name="old_model", dimension=2)
    monkeypatch.setattr(mod, "resolve_collection_names", lambda **kwargs: ["kb"])
    monkeypatch.setattr(mod, "resolve_target_model", lambda model_id: (same, adapter))

    report = mod.embed_collections(same.id)

    assert "both write" in report.collections[0].failures[0].error
    assert report.failed
    assert adapter.calls == []
    assert {row["chunk_id"] for row in _rows_in(scope.source_table("old-model"))} == {
        "c1"
    }
