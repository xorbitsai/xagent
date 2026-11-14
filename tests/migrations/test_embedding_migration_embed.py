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
    model: Optional[str] = None,
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
        "model": model if model is not None else collection_model(collection),
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


def test_one_limiter_paces_every_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    class _Clock:
        def __init__(self) -> None:
            self.now = 0.0
            self.slept: List[float] = []

        def monotonic(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.slept.append(round(seconds, 6))
            self.now += seconds

    clock = _Clock()
    monkeypatch.setattr(
        mod, "time", SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep)
    )
    _save_collection()
    _save_collection(name="kb2")
    _source(
        [
            _row("c1"),
            _row("c2"),
            _row("c3", collection="kb2"),
            _row("c4", collection="kb2"),
        ]
    )
    adapter = _Adapter()
    monkeypatch.setattr(mod, "resolve_target_model", lambda model_id: (TARGET, adapter))

    report = mod.embed_collections(
        TARGET.id, collections=["kb", "kb2"], rate_limit=2.0, batch_size=2
    )

    assert [item.chunks_embedded for item in report.collections] == [2, 2]
    assert [len(call) for call in adapter.calls] == [2, 2]
    # Two texts per second, one limiter for both collections: the second
    # collection's batch waits for the budget the first one spent.
    assert clock.slept == [1.0]


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


def test_a_target_that_writes_the_source_table_is_skipped_before_writing(
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

    assert report.collections[0].collection_skip == scope.SKIP_SHARED_TABLE
    assert adapter.calls == []
    assert {row["chunk_id"] for row in _rows_in(scope.source_table("old-model"))} == {
        "c1"
    }


def test_a_target_that_aliases_a_source_fallback_never_writes_or_reruns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The target's writer table is the source's provider-name fallback. A first
    # copy there would make every later run see rows in two source spellings.
    _save_collection(model_id="old-config", dimension=2)
    _table(scope.source_table("old-config"), [_row("c1")])
    monkeypatch.setattr(
        scope,
        "_legacy_spelling",
        lambda model_id: (
            "embeddings_text_embedding_3_small" if model_id == "old-config" else None
        ),
    )
    target = EmbeddingModelConfig(
        id="text-embedding-3-small", model_name="other-provider", dimension=2
    )
    adapter = _Adapter()
    monkeypatch.setattr(mod, "resolve_collection_names", lambda **kwargs: ["kb"])
    monkeypatch.setattr(mod, "resolve_target_model", lambda model_id: (target, adapter))

    first = mod.embed_collections(target.id)
    second = mod.embed_collections(target.id)

    assert [item.collection_skip for item in first.collections] == [
        scope.SKIP_SHARED_TABLE
    ]
    assert [item.collection_skip for item in second.collections] == [
        scope.SKIP_SHARED_TABLE
    ]
    assert adapter.calls == []
    assert scope.source_table(target.id) not in scope.table_names(scope.connection())


def test_a_target_that_is_the_bound_model_copies_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_collection()
    _source([_row("c1")])
    same = EmbeddingModelConfig(id=SOURCE_MODEL, model_name=SOURCE_MODEL, dimension=2)
    adapter = _Adapter()
    monkeypatch.setattr(mod, "resolve_collection_names", lambda **kwargs: ["kb"])
    monkeypatch.setattr(mod, "resolve_target_model", lambda model_id: (same, adapter))

    report = mod.embed_collections(SOURCE_MODEL)

    assert report.collections[0].collection_skip == scope.SKIP_TARGET_IS_SOURCE
    assert adapter.calls == []


# --- a delete or retirement landing during the copy ---


def _delete_document(doc_id: str, *, everywhere: bool) -> int:
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        get_vector_index_store,
    )

    model_tag = None if everywhere else TARGET.id
    return get_vector_index_store().delete_embedding_records(
        "kb", doc_id, model_tag=model_tag, is_admin=True
    )


def test_a_delete_that_lands_before_the_upsert_does_not_survive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_collection()
    _source([_row("c1"), _row("c2", doc_id="d2")])
    real = mod.write_vectors_to_db
    injected = {"done": False}

    def _delete_then_write(**kwargs: Any) -> Any:
        if not injected["done"]:
            injected["done"] = True
            # The document delete commits while this batch is being encoded.
            _delete_document("d1", everywhere=True)
        return real(**kwargs)

    monkeypatch.setattr(mod, "write_vectors_to_db", _delete_then_write)

    report = _run()

    assert report.failures == []
    assert report.chunks_removed == 1
    remaining = {row["doc_id"] for row in _rows_in(scope.source_table(TARGET.id))}
    assert remaining == {"d2"}


def test_a_delete_that_lands_after_the_upsert_clears_the_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_collection()
    _source([_row("c1"), _row("c2", doc_id="d2")])
    real = mod.write_vectors_to_db
    injected = {"done": False}

    def _write_then_delete(**kwargs: Any) -> Any:
        result = real(**kwargs)
        if not injected["done"]:
            injected["done"] = True
            # Committed after the copy's revalidation: the delete path clears
            # every model table itself.
            _delete_document("d1", everywhere=True)
        return result

    monkeypatch.setattr(mod, "write_vectors_to_db", _write_then_delete)

    report = _run()

    assert report.failures == []
    remaining = {row["doc_id"] for row in _rows_in(scope.source_table(TARGET.id))}
    assert remaining == {"d2"}


# --- reads ---


def test_an_all_held_rerun_reads_the_source_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_collection()
    _source(
        [
            _row("c1"),
            _row("c2", doc_id="d2"),
            _row("c3"),
            _row("c4", doc_id="d2"),
        ]
    )
    _seed_target(["c1", "c2", "c3", "c4"])
    calls: List[Any] = []
    real = scope.iter_source_chunks

    def _counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs.get("doc_id"))
        return real(*args, **kwargs)

    monkeypatch.setattr(mod, "iter_source_chunks", _counted)

    report = _run()

    assert report.documents_unchanged == 2
    assert report.chunks_embedded == 0
    assert calls == []


def test_only_documents_with_pending_chunks_are_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_collection()
    _source([_row("c1"), _row("c2", doc_id="d2"), _row("c3", doc_id="d1")])
    _seed_target(["c1", "c2"])
    calls: List[Any] = []
    real = scope.iter_source_chunks

    def _counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs.get("doc_id"))
        return real(*args, **kwargs)

    monkeypatch.setattr(mod, "iter_source_chunks", _counted)

    report = _run()

    assert report.chunks_embedded == 1
    assert calls == ["d1"] * 2  # the copy's read and the reconciler's re-read


# --- one table set per model, whatever the spelling ---


@pytest.mark.parametrize(
    ("source_id", "target_id", "legacy"),
    [
        ("old-model", "old_model", None),
        ("BAAI/old-model", "baai_old_model", None),
        (
            "old-config",
            "text-embedding-3-small",
            {"old-config": "embeddings_text_embedding_3_small"},
        ),
        (
            "old-config",
            "text-embedding-3-small",
            {"text-embedding-3-small": "embeddings_old_config"},
        ),
    ],
    ids=[
        "hyphen-underscore",
        "once-twice-applied",
        "target-is-source-fallback",
        "source-is-target-fallback",
    ],
)
def test_a_target_that_shares_any_readable_table_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    source_id: str,
    target_id: str,
    legacy: Optional[Dict[str, str]],
) -> None:
    monkeypatch.setattr(
        scope, "_legacy_spelling", lambda model_id: (legacy or {}).get(model_id)
    )
    _save_collection(model_id=source_id, dimension=2)
    _table(scope.source_table(source_id), [_row("c1")])
    target = EmbeddingModelConfig(id=target_id, model_name=target_id, dimension=2)
    adapter = _Adapter()
    monkeypatch.setattr(mod, "resolve_collection_names", lambda **kwargs: ["kb"])
    monkeypatch.setattr(mod, "resolve_target_model", lambda model_id: (target, adapter))

    report = mod.embed_collections(target_id)

    assert report.collections[0].collection_skip == scope.SKIP_SHARED_TABLE
    assert adapter.calls == []
    assert {row["chunk_id"] for row in _rows_in(scope.source_table(source_id))} == {
        "c1"
    }
    assert not scope.target_chunk_ids("kb", target_id)


def test_a_row_another_model_wrote_in_the_target_table_refuses_the_migration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``new-model`` and ``new_model`` are distinct bindings with one writer
    # table: search would serve the other model's rows after the flip.
    _save_collection()
    _source([_row("c1")])
    _table(scope.source_table("new-model"), [_row("c9", model="new_model")])
    monkeypatch.setattr(
        mod,
        "resolve_collection_names",
        lambda **kwargs: pytest.fail("refused collections are not copied"),
    )

    eligibility = scope.source_eligibility("kb", target_model_id="new-model")

    assert eligibility.reason == scope.SKIP_TARGET_ROWS_IN_OTHER_TABLE


def test_a_target_fallback_table_holding_the_collection_refuses_the_migration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_collection()
    _source([_row("c1")])
    _table("embeddings_new_legacy", [_row("c9")])
    monkeypatch.setattr(
        scope,
        "_legacy_spelling",
        lambda model_id: ("embeddings_new_legacy" if model_id == "new-model" else None),
    )

    assert (
        scope.source_eligibility("kb", target_model_id="new-model").reason
        == scope.SKIP_TARGET_ROWS_IN_OTHER_TABLE
    )


def test_the_target_holds_only_chunk_ids_its_own_model_wrote() -> None:
    from xagent.migrations.embedding.scope import target_chunk_ids

    _save_collection()
    _source([_row("c1")])
    _table(
        scope.source_table(TARGET.id),
        [_row("held", model=TARGET.id), _row("theirs", model="new_model")],
    )

    assert target_chunk_ids("kb", TARGET.id) == {"held"}


def test_a_target_row_without_a_model_value_is_not_held() -> None:
    from xagent.migrations.embedding.scope import target_chunk_ids

    _save_collection()
    _source([_row("c1")])
    _table(scope.source_table(TARGET.id), [_row("c9", model="")])

    assert target_chunk_ids("kb", TARGET.id) == set()


# --- the delete contract is read from the core tables ---


def test_a_delete_that_takes_the_core_rows_leaves_no_target_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_collection()
    core_schema = pa.schema(
        [
            ("collection", pa.string()),
            ("doc_id", pa.string()),
            ("chunk_id", pa.string()),
            ("user_id", pa.int64()),
        ]
    )
    _table(
        "chunks",
        [
            {"collection": "kb", "doc_id": "d1", "chunk_id": "c1", "user_id": 7},
            {"collection": "kb", "doc_id": "d2", "chunk_id": "c2", "user_id": 7},
        ],
        schema=core_schema,
    )
    _source([_row("c1"), _row("c2", doc_id="d2")])
    real = mod.write_vectors_to_db
    injected = {"done": False}

    def _delete_core_rows_then_write(**kwargs: Any) -> Any:
        if not injected["done"]:
            injected["done"] = True
            # The cascade deletes the core rows first and sweeps embedding
            # tables it can see; this target table does not exist yet.
            scope.connection().open_table("chunks").delete("doc_id = 'd1'")
        return real(**kwargs)

    monkeypatch.setattr(mod, "write_vectors_to_db", _delete_core_rows_then_write)

    report = _run()

    assert report.failures == []
    remaining = {row["doc_id"] for row in _rows_in(scope.source_table(TARGET.id))}
    assert remaining == {"d2"}
    # The source row is still there: only the core table says the chunk is gone.
    assert {row["doc_id"] for row in _rows_in(scope.source_table(SOURCE_MODEL))} == {
        "d1",
        "d2",
    }


def test_the_documents_cascade_sweeps_a_table_a_writer_created_meantime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.LanceDB import schema_manager
    from xagent.core.tools.core.RAG_tools.storage import lancedb_stores

    for name in (
        "ensure_documents_table",
        "ensure_parses_table",
        "ensure_chunks_table",
        "ensure_main_pointers_table",
        "ensure_ingestion_runs_table",
    ):
        monkeypatch.setattr(schema_manager, name, lambda *args, **kwargs: None)

    conn = scope.connection()
    core = pa.schema(
        [
            ("collection", pa.string()),
            ("doc_id", pa.string()),
            ("chunk_id", pa.string()),
            ("user_id", pa.int64()),
        ]
    )
    documents = pa.Table.from_pylist(
        [{"collection": "kb", "doc_id": "d1", "chunk_id": "", "user_id": 7}],
        schema=core,
    )
    conn.create_table("documents", data=documents)
    conn.create_table("chunks", data=documents)

    late = scope.source_table(TARGET.id)
    real_delete = lancedb_stores._vis_delete_rows_by_filters
    created: List[str] = []

    def _delete_then_writer_commits(table: Any, filters: List[str]) -> int:
        count = real_delete(table, filters)
        if count and not created:
            # A writer that sampled the source while these core rows still
            # existed commits its target rows while the cascade runs.
            conn.create_table(
                late,
                data=pa.Table.from_pylist(
                    [_row("c1", model=TARGET.id)], schema=_SCHEMA
                ),
            )
            created.append(late)
        return count

    monkeypatch.setattr(
        lancedb_stores, "_vis_delete_rows_by_filters", _delete_then_writer_commits
    )

    lancedb_stores._vis_cascade_delete_documents(
        conn,
        collection="kb",
        doc_ids=["d1"],
        user_id=None,
        is_admin=True,
        preview_only=False,
        confirm=True,
    )

    assert created == [late]
    assert _rows_in(late) == []
