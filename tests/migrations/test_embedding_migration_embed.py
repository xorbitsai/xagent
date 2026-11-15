"""Warm-up: write the target's vectors for backed, usable source rows only."""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import pytest

from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import EmbeddingModelConfig
from xagent.core.model.storage.error import ModelNotFoundError
from xagent.migrations.embedding import embed as mod
from xagent.migrations.embedding import scope

LEASES = "collection_write_leases"

SOURCE = "BAAI/old-model"
TARGET = "new-model"
VECTOR = [0.25, 0.75]


def _config(model_id: str, **overrides: Any) -> EmbeddingModelConfig:
    fields: Dict[str, Any] = {
        "id": model_id,
        "model_name": model_id,
        "model_provider": "dashscope",
        "api_key": "k",
        "dimension": 2,
        "abilities": ["embedding"],
    }
    fields.update(overrides)
    return EmbeddingModelConfig(**fields)


class _Hub:
    def __init__(self, models: Dict[str, EmbeddingModelConfig]) -> None:
        self.models = models

    def load(self, model_id: str, **_kwargs: Any) -> EmbeddingModelConfig:
        for config in self.models.values():
            if model_id in (config.id, config.model_name):
                return config
        raise ModelNotFoundError(model_id)

    def list(self) -> Dict[str, EmbeddingModelConfig]:
        return self.models

    def close(self) -> None:
        pass


class _Adapter(BaseEmbedding):
    def __init__(self) -> None:
        self.calls: List[List[str]] = []
        self.builds = 0
        self.before_encode: Optional[Callable[[List[str]], None]] = None
        self.vectors: Callable[[List[str]], Any] = lambda texts: [
            list(VECTOR) for _ in texts
        ]

    def encode(self, text: Any, dimension: Any = None, instruct: Any = None) -> Any:
        texts = [text] if isinstance(text, str) else list(text)
        self.calls.append(texts)
        if self.before_encode is not None:
            self.before_encode(texts)
        return self.vectors(texts)

    def get_dimension(self) -> Optional[int]:
        return 2

    @property
    def abilities(self) -> List[str]:
        return ["embedding"]


@pytest.fixture(autouse=True)
def hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[_Hub]:
    """A temporary LanceDB directory and SQL file; the hub is served in memory."""
    from xagent.core.tools.core.RAG_tools.storage import reset_rag_storage_for_tests
    from xagent.core.tools.core.RAG_tools.utils import model_resolver

    lancedb_dir = tmp_path / "warmup-lancedb"
    lancedb_dir.mkdir()
    monkeypatch.setenv("LANCEDB_DIR", str(lancedb_dir))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'hub.db'}")
    model_resolver._reset_model_hub_cache()
    served = _Hub({SOURCE: _config(SOURCE), TARGET: _config(TARGET)})
    monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", lambda: served)
    reset_rag_storage_for_tests()
    yield served
    reset_rag_storage_for_tests()
    model_resolver._reset_model_hub_cache()


@pytest.fixture(autouse=True)
def index_calls(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    from xagent.core.tools.core.RAG_tools.core.schemas import IndexResult
    from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
        LanceDBVectorIndexStore,
    )

    record = SimpleNamespace(calls=[], status="below_threshold")

    def _create_index(self: Any, model_tag: str, readonly: bool = False) -> Any:
        record.calls.append((model_tag, readonly))
        return IndexResult(status=record.status, advice=None, fts_enabled=True)

    monkeypatch.setattr(LanceDBVectorIndexStore, "create_index", _create_index)
    return record


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> _Adapter:
    from xagent.core.tools.core.RAG_tools.utils import model_resolver

    built = _Adapter()

    def _factory(*args: Any, **kwargs: Any) -> _Adapter:
        built.builds += 1
        return built

    monkeypatch.setattr(model_resolver, "create_embedding_adapter", _factory)
    return built


def _save_collection(
    name: str = "kb", model_id: Optional[str] = SOURCE, dimension: Optional[int] = 2
) -> None:
    import asyncio

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


def _store() -> Any:
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        get_vector_index_store,
    )

    return get_vector_index_store()


def _seed_core(
    chunks: Dict[str, Sequence[str]],
    *,
    collection: str = "kb",
    stale: Sequence[str] = (),
) -> None:
    """Document, parse and chunk rows; a ``stale`` chunk was made from older content."""
    from xagent.core.tools.core.RAG_tools.utils.metadata_utils import (
        serialize_metadata,
    )

    now = datetime.now(timezone.utc)
    store = _store()
    store.upsert_documents(
        [
            {
                "collection": collection,
                "doc_id": doc_id,
                "file_id": None,
                "source_path": f"/{doc_id}.txt",
                "file_type": "txt",
                "content_hash": "C2",
                "uploaded_at": now,
                "title": None,
                "language": None,
                "user_id": None,
            }
            for doc_id in chunks
        ]
    )
    store.upsert_parses(
        [
            {
                "collection": collection,
                "doc_id": doc_id,
                "parse_hash": "p",
                "parser": "local:default@v1.0.0",
                "created_at": now,
                "params_json": json.dumps({"content_hash": "C2"}),
                "parsed_content": "[]",
                "user_id": None,
            }
            for doc_id in chunks
        ]
    )
    store.upsert_chunks(
        [
            {
                "collection": collection,
                "doc_id": doc_id,
                "parse_hash": "p",
                "chunk_id": chunk_id,
                "index": index,
                "text": chunk_id,
                "page_number": None,
                "section": None,
                "anchor": None,
                "json_path": None,
                "chunk_hash": "x",
                "config_hash": "cfg-old" if chunk_id in stale else "cfg-new",
                "created_at": now,
                "metadata": serialize_metadata(
                    {"content_hash": "C1" if chunk_id in stale else "C2"}
                ),
                "user_id": None,
            }
            for doc_id, chunk_ids in chunks.items()
            for index, chunk_id in enumerate(chunk_ids)
        ]
    )


def _src(
    chunk_id: str,
    doc_id: str = "d1",
    *,
    user_id: Optional[int] = 7,
    text: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "user_id": user_id,
        "text": f"text {chunk_id}" if text is None else text,
        "metadata": metadata,
    }


def _write(
    rows: List[Dict[str, Any]],
    *,
    model: str = SOURCE,
    collection: str = "kb",
    vector: Optional[List[float]] = None,
) -> None:
    """Rows written through the ingestion write path, one call per owner."""
    from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData
    from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
        write_vectors_to_db,
    )

    owners: Dict[Optional[int], List[Dict[str, Any]]] = {}
    for row in rows:
        owners.setdefault(row["user_id"], []).append(row)
    for owner, items in owners.items():
        write_vectors_to_db(
            collection,
            [
                ChunkEmbeddingData(
                    doc_id=row["doc_id"],
                    chunk_id=row["chunk_id"],
                    parse_hash="p",
                    model=model,
                    vector=vector or [0.5, 0.5],
                    text=row["text"],
                    chunk_hash=f"h-{row['chunk_id']}",
                    metadata=row["metadata"],
                )
                for row in items
            ],
            create_index=False,
            user_id=owner,
            verify_bound=False,
        )


def _rows(table: str) -> List[Dict[str, Any]]:
    conn = scope.connection()
    if table not in scope.table_names(conn):
        return []
    return conn.open_table(table).search().where("true").to_arrow().to_pylist()


def _target() -> List[Dict[str, Any]]:
    return _rows(scope.source_table(TARGET))


def _target_ids(doc_id: Optional[str] = None) -> List[str]:
    return sorted(
        row["chunk_id"]
        for row in _target()
        if doc_id is None or row["doc_id"] == doc_id
    )


def _versions() -> Dict[str, int]:
    conn = scope.connection()
    return {name: conn.open_table(name).version for name in scope.table_names(conn)}


def _run(*, dry_run: bool = False, rate_limit: float = 0.0) -> mod.CollectionReport:
    reports = mod.warm_up(
        TARGET, collections=["kb"], dry_run=dry_run, rate_limit=rate_limit
    )
    assert len(reports) == 1
    return reports[0]


# --- what is written ---


def test_a_written_row_carries_its_source_row_and_the_target_vector(
    adapter: _Adapter,
) -> None:
    _save_collection()
    _seed_core({"d1": ["c1", "c2", "c3"]})
    _write(
        [
            _src("c1", metadata={"page": 1}),
            _src("c2", user_id=8),
            _src("c3", user_id=8, metadata={"a": [1, 2]}),
        ]
    )

    report = _run()

    assert (report.failures, report.error, report.written) == ({}, None, 3)
    assert sorted(adapter.calls) == [["text c1"], ["text c2", "text c3"]]
    source = {row["chunk_id"]: row for row in _rows(scope.source_table(SOURCE))}
    written = {row["chunk_id"]: row for row in _target()}
    assert set(written) == {"c1", "c2", "c3"}
    for chunk_id, row in written.items():
        origin = source[chunk_id]
        for column in (
            "collection",
            "doc_id",
            "chunk_id",
            "parse_hash",
            "user_id",
            "text",
            "chunk_hash",
        ):
            assert row[column] == origin[column], column
        assert json.loads(row["metadata"] or "null") == json.loads(
            origin["metadata"] or "null"
        )
        assert row["model"] == TARGET
        assert row["vector"] == pytest.approx(VECTOR)


def test_a_second_run_writes_nothing(adapter: _Adapter) -> None:
    _save_collection()
    _seed_core({"d1": ["c1"], "d2": ["c2"]})
    _write([_src("c1"), _src("c2", "d2")])
    assert _run().written == 2
    adapter.calls.clear()
    before = _versions()

    report = _run()

    assert (report.pending, report.written, report.removed) == (0, 0, 0)
    assert adapter.calls == []
    assert {k: v for k, v in _versions().items() if k != LEASES} == {
        k: v for k, v in before.items() if k != LEASES
    }


def test_a_chunk_the_target_holds_is_not_encoded_again(adapter: _Adapter) -> None:
    _save_collection()
    _seed_core({"d1": ["c1", "c2"]})
    _write([_src("c1"), _src("c2")])
    _write([_src("c1")], model=TARGET, vector=list(VECTOR))

    report = _run()

    assert adapter.calls == [["text c2"]]
    assert (report.documents, report.pending, report.written) == (1, 1, 1)


def test_a_target_row_with_an_unusable_vector_is_replaced(adapter: _Adapter) -> None:
    _save_collection()
    _seed_core({"d1": ["c1"]})
    _write([_src("c1")])
    _write([_src("c1")], model=TARGET, vector=[0.0, 0.0])

    report = _run()

    assert report.written == 1
    assert [row["vector"] for row in _target()] == [pytest.approx(VECTOR)]


def test_a_target_row_of_another_dimension_is_not_held(adapter: _Adapter) -> None:
    _save_collection()
    _seed_core({"d1": ["c1"]})
    _write([_src("c1")])
    _write([_src("c1")], model=TARGET, vector=[0.5, 0.5, 0.5])

    report = _run()

    assert (report.pending, list(report.failures)) == (1, ["d1"])


def test_a_bound_collection_without_rows_writes_nothing(
    adapter: _Adapter, index_calls: SimpleNamespace
) -> None:
    _save_collection()

    report = _run()

    assert (report.failed, report.pending, report.skip_reason) == (False, 0, None)
    assert (adapter.calls, index_calls.calls, _target()) == ([], [], [])


@pytest.mark.parametrize("core", ["other-document-only", "no-core-table"])
def test_rows_no_core_chunk_row_backs_are_counted_not_written(
    adapter: _Adapter, core: str
) -> None:
    _save_collection()
    if core == "other-document-only":
        _seed_core({"d1": ["c1"]})
    _write([_src("c1"), _src("c2"), _src("c3", "d2")])

    report = _run()

    if core == "other-document-only":
        assert report.skipped == {
            scope.SKIP_NO_CORE_CHUNK: 1,
            mod.SKIP_NO_CORE_DOCUMENT: 1,
        }
        assert adapter.calls == [["text c1"]]
        assert _target_ids() == ["c1"]
    else:
        assert report.skipped == {mod.SKIP_NO_CORE_DOCUMENT: 3}
        assert adapter.calls == []
        assert _target_ids() == []
    assert report.failures == {}


def test_excluded_rows_beside_a_usable_one_are_counted_not_written(
    adapter: _Adapter,
) -> None:
    _save_collection()
    _seed_core({"d1": ["c1", "c2", "c3"]})
    _write(
        [
            _src("c1"),
            _src("c2", user_id=None),
            _src("c3", text=""),
            _src("c4", ""),
            _src("", "d1"),
        ]
    )

    report = _run()

    assert report.skipped == {
        scope.SKIP_NO_OWNER: 1,
        scope.SKIP_NO_TEXT: 1,
        scope.SKIP_NO_DOC_ID: 1,
        scope.SKIP_NO_CHUNK_ID: 1,
    }
    assert (report.failures, report.written) == ({}, 1)
    assert _target_ids() == ["c1"]


# --- a dry run, a refusal and the binding ---


def test_a_dry_run_reads_only(adapter: _Adapter) -> None:
    _save_collection()
    _seed_core({"d1": ["c1", "c2"]})
    _write([_src("c1"), _src("c2")])
    before = _versions()

    report = _run(dry_run=True)

    assert (report.documents, report.pending, report.written) == (1, 2, 0)
    assert (report.failures, adapter.builds, adapter.calls) == ({}, 0, [])
    assert _versions() == before


@pytest.mark.parametrize("case", ["unbound", "target-is-source", "foreign-target-row"])
def test_a_refused_collection_gets_no_write(adapter: _Adapter, case: str) -> None:
    bound = {"unbound": None, "target-is-source": TARGET}.get(case, SOURCE)
    _save_collection(model_id=bound, dimension=2 if bound else None)
    _seed_core({"d1": ["c1"]})
    _write([_src("c1")], model=bound or SOURCE)
    if case == "foreign-target-row":
        _write([_src("c9")], model="new_model")
    before = _versions()

    report = _run()

    assert (
        report.skip_reason
        == {
            "unbound": scope.SKIP_UNBOUND,
            "target-is-source": scope.SKIP_TARGET_IS_SOURCE,
            "foreign-target-row": scope.SKIP_TARGET_ROWS_IN_OTHER_TABLE,
        }[case]
    )
    assert adapter.calls == []
    assert _versions() == before


def test_a_collection_name_the_write_path_would_strip_is_refused(
    adapter: _Adapter,
) -> None:
    _save_collection()
    _seed_core({"d1": ["c1"]})
    _write([_src("c1")])
    conn = scope.connection()
    for table in (scope.METADATA_TABLE, scope.CORE_TABLE, scope.source_table(SOURCE)):
        column = "name" if table == scope.METADATA_TABLE else "collection"
        conn.open_table(table).update(where=f"{column} = 'kb'", values={column: " kb"})

    [report] = mod.warm_up(TARGET, collections=[" kb"])

    assert report.skip_reason == mod.SKIP_NAME_NOT_STRIPPED
    assert adapter.calls == []
    assert _target() == []


def test_a_run_changes_no_table_but_the_target_writer_table(
    adapter: _Adapter,
) -> None:
    _save_collection()
    _seed_core({"d1": ["c1"]})
    _write([_src("c1")])
    before = _versions()

    assert _run().written == 1

    after = _versions()
    writer = scope.source_table(TARGET)
    assert set(after) == set(before) | {writer, LEASES}
    assert {name: after[name] for name in before} == before
    assert _rows(LEASES) == []
    assert scope.collection_info("kb").embedding_model_id == SOURCE


# --- deletes landing during the write ---


def test_a_row_committed_before_a_deletes_second_sweep_goes_with_the_delete(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xagent.core.tools.core.RAG_tools.storage import lancedb_stores

    _save_collection()
    _seed_core({"d1": ["c1", "c2"]})
    _write([_src("c1"), _src("c2")])
    chunks_gone, resume = threading.Event(), threading.Event()
    real_delete = lancedb_stores._vis_delete_rows_by_filters

    def _pause_after_the_chunk_rows(table: Any, filters: List[str]) -> int:
        count = real_delete(table, filters)
        if getattr(table, "name", "") == "chunks":
            chunks_gone.set()
            assert resume.wait(30)
        return count

    monkeypatch.setattr(
        lancedb_stores, "_vis_delete_rows_by_filters", _pause_after_the_chunk_rows
    )
    errors: List[BaseException] = []

    def _delete() -> None:
        try:
            _store().delete_documents_data("kb", ["d1"], None, True)
        except BaseException as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    deleter = threading.Thread(target=_delete)

    def _delete_starts_during_the_provider_call(texts: List[str]) -> None:
        deleter.start()
        assert chunks_gone.wait(30)

    adapter.before_encode = _delete_starts_during_the_provider_call
    real_write = mod.write_vectors_to_db
    seen: Dict[str, List[str]] = {}

    def _write_then_let_the_delete_finish(*args: Any, **kwargs: Any) -> Any:
        result = real_write(*args, **kwargs)
        seen["committed"] = _target_ids("d1")
        resume.set()
        deleter.join(30)
        seen["after the delete"] = _target_ids("d1")
        return result

    monkeypatch.setattr(mod, "write_vectors_to_db", _write_then_let_the_delete_finish)

    report = _run()

    assert errors == []
    assert seen == {"committed": ["c1", "c2"], "after the delete": []}
    assert (report.failures, report.removed) == ({}, 0)


def test_a_row_committed_after_a_deletes_second_sweep_is_removed_by_the_recheck(
    adapter: _Adapter,
) -> None:
    from xagent.core.tools.core.RAG_tools.management import collections as mc

    _save_collection()
    _seed_core({"d1": ["c1", "c2"], "d2": ["c3"]})
    _write([_src("c1"), _src("c2"), _src("c3", "d2")])

    def _whole_delete_inside_the_provider_call(texts: List[str]) -> None:
        if "text c1" in texts:
            result = mc.delete_document("kb", "d1", 1, is_admin=True)
            assert result.status == "success", result.message

    adapter.before_encode = _whole_delete_inside_the_provider_call

    report = _run()

    assert (report.failures, report.written, report.removed) == ({}, 3, 2)
    assert _target_ids() == ["c3"]


def test_a_retirement_during_the_provider_call_leaves_no_retired_row(
    adapter: _Adapter,
) -> None:
    _save_collection()
    _seed_core({"d1": ["c1", "c2"]}, stale=["c1"])
    _write([_src("c1"), _src("c2")])

    def _retire(texts: List[str]) -> None:
        assert _handle().retire_superseded_chunks("d1", "p") == ["c1"]

    adapter.before_encode = _retire

    report = _run()

    assert (report.failures, report.removed) == ({}, 1)
    assert _target_ids() == ["c2"]


def test_the_recheck_removes_only_rows_it_attempted(adapter: _Adapter) -> None:
    _save_collection()
    _seed_core({"d1": ["c1", "c9"]})
    _write([_src("c1"), _src("c9")])
    _write([_src("c9")], model=TARGET, vector=list(VECTOR))

    def _core_rows_go(texts: List[str]) -> None:
        scope.connection().open_table("chunks").delete("doc_id = 'd1'")

    adapter.before_encode = _core_rows_go

    report = _run()

    assert adapter.calls == [["text c1"]]
    assert report.removed == 1
    assert _target_ids() == ["c9"]


def test_a_write_that_fails_partway_is_rechecked(adapter: _Adapter) -> None:
    _save_collection()
    chunk_ids = [f"c{index:02d}" for index in range(12)]
    _seed_core({"d1": chunk_ids})
    _write([_src(chunk_id) for chunk_id in chunk_ids])

    def _second_batch_fails_after_the_core_rows_go(texts: List[str]) -> None:
        if len(adapter.calls) == 2:
            scope.connection().open_table("chunks").delete("doc_id = 'd1'")
            raise RuntimeError("provider down")

    adapter.before_encode = _second_batch_fails_after_the_core_rows_go

    report = _run()

    assert list(report.failures) == ["d1"]
    assert (report.written, report.removed) == (10, 10)
    assert _target_ids() == []


# --- the write lease ---


@pytest.fixture
def stall(monkeypatch: pytest.MonkeyPatch) -> Callable[[], None]:
    """Move the lease clock past the TTL, as a process stalled that long would."""
    from xagent.core.tools.core.RAG_tools.management import write_leases

    offset = [0.0]
    clock = SimpleNamespace(
        monotonic=lambda: time.monotonic() + offset[0], sleep=time.sleep
    )
    monkeypatch.setattr(write_leases, "time", clock)

    def _stall() -> None:
        offset[0] += write_leases.DEFAULT_LEASE_TTL_SECONDS + 1

    return _stall


@pytest.fixture
def compactions(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    """Calls to the store's compaction, with whether the lease was held."""
    from xagent.core.tools.core.RAG_tools.management.migration_state import (
        holds_lease,
    )
    from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
        LanceDBVectorIndexStore,
    )

    calls: List[Any] = []
    real = LanceDBVectorIndexStore.compact_tables

    def _compact(self: Any, table_names: Sequence[str], policy: Any = None) -> Any:
        calls.append((list(table_names), holds_lease("kb")))
        return real(self, table_names, policy)

    monkeypatch.setattr(LanceDBVectorIndexStore, "compact_tables", _compact)
    return calls


def _target_tables() -> List[str]:
    from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
        embeddings_table_candidates,
    )

    return list(dict.fromkeys(embeddings_table_candidates(TARGET)))


def _freeze() -> None:
    from xagent.core.tools.core.RAG_tools.management import migration_state as ms

    ms.write_migration_state(
        "kb", ms.CollectionMigrationState(state=ms.FROZEN, source_model_id=SOURCE)
    )


def test_the_lease_is_held_only_while_a_document_is_written(
    adapter: _Adapter,
) -> None:
    from xagent.core.tools.core.RAG_tools.management import write_leases

    _save_collection()
    _seed_core({"d1": ["c1"]})
    _write([_src("c1")])
    seen: List[List[str]] = []
    adapter.before_encode = lambda texts: seen.append(
        [lease.operation for lease in write_leases.live_leases("kb")]
    )

    assert _run().written == 1
    assert seen == [[mod.LEASE_OPERATION]]
    assert write_leases.live_leases("kb") == []


def test_a_freeze_stops_the_collection_at_its_next_document(
    adapter: _Adapter,
    index_calls: SimpleNamespace,
    compactions: List[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "COMPACT_EVERY_DOCUMENTS", 1, raising=False)
    _save_collection()
    _seed_core({"d1": ["c1"], "d2": ["c2"]})
    _write([_src("c1"), _src("c2", "d2")])
    adapter.before_encode = lambda texts: _freeze()

    report = _run()

    assert (report.stopped, report.failed, report.written) == (
        mod.STOP_FROZEN,
        False,
        1,
    )
    assert (_target_ids(), adapter.calls, index_calls.calls) == (
        ["c1"],
        [["text c1"]],
        [],
    )
    assert compactions == [(_target_tables(), True)]


def test_a_frozen_collection_gets_no_index_build(
    adapter: _Adapter, index_calls: SimpleNamespace, compactions: List[Any]
) -> None:
    _save_collection()
    _seed_core({"d1": ["c1"]})
    _write([_src("c1")])
    _write([_src("c1")], model=TARGET, vector=list(VECTOR))
    _freeze()

    report = _run()

    assert (report.stopped, report.pending, index_calls.calls) == (
        mod.STOP_FROZEN,
        0,
        [],
    )
    assert compactions == []


@pytest.mark.parametrize("change", ["rebound", "metadata-row-gone"])
def test_a_binding_changed_under_the_run_stops_the_collection(
    adapter: _Adapter, index_calls: SimpleNamespace, change: str
) -> None:
    _save_collection()
    _seed_core({"d1": ["c1"], "d2": ["c2"]})
    _write([_src("c1"), _src("c2", "d2")])

    def _change(texts: List[str]) -> None:
        if change == "rebound":
            _save_collection(model_id=TARGET)
        else:
            metadata = scope.connection().open_table(scope.METADATA_TABLE)
            metadata.delete("name = 'kb'")

    adapter.before_encode = _change

    report = _run()

    assert (report.stopped, report.failed) == (mod.STOP_REBOUND, False)
    assert (_target_ids(), index_calls.calls) == (["c1"], [])


def test_a_lease_lost_before_the_write_writes_nothing(
    adapter: _Adapter, stall: Callable[[], None]
) -> None:
    _save_collection()
    _seed_core({"d1": ["c1"]})
    _write([_src("c1")])
    adapter.before_encode = lambda texts: stall()

    report = _run()

    assert list(report.failures) == ["d1"]
    assert "lease" in report.failures["d1"]
    assert (report.written, _target()) == (0, [])


def test_a_lease_lost_before_the_delete_keeps_the_rows(
    adapter: _Adapter, stall: Callable[[], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    from xagent.core.tools.core.RAG_tools.management import collections as mc

    _save_collection()
    _seed_core({"d1": ["c1", "c2"]})
    _write([_src("c1"), _src("c2")])

    def _delete(texts: List[str]) -> None:
        result = mc.delete_document("kb", "d1", 1, is_admin=True)
        assert result.status == "success", result.message

    adapter.before_encode = _delete
    real = mod.write_vectors_to_db

    def _write_then_stall(*args: Any, **kwargs: Any) -> Any:
        response = real(*args, **kwargs)
        stall()
        return response

    monkeypatch.setattr(mod, "write_vectors_to_db", _write_then_stall)

    report = _run()

    assert list(report.failures) == ["d1"]
    assert "lease" in report.failures["d1"]
    assert (report.written, report.removed) == (2, 0)
    assert _target_ids() == ["c1", "c2"]


def test_the_target_table_is_compacted_in_the_lease_every_few_documents_and_at_the_end(
    adapter: _Adapter, compactions: List[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "COMPACT_EVERY_DOCUMENTS", 2, raising=False)
    _save_collection()
    _seed_core({f"d{index}": [f"c{index}"] for index in range(5)})
    _write([_src(f"c{index}", f"d{index}") for index in range(5)])
    compactions.clear()

    assert _run().written == 5
    assert compactions == [(_target_tables(), True)] * 3


def test_a_lease_lost_before_a_documents_compaction_skips_only_that_one(
    adapter: _Adapter,
    stall: Callable[[], None],
    compactions: List[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "COMPACT_EVERY_DOCUMENTS", 1, raising=False)
    _save_collection()
    _seed_core({"d1": ["c1"]})
    _write([_src("c1")])
    compactions.clear()
    real = mod.write_vectors_to_db

    def _write_then_stall(*args: Any, **kwargs: Any) -> Any:
        response = real(*args, **kwargs)
        stall()
        return response

    monkeypatch.setattr(mod, "write_vectors_to_db", _write_then_stall)

    report = _run()

    assert (list(report.failures), report.written) == (["d1"], 1)
    assert "lease" in report.failures["d1"]
    assert compactions == [(_target_tables(), True)]


# --- failures ---


def test_blank_source_metadata_is_written_as_none(adapter: _Adapter) -> None:
    _save_collection()
    _seed_core({"d1": ["c1"]})
    _write([_src("c1")])
    scope.connection().open_table(scope.source_table(SOURCE)).update(
        where="chunk_id = 'c1'", values={"metadata": " "}
    )

    report = _run()

    assert (report.failures, report.written, _target_ids()) == ({}, 1, ["c1"])


def test_a_recheck_that_cannot_delete_keeps_the_write_error(
    adapter: _Adapter, stall: Callable[[], None]
) -> None:
    _save_collection()
    chunk_ids = [f"c{index:02d}" for index in range(12)]
    _seed_core({"d1": chunk_ids})
    _write([_src(chunk_id) for chunk_id in chunk_ids])

    def _second_batch_fails_with_the_lease_gone(texts: List[str]) -> None:
        if len(adapter.calls) == 2:
            scope.connection().open_table("chunks").delete("doc_id = 'd1'")
            stall()
            raise RuntimeError("provider down")

    adapter.before_encode = _second_batch_fails_with_the_lease_gone

    report = _run()

    assert report.failures == {"d1": "RuntimeError: provider down"}
    assert (report.written, report.removed, len(_target_ids())) == (10, 0, 10)


@pytest.mark.parametrize(
    "vectors",
    [
        [[0.25, 0.75, 0.5]],
        [[0.0, 0.0]],
        [[float("nan"), 0.5]],
        [],
    ],
    ids=["wrong-dimension", "zero", "not-finite", "too-few"],
)
def test_an_unusable_provider_answer_fails_the_document(
    adapter: _Adapter, vectors: List[List[float]]
) -> None:
    _save_collection()
    _seed_core({"d1": ["c1"]})
    _write([_src("c1")])
    adapter.vectors = lambda texts: vectors

    report = _run()

    assert list(report.failures) == ["d1"]
    assert (report.written, _target_ids()) == (0, [])


def test_metadata_that_is_not_an_object_fails_only_its_document(
    adapter: _Adapter,
) -> None:
    _save_collection()
    _seed_core({"d1": ["c1"], "d2": ["c2"]})
    _write([_src("c1"), _src("c2", "d2")])
    scope.connection().open_table(scope.source_table(SOURCE)).update(
        where="chunk_id = 'c1'", values={"metadata": "[1]"}
    )

    report = _run()

    assert list(report.failures) == ["d1"]
    assert (report.written, _target_ids()) == (1, ["c2"])
    assert adapter.calls == [["text c2"]]


def test_consecutive_document_failures_abort_the_run(adapter: _Adapter) -> None:
    docs = {f"d{index}": [f"c{index}"] for index in range(6)}
    _save_collection()
    _save_collection("kb2")
    _seed_core(docs)
    _seed_core({"e1": ["x1"]}, collection="kb2")
    _write([_src(f"c{index}", f"d{index}") for index in range(6)])
    _write([_src("x1", "e1")], collection="kb2")

    def _fail(texts: List[str]) -> None:
        raise RuntimeError("bad credential")

    adapter.before_encode = _fail

    reports = mod.warm_up(TARGET)

    assert [report.name for report in reports] == ["kb"]
    assert reports[0].aborted
    assert len(reports[0].failures) == mod.MAX_CONSECUTIVE_DOCUMENT_FAILURES
    assert len(adapter.calls) == mod.MAX_CONSECUTIVE_DOCUMENT_FAILURES


def test_failures_between_successes_do_not_abort(adapter: _Adapter) -> None:
    _save_collection()
    _seed_core({f"d{index}": [f"c{index}"] for index in range(10)})
    _write([_src(f"c{index}", f"d{index}") for index in range(10)])

    def _even_documents_fail(texts: List[str]) -> None:
        if int(texts[0].removeprefix("text c")) % 2 == 0:
            raise RuntimeError("refused")

    adapter.before_encode = _even_documents_fail

    report = _run()

    assert (len(report.failures), report.written, report.aborted) == (5, 5, False)


def test_one_collection_failing_does_not_stop_the_others(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    _save_collection()
    _save_collection("kb2")
    _seed_core({"e1": ["x1"]}, collection="kb2")
    _write([_src("x1", "e1")], collection="kb2")
    real = mod.precompute_eligibility

    def _kb_fails(name: str, **kwargs: Any) -> Any:
        if name == "kb":
            raise OSError("read failed")
        return real(name, **kwargs)

    monkeypatch.setattr(mod, "precompute_eligibility", _kb_fails)

    reports = mod.warm_up(TARGET)

    assert [(r.name, r.error, r.written) for r in reports] == [
        ("kb", "OSError: read failed", 0),
        ("kb2", None, 1),
    ]


def test_a_collection_gone_before_it_is_read_is_skipped(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    _save_collection()

    def _gone(name: str, **kwargs: Any) -> Any:
        raise scope.MissingCollectionError(name)

    monkeypatch.setattr(mod, "precompute_eligibility", _gone)

    report = _run()

    assert (report.skip_reason, report.failed) == (scope.SKIP_NO_METADATA, False)


# --- indexes ---


def test_indexes_are_built_only_where_the_target_holds_rows(
    adapter: _Adapter, index_calls: SimpleNamespace
) -> None:
    _save_collection()
    _save_collection("kb2")
    _seed_core({"d1": ["c1"]})
    _write([_src("c1")])
    _write([_src("c0")], model=TARGET, vector=list(VECTOR))
    _write([_src("x1", "e1")], collection="kb2")

    reports = mod.warm_up(TARGET, collections=["kb2"])

    assert (reports[0].skipped, reports[0].failed) == (
        {mod.SKIP_NO_CORE_DOCUMENT: 1},
        False,
    )
    assert index_calls.calls == []

    assert _run().written == 1
    assert index_calls.calls == [("new_model", False)]


def test_no_target_table_builds_no_index(
    adapter: _Adapter, index_calls: SimpleNamespace
) -> None:
    _save_collection()
    _write([_src("c1")])

    report = _run()

    assert (report.failed, index_calls.calls) == (False, [])


@pytest.mark.parametrize("status", ["failed", "index_corrupted"])
def test_a_failed_index_build_fails_the_collection(
    adapter: _Adapter, index_calls: SimpleNamespace, status: str
) -> None:
    _save_collection()
    _seed_core({"d1": ["c1"]})
    _write([_src("c1")])
    index_calls.status = status

    report = _run()

    assert report.index_error and status in report.index_error
    assert mod.main(["--model", TARGET]) == 1


# --- target, exit codes, pacing ---


@pytest.mark.parametrize(
    "case", ["unknown", "alias", "no-dimension", "adapter-of-another-model"]
)
def test_a_target_the_hub_does_not_confirm_never_starts(
    adapter: _Adapter,
    monkeypatch: pytest.MonkeyPatch,
    hub: _Hub,
    case: str,
) -> None:
    from xagent.core.tools.core.RAG_tools.utils import model_resolver

    model = TARGET
    if case == "unknown":
        model = "no-such-model"
    elif case == "alias":
        hub.models[TARGET] = _config(TARGET, model_name="alias-name")
        model = "alias-name"
    elif case == "no-dimension":
        hub.models[TARGET] = _config(TARGET, dimension=None)
    else:
        monkeypatch.setattr(
            model_resolver,
            "resolve_embedding_adapter",
            lambda model_id, **kwargs: (_config("other-model"), adapter),
        )
    opened: List[str] = []
    monkeypatch.setattr(mod, "connection", lambda: opened.append("lancedb"))

    assert mod.main(["--model", model]) == 2
    assert (opened, adapter.calls) == ([], [])


def test_an_unknown_collection_never_starts(adapter: _Adapter) -> None:
    _save_collection()
    _seed_core({"d1": ["c1"]})
    _write([_src("c1")])

    assert mod.main(["--model", TARGET, "--collection", "nope"]) == 2
    assert adapter.calls == []


@pytest.mark.parametrize("rate", ["-1", "nan", "inf"])
def test_an_invalid_rate_limit_never_starts(adapter: _Adapter, rate: str) -> None:
    assert mod.main(["--model", TARGET, "--rate-limit", rate]) == 2
    assert adapter.builds == 0


def test_exit_codes_follow_the_outcome(adapter: _Adapter) -> None:
    assert mod.main(["--model", TARGET]) == 1

    _save_collection()
    _save_collection("kb2", model_id=None, dimension=None)
    _seed_core({"d1": ["c1"]})
    _write([_src("c1")])
    assert mod.main(["--model", TARGET]) == 0
    assert mod.main(["--model", TARGET, "--dry-run"]) == 0

    adapter.vectors = lambda texts: []
    _seed_core({"d2": ["c2"]})
    _write([_src("c2", "d2")])
    assert mod.main(["--model", TARGET]) == 1


def test_the_run_ends_with_a_summary_of_stopped_failed_and_skipped(
    adapter: _Adapter, caplog: pytest.LogCaptureFixture
) -> None:
    _save_collection()
    _save_collection("kb2")
    _save_collection("kb3", model_id=None, dimension=None)
    _seed_core({"d1": ["c1"]})
    _seed_core({"e1": ["x1"]}, collection="kb2")
    _write([_src("c1")])
    _write([_src("x1", "e1")], collection="kb2")
    _freeze()

    def _fail(texts: List[str]) -> None:
        raise RuntimeError("refused")

    adapter.before_encode = _fail

    with caplog.at_level(logging.INFO, logger=mod.__name__):
        assert mod.main(["--model", TARGET]) == 1

    [summary] = [r.getMessage() for r in caplog.records if "stopped:" in r.getMessage()]
    assert "stopped: kb (frozen)" in summary
    assert "failed: kb2" in summary
    assert "skipped: 1" in summary


def test_help_states_what_is_written_and_what_the_exit_codes_mean(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        mod.main(["--help"])
    text = " ".join(capsys.readouterr().out.split())

    for phrase in (
        "Not a completeness claim",
        "no-core-chunk",
        "'frozen'",
        "'binding-changed'",
        "a stop is not a failure",
        "Five failed documents in a row in one collection abort the run",
        "0 does not mean every collection was warmed up",
        "2 the run never started",
        "writes nothing to LanceDB",
        "not retried",
    ):
        assert phrase in text, phrase


@pytest.mark.parametrize("dry_run", [False, True])
def test_the_rate_limit_paces_every_call_and_unlimited_warns(
    adapter: _Adapter,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    dry_run: bool,
) -> None:
    clock = SimpleNamespace(now=0.0, slept=[])

    def _sleep(seconds: float) -> None:
        clock.slept.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(
        mod, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=_sleep)
    )
    _save_collection()
    _seed_core({"d1": ["c1", "c2", "c3"]})
    _write([_src("c1"), _src("c2"), _src("c3", user_id=8)])

    with caplog.at_level(logging.WARNING, logger=mod.__name__):
        _run(dry_run=dry_run, rate_limit=2.0)
        _run(dry_run=dry_run)

    if dry_run:
        assert (clock.slept, adapter.calls) == ([], [])
        assert "No --rate-limit" not in caplog.text
    else:
        assert [len(texts) for texts in adapter.calls] == [2, 1]
        assert clock.slept == [pytest.approx(1.0)]
        assert "No --rate-limit" in caplog.text


def _handle() -> Any:
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

    return LanceDBCollectionHandle(
        KBCollectionContext(
            collection="kb",
            user_scope=KBUserScope(user_id=None, is_admin=True),
            access_mode=KBAccessMode.WRITE,
            allow_create=True,
            hide_missing=True,
            metadata_store=get_metadata_store(),
            vector_index_store=get_vector_index_store(),
            ingestion_status_store=get_ingestion_status_store(),
            main_pointer_store=get_main_pointer_store(),
            backend=KBStorageBackend.LANCEDB,
            capabilities=KBBackendCapabilities.lancedb(),
            collection_info=None,
        )
    )
