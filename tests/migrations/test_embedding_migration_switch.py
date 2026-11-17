"""Switch: bind a collection to the target only when the target equals the source."""

from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import pyarrow as pa
import pytest

from tests.migrations.embedding_migration_shared import (
    SOURCE,
    TARGET,
    VECTOR,
    VENDOR_TARGET,
    _Adapter,
    _retype_owner,
    _rows,
    _save_collection,
    _seed_core,
    _seed_target,
    _src,
    _target,
    _versions,
    _write,
)
from xagent.core.tools.core.RAG_tools.core.exceptions import CollectionMigratingError
from xagent.core.tools.core.RAG_tools.management import embedding_switch
from xagent.core.tools.core.RAG_tools.management import migration_state as ms
from xagent.core.tools.core.RAG_tools.management import write_leases
from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
    LanceDBVectorIndexStore,
)
from xagent.migrations.embedding import scope
from xagent.migrations.embedding import switch as mod

pytestmark = pytest.mark.usefixtures("hub", "index_calls")

OTHER_RUN = f"{mod.HOLDER}:{'0' * 12}"
REAL_CREATE_INDEX = LanceDBVectorIndexStore.create_index


def _seed(chunks: Dict[str, List[str]], *, name: str = "kb") -> None:
    _save_collection(name)
    _seed_core(chunks, collection=name)
    _write([_src(c, d) for d, ids in chunks.items() for c in ids], collection=name)


def _held(name: str = "kb", target: str = TARGET) -> List[Tuple[str, str, str]]:
    return sorted(
        (r["doc_id"], r["chunk_id"], r["text"])
        for r in _rows(scope.source_table(target))
        if r["collection"] == name
    )


def _source(name: str = "kb") -> List[Tuple[str, str, str]]:
    return sorted(
        (r["doc_id"], r["chunk_id"], r["text"])
        for r in _rows(scope.source_table(SOURCE))
        if r["collection"] == name
    )


def _binding(name: str = "kb") -> Optional[str]:
    return scope.collection_info(name).embedding_model_id


def _state(name: str = "kb") -> ms.CollectionMigrationState:
    return ms.read_migration_state(name)


def _assert_switched(name: str = "kb", target: str = TARGET) -> None:
    assert _held(name, target) == _source(name)
    assert _binding(name) == target
    assert (_state(name).state, _state(name).owner) == (ms.IDLE, None)


def _assert_untouched(name: str = "kb") -> None:
    assert _binding(name) == SOURCE
    assert _state(name).state == ms.IDLE


def _never_frozen(name: str = "kb") -> None:
    conn = scope.connection()
    if ms.MIGRATION_TABLE in scope.table_names(conn):
        rows = (
            conn.open_table(ms.MIGRATION_TABLE)
            .search()
            .where(f"name = '{name}'")
            .to_arrow()
            .to_pylist()
        )
        assert rows == []


def _ingest(chunk_id: str, model: str) -> None:
    """The ingestion write path, with its guard against a rebound collection."""
    from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData
    from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
        write_vectors_to_db,
    )

    write_vectors_to_db(
        "kb",
        [
            ChunkEmbeddingData(
                doc_id="d2",
                chunk_id=chunk_id,
                parse_hash="p",
                model=model,
                vector=list(VECTOR),
                text=f"text {chunk_id}",
                chunk_hash="h",
            )
        ],
        create_index=False,
        user_id=7,
    )


def _run(target: str = TARGET, **kwargs: Any) -> mod.SwitchReport:
    kwargs.setdefault("drain_timeout", 0.5)
    [report] = mod.switch(target, collections=["kb"], **kwargs)
    return report


def _killed(monkeypatch: pytest.MonkeyPatch) -> Callable[[], None]:
    """Make the next run leave its window as a killed process would; undo after."""
    monkeypatch.setattr(mod, "release_migration_window", lambda window: None)
    return lambda: monkeypatch.setattr(
        mod, "release_migration_window", ms.release_migration_window
    )


def _source_scans(monkeypatch: pytest.MonkeyPatch) -> List[bool]:
    """Every pass over the source table, as whether ``kb`` was frozen during it."""
    from xagent.migrations.embedding import embed

    scans: List[bool] = []

    def _counted(conn: Any, table: str, *args: Any, **kwargs: Any) -> Any:
        if table == scope.source_table(SOURCE):
            scans.append(ms.read_migration_state("kb").state == ms.FROZEN)
        return scope.iter_rows(conn, table, *args, **kwargs)

    monkeypatch.setattr(mod, "iter_rows", _counted)
    monkeypatch.setattr(embed, "iter_rows", _counted)
    return scans


# --- switched ---


@pytest.mark.parametrize("target", [TARGET, VENDOR_TARGET])
def test_a_partial_warm_up_is_completed_and_the_collection_switched(
    adapter: _Adapter, hub: Any, target: str
) -> None:
    _seed_target(hub, target, writer=True)
    _seed({"d1": ["c1", "c2", "c3"]})
    _write([_src("c1")], model=target, vector=list(VECTOR))
    _write([_src("gone")], model=target, vector=list(VECTOR))

    report = _run(target)

    assert (report.outcome, report.error, report.refused) == ("switched", None, None)
    assert (report.removed, report.written) == (1, 2)
    assert [sorted(texts) for texts in adapter.calls] == [["text c2", "text c3"]]
    _assert_switched(target=target)


@pytest.mark.parametrize("target", [TARGET, VENDOR_TARGET])
def test_after_the_switch_search_reads_the_target_and_ingestion_writes_it(
    adapter: _Adapter, hub: Any, target: str
) -> None:
    from xagent.core.tools.core.RAG_tools.pipelines.document_search import (
        run_document_search,
    )

    _seed_target(hub, target, writer=True)
    _seed({"d1": ["c1", "c2"], "d2": ["c3"]})

    def _search(kind: str = "dense", query: str = "text c1") -> List[str]:
        result = run_document_search(
            "kb",
            query,
            config={"search_type": kind, "embedding_model_id": SOURCE},
            is_admin=True,
        )
        assert result.status == "success", result.message
        return sorted(item.chunk_id for item in result.results)

    before = _search()
    assert before == ["c1", "c2", "c3"]
    assert _run(target).outcome == "switched"
    conn = scope.connection()
    conn.open_table(scope.source_table(SOURCE)).delete("collection = 'kb'")

    assert _search() == before
    # The recorder stands in for create_index; this is the FTS index it would build.
    conn.open_table(scope.source_table(target)).create_fts_index(
        "text", replace=True, with_position=True
    )
    assert _search("sparse", "c3") == ["c3"]
    _ingest("c4", target)
    assert ("d2", "c4", "text c4") in _held(target=target)
    with pytest.raises(CollectionMigratingError):
        _ingest("c5", SOURCE)


def test_a_target_row_of_other_text_is_embedded_again(adapter: _Adapter) -> None:
    _seed({"d1": ["c1", "c2"]})
    _write([_src("c1", text="old text"), _src("c2")], model=TARGET, vector=list(VECTOR))

    report = _run()

    assert (report.outcome, report.removed, report.written) == ("switched", 1, 1)
    assert adapter.calls == [["text c1"]]
    _assert_switched()


def test_duplicate_target_rows_of_one_chunk_are_replaced_by_one(
    adapter: _Adapter,
) -> None:
    _seed({"d1": ["c1"]})
    _write([_src("c1")], model=TARGET, vector=list(VECTOR))
    table = scope.connection().open_table(scope.source_table(TARGET))
    table.add(table.to_arrow())
    assert len(_held()) == 2

    report = _run()

    assert (report.outcome, report.stale, report.removed) == ("switched", 2, 2)
    assert "1 rows to embed, 2 to delete; wrote 1; deleted 2" in mod.describe(report)
    _assert_switched()


def test_a_delete_that_raises_keeps_the_rows_already_deleted(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed({"d1": ["c1"], "d2": ["c2"]})
    rows = [_src("c1", text="old"), _src("c2", "d2", text="old")]
    _write(rows, model=TARGET, vector=list(VECTOR))
    table_type = type(scope.connection().open_table(scope.source_table(TARGET)))
    real_delete = table_type.delete

    def _second_document_fails(self: Any, where: str, *args: Any, **kw: Any) -> Any:
        if "doc_id = 'd2'" in where:
            raise OSError("delete failed")
        return real_delete(self, where, *args, **kw)

    monkeypatch.setattr(table_type, "delete", _second_document_fails)
    report = _run()

    assert (report.outcome, report.removed, report.error) == (
        None,
        1,
        "OSError: delete failed",
    )
    assert "deleted 1" in mod.describe(report)
    assert (_binding(), _state().state) == (SOURCE, ms.IDLE)


def test_the_comparison_keeps_a_digest_and_a_flag_per_row_not_the_vector(
    adapter: _Adapter,
) -> None:
    from xagent.migrations.embedding import embed

    _seed({"d1": ["c1"]})
    _write([_src("c1")], model=TARGET, vector=list(VECTOR))
    target, conn, _ = embed.start_run(
        TARGET, collections=["kb"], dry_run=True, rate_limit=0
    )
    report = mod.SwitchReport(name="kb", source_model_id=SOURCE)

    source, held = mod.read_both(report, target, conn)

    [[(digest, usable)]] = held.values()
    assert (len(digest), usable) == (32, True)
    assert (source, held) == (
        {("d1", "c1"): [(digest, True)]},
        {("d1", "c1"): [(digest, True)]},
    )


def test_a_target_row_with_an_unusable_vector_is_embedded_again(
    adapter: _Adapter,
) -> None:
    _seed({"d1": ["c1"]})
    _write([_src("c1")], model=TARGET, vector=[0.0, 0.0])

    assert _run().outcome == "switched"
    assert [r["vector"] for r in _target()] == [pytest.approx(VECTOR)]
    _assert_switched()


def _warm_up() -> None:
    from xagent.migrations.embedding import embed

    [warmed] = embed.warm_up(TARGET, collections=["kb"])
    assert (warmed.failures, warmed.error, warmed.written) == ({}, None, 1)


@pytest.mark.parametrize(
    "metadata", [None, "", "{}", '{"b": [1, {"y": 2, "x": 1}], "a": "z"}']
)
def test_rows_the_warm_up_wrote_leave_nothing_to_fill(
    adapter: _Adapter, metadata: Optional[str]
) -> None:
    _seed({"d1": ["c1"]})
    if metadata is not None:
        table = scope.connection().open_table(scope.source_table(SOURCE))
        table.update(where="chunk_id = 'c1'", values={"metadata": metadata})
    _warm_up()
    adapter.calls.clear()

    report = _run()

    assert (report.outcome, report.written, report.removed) == ("switched", 0, 0)
    assert adapter.calls == []


def test_a_source_owner_stored_as_text_leaves_nothing_to_fill(
    adapter: _Adapter,
) -> None:
    _seed({"d1": ["c1"]})
    conn = scope.connection()
    name = scope.source_table(SOURCE)
    table = conn.open_table(name).to_arrow()
    owners = pa.array([str(v) for v in table.column("user_id").to_pylist()])
    index = table.schema.get_field_index("user_id")
    conn.drop_table(name)
    conn.create_table(name, table.set_column(index, "user_id", owners))
    _warm_up()
    adapter.calls.clear()

    report = _run()

    assert (report.outcome, report.written, report.removed) == ("switched", 0, 0)
    assert adapter.calls == []


@pytest.mark.parametrize(
    "field, value",
    [
        ("metadata", '{"page": 9}'),
        ("chunk_hash", "other"),
        ("parse_hash", "other"),
        ("user_id", 8),
    ],
)
def test_a_target_row_differing_in_a_copied_field_is_written_again(
    adapter: _Adapter, field: str, value: Any
) -> None:
    _seed({"d1": ["c1"]})
    _warm_up()
    table = scope.connection().open_table(scope.source_table(TARGET))
    table.update(where="chunk_id = 'c1'", values={field: value})

    report = _run()

    assert (report.outcome, report.removed, report.written) == ("switched", 1, 1)
    [source] = _rows(scope.source_table(SOURCE))
    assert [row[field] for row in _target()] == [source[field]]


def test_an_already_switched_collection_takes_no_window(adapter: _Adapter) -> None:
    _seed({"d1": ["c1"]})
    assert _run().outcome == "switched"
    before = _versions()
    adapter.calls.clear()

    report = _run()

    assert (report.outcome, report.ok) == ("already-switched", True)
    assert (_versions(), adapter.calls) == (before, [])


def test_an_empty_bound_collection_is_switched(adapter: _Adapter) -> None:
    _save_collection()

    assert _run().outcome == "switched"
    assert (_binding(), _state().state) == (TARGET, ms.IDLE)


# --- refused before the freeze ---


@pytest.mark.parametrize("case", ["no-core-chunk", "no-core-document", "no-owner"])
def test_an_unusable_source_row_refuses_without_freezing(
    adapter: _Adapter, case: str
) -> None:
    _seed({"d1": ["c1"]})
    if case == "no-owner":
        _write([_src("c2", user_id=None)])
        _seed_core({"d1": ["c1", "c2"]})
    else:
        _write([_src("c2", "d1" if case == "no-core-chunk" else "d9")])
    before = _versions()

    report = _run()

    assert report.refused == "unusable-source-rows"
    assert report.skipped == {case: 1}
    assert (_versions(), adapter.calls) == (before, [])
    _never_frozen()
    _assert_untouched()


@pytest.mark.parametrize(
    "field, raw",
    [
        ("metadata", "null"),
        ("metadata", "[1, 2]"),
        ("metadata", "not json"),
        ("chunk_hash", None),
        ("parse_hash", None),
    ],
)
def test_a_source_row_the_write_would_reject_refuses_without_freezing(
    adapter: _Adapter, field: str, raw: Optional[str]
) -> None:
    _seed({"d1": ["c1"]})
    table = scope.connection().open_table(scope.source_table(SOURCE))
    if raw is None:
        table.update(where="chunk_id = 'c1'", values_sql={field: "NULL"})
    else:
        table.update(where="chunk_id = 'c1'", values={field: raw})
    before = _versions()

    report = _run()

    assert report.refused == "unusable-source-rows"
    assert report.skipped == {"unwritable": 1}
    assert (_versions(), adapter.calls) == (before, [])
    _never_frozen()
    _assert_untouched()


def test_more_than_max_fill_refuses_without_freezing(adapter: _Adapter) -> None:
    _seed({"d1": ["c1", "c2", "c3"]})
    _write([_src("gone")], model=TARGET, vector=list(VECTOR))
    before = _versions()

    report = _run(max_fill=3)

    assert (report.refused, report.pending, report.stale) == ("over-max-fill", 3, 1)
    assert (_versions(), adapter.calls) == (before, [])
    _never_frozen()
    assert _run(max_fill=4).outcome == "switched"


def _duplicate_source_row() -> None:
    table = scope.connection().open_table(scope.source_table(SOURCE))
    table.add(table.search().where("chunk_id = 'c1'").to_arrow())


def test_a_duplicated_source_row_refuses_without_freezing(adapter: _Adapter) -> None:
    _seed({"d1": ["c1"]})
    _duplicate_source_row()

    report = _run()

    assert (report.refused, adapter.calls) == ("duplicate-source-rows", [])
    _never_frozen()


def test_a_source_row_duplicated_during_the_drain_is_refused(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed({"d1": ["c1"]})
    real_wait = mod.wait_for_no_leases

    def _duplicated_meanwhile(name: str, **kwargs: Any) -> Any:
        _duplicate_source_row()
        return real_wait(name, **kwargs)

    monkeypatch.setattr(mod, "wait_for_no_leases", _duplicated_meanwhile)

    report = _run()

    assert (report.refused, adapter.calls, _held()) == ("duplicate-source-rows", [], [])
    _assert_untouched()


def test_a_collection_frozen_by_another_run_is_refused(adapter: _Adapter) -> None:
    _seed({"d1": ["c1"]})
    window = ms.claim_migration_window("kb", holder="other")

    report = _run()

    assert report.refused == "frozen"
    assert (_state().owner, _binding(), adapter.calls) == (window.owner, SOURCE, [])


def test_a_frozen_collection_already_on_the_target_reports_its_binding(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed({"d1": ["c1"]})
    restore = _killed(monkeypatch)
    _run()
    restore()

    report = _run()

    line = mod.describe(report)
    assert report.refused == "frozen" and f"bound to {TARGET}" in line
    assert "rerun with --collection kb --take-over" in line
    assert "source model" not in line
    assert (_binding(), _state().state) == (TARGET, ms.FROZEN)


def test_a_collection_another_run_switched_meanwhile_reports_its_binding(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed({"d1": ["c1"]})
    original = mod.open_window
    inner: List[mod.SwitchReport] = []

    def _other_run_first(report: Any) -> Any:
        monkeypatch.setattr(mod, "open_window", original)
        inner.extend(mod.switch(TARGET, collections=["kb"], drain_timeout=0.5))
        return original(report)

    monkeypatch.setattr(mod, "open_window", _other_run_first)

    outer = _run()

    assert (inner[0].outcome, outer.refused) == ("switched", "scope-changed")
    assert mod.describe(outer).endswith(f"bound to {TARGET}")
    _assert_switched()


def test_over_max_fill_made_only_of_rows_to_delete_says_so_on_its_line(
    adapter: _Adapter, caplog: pytest.LogCaptureFixture
) -> None:
    _seed({"d1": ["c1"]})
    extra = [_src("x1"), _src("x2"), _src("x3")]
    _write([_src("c1")] + extra, model=TARGET, vector=list(VECTOR))

    with caplog.at_level(logging.INFO, logger=mod.__name__):
        report = _run(max_fill=2)

    assert (report.refused, report.pending, report.stale) == ("over-max-fill", 0, 3)
    assert (
        "refused (over-max-fill: 0 to embed + 3 to delete > 2; raise --max-fill)"
        in mod.describe(report)
    )
    assert "Switch done: refused (over-max-fill): 1 (kb)" in caplog.text
    _never_frozen()


# --- inside the window ---


def test_a_live_lease_is_waited_for_then_refuses_and_releases(
    adapter: _Adapter,
) -> None:
    _seed({"d1": ["c1"]})
    lease = write_leases.acquire_lease("kb", "test")
    try:
        report = _run(drain_timeout=0.3)
    finally:
        write_leases.release_leases([lease])

    assert report.refused == "writers-still-active"
    assert (_held(), adapter.calls) == ([], [])
    _assert_untouched()


def test_the_switch_waits_for_a_writer_to_finish(adapter: _Adapter) -> None:
    _seed({"d1": ["c1"]})
    lease = write_leases.acquire_lease("kb", "test")
    threading.Timer(0.5, write_leases.release_leases, args=([lease],)).start()
    started = time.monotonic()

    report = _run(drain_timeout=10)

    assert report.outcome == "switched"
    assert time.monotonic() - started >= 0.5
    _assert_switched()


def test_a_write_admitted_before_the_freeze_landing_in_the_drain_is_switched(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed({"d1": ["c1"]})
    lease = write_leases.acquire_lease("kb", "ingestion")

    def _writer_finishes(name: str, **kwargs: Any) -> Any:
        assert _state().state == ms.FROZEN
        _seed_core({"d1": ["c1", "late"]})
        _write([_src("late")])
        write_leases.release_leases([lease])
        return write_leases.wait_for_no_leases(name, **kwargs)

    monkeypatch.setattr(mod, "wait_for_no_leases", _writer_finishes)

    assert _run().outcome == "switched"
    assert [c for _, c, _ in _held()] == ["c1", "late"]
    _assert_switched()


def test_a_missed_flip_is_retried(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xagent.core.tools.core.RAG_tools.management.collection_manager import (
        mark_collection_accessed_sync,
    )

    _seed({"d1": ["c1"]})
    real = embedding_switch.write_collection_binding_sync
    calls: List[str] = []

    def _searched_meanwhile(name: str, *args: Any, **kwargs: Any) -> bool:
        calls.append(name)
        if len(calls) <= 2:
            mark_collection_accessed_sync(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(
        embedding_switch, "write_collection_binding_sync", _searched_meanwhile
    )

    assert _run().outcome == "switched"
    assert len(calls) == 3
    _assert_switched()


def test_exhausted_flip_retries_release_and_leave_the_binding(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xagent.core.tools.core.RAG_tools.management.collection_manager import (
        mark_collection_accessed_sync,
    )

    _seed({"d1": ["c1"]})
    real = embedding_switch.write_collection_binding_sync
    calls: List[str] = []

    def _always_searched(name: str, *args: Any, **kwargs: Any) -> bool:
        calls.append(name)
        mark_collection_accessed_sync(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(
        embedding_switch, "write_collection_binding_sync", _always_searched
    )

    report = _run()

    assert report.outcome is None and "changed while switching" in str(report.error)
    assert len(calls) == mod.FLIP_ATTEMPTS
    _assert_untouched()


def test_a_flip_that_finds_another_binding_is_not_retried(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed({"d1": ["c1"]})
    real = embedding_switch.write_collection_binding_sync
    calls: List[str] = []

    def _rebound_meanwhile(name: str, *args: Any, **kwargs: Any) -> bool:
        calls.append(name)
        if len(calls) == 1:
            real(name, "third-model", 2)
            return False
        return real(name, *args, **kwargs)

    monkeypatch.setattr(
        embedding_switch, "write_collection_binding_sync", _rebound_meanwhile
    )

    report = _run()

    assert (report.outcome, len(calls), _binding()) == (None, 1, "third-model")
    assert _state().state == ms.IDLE


def test_a_target_that_still_differs_after_the_fill_is_not_switched(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed({"d1": ["c1", "c2"]})
    monkeypatch.setattr(mod, "write_rows", lambda *args, **kwargs: None)

    report = _run()

    assert report.outcome is None and "differs from source" in str(report.error)
    _assert_untouched()


def test_a_binding_changed_after_the_claim_is_refused(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xagent.core.tools.core.RAG_tools.management.collection_manager import (
        write_collection_binding_sync,
    )

    _seed({"d1": ["c1"]})
    real_claim = mod.claim_migration_window

    def _rebound_after_the_claim(name: str, **kwargs: Any) -> Any:
        window = real_claim(name, **kwargs)
        write_collection_binding_sync(name, "third-model", 2)
        return window

    monkeypatch.setattr(mod, "claim_migration_window", _rebound_after_the_claim)

    report = _run()

    assert (report.refused, report.outcome) == ("scope-changed", None)
    assert mod.describe(report).endswith("bound to third-model")
    assert (_binding(), _state().state, _held()) == ("third-model", ms.IDLE, [])


def test_a_target_row_the_source_lacks_after_the_fill_is_not_switched(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed({"d1": ["c1"]})
    real_write = mod.write_rows

    def _write_and_stray(*args: Any, **kwargs: Any) -> None:
        real_write(*args, **kwargs)
        _write([_src("stray")], model=TARGET, vector=list(VECTOR))

    monkeypatch.setattr(mod, "write_rows", _write_and_stray)

    report = _run()

    assert report.outcome is None and "differs from source at d1/stray" in str(
        report.error
    )
    _assert_untouched()


def test_a_run_that_lost_its_window_before_the_index_call_makes_none(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch, index_calls: SimpleNamespace
) -> None:
    _seed({"d1": ["c1"]})
    real_write = mod.write_rows

    def _write_then_lose(*args: Any, **kwargs: Any) -> None:
        real_write(*args, **kwargs)
        ms.take_over_migration_window("kb", _state(), holder="other")

    monkeypatch.setattr(mod, "write_rows", _write_then_lose)

    report = _run()

    assert (report.outcome, index_calls.calls) == (None, [])
    assert "not frozen by this run" in str(report.error)


@pytest.mark.parametrize("status", ["failed", "index_corrupted"])
def test_a_failed_index_build_is_not_switched(
    adapter: _Adapter, index_calls: SimpleNamespace, status: str
) -> None:
    _seed({"d1": ["c1"]})
    index_calls.status = status

    report = _run()

    assert report.outcome is None and status in str(report.error)
    _assert_untouched()


def test_a_provider_failure_releases_and_a_rerun_completes(
    adapter: _Adapter,
) -> None:
    _seed({"d1": ["c1"], "d2": ["c2"]})

    def _second_call_fails(texts: List[str]) -> None:
        if len(adapter.calls) == 2:
            raise RuntimeError("provider down")

    adapter.before_encode = _second_call_fails

    report = _run()

    assert report.outcome is None and "provider down" in str(report.error)
    assert report.written == 1
    _assert_untouched()
    adapter.before_encode = None
    assert _run().outcome == "switched"
    _assert_switched()


@pytest.mark.parametrize("moment", ["after-drain", "mid-fill", "before-flip"])
def test_a_run_that_lost_its_window_never_touches_the_target_again(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch, moment: str
) -> None:
    _seed({"d1": ["c1", "c2"], "d2": ["c3"]})
    _write([_src("gone")], model=TARGET, vector=list(VECTOR))
    taken: Dict[str, Any] = {}

    def _take_over() -> None:
        if not taken:
            taken["window"] = ms.take_over_migration_window(
                "kb", _state(), holder="other"
            )
            taken["held"] = _held()

    if moment == "after-drain":
        real_wait = mod.wait_for_no_leases

        def _wait(name: str, **kwargs: Any) -> Any:
            left = real_wait(name, **kwargs)
            _take_over()
            return left

        monkeypatch.setattr(mod, "wait_for_no_leases", _wait)
    elif moment == "mid-fill":
        adapter.before_encode = lambda texts: (
            _take_over() if len(adapter.calls) == 2 else None
        )
    else:
        real_compare = mod.compare

        def _compare(*args: Any) -> Any:
            result = real_compare(*args)
            _take_over()
            return result

        monkeypatch.setattr(mod, "compare", _compare)

    report = _run()
    line = mod.describe(report)

    assert report.outcome is None and f"failed: {report.error}" in line
    assert f"window lost: {report.window_lost}" in line and report.window_lost
    assert _held() == taken["held"]
    assert _binding() == SOURCE
    assert (_state().state, _state().owner) == (ms.FROZEN, taken["window"].owner)


def test_a_window_lost_after_the_binding_write_is_summarised_apart(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _seed({"d1": ["c1"]})
    real_flip = mod.flip

    def _flip_then_lose(*args: Any) -> None:
        real_flip(*args)
        ms.take_over_migration_window("kb", _state(), holder="other")

    monkeypatch.setattr(mod, "flip", _flip_then_lose)
    with caplog.at_level(logging.INFO, logger=mod.__name__):
        report = _run()

    assert (report.outcome, report.error, _binding()) == ("switched", None, TARGET)
    assert report.window_lost and not report.ok
    assert f"window lost: {report.window_lost}" in mod.describe(report)
    assert "Switch done: switched, window lost: 1 (kb)" in caplog.text


def test_a_metadata_row_gone_under_the_freeze_fails_the_collection(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed({"d1": ["c1"]})
    real_wait = mod.wait_for_no_leases

    def _wait_then_delete(name: str, **kwargs: Any) -> Any:
        scope.connection().open_table(scope.METADATA_TABLE).delete("name = 'kb'")
        return real_wait(name, **kwargs)

    monkeypatch.setattr(mod, "wait_for_no_leases", _wait_then_delete)

    report = _run()

    assert (report.skip_reason, report.ok) == (None, False)
    assert str(report.error).startswith("RuntimeError")
    assert _state().state == ms.IDLE


def test_a_delete_leaves_another_collections_rows_with_the_same_keys(
    adapter: _Adapter,
) -> None:
    _seed({"d1": ["c1"]})
    _write([_src("c1", text="old text")], model=TARGET, vector=list(VECTOR))
    _write([_src("c1")], model=TARGET, collection="kb2", vector=list(VECTOR))

    assert _run().removed == 1

    _assert_switched()
    assert _held("kb2") == [("d1", "c1", "text c1")]


def test_a_target_row_another_model_wrote_after_the_fill_is_not_switched(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed({"d1": ["c1"]})
    real_compare = mod.compare

    def _relabel_then_compare(*args: Any) -> Any:
        scope.connection().open_table(scope.source_table(TARGET)).update(
            where="chunk_id = 'c1'", values={"model": "other-model"}
        )
        return real_compare(*args)

    monkeypatch.setattr(mod, "compare", _relabel_then_compare)

    report = _run()

    assert report.outcome is None and "differs from source at d1/c1" in str(
        report.error
    )
    _assert_untouched()


def test_a_collection_name_the_write_path_would_strip_is_skipped(
    adapter: _Adapter,
) -> None:
    _seed({"d1": ["c1"]})
    conn = scope.connection()
    for table in (scope.METADATA_TABLE, scope.CORE_TABLE, scope.source_table(SOURCE)):
        column = "name" if table == scope.METADATA_TABLE else "collection"
        conn.open_table(table).update(where=f"{column} = 'kb'", values={column: " kb"})

    [report] = mod.switch(TARGET, collections=[" kb"], drain_timeout=0.5)

    assert report.skip_reason == mod.SKIP_NAME_NOT_STRIPPED
    assert (adapter.calls, _target()) == ([], [])
    _never_frozen(" kb")


def test_the_real_index_call_runs_on_the_writer_table_inside_the_window(
    adapter: _Adapter, hub: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _seed_target(hub, VENDOR_TARGET, writer=True)
    [once] = [t for t in scope.own_tables(VENDOR_TARGET) if t != writer]
    _seed({"d1": ["c1", "c2"]})
    states: List[str] = []

    def _create_index(self: Any, model_tag: str, readonly: bool = False) -> Any:
        states.append(_state().state)
        return REAL_CREATE_INDEX(self, model_tag, readonly=readonly)

    monkeypatch.setattr(LanceDBVectorIndexStore, "create_index", _create_index)

    assert _run(VENDOR_TARGET).outcome == "switched"

    conn = scope.connection()
    assert states == [ms.FROZEN]
    assert conn.open_table(writer).list_indices()
    assert conn.open_table(once).list_indices() == []
    _assert_switched(target=VENDOR_TARGET)


def test_a_switched_collection_reads_the_source_once_per_plan_and_once_to_compare(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed({"d1": ["c1"], "d2": ["c2"]})
    scans = _source_scans(monkeypatch)

    assert _run().outcome == "switched"
    assert scans == [False, True, True]
    _assert_switched()


# --- resuming a killed run ---


@pytest.mark.parametrize("killed_at", ["after-claim", "mid-fill", "after-flip"])
def test_take_over_resumes_a_collection_left_frozen(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch, killed_at: str
) -> None:
    _seed({"d1": ["c1"], "d2": ["c2"]})
    if killed_at == "after-claim":
        ms.claim_migration_window("kb", holder=OTHER_RUN)
    else:
        restore = _killed(monkeypatch)
        if killed_at == "mid-fill":

            def _killed_mid_fill(texts: List[str]) -> None:
                if len(adapter.calls) == 2:
                    raise RuntimeError("killed")

            adapter.before_encode = _killed_mid_fill
        _run()
        restore()
        adapter.before_encode = None
    assert _state().state == ms.FROZEN

    refused = _run()
    resumed = _run(take_over=True)

    assert (refused.refused, refused.ok) == ("frozen", False)
    assert resumed.ok
    assert resumed.outcome == (
        "already-switched" if killed_at == "after-flip" else "switched"
    )
    _assert_switched()


def test_a_failed_unfreeze_is_reported_and_resumable(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _seed({"d1": ["c1"]})

    def _store_down(window: Any) -> None:
        raise OSError("store down")

    monkeypatch.setattr(mod, "release_migration_window", _store_down)

    assert mod.main(["--model", TARGET]) == 1
    assert "LEFT FROZEN" in caplog.text and "--take-over" in caplog.text
    assert (_binding(), _state().state) == (TARGET, ms.FROZEN)
    monkeypatch.setattr(mod, "release_migration_window", ms.release_migration_window)
    assert mod.main(["--model", TARGET, "--collection", "kb", "--take-over"]) == 0
    _assert_switched()


@pytest.mark.parametrize(
    "fails", ["read-back", "before-write", "held-elsewhere", "claimed-meanwhile"]
)
def test_a_claim_that_raised_is_reported_left_frozen_only_if_it_froze(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch, fails: str
) -> None:
    _seed({"d1": ["c1"]})
    others = []
    if fails == "held-elsewhere":
        others.append(ms.claim_migration_window("kb", holder=OTHER_RUN))
    name = "read_migration_state" if fails == "read-back" else "_ensure_table"
    real = getattr(ms, name)
    calls: List[str] = []

    def _once(*args: Any, **kwargs: Any) -> Any:
        if not calls:
            calls.append(name)
            if fails == "claimed-meanwhile":
                others.append(ms.claim_migration_window("kb", holder=OTHER_RUN))
            raise OSError("store hiccup")
        return real(*args, **kwargs)

    monkeypatch.setattr(ms, name, _once)
    report = _run()
    line = mod.describe(report)

    if fails == "held-elsewhere":
        assert (calls, report.refused, report.error) == ([], "frozen", None)
    else:
        assert (calls, report.error) == ([name], "OSError: store hiccup")
    assert (_binding(), adapter.calls) == (SOURCE, [])
    if fails == "read-back":
        assert "LEFT FROZEN (still frozen by this run)" in line
        assert _state().state == ms.FROZEN
        assert str(_state().owner).startswith(f"{report.holder}:")
        assert _run(take_over=True).outcome == "switched"
        _assert_switched()
    else:
        assert report.left_frozen is None and "LEFT FROZEN" not in line
        assert _state().owner == (others[0].owner if others else None)
        for window in others:
            embedding_switch._held_state(window)


@pytest.mark.parametrize("fails", ["fence", "before-update"])
@pytest.mark.parametrize("bound", ["source", "target"])
def test_a_take_over_that_raised_is_reported_left_frozen_only_if_it_took_the_row(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch, bound: str, fails: str
) -> None:
    from xagent.core.tools.core.RAG_tools.management import collection_manager

    _seed({"d1": ["c1"]})
    if bound == "source":
        ms.claim_migration_window("kb", holder=OTHER_RUN)
    else:
        restore = _killed(monkeypatch)
        _run()
        restore()
    previous = _state().owner

    def _raise(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("store error")

    if fails == "fence":
        monkeypatch.setattr(collection_manager, "fence_collection_binding_sync", _raise)
    else:
        monkeypatch.setattr(mod, "take_over_migration_window", _raise)

    assert mod.main(["--model", TARGET, "--collection", "kb", "--take-over"]) == 1
    report = _run(take_over=True)
    line = mod.describe(report)
    took = fails == "fence"
    assert report.error == "RuntimeError: store error"
    assert ("LEFT FROZEN" in line, bool(report.left_frozen)) == (took, took)
    assert _state().state == ms.FROZEN
    assert str(_state().owner).startswith(f"{report.holder}:") == took
    assert (_state().owner == previous) == (not took)
    assert _binding() == (SOURCE if bound == "source" else TARGET)


@pytest.mark.parametrize("write", ["transition", "unfreeze"])
def test_a_window_write_that_raised_after_its_commit_is_judged_by_the_row(
    adapter: _Adapter,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    write: str,
) -> None:
    _seed({"d1": ["c1"]})
    real = ms._update_row
    calls: List[str] = []

    def _commit_then_raise(conn: Any, where: str, values: Dict[str, Any]) -> bool:
        result = real(conn, where, values)
        if write == "transition":
            hit = "target_model_id" in values
        else:
            hit = values.get("state") == ms.IDLE
        if hit and not calls:
            calls.append(write)
            raise OSError("store hiccup")
        return result

    monkeypatch.setattr(ms, "_update_row", _commit_then_raise)
    with caplog.at_level(logging.INFO, logger=mod.__name__):
        report = _run()
    line = mod.describe(report)

    assert calls == [write] and not report.ok
    if write == "transition":
        assert report.error == "OSError: store hiccup" and report.outcome is None
        assert "LEFT FROZEN (still frozen by this run)" in line
        assert report.window_lost is None and "window lost" not in line
        assert _state().state == ms.FROZEN and _binding() == SOURCE
        assert str(_state().owner).startswith(f"{report.holder}:")
        assert "Switch done: left frozen: 1 (kb)" in caplog.text
        assert _run(take_over=True).outcome == "switched"
        _assert_switched()
    else:
        assert (report.outcome, report.left_frozen) == ("switched", None)
        assert "failed: unfreeze OSError: store hiccup" in line
        assert "LEFT FROZEN" not in line and "window lost" not in line
        _assert_switched()
        assert "Switch done: switched, failed: 1 (kb)" in caplog.text


def test_only_a_collection_that_tried_a_window_reads_its_row_at_the_end(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed({"d1": ["c1"]})
    _seed({"e1": ["x1", "x2"]}, name="kb3")
    _save_collection("kb2", model_id=None, dimension=None)

    reads: List[str] = []

    def _unreadable_after_the_first_read(name: str) -> Any:
        reads.append(name)
        if reads.count(name) > 1:
            raise OSError("store down")
        return ms.read_migration_state(name)

    monkeypatch.setattr(mod, "read_migration_state", _unreadable_after_the_first_read)
    reports = {r.name: r for r in mod.switch(TARGET, max_fill=1, drain_timeout=0.5)}

    assert (reports["kb"].outcome, reports["kb"].ok) == ("switched", False)
    assert "LEFT FROZEN (state unreadable: OSError)" in mod.describe(reports["kb"])
    _assert_switched()
    assert reports["kb3"].refused == "over-max-fill"
    assert reports["kb2"].skip_reason == "unbound"
    assert reports["kb3"].left_frozen is None and reports["kb2"].left_frozen is None
    assert sorted(reads) == ["kb", "kb", "kb3"]


def test_a_state_unreadable_before_planning_fails_the_collection_untouched(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed({"d1": ["c1"]})
    scans = _source_scans(monkeypatch)

    def _down(name: str) -> Any:
        raise OSError("store down")

    monkeypatch.setattr(mod, "read_migration_state", _down)
    report = _run()

    assert (report.error, report.tried_window) == ("OSError: store down", False)
    assert (report.left_frozen, scans, adapter.calls) == (None, [], [])
    assert _binding() == SOURCE
    _never_frozen()


@pytest.mark.parametrize("take_over", [False, True])
def test_a_collection_a_dead_run_froze_is_planned_only_after_a_take_over(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch, take_over: bool
) -> None:
    _seed({"d1": ["c1", "c2"]})
    dead = ms.claim_migration_window("kb", holder=OTHER_RUN)
    scans = _source_scans(monkeypatch)

    report = _run(max_fill=1, take_over=take_over)
    line = mod.describe(report)

    assert (_binding(), adapter.calls, report.written) == (SOURCE, [], 0)
    assert "LEFT FROZEN" not in line and report.left_frozen is None
    if take_over:
        assert report.refused == "over-max-fill" and scans == [True]
        assert (
            "refused (over-max-fill: 2 to embed + 0 to delete > 1; run the warm-up, "
            "or raise --max-fill)" in line
        )
        assert (_state().state, _state().owner) == (ms.IDLE, None)
    else:
        assert report.refused == "frozen" and scans == []
        assert "refused (frozen by a run that holds or left the window;" in line
        assert (_state().state, _state().owner) == (ms.FROZEN, dead.owner)


def test_a_row_frozen_after_the_read_before_planning_is_not_taken_over(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed({"d1": ["c1"]})
    others: List[Any] = []

    def _frozen_right_after(name: str) -> Any:
        state = ms.read_migration_state(name)
        if not others:
            others.append(ms.claim_migration_window(name, holder=OTHER_RUN))
        return state

    monkeypatch.setattr(mod, "read_migration_state", _frozen_right_after)
    report = _run(take_over=True)

    assert (report.refused, report.left_frozen, adapter.calls) == ("frozen", None, [])
    assert (_state().owner, _binding()) == (others[0].owner, SOURCE)
    embedding_switch._held_state(others[0])


# --- the run ---


@pytest.mark.parametrize("target", [TARGET, VENDOR_TARGET])
def test_a_missing_target_table_never_starts(
    adapter: _Adapter, hub: Any, caplog: pytest.LogCaptureFixture, target: str
) -> None:
    _seed({"d1": ["c1"]})
    writer = _seed_target(hub, target, writer=False)
    before = _versions()

    assert mod.main(["--model", target, "--collection", "kb"]) == 2

    assert f"{writer} does not exist; this command writes only" in caplog.text
    assert (_versions(), adapter.calls) == (before, [])
    _never_frozen()


def test_a_target_table_the_write_path_would_convert_never_starts(
    adapter: _Adapter, caplog: pytest.LogCaptureFixture
) -> None:
    _seed({"d1": ["c1"]})
    writer = _retype_owner(pa.string())
    before = _versions()

    assert mod.main(["--model", TARGET, "--collection", "kb"]) == 2

    assert f"{writer} stores user_id as string, a type the write path" in caplog.text
    assert (_versions(), adapter.calls) == (before, [])
    _never_frozen()


@pytest.mark.parametrize("release", ["released", "release-fails"])
@pytest.mark.parametrize("target", [TARGET, VENDOR_TARGET])
def test_a_target_table_gone_inside_the_window_unfreezes_and_stops_the_run(
    adapter: _Adapter,
    hub: Any,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    release: str,
) -> None:
    _seed({"d1": ["c1"]})
    _seed({"e1": ["x1"]}, name="kb3")
    writer = _seed_target(hub, target, writer=True)
    adapter.before_encode = lambda _texts: scope.connection().drop_table(writer)
    if release == "release-fails":

        def _fail(window: Any) -> None:
            raise RuntimeError("store down")

        monkeypatch.setattr(mod, "release_migration_window", _fail)

    with caplog.at_level(logging.INFO, logger=mod.__name__):
        [report] = mod.switch(target, drain_timeout=0.5)

    assert report.aborted and f"{writer} does not exist" in str(report.error)
    assert (report.outcome, report.written, len(adapter.calls)) == (None, 0, 1)
    assert (report.left_frozen is not None) == (release == "release-fails")
    unfreeze = "unfreeze RuntimeError: store down" in str(report.error)
    assert unfreeze == (release == "release-fails")
    frozen = ms.FROZEN if release == "release-fails" else ms.IDLE
    assert (_state().state, _binding(), _binding("kb3")) == (frozen, SOURCE, SOURCE)
    _never_frozen("kb3")
    assert writer not in scope.table_names(scope.connection())
    assert "Run aborted after kb" in caplog.text


def _other_engine(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    _save_collection(name)
    real = scope.collection_engine
    monkeypatch.setattr(
        scope,
        "collection_engine",
        lambda info: "milvus" if info.name == name else real(info),
    )


def test_exit_codes_follow_the_outcome(
    adapter: _Adapter, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert mod.main(["--model", TARGET]) == 1

    _seed({"d1": ["c1"]})
    _save_collection("kb2", model_id=None, dimension=None)
    _other_engine(monkeypatch, "kbm")
    assert mod.main(["--model", TARGET]) == 0
    assert mod.main(["--model", TARGET]) == 0
    assert _binding("kbm") == SOURCE

    _save_collection("kb4", dimension=None)
    assert mod.main(["--model", TARGET, "--collection", "kb4"]) == 1

    _seed({"e1": ["x1", "x2"]}, name="kb3")
    assert mod.main(["--model", TARGET, "--max-fill", "1"]) == 1
    assert mod.main(["--model", TARGET, "--collection", "nope"]) == 2
    assert mod.main(["--model", TARGET, "--max-fill", "-1"]) == 2
    assert mod.main(["--model", TARGET, "--drain-timeout", "nan"]) == 2
    assert caplog.text.count("--max-fill and --drain-timeout must be >= 0") == 2


def test_the_run_ends_with_a_summary_per_outcome(
    adapter: _Adapter, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed({"d1": ["c1"]})
    _save_collection("kb2", model_id=None, dimension=None)
    _other_engine(monkeypatch, "kbm")
    _save_collection("kb4", dimension=None)
    _seed({"e1": ["x1", "x2"]}, name="kb3")

    with caplog.at_level(logging.INFO, logger=mod.__name__):
        assert mod.main(["--model", TARGET, "--max-fill", "1"]) == 1

    [line] = [r.getMessage() for r in caplog.records if "Switch done" in r.getMessage()]
    for part in (
        "switched: 1 (kb)",
        "skipped (unbound): 1 (kb2)",
        "refused (over-max-fill): 1 (kb3)",
        "skipped (bound-without-dimension): 1 (kb4)",
        "skipped (other-engine): 1 (kbm)",
    ):
        assert part in line, line


def test_the_summary_files_each_collection_under_its_worst_outcome() -> None:
    reports = [
        mod.SwitchReport(name="a", outcome="switched", error="e", left_frozen="x"),
        mod.SwitchReport(name="b", error="e", refused="frozen"),
        mod.SwitchReport(name="c", outcome="switched", refused="frozen"),
        mod.SwitchReport(name="d", outcome="switched"),
        mod.SwitchReport(name="e", refused="frozen", window_lost="x"),
        mod.SwitchReport(name="f", outcome="switched", error="e", window_lost="x"),
        mod.SwitchReport(name="g", outcome="already-switched", window_lost="x"),
        mod.SwitchReport(name="h", outcome="switched", error="e"),
        mod.SwitchReport(name="i", refused="frozen", window_lost="x", left_frozen="x"),
        mod.SwitchReport(
            name="j", outcome="switched", error="e", window_lost="x", left_frozen="x"
        ),
    ]

    assert mod.summary(reports) == (
        "failed: 3 (b, e, g); left frozen: 3 (a, i, j); refused (frozen): 1 (c); "
        "switched: 1 (d); switched, failed: 1 (h); switched, window lost: 1 (f)"
    )


def test_a_binding_unreadable_for_the_report_line_hides_no_outcome(
    adapter: _Adapter, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed({"d1": ["c1"]})
    _seed({"e1": ["x1"]}, name="kb3")

    def _unreadable(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("metadata store down")

    monkeypatch.setattr(mod, "collection_info", _unreadable)
    with caplog.at_level(logging.INFO, logger=mod.__name__):
        assert mod.main(["--model", TARGET]) == 0

    lines = [r.getMessage() for r in caplog.records]
    for name in ("kb", "kb3"):
        _assert_switched(name)
        assert f"{name}: source model {SOURCE}" in "\n".join(lines)
    assert (
        sum(line.endswith("binding unreadable (RuntimeError)") for line in lines) == 2
    )


def test_help_states_outcomes_reasons_and_how_to_resume(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        mod.main(["--help"])
    text = " ".join(capsys.readouterr().out.split())

    for phrase in (
        "does not change the binding of a collection it does not report as switched",
        "every report line but a skipped one ends with the binding",
        "over-max-fill (run the warm-up, or raise --max-fill; the warm-up embeds the "
        "writable source rows whose key the target does not hold with a vector it "
        "wrote and can use, and touches no other target row: not one without a "
        "source row, not one that differs only in a copied field)",
        "The time a collection stays frozen grows with its rows even when nothing "
        "needs filling; --max-fill bounds only the writes",
        "--take-over applies only to the collections named with --collection",
        "unusable-source-rows",
        "writers-still-active",
        "rerunning with --collection NAME --take-over",
        "0 nothing was refused, failed, left frozen or lost its window, and every "
        "skipped collection was unbound, no-metadata or other-engine",
        "1 a collection was refused, failed, left frozen, lost its window, or skipped "
        "for any other reason (left as it was; it needs attention)",
        "2 the run never started",
        "A failed provider call is not retried",
        "The target model's table must exist",
        "if it is gone, that collection fails with its binding unchanged and the run "
        "stops",
        "A collection still frozen by this run when its line is written is reported "
        "LEFT FROZEN",
        '"window lost": the unfreeze did not find the row as this run last wrote it, '
        "and it is not frozen by this run; rerun (already-switched if the binding "
        "moved)",
        "2 the run never started, including when the target's table does not exist",
    ):
        assert phrase in text, phrase


def test_take_over_without_a_named_collection_never_starts(
    adapter: _Adapter, caplog: pytest.LogCaptureFixture
) -> None:
    _seed({"d1": ["c1"]})
    dead = ms.claim_migration_window("kb", holder=OTHER_RUN)

    assert mod.main(["--model", TARGET, "--take-over"]) == 2

    assert "Switch did not start: --take-over applies only with --collection" in (
        caplog.text
    )
    assert (_state().owner, _binding(), adapter.calls) == (dead.owner, SOURCE, [])
    with pytest.raises(scope.MigrationCliError):
        mod.switch(TARGET, collections=[], take_over=True)


@pytest.mark.parametrize(
    "args",
    [
        ["--rate-limit", "-1"],
        ["--rate-limit", "inf"],
        ["--drain-timeout", "-1"],
        ["--max-fill", "-1"],
    ],
)
def test_an_invalid_number_never_starts(
    adapter: _Adapter, monkeypatch: pytest.MonkeyPatch, args: List[str]
) -> None:
    _seed({"d1": ["c1"]})
    started: List[Any] = []
    monkeypatch.setattr(mod, "start_run", lambda *a, **k: started.append(a))

    assert mod.main(["--model", TARGET, *args]) == 2

    assert (started, adapter.calls, _binding()) == ([], [], SOURCE)
    _never_frozen()
