"""Tests for the multi-collection rollout and the rollback entry point."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from xagent.core.model.model import EmbeddingModelConfig
from xagent.migrations.embedding import rollback as rollback_mod
from xagent.migrations.embedding import switch as mod
from xagent.migrations.embedding.embed import RateLimiter


def _report(name: str, result: str = mod.RESULT_SWITCHED, reason: str = "") -> Any:
    return mod.SwitchReport(
        name=name, target_model_id="new-model", result=result, reason=reason
    )


class _Recorder:
    """Stands in for one window, recording how the rollout called it."""

    def __init__(self, behaviour: Optional[Dict[str, Any]] = None) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.behaviour = behaviour or {}

    def __call__(self, name: str, target: str, **kwargs: Any) -> Any:
        self.calls.append({"name": name, "target": target, **kwargs})
        action = self.behaviour.get(name)
        if action is not None:
            return action(name)
        return _report(name)


def _wire(
    monkeypatch: pytest.MonkeyPatch,
    names: List[str],
    recorder: Optional[_Recorder] = None,
    *,
    acceptance: Any = None,
) -> _Recorder:
    recorder = recorder or _Recorder()
    monkeypatch.setattr(mod, "resolve_collection_names", lambda **kwargs: list(names))
    monkeypatch.setattr(mod, "switch_collection", recorder)
    monkeypatch.setattr(mod, "InflightJobWatcher", lambda: SimpleNamespace())
    if acceptance is not None:
        monkeypatch.setattr(mod, "check_collection", acceptance)
    return recorder


def test_every_collection_in_scope_runs_under_one_owner_and_one_limiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _wire(monkeypatch, ["a", "b", "c"])

    reports = mod.switch_collections("new-model", collections=["a", "b", "c"])

    assert [report.name for report in reports] == ["a", "b", "c"]
    assert [call["name"] for call in recorder.calls] == ["a", "b", "c"]
    assert len({call["owner"] for call in recorder.calls}) == 1
    assert len({id(call["rate_limiter"]) for call in recorder.calls}) == 1
    assert isinstance(recorder.calls[0]["rate_limiter"], RateLimiter)
    assert len({id(call["job_watcher"]) for call in recorder.calls}) == 1


def test_one_crashed_window_does_not_stop_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _explode(name: str) -> Any:
        raise RuntimeError("boom")

    recorder = _wire(monkeypatch, ["a", "b"], _Recorder({"a": _explode}))

    reports = mod.switch_collections("new-model", concurrency=2)

    assert [report.result for report in reports] == [
        mod.RESULT_FAILED,
        mod.RESULT_SWITCHED,
    ]
    assert "RuntimeError" in reports[0].reason
    assert [call["name"] for call in recorder.calls] == ["a", "b"]


def test_an_interrupt_keeps_the_window_report_and_stops_new_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _interrupt(name: str) -> Any:
        raise mod.WindowInterrupted(_report(name, mod.RESULT_INTERRUPTED))

    recorder = _wire(monkeypatch, ["a", "b", "c"], _Recorder({"a": _interrupt}))

    reports = mod.switch_collections("new-model")

    assert [report.result for report in reports] == [
        mod.RESULT_INTERRUPTED,
        mod.RESULT_NOT_STARTED,
        mod.RESULT_NOT_STARTED,
    ]
    assert [call["name"] for call in recorder.calls] == ["a"]


def test_an_empty_scope_reports_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    _wire(monkeypatch, [])

    assert mod.switch_collections("new-model") == []


def test_two_windows_really_run_at_the_same_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    inside = threading.Barrier(2, timeout=5.0)

    def _window(name: str, target: str, **kwargs: Any) -> Any:
        try:
            inside.wait()
        except threading.BrokenBarrierError:
            return _report(name, mod.RESULT_FAILED, "alone")
        return _report(name)

    _wire(monkeypatch, ["a", "b"])
    monkeypatch.setattr(mod, "switch_collection", _window)

    reports = mod.switch_collections("new-model", concurrency=2)

    assert [report.result for report in reports] == [
        mod.RESULT_SWITCHED,
        mod.RESULT_SWITCHED,
    ]


class _Args(SimpleNamespace):
    pass


def _cli_args(**overrides: Any) -> _Args:
    args = dict(
        model="new-model",
        collection=None,
        team=None,
        concurrency=1,
        batch_size=16,
        rate_limit=0.0,
        wait_seconds=0.0,
        use_async=False,
    )
    args.update(overrides)
    return _Args(**args)


def _cli(
    monkeypatch: pytest.MonkeyPatch,
    args: _Args,
    reports: List[Any],
    seen: Optional[Dict[str, Any]] = None,
) -> int:
    class _Parser:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def parse_args(self, argv: Any) -> _Args:
            return args

        def add_argument(self, *a: Any, **k: Any) -> None:
            pass

        def add_mutually_exclusive_group(self, *a: Any, **k: Any) -> "_Parser":
            return self

    monkeypatch.setattr(mod.argparse, "ArgumentParser", _Parser)

    def _switch(target: str, **kwargs: Any) -> List[Any]:
        if seen is not None:
            seen.update(target=target, **kwargs)
        return reports

    monkeypatch.setattr(mod, "switch_collections", _switch)
    return mod.run_cli(mod.MODE_SWITCH, [], description="x")


def test_the_cli_forwards_its_mode_and_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: Dict[str, Any] = {}
    args = _cli_args(
        collection=["kb"], team=None, concurrency=3, wait_seconds=7.0, batch_size=9
    )

    _cli(monkeypatch, args, [_report("kb")], seen)

    assert seen["mode"] == mod.MODE_SWITCH
    assert seen["collections"] == ["kb"]
    assert seen["concurrency"] == 3
    assert seen["wait_seconds"] == 7.0
    assert seen["cfg"].embedding_batch_size == 9
    assert seen["cfg"].embedding_use_async is False


def test_the_rollback_cli_forwards_the_rollback_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: Dict[str, Any] = {}
    args = _cli_args()

    class _Parser:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def parse_args(self, argv: Any) -> _Args:
            return args

        def add_argument(self, *a: Any, **k: Any) -> None:
            pass

        def add_mutually_exclusive_group(self, *a: Any, **k: Any) -> "_Parser":
            return self

    monkeypatch.setattr(mod.argparse, "ArgumentParser", _Parser)
    monkeypatch.setattr(
        mod, "switch_collections", lambda *a, **k: seen.update(k) or [_report("a")]
    )

    mod.run_cli(mod.MODE_ROLLBACK, [], description="x")

    assert seen["mode"] == mod.MODE_ROLLBACK


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"concurrency": 0}, 2),
        ({"batch_size": 0}, 2),
        ({"rate_limit": -1.0}, 2),
        ({"rate_limit": float("inf")}, 2),
    ],
)
def test_the_rollout_cli_refuses_impossible_flags(
    monkeypatch: pytest.MonkeyPatch, overrides: Dict[str, Any], code: int
) -> None:
    assert _cli(monkeypatch, _cli_args(**overrides), [_report("a")]) == code


def test_the_rollout_cli_exits_zero_only_when_every_collection_switched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _cli(monkeypatch, _cli_args(), [_report("a"), _report("b")]) == 0
    for result in (
        mod.RESULT_REFUSED,
        mod.RESULT_FAILED,
        mod.RESULT_NOT_STARTED,
        mod.RESULT_INTERRUPTED,
    ):
        reports = [_report("a"), _report("b", result)]
        assert _cli(monkeypatch, _cli_args(), reports) == 1


def test_an_empty_scope_exits_one(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _cli(monkeypatch, _cli_args(), []) == 1


def test_rollback_is_the_same_cli_in_rollback_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: Dict[str, Any] = {}

    def _run_cli(mode: str, argv: Any, *, description: str) -> int:
        seen.update(mode=mode, description=description)
        return 0

    monkeypatch.setattr(rollback_mod, "run_cli", _run_cli)

    assert rollback_mod.main([]) == 0
    assert seen["mode"] == mod.MODE_ROLLBACK


def test_the_switch_command_runs_the_scope_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: Dict[str, Any] = {}

    def _run_cli(mode: str, argv: Any, *, description: str) -> int:
        seen.update(mode=mode)
        return 0

    monkeypatch.setattr(mod, "run_cli", _run_cli)

    assert mod.main([]) == 0
    assert seen["mode"] == mod.MODE_SWITCH


# --- rollback end to end: a real LanceDB, no mocked copy, check or flip ---


class _Adapter:
    def encode(self, texts: Any, **kwargs: Any) -> List[List[float]]:
        items = texts if isinstance(texts, list) else [texts]
        return [[0.25, 0.75] for _ in items]


class _NoJobs:
    def running(self, collection: str) -> List[str]:
        return []


def _rows(count: int) -> List[Dict[str, Any]]:
    return [
        {
            "collection": "kb",
            "doc_id": "doc-1",
            "chunk_id": f"chunk-{index}",
            "parse_hash": "h",
            "text": f"text {index}",
            "chunk_hash": f"ch-{index}",
            "metadata": None,
            "user_id": 7,
            "vector": [0.5, 0.5],
        }
        for index in range(count)
    ]


def test_a_real_rollback_copies_accepts_and_flips_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    import pyarrow as pa

    from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
    from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
        embeddings_table_candidates,
    )
    from xagent.core.tools.core.RAG_tools.management import collection_manager as cm
    from xagent.core.tools.core.RAG_tools.management import embedding_switch as es
    from xagent.core.tools.core.RAG_tools.management.migration_state import (
        IDLE,
        CollectionMigrationState,
        read_migration_state,
        write_migration_state,
    )
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        get_vector_index_store,
    )

    conn = get_vector_index_store().get_raw_connection()
    conn.create_table(
        embeddings_table_candidates("new-model")[0],
        data=pa.Table.from_pylist(_rows(3)),
    )
    # The old table already holds a legacy ownerless row that acceptance
    # excludes; it is written through the production writer, then orphaned.
    from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData
    from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
        write_vectors_to_db,
    )

    write_vectors_to_db(
        collection="kb",
        embeddings=[
            ChunkEmbeddingData(
                doc_id="doc-1",
                chunk_id="legacy-1",
                parse_hash="h",
                model="old-model",
                vector=[0.5, 0.5],
                text="legacy text",
                chunk_hash="ch-legacy",
                metadata=None,
            )
        ],
        create_index=False,
        user_id=7,
    )
    conn.open_table(embeddings_table_candidates("old-model")[0]).update(
        "chunk_id = 'legacy-1'", {"user_id": None}
    )
    asyncio.run(
        cm.collection_manager.save_collection(
            CollectionInfo(
                name="kb", embedding_model_id="new-model", embedding_dimension=2
            )
        )
    )
    write_migration_state(
        "kb",
        CollectionMigrationState(
            state=IDLE,
            source_model_id="old-model",
            target_model_id="new-model",
            mode=mod.MODE_SWITCH,
        ),
    )
    config = EmbeddingModelConfig(id="old-model", model_name="stub", dimension=2)
    monkeypatch.setattr(
        mod, "resolve_target_model", lambda model_id: (config, _Adapter())
    )
    monkeypatch.setattr(
        cm, "resolve_embedding_adapter", lambda model_id, **k: (config, object())
    )
    monkeypatch.setattr(
        es,
        "_load_model_from_hub",
        lambda model_id, *a, **k: SimpleNamespace(id="old-model", dimension=2),
    )
    monkeypatch.setattr(mod, "live_leases", lambda name: [])

    report = mod.switch_collection(
        "kb",
        "old-model",
        mode=mod.MODE_ROLLBACK,
        job_watcher=_NoJobs(),
        wait_seconds=0.0,
    )

    assert report.result == mod.RESULT_SWITCHED, report.reason
    assert report.rows_copied == 3
    binding = asyncio.run(cm.collection_manager.get_collection("kb"))
    assert binding.embedding_model_id == "old-model"
    table = conn.open_table(embeddings_table_candidates("old-model")[0])
    copied = table.search().where("collection = 'kb'").to_arrow().to_pylist()
    assert sorted(row["chunk_id"] for row in copied) == [
        "chunk-0",
        "chunk-1",
        "chunk-2",
        "legacy-1",
    ]
    assert {row["text"] for row in copied} >= {"text 0", "text 1", "text 2"}
    # The legacy ownerless row survived: prune never deletes excluded rows.
    assert "legacy text" in {row["text"] for row in copied}
    assert any(
        index.index_type == "FTS" and "text" in index.columns
        for index in table.list_indices()
    )
    state = read_migration_state("kb")
    assert (state.state, state.source_model_id, state.target_model_id) == (
        IDLE,
        "new-model",
        "old-model",
    )
    assert state.mode == mod.MODE_ROLLBACK


class _FakeFuture:
    """A future whose result() delivers an interrupt the way Ctrl-C would."""

    def __init__(self, name: str, first: Any, later: Any) -> None:
        self.name = name
        self._first = first
        self._later = later
        self._calls = 0

    def result(self) -> Any:
        self._calls += 1
        outcome = self._first if self._calls == 1 else self._later
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def cancel(self) -> bool:
        return False

    def cancelled(self) -> bool:
        return False


class _FakePool:
    """Two futures: 'a' got the interrupt, 'b' is still running when it hits."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.futures: Dict[Any, str] = {}

    def __enter__(self) -> "_FakePool":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def submit(self, fn: Any, name: str) -> Any:
        if name == "a":
            future = _FakeFuture(
                "a", KeyboardInterrupt(), _report(name, mod.RESULT_INTERRUPTED)
            )
        else:
            future = _FakeFuture("b", KeyboardInterrupt(), KeyboardInterrupt())
        self.futures[future] = name
        return future


def _second_interrupt(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    described: List[Any] = []
    _wire(monkeypatch, ["a", "b"])
    monkeypatch.setattr(mod, "ThreadPoolExecutor", _FakePool)
    monkeypatch.setattr(mod, "describe", lambda report: described.append(report))
    return described


def test_a_second_ctrl_c_re_raises_instead_of_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _second_interrupt(monkeypatch)

    with pytest.raises(KeyboardInterrupt):
        mod.switch_collections("new-model", concurrency=2)


def test_a_running_window_is_not_relabelled_failed_on_the_second_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    described = _second_interrupt(monkeypatch)

    with pytest.raises(KeyboardInterrupt):
        mod.switch_collections("new-model", concurrency=2)

    assert {report.name: report.result for report in described} == {
        "a": mod.RESULT_INTERRUPTED,
        "b": mod.RESULT_RUNNING_AT_EXIT,
    }
