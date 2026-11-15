"""One run's migration window: claim, owned writes, take-over and binding fence.

Every case runs on real LanceDB; the races run in separate processes.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Callable, Optional

import lancedb
import pyarrow as pa
import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import (
    CollectionMigratingError,
    CollectionNotFrozenError,
)
from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
from xagent.core.tools.core.RAG_tools.management import collection_manager as cm
from xagent.core.tools.core.RAG_tools.management import embedding_switch as es
from xagent.core.tools.core.RAG_tools.management import migration_state as state_module
from xagent.core.tools.core.RAG_tools.management.migration_state import (
    FROZEN,
    IDLE,
    MIGRATION_TABLE,
    CollectionMigrationState,
    MigrationWindow,
    claim_migration_window,
    clear_migration_state,
    ensure_collection_writable,
    read_migration_state,
    record_migration_transition,
    release_migration_window,
    take_over_migration_window,
    write_migration_state,
)
from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store
from xagent.core.tools.core.RAG_tools.utils.string_utils import (
    validate_collection_name,
)

OLD_MODEL, OLD_DIM = "model-old", 4
NEW_MODEL, NEW_DIM = "model-new", 8
_OLD_SCHEMA = pa.schema(
    [
        ("name", pa.string()),
        ("state", pa.string()),
        ("source_model_id", pa.string()),
        ("target_model_id", pa.string()),
        ("updated_at", pa.timestamp("us")),
    ]
)


def _conn() -> Any:
    return get_metadata_store().get_raw_connection()


def _rows(name: str) -> list[dict[str, Any]]:
    table = lancedb.connect(_conn().uri).open_table(MIGRATION_TABLE)
    return table.search().where(f"name = '{name}'").to_arrow().to_pylist()


def _bind_models(monkeypatch: pytest.MonkeyPatch) -> None:
    models = {OLD_MODEL: OLD_DIM, NEW_MODEL: NEW_DIM}
    monkeypatch.setattr(
        es,
        "_load_model_from_hub",
        lambda model_id, *_: SimpleNamespace(id=model_id, dimension=models[model_id]),
    )


def _save(name: str = "kb") -> None:
    asyncio.run(
        cm.collection_manager.save_collection(
            CollectionInfo(
                name=name, embedding_model_id=OLD_MODEL, embedding_dimension=OLD_DIM
            )
        )
    )


def _save_legacy(name: str = "kb") -> None:
    asyncio.run(get_metadata_store().ensure_collection_metadata_table())
    table = _conn().open_table("collection_metadata")
    row = {
        "name": name,
        "schema_version": "0.0.0",
        "embedding_model_id": OLD_MODEL,
        "embedding_dimension": OLD_DIM,
        "documents": 0,
        "embeddings": 0,
        "document_names": "[]",
        "extra_metadata": "{}",
    }
    table.add(pa.Table.from_pylist([row], schema=table.schema))
    stored = table.search().where(f"name = '{name}'").to_arrow()
    assert stored["updated_at"].to_pylist() == [None]


def _binding(name: str = "kb") -> Optional[str]:
    return asyncio.run(cm.collection_manager.get_collection(name)).embedding_model_id


def _claim_or_report(holder: str, name: str) -> Optional[str]:
    try:
        return claim_migration_window(name, holder=holder).owner
    except CollectionMigratingError:
        return None
    except Exception as exc:
        # Reported, not raised: a raising process strands the others at the barrier.
        return f"error: {exc!r}"


def _claim_rounds(holder: str, names: list[str], start: Any) -> list[Any]:
    won = []
    for name in names:
        start.wait()
        won.append(_claim_or_report(holder, name))
    return won


def _race_claims(names: list[str], runs: int = 4) -> list[list[Any]]:
    spawn = multiprocessing.get_context("spawn")
    with spawn.Manager() as manager:
        start = manager.Barrier(runs)
        with spawn.Pool(runs) as pool:
            results = [
                pool.apply_async(_claim_rounds, (f"run{i}", names, start))
                for i in range(runs)
            ]
            return [result.get(timeout=240) for result in results]


def _assert_one_winner_each(names: list[str], outcomes: list[list[Any]]) -> None:
    assert not [o for won in outcomes for o in won if str(o).startswith("error")]
    for index, name in enumerate(names):
        winners = [won[index] for won in outcomes if won[index] is not None]
        assert len(winners) == 1, (name, outcomes)
        rows = _rows(name)
        assert len(rows) == 1, (name, rows)
        assert rows[0]["state"] == FROZEN
        assert rows[0]["owner"] == winners[0]


def test_concurrent_first_claims_have_exactly_one_winner() -> None:
    names = [f"fresh{i}" for i in range(4)]

    _assert_one_winner_each(names, _race_claims(names))


def test_concurrent_claims_of_an_idle_row_have_exactly_one_winner() -> None:
    names = [f"idle{i}" for i in range(4)]
    for name in names:
        write_migration_state(name, CollectionMigrationState(state=IDLE))

    _assert_one_winner_each(names, _race_claims(names))


def _old_table(*rows: dict[str, Any]) -> None:
    _conn().create_table(MIGRATION_TABLE, schema=_OLD_SCHEMA).add(
        pa.Table.from_pylist(list(rows), schema=_OLD_SCHEMA)
    )


def test_an_old_table_is_read_unchanged_and_upgraded_by_concurrent_claims() -> None:
    legacy = {"name": "legacy", "state": FROZEN, "target_model_id": NEW_MODEL}
    _old_table(legacy, {"name": "done", "state": IDLE, "source_model_id": OLD_MODEL})

    assert read_migration_state("legacy").state == FROZEN
    assert "owner" not in _conn().open_table(MIGRATION_TABLE).schema.names

    names = [f"upgrade{i}" for i in range(4)]
    outcomes = _race_claims_one_name_each(names)

    assert all(o and not o.startswith("error") for o in outcomes), outcomes
    assert "owner" in _conn().open_table(MIGRATION_TABLE).schema.names
    assert read_migration_state("legacy") == CollectionMigrationState(
        state=FROZEN, target_model_id=NEW_MODEL
    )
    assert read_migration_state("done").source_model_id == OLD_MODEL


def _claim_one(holder: str, name: str, start: Any) -> Optional[str]:
    start.wait()
    return _claim_or_report(holder, name)


def _race_claims_one_name_each(names: list[str]) -> list[Any]:
    spawn = multiprocessing.get_context("spawn")
    with spawn.Manager() as manager:
        start = manager.Barrier(len(names))
        with spawn.Pool(len(names)) as pool:
            results = [
                pool.apply_async(_claim_one, (f"run{i}", name, start))
                for i, name in enumerate(names)
            ]
            return [result.get(timeout=240) for result in results]


def test_a_claim_waits_for_a_table_another_process_is_still_creating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_migration_state("other", CollectionMigrationState(state=IDLE))
    real_conn = _conn()
    opens = []

    class _CreatedElsewhere:
        def __getattr__(self, name: str) -> Any:
            if name != "open_table":
                return getattr(real_conn, name)

            def _open(table_name: str) -> Any:
                opens.append(table_name)
                if len(opens) == 1:
                    raise ValueError(f"Table '{table_name}' was not found")
                return real_conn.open_table(table_name)

            return _open

    monkeypatch.setattr(
        state_module,
        "get_metadata_store",
        lambda: SimpleNamespace(get_raw_connection=lambda: _CreatedElsewhere()),
    )

    window = claim_migration_window("kb", holder="first")

    assert len(opens) > 1
    assert read_migration_state("kb").owner == window.owner


def test_a_frozen_collection_is_refused_with_its_owner_and_time() -> None:
    window = claim_migration_window("kb", holder="first")

    with pytest.raises(CollectionMigratingError) as refused:
        claim_migration_window("kb", holder="second")

    assert refused.value.details["owner"] == window.owner
    assert refused.value.details["updated_at"] == window.updated_at
    assert read_migration_state("kb").owner == window.owner


def test_a_claim_keeps_the_rollback_origin_and_admission_sees_it() -> None:
    window = claim_migration_window("kb", holder="first")
    record_migration_transition(window, OLD_MODEL, NEW_MODEL)
    release_migration_window(window)
    ensure_collection_writable("kb")

    claim_migration_window("kb", holder="second")

    state = read_migration_state("kb")
    assert (state.source_model_id, state.target_model_id) == (OLD_MODEL, NEW_MODEL)
    with pytest.raises(CollectionMigratingError):
        ensure_collection_writable("kb")


def test_a_claim_loses_when_the_idle_row_changed_after_its_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_migration_state("kb", CollectionMigrationState(state=IDLE))
    real_read = state_module._read_row
    reads = []

    def _read_then_rewrite(name: str, store: Any = None) -> Any:
        row = real_read(name, store)
        if not reads:
            reads.append(row)
            write_migration_state(name, CollectionMigrationState(state=IDLE))
        return row

    monkeypatch.setattr(state_module, "_read_row", _read_then_rewrite)

    with pytest.raises(CollectionMigratingError):
        claim_migration_window("kb", holder="late")

    assert read_migration_state("kb").owner is None
    assert read_migration_state("kb").state == IDLE


def test_a_claim_trusts_the_row_it_reads_back_over_a_reported_insert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = claim_migration_window("kb", holder="first")
    real_read = state_module._read_row
    reads = []

    def _absent_on_first_read(name: str, store: Any = None) -> Any:
        reads.append(name)
        return None if len(reads) == 1 else real_read(name, store)

    monkeypatch.setattr(state_module, "_read_row", _absent_on_first_read)
    monkeypatch.setattr(state_module, "_insert_new_row", lambda *_: True)

    with pytest.raises(CollectionMigratingError):
        claim_migration_window("kb", holder="second")

    assert read_migration_state("kb").owner == first.owner


def test_owned_writes_refuse_another_owner_at_the_same_version() -> None:
    window = claim_migration_window("kb", holder="first")
    intruder = MigrationWindow("kb", "intruder", window.updated_at)

    with pytest.raises(CollectionNotFrozenError):
        record_migration_transition(intruder, OLD_MODEL, NEW_MODEL)
    with pytest.raises(CollectionNotFrozenError):
        release_migration_window(intruder)

    state = read_migration_state("kb")
    assert (state.state, state.owner, state.target_model_id) == (
        FROZEN,
        window.owner,
        None,
    )


def test_owned_writes_refuse_a_version_the_owner_already_moved_past() -> None:
    window = claim_migration_window("kb", holder="first")
    stale = replace(window)
    record_migration_transition(window, OLD_MODEL, NEW_MODEL)

    with pytest.raises(CollectionNotFrozenError):
        record_migration_transition(stale, NEW_MODEL, OLD_MODEL)
    with pytest.raises(CollectionNotFrozenError):
        release_migration_window(stale)

    state = read_migration_state("kb")
    assert (state.state, state.source_model_id) == (FROZEN, OLD_MODEL)
    assert state.updated_at == window.updated_at


def test_a_take_over_is_refused_once_the_observed_row_changed() -> None:
    window = claim_migration_window("kb", holder="first")
    observed = read_migration_state("kb")
    record_migration_transition(window, OLD_MODEL, NEW_MODEL)

    with pytest.raises(CollectionMigratingError):
        take_over_migration_window("kb", observed, holder="operator")

    release_migration_window(window)
    assert read_migration_state("kb").state == IDLE


def test_a_take_over_is_refused_for_another_owner_at_the_same_version() -> None:
    claim_migration_window("kb", holder="first")
    observed = replace(read_migration_state("kb"), owner="someone-else")

    with pytest.raises(CollectionMigratingError):
        take_over_migration_window("kb", observed, holder="operator")


def test_a_taken_over_run_can_no_longer_write_and_the_taker_can() -> None:
    window = claim_migration_window("kb", holder="first")
    record_migration_transition(window, OLD_MODEL, NEW_MODEL)

    taker = take_over_migration_window(
        "kb", read_migration_state("kb"), holder="operator"
    )

    with pytest.raises(CollectionNotFrozenError):
        release_migration_window(window)
    state = read_migration_state("kb")
    assert (state.state, state.owner) == (FROZEN, taker.owner)
    assert (state.source_model_id, state.target_model_id) == (OLD_MODEL, NEW_MODEL)
    release_migration_window(taker)
    assert read_migration_state("kb").state == IDLE


def test_a_take_over_of_an_idle_row_is_refused() -> None:
    write_migration_state("kb", CollectionMigrationState(state=IDLE))

    with pytest.raises(CollectionNotFrozenError):
        take_over_migration_window("kb", read_migration_state("kb"), holder="op")


def test_an_unowned_frozen_row_from_an_old_table_can_be_taken_over() -> None:
    _old_table({"name": "legacy", "state": FROZEN, "source_model_id": OLD_MODEL})

    window = take_over_migration_window(
        "legacy", read_migration_state("legacy"), holder="operator"
    )
    release_migration_window(window)

    state = read_migration_state("legacy")
    assert (state.state, state.owner, state.source_model_id) == (
        IDLE,
        None,
        OLD_MODEL,
    )


def test_unowned_writers_refuse_a_held_window() -> None:
    window = claim_migration_window("kb", holder="first")

    with pytest.raises(CollectionMigratingError):
        write_migration_state("kb", CollectionMigrationState(state=IDLE))
    with pytest.raises(CollectionMigratingError):
        clear_migration_state("kb")

    state = read_migration_state("kb")
    assert (state.state, state.owner) == (FROZEN, window.owner)
    release_migration_window(window)
    clear_migration_state("kb")
    assert _rows("kb") == []


def test_a_stale_run_cannot_replace_the_record_of_a_later_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save()
    _bind_models(monkeypatch)
    stalled = claim_migration_window("kb", holder="stalled")
    es.switch_collection_embedding_model(stalled, NEW_MODEL)

    later = take_over_migration_window(
        "kb", read_migration_state("kb"), holder="operator"
    )
    release_migration_window(later)
    again = claim_migration_window("kb", holder="later")
    es.switch_collection_embedding_model(again, OLD_MODEL)
    release_migration_window(again)

    with pytest.raises(CollectionNotFrozenError):
        release_migration_window(stalled)
    with pytest.raises(CollectionNotFrozenError):
        es.switch_collection_embedding_model(stalled, NEW_MODEL)
    state = read_migration_state("kb")
    assert (state.state, state.source_model_id, state.target_model_id) == (
        IDLE,
        NEW_MODEL,
        OLD_MODEL,
    )
    assert _binding() == OLD_MODEL


def test_the_switch_records_the_binding_read_after_the_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save()
    _bind_models(monkeypatch)
    first = claim_migration_window("kb", holder="first")
    es.switch_collection_embedding_model(first, NEW_MODEL)
    release_migration_window(first)

    second = claim_migration_window("kb", holder="second")
    result = es.switch_collection_embedding_model(second, OLD_MODEL)

    assert result.source_model_id == NEW_MODEL
    assert read_migration_state("kb").source_model_id == NEW_MODEL
    assert _binding() == OLD_MODEL


def _take_over_after_the_frozen_check(
    monkeypatch: pytest.MonkeyPatch, then: Callable[[MigrationWindow], Any]
) -> None:
    real_read = cm.read_migration_state

    def _taken_over_after_the_check(name: str) -> Any:
        state = real_read(name)
        monkeypatch.setattr(cm, "read_migration_state", real_read)
        then(take_over_migration_window(name, real_read(name), holder="operator"))
        return state

    monkeypatch.setattr(cm, "read_migration_state", _taken_over_after_the_check)


@pytest.mark.parametrize("save", [_save, _save_legacy], ids=["row", "legacy-row"])
def test_a_flip_prepared_before_a_take_over_cannot_land_after_it(
    monkeypatch: pytest.MonkeyPatch, save: Callable[[], None]
) -> None:
    save()
    _bind_models(monkeypatch)
    stalled = claim_migration_window("kb", holder="stalled")
    _take_over_after_the_frozen_check(monkeypatch, release_migration_window)

    with pytest.raises(ValueError, match="changed while switching"):
        es.switch_collection_embedding_model(stalled, NEW_MODEL)

    assert _binding() == OLD_MODEL
    assert read_migration_state("kb").state == IDLE


def test_a_taken_over_run_is_refused_even_once_its_target_is_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save()
    _bind_models(monkeypatch)
    stalled = claim_migration_window("kb", holder="stalled")

    def _switch_and_release(taker: MigrationWindow) -> None:
        es.switch_collection_embedding_model(taker, NEW_MODEL)
        release_migration_window(taker)

    _take_over_after_the_frozen_check(monkeypatch, _switch_and_release)

    with pytest.raises(CollectionNotFrozenError):
        es.switch_collection_embedding_model(stalled, NEW_MODEL)

    assert _binding() == NEW_MODEL
    state = read_migration_state("kb")
    assert (state.state, state.source_model_id, state.target_model_id) == (
        IDLE,
        OLD_MODEL,
        NEW_MODEL,
    )


def test_a_legacy_row_never_written_since_creation_can_be_switched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_legacy()
    _bind_models(monkeypatch)
    window = claim_migration_window("kb", holder="first")

    es.switch_collection_embedding_model(window, NEW_MODEL)

    assert _binding() == NEW_MODEL


def test_a_take_over_of_a_collection_without_a_metadata_row_succeeds(
    caplog: pytest.LogCaptureFixture,
) -> None:
    window = claim_migration_window("ghost", holder="first")

    taker = take_over_migration_window(
        "ghost", read_migration_state("ghost"), holder="operator"
    )

    assert read_migration_state("ghost").owner == taker.owner != window.owner
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR] == []


def test_the_insert_guard_row_cannot_be_a_collection() -> None:
    claim_migration_window("kb", holder="first")

    table = _conn().open_table(MIGRATION_TABLE)
    others = {row["name"] for row in table.to_arrow().to_pylist()} - {"kb"}

    assert len(others) == 1
    with pytest.raises(ValueError):
        validate_collection_name(others.pop())


@pytest.mark.parametrize(
    "write",
    [
        lambda name: claim_migration_window(name, holder="run"),
        lambda name: take_over_migration_window(
            name, CollectionMigrationState(state=FROZEN), holder="run"
        ),
        lambda name: write_migration_state(name, CollectionMigrationState(FROZEN)),
        clear_migration_state,
    ],
    ids=["claim", "take-over", "write", "clear"],
)
def test_every_writer_refuses_the_insert_guard_name(
    write: Callable[[str], Any],
) -> None:
    claim_migration_window("kb", holder="first")
    version = _conn().open_table(MIGRATION_TABLE).version

    with pytest.raises(ValueError, match="reserved"):
        write(state_module._INSERT_GUARD)

    assert _conn().open_table(MIGRATION_TABLE).version == version
