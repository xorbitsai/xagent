"""Tests for the per-collection embedding switch window."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from xagent.core.model.model import EmbeddingModelConfig
from xagent.core.tools.core.RAG_tools.core.schemas import IngestionConfig
from xagent.core.tools.core.RAG_tools.management.migration_state import (
    FROZEN,
    IDLE,
    CollectionMigrationState,
    read_migration_state,
    read_raw_state,
    write_migration_state,
)
from xagent.migrations.embedding import scope as mod_scope
from xagent.migrations.embedding import switch as mod
from xagent.migrations.embedding.acceptance import AcceptanceResult
from xagent.migrations.embedding.embed import CollectionReport, DocumentFailure


class _FakeWatcher:
    def __init__(self, jobs: Optional[List[str]] = None) -> None:
        self.jobs = jobs or []
        self.calls: List[str] = []

    def running(self, collection: str) -> List[str]:
        self.calls.append(collection)
        return list(self.jobs)


def _wire(
    monkeypatch: pytest.MonkeyPatch,
    *,
    engine: str = "lancedb",
    bound: Optional[str] = "old-model",
    info: Any = "default",
    embed: Any = None,
    acceptance: Any = None,
    jobs: Optional[List[str]] = None,
    leases: Optional[List[Any]] = None,
    create_at: Any = "same",
    reason: Optional[str] = None,
) -> Dict[str, List[Any]]:
    calls: Dict[str, List[Any]] = {
        "switch": [],
        "embed": [],
        "accept": [],
        "prune": [],
    }
    if info == "default":
        info_obj: Any = SimpleNamespace(
            embedding_model_id=bound,
            created_at=_CREATED if create_at == "same" else create_at,
        )
    else:
        info_obj = info
    state = {"info": info_obj}

    def _collection_info(name: str, conn: Any = None) -> Any:
        if state["info"] is None:
            raise mod.MissingCollectionError(f"collection '{name}' does not exist")
        if isinstance(state["info"], BaseException):
            raise state["info"]
        return state["info"]

    monkeypatch.setattr(mod, "collection_info", _collection_info)

    def _eligibility(name: str) -> Any:
        from xagent.migrations.embedding.scope import SKIP_UNBOUND, Eligibility

        item = _collection_info(name)
        if reason is not None:
            return Eligibility(engine=engine, reason=reason)
        if engine != "lancedb":
            return Eligibility(engine=engine, reason=mod_scope.SKIP_OTHER_ENGINE)
        model = (item.embedding_model_id or "").strip() or None
        if not model:
            return Eligibility(engine=engine, reason=SKIP_UNBOUND)
        return Eligibility(
            model_id=model, dimension=2, table="embeddings_stub", engine=engine
        )

    monkeypatch.setattr(mod, "source_eligibility", _eligibility)
    monkeypatch.setattr(
        mod,
        "resolve_target_model",
        lambda model_id: (
            EmbeddingModelConfig(id=model_id, model_name="stub", dimension=2),
            SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(mod, "connection", lambda: object())
    monkeypatch.setattr(
        mod,
        "collection_identity",
        lambda name: (True, mod.collection_info(name).created_at),
    )
    monkeypatch.setattr(
        mod,
        "prune_target_only",
        lambda *args, **kwargs: calls["prune"].append(args) or 0,
    )
    monkeypatch.setattr(mod, "target_index_problem", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mod,
        "embed_collection",
        embed
        or (
            lambda *args, **kwargs: calls["embed"].append(kwargs)
            or CollectionReport(
                name="kb", documents_embedded=2, chunks_embedded=5, documents_total=2
            )
        ),
    )
    monkeypatch.setattr(
        mod,
        "check_collection",
        acceptance
        or (
            lambda *args, **kwargs: calls["accept"].append(kwargs)
            or AcceptanceResult(ok=True, checks={"ownerless_excluded": 0})
        ),
    )
    monkeypatch.setattr(mod, "live_leases", lambda name: list(leases or []))
    monkeypatch.setattr(mod, "FINAL_LEASE_WAIT_SECONDS", 0.0)

    def _flip(name: str, model: str) -> None:
        calls["switch"].append((name, model))
        current = state["info"]
        state["info"] = SimpleNamespace(
            embedding_model_id=model, created_at=getattr(current, "created_at", None)
        )

    monkeypatch.setattr(mod, "switch_collection_embedding_model", _flip)
    return calls


_CREATED = datetime(2026, 1, 1)


def _watcher(jobs: Optional[List[str]] = None) -> _FakeWatcher:
    return _FakeWatcher(jobs)


def test_a_healthy_collection_is_frozen_switched_and_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_SWITCHED
    assert report.source_model_id == "old-model"
    assert report.rows_copied == 5
    assert report.documents_copied == 2
    assert calls["switch"] == [("kb", "new-model")]
    state = read_migration_state("kb")
    assert state.state == IDLE
    assert state.source_model_id == "old-model"
    assert state.target_model_id == "new-model"
    assert state.mode == mod.MODE_SWITCH
    assert state.owner is None


def test_a_refusal_restores_the_previous_row_byte_for_byte(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_migration_state(
        "kb",
        CollectionMigrationState(
            state=IDLE,
            source_model_id="model-a",
            target_model_id="old-model",
            mode=mod.MODE_SWITCH,
        ),
    )

    def _failing_acceptance(*args: Any, **kwargs: Any) -> AcceptanceResult:
        return AcceptanceResult(ok=False, failures=["chunks differ"])

    calls = _wire(monkeypatch, acceptance=_failing_acceptance)
    before = read_raw_state("kb")

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason.startswith("acceptance-failed")
    assert calls["switch"] == []
    # The completed-switch record from before the window survives intact.
    assert read_raw_state("kb") == before


def test_a_window_owned_by_another_run_is_not_resumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(monkeypatch)
    write_migration_state(
        "kb",
        CollectionMigrationState(
            state=FROZEN,
            source_model_id="old-model",
            target_model_id="new-model",
            owner="other-host:1:abcd",
        ),
    )

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason.startswith("window-in-progress")
    assert read_migration_state("kb").state == FROZEN


def _age_row(name: str, seconds: float) -> None:
    """Rewrite the row's timestamp: the public writer always stamps now."""
    from xagent.core.tools.core.RAG_tools.management.migration_state import (
        MIGRATION_TABLE,
    )
    from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store

    table = get_metadata_store().get_raw_connection().open_table(MIGRATION_TABLE)
    table.update(
        f"name = '{name}'",
        {
            "updated_at": datetime.now(timezone.utc).replace(tzinfo=None)
            - timedelta(seconds=seconds)
        },
    )


def test_a_stale_window_is_resumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)
    write_migration_state(
        "kb",
        CollectionMigrationState(
            state=FROZEN,
            source_model_id="old-model",
            target_model_id="new-model",
            owner="dead-host:9:beef",
        ),
    )
    _age_row("kb", mod.STALE_FREEZE_SECONDS + 60)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_SWITCHED
    assert calls["switch"] == [("kb", "new-model")]


def test_our_own_window_is_resumed(monkeypatch: pytest.MonkeyPatch) -> None:
    _wire(monkeypatch)
    write_migration_state(
        "kb",
        CollectionMigrationState(
            state=FROZEN,
            source_model_id="old-model",
            target_model_id="new-model",
            owner="me",
        ),
    )

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), owner="me", wait_seconds=0.0
    )

    assert report.result == mod.RESULT_SWITCHED


def test_a_frozen_window_for_another_target_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)
    write_migration_state(
        "kb",
        CollectionMigrationState(
            state=FROZEN,
            source_model_id="old-model",
            target_model_id="third-model",
            owner=None,
        ),
    )

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason.startswith("frozen-for-another-target")
    assert calls["switch"] == []
    assert read_migration_state("kb").state == FROZEN


def test_a_collection_already_on_the_target_needs_no_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch, bound="new-model")

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_ALREADY
    assert calls["switch"] == []
    assert read_migration_state("kb").state == IDLE


@pytest.mark.parametrize(
    ("engine", "bound", "info", "reason"),
    [
        ("milvus", "old-model", "default", "other-engine"),
        ("lancedb", "old-model", None, "unknown-collection"),
        ("lancedb", None, "default", "unbound"),
    ],
)
def test_the_window_refuses_collections_it_cannot_switch(
    monkeypatch: pytest.MonkeyPatch,
    engine: str,
    bound: Optional[str],
    info: Any,
    reason: str,
) -> None:
    calls = _wire(monkeypatch, engine=engine, bound=bound, info=info)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason == reason
    assert calls["switch"] == []


def test_an_unresolvable_target_is_refused_before_freezing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(monkeypatch)

    def _raise(model_id: str) -> Any:
        raise mod.MigrationCliError(f"'{model_id}' is not configured")

    monkeypatch.setattr(mod, "resolve_target_model", _raise)

    report = mod.switch_collection(
        "kb", "alias", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert "not configured" in report.reason
    assert read_migration_state("kb").state == IDLE


def test_in_flight_writers_refuse_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(["job-1"]), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason.startswith("in-flight-writers")
    assert calls["switch"] == []
    assert read_migration_state("kb").state == IDLE


def test_a_live_lease_also_holds_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(monkeypatch, leases=[SimpleNamespace(operation="wipe", owner="h:2")])

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason.startswith("in-flight-writers")


@pytest.mark.parametrize(
    ("before", "after", "result"),
    [
        ((True, _CREATED), (True, _CREATED + timedelta(seconds=1)), "refused"),
        ((True, _CREATED), (False, None), "refused"),
        ((True, None), (False, None), "refused"),
        ((True, _CREATED), (True, _CREATED), "switched"),
    ],
)
def test_a_rename_is_detected_by_stored_identity(
    monkeypatch: pytest.MonkeyPatch, before: Any, after: Any, result: str
) -> None:
    _wire(monkeypatch)
    seen = iter([before, after])
    monkeypatch.setattr(mod, "collection_identity", lambda name: next(seen))

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == result
    if result == "refused":
        assert report.reason == "renamed"


def test_a_legacy_row_without_created_at_is_not_a_rename(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(monkeypatch)
    monkeypatch.setattr(mod, "collection_identity", lambda name: (True, None))

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_SWITCHED


@pytest.mark.parametrize("reason", ["no-owner", "no-text", "no-doc-id"])
def test_excluded_rows_are_reported_not_refused(
    monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    def _embed(*args: Any, **kwargs: Any) -> CollectionReport:
        return CollectionReport(name="kb", skipped={reason: 2})

    seen: Dict[str, Any] = {}

    def _acceptance(*args: Any, **kwargs: Any) -> AcceptanceResult:
        seen.update(kwargs)
        return AcceptanceResult(ok=True)

    calls = _wire(monkeypatch, embed=_embed, acceptance=_acceptance)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_SWITCHED
    assert report.excluded_rows == {reason: 2}
    assert seen["source_model_id"] == "old-model"
    assert seen["target_model_id"] == "new-model"
    assert calls["switch"] == [("kb", "new-model")]


def test_a_failed_copy_still_refuses_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _embed(*args: Any, **kwargs: Any) -> CollectionReport:
        return CollectionReport(
            name="kb",
            failures=[DocumentFailure("d", "1 chunks missing from the target model")],
        )

    calls = _wire(monkeypatch, embed=_embed)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason.startswith("embed-failed")
    assert calls["switch"] == []


def test_document_failures_refuse_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _embed(*args: Any, **kwargs: Any) -> CollectionReport:
        return CollectionReport(
            name="kb", documents_total=1, failures=[DocumentFailure("d", "boom")]
        )

    calls = _wire(monkeypatch, embed=_embed)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason.startswith("embed-failed")
    assert calls["switch"] == []


def test_writers_reappearing_before_the_flip_refuse_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)
    counts = {"n": 0}

    def _leases(name: str) -> List[Any]:
        counts["n"] += 1
        if counts["n"] > 1:
            return [SimpleNamespace(operation="late", owner="h:1")]
        return []

    monkeypatch.setattr(mod, "live_leases", _leases)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason.startswith("writers-restarted")
    assert calls["switch"] == []


def test_a_switch_error_reports_failure_and_unfreezes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)

    def _explode(name: str, model: str) -> None:
        calls["switch"].append((name, model))
        raise RuntimeError("binding write exploded")

    monkeypatch.setattr(mod, "switch_collection_embedding_model", _explode)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_FAILED
    assert "RuntimeError" in report.reason
    assert read_migration_state("kb").state == IDLE


def test_a_failed_record_after_the_flip_is_reported_as_switched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)

    def _write(name: str, state: CollectionMigrationState) -> Any:
        if state.state == IDLE and state.target_model_id is not None:
            raise OSError("metadata store gone")
        write_migration_state(name, state)
        return state

    monkeypatch.setattr(mod, "write_migration_state", _write)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_SWITCHED
    assert report.reason.startswith("left-frozen")
    assert calls["switch"] == [("kb", "new-model")]


def test_the_run_config_and_rate_limit_reach_the_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)
    cfg = IngestionConfig(
        embedding_model_id="new-model", embedding_batch_size=7, embedding_use_async=True
    )

    mod.switch_collection(
        "kb",
        "new-model",
        job_watcher=_watcher(),
        cfg=cfg,
        rate_limit=3.0,
        wait_seconds=0.0,
    )

    assert calls["embed"][0]["cfg"] is cfg
    assert calls["embed"][0]["dry_run"] is False


def test_rollback_needs_a_recorded_switch_back_to_its_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch, bound="new-model")

    report = mod.switch_collection(
        "kb",
        "old-model",
        job_watcher=_watcher(),
        mode=mod.MODE_ROLLBACK,
        wait_seconds=0.0,
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason == "no-recorded-switch"
    assert calls["switch"] == []


def test_rollback_refuses_when_the_record_is_for_another_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(monkeypatch, bound="new-model")
    write_migration_state(
        "kb",
        CollectionMigrationState(
            state=IDLE,
            source_model_id="other-model",
            target_model_id="new-model",
            mode=mod.MODE_SWITCH,
        ),
    )

    report = mod.switch_collection(
        "kb",
        "old-model",
        job_watcher=_watcher(),
        mode=mod.MODE_ROLLBACK,
        wait_seconds=0.0,
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason == "recorded-source-is-other-model"


def test_rollback_undoes_a_recorded_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _wire(monkeypatch, bound="new-model")
    write_migration_state(
        "kb",
        CollectionMigrationState(
            state=IDLE,
            source_model_id="old-model",
            target_model_id="new-model",
            mode=mod.MODE_SWITCH,
        ),
    )

    report = mod.switch_collection(
        "kb",
        "old-model",
        job_watcher=_watcher(),
        mode=mod.MODE_ROLLBACK,
        wait_seconds=0.0,
    )

    assert report.result == mod.RESULT_SWITCHED
    assert calls["switch"] == [("kb", "old-model")]
    state = read_migration_state("kb")
    assert (state.state, state.source_model_id, state.target_model_id) == (
        IDLE,
        "new-model",
        "old-model",
    )
    assert state.mode == mod.MODE_ROLLBACK


def test_wait_until_idle_reports_what_is_left_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 100.0}
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        mod.time, "sleep", lambda seconds: clock.update(now=clock["now"] + seconds)
    )
    monkeypatch.setattr(
        mod,
        "live_leases",
        lambda name: [SimpleNamespace(operation="wipe", owner="h:2")],
    )

    left = mod._wait_until_idle(
        "kb", timeout=2.0, poll=1.0, watcher=_watcher(["job-1"])
    )

    assert "job:job-1" in left
    assert "lease:wipe@h:2" in left


def test_the_freeze_is_written_before_the_first_lease_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(monkeypatch)
    events: List[str] = []
    claim = mod.claim_freeze

    def _claim(name: str, **kwargs: Any) -> str:
        events.append(f"claim:{read_migration_state(name).state}")
        outcome = claim(name, **kwargs)
        events.append(f"state:{read_migration_state(name).state}")
        return outcome

    def _leases(name: str) -> List[Any]:
        events.append("leases")
        return []

    monkeypatch.setattr(mod, "claim_freeze", _claim)
    monkeypatch.setattr(mod, "live_leases", _leases)

    mod.switch_collection("kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0)

    assert events[:3] == [f"claim:{IDLE}", f"state:{FROZEN}", "leases"]


def test_the_job_watcher_reads_running_ingest_jobs_from_the_database(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xagent.web.models.background_job import BackgroundJob, BackgroundJobStatus
    from xagent.web.models.database import get_session_local, init_db
    from xagent.web.models.user import User

    init_db(f"sqlite:///{tmp_path / 'watcher.db'}")
    sessions = get_session_local()
    session = sessions()
    try:
        session.add(User(id=1, username="watcher", password_hash="x"))
        session.commit()
        for job_id, status, collection in (
            ("running-kb", BackgroundJobStatus.RUNNING.value, "kb"),
            ("running-other", BackgroundJobStatus.RUNNING.value, "other"),
            ("pending-kb", BackgroundJobStatus.PENDING.value, "kb"),
        ):
            session.add(
                BackgroundJob(
                    id=job_id,
                    user_id=1,
                    job_type="kb.ingest.document",
                    status=status,
                    payload={"collection": collection},
                )
            )
        session.commit()
    finally:
        session.close()

    from xagent.web.models import database

    # The command configures the application's database; here that is the
    # sqlite file the test just filled.
    monkeypatch.setattr(database, "configure_db", lambda *a, **k: None)
    monkeypatch.setattr(database, "get_session_local", lambda: sessions)

    watcher = mod.InflightJobWatcher()

    assert watcher.running("kb") == ["running-kb"]
    assert watcher.running("other") == ["running-other"]


def _frozen(
    owner: Optional[str], mode: str = "switch", target: str = "new-model"
) -> None:
    write_migration_state(
        "kb",
        CollectionMigrationState(
            state=FROZEN,
            source_model_id="old-model",
            target_model_id=target,
            owner=owner,
            mode=mode,
        ),
    )


def test_a_refusal_on_a_fresh_collection_leaves_no_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(
        monkeypatch,
        acceptance=lambda *a, **k: AcceptanceResult(ok=False, failures=["x"]),
    )

    mod.switch_collection("kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0)

    assert read_raw_state("kb") is None


def test_a_refused_resumed_window_unfreezes_without_a_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(
        monkeypatch,
        acceptance=lambda *a, **k: AcceptanceResult(ok=False, failures=["x"]),
    )
    _frozen("me")

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), owner="me", wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    state = read_migration_state("kb")
    assert (state.state, state.target_model_id, state.mode) == (IDLE, None, None)


def test_an_ownerless_fresh_freeze_is_not_taken_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)
    _frozen(None)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason.startswith("window-in-progress")
    assert calls["switch"] == []


def test_the_heartbeat_keeps_a_long_window_from_going_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _frozen("me")
    _age_row("kb", mod.STALE_FREEZE_SECONDS + 60)

    assert mod._Heartbeat("kb", "me").beat() is True
    assert not mod._may_resume(read_migration_state("kb"), "someone-else")


def test_a_heartbeat_on_a_row_someone_else_owns_is_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _frozen("other")
    beat = mod._Heartbeat("kb", "me")

    assert beat.beat() is False
    assert beat.lost is True


def test_a_window_that_lost_its_row_does_not_flip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)
    monkeypatch.setattr(mod, "heartbeat_freeze", lambda name, owner: False)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason == "ownership-lost"
    assert calls["switch"] == []


def test_a_crash_after_the_flip_is_recorded_as_the_switch_by_the_rerun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch, bound="new-model")
    _frozen("dead-host:9:beef")
    _age_row("kb", mod.STALE_FREEZE_SECONDS + 60)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_SWITCHED
    assert calls["switch"] == []
    state = read_migration_state("kb")
    assert (state.state, state.source_model_id, state.target_model_id, state.mode) == (
        IDLE,
        "old-model",
        "new-model",
        mod.MODE_SWITCH,
    )


def test_a_flipped_window_still_heartbeating_is_left_to_its_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(monkeypatch, bound="new-model")
    _frozen("live-host:1:cafe")

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert read_migration_state("kb").state == FROZEN


@pytest.mark.parametrize(
    ("frozen_mode", "mode", "target", "reason"),
    [
        ("switch", "rollback", "old-model", "frozen-for-another-target: new-model"),
        ("switch", "rollback", "new-model", "frozen-for-switch"),
        ("rollback", "switch", "new-model", "frozen-for-rollback"),
    ],
)
def test_a_frozen_row_only_resumes_under_its_own_mode(
    monkeypatch: pytest.MonkeyPatch,
    frozen_mode: str,
    mode: str,
    target: str,
    reason: str,
) -> None:
    calls = _wire(monkeypatch, bound="old-model")
    _frozen("me", mode=frozen_mode)

    report = mod.switch_collection(
        "kb", target, job_watcher=_watcher(), owner="me", mode=mode, wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason == reason
    assert calls["switch"] == []


def test_a_frozen_rollback_window_resumes_as_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch, bound="new-model")
    write_migration_state(
        "kb",
        CollectionMigrationState(
            state=FROZEN,
            source_model_id="new-model",
            target_model_id="old-model",
            owner="me",
            mode=mod.MODE_ROLLBACK,
        ),
    )

    report = mod.switch_collection(
        "kb",
        "old-model",
        job_watcher=_watcher(),
        owner="me",
        mode=mod.MODE_ROLLBACK,
        wait_seconds=0.0,
    )

    assert report.result == mod.RESULT_SWITCHED
    assert calls["switch"] == [("kb", "old-model")]


def test_ctrl_c_before_the_flip_restores_the_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_migration_state(
        "kb",
        CollectionMigrationState(
            state=IDLE,
            source_model_id="model-a",
            target_model_id="old-model",
            mode=mod.MODE_SWITCH,
        ),
    )
    before = read_raw_state("kb")

    def _interrupt(*args: Any, **kwargs: Any) -> AcceptanceResult:
        raise KeyboardInterrupt

    calls = _wire(monkeypatch, acceptance=_interrupt)

    with pytest.raises(mod.WindowInterrupted) as caught:
        mod.switch_collection(
            "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
        )

    assert caught.value.report.result == mod.RESULT_INTERRUPTED
    assert calls["switch"] == []
    assert read_raw_state("kb") == before


def test_ctrl_c_after_the_flip_records_the_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)
    flip = mod.switch_collection_embedding_model

    def _flip_then_interrupt(name: str, model: str) -> None:
        flip(name, model)
        raise KeyboardInterrupt

    monkeypatch.setattr(mod, "switch_collection_embedding_model", _flip_then_interrupt)

    with pytest.raises(mod.WindowInterrupted) as caught:
        mod.switch_collection(
            "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
        )

    assert caught.value.report.result == mod.RESULT_SWITCHED
    assert calls["switch"] == [("kb", "new-model")]
    state = read_migration_state("kb")
    assert (state.state, state.source_model_id, state.target_model_id) == (
        IDLE,
        "old-model",
        "new-model",
    )


def test_a_missing_target_index_refuses_the_flip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)
    monkeypatch.setattr(
        mod, "target_index_problem", lambda *a, **k: "no full-text index on text"
    )

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason.startswith("target-index-missing")
    assert calls["switch"] == []


def test_prune_runs_inside_the_window_before_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: List[str] = []

    def _acceptance(*args: Any, **kwargs: Any) -> AcceptanceResult:
        order.append("accept")
        return AcceptanceResult(ok=True)

    _wire(monkeypatch, acceptance=_acceptance)

    def _prune(name: str, source: str, target: str, **kwargs: Any) -> int:
        order.append(f"prune:{source}->{target}:{read_migration_state(name).state}")
        return 4

    monkeypatch.setattr(mod, "prune_target_only", _prune)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert order == [f"prune:old-model->new-model:{FROZEN}", "accept"]
    assert report.rows_pruned == 4


def _table_for(model_id: str) -> str:
    from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
        embeddings_table_candidates,
    )

    return embeddings_table_candidates(model_id)[0]


OLD = _table_for("old-model")
NEW = _table_for("new-model")


def _embeddings(conn: Any, name: str, keys: List[Any]) -> None:
    import pyarrow as pa

    schema = pa.schema(
        [
            ("collection", pa.string()),
            ("doc_id", pa.string()),
            ("chunk_id", pa.string()),
            ("parse_hash", pa.string()),
            ("user_id", pa.int64()),
            ("text", pa.string()),
            ("vector", pa.list_(pa.float32(), 2)),
        ]
    )
    rows = [
        {
            "collection": collection,
            "doc_id": doc,
            "chunk_id": chunk,
            "parse_hash": "h",
            "user_id": 1,
            "text": "t",
            "vector": [0.5, 0.5],
        }
        for collection, doc, chunk in keys
    ]
    conn.create_table(name, data=pa.Table.from_pylist(rows, schema=schema))


def _keys(conn: Any, name: str) -> List[Any]:
    rows = conn.open_table(name).to_arrow().to_pylist()
    return sorted((r["collection"], r["doc_id"], r["chunk_id"]) for r in rows)


def test_prune_deletes_only_rows_the_source_no_longer_has() -> None:
    conn = mod.connection()
    _embeddings(conn, OLD, [("kb", "a", "a1"), ("kb", "b", "b1")])
    _embeddings(
        conn,
        NEW,
        [
            ("kb", "a", "a1"),
            ("kb", "a", "gone"),
            ("kb", "c", "c1"),
            ("other", "z", "z"),
        ],
    )

    pruned = mod.prune_target_only("kb", "old-model", "new-model", conn=conn)

    assert pruned == 2
    assert _keys(mod.connection(), NEW) == [
        ("kb", "a", "a1"),
        ("other", "z", "z"),
    ]


@pytest.mark.parametrize("failing", [OLD, NEW])
def test_a_failed_read_on_either_side_prunes_nothing(
    monkeypatch: pytest.MonkeyPatch, failing: str
) -> None:
    conn = mod.connection()
    _embeddings(conn, OLD, [("kb", "a", "a1")])
    _embeddings(conn, NEW, [("kb", "a", "a1"), ("kb", "a", "x")])
    real = mod.iter_rows

    def _iter_rows(conn: Any, table: str, *args: Any, **kwargs: Any) -> Any:
        if table == failing:
            raise RuntimeError("corrupt table")
        return real(conn, table, *args, **kwargs)

    monkeypatch.setattr(mod, "iter_rows", _iter_rows)

    with pytest.raises(RuntimeError, match="corrupt table"):
        mod.prune_target_only("kb", "old-model", "new-model", conn=conn)

    assert len(_keys(mod.connection(), NEW)) == 2


def test_prune_without_a_target_table_is_a_no_op() -> None:
    conn = mod.connection()
    _embeddings(conn, OLD, [("kb", "a", "a1")])

    assert mod.prune_target_only("kb", "old-model", "new-model", conn=conn) == 0


def test_the_target_needs_a_full_text_index_before_the_flip() -> None:
    conn = mod.connection()
    _embeddings(conn, NEW, [("kb", "a", "a1")])

    assert mod.target_index_problem("new-model", conn=conn) == (
        "no full-text index on text"
    )

    conn.open_table(NEW).create_fts_index("text", replace=True)

    assert mod.target_index_problem("new-model", conn=conn) is None


def test_the_target_needs_a_vector_index_above_the_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = mod.connection()
    _embeddings(conn, NEW, [("kb", "a", "a1")])
    conn.open_table(NEW).create_fts_index("text", replace=True)
    monkeypatch.setattr(
        mod, "IndexPolicy", lambda: SimpleNamespace(enable_threshold_rows=1)
    )

    assert mod.target_index_problem("new-model", conn=conn) == "no vector index"


def test_stored_identity_reads_the_metadata_row() -> None:
    import pyarrow as pa

    conn = mod.connection()
    conn.create_table(
        mod.METADATA_TABLE,
        data=pa.Table.from_pylist(
            [{"name": "kb", "created_at": _CREATED}],
            schema=pa.schema(
                [("name", pa.string()), ("created_at", pa.timestamp("us"))]
            ),
        ),
    )

    assert mod.collection_identity("kb") == (True, _CREATED)
    assert mod.collection_identity("renamed-away") == (False, None)


def test_a_metadata_table_without_created_at_has_no_identity() -> None:
    import pyarrow as pa

    mod.connection().create_table(
        mod.METADATA_TABLE,
        data=pa.Table.from_pylist(
            [{"name": "kb"}], schema=pa.schema([("name", pa.string())])
        ),
    )

    assert mod.collection_identity("kb") == (True, None)


def test_a_refusal_after_another_run_took_the_row_leaves_it_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _stolen(*args: Any, **kwargs: Any) -> AcceptanceResult:
        _frozen("thief")
        return AcceptanceResult(ok=False, failures=["x"])

    _wire(monkeypatch, acceptance=_stolen)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert read_migration_state("kb").owner == "thief"


def test_a_flipped_window_with_a_fresh_ownerless_row_is_not_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(monkeypatch, bound="new-model")
    _frozen(None)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert read_migration_state("kb").state == FROZEN


def test_rollback_onto_a_frozen_row_without_a_target_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch, bound="new-model")
    write_migration_state(
        "kb",
        CollectionMigrationState(state=FROZEN, owner="me", mode=mod.MODE_ROLLBACK),
    )

    report = mod.switch_collection(
        "kb",
        "old-model",
        job_watcher=_watcher(),
        owner="me",
        mode=mod.MODE_ROLLBACK,
        wait_seconds=0.0,
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason == "no-recorded-switch"
    assert calls["switch"] == []


def test_prune_does_not_keep_a_row_another_collection_still_has() -> None:
    conn = mod.connection()
    _embeddings(conn, OLD, [("kb", "a", "a1"), ("other", "g", "gone")])
    _embeddings(conn, NEW, [("kb", "a", "a1"), ("kb", "g", "gone")])

    assert mod.prune_target_only("kb", "old-model", "new-model", conn=conn) == 1


def test_an_unreadable_binding_after_a_failure_leaves_the_row_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _explode(*args: Any, **kwargs: Any) -> AcceptanceResult:
        raise RuntimeError("boom")

    _wire(monkeypatch, acceptance=_explode)
    monkeypatch.setattr(mod, "_bound_to", lambda name, attempts=3: None)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0, owner="me"
    )

    assert report.result == mod.RESULT_FAILED
    assert "binding-unreadable" in report.reason
    state = read_migration_state("kb")
    assert (state.state, state.owner) == (FROZEN, "me")


def test_an_unreadable_metadata_row_refuses_instead_of_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(monkeypatch, info=OSError("metadata store unavailable"))

    report = mod.switch_collection("kb", "new-model")

    assert report.result == mod.RESULT_REFUSED
    assert report.reason.startswith("metadata-unreadable")
    assert "metadata store unavailable" in report.reason


def test_the_window_uses_the_precompute_eligibility_predicate() -> None:
    # One function, so the copy and the switch can never disagree about which
    # collections may be migrated.
    assert mod.source_eligibility is mod_scope.source_eligibility


@pytest.mark.parametrize(
    "reason",
    [
        mod_scope.SKIP_ROWS_IN_OTHER_TABLE,
        mod_scope.SKIP_ROWS_IN_MULTIPLE_TABLES,
        mod_scope.SKIP_BOUND_WITHOUT_DIMENSION,
        mod_scope.SKIP_MISSING_COLUMN,
        mod_scope.SKIP_UNBOUND,
    ],
)
def test_an_ineligible_collection_is_refused_with_its_reason(
    monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    calls = _wire(monkeypatch, reason=reason)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason == reason
    assert calls["switch"] == []
    assert read_migration_state("kb").state == IDLE


def test_a_target_that_writes_the_source_table_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch, bound="old-model")

    report = mod.switch_collection(
        "kb", "old_model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert "both write" in report.reason
    assert calls["switch"] == []
    assert read_migration_state("kb").state == IDLE
