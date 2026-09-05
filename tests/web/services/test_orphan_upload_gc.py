"""Orphan GC of task-less public uploads (#973, PR3)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import inspect as sa_inspect

import xagent.web.services.orphan_upload_gc as orphan_upload_gc
from xagent.web.models.database import (
    Base,
    get_db,
    get_engine,
    get_session_local,
    init_db,
)
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services.file_turn import bind_turn_files_no_commit
from xagent.web.services.orphan_upload_gc import (
    TASKLESS_SHARE_UPLOAD_SOURCE,
    OrphanUploadSweepCursor,
    _claim_orphan,
    _OrphanUploadCandidate,
    _run_gc_tick,
    cleanup_detached_uploaded_files,
    cleanup_orphaned_taskless_uploads,
)
from xagent.web.services.uploaded_file_recovery import (
    get_stale_uploaded_file_compensation_candidates,
)
from xagent.web.services.workforce_runs import (
    WorkforceRunError,
    _bind_selected_files_to_task,
)

DAY = 24 * 60 * 60


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'orphan_gc.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


@pytest.fixture(autouse=True)
def configured_uploads_root(tmp_path, monkeypatch):
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path))


@pytest.fixture()
def owner(db_session) -> User:
    user = User(username="gc-owner", password_hash="h", is_admin=False)
    db_session.add(user)
    db_session.commit()
    return user


def _mk_upload(
    db_session,
    owner: User,
    tmp_path: Path,
    *,
    name: str,
    marker: str | None,
    task_id: int | None,
    age_days: float,
    detached_reason: str | None = None,
    detached_age_days: float | None = None,
    under_user_root: bool = True,
) -> tuple[UploadedFile, Path]:
    """One registered upload in the exact shape the public path leaves it:
    ``available`` with a durable storage key (the only state a marked row can
    exist in — the registration pipeline never commits anything else)."""
    root = tmp_path / f"user_{int(owner.id)}" if under_user_root else tmp_path
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_bytes(b"payload")
    now = datetime.now(timezone.utc)
    file_id = str(uuid4())
    row = UploadedFile(
        file_id=file_id,
        user_id=int(owner.id),
        task_id=task_id,
        filename=name,
        storage_path=str(path),
        storage_key=f"users/{int(owner.id)}/uploads/{file_id}/{name}",
        storage_status="available",
        file_size=path.stat().st_size,
        upload_source=marker,
        detached_reason=detached_reason,
        detached_at=(
            now - timedelta(days=detached_age_days)
            if detached_age_days is not None
            else None
        ),
        created_at=now - timedelta(days=age_days),
        updated_at=now - timedelta(days=age_days),
    )
    db_session.add(row)
    db_session.commit()
    db_session.refresh(row)
    return row, path


def _candidate(row: UploadedFile) -> _OrphanUploadCandidate:
    return _OrphanUploadCandidate(
        row_id=int(row.id),
        user_id=int(row.user_id),
        file_id=str(row.file_id),
        storage_key=str(row.storage_key),
        storage_path=str(row.storage_path),
        created_at=row.created_at,
        detached_reason=row.detached_reason,
        detached_at=row.detached_at,
    )


def _make_task(db_session, owner: User) -> int:
    task = Task(
        user_id=int(owner.id),
        title="t",
        description="t",
        status=TaskStatus.PENDING,
    )
    db_session.add(task)
    db_session.commit()
    return int(task.id)


def test_create_all_declares_the_gc_index(db_session) -> None:
    """Fresh installations stamp Alembic head BEFORE create_all(), so the
    migration never runs there — the model metadata itself must produce the
    GC index (#996 review)."""
    indexes = {
        ix["name"] for ix in sa_inspect(get_engine()).get_indexes("uploaded_files")
    }
    assert "ix_uploaded_files_orphan_gc" in indexes


def test_reaps_aged_marked_unbound_upload(db_session, owner, tmp_path) -> None:
    row, path = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="orphan.txt",
        marker=TASKLESS_SHARE_UPLOAD_SOURCE,
        task_id=None,
        age_days=5,
    )
    row_id = int(row.id)

    result = cleanup_orphaned_taskless_uploads(db_session, older_than_seconds=2 * DAY)

    assert (result.scanned, result.deleted) == (1, 1)
    assert db_session.query(UploadedFile).filter_by(id=row_id).first() is None
    assert not path.exists()  # on-disk file removed too


def test_spares_marked_but_recent_upload(db_session, owner, tmp_path) -> None:
    row, path = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="fresh.txt",
        marker=TASKLESS_SHARE_UPLOAD_SOURCE,
        task_id=None,
        age_days=0,
    )
    row_id = int(row.id)

    result = cleanup_orphaned_taskless_uploads(db_session, older_than_seconds=2 * DAY)

    assert (result.scanned, result.deleted) == (0, 0)
    assert db_session.query(UploadedFile).filter_by(id=row_id).first() is not None
    assert path.exists()


def test_spares_unmarked_unbound_upload(db_session, owner, tmp_path) -> None:
    """A logged-in user's aged, un-sent draft (no marker) must never be reaped
    by the task_id-IS-NULL sweep."""
    row, path = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="draft.txt",
        marker=None,
        task_id=None,
        age_days=10,
    )
    row_id = int(row.id)

    result = cleanup_orphaned_taskless_uploads(db_session, older_than_seconds=2 * DAY)

    assert result.deleted == 0
    assert db_session.query(UploadedFile).filter_by(id=row_id).first() is not None
    assert path.exists()


def test_spares_marked_but_bound_upload(db_session, owner, tmp_path) -> None:
    """Once a marked upload is bound to a task (run started), it is no longer an
    orphan and must be kept."""
    task_id = _make_task(db_session, owner)
    row, path = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="bound.txt",
        marker=TASKLESS_SHARE_UPLOAD_SOURCE,
        task_id=task_id,
        age_days=5,
    )
    row_id = int(row.id)

    result = cleanup_orphaned_taskless_uploads(db_session, older_than_seconds=2 * DAY)

    assert result.deleted == 0
    assert db_session.query(UploadedFile).filter_by(id=row_id).first() is not None
    assert path.exists()


def test_claim_requires_exact_available_status(db_session, owner, tmp_path) -> None:
    """The claim CAS demands the exact prior state: a bound row fails on
    ``task_id IS NULL``; a second claim on the same row fails on
    ``storage_status == 'available'`` — overlapping sweeps are mutually
    exclusive, never co-owners."""
    task_id = _make_task(db_session, owner)
    bound, _ = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="claim-bound.txt",
        marker=TASKLESS_SHARE_UPLOAD_SOURCE,
        task_id=task_id,
        age_days=5,
    )
    unbound, _ = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="claim-unbound.txt",
        marker=TASKLESS_SHARE_UPLOAD_SOURCE,
        task_id=None,
        age_days=5,
    )

    assert _claim_orphan(db_session, _candidate(bound)) is None

    token = _claim_orphan(db_session, _candidate(unbound))
    assert token is not None
    db_session.refresh(unbound)
    assert unbound.storage_status == "compensating"
    assert unbound.updated_at == token  # persisted generation token

    # An overlapping sweep claiming the same row loses outright.
    assert _claim_orphan(db_session, _candidate(unbound)) is None


def test_claimed_row_is_excluded_from_ws_turn_binding(
    db_session, owner, tmp_path
) -> None:
    """The WS-turn binder's conditional update must skip a claimed row rather
    than resurrect it."""
    task_id = _make_task(db_session, owner)
    row, _ = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="claimed-ws.txt",
        marker=TASKLESS_SHARE_UPLOAD_SOURCE,
        task_id=None,
        age_days=5,
    )
    assert _claim_orphan(db_session, _candidate(row)) is not None

    missing = bind_turn_files_no_commit(
        file_ids=[str(row.file_id)],
        task_id=task_id,
        owner_user_id=int(owner.id),
        db=db_session,
    )

    assert missing == [str(row.file_id)]  # bind refused the claimed row
    db_session.rollback()
    db_session.refresh(row)
    assert row.task_id is None


def test_claimed_row_is_excluded_from_workforce_run_binding(
    db_session, owner, tmp_path
) -> None:
    """The workforce run-start binder — the binder actually used by the
    task-less share/widget path — must refuse a claimed row via the shared
    conditional-update binder instead of overwriting the claim with a stale
    ORM assignment (#996 review)."""
    task = Task(
        user_id=int(owner.id),
        title="wf",
        description="wf",
        status=TaskStatus.PENDING,
    )
    db_session.add(task)
    db_session.commit()

    claimed, _ = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="claimed-wf.txt",
        marker=TASKLESS_SHARE_UPLOAD_SOURCE,
        task_id=None,
        age_days=5,
    )
    assert _claim_orphan(db_session, _candidate(claimed)) is not None

    with pytest.raises(WorkforceRunError):
        _bind_selected_files_to_task(db_session, owner, task, [str(claimed.file_id)])
    db_session.rollback()
    db_session.refresh(claimed)
    assert claimed.task_id is None
    assert claimed.storage_status == "compensating"  # claim intact

    # And the healthy path still binds.
    free, _ = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="free-wf.txt",
        marker=TASKLESS_SHARE_UPLOAD_SOURCE,
        task_id=None,
        age_days=5,
    )
    _bind_selected_files_to_task(db_session, owner, task, [str(free.file_id)])
    db_session.commit()
    db_session.refresh(free)
    assert free.task_id == int(task.id)


def test_row_bound_between_fetch_and_claim_is_spared(
    db_session, owner, tmp_path, monkeypatch
) -> None:
    """A run-start bind that commits after the batch query but before the
    claim must win: the claim's predicate no longer matches, so the metadata
    row, local copy, and durable object survive because cleanup has not yet
    won the claim."""
    task_id = _make_task(db_session, owner)
    row, path = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="raced.txt",
        marker=TASKLESS_SHARE_UPLOAD_SOURCE,
        task_id=None,
        age_days=5,
    )
    row_id = int(row.id)

    real_claim = orphan_upload_gc._claim_orphan

    def bind_then_claim(db, candidate) -> datetime | None:
        # Emulate the concurrent run-start committing its bind first.
        db.query(UploadedFile).filter(UploadedFile.id == candidate.row_id).update(
            {UploadedFile.task_id: task_id}, synchronize_session=False
        )
        db.commit()
        return real_claim(db, candidate)

    monkeypatch.setattr(orphan_upload_gc, "_claim_orphan", bind_then_claim)

    result = cleanup_orphaned_taskless_uploads(db_session, older_than_seconds=2 * DAY)

    assert result.deleted == 0
    survivor = db_session.query(UploadedFile).filter_by(id=row_id).one()
    assert survivor.task_id == task_id
    assert survivor.storage_status == "available"  # never claimed
    assert path.exists()  # GC never obtained cleanup ownership


def test_gc_never_unlinks_an_external_shared_path(
    db_session, owner, tmp_path, monkeypatch
) -> None:
    row, path = _mk_upload(
        db_session,
        owner,
        tmp_path / "external",
        name="shared.txt",
        marker=None,
        task_id=None,
        age_days=30,
        detached_reason="task_deleted",
        detached_age_days=8,
        under_user_root=False,
    )
    row_id = int(row.id)
    monkeypatch.setattr(
        orphan_upload_gc,
        "delete_uploaded_file_compensation_object",
        lambda **_kwargs: "absent",
    )

    result = cleanup_detached_uploaded_files(
        db_session,
        older_than_seconds=7 * DAY,
    )

    assert (result.scanned, result.deleted) == (1, 1)
    assert db_session.query(UploadedFile).filter_by(id=row_id).first() is None
    assert path.exists()


def test_detached_gc_uses_detach_time_and_spares_unmarked_drafts(
    db_session, owner, tmp_path, monkeypatch
) -> None:
    old, old_path = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="old-detached.txt",
        marker=None,
        task_id=None,
        age_days=30,
        detached_reason="task_deleted",
        detached_age_days=8,
    )
    recent, recent_path = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="recent-detached.txt",
        marker=None,
        task_id=None,
        age_days=30,
        detached_reason="task_deleted",
        detached_age_days=1,
    )
    draft, draft_path = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="draft.txt",
        marker=None,
        task_id=None,
        age_days=30,
    )
    old_id, recent_id, draft_id = int(old.id), int(recent.id), int(draft.id)
    monkeypatch.setattr(
        orphan_upload_gc,
        "delete_uploaded_file_compensation_object",
        lambda **_kwargs: "absent",
    )

    result = cleanup_detached_uploaded_files(
        db_session,
        older_than_seconds=7 * DAY,
    )

    assert (result.scanned, result.deleted) == (1, 1)
    assert db_session.query(UploadedFile).filter_by(id=old_id).first() is None
    assert not old_path.exists()
    assert db_session.query(UploadedFile).filter_by(id=recent_id).first() is not None
    assert recent_path.exists()
    assert db_session.query(UploadedFile).filter_by(id=draft_id).first() is not None
    assert draft_path.exists()


def test_taskless_gc_never_bypasses_detached_retention_for_hybrid_rows(
    db_session, owner, tmp_path, monkeypatch
) -> None:
    hybrid, hybrid_path = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="recent-hybrid.txt",
        marker=TASKLESS_SHARE_UPLOAD_SOURCE,
        task_id=None,
        age_days=30,
        detached_reason="task_deleted",
        detached_age_days=1,
    )
    ordinary, ordinary_path = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="ordinary-taskless.txt",
        marker=TASKLESS_SHARE_UPLOAD_SOURCE,
        task_id=None,
        age_days=10,
    )
    hybrid_id, ordinary_id = int(hybrid.id), int(ordinary.id)
    monkeypatch.setattr(
        orphan_upload_gc,
        "delete_uploaded_file_compensation_object",
        lambda **_kwargs: "absent",
    )

    taskless = cleanup_orphaned_taskless_uploads(
        db_session,
        older_than_seconds=2 * DAY,
    )

    assert (taskless.scanned, taskless.deleted) == (1, 1)
    assert db_session.query(UploadedFile).filter_by(id=ordinary_id).first() is None
    assert not ordinary_path.exists()
    assert db_session.query(UploadedFile).filter_by(id=hybrid_id).first() is not None
    assert hybrid_path.exists()


def test_detached_claim_rejects_a_newer_detach_generation(
    db_session, owner, tmp_path
) -> None:
    row, _ = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="detached-again.txt",
        marker=None,
        task_id=None,
        age_days=30,
        detached_reason="task_deleted",
        detached_age_days=8,
    )
    stale_candidate = _candidate(row)
    newer_detached_at = datetime.now(timezone.utc)
    db_session.query(UploadedFile).filter_by(id=int(row.id)).update(
        {UploadedFile.detached_at: newer_detached_at},
        synchronize_session=False,
    )
    db_session.commit()

    assert _claim_orphan(db_session, stale_candidate) is None
    survivor = db_session.query(UploadedFile).filter_by(id=int(row.id)).one()
    assert survivor.storage_status == "available"


def test_claim_commit_failure_performs_no_physical_io(
    db_session, owner, tmp_path, monkeypatch
) -> None:
    row, path = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="commit-failed.txt",
        marker=None,
        task_id=None,
        age_days=30,
        detached_reason="task_deleted",
        detached_age_days=8,
    )
    durable_calls: list[str] = []
    monkeypatch.setattr(
        orphan_upload_gc,
        "delete_uploaded_file_compensation_object",
        lambda **_kwargs: durable_calls.append("delete") or "absent",
    )
    monkeypatch.setattr(
        db_session,
        "commit",
        lambda: (_ for _ in ()).throw(RuntimeError("commit failed")),
    )

    result = cleanup_detached_uploaded_files(
        db_session,
        older_than_seconds=7 * DAY,
    )

    assert (result.scanned, result.deleted) == (1, 0)
    assert durable_calls == []
    assert path.exists()


def test_pages_thread_the_cursor_and_drain_the_backlog(
    db_session, owner, tmp_path
) -> None:
    """One bounded page per call; the caller (the GC loop) threads
    ``next_cursor``, and a short page signals the end of the backlog."""
    for i in range(5):
        _mk_upload(
            db_session,
            owner,
            tmp_path,
            name=f"backlog-{i}.txt",
            marker=TASKLESS_SHARE_UPLOAD_SOURCE,
            task_id=None,
            age_days=5,
        )

    total_deleted = 0
    cursor = None
    pages = 0
    while True:
        result = cleanup_orphaned_taskless_uploads(
            db_session, older_than_seconds=2 * DAY, batch_size=2, after=cursor
        )
        total_deleted += result.deleted
        pages += 1
        assert result.scanned <= 2  # never more than one bounded page
        if result.scanned < 2:
            break
        cursor = result.next_cursor

    assert total_deleted == 5
    assert pages == 3
    assert db_session.query(UploadedFile).count() == 0


@pytest.mark.asyncio
async def test_tick_drains_full_pages_before_sleeping(
    db_session, owner, tmp_path
) -> None:
    """One tick keeps consuming FULL pages until a short page — the reaper's
    per-tick capacity is pages × batch, not one page per long sleep, so
    sustained task-less ingress cannot outrun it (#996 review). Exhausting
    the page budget instead returns a live cursor for the next tick."""
    for i in range(5):
        _mk_upload(
            db_session,
            owner,
            tmp_path,
            name=f"tick-{i}.txt",
            marker=TASKLESS_SHARE_UPLOAD_SOURCE,
            task_id=None,
            age_days=5,
        )

    # Plenty of budget: a single tick drains the whole backlog (3 pages)
    # and signals the drained state with a reset (None) cursor.
    cursor = await _run_gc_tick(
        ttl_seconds=2 * DAY,
        batch_size=2,
        max_pages=10,
        after=None,
        session_factory=get_session_local(),
    )
    assert cursor is None
    assert db_session.query(UploadedFile).count() == 0

    # Budget exhaustion on full pages: the tick stops early but hands the
    # next tick a live cursor to resume from, preserving fairness.
    for i in range(3):
        _mk_upload(
            db_session,
            owner,
            tmp_path,
            name=f"budget-{i}.txt",
            marker=TASKLESS_SHARE_UPLOAD_SOURCE,
            task_id=None,
            age_days=5,
        )
    cursor = await _run_gc_tick(
        ttl_seconds=2 * DAY,
        batch_size=1,
        max_pages=2,
        after=None,
        session_factory=get_session_local(),
    )
    assert cursor is not None
    assert db_session.query(UploadedFile).count() == 1

    cursor = await _run_gc_tick(
        ttl_seconds=2 * DAY,
        batch_size=1,
        max_pages=2,
        after=cursor,
        session_factory=get_session_local(),
    )
    assert cursor is None
    assert db_session.query(UploadedFile).count() == 0


@pytest.mark.asyncio
async def test_combined_tick_keeps_separate_fair_cursors(
    db_session, owner, tmp_path, monkeypatch
) -> None:
    for index in range(3):
        _mk_upload(
            db_session,
            owner,
            tmp_path,
            name=f"fair-taskless-{index}.txt",
            marker=TASKLESS_SHARE_UPLOAD_SOURCE,
            task_id=None,
            age_days=10 - index,
        )
        _mk_upload(
            db_session,
            owner,
            tmp_path,
            name=f"fair-detached-{index}.txt",
            marker=None,
            task_id=None,
            age_days=30,
            detached_reason="task_deleted",
            detached_age_days=10 - index,
        )
    monkeypatch.setattr(
        orphan_upload_gc,
        "delete_uploaded_file_compensation_object",
        lambda **_kwargs: "absent",
    )

    first = await orphan_upload_gc._run_combined_gc_tick(
        ttl_seconds=2 * DAY,
        detached_retention_seconds=7 * DAY,
        batch_size=1,
        max_pages=2,
        after=None,
        detached_after=None,
        session_factory=get_session_local(),
    )

    assert first.taskless.backlog is True
    assert first.detached.backlog is True
    assert first.taskless.cursor is not None
    assert first.detached.cursor is not None
    assert db_session.query(UploadedFile).count() == 4

    second = await orphan_upload_gc._run_combined_gc_tick(
        ttl_seconds=2 * DAY,
        detached_retention_seconds=7 * DAY,
        batch_size=1,
        max_pages=2,
        after=first.taskless.cursor,
        detached_after=first.detached.cursor,
        session_factory=get_session_local(),
    )

    assert second.taskless.cursor != first.taskless.cursor
    assert second.detached.cursor != first.detached.cursor
    assert db_session.query(UploadedFile).count() == 2


@pytest.mark.asyncio
async def test_loop_defers_long_sleep_until_backlog_is_drained(monkeypatch) -> None:
    """The long poll sleep is reserved for a drained backlog: while the tick
    returns a live cursor the loop continues after only a short breather, so
    drain throughput is not throttled by the poll interval (#996 review). A
    failed tick also takes the long sleep, so a persistent error cannot
    hot-loop."""
    ticks = iter(
        [
            (datetime.now(timezone.utc), 11),  # budget exhausted, backlog left
            (datetime.now(timezone.utc), 22),  # still draining
            RuntimeError("tick blew up"),  # failure -> long sleep
            None,  # short page -> drained -> long sleep
        ]
    )
    seen_cursors: list[OrphanUploadSweepCursor | None] = []

    async def fake_tick(*, after, **_kwargs):
        seen_cursors.append(after)
        outcome = next(ticks)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 4:
            raise asyncio.CancelledError

    monkeypatch.setattr(orphan_upload_gc, "_run_gc_tick", fake_tick)
    monkeypatch.setattr(orphan_upload_gc.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await orphan_upload_gc.run_orphan_upload_gc_loop(
            poll_interval_seconds=3600,
            ttl_seconds=2 * DAY,
            backlog_continue_delay_seconds=0.5,
        )

    assert sleeps == [0.5, 0.5, 3600, 3600]
    # The live cursor is threaded into the immediately following tick.
    assert seen_cursors[1] == (seen_cursors[1][0], 11)
    assert seen_cursors[2] == (seen_cursors[2][0], 22)


def test_cursor_advances_past_failing_rows(
    db_session, owner, tmp_path, monkeypatch
) -> None:
    """``next_cursor`` points past spared AND failing rows, so a permanently
    failing oldest page cannot starve newer orphans across ticks — the loop
    resumes AFTER the poison row on its next page (#996 review)."""
    oldest, _ = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="poison.txt",
        marker=TASKLESS_SHARE_UPLOAD_SOURCE,
        task_id=None,
        age_days=9,
    )
    newer, newer_path = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="newer.txt",
        marker=TASKLESS_SHARE_UPLOAD_SOURCE,
        task_id=None,
        age_days=5,
    )
    poison_id = int(oldest.id)
    newer_id = int(newer.id)

    real_reap = orphan_upload_gc._reap_orphan

    def poisoned_reap(db, candidate) -> bool:
        if candidate.row_id == poison_id:
            raise RuntimeError("undeletable row")
        return real_reap(db, candidate)

    monkeypatch.setattr(orphan_upload_gc, "_reap_orphan", poisoned_reap)

    first = cleanup_orphaned_taskless_uploads(
        db_session, older_than_seconds=2 * DAY, batch_size=1
    )
    assert (first.scanned, first.deleted) == (1, 0)  # poison page, no progress
    assert first.next_cursor is not None

    second = cleanup_orphaned_taskless_uploads(
        db_session, older_than_seconds=2 * DAY, batch_size=1, after=first.next_cursor
    )
    assert (second.scanned, second.deleted) == (1, 1)

    assert db_session.query(UploadedFile).filter_by(id=poison_id).first() is not None
    assert db_session.query(UploadedFile).filter_by(id=newer_id).first() is None
    assert not newer_path.exists()


def test_unresolved_durable_delete_hands_off_to_stale_recovery(
    db_session, owner, tmp_path, monkeypatch
) -> None:
    """When the durable delete cannot confirm ``absent``, GC must NOT settle
    the metadata (that could strand a live object): the row stays claimed and
    becomes a stale-compensation recovery candidate, which owns the retry."""
    row, path = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="deferred.txt",
        marker=TASKLESS_SHARE_UPLOAD_SOURCE,
        task_id=None,
        age_days=5,
    )
    row_id = int(row.id)

    monkeypatch.setattr(
        orphan_upload_gc,
        "delete_uploaded_file_compensation_object",
        lambda **_kwargs: "unknown",
    )
    result = cleanup_orphaned_taskless_uploads(db_session, older_than_seconds=2 * DAY)

    assert result.deleted == 0
    survivor = db_session.query(UploadedFile).filter_by(id=row_id).one()
    assert survivor.storage_status == "compensating"
    assert not path.exists()  # local copy already removed pre-claim

    # The claimed row is exactly what the generic recovery loop scans for.
    candidates = get_stale_uploaded_file_compensation_candidates(
        db_session,
        cutoff=datetime.now(timezone.utc) + timedelta(seconds=1),
        limit=10,
    )
    assert [c.row_id for c in candidates] == [row_id]

    # A later sweep does not fight the recovery loop for the claimed row.
    monkeypatch.undo()
    later = cleanup_orphaned_taskless_uploads(db_session, older_than_seconds=2 * DAY)
    assert (later.scanned, later.deleted) == (0, 0)


def test_reap_deletes_durable_object_via_compensation_helper(
    db_session, owner, tmp_path, monkeypatch
) -> None:
    """The durable object is removed through the session-less compensation
    delete (keyed by owner + exact storage key) — not via a status-gated path
    that a claimed row would silently skip (#996 review)."""
    row, _ = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="durable.txt",
        marker=TASKLESS_SHARE_UPLOAD_SOURCE,
        task_id=None,
        age_days=5,
    )
    expected_key = str(row.storage_key)
    seen: list[tuple[int, str]] = []

    def recording_delete(*, user_id: int, storage_key: str) -> str:
        seen.append((user_id, storage_key))
        return "absent"

    monkeypatch.setattr(
        orphan_upload_gc,
        "delete_uploaded_file_compensation_object",
        recording_delete,
    )
    result = cleanup_orphaned_taskless_uploads(db_session, older_than_seconds=2 * DAY)

    assert result.deleted == 1
    assert seen == [(int(owner.id), expected_key)]
