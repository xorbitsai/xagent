"""The durable record of external cleanup a task deletion still owes (#2587).

Runs on SQLite and, when ``XAGENT_TEST_POSTGRES_URL`` is set, on PostgreSQL
through the shared ``engine`` fixture. Everything here is observed through the
module's own interface -- ``list_cleanup_obligations`` and the batch report --
and through the resource itself (the directory on disk, the provider's calls),
never by reading the table directly.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from xagent.web.models.database import Base
from xagent.web.services.task_cleanup_obligations import (
    CLAIM_LEASE,
    CleanupObligationStatus,
    CleanupResourceKind,
    CleanupRetryReport,
    _claim_sync,
    _due_candidates_sync,
    extension_obligation,
    list_cleanup_obligations,
    record_cleanup_obligations_no_commit,
    run_cleanup_obligation_batch,
    run_cleanup_obligation_loop,
    settle_cleanup_attempt_no_commit,
    workspace_obligation,
)
from xagent.web.services.task_workspace_cleanup import WorkspaceCleanupTarget

engine = engine_fixture

NOW = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _uploads_root(tmp_path, monkeypatch) -> Path:
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads))
    monkeypatch.setenv("XAGENT_EXTERNAL_UPLOAD_DIRS", "")
    return uploads


@pytest.fixture
def sessions(engine) -> sessionmaker[Session]:
    """Sessions shaped like production's: ``autoflush=False``."""
    Base.metadata.create_all(engine)
    return sa.orm.sessionmaker(bind=engine, autoflush=False)


def _workspace_target(
    base: Path, task_id: int, owner_id: int = 7
) -> WorkspaceCleanupTarget:
    return WorkspaceCleanupTarget(
        task_id=task_id, owner_id=owner_id, base_dirs=(str(base),)
    )


def test_retry_backoff_doubles_then_caps_without_overflowing() -> None:
    """Five minutes, doubling: the default budget's delays add up to about ten
    and a half hours without reaching the six-hour cap, which the ninth
    attempt is the first to hit. A huge attempt count returns the cap rather
    than overflowing."""
    from xagent.web.services.task_cleanup_obligations import _retry_backoff

    expected_minutes = [5, 10, 20, 40, 80, 160, 320]
    for attempts, minutes in enumerate(expected_minutes, start=1):
        assert _retry_backoff(attempts) == timedelta(minutes=minutes)

    assert _retry_backoff(9) == timedelta(hours=6)
    assert _retry_backoff(100) == timedelta(hours=6)


def test_a_recorded_obligation_is_listed_until_it_is_completed(
    sessions, tmp_path
) -> None:
    with sessions() as db:
        [obligation] = record_cleanup_obligations_no_commit(
            db, [workspace_obligation(_workspace_target(tmp_path, 41))], now=NOW
        )
        db.commit()

    with sessions() as db:
        [listed] = list_cleanup_obligations(db)
        assert listed.task_id == 41
        assert listed.owner_id == 7
        assert listed.kind is CleanupResourceKind.WORKSPACE
        assert listed.status is CleanupObligationStatus.PENDING
        assert listed.attempts == 0

        assert settle_cleanup_attempt_no_commit(db, obligation, error=None) is None
        db.commit()

    with sessions() as db:
        assert list_cleanup_obligations(db) == []


def test_a_reused_task_id_owes_its_own_cleanup_next_to_the_old_one(
    sessions, tmp_path
) -> None:
    """SQLite hands a deleted task's id to the next task. When that task is
    deleted too, its obligation must be recorded and settled on its own -- not
    folded into, or settled against, what the earlier holder of the id left."""
    old_target = _workspace_target(tmp_path / "old", 42)
    with sessions() as db:
        [old] = record_cleanup_obligations_no_commit(
            db, [workspace_obligation(old_target, scope_resolved=False)], now=NOW
        )
        settle_cleanup_attempt_no_commit(db, old, error=None, now=NOW)
        db.commit()

    with sessions() as db:
        [new] = record_cleanup_obligations_no_commit(
            db, [workspace_obligation(_workspace_target(tmp_path / "new", 42))], now=NOW
        )
        db.commit()
    with sessions() as db:
        status = settle_cleanup_attempt_no_commit(
            db, new, error="OSError: busy", now=NOW
        )
        db.commit()

    assert status is CleanupObligationStatus.PENDING
    with sessions() as db:
        listed = {o.id: o for o in list_cleanup_obligations(db)}
    assert listed[old.id].status is CleanupObligationStatus.ABANDONED
    assert listed[new.id].status is CleanupObligationStatus.PENDING
    assert listed[new.id].locator["base_dirs"] == [str(tmp_path / "new")]


def test_an_inline_outcome_does_not_overwrite_a_drivers_claim(
    sessions, tmp_path, failing_workspace_removal
) -> None:
    """An inline removal that outlasted the lease: the driver claimed the row
    meanwhile, so the late inline outcome must not settle it."""
    import asyncio as _asyncio

    base = tmp_path / "user_7"
    _make_workspace(base, 44)
    with sessions() as db:
        [recorded] = record_cleanup_obligations_no_commit(
            db,
            [workspace_obligation(_workspace_target(base, 44))],
            now=NOW,
            inline_attempt=True,
        )
        db.commit()
    later = NOW + timedelta(hours=1)
    # The driver takes it once the lease has lapsed, and its release fails.
    failing_workspace_removal["failures"] = 1
    report = _asyncio.run(run_cleanup_obligation_batch(sessions, now=later))
    assert report.retrying == 1

    # The inline attempt finally reports success -- for attempt 0, long gone.
    with sessions() as db:
        status = settle_cleanup_attempt_no_commit(db, recorded, error=None, now=later)
        db.commit()

    assert status is CleanupObligationStatus.PENDING
    with sessions() as db:
        [owed] = list_cleanup_obligations(db)
        assert owed.attempts == 1
        assert "read-only" in (owed.last_error or "")


def test_an_obligation_recorded_in_a_rolled_back_transaction_does_not_exist(
    sessions, tmp_path
) -> None:
    """Same transaction as the row deletion: no deletion, no obligation."""
    with sessions() as db:
        record_cleanup_obligations_no_commit(
            db, [workspace_obligation(_workspace_target(tmp_path, 43))], now=NOW
        )
        db.rollback()

    with sessions() as db:
        assert list_cleanup_obligations(db) == []


def _make_workspace(base: Path, task_id: int) -> Path:
    workspace = base / f"web_task_{task_id}"
    (workspace / "output").mkdir(parents=True)
    (workspace / "output" / "result.txt").write_text("payload")
    return workspace


def _record(sessions, *obligations, inline_attempt=False) -> list:
    with sessions() as db:
        recorded = record_cleanup_obligations_no_commit(
            db, obligations, now=NOW, inline_attempt=inline_attempt
        )
        db.commit()
    return recorded


@pytest.mark.asyncio
async def test_the_driver_removes_a_workspace_whose_task_row_is_long_gone(
    sessions, tmp_path
) -> None:
    """The locator alone is enough: no task row ever existed for id 51."""
    base = tmp_path / "user_7"
    workspace = _make_workspace(base, 51)
    _record(sessions, workspace_obligation(_workspace_target(base, 51)))

    report = await run_cleanup_obligation_batch(sessions, now=NOW)

    assert not workspace.exists()
    assert report.completed == 1
    with sessions() as db:
        assert list_cleanup_obligations(db) == []


@pytest.mark.asyncio
async def test_a_workspace_that_is_already_gone_completes_its_obligation(
    sessions, tmp_path
) -> None:
    _record(sessions, workspace_obligation(_workspace_target(tmp_path / "user_7", 52)))

    report = await run_cleanup_obligation_batch(sessions, now=NOW)

    assert report.completed == 1
    with sessions() as db:
        assert list_cleanup_obligations(db) == []


@pytest.mark.asyncio
async def test_the_driver_leaves_an_obligation_that_is_not_yet_due(
    sessions, tmp_path
) -> None:
    """An inline attempt is in flight until the lease runs out."""
    base = tmp_path / "user_7"
    workspace = _make_workspace(base, 53)
    _record(
        sessions,
        workspace_obligation(_workspace_target(base, 53)),
        inline_attempt=True,
    )

    report = await run_cleanup_obligation_batch(sessions, now=NOW)

    assert workspace.exists()
    assert report.claimed == 0
    with sessions() as db:
        assert [o.task_id for o in list_cleanup_obligations(db)] == [53]


@pytest.fixture
def failing_workspace_removal(monkeypatch):
    """Make workspace removal fail ``failures`` times, then really remove."""
    from xagent.web.services import task_cleanup_obligations as module

    real_remove = module.remove_task_workspace
    state = {"failures": 0, "calls": 0}

    def _remove(target):
        state["calls"] += 1
        if state["failures"] > 0:
            state["failures"] -= 1
            raise PermissionError("workspace volume is read-only")
        real_remove(target)

    monkeypatch.setattr(module, "remove_task_workspace", _remove)
    return state


@pytest.mark.asyncio
async def test_a_transient_failure_is_retried_later_and_then_completes(
    sessions, tmp_path, failing_workspace_removal
) -> None:
    base = tmp_path / "user_7"
    workspace = _make_workspace(base, 61)
    _record(sessions, workspace_obligation(_workspace_target(base, 61)))
    failing_workspace_removal["failures"] = 1

    first = await run_cleanup_obligation_batch(sessions, now=NOW)

    assert first.retrying == 1
    assert workspace.exists()
    with sessions() as db:
        [pending] = list_cleanup_obligations(db)
        assert pending.status is CleanupObligationStatus.PENDING
        assert pending.attempts == 1
        assert "read-only" in (pending.last_error or "")

    # Not retried before its backoff has passed...
    early = await run_cleanup_obligation_batch(sessions, now=NOW + timedelta(minutes=1))
    assert early.claimed == 0

    # ...and completed once it has.
    later = await run_cleanup_obligation_batch(sessions, now=NOW + timedelta(hours=1))
    assert later.completed == 1
    assert not workspace.exists()
    with sessions() as db:
        assert list_cleanup_obligations(db) == []


@pytest.mark.asyncio
async def test_a_candidate_already_at_the_attempt_budget_is_exhausted_without_a_release(
    sessions, tmp_path, monkeypatch
) -> None:
    """A release that hangs, or a process that dies on every attempt before it
    can write an outcome, would otherwise be reclaimed every ``CLAIM_LEASE``
    forever: the outcome-side budget check never runs for an attempt that
    never finishes. The claim step itself must refuse to hand out a claim
    once the budget is already spent."""
    from xagent.web.services import task_cleanup_obligations as module

    base = tmp_path / "user_7"
    _make_workspace(base, 65)
    with sessions() as db:
        [recorded] = record_cleanup_obligations_no_commit(
            db, [workspace_obligation(_workspace_target(base, 65))], now=NOW
        )
        db.commit()
    # Simulate two prior claims that never reported back: attempts sits at 2
    # already, with a due (lapsed-lease) next_attempt_at.
    with sessions() as db:
        db.execute(
            sa.update(module.TaskCleanupObligation)
            .where(module.TaskCleanupObligation.id == recorded.id)
            .values(attempts=2, next_attempt_at=NOW)
        )
        db.commit()

    calls = {"n": 0}
    real_remove = module.remove_task_workspace

    def _counting_remove(target):
        calls["n"] += 1
        real_remove(target)

    monkeypatch.setattr(module, "remove_task_workspace", _counting_remove)

    report = await run_cleanup_obligation_batch(sessions, now=NOW, max_attempts=2)

    assert calls["n"] == 0
    assert report.exhausted == 1
    with sessions() as db:
        [terminal] = list_cleanup_obligations(
            db, statuses=[CleanupObligationStatus.EXHAUSTED]
        )
        assert terminal.attempts == 2
        assert "without a recorded outcome" in (terminal.last_error or "")


@pytest.mark.asyncio
async def test_an_obligation_that_keeps_failing_ends_on_the_reconciliation_list(
    sessions, tmp_path, failing_workspace_removal
) -> None:
    base = tmp_path / "user_7"
    _make_workspace(base, 62)
    _record(sessions, workspace_obligation(_workspace_target(base, 62)))
    failing_workspace_removal["failures"] = 100

    when = NOW
    for _ in range(3):
        await run_cleanup_obligation_batch(sessions, now=when, max_attempts=3)
        when += timedelta(days=1)
    after_budget = await run_cleanup_obligation_batch(
        sessions, now=when, max_attempts=3
    )

    assert after_budget.claimed == 0
    assert failing_workspace_removal["calls"] == 3
    with sessions() as db:
        [terminal] = list_cleanup_obligations(
            db,
            statuses=[
                CleanupObligationStatus.EXHAUSTED,
                CleanupObligationStatus.ABANDONED,
            ],
        )
        assert terminal.task_id == 62
        assert terminal.status is CleanupObligationStatus.EXHAUSTED
        assert terminal.attempts == 3


@pytest.mark.asyncio
async def test_a_reused_task_id_is_never_cleaned_up_under_its_new_owner(
    sessions, tmp_path
) -> None:
    """SQLite hands a deleted task's id to the next task. The obligation left
    by the old one must not delete the new one's workspace."""
    from xagent.web.models.task import Task
    from xagent.web.models.user import User

    base = tmp_path / "user_7"
    _record(sessions, workspace_obligation(_workspace_target(base, 63)))
    with sessions() as db:
        user = User(username="reuse-owner", password_hash="unused")
        db.add(user)
        db.flush()
        db.add(Task(id=63, user_id=int(user.id), title="new owner of id 63"))
        db.commit()
    live_workspace = _make_workspace(base, 63)

    report = await run_cleanup_obligation_batch(sessions, now=NOW)

    assert report.abandoned == 1
    assert live_workspace.exists()
    with sessions() as db:
        [abandoned] = list_cleanup_obligations(db)
        assert abandoned.status is CleanupObligationStatus.ABANDONED


@pytest.mark.asyncio
async def test_a_task_id_reused_during_the_release_itself_is_logged_not_fenced(
    sessions, tmp_path, caplog, monkeypatch
) -> None:
    """The live-task check races only the release itself (``rmtree``, the
    provider round trip), not "one query" -- a task can be created any time
    between the pre-release check and the release returning. That window is
    not fenced (doing so would need a lock this module does not take), but a
    task created inside it deserves an operator's attention: log it, with
    ids, rather than silently letting the release's own outcome stand as if
    nothing had happened. The recorded outcome itself is unchanged -- this is
    an observability addition, not a new abandonment path."""
    import logging

    from xagent.web.services import task_cleanup_obligations as module

    base = tmp_path / "user_7"
    workspace = _make_workspace(base, 67)
    _record(sessions, workspace_obligation(_workspace_target(base, 67)))

    live_calls = {"n": 0}

    def _goes_live_during_the_release(session_factory, task_id):
        live_calls["n"] += 1
        if live_calls["n"] == 1:
            return False  # pre-release check: no live task yet.
        return True  # post-release check: a task now owns this id.

    monkeypatch.setattr(module, "_task_id_is_live_sync", _goes_live_during_the_release)
    with caplog.at_level(
        logging.ERROR, logger="xagent.web.services.task_cleanup_obligations"
    ):
        report = await run_cleanup_obligation_batch(sessions, now=NOW)

    # The release still ran and completed normally -- outcome unchanged.
    assert not workspace.exists()
    assert report.completed == 1
    with sessions() as db:
        assert list_cleanup_obligations(db) == []

    reused = [r for r in caplog.records if "may have touched" in r.message]
    assert len(reused) == 1
    assert "67" in reused[0].message


def test_claim_sync_is_a_compare_and_set(sessions, tmp_path) -> None:
    """The same due-candidate snapshot must be able to win the claim exactly
    once. This is the primitive ``run_cleanup_obligation_batch`` relies on to
    let two web replicas race a claim safely; exercise it directly rather
    than only through the batch, which hides whether the second call was
    refused by the CAS or simply never attempted."""
    _record(sessions, workspace_obligation(_workspace_target(tmp_path, 94)))

    [candidate] = _due_candidates_sync(sessions, now=NOW, limit=10)

    first = _claim_sync(sessions, candidate, now=NOW, max_attempts=8)
    second = _claim_sync(sessions, candidate, now=NOW, max_attempts=8)

    assert first is not None
    assert first.attempts == candidate.attempts + 1
    assert second is None


@pytest.mark.asyncio
async def test_a_claim_that_never_settles_is_reclaimed_once_its_lease_lapses(
    sessions, tmp_path, failing_workspace_removal
) -> None:
    """A process that claims an obligation and then dies (or hangs) before it
    can write any outcome must not strand the obligation forever: once
    ``CLAIM_LEASE`` has passed, the row is due again and a fresh batch can
    reclaim it, with ``attempts`` reflecting both the abandoned claim and the
    new one."""
    base = tmp_path / "user_7"
    _make_workspace(base, 95)
    _record(sessions, workspace_obligation(_workspace_target(base, 95)))

    [candidate] = _due_candidates_sync(sessions, now=NOW, limit=10)
    claimed = _claim_sync(sessions, candidate, now=NOW, max_attempts=8)
    assert claimed is not None
    assert claimed.attempts == 1
    # No settle follows -- this simulates a process that died mid-release.

    # The next release also fails, so the row stays pending afterwards and
    # ``attempts`` can be observed rather than the row being discharged.
    failing_workspace_removal["failures"] = 1
    later = NOW + CLAIM_LEASE
    report = await run_cleanup_obligation_batch(sessions, now=later, max_attempts=8)

    assert report.retrying == 1
    with sessions() as db:
        [pending] = list_cleanup_obligations(db)
        assert pending.status is CleanupObligationStatus.PENDING
        assert pending.attempts == 2


@pytest.mark.asyncio
async def test_an_obligation_in_flight_is_not_claimed_by_a_second_driver(
    sessions, tmp_path, monkeypatch
) -> None:
    """Two web replicas each run a driver. While one is releasing an
    obligation, the other's batch must not take the same row."""
    import asyncio as _asyncio

    from xagent.web.services import task_cleanup_obligations as module

    base = tmp_path / "user_7"
    _make_workspace(base, 64)
    _record(sessions, workspace_obligation(_workspace_target(base, 64)))
    real_remove = module.remove_task_workspace
    competing: list = []

    def _remove_while_another_driver_runs(target):
        competing.append(_asyncio.run(run_cleanup_obligation_batch(sessions, now=NOW)))
        real_remove(target)

    monkeypatch.setattr(
        module, "remove_task_workspace", _remove_while_another_driver_runs
    )

    report = await run_cleanup_obligation_batch(sessions, now=NOW)

    assert report.completed == 1
    assert [other.claimed for other in competing] == [0]


class _Provider:
    """A runtime extension whose release fails ``failures`` times first."""

    def __init__(self, *, failures: int = 0) -> None:
        self.failures = failures
        self.deleted: list[tuple[int, int, str | None]] = []

    async def on_task_created(self, context, configuration) -> None:
        return None

    async def build_runtime(self, context):
        return None

    async def public_metadata(self, context):
        return None

    async def on_task_deleted(self, context) -> None:
        self.deleted.append((context.task_id, context.user_id, context.source))
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("provider is down")


@pytest.fixture
def provider():
    from xagent.web.services.task_runtime import (
        register_task_extension,
        unregister_task_extension,
    )

    instance = _Provider()
    register_task_extension("sandbox_lease", instance)
    yield instance
    unregister_task_extension("sandbox_lease")


@pytest.mark.asyncio
async def test_an_extension_is_released_after_its_task_row_is_gone(
    sessions, provider
) -> None:
    """The provider context is rebuilt from the locator alone."""
    provider.failures = 1
    _record(
        sessions,
        extension_obligation(
            task_id=71, user_id=7, source="webhook", extension="sandbox_lease"
        ),
    )

    first = await run_cleanup_obligation_batch(sessions, now=NOW)
    later = await run_cleanup_obligation_batch(sessions, now=NOW + timedelta(hours=1))

    assert first.retrying == 1
    assert later.completed == 1
    assert provider.deleted == [(71, 7, "webhook"), (71, 7, "webhook")]
    with sessions() as db:
        assert list_cleanup_obligations(db) == []


@pytest.mark.asyncio
async def test_an_extension_that_is_no_longer_registered_stays_owed(
    sessions,
) -> None:
    """Dropping the obligation because the provider was unloaded would lose the
    only record that its state still exists somewhere."""
    _record(
        sessions,
        extension_obligation(
            task_id=72, user_id=7, source=None, extension="unloaded_provider"
        ),
    )

    report = await run_cleanup_obligation_batch(sessions, now=NOW)

    assert report.retrying == 1
    with sessions() as db:
        [owed] = list_cleanup_obligations(db)
        assert owed.kind is CleanupResourceKind.RUNTIME_EXTENSION
        assert owed.key == "unloaded_provider"
        assert "not registered" in (owed.last_error or "")


@pytest.mark.asyncio
async def test_an_unregistered_extension_never_exhausts_its_budget(sessions) -> None:
    """A provider that is not registered here is not a failure of *this*
    obligation -- it is a fact about this process, which may not be the one
    that eventually has the provider loaded. Contradicting the design (a
    binding whose provider is not registered stays owed until one that can
    release it is), spending the attempt budget on it would let it go
    EXHAUSTED before the right process ever gets a turn. A small
    ``max_attempts`` here means a normal failure would exhaust in a batch or
    two; this must not, no matter how many batches run."""
    _record(
        sessions,
        extension_obligation(
            task_id=74, user_id=7, source=None, extension="never_loaded_here"
        ),
    )

    when = NOW
    for _ in range(6):
        report = await run_cleanup_obligation_batch(sessions, now=when, max_attempts=2)
        assert report.retrying == 1
        when += timedelta(hours=1, minutes=1)

    with sessions() as db:
        [owed] = list_cleanup_obligations(db)
        assert owed.status is CleanupObligationStatus.PENDING
        assert owed.attempts == 0
        assert "not registered" in (owed.last_error or "")


def test_an_abandoned_extension_is_listed_but_never_retried(sessions) -> None:
    """What an admin's force delete leaves behind: on the list, off the queue."""
    _record(
        sessions,
        extension_obligation(
            task_id=73,
            user_id=7,
            source=None,
            extension="sandbox_lease",
            status=CleanupObligationStatus.ABANDONED,
            reason="force delete: provider is down",
        ),
    )

    import asyncio as _asyncio

    report = _asyncio.run(run_cleanup_obligation_batch(sessions, now=NOW))

    assert report.claimed == 0
    with sessions() as db:
        [listed] = list_cleanup_obligations(db)
        assert listed.status is CleanupObligationStatus.ABANDONED
        assert listed.last_error == "force delete: provider is down"


def test_a_failed_inline_attempt_hands_the_obligation_to_the_driver(
    sessions, tmp_path
) -> None:
    [obligation] = _record(
        sessions,
        workspace_obligation(_workspace_target(tmp_path, 81)),
        inline_attempt=True,
    )

    with sessions() as db:
        status = settle_cleanup_attempt_no_commit(
            db, obligation, error="OSError: busy", now=NOW
        )
        db.commit()

    assert status is CleanupObligationStatus.PENDING
    with sessions() as db:
        [pending] = list_cleanup_obligations(db)
        assert pending.attempts == 1
        assert pending.last_error == "OSError: busy"


def test_exhaustion_is_logged_with_ids_only_once_the_write_actually_lands(
    sessions, tmp_path, caplog
) -> None:
    """The "gave up" line names the obligation, and fires only for a write
    the fence let through -- a stale outcome for a row that has moved on logs
    nothing."""
    import logging

    [obligation] = _record(
        sessions, workspace_obligation(_workspace_target(tmp_path, 66))
    )

    with caplog.at_level(
        logging.ERROR, logger="xagent.web.services.task_cleanup_obligations"
    ):
        with sessions() as db:
            status = settle_cleanup_attempt_no_commit(
                db, obligation, error="OSError: still busy", now=NOW, max_attempts=1
            )
            db.commit()

    assert status is CleanupObligationStatus.EXHAUSTED
    gave_up = [r for r in caplog.records if "gave up" in r.message]
    assert len(gave_up) == 1
    message = gave_up[0].message
    assert str(obligation.id) in message
    assert str(obligation.task_id) in message
    assert obligation.kind.value in message
    assert "1 attempt" in message

    caplog.clear()
    # A second, stale settle for the same (already-terminal) obligation must
    # not write anything -- and therefore must not log "gave up" again either.
    with caplog.at_level(
        logging.ERROR, logger="xagent.web.services.task_cleanup_obligations"
    ):
        with sessions() as db:
            settle_cleanup_attempt_no_commit(
                db, obligation, error="OSError: still busy", now=NOW, max_attempts=1
            )
            db.commit()
    assert not [r for r in caplog.records if "gave up" in r.message]


def test_an_unresolved_scope_is_never_reported_as_cleaned(sessions, tmp_path) -> None:
    """Clearing the unscoped candidates cannot prove the scoped workspace is
    gone, so the obligation ends on the operator's list instead."""
    [obligation] = _record(
        sessions,
        workspace_obligation(_workspace_target(tmp_path, 82), scope_resolved=False),
    )

    with sessions() as db:
        status = settle_cleanup_attempt_no_commit(db, obligation, error=None, now=NOW)
        db.commit()

    assert status is CleanupObligationStatus.ABANDONED
    with sessions() as db:
        [abandoned] = list_cleanup_obligations(
            db, statuses=[CleanupObligationStatus.ABANDONED]
        )
        assert abandoned.task_id == 82


async def _wait_until(predicate, *, timeout: float = 5.0) -> None:
    import asyncio as _asyncio

    deadline = _asyncio.get_running_loop().time() + timeout
    while not predicate():
        if _asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached before the timeout")
        await _asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_the_loop_works_off_what_is_due_and_keeps_running(
    sessions, tmp_path
) -> None:
    import asyncio as _asyncio

    base = tmp_path / "user_7"
    first = _make_workspace(base, 91)
    with sessions() as db:
        record_cleanup_obligations_no_commit(
            db, [workspace_obligation(_workspace_target(base, 91))]
        )
        db.commit()

    loop = _asyncio.create_task(
        run_cleanup_obligation_loop(sessions, poll_interval_seconds=0.01)
    )
    try:
        await _wait_until(lambda: not first.exists())

        # Still running: an obligation recorded later is picked up too.
        second = _make_workspace(base, 92)
        with sessions() as db:
            record_cleanup_obligations_no_commit(
                db, [workspace_obligation(_workspace_target(base, 92))]
            )
            db.commit()
        await _wait_until(lambda: not second.exists())
    finally:
        loop.cancel()
        with pytest.raises(_asyncio.CancelledError):
            await loop


@pytest.mark.asyncio
async def test_a_failing_batch_does_not_stop_the_loop(
    sessions, tmp_path, monkeypatch
) -> None:
    import asyncio as _asyncio

    from xagent.web.services import task_cleanup_obligations as module

    real_batch = module.run_cleanup_obligation_batch
    calls = {"n": 0}

    async def _flaky_batch(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("database went away")
        return await real_batch(*args, **kwargs)

    monkeypatch.setattr(module, "run_cleanup_obligation_batch", _flaky_batch)
    base = tmp_path / "user_7"
    workspace = _make_workspace(base, 93)
    with sessions() as db:
        record_cleanup_obligations_no_commit(
            db, [workspace_obligation(_workspace_target(base, 93))]
        )
        db.commit()

    loop = _asyncio.create_task(
        run_cleanup_obligation_loop(sessions, poll_interval_seconds=0.01)
    )
    try:
        await _wait_until(lambda: not workspace.exists())
    finally:
        loop.cancel()
        with pytest.raises(_asyncio.CancelledError):
            await loop
    assert calls["n"] >= 2


@pytest.mark.asyncio
async def test_the_loop_treats_a_full_due_page_as_backlog_even_when_claims_lag(
    monkeypatch,
) -> None:
    """Several replicas can split one due page: each claims only some of it,
    so ``claimed`` alone never reaches ``batch_size`` even though there is
    plainly more work due right now. ``due`` -- how many candidates the batch
    fetched -- is what should decide the pause, not how many this replica
    happened to win."""
    import asyncio as _asyncio

    from xagent.web.services import task_cleanup_obligations as module

    batch_size = 5
    sleeps: list[float] = []

    async def _mostly_claimed_elsewhere(*args, **kwargs):
        return CleanupRetryReport(due=batch_size, claimed=1, completed=1)

    async def _recording_sleep(seconds, *args, **kwargs):
        sleeps.append(seconds)
        raise _asyncio.CancelledError

    monkeypatch.setattr(
        module, "run_cleanup_obligation_batch", _mostly_claimed_elsewhere
    )
    monkeypatch.setattr(module.asyncio, "sleep", _recording_sleep)

    # ``poll_interval_seconds`` is deliberately huge: if the loop mistakenly
    # fell back to it instead of ``backlog_pause_seconds``, the recorded sleep
    # would be 1000.0 rather than the short backlog pause.
    with pytest.raises(_asyncio.CancelledError):
        await run_cleanup_obligation_loop(
            lambda: None,  # never used: the batch is replaced above
            poll_interval_seconds=1000.0,
            batch_size=batch_size,
            backlog_pause_seconds=0.01,
        )

    assert sleeps == [0.01]


@pytest.mark.asyncio
async def test_the_loop_does_not_hot_loop_when_every_claim_fails(
    monkeypatch,
) -> None:
    """A full due page with nothing actually claimed (e.g. a DB outage on the
    claim step) must not spin the loop at ``backlog_pause_seconds`` forever --
    ``claimed > 0`` guards the backlog path for exactly this case."""
    import asyncio as _asyncio

    from xagent.web.services import task_cleanup_obligations as module

    batch_size = 5
    calls = {"n": 0}
    sleeps: list[float] = []

    async def _nothing_claimed(*args, **kwargs):
        calls["n"] += 1
        return CleanupRetryReport(due=batch_size, claimed=0)

    async def _recording_sleep(seconds, *args, **kwargs):
        sleeps.append(seconds)
        raise _asyncio.CancelledError

    monkeypatch.setattr(module, "run_cleanup_obligation_batch", _nothing_claimed)
    monkeypatch.setattr(module.asyncio, "sleep", _recording_sleep)

    with pytest.raises(_asyncio.CancelledError):
        await run_cleanup_obligation_loop(
            lambda: None,  # never used: the batch is replaced above
            poll_interval_seconds=1000.0,
            batch_size=batch_size,
            backlog_pause_seconds=0.01,
        )

    assert sleeps == [1000.0]
