"""Per-team retention period overrides (#2600).

The deployment layer registers a resolver keyed by ``user_id``; the purge
applies it at both the batch scan and each task's locked assessment. Runs on
SQLite and PostgreSQL through the shared ``engine`` fixture, like the purge's
own row-level tests.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.user import User
from xagent.web.services import task_retention_overrides as overrides_module
from xagent.web.services.task_retention_overrides import (
    INHERIT,
    RETENTION_OVERRIDE_MAX_DAYS,
    RetentionOverride,
    RetentionPeriods,
    load_retention_overrides,
    resolve_task_retention_periods,
    set_retention_override_resolver,
)
from xagent.web.services.task_retention_purge import (
    RetentionPurgeAction,
    RetentionPurgeReport,
    purge_task,
    run_retention_purge_batch,
    select_purge_candidates,
)

engine = engine_fixture

NOW = datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc)
DEFAULTS = RetentionPeriods(conversation_days=365, trace_days=90)


@pytest.fixture(autouse=True)
def _no_resolver_leaks():
    """The resolver is process-wide; never let one test's leak into another."""
    set_retention_override_resolver(None)
    yield
    set_retention_override_resolver(None)


@pytest.fixture
def sessions(engine) -> sessionmaker[Session]:
    Base.metadata.create_all(engine)
    return sa.orm.sessionmaker(bind=engine, autoflush=False)


def _age(days: int) -> datetime:
    return NOW - timedelta(days=days)


def _seed(db: Session, *, username: str, age_days: int, trace: bool = True) -> int:
    """A terminal task of one fresh user, anchored ``age_days`` ago."""
    user = User(username=username, password_hash="unused")
    db.add(user)
    db.flush()
    task = Task(
        user_id=user.id,
        title="retention override fixture",
        status=TaskStatus.COMPLETED,
        last_activity_at=_age(age_days),
    )
    db.add(task)
    db.flush()
    if trace:
        db.add(
            TraceEvent(
                task_id=task.id,
                event_id=f"evt-{task.id}",
                event_type="agent_execution_checkpoint",
                timestamp=NOW,
                data={},
            )
        )
    db.commit()
    return int(task.id)


def _user_of(sessions: sessionmaker[Session], task_id: int) -> int:
    with sessions() as db:
        return int(
            db.execute(sa.select(Task.user_id).where(Task.id == task_id)).scalar_one()
        )


def _state(sessions: sessionmaker[Session], task_id: int) -> str:
    """``gone``, ``traces_gone`` or ``intact``."""
    with sessions() as db:
        if db.get(Task, task_id) is None:
            return "gone"
        traces = db.execute(
            sa.select(sa.func.count())
            .select_from(TraceEvent)
            .where(TraceEvent.task_id == task_id)
        ).scalar_one()
        return "intact" if traces else "traces_gone"


def _batch(sessions: sessionmaker[Session], **kwargs: object) -> RetentionPurgeReport:
    """A batch on :data:`DEFAULTS`, bypassing the PostgreSQL-only gate.

    The gate is covered in ``test_task_retention_purge.py``; the row semantics
    here must hold on both dialects.
    """
    import xagent.web.services.task_retention_purge as purge_module

    original = purge_module.ensure_retention_purge_supported
    purge_module.ensure_retention_purge_supported = lambda db: None  # type: ignore[assignment]
    try:
        kwargs.setdefault("periods", (DEFAULTS.conversation_days, DEFAULTS.trace_days))
        return run_retention_purge_batch(
            sessions,
            now=NOW,
            limit=100,
            dry_run=False,
            **kwargs,  # type: ignore[arg-type]
        )
    finally:
        purge_module.ensure_retention_purge_supported = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Unregistered: exactly today's behaviour.
# ---------------------------------------------------------------------------


def test_unregistered_resolves_the_configured_periods_without_a_query(sessions) -> None:
    with sessions() as db:
        task_id = _seed(db, username="u0", age_days=1)
    statements: list[str] = []
    with sessions() as db:
        sa.event.listen(
            db.get_bind(),
            "before_cursor_execute",
            lambda *args: statements.append(args[2]),
        )
        assert resolve_task_retention_periods(db, task_id, DEFAULTS) == DEFAULTS
    assert statements == []


def test_unregistered_scan_is_the_global_scan(sessions) -> None:
    """Not just the same rows: the same SQL, so nothing about the plan moves."""
    with sessions() as db:
        snapshot = load_retention_overrides(db, DEFAULTS)
        assert snapshot is not None
        assert snapshot.periods_by_user == {} and snapshot.refused_users == frozenset()

        def due(periods: RetentionPeriods) -> sa.ColumnElement[bool]:
            return Task.id > (periods.conversation_days or 0)

        assert str(snapshot.scan_condition(Task.user_id, due)) == str(due(DEFAULTS))


# ---------------------------------------------------------------------------
# Resolution rules.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        (RetentionOverride(conversation_days=30), RetentionPeriods(30, 90)),
        (RetentionOverride(trace_days=7), RetentionPeriods(365, 7)),
        (RetentionOverride(730, 730), RetentionPeriods(730, 730)),
        (RetentionOverride(), DEFAULTS),
    ],
)
def test_an_override_replaces_only_the_legs_it_sets(
    sessions, override, expected
) -> None:
    set_retention_override_resolver(lambda db: {7: override})
    with sessions() as db:
        snapshot = load_retention_overrides(db, DEFAULTS)
    assert snapshot is not None
    assert snapshot.periods_for(7) == expected
    assert snapshot.periods_for(8) == DEFAULTS


@pytest.mark.parametrize(
    "value",
    [None, 0, -1, RETENTION_OVERRIDE_MAX_DAYS + 1, True, 30.0, "30"],
)
def test_an_unusable_value_refuses_that_user_only(sessions, value) -> None:
    """Not clamped, not treated as inherit, not treated as unlimited."""
    set_retention_override_resolver(
        lambda db: {
            7: RetentionOverride(conversation_days=value),  # type: ignore[arg-type]
            8: RetentionOverride(conversation_days=30),
        }
    )
    with sessions() as db:
        snapshot = load_retention_overrides(db, DEFAULTS)
    assert snapshot is not None
    assert snapshot.periods_for(7) is None
    assert snapshot.periods_for(8) == RetentionPeriods(30, 90)


def test_an_unusable_value_refuses_the_user_even_on_a_leg_the_deployment_left_off(
    sessions,
) -> None:
    """``_apply`` must check usability before it checks whether the leg is on.

    The deployment here has no trace period at all (``trace_days=None``); a
    resolver that sets ``trace_days=0`` is unusable regardless. Checking
    ``default is None`` first would treat the off leg as nothing to validate
    and let the unusable ``0`` pass silently as "leg stays off", refusing
    nobody -- exactly the leak "on a leg the deployment left off" exists to
    close.
    """
    defaults = RetentionPeriods(conversation_days=365, trace_days=None)
    set_retention_override_resolver(lambda db: {7: RetentionOverride(trace_days=0)})
    with sessions() as db:
        snapshot = load_retention_overrides(db, defaults)
    assert snapshot is not None
    assert snapshot.periods_for(7) is None


@pytest.mark.parametrize(
    ("defaults", "override", "expected"),
    [
        # Trace unset in the environment: the deployment's trace period *is*
        # its conversation period, so an inherited trace follows the team's.
        (
            RetentionPeriods(365, 365),
            RetentionOverride(730),
            RetentionPeriods(730, 730),
        ),
        (RetentionPeriods(365, 365), RetentionOverride(30), RetentionPeriods(30, 30)),
        # A trace period of its own stays what the deployment configured.
        (RetentionPeriods(365, 90), RetentionOverride(730), RetentionPeriods(730, 90)),
        (
            RetentionPeriods(365, 500),
            RetentionOverride(730),
            RetentionPeriods(730, 500),
        ),
        (RetentionPeriods(None, 90), RetentionOverride(30), RetentionPeriods(None, 90)),
    ],
)
def test_an_inherited_trace_period_follows_a_trace_that_follows_the_conversation(
    sessions, defaults, override, expected
) -> None:
    set_retention_override_resolver(lambda db: {7: override})
    with sessions() as db:
        snapshot = load_retention_overrides(db, defaults)
    assert snapshot is not None and snapshot.periods_for(7) == expected


def test_extending_the_conversation_keeps_a_following_trace_too(sessions) -> None:
    """Deployment sets only the conversation period; a team extends it."""
    with sessions() as db:
        task_id = _seed(db, username="follow", age_days=400)
    user_id = _user_of(sessions, task_id)
    set_retention_override_resolver(lambda db: {user_id: RetentionOverride(730)})

    report = _batch(sessions, periods=(365, 365))

    assert _state(sessions, task_id) == "intact"
    assert report.purged == 0


def test_an_override_never_enables_a_leg_the_deployment_left_off(sessions) -> None:
    trace_only = RetentionPeriods(conversation_days=None, trace_days=90)
    set_retention_override_resolver(lambda db: {7: RetentionOverride(30, 7)})
    with sessions() as db:
        snapshot = load_retention_overrides(db, trace_only)
    assert snapshot is not None
    assert snapshot.periods_for(7) == RetentionPeriods(None, 7)


@pytest.mark.parametrize(
    "result",
    [
        None,
        [(7, RetentionOverride())],
        {"7": RetentionOverride()},
        {True: RetentionOverride()},
    ],
)
def test_a_malformed_result_fails_the_whole_resolution(sessions, result) -> None:
    set_retention_override_resolver(lambda db: result)
    with sessions() as db:
        assert load_retention_overrides(db, DEFAULTS) is None


def test_a_value_of_the_wrong_type_refuses_that_user(sessions) -> None:
    set_retention_override_resolver(lambda db: {7: {"conversation_days": 30}})
    with sessions() as db:
        snapshot = load_retention_overrides(db, DEFAULTS)
    assert snapshot is not None and snapshot.periods_for(7) is None


def test_a_raising_resolver_fails_closed_and_logs(sessions, caplog) -> None:
    def broken(db: Session) -> dict[int, RetentionOverride]:
        raise RuntimeError("settings table unreachable")

    set_retention_override_resolver(broken)
    with caplog.at_level(logging.WARNING, logger=overrides_module.__name__):
        with sessions() as db:
            assert load_retention_overrides(db, DEFAULTS) is None
            assert resolve_task_retention_periods(db, 1, DEFAULTS) is None
    assert "settings table unreachable" in caplog.text


def test_registering_a_non_callable_is_refused() -> None:
    with pytest.raises(TypeError):
        set_retention_override_resolver(30)  # type: ignore[arg-type]


def test_inherit_is_the_default_and_is_not_none() -> None:
    assert RetentionOverride().conversation_days is INHERIT
    assert INHERIT is not None


# ---------------------------------------------------------------------------
# End to end through run_retention_purge_batch.
# ---------------------------------------------------------------------------


def test_a_shorter_and_a_longer_team_period_each_change_only_their_own_tasks(
    sessions,
) -> None:
    with sessions() as db:
        shorter = _seed(db, username="short", age_days=60)
        longer_conversation = _seed(db, username="long-c", age_days=400)
        longer_trace = _seed(db, username="long-t", age_days=120)
        default_conversation = _seed(db, username="default-c", age_days=400)
        default_trace = _seed(db, username="default-t", age_days=120)
        untouched = _seed(db, username="default-young", age_days=60)
    short_user = _user_of(sessions, shorter)
    long_users = {
        _user_of(sessions, longer_conversation),
        _user_of(sessions, longer_trace),
    }
    set_retention_override_resolver(
        lambda db: {
            short_user: RetentionOverride(conversation_days=30, trace_days=30),
            **{user: RetentionOverride(730, 700) for user in long_users},
        }
    )

    report = _batch(sessions)

    assert _state(sessions, shorter) == "gone"
    assert _state(sessions, longer_conversation) == "intact"
    assert _state(sessions, longer_trace) == "intact"
    assert _state(sessions, default_conversation) == "gone"
    assert _state(sessions, default_trace) == "traces_gone"
    assert _state(sessions, untouched) == "intact"
    # The longer team's tasks are not even scanned: a global-period scan would
    # re-admit them on every sweep only for the lock to refuse them.
    assert report.eligible == 3
    assert (report.purged_conversations, report.purged_traces) == (2, 1)
    assert report.skipped_busy == 0 and report.skipped_not_due == 0


def test_the_assessment_uses_the_period_current_when_the_task_is_locked(
    sessions,
) -> None:
    """A team that extends between the scan and the lock keeps its task.

    400 days is past both global periods, so an assessment that fell back to
    the global periods -- or took the shorter of team and global -- would
    purge it. Only the team's current 730 days keeps it.
    """
    with sessions() as db:
        task_id = _seed(db, username="racing", age_days=400)
    user_id = _user_of(sessions, task_id)
    calls = {"n": 0}

    def extends_after_the_scan(db: Session) -> dict[int, RetentionOverride]:
        calls["n"] += 1
        days = 30 if calls["n"] == 1 else 730
        return {user_id: RetentionOverride(days, days)}

    set_retention_override_resolver(extends_after_the_scan)

    report = _batch(sessions)

    assert report.eligible == 1
    assert report.skipped_not_due == 1 and report.skipped_busy == 0
    assert _state(sessions, task_id) == "intact"


def test_a_resolver_failing_at_the_scan_purges_nothing(sessions) -> None:
    with sessions() as db:
        task_id = _seed(db, username="scan-fail", age_days=400)

    def broken(db: Session) -> dict[int, RetentionOverride]:
        raise RuntimeError("down")

    set_retention_override_resolver(broken)

    report = _batch(sessions)

    assert report.eligible == 0 and report.purged == 0
    assert _state(sessions, task_id) == "intact"


def test_a_resolver_failing_at_the_lock_skips_the_task(sessions) -> None:
    with sessions() as db:
        task_id = _seed(db, username="lock-fail", age_days=400)
    calls = {"n": 0}

    def fails_after_the_scan(db: Session) -> dict[int, RetentionOverride]:
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("down")
        return {}

    set_retention_override_resolver(fails_after_the_scan)

    report = _batch(sessions)

    assert report.eligible == 1
    assert report.skipped_override_unresolved == 1 and report.purged == 0
    assert _state(sessions, task_id) == "intact"


def test_a_per_task_failure_warns_once_per_batch_not_once_per_task(
    sessions, caplog
) -> None:
    """A resolver that fails only after the scan -- a rate-limited settings
    service that answers the scan and refuses the per-task reads -- must
    still reach WARNING, but once per batch, not once per task. The per-task
    reads log at DEBUG with the traceback; the batch logs one WARNING
    naming the count. The scan succeeds here, so nothing else would warn.
    """
    with sessions() as db:
        for index in range(3):
            _seed(db, username=f"persistent-fail-{index}", age_days=400)
    calls = {"n": 0}

    def fails_after_the_scan(db: Session) -> dict[int, RetentionOverride]:
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("down")
        return {}

    set_retention_override_resolver(fails_after_the_scan)

    with caplog.at_level(logging.DEBUG, logger=overrides_module.__name__):
        report = _batch(sessions)

    assert report.skipped_override_unresolved == 3
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert [r.name for r in warnings] == ["xagent.web.services.task_retention_purge"]
    assert "3 task(s)" in warnings[0].getMessage()
    debug_failures = [
        r
        for r in caplog.records
        if r.levelno == logging.DEBUG and "resolver failed" in r.getMessage()
    ]
    assert len(debug_failures) == 3
    assert all(r.exc_info for r in debug_failures)


def test_an_out_of_cap_value_keeps_that_teams_tasks_out_of_the_purge(
    sessions,
) -> None:
    with sessions() as db:
        refused = _seed(db, username="over-cap", age_days=400)
        other = _seed(db, username="default", age_days=400)
    refused_user = _user_of(sessions, refused)
    set_retention_override_resolver(
        lambda db: {refused_user: RetentionOverride(RETENTION_OVERRIDE_MAX_DAYS + 1)}
    )

    report = _batch(sessions)

    assert report.eligible == 1
    assert _state(sessions, refused) == "intact"
    assert _state(sessions, other) == "gone"
    with sessions() as db:
        action = purge_task(db, refused, now=NOW, conversation_days=365, trace_days=90)
    assert action is RetentionPurgeAction.SKIPPED_OVERRIDE_UNRESOLVED
    assert _state(sessions, refused) == "intact"


def test_a_missing_task_is_skipped_busy_not_unresolved(sessions) -> None:
    """A registered resolver with nothing to say about the task must not
    turn a row that vanished before the lock into ``skipped_override_unresolved``.

    ``resolve_task_retention_periods`` looks the task's ``user_id`` up to key
    the resolver's mapping; a task deleted between the scan and this call has
    none, and it must fall back to ``defaults`` rather than refuse (returning
    ``None``) -- the assessment that follows is what correctly reports a
    missing row, as ``skipped_busy``. A regression that returned ``None`` for
    the missing-task branch would instead report
    ``skipped_override_unresolved``, which is what this pins.
    """
    set_retention_override_resolver(lambda db: {})
    with sessions() as db:
        task_id = _seed(db, username="vanished", age_days=1, trace=False)
        db.execute(sa.delete(Task).where(Task.id == task_id))
        db.commit()

    with sessions() as db:
        action = purge_task(db, task_id, now=NOW, conversation_days=365, trace_days=90)
    assert action is RetentionPurgeAction.SKIPPED_BUSY


def test_a_quiescent_task_with_no_anchor_is_skipped_busy(sessions) -> None:
    """A COMPLETED task with no ``last_activity_at``, no ``created_at`` and no
    message has nothing to measure a period against, so it must be skipped as
    busy rather than reported not-due. This pins the ``assessment.anchor is
    not None`` half of the ``NOT_ELIGIBLE`` branch in ``_purge_task``: without
    it, a quiescent task with no anchor would be misreported as
    ``skipped_not_due`` -- "not due yet" implies a period that simply has not
    elapsed, which is false when there is nothing to measure at all.
    """
    with sessions() as db:
        task_id = _seed(db, username="no-anchor", age_days=1, trace=False)
        db.execute(
            sa.update(Task)
            .where(Task.id == task_id)
            .values(created_at=None, last_activity_at=None)
        )
        db.commit()

    with sessions() as db:
        action = purge_task(db, task_id, now=NOW, conversation_days=365, trace_days=90)
    assert action is RetentionPurgeAction.SKIPPED_BUSY


def test_an_override_on_a_leg_the_deployment_left_off_deletes_nothing(
    sessions,
) -> None:
    with sessions() as db:
        young = _seed(db, username="leg-off-young", age_days=60)
        old = _seed(db, username="leg-off-old", age_days=100)
    users = {_user_of(sessions, young), _user_of(sessions, old)}
    set_retention_override_resolver(
        lambda db: {user: RetentionOverride(conversation_days=30) for user in users}
    )

    report = _batch(sessions, periods=(None, 90))

    assert _state(sessions, young) == "intact"
    assert _state(sessions, old) == "traces_gone"
    assert report.purged_conversations == 0


def test_the_scan_honours_a_shorter_period_it_would_otherwise_miss(sessions) -> None:
    with sessions() as db:
        task_id = _seed(db, username="scan-short", age_days=60)
    user_id = _user_of(sessions, task_id)

    with sessions() as db:
        assert (
            select_purge_candidates(
                db, now=NOW, conversation_days=365, trace_days=90, limit=10
            )
            == []
        )
        snapshot_resolver = {user_id: RetentionOverride(30, 30)}
        set_retention_override_resolver(lambda db: snapshot_resolver)
        snapshot = load_retention_overrides(db, DEFAULTS)
        assert select_purge_candidates(
            db,
            now=NOW,
            conversation_days=365,
            trace_days=90,
            limit=10,
            overrides=snapshot,
        ) == [task_id]


def test_a_snapshot_for_other_periods_is_refused_by_the_scan(sessions) -> None:
    with sessions() as db:
        snapshot = load_retention_overrides(db, RetentionPeriods(30, 30))
        with pytest.raises(ValueError):
            select_purge_candidates(
                db,
                now=NOW,
                conversation_days=365,
                trace_days=90,
                limit=10,
                overrides=snapshot,
            )


def test_a_refused_team_is_reported_once_per_batch_not_once_per_task(
    sessions, caplog
) -> None:
    with sessions() as db:
        refused = _seed(db, username="refused", age_days=400)
        for index in range(3):
            _seed(db, username=f"plain-{index}", age_days=400)
    refused_user = _user_of(sessions, refused)
    set_retention_override_resolver(
        lambda db: {refused_user: RetentionOverride(conversation_days=0)}
    )

    with caplog.at_level(logging.WARNING, logger=overrides_module.__name__):
        report = _batch(sessions)

    assert report.purged_conversations == 3
    assert _state(sessions, refused) == "intact"
    assert [r.getMessage() for r in caplog.records if "refused" in r.getMessage()] == [
        f"retention overrides refused for 1 user(s): unusable period "
        f"(must be 1..{RETENTION_OVERRIDE_MAX_DAYS} days or inherit); "
        f"their tasks are kept; user ids (first 20): [{refused_user}]"
    ]
