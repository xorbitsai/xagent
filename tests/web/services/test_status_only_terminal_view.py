"""The terminal view a status-only (SDK) caller reads for a stopped task.

``status_only_terminal_view`` answers one question from the stored rows alone:
has automatic resume of this PAUSED SDK task stopped for good, with nothing
queued that could still move it? The cases below seed the rows directly
(matrix) or drive the production writers (exits); expectations are written
here as literals, never taken from the code under test. Runs on SQLite and,
when ``XAGENT_TEST_POSTGRES_URL`` is set, on PostgreSQL.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from tests.web.services.test_auto_resume_cancel import AT as CANCEL_AT
from tests.web.services.test_task_auto_resume import (
    PV,
    RUN,
    _channel_task,
    _dispatched,
    _seed,
    _user,
)
from tests.web.services.test_task_auto_resume import factory as factory_fixture
from xagent.core.agent.interruption import InterruptionReason
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_auto_recovery import TaskAutoRecovery, TaskAutoRecoveryState
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.services import task_auto_resume
from xagent.web.services.task_auto_recovery import record_interruption_no_commit
from xagent.web.services.task_auto_resume import (
    auto_resume_command_id,
    cancel_pending_auto_resume_no_commit,
    check_auto_resume_claim_sync,
    run_auto_resume_tick,
    status_only_terminal_view,
    supersede_pending_auto_resume_no_commit,
)
from xagent.web.services.task_lease_service import utc_now

engine = engine_fixture
factory = factory_fixture

# The client-visible texts, as the contract states them; a stop text ends with
# the row's reason (see ``_text``).
EXHAUSTED = (
    "Task failed after a system interruption; "
    "automatic resume stopped after repeated attempts"
)
EXPIRED = (
    "Task failed after a system interruption; "
    "it was not resumed within the automatic-resume window"
)
DISPATCH_FAILED = (
    "Task failed after a system interruption; automatic resume could not be started"
)
INELIGIBLE = (
    "Task failed after a system interruption; it can no longer be resumed automatically"
)
DISABLED = "Task failed after a system interruption; automatic resume is disabled"
NEW_TURN = "Task failed: the new message could not be started."
OTHER_MANUAL = (
    "Task failed after a system interruption; automatic resume will not continue it"
)

STOPPED_AT = datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc)
LAST_COMMAND = auto_resume_command_id(PV, 1, RUN)


def _text(state: str, detail: str | None, reason: str) -> str:
    """The literal text table of the contract."""
    if state == "manual" and detail == "new_turn":
        return NEW_TURN
    if state == "exhausted":
        body = EXHAUSTED
    elif state == "expired":
        body = EXPIRED
    elif state == "dispatch_failed":
        body = DISPATCH_FAILED
    elif state in ("ineligible", "superseded"):
        body = INELIGIBLE
    else:
        assert state == "manual"
        body = DISABLED if detail == "auto_resume_disabled" else OTHER_MANUAL
    return f"{body} (reason: {reason})."


def _view(factory: sessionmaker, task_id: int):
    with factory() as db:
        return status_only_terminal_view(db, task_id)


def _stored_updated_at(factory: sessionmaker, task_id: int):
    with factory() as db:
        return db.scalar(
            sa.select(TaskAutoRecovery.updated_at).where(
                TaskAutoRecovery.task_id == task_id
            )
        )


# --------------------------------------------------------------------------
# The decision table, written out
# --------------------------------------------------------------------------

_STOP_STATES = {"exhausted", "expired", "dispatch_failed", "ineligible", "superseded"}
_PERSON_KINDS = {"resume", "resume_input", "pause", "cancel"}
_OPENING_KINDS = {"start", "resume", "resume_input", "message"}


def _expected(
    *,
    source: str,
    status: str,
    control: str,
    state: str,
    detail: str | None,
    reason: str,
    delta: int,
    last_command: bool,
    lost: bool = False,
    rejected: str = "none",
    after: list[tuple[str, str, str]],
    queued: list[tuple[str, str]],
) -> str | None:
    """The text a terminal cell reads, or None. Rules 1-6, literally."""
    if source != "sdk":
        return None
    if status != "paused" or control != "paused":
        return None
    if reason == "user_pause":
        return None
    stop = state in _STOP_STATES or (state == "manual" and detail != "user_cancelled")
    if not stop:
        return None
    person = any(
        kind in _PERSON_KINDS and not command_id.startswith("auto-resume:")
        for kind, _status, command_id in after
    )
    if delta == 0:
        fence = True
    elif delta == 2:
        fence = (
            state in ("dispatch_failed", "manual")
            and last_command
            and not lost
            and not person
        )
    else:
        fence = False
    if not fence:
        return None
    if any(
        kind in _OPENING_KINDS and status_ in ("pending", "processing")
        for kind, status_ in queued
    ):
        return None
    if rejected == "at_current":
        return NEW_TURN
    return _text(state, detail, reason)


def _add_command(
    db: Session,
    task_id: int,
    *,
    kind: str,
    status: str,
    command_id: str,
    target_run_id: str | None = RUN,
    target_state_version: int | None = None,
) -> None:
    db.add(
        TaskExecutionCommand(
            task_id=task_id,
            actor_user_id=db.get(Task, task_id).user_id,
            command_id=command_id,
            kind=kind,
            payload={},
            target_run_id=target_run_id,
            target_state_version=target_state_version,
            status=status,
        )
    )
    db.flush()


def _seed_cell(
    factory: sessionmaker,
    *,
    source: str,
    control: str,
    state: str,
    detail: str | None,
    reason: str,
    delta: int,
    last_command: bool,
    lost: bool = False,
    rejected: str = "none",
    after: list[tuple[str, str, str]],
    queued: list[tuple[str, str]],
) -> int:
    now = utc_now()
    with factory() as db:
        task = _bare_task(
            db, source=source, control_state=control, state_version=PV + delta
        )
        db.add(
            TaskAutoRecovery(
                task_id=task.id,
                run_id=RUN,
                reason=reason,
                state=state,
                state_detail=detail,
                paused_state_version=PV,
                interrupted_at=now - timedelta(seconds=60),
                episode_started_at=now - timedelta(seconds=60),
                last_command_id=LAST_COMMAND if last_command else None,
                updated_at=STOPPED_AT,
            )
        )
        db.flush()
        if last_command:
            _add_command(
                db,
                int(task.id),
                kind="resume",
                status="completed",
                command_id=LAST_COMMAND,
            )
        for kind, status, command_id in after:
            _add_command(
                db, int(task.id), kind=kind, status=status, command_id=command_id
            )
        for index, (kind, status) in enumerate(queued):
            _add_command(
                db,
                int(task.id),
                kind=kind,
                status=status,
                command_id=f"queued-{index}-{uuid.uuid4().hex[:6]}",
            )
        if rejected != "none":
            # A START the caller's append staged, rejected at the task's
            # current version (``at_current``) or at an older one; or a
            # failed RESUME at the current version (``resume_at_current``),
            # which is not a rejected new message. Its id carries the
            # sweeper's prefix so it is not a person's command.
            resume = rejected == "resume_at_current"
            _add_command(
                db,
                int(task.id),
                kind="resume" if resume else "start",
                status="failed",
                command_id=(
                    f"{'auto-resume:' if resume else ''}rejected-{uuid.uuid4().hex[:6]}"
                ),
                target_run_id="run-of-the-new-turn",
                target_state_version=PV
                + delta
                - (0 if rejected in ("at_current", "resume_at_current") else 1),
            )
        if lost:
            # The dispatch command is gone while the row still names it.
            db.execute(
                sa.delete(TaskExecutionCommand).where(
                    TaskExecutionCommand.task_id == task.id,
                    TaskExecutionCommand.command_id == LAST_COMMAND,
                )
            )
        db.commit()
        return int(task.id)


def _bare_task(db: Session, **fields: Any) -> Task:
    """A PAUSED SDK task on ``RUN`` at ``PV``; ``fields`` override."""
    task = Task(
        user_id=_user(db).id,
        title="View",
        description="view test",
        execution_mode="balanced",
        **{
            "source": "sdk",
            "status": TaskStatus.PAUSED,
            "control_state": "paused",
            "run_id": RUN,
            "state_version": PV,
            **fields,
        },
    )
    db.add(task)
    db.flush()
    return task


# fence name -> (delta, last_command_id set, commands after the last dispatch)
_FENCES: dict[str, tuple[int, bool, list[tuple[str, str, str]]]] = {
    "equal": (0, True, []),
    "plus1": (1, True, []),
    "plus2_no_person": (2, True, []),
    "plus2_person_resume": (2, True, [("resume", "completed", "web-resume-1")]),
    "plus2_person_resume_input": (
        2,
        True,
        [("resume_input", "completed", "web-answer-1")],
    ),
    "plus2_person_pause": (2, True, [("pause", "completed", "web-pause-1")]),
    "plus2_person_cancel": (2, True, [("cancel", "completed", "web-cancel-1")]),
    "plus2_person_upper_prefix": (
        2,
        True,
        [("resume", "completed", "AUTO-RESUME:person-1")],
    ),
    "plus2_auto_resume_later": (
        2,
        True,
        [("resume", "completed", "auto-resume:later-1")],
    ),
    "plus2_sdk_start_failed": (2, True, [("start", "failed", "start-2")]),
    "plus2_no_last_command": (2, False, []),
}
# The same, with the dispatch command deleted from the table.
_LOST_FENCES: dict[str, tuple[int, bool, list[tuple[str, str, str]]]] = {
    "plus2_lost_dispatch_person_resume": (
        2,
        True,
        [("resume", "completed", "web-resume-2")],
    ),
    "plus2_lost_dispatch_no_person": (2, True, []),
}
_DETAILS = [None, "auto_resume_disabled", "new_turn", "user_cancelled"]
_REASONS = ["lease_expired", "llm_unavailable", "user_pause"]
_COMMANDS: dict[str, list[tuple[str, str]]] = {
    "none": [],
    "start_pending": [("start", "pending")],
    "message_pending": [("message", "pending")],
    "resume_processing": [("resume", "processing")],
    "cancel_pending": [("cancel", "pending")],
    "start_completed": [("start", "completed")],
}


def test_every_recovery_state_is_in_the_matrix():
    assert len(TaskAutoRecoveryState) == 10
    assert _STOP_STATES | {
        "manual",
        "scheduled",
        "dispatched",
        "stale",
        "disabled",
    } == {state.value for state in TaskAutoRecoveryState}


@pytest.mark.parametrize("state", [s.value for s in TaskAutoRecoveryState])
def test_view_matrix(factory, state):
    mismatches: list[str] = []

    def check(label: str, expected: str | None, **cell: Any) -> None:
        tid = _seed_cell(factory, state=state, **cell)
        view = _view(factory, tid)
        if expected is None:
            if view is not None:
                mismatches.append(f"{label}: expected None, got {view}")
            return
        if view is None:
            mismatches.append(f"{label}: expected terminal, got None")
            return
        if (view.state, view.error) != (state, expected):
            mismatches.append(f"{label}: got {(view.state, view.error)}")
        if view.reason != cell["reason"]:
            mismatches.append(f"{label}: reason {view.reason}")
        if view.stopped_at != _stored_updated_at(factory, tid):
            mismatches.append(f"{label}: stopped_at {view.stopped_at}")

    for detail in _DETAILS:
        for reason in _REASONS:
            fences = [(f, v, False) for f, v in _FENCES.items()] + [
                (f, v, True) for f, v in _LOST_FENCES.items()
            ]
            for fence, (delta, last, after), lost in fences:
                # A START rejected at the task's current version, or at an
                # older one, changes the text of a terminal cell only.
                for rejected in (
                    ("none", "at_current", "at_older", "resume_at_current")
                    if reason == "llm_unavailable"
                    else ("none",)
                ):
                    cell = dict(
                        source="sdk",
                        control="paused",
                        detail=detail,
                        reason=reason,
                        delta=delta,
                        last_command=last,
                        lost=lost,
                        rejected=rejected,
                        after=after,
                        queued=[],
                    )
                    check(
                        f"{detail}/{reason}/{fence}/{rejected}",
                        _expected(status="paused", state=state, **cell),
                        **cell,
                    )

    for command, queued in _COMMANDS.items():
        for control in ("paused", "resume_requested"):
            for source in ("sdk", "internal", "trigger", "a2a"):
                cell = dict(
                    source=source,
                    control=control,
                    detail=None,
                    reason="lease_expired",
                    delta=0,
                    last_command=False,
                    after=[],
                    queued=queued,
                )
                check(
                    f"{command}/{control}/{source}",
                    _expected(status="paused", state=state, **cell),
                    **cell,
                )

    assert mismatches == []


def test_view_none_for_a_task_that_is_not_paused(factory):
    tid = _seed_cell(
        factory,
        source="sdk",
        control="paused",
        state="exhausted",
        detail=None,
        reason="lease_expired",
        delta=0,
        last_command=False,
        after=[],
        queued=[],
    )
    for status in (TaskStatus.RUNNING, TaskStatus.FAILED):
        with factory() as db:
            db.get(Task, tid).status = status
            db.commit()
        assert _view(factory, tid) is None, status


def test_view_none_for_a_missing_task(factory):
    assert _view(factory, 987_654) is None


# --------------------------------------------------------------------------
# One statement; a failed read is not terminal
# --------------------------------------------------------------------------


def test_view_uses_one_statement(factory, engine):
    tid = _seed_cell(
        factory,
        source="sdk",
        control="paused",
        state="dispatch_failed",
        detail=None,
        reason="lease_expired",
        delta=2,
        last_command=True,
        after=[],
        queued=[],
    )
    statements: list[str] = []

    def count(_conn, _cursor, statement, *_args):
        statements.append(statement)

    with factory() as db:
        db.connection()  # BEGIN outside the count
        sa.event.listen(engine, "before_cursor_execute", count)
        try:
            view = status_only_terminal_view(db, tid)
        finally:
            sa.event.remove(engine, "before_cursor_execute", count)
    assert view is not None
    savepoint = ("SAVEPOINT", "RELEASE SAVEPOINT", "ROLLBACK TO SAVEPOINT")
    real = [s for s in statements if not s.lstrip().upper().startswith(savepoint)]
    assert len(real) == 1, real
    assert real[0].lstrip().upper().startswith("SELECT")


@pytest.mark.parametrize("case", ["no_row", "older_run_row", "read_raises"])
def test_view_none_without_current_row(factory, case, caplog, monkeypatch):
    tid = _seed_cell(
        factory,
        source="sdk",
        control="paused",
        state="exhausted",
        detail=None,
        reason="lease_expired",
        delta=0,
        last_command=False,
        after=[],
        queued=[],
    )
    if case == "no_row":
        with factory() as db:
            db.execute(sa.delete(TaskAutoRecovery))
            db.commit()
    elif case == "older_run_row":
        with factory() as db:
            db.get(TaskAutoRecovery, tid).run_id = "run-before"
            db.commit()
    with factory() as db:
        if case == "read_raises":
            real_execute = db.execute

            def failing(statement, *args, **kwargs):
                # A real database error, so that on PostgreSQL the enclosing
                # transaction is aborted unless a SAVEPOINT contains it.
                real_execute(sa.text("SELECT 1 FROM no_such_table_for_the_view"))
                return real_execute(statement, *args, **kwargs)

            monkeypatch.setattr(db, "execute", failing)
        with caplog.at_level(logging.WARNING, logger=task_auto_resume.logger.name):
            assert status_only_terminal_view(db, tid) is None
        if case == "read_raises":
            monkeypatch.undo()
            warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert len(warnings) == 1
            # The caller's transaction survives the failed statement.
            assert db.scalar(sa.select(sa.func.count()).select_from(Task)) >= 1
        else:
            assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# --------------------------------------------------------------------------
# Every exit that stops automatic resume, driven through production code
# --------------------------------------------------------------------------


def _complete_commands(factory: sessionmaker, tid: int) -> None:
    with factory() as db:
        db.execute(
            sa.update(TaskExecutionCommand)
            .where(TaskExecutionCommand.task_id == tid)
            .values(status="completed")
        )
        db.commit()


def _claim(tid: int, *, version: int = PV, attempt: int = 1):
    return check_auto_resume_claim_sync(
        task_id=tid,
        command_id=LAST_COMMAND,
        auto_resume={
            "expected_run_id": RUN,
            "expected_state_version": PV,
            "reason": "lease_expired",
            "attempt": attempt,
        },
        status=TaskStatus.PAUSED,
        control_state="paused",
        run_id=RUN,
        state_version=version,
        attempt_count=attempt,
    )


def _tick_exit(setup: Callable[[Session, datetime], int], *, env=None):
    """An exit reached by a sweeper tick over one seeded row."""

    def drive(factory, monkeypatch):
        for key, value in (env or {}).items():
            monkeypatch.setenv(key, value)
        now = utc_now()
        with factory() as db:
            tid = setup(db, now)
            db.commit()
        run_auto_resume_tick(now=now)
        return tid

    return drive


def _record_exit(
    reason: InterruptionReason = InterruptionReason.LEASE_EXPIRED,
    *,
    ago: float = 30,
    task_fields: dict[str, Any] | None = None,
    continuing: bool = False,
    env: dict[str, str] | None = None,
):
    """An exit reached when settlement or lease recovery records the pause."""

    def drive(factory, monkeypatch):
        for key, value in (env or {}).items():
            monkeypatch.setenv(key, value)
        now = utc_now()
        with factory() as db:
            if continuing:
                tid = _seed(
                    db,
                    now,
                    source="sdk",
                    state="dispatched",
                    total=20,
                    no_progress=3,
                    last_command_id=LAST_COMMAND,
                )
                db.flush()
                task = db.get(Task, tid)
            else:
                task = _bare_task(db, **(task_fields or {}))
            row = record_interruption_no_commit(
                db,
                task=task,
                reason=reason,
                task_status=TaskStatus.PAUSED,
                interrupted_at=now - timedelta(seconds=ago),
                progress_marker=None,
                gated_by_settlement_switches=False,
                now=now,
            )
            assert row is not None
            tid = int(task.id)
            db.commit()
        return tid

    return drive


def _housekeeping(
    command_status: str | None, *, rolled_back: bool = False, **seed: Any
):
    def setup(db: Session, now: datetime) -> int:
        return _dispatched(
            db,
            now,
            command_status=command_status,
            source="sdk",
            **({"state_version": PV + 2} if rolled_back else {}),
            **seed,
        )

    return setup


def _claim_exit(*, version: int, attempt: int):
    """The claim-time guard stopped the row while its own RESUME is running."""

    def drive(factory, monkeypatch):
        monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_ENABLED", "false")
        now = utc_now()
        with factory() as db:
            tid = _dispatched(
                db,
                now,
                command_status="processing",
                source="sdk",
                **({"state_version": version} if version != PV else {}),
            )
            db.commit()
        decision = _claim(tid, version=version, attempt=attempt)
        assert decision.proceed is False
        assert _view(factory, tid) is None, "the claiming RESUME is still running"
        _complete_commands(factory, tid)
        return tid

    return drive


def _channel_exit(act: Callable[[Session, Task], Any], state: str):
    """Exits only a channel task reaches, driven on an SDK task given a
    channel (production never gives an SDK task one), which checks that the
    state they write reads as None."""

    def drive(factory, monkeypatch):
        now = utc_now()
        with factory() as db:
            tid = _channel_task(db, now, state=state, source="sdk")
            db.commit()
        with factory() as db:
            task = db.execute(
                sa.select(Task).where(Task.id == tid).with_for_update()
            ).scalar_one()
            assert act(db, task)
            db.commit()
        return tid

    return drive


def _dispatch_stale():
    def setup(db: Session, now: datetime) -> int:
        return _seed(db, now, source="sdk", state_version=PV + 1)

    return setup


def _drain(**seed: Any):
    def setup(db: Session, now: datetime) -> int:
        return _seed(db, now, source="sdk", due_in=3_600, **seed)

    return setup


_OFF = {"XAGENT_TASK_AUTO_RESUME_ENABLED": "false"}

# exit id -> (driver, expected recovery state, expected text); None = no view
_EXITS: dict[str, tuple[Callable[..., int], str | None, str | None]] = {
    "dispatch-ineligible": (
        _tick_exit(
            lambda db, now: _seed(
                db, now, source="sdk", agent_config={"workforce_run_id": 999_999}
            )
        ),
        "ineligible",
        INELIGIBLE,
    ),
    "limit-exhausted_total": (
        _tick_exit(lambda db, now: _seed(db, now, source="sdk", total=20)),
        "exhausted",
        EXHAUSTED,
    ),
    "limit-exhausted_elapsed": (
        _tick_exit(
            lambda db, now: _seed(
                db,
                now,
                source="sdk",
                reason="llm_unavailable",
                episode_ago=7_200,
                total=5,
            )
        ),
        "exhausted",
        EXHAUSTED,
    ),
    "limit-expired": (
        _tick_exit(
            lambda db, now: _seed(db, now, source="sdk", interrupted_ago=86_400 + 60)
        ),
        "expired",
        EXPIRED,
    ),
    "limit-no_policy": (
        _tick_exit(
            lambda db, now: _seed(db, now, source="sdk", reason="not_recoverable")
        ),
        "manual",
        OTHER_MANUAL,
    ),
    "housekeeping-disabled": (
        _tick_exit(_housekeeping("completed"), env=_OFF),
        "manual",
        DISABLED,
    ),
    "housekeeping-disabled-plus2": (
        _tick_exit(_housekeeping("failed", rolled_back=True), env=_OFF),
        "manual",
        DISABLED,
    ),
    "housekeeping-dispatch_failed": (
        _tick_exit(_housekeeping("failed")),
        "dispatch_failed",
        DISPATCH_FAILED,
    ),
    "housekeeping-dispatch_failed_lost_command": (
        _tick_exit(_housekeeping(None)),
        "dispatch_failed",
        DISPATCH_FAILED,
    ),
    "housekeeping-dispatch_failed-plus2": (
        _tick_exit(_housekeeping("failed", rolled_back=True)),
        "dispatch_failed",
        DISPATCH_FAILED,
    ),
    "reschedule-limit": (
        _tick_exit(_housekeeping("completed", no_progress=3, total=3)),
        "exhausted",
        EXHAUSTED,
    ),
    "drain-disabled": (_tick_exit(_drain(), env=_OFF), "manual", DISABLED),
    "claim-disabled": (
        _claim_exit(version=PV, attempt=1),
        "manual",
        DISABLED,
    ),
    "refence-disabled": (
        _claim_exit(version=PV + 2, attempt=2),
        "manual",
        DISABLED,
    ),
    "record-exhausted": (_record_exit(continuing=True), "exhausted", EXHAUSTED),
    "record-expired": (_record_exit(ago=86_400 + 60), "expired", EXPIRED),
    "record-ineligible": (
        _record_exit(task_fields={"agent_config": {"workforce_run_id": 999_999}}),
        "ineligible",
        INELIGIBLE,
    ),
    "record-auto_resume_disabled": (
        _record_exit(env=_OFF),
        "manual",
        DISABLED,
    ),
    # Exits that are not a stop the caller can be told about.
    "dispatch-stale": (_tick_exit(_dispatch_stale()), "stale", None),
    "drain-stale": (_tick_exit(_drain(state_version=PV + 1), env=_OFF), "stale", None),
    "claim-stale": (
        lambda factory, monkeypatch: _stale_claim(factory),
        "stale",
        None,
    ),
    "channel-supersede": (
        _channel_exit(
            lambda db, task: supersede_pending_auto_resume_no_commit(
                db, task, now=utc_now()
            ),
            "scheduled",
        ),
        "stale",
        None,
    ),
    "person-cancel": (
        _channel_exit(
            lambda db, task: cancel_pending_auto_resume_no_commit(
                db, task, at=CANCEL_AT
            ).cancelled,
            "scheduled",
        ),
        "manual",
        None,
    ),
    "record-user_pause": (
        _record_exit(InterruptionReason.USER_PAUSE),
        "manual",
        None,
    ),
    "record-ineligible_user_pause": (
        _record_exit(
            InterruptionReason.USER_PAUSE,
            task_fields={"agent_config": {"workforce_run_id": 999_999}},
        ),
        "ineligible",
        None,
    ),
}


def _stale_claim(factory) -> int:
    now = utc_now()
    with factory() as db:
        tid = _dispatched(db, now, command_status="processing", source="sdk")
        db.commit()
    decision = _claim(tid, version=PV + 1)
    assert decision.proceed is False
    return tid


@pytest.mark.parametrize("exit_id", sorted(_EXITS))
def test_every_exit_projects(factory, monkeypatch, exit_id):
    """Each place that stops automatic resume leaves a row the view reads as
    the text it owes, or as None where the caller must keep waiting.

    ``channel-supersede`` and ``person-cancel`` are reachable only for
    channel tasks in production (an SDK task has no channel); they are driven
    on an SDK task given a channel, which checks that the state they write
    reads as None."""
    driver, state, text = _EXITS[exit_id]
    tid = driver(factory, monkeypatch)

    with factory() as db:
        row = db.get(TaskAutoRecovery, tid)
        assert row.state == state, (row.state, row.state_detail)
        db.expunge(row)
    view = _view(factory, tid)
    if text is None:
        assert view is None
    else:
        assert view is not None
        assert (view.state, view.error, view.reason) == (
            state,
            f"{text} (reason: {row.reason}).",
            row.reason,
        )
        assert view.stopped_at == _stored_updated_at(factory, tid)
