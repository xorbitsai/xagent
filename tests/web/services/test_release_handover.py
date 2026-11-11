"""Hand-over of unfinished runs to a successor (``release_handover``).

Runs on SQLite, and on PostgreSQL when ``XAGENT_TEST_POSTGRES_URL`` is set,
through the shared ``engine`` fixture. The lease owner's pause is modelled as
production does it: the PAUSE command completes when the pause is *requested*
(``pause_requested``), and the run rests ``PAUSED`` only at a later step
boundary. The checkpoint read is replaced by a scripted verdict.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from xagent.web.models import database
from xagent.web.models.agent import Agent
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_auto_recovery import TaskAutoRecovery
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.trigger import (
    AgentTrigger,
    TriggerRun,
    TriggerRunStatus,
    TriggerType,
)
from xagent.web.models.user import User
from xagent.web.models.user_channel import UserChannel
from xagent.web.services import release_handover, task_auto_resume
from xagent.web.services.release_handover import (
    HandoverOrderError,
    begin_handover_sync,
    handover_command_id,
    settle_handover_sync,
)
from xagent.web.services.release_hold import release_hold
from xagent.web.services.task_auto_recovery import (
    HANDOVER_DELIVERY_AGENT_CONFIG_KEY,
    _user_command_since_dispatch,
    handover_pause_owns_run,
    record_lease_expiry_interruption_no_commit,
    verdict_interruption_reason,
)
from xagent.web.services.task_auto_resume import (
    cancel_pending_auto_resume_no_commit,
    check_auto_resume_claim_sync,
    run_auto_resume_tick,
)
from xagent.web.services.task_command_transport import (
    COMMAND_COMPLETED,
    COMMAND_FAILED,
    COMMAND_PENDING,
    TaskCommandKind,
    stage_task_command,
)
from xagent.web.services.task_execution_controller import (
    TaskControlState,
    apply_task_control_transition,
)
from xagent.web.services.task_lease_service import (
    CheckpointRecoveryResolution,
    CheckpointRecoveryVerdict,
    TaskLeaseRecoveryCandidate,
    utc_now,
)

engine = engine_fixture

RELEASE = "rel-1"
RUNNER = "runner-old"
RUN = "run-handover"


@pytest.fixture
def factory(engine, monkeypatch) -> Iterator[sessionmaker]:
    Base.metadata.create_all(engine)
    result = sessionmaker(engine, autoflush=False)
    monkeypatch.setattr(database, "_SessionLocal", result)
    monkeypatch.setattr(database, "_engine", engine)
    for name in (
        "XAGENT_TASK_AUTO_RESUME_ENABLED",
        "XAGENT_SHARED_TASK_EXECUTION_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_ENABLED", "true")
    for module in (release_handover, task_auto_resume):
        monkeypatch.setattr(module, "notify_task_command_dispatcher", lambda: None)
    # The draining host's hold is closed for the whole release.
    release_hold.close("test release")
    yield result
    release_hold.reopen()


@pytest.fixture
def checkpoint(monkeypatch) -> dict[str, Any]:
    """The checkpoint read: recoverable by default, scriptable per test."""
    state: dict[str, Any] = {
        "verdict": CheckpointRecoveryVerdict.RECOVERABLE,
        "tool_in_flight": False,
    }
    monkeypatch.setattr(
        release_handover,
        "resolve_checkpoint_recovery_with_data",
        lambda _db, _candidate: CheckpointRecoveryResolution(state["verdict"]),
    )
    monkeypatch.setattr(
        release_handover, "resolution_progress_marker", lambda *_a: "m1"
    )
    monkeypatch.setattr(
        release_handover,
        "resolution_tool_call_in_flight",
        lambda _r: state["tool_in_flight"],
    )
    return state


def _task(db: Session, *, channel: bool = False, **fields: Any) -> int:
    user = User(username=f"handover-{uuid.uuid4().hex[:8]}", password_hash="x")
    db.add(user)
    db.flush()
    if channel:
        row = UserChannel(
            user_id=user.id, channel_type="slack", channel_name="ops", config={}
        )
        db.add(row)
        db.flush()
        fields["channel_id"] = row.id
    values: dict[str, Any] = {
        "status": TaskStatus.RUNNING,
        "control_state": "running",
        "run_id": RUN,
        "runner_id": RUNNER,
        "state_version": 3,
    }
    values.update(fields)
    task = Task(
        user_id=user.id,
        title="Handover",
        description="handover test",
        execution_mode="balanced",
        **values,
    )
    db.add(task)
    db.flush()
    return int(task.id)


def _commands(factory: sessionmaker, task_id: int) -> list[TaskExecutionCommand]:
    with factory() as db:
        rows = list(
            db.scalars(
                sa.select(TaskExecutionCommand)
                .where(TaskExecutionCommand.task_id == task_id)
                .order_by(TaskExecutionCommand.id)
            )
        )
        db.expunge_all()
        return rows


def _row(factory: sessionmaker, task_id: int) -> TaskAutoRecovery | None:
    with factory() as db:
        row = db.get(TaskAutoRecovery, task_id)
        db.expunge_all()
        return row


def _transition(
    factory: sessionmaker,
    task_id: int,
    control_state: TaskControlState,
    status: TaskStatus | None = None,
) -> None:
    """One real control transition: it bumps ``state_version`` by exactly one."""
    with factory() as db:
        apply_task_control_transition(
            db.get(Task, task_id), control_state, status=status, expected_run_id=RUN
        )
        db.commit()


def _pause_requested(factory: sessionmaker, task_id: int) -> None:
    """Phase one of the lease owner's pause: the command completes, the run still runs."""
    _transition(factory, task_id, TaskControlState.PAUSE_REQUESTED)
    with factory() as db:
        for command in db.scalars(
            sa.select(TaskExecutionCommand).where(
                TaskExecutionCommand.task_id == task_id,
                TaskExecutionCommand.command_id.like("release-handover:%"),
                TaskExecutionCommand.status != COMMAND_COMPLETED,
            )
        ):
            command.status = COMMAND_COMPLETED
        db.commit()


def _step_boundary(factory: sessionmaker, task_id: int) -> None:
    """Phase two: the run reaches its next step boundary and rests PAUSED."""
    _transition(factory, task_id, TaskControlState.PAUSED, TaskStatus.PAUSED)


def _continue(
    factory: sessionmaker, task_id: int, kind: TaskCommandKind | None = None
) -> None:
    """A person (or the sweeper) continues the paused run, which keeps its run id."""
    if kind is not None:
        _stage(
            factory,
            task_id,
            kind,
            f"user:{uuid.uuid4().hex[:12]}",
            status=COMMAND_COMPLETED,
        )
    _transition(factory, task_id, TaskControlState.RUNNING, TaskStatus.RUNNING)


def _handed_over(factory: sessionmaker, task_id: int, release: str = RELEASE) -> None:
    begin_handover_sync(release, {RUNNER})
    _pause_requested(factory, task_id)
    _step_boundary(factory, task_id)


def _user_command(factory: sessionmaker, task_id: int, kind: TaskCommandKind) -> None:
    with factory() as db:
        user_id = db.get(Task, task_id).user_id
        stage_task_command(
            db,
            task_id=task_id,
            actor_user_id=user_id,
            command_id=f"user:{uuid.uuid4().hex[:12]}",
            kind=kind,
            payload={},
            target_run_id=RUN,
        )
        db.commit()


def _stage(
    factory: sessionmaker,
    task_id: int,
    kind: TaskCommandKind,
    command_id: str,
    *,
    status: str,
    handover: bool = False,
    reserved: bool = False,
) -> None:
    with factory() as db:
        stage_task_command(
            db,
            task_id=task_id,
            actor_user_id=db.get(Task, task_id).user_id,
            command_id=command_id,
            kind=kind,
            payload={},
            target_run_id=RUN,
            handover=handover,
            reserved=reserved,
        )
        db.flush()
        db.execute(
            sa.update(TaskExecutionCommand)
            .where(
                TaskExecutionCommand.task_id == task_id,
                TaskExecutionCommand.command_id == command_id,
            )
            .values(status=status)
        )
        db.commit()


def _set(factory: sessionmaker, task_id: int, **fields: Any) -> None:
    with factory() as db:
        for name, value in fields.items():
            setattr(db.get(Task, task_id), name, value)
        db.commit()


# -- begin ------------------------------------------------------------------


def test_begin_stages_one_pause_per_run_and_runner_and_is_idempotent(factory):
    with factory() as db:
        mine = _task(db)
        other_runner = _task(db, runner_id="runner-elsewhere")
        db.commit()

    report = begin_handover_sync(RELEASE, {RUNNER})

    assert (report.staged, report.kept) == (1, 0)
    commands = _commands(factory, mine)
    assert [(c.kind, c.command_id, c.target_run_id) for c in commands] == [
        ("pause", handover_command_id(RELEASE, mine, RUN), RUN)
    ]
    assert _commands(factory, other_runner) == []
    again = begin_handover_sync(RELEASE, {RUNNER})
    assert (again.staged, again.kept) == (0, 0)
    assert len(_commands(factory, mine)) == 1


def test_begin_covers_every_runner_of_a_pool(factory):
    with factory() as db:
        ids = [_task(db, runner_id=f"worker-{n}") for n in range(3)]
        db.commit()

    report = begin_handover_sync(RELEASE, {"worker-0", "worker-1", "worker-2"})

    assert report.staged == 3
    assert all(len(_commands(factory, i)) == 1 for i in ids)


@pytest.mark.parametrize(
    "fields",
    [
        {"source": "a2a"},
        {"source": "external"},
        {"source": "widget"},
        {"source": "shared_link"},
        {"control_state": "pause_requested"},
        {"channel": True},
        {"source": "sdk"},
    ],
    ids=["a2a", "external", "widget", "shared_link", "user-pause", "channel", "sdk"],
)
def test_begin_leaves_runs_the_successor_would_not_resume_to_the_drain(
    factory, fields, monkeypatch
):
    # Shared execution makes a channel task eligible for automatic recovery;
    # the hand-over still leaves it to the drain.
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    with factory() as db:
        task_id = _task(db, **fields)
        db.commit()

    report = begin_handover_sync(RELEASE, {RUNNER})

    assert (report.staged, report.kept) == (0, 1)
    assert _commands(factory, task_id) == []


@pytest.mark.parametrize(
    ("agent_config", "staged"),
    [
        ({HANDOVER_DELIVERY_AGENT_CONFIG_KEY: True}, 1),
        ({HANDOVER_DELIVERY_AGENT_CONFIG_KEY: False}, 0),
        ({HANDOVER_DELIVERY_AGENT_CONFIG_KEY: "true"}, 0),
        ({}, 0),
    ],
    ids=["marked", "false", "not-a-bool", "unmarked"],
)
def test_begin_hands_over_a_channel_task_only_when_its_host_marked_the_reply(
    factory, monkeypatch, agent_config, staged
):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    with factory() as db:
        task_id = _task(db, channel=True, agent_config=agent_config)
        db.commit()

    report = begin_handover_sync(RELEASE, {RUNNER})

    assert (report.staged, report.kept) == (staged, 1 - staged)
    assert len(_commands(factory, task_id)) == staged


_MARK = {HANDOVER_DELIVERY_AGENT_CONFIG_KEY: True}


@pytest.mark.parametrize(
    ("fields", "staged"),
    [
        ({"channel": True, "source": "external", "agent_config": _MARK}, 1),
        ({"channel": True, "source": "external"}, 0),
        (
            {
                "channel": True,
                "source": "external",
                "agent_config": {HANDOVER_DELIVERY_AGENT_CONFIG_KEY: "true"},
            },
            0,
        ),
        # The mark lifts only the external source, and only for a channel task.
        ({"source": "external", "agent_config": _MARK}, 0),
        ({"channel": True, "source": "a2a", "agent_config": _MARK}, 0),
        (
            {
                "channel": True,
                "source": "external",
                "agent_config": _MARK,
                "control_state": "pause_requested",
            },
            0,
        ),
    ],
    ids=[
        "marked-external-channel",
        "unmarked-external-channel",
        "not-a-bool",
        "marked-external-without-channel",
        "marked-a2a-channel",
        "marked-external-user-paused",
    ],
)
def test_a_marked_external_channel_task_is_handed_over_and_nothing_else_changes(
    factory, monkeypatch, fields, staged
):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    with factory() as db:
        task_id = _task(db, **fields)
        db.commit()

    assert begin_handover_sync(RELEASE, {RUNNER}).staged == staged
    assert len(_commands(factory, task_id)) == staged


def test_a_marked_external_channel_task_needs_shared_execution(factory, monkeypatch):
    monkeypatch.delenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", raising=False)
    with factory() as db:
        task_id = _task(db, channel=True, source="external", agent_config=_MARK)
        db.commit()

    assert begin_handover_sync(RELEASE, {RUNNER}).staged == 0
    assert _commands(factory, task_id) == []


def test_the_successor_resumes_a_marked_external_channel_run_exactly_once(
    factory, checkpoint, monkeypatch
):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    with factory() as db:
        task_id = _task(db, channel=True, source="external", agent_config=_MARK)
        db.commit()
    _handed_over(factory, task_id)
    assert settle_handover_sync(RELEASE).recorded == 1

    now = utc_now() + timedelta(seconds=30)
    assert run_auto_resume_tick(now=now).staged_commands == 1
    assert run_auto_resume_tick(now=now).staged_commands == 0
    resumes = [c for c in _commands(factory, task_id) if c.kind == "resume"]
    assert len(resumes) == 1


def test_an_unmarked_external_channel_run_is_never_resumed(
    factory, checkpoint, monkeypatch
):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    with factory() as db:
        task_id = _task(db, channel=True, source="external")
        db.commit()
    _handed_over(factory, task_id)
    settle_handover_sync(RELEASE)

    now = utc_now() + timedelta(seconds=30)
    assert run_auto_resume_tick(now=now).staged_commands == 0
    assert [c for c in _commands(factory, task_id) if c.kind == "resume"] == []


def test_begin_stages_nothing_when_automatic_resume_is_off(factory, monkeypatch):
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_ENABLED", "false")
    with factory() as db:
        task_id = _task(db)
        db.commit()

    assert begin_handover_sync(RELEASE, {RUNNER}).staged == 0
    assert _commands(factory, task_id) == []


def test_one_task_that_cannot_be_staged_does_not_fail_the_others(factory, monkeypatch):
    with factory() as db:
        bad = _task(db)
        good = _task(db)
        db.commit()
    real = release_handover.stage_task_command

    def flaky(db, **kwargs):
        if kwargs["task_id"] == bad:
            raise RuntimeError("staging failed")
        return real(db, **kwargs)

    monkeypatch.setattr(release_handover, "stage_task_command", flaky)

    report = begin_handover_sync(RELEASE, {RUNNER})

    assert (report.staged, report.kept) == (1, 1)
    assert _commands(factory, bad) == []
    assert len(_commands(factory, good)) == 1


def test_begin_refuses_a_host_that_is_not_closed_or_is_already_sealed(factory):
    release_hold.reopen()
    with pytest.raises(HandoverOrderError):
        begin_handover_sync(RELEASE, {RUNNER})
    release_hold.close("test release")
    release_hold.seal("test release")
    with pytest.raises(HandoverOrderError):
        begin_handover_sync(RELEASE, {RUNNER})


def test_settle_refuses_a_host_whose_hold_is_still_open(factory):
    release_hold.reopen()
    with pytest.raises(HandoverOrderError):
        settle_handover_sync(RELEASE)


def test_the_handover_command_id_namespace_is_not_open_to_ingress(factory):
    with factory() as db:
        task_id = _task(db)
        user_id = db.get(Task, task_id).user_id
        db.commit()
    with factory() as db, pytest.raises(ValueError, match="reserved"):
        stage_task_command(
            db,
            task_id=task_id,
            actor_user_id=user_id,
            command_id="release-handover:forged:1",
            kind=TaskCommandKind.PAUSE,
            payload={},
            target_run_id=RUN,
        )


# -- settle -----------------------------------------------------------------


def test_settle_waits_through_both_phases_of_the_pause_before_recording(
    factory, checkpoint
):
    with factory() as db:
        task_id = _task(db)
        db.commit()
    begin_handover_sync(RELEASE, {RUNNER})

    queued = settle_handover_sync(RELEASE)
    assert (queued.recorded, queued.unsettled) == (0, 1)

    # The command completed at ``pause_requested``; the run is still stepping.
    # This is the window in which the host must not tear down.
    _pause_requested(factory, task_id)
    requested = settle_handover_sync(RELEASE)
    assert (requested.recorded, requested.unsettled) == (0, 1)
    assert _row(factory, task_id) is None

    _step_boundary(factory, task_id)
    settled = settle_handover_sync(RELEASE)

    assert (settled.recorded, settled.unsettled) == (1, 0)
    row = _row(factory, task_id)
    assert row is not None
    assert (row.reason, row.state, row.run_id) == ("shutdown", "scheduled", RUN)
    assert row.next_attempt_at is not None
    assert row.progress_marker == "m1"
    assert settle_handover_sync(RELEASE).recorded == 0


def test_a_run_handed_over_in_two_consecutive_releases_is_recorded_both_times(
    factory, checkpoint
):
    with factory() as db:
        task_id = _task(db)
        db.commit()
    _handed_over(factory, task_id, "rel-1")
    assert settle_handover_sync("rel-1").recorded == 1

    # The successor resumed it on the same run id (a resume keeps the run).
    with factory() as db:
        task = db.get(Task, task_id)
        task.status, task.control_state = TaskStatus.RUNNING, "running"
        task.state_version = int(task.state_version) + 2
        db.commit()
    _handed_over(factory, task_id, "rel-2")

    assert settle_handover_sync("rel-2").recorded == 1
    row = _row(factory, task_id)
    with factory() as db:
        assert row.paused_state_version == db.get(Task, task_id).state_version
    assert settle_handover_sync("rel-2").recorded == 0


def test_a_user_pause_or_cancel_after_the_handover_keeps_the_run_with_its_user(
    factory, checkpoint
):
    with factory() as db:
        paused = _task(db)
        cancelled = _task(db)
        db.commit()
    for task_id in (paused, cancelled):
        begin_handover_sync(RELEASE, {RUNNER})
        _pause_requested(factory, task_id)
    _user_command(factory, paused, TaskCommandKind.PAUSE)
    _user_command(factory, cancelled, TaskCommandKind.CANCEL)
    _step_boundary(factory, paused)
    _step_boundary(factory, cancelled)

    report = settle_handover_sync(RELEASE)

    assert (report.recorded, report.unsettled) == (0, 0)
    assert _row(factory, paused) is None and _row(factory, cancelled) is None


def test_a_failed_handover_command_is_counted_and_never_blocks_the_stop(
    factory, checkpoint
):
    with factory() as db:
        task_id = _task(db)
        db.commit()
    begin_handover_sync(RELEASE, {RUNNER})
    with factory() as db:
        for command in db.scalars(sa.select(TaskExecutionCommand)):
            command.status = COMMAND_FAILED
        db.commit()

    report = settle_handover_sync(RELEASE)

    assert (report.recorded, report.unsettled, report.failed) == (0, 0, 1)
    assert _row(factory, task_id) is None
    # Not staged again: the run drains.
    assert begin_handover_sync(RELEASE, {RUNNER}).staged == 0


def test_a_run_that_moved_on_is_not_recorded(factory, checkpoint):
    with factory() as db:
        task_id = _task(db)
        db.commit()
    _handed_over(factory, task_id)
    with factory() as db:
        db.get(Task, task_id).run_id = "run-next-turn"
        db.commit()

    assert settle_handover_sync(RELEASE).recorded == 0
    assert _row(factory, task_id) is None


def test_a_legacy_checkpoint_with_a_started_tool_call_rests_manual(factory, checkpoint):
    checkpoint["tool_in_flight"] = True
    with factory() as db:
        task_id = _task(db)
        db.commit()
    _handed_over(factory, task_id)

    assert settle_handover_sync(RELEASE).recorded == 1

    row = _row(factory, task_id)
    assert (row.state, row.state_detail) == ("manual", "tool_in_flight")
    assert (
        run_auto_resume_tick(now=utc_now() + timedelta(seconds=30)).staged_commands == 0
    )


def test_a_checkpoint_that_cannot_be_read_yet_is_pending_and_a_missing_one_is_left(
    factory, checkpoint
):
    with factory() as db:
        task_id = _task(db)
        db.commit()
    _handed_over(factory, task_id)

    checkpoint["verdict"] = CheckpointRecoveryVerdict.INDETERMINATE
    assert settle_handover_sync(RELEASE).unsettled == 1
    checkpoint["verdict"] = CheckpointRecoveryVerdict.NOT_RECOVERABLE
    report = settle_handover_sync(RELEASE)
    assert (report.recorded, report.unsettled) == (0, 0)
    # Never resumed, but visible: a manual row says why, so the sweeper does not
    # look at the run again on every tick.
    row = _row(factory, task_id)
    assert row is not None
    assert (row.reason, row.state) == ("not_recoverable", "manual")
    assert run_auto_resume_tick(now=utc_now() + timedelta(seconds=30)).handed_over == 0
    assert _commands(factory, task_id)[-1].kind == "pause"


# -- the successor ----------------------------------------------------------


def test_the_successor_resumes_the_handed_over_run_exactly_once(factory, checkpoint):
    with factory() as db:
        task_id = _task(db)
        db.commit()
    _handed_over(factory, task_id)
    settle_handover_sync(RELEASE)

    now = utc_now() + timedelta(seconds=30)
    assert run_auto_resume_tick(now=now).staged_commands == 1
    assert run_auto_resume_tick(now=now).staged_commands == 0

    resumes = [c for c in _commands(factory, task_id) if c.kind == "resume"]
    assert len(resumes) == 1
    assert resumes[0].command_id.startswith("auto-resume:")


def test_a_user_cancel_during_the_handover_keeps_the_run_paused(factory, checkpoint):
    with factory() as db:
        task_id = _task(db)
        db.commit()
    _handed_over(factory, task_id)
    settle_handover_sync(RELEASE)

    with factory() as db:
        task = db.execute(
            sa.select(Task).where(Task.id == task_id).with_for_update()
        ).scalar_one()
        cancel = cancel_pending_auto_resume_no_commit(db, task, at="web_cancel")
        assert cancel.cancelled
        db.commit()

    now = utc_now() + timedelta(seconds=30)
    assert run_auto_resume_tick(now=now).staged_commands == 0
    assert [c.kind for c in _commands(factory, task_id)] == ["pause"]
    with factory() as db:
        task = db.get(Task, task_id)
        assert (task.status, task.control_state) == (TaskStatus.PAUSED, "paused")


# -- round 2: one predicate, a durable marker, the user's pause first -----------


def test_begin_skips_a_run_whose_user_pause_is_queued_but_not_applied(factory):
    with factory() as db:
        task_id = _task(db)
        db.commit()
    _stage(
        factory, task_id, TaskCommandKind.PAUSE, "user:early", status=COMMAND_PENDING
    )

    report = begin_handover_sync(RELEASE, {RUNNER})

    assert (report.staged, report.kept) == (0, 1)
    assert [c.command_id for c in _commands(factory, task_id)] == ["user:early"]


def test_a_user_pause_queued_before_the_handover_keeps_the_run_with_its_user(
    factory, checkpoint
):
    # Queued before the hand-over began and still waiting when it completed.
    with factory() as db:
        task_id = _task(db)
        db.commit()
    _stage(
        factory, task_id, TaskCommandKind.PAUSE, "user:early", status=COMMAND_PENDING
    )
    _stage(
        factory,
        task_id,
        TaskCommandKind.PAUSE,
        handover_command_id(RELEASE, task_id, RUN),
        status=COMMAND_COMPLETED,
        handover=True,
    )
    _transition(factory, task_id, TaskControlState.PAUSE_REQUESTED)
    _step_boundary(factory, task_id)

    report = settle_handover_sync(RELEASE)

    assert (report.recorded, report.unsettled) == (0, 0)
    assert _row(factory, task_id) is None
    with factory() as db:
        assert not handover_pause_owns_run(db, task_id, RUN)


def test_a_person_pause_before_an_earlier_resume_is_not_this_stops(factory, checkpoint):
    # The run id survives a resume: a pause the user lifted long ago must not
    # make a later hand-over look like a user pause.
    with factory() as db:
        task_id = _task(db)
        db.commit()
    _stage(
        factory, task_id, TaskCommandKind.PAUSE, "user:old", status=COMMAND_COMPLETED
    )
    _stage(
        factory, task_id, TaskCommandKind.RESUME, "user:lift", status=COMMAND_COMPLETED
    )
    _handed_over(factory, task_id)

    assert settle_handover_sync(RELEASE).recorded == 1


def test_a_run_that_ends_between_begin_and_settle_is_neither_recorded_nor_pending(
    factory, checkpoint
):
    with factory() as db:
        done = _task(db)
        failed = _task(db)
        db.commit()
    begin_handover_sync(RELEASE, {RUNNER})
    for task_id, status in ((done, TaskStatus.COMPLETED), (failed, TaskStatus.FAILED)):
        _pause_requested(factory, task_id)
        _set(factory, task_id, status=status, control_state="paused")

    report = settle_handover_sync(RELEASE)

    assert (report.recorded, report.unsettled, report.failed) == (0, 0, 0)
    assert _row(factory, done) is None and _row(factory, failed) is None


def test_settle_returns_the_recovery_notice_of_what_it_committed(factory, checkpoint):
    with factory() as db:
        task_id = _task(db)
        db.commit()
    _handed_over(factory, task_id)

    report = settle_handover_sync(RELEASE)

    assert [(n.task_id, n.run_id) for n in report.notices] == [(task_id, RUN)]
    with factory() as db:
        assert report.notices[0].state_version == db.get(Task, task_id).state_version
    assert report.notices[0].auto_resume
    assert settle_handover_sync(RELEASE).notices == ()


def test_the_handover_pause_is_not_a_person_command_for_the_resume_episode(factory):
    with factory() as db:
        task_id = _task(db)
        db.commit()
    _stage(
        factory,
        task_id,
        TaskCommandKind.RESUME,
        "auto-resume:1:1:run",
        status=COMMAND_COMPLETED,
        reserved=True,
    )
    _stage(
        factory,
        task_id,
        TaskCommandKind.PAUSE,
        handover_command_id(RELEASE, task_id, RUN),
        status=COMMAND_COMPLETED,
        handover=True,
    )
    with factory() as db:
        assert not _user_command_since_dispatch(db, task_id, "auto-resume:1:1:run")
    _stage(factory, task_id, TaskCommandKind.PAUSE, "user:p", status=COMMAND_COMPLETED)
    with factory() as db:
        assert _user_command_since_dispatch(db, task_id, "auto-resume:1:1:run")


def test_the_successor_claim_resumes_the_run_and_a_user_pause_before_it_wins(
    factory, checkpoint
):
    with factory() as db:
        resumed = _task(db)
        paused = _task(db)
        db.commit()
    for task_id in (resumed, paused):
        _handed_over(factory, task_id)
    settle_handover_sync(RELEASE)
    assert (
        run_auto_resume_tick(now=utc_now() + timedelta(seconds=30)).staged_commands == 2
    )
    # A user pause lands between the recording and the claim.
    _set(factory, paused, state_version=sa.text("state_version + 1"))

    def claim(task_id: int):
        command = next(c for c in _commands(factory, task_id) if c.kind == "resume")
        with factory() as db:
            task = db.get(Task, task_id)
            return check_auto_resume_claim_sync(
                task_id=task_id,
                command_id=command.command_id,
                auto_resume=command.payload["auto_resume"],
                status=task.status,
                control_state=task.control_state,
                run_id=task.run_id,
                state_version=task.state_version,
                attempt_count=1,
            )

    assert claim(resumed).proceed
    stale = claim(paused)
    assert (stale.proceed, stale.why) == (False, "stale")


def test_the_sweeper_records_a_handed_over_pause_nobody_settled(factory, checkpoint):
    # The controller died after the pause applied: settle never ran.
    with factory() as db:
        task_id = _task(db)
        db.commit()
    _handed_over(factory, task_id)
    release_hold.reopen()

    report = run_auto_resume_tick(now=utc_now())

    assert report.handed_over == 1
    assert [(n.task_id, n.run_id) for n in report.notices] == [(task_id, RUN)]
    row = _row(factory, task_id)
    assert (row.reason, row.state) == ("shutdown", "scheduled")
    again = run_auto_resume_tick(now=utc_now())
    assert again.handed_over == 0
    # The process that is still serving the run resumes it, after an abort.
    assert (
        run_auto_resume_tick(now=utc_now() + timedelta(seconds=30)).staged_commands == 1
    )


def test_the_sweeper_leaves_a_user_paused_run_and_acts_only_when_enabled(
    factory, checkpoint, monkeypatch
):
    with factory() as db:
        user_paused = _task(db)
        plain = _task(db)
        db.commit()
    _handed_over(factory, user_paused)
    _user_command(factory, user_paused, TaskCommandKind.PAUSE)
    _handed_over(factory, plain, "rel-2")
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_ENABLED", "false")
    assert run_auto_resume_tick(now=utc_now()).handed_over == 0
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_ENABLED", "true")

    assert run_auto_resume_tick(now=utc_now()).handed_over == 1

    assert _row(factory, user_paused) is None
    assert _row(factory, plain) is not None


def test_the_sweeper_ignores_handovers_older_than_its_window(factory, checkpoint):
    with factory() as db:
        task_id = _task(db)
        db.commit()
    _handed_over(factory, task_id)

    assert run_auto_resume_tick(now=utc_now() + timedelta(hours=3)).handed_over == 0
    assert _row(factory, task_id) is None


def _lease_recovered_reason(factory: sessionmaker, task_id: int) -> str | None:
    """Lease recovery of a run that died PAUSE_REQUESTED, reduced to its recording."""
    with factory() as db:
        task = db.get(Task, task_id)
        candidate = TaskLeaseRecoveryCandidate(
            task_id=task_id,
            runner_id=RUNNER,
            run_id=RUN,
            lease_expires_at=utc_now(),
            state_version=int(task.state_version),
            last_checkpoint_event_id=None,
            last_checkpoint_trace_event_id=None,
            attempt_id=None,
            control_state="pause_requested",
        )
        task.status, task.control_state = TaskStatus.PAUSED, "paused"
        task.state_version = int(task.state_version) + 1
        recorded = record_lease_expiry_interruption_no_commit(
            db,
            task=task,
            candidate=candidate,
            resolution=CheckpointRecoveryResolution(
                CheckpointRecoveryVerdict.RECOVERABLE
            ),
            task_status=TaskStatus.PAUSED,
            recovered_at=utc_now(),
        )
        db.commit()
        return recorded.reason if recorded is not None else None


def test_lease_recovery_of_a_run_that_died_pause_requested_records_shutdown(
    factory, checkpoint
):
    with factory() as db:
        task_id = _task(db)
        db.commit()
    begin_handover_sync(RELEASE, {RUNNER})
    _pause_requested(factory, task_id)

    assert _lease_recovered_reason(factory, task_id) == "shutdown"


def test_lease_recovery_keeps_user_pause_when_a_person_paused_too(factory, checkpoint):
    with factory() as db:
        task_id = _task(db)
        db.commit()
    begin_handover_sync(RELEASE, {RUNNER})
    _pause_requested(factory, task_id)
    _user_command(factory, task_id, TaskCommandKind.PAUSE)

    assert _lease_recovered_reason(factory, task_id) == "user_pause"


def test_the_verdict_reason_is_shutdown_only_for_a_handover_pause():
    for handover, expected in ((True, "shutdown"), (False, "user_pause")):
        reason = verdict_interruption_reason(
            CheckpointRecoveryVerdict.RECOVERABLE,
            control_state="pause_requested",
            recoverable_reason=release_handover.InterruptionReason.LEASE_EXPIRED,
            handover_pause=handover,
        )
        assert reason is not None and reason.value == expected


# -- round 3: ownership is anchored on the run's state_version ---------------------


def _owns(factory: sessionmaker, task_id: int) -> bool:
    with factory() as db:
        return handover_pause_owns_run(db, task_id, RUN)


def test_the_stage_snapshots_the_state_version_the_ownership_is_anchored_on(factory):
    with factory() as db:
        task_id = _task(db, state_version=11)
        db.commit()
    begin_handover_sync(RELEASE, {RUNNER})

    assert _commands(factory, task_id)[0].target_state_version == 11
    # Real transitions: the request is one step, the rest at the boundary another.
    _pause_requested(factory, task_id)
    _step_boundary(factory, task_id)
    with factory() as db:
        assert db.get(Task, task_id).state_version == 13
    assert _owns(factory, task_id)


def test_a_repause_after_the_successor_resumed_it_is_not_the_handovers(
    factory, checkpoint
):
    with factory() as db:
        task_id = _task(db)
        db.commit()
    _handed_over(factory, task_id)
    assert settle_handover_sync(RELEASE).recorded == 1
    fenced = _row(factory, task_id).paused_state_version
    # The successor's automatic resume keeps the run id; a later pause with no
    # recovery row (lease loss of the resumed run) is nobody's to auto-resume.
    _continue(factory, task_id)
    _transition(factory, task_id, TaskControlState.PAUSED, TaskStatus.PAUSED)

    assert not _owns(factory, task_id)
    report = run_auto_resume_tick(now=utc_now())
    assert report.handed_over == 0
    assert settle_handover_sync(RELEASE).recorded == 0
    assert _row(factory, task_id).paused_state_version == fenced


@pytest.mark.parametrize(
    "kind",
    [TaskCommandKind.MESSAGE, TaskCommandKind.RESUME_INPUT, TaskCommandKind.RESUME],
    ids=["message", "resume-input", "resume"],
)
def test_a_run_its_user_paused_and_continued_is_handed_over_later(
    factory, checkpoint, kind
):
    # The user paused the run (a completed person PAUSE, long settled) and
    # continued it on the same run id: no RESUME command marks the epoch for a
    # message or an input answer. A later hand-over of that run is the
    # platform's.
    with factory() as db:
        task_id = _task(db)
        db.commit()
    _stage(
        factory, task_id, TaskCommandKind.PAUSE, "user:old", status=COMMAND_COMPLETED
    )
    _transition(factory, task_id, TaskControlState.PAUSE_REQUESTED)
    _step_boundary(factory, task_id)
    _continue(factory, task_id, kind)
    _handed_over(factory, task_id)

    assert _owns(factory, task_id)
    assert settle_handover_sync(RELEASE).recorded == 1
    assert _row(factory, task_id).reason == "shutdown"


@pytest.mark.parametrize(
    ("error", "owned"),
    [
        ("Task is already paused", False),
        ("No live execution found to pause", False),
        ("some other failure", True),
    ],
    ids=["already-paused", "no-live-execution", "unrelated-failure"],
)
def test_a_person_pause_rejected_for_finding_the_run_paused_is_still_the_persons(
    factory, checkpoint, error, owned
):
    # The person's PAUSE lands after the hand-over rested the run: it is
    # rejected for finding nothing left to pause and ends FAILED. It was the
    # person's request all the same.
    with factory() as db:
        task_id = _task(db)
        db.commit()
    _handed_over(factory, task_id)
    _stage(factory, task_id, TaskCommandKind.PAUSE, "user:late", status=COMMAND_FAILED)
    with factory() as db:
        db.execute(
            sa.update(TaskExecutionCommand)
            .where(TaskExecutionCommand.command_id == "user:late")
            .values(error=error)
        )
        db.commit()

    assert _owns(factory, task_id) is owned
    assert settle_handover_sync(RELEASE).recorded == int(owned)
    assert run_auto_resume_tick(now=utc_now()).handed_over == 0


def test_a_run_that_settled_paused_before_its_pause_request_was_written_is_owned(
    factory, checkpoint
):
    # The interrupt reached the run, which rested PAUSED before the fenced write
    # of PAUSE_REQUESTED; the command completes as applied one step on.
    with factory() as db:
        task_id = _task(db)
        db.commit()
    begin_handover_sync(RELEASE, {RUNNER})
    _transition(factory, task_id, TaskControlState.PAUSED, TaskStatus.PAUSED)
    with factory() as db:
        db.execute(sa.update(TaskExecutionCommand).values(status=COMMAND_COMPLETED))
        db.commit()

    assert _owns(factory, task_id)
    assert settle_handover_sync(RELEASE).recorded == 1


def test_the_sweeper_does_not_let_user_paused_tasks_starve_newer_handovers(
    factory, checkpoint, monkeypatch
):
    monkeypatch.setattr(task_auto_resume, "HANDOVER_PROMOTE_LIMIT", 2)
    with factory() as db:
        doubled = _task(db, runner_id="r-doubled")
        users = {_task(db, runner_id=f"r-user{n}"): f"r-user{n}" for n in range(2)}
        genuine = {_task(db, runner_id=f"r-gen{n}"): f"r-gen{n}" for n in range(2)}
        db.commit()
    # Two releases staged ``doubled`` at the same version: two completed
    # commands, one task.
    begin_handover_sync("rel-a", {"r-doubled"})
    begin_handover_sync("rel-b", {"r-doubled"})
    _pause_requested(factory, doubled)
    _step_boundary(factory, doubled)
    for task_id, runner in (*users.items(), *genuine.items()):
        begin_handover_sync("rel-c", {runner})
        _pause_requested(factory, task_id)
        _step_boundary(factory, task_id)
    for task_id in users:
        _user_command(factory, task_id, TaskCommandKind.PAUSE)
    release_hold.reopen()

    first = run_auto_resume_tick(now=utc_now())
    second = run_auto_resume_tick(now=utc_now())

    # Two slots: the user-paused runs take none, the doubled task takes one.
    assert (first.handed_over, second.handed_over) == (2, 1)
    assert all(_row(factory, t) is not None for t in (doubled, *genuine))
    assert all(_row(factory, t) is None for t in users)


def test_begin_reads_and_stages_each_run_inside_its_own_savepoint(factory, monkeypatch):
    # PostgreSQL aborts the transaction on a failed statement: every read of one
    # run's turn must be inside the SAVEPOINT that is rolled back, or the later
    # runs and the final commit are lost with it.
    with factory() as db:
        bad = _task(db)
        good = _task(db)
        db.commit()
    seen: list[tuple[str, bool]] = []
    real_eligible = release_handover.shutdown_handoff_eligible
    real_queued = release_handover._person_stop_queued

    def eligible(db, task_id):
        seen.append(("eligible", db.in_nested_transaction()))
        if task_id == bad:
            raise RuntimeError("eligibility read failed")
        return real_eligible(db, task_id)

    def queued(db, task_id, run_id):
        seen.append(("queued", db.in_nested_transaction()))
        return real_queued(db, task_id, run_id)

    monkeypatch.setattr(release_handover, "shutdown_handoff_eligible", eligible)
    monkeypatch.setattr(release_handover, "_person_stop_queued", queued)

    report = begin_handover_sync(RELEASE, {RUNNER})

    assert (report.staged, report.kept) == (1, 1)
    assert seen and all(inside for _, inside in seen)
    assert _commands(factory, bad) == []
    assert len(_commands(factory, good)) == 1


def test_the_sweeper_stamps_the_pause_completion_time_as_the_interruption(
    factory, checkpoint
):
    with factory() as db:
        task_id = _task(db)
        db.commit()
    _handed_over(factory, task_id)
    stopped_at = utc_now() - timedelta(minutes=90)
    with factory() as db:
        db.execute(
            sa.update(TaskExecutionCommand).values(
                completed_at=stopped_at, updated_at=stopped_at
            )
        )
        db.commit()
    release_hold.reopen()

    assert run_auto_resume_tick(now=utc_now()).handed_over == 1

    interrupted = _row(factory, task_id).interrupted_at
    if interrupted.tzinfo is None:
        interrupted = interrupted.replace(tzinfo=stopped_at.tzinfo)
    assert abs(interrupted - stopped_at) < timedelta(seconds=2)


def _trigger_run(factory: sessionmaker, task_id: int) -> int:
    with factory() as db:
        user_id = db.get(Task, task_id).user_id
        agent = Agent(user_id=user_id, name=f"agent-{task_id}")
        db.add(agent)
        db.flush()
        trigger = AgentTrigger(
            user_id=user_id,
            agent_id=agent.id,
            type=TriggerType.SCHEDULED.value,
            name=f"trigger-{task_id}",
            config={},
        )
        db.add(trigger)
        db.flush()
        run = TriggerRun(
            trigger_id=trigger.id,
            task_id=task_id,
            status=TriggerRunStatus.RUNNING.value,
            idempotency_key=f"handover-{task_id}",
        )
        db.add(run)
        db.commit()
        return int(run.id)


def _trigger_status(factory: sessionmaker, run_id: int) -> str:
    with factory() as db:
        return str(db.get(TriggerRun, run_id).status)


def test_a_handover_that_lands_manual_ends_the_trigger_run_and_a_scheduled_one_keeps_it(
    factory, checkpoint
):
    with factory() as db:
        scheduled = _task(db)
        manual = _task(db)
        db.commit()
    scheduled_run = _trigger_run(factory, scheduled)
    manual_run = _trigger_run(factory, manual)
    _handed_over(factory, scheduled, "rel-1")
    assert settle_handover_sync("rel-1").recorded == 1
    # No resumable checkpoint for the second run: its row is ``manual``.
    checkpoint["verdict"] = CheckpointRecoveryVerdict.NOT_RECOVERABLE
    _handed_over(factory, manual, "rel-2")
    assert settle_handover_sync("rel-2").recorded == 0

    assert _row(factory, manual).state == "manual"
    # A scheduled resume settles the TriggerRun later; a manual one ends it now.
    assert _trigger_status(factory, scheduled_run) == "running"
    assert _trigger_status(factory, manual_run) == "failed"


def test_a_handover_whose_resume_window_is_spent_is_reported_stopped_not_recorded(
    factory, checkpoint, monkeypatch
):
    with factory() as db:
        task_id = _task(db)
        db.commit()
    _handed_over(factory, task_id)
    # The pause completed longer ago than the window: the row says ``expired``.
    long_ago = utc_now() - timedelta(days=30)
    with factory() as db:
        db.execute(
            sa.update(TaskExecutionCommand).values(
                completed_at=long_ago, updated_at=long_ago
            )
        )
        db.commit()

    report = settle_handover_sync(RELEASE)

    assert report.recorded == 0 and report.unsettled == 0
    assert _row(factory, task_id).state in ("expired", "exhausted")
    assert run_auto_resume_tick(now=utc_now()).staged_commands == 0


def test_a_run_pause_requested_one_step_after_the_stage_is_owned_until_it_moves_on(
    factory,
):
    # The window settlement and lease recovery read: RUNNING, pause requested.
    with factory() as db:
        task_id = _task(db)
        db.commit()
    begin_handover_sync(RELEASE, {RUNNER})
    _pause_requested(factory, task_id)
    assert _owns(factory, task_id)

    # Something else moved the run (a second request, a takeover): not ours.
    _transition(factory, task_id, TaskControlState.PAUSE_REQUESTED)
    assert not _owns(factory, task_id)
