"""Lease-expiry recovery records why it stopped a run, and changes nothing else.

Runs on SQLite and, when ``XAGENT_TEST_POSTGRES_URL`` is set, on PostgreSQL
through the shared ``engine`` fixture (CI runs the PostgreSQL cells in the
task-lease step of test-migrations.yml). Recovery cases drive the real
recovery batch, so SQLite exercises the compare-and-swap path and PostgreSQL
the row-lock path; only the switch-gating case calls the recorder directly.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from xagent.core.agent.checkpoint import CHECKPOINT_SCHEMA_VERSION, CHECKPOINT_TYPE
from xagent.core.agent.interruption import InterruptionReason
from xagent.web.models.agent import Agent
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.task_auto_recovery import TaskAutoRecovery, TaskRecoveryEvent
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.models.trigger import (
    AgentTrigger,
    TriggerRun,
    TriggerRunStatus,
    TriggerType,
)
from xagent.web.models.user import User
from xagent.web.models.user_channel import UserChannel
from xagent.web.models.workforce import Workforce, WorkforceRun
from xagent.web.services import task_auto_recovery, task_lease_recovery
from xagent.web.services.task_auto_recovery import record_interruption_no_commit
from xagent.web.services.task_execution_event_writer import append_fact_no_commit
from xagent.web.services.task_lease_recovery import (
    TASK_LEASE_EXPIRED_ERROR,
    TASK_LEASE_PAUSED_TRIGGER_ERROR,
    TASK_UNKNOWN_TOOL_EFFECT_ERROR,
    recover_expired_task_leases_batch_isolated,
)
from xagent.web.services.task_lease_service import (
    TASK_RUN_ID_TRACE_FIELD,
    CheckpointRecoveryResolution,
    CheckpointRecoveryVerdict,
    utc_now,
)
from xagent.web.services.trace_message_storage import (
    encode_checkpoint_data_for_storage,
)

engine = engine_fixture


@pytest.fixture
def factory(engine, monkeypatch) -> sessionmaker:
    Base.metadata.create_all(engine)
    result = sessionmaker(engine)
    monkeypatch.setattr("xagent.web.models.database.get_session_local", lambda: result)
    # Selects the PostgreSQL row-lock path or the SQLite CAS path.
    monkeypatch.setattr("xagent.web.models.database.get_engine", lambda: engine)
    monkeypatch.setattr(
        task_lease_recovery, "invalidate_task_cache_best_effort", lambda _id: None
    )
    monkeypatch.delenv("XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED", raising=False)
    return result


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _user(db: Session) -> User:
    user = User(username=f"interruption-{uuid.uuid4().hex[:8]}", password_hash="x")
    db.add(user)
    db.flush()
    return user


def _snapshot(*, messages: int, iteration: int = 0) -> dict[str, Any]:
    return {
        "pattern": "ReActPattern",
        "context": {
            "messages": [
                {
                    "role": "user" if index % 2 == 0 else "assistant",
                    "content": f"m{index}",
                }
                for index in range(messages)
            ]
        },
        "pattern_state": {"current_iteration": iteration, "tool_ledger": {}},
    }


def _expire(task: Task, *, run_id: str) -> None:
    """Put ``task`` back in a RUNNING run whose lease has already expired."""
    task.status = TaskStatus.RUNNING
    task.control_state = "running"
    task.runner_id = f"dead-runner-{uuid.uuid4().hex[:6]}"
    task.run_id = run_id
    task.lease_expires_at = utc_now() - timedelta(seconds=5)
    task.last_heartbeat_at = utc_now() - timedelta(seconds=10)
    task.error_message = None


def _legacy_checkpoint(
    db: Session, task: Task, *, messages: int, iteration: int = 0, encode=False
) -> None:
    event_id = f"checkpoint-{uuid.uuid4().hex[:10]}"
    data: dict[str, Any] = {
        "checkpoint_type": CHECKPOINT_TYPE,
        "snapshot": _snapshot(messages=messages, iteration=iteration),
        TASK_RUN_ID_TRACE_FIELD: task.run_id,
    }
    if encode:
        data = encode_checkpoint_data_for_storage(
            db, task_id=int(task.id), data=data, use_v2=True
        )
        # The marker must come from decoded refs, not the stored markers.
        assert "__encoding" in data["snapshot"]["context"]["messages"]
    db.add(
        TraceEvent(
            task_id=task.id,
            event_id=event_id,
            event_type="system_update_general",
            timestamp=utc_now(),
            data=data,
        )
    )
    task.last_checkpoint_event_id = event_id
    task.last_checkpoint_trace_event_id = None


def _event_checkpoint(db: Session, task: Task, *, messages: int) -> None:
    task.conversation_storage_version = 2
    db.flush()
    append_fact_no_commit(
        db,
        task_id=int(task.id),
        kind="recovery_state",
        key=f"runtime:{uuid.uuid4()}",
        payload={
            "data": {
                "checkpoint_type": CHECKPOINT_TYPE,
                "snapshot_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "execution_id": str(task.id),
                "snapshot": _snapshot(messages=messages, iteration=2),
            }
        },
        run_id=task.run_id,
    )


def _task(
    db: Session,
    user: User,
    *,
    checkpoint: str | None = "legacy",
    messages: int = 2,
    **fields: Any,
) -> Task:
    task = Task(
        user_id=user.id,
        title="Interrupted task",
        description="lease expiry interruption test",
        execution_mode="balanced",
        state_version=3,
        output="stale output",
        **fields,
    )
    _expire(task, run_id=f"run-{uuid.uuid4().hex[:8]}")
    if "control_state" in fields:
        task.control_state = fields["control_state"]
    db.add(task)
    db.flush()
    if checkpoint == "legacy":
        _legacy_checkpoint(db, task, messages=messages, iteration=1)
    elif checkpoint == "encoded":
        _legacy_checkpoint(db, task, messages=messages, iteration=1, encode=True)
    elif checkpoint == "events":
        _event_checkpoint(db, task, messages=messages)
    db.commit()
    return task


def _recover() -> int:
    return recover_expired_task_leases_batch_isolated(
        cutoff=utc_now(), batch_size=10, after=None
    ).recovered


def _state(factory: sessionmaker, task_id: int) -> tuple[Task, Any, list[Any]]:
    with factory() as db:
        task = db.get(Task, task_id)
        row = db.get(TaskAutoRecovery, task_id)
        events = list(
            db.scalars(
                sa.select(TaskRecoveryEvent)
                .where(TaskRecoveryEvent.task_id == task_id)
                .order_by(TaskRecoveryEvent.id)
            )
        )
        db.expunge_all()
    return task, row, events


@pytest.mark.parametrize("checkpoint", ["legacy", "encoded", "events"])
def test_recoverable_run_pauses_and_records_lease_expired(factory, checkpoint):
    with factory() as db:
        task_id = int(_task(db, _user(db), checkpoint=checkpoint, messages=3).id)
    before = utc_now()

    assert _recover() == 1

    task, row, events = _state(factory, task_id)
    # The task row is exactly what lease recovery writes without metadata.
    assert task.status == TaskStatus.PAUSED
    assert task.control_state == "paused"
    assert task.state_version == 4
    assert task.error_message is None
    assert task.runner_id is None and task.lease_expires_at is None
    assert row.run_id == task.run_id
    assert row.reason == "lease_expired"
    assert row.state == "manual"
    assert row.state_detail is None
    assert row.paused_state_version == task.state_version
    assert _aware(row.interrupted_at) >= before
    assert _aware(row.episode_started_at) == _aware(row.interrupted_at)
    assert (row.no_progress_resumes, row.total_resumes) == (0, 0)
    iterations = 2 if checkpoint == "events" else 1
    assert row.progress_marker == f"m3:i{iterations}:t0:p0:s0"
    assert row.next_attempt_at is None
    assert [(e.event, e.reason, e.run_id) for e in events] == [
        ("interrupted", "lease_expired", task.run_id)
    ]
    assert events[0].detail == {
        "task_status": "paused",
        "state": "manual",
        "kind": "normal",
    }


def test_pause_requested_at_crash_records_user_pause(factory):
    with factory() as db:
        task_id = int(_task(db, _user(db), control_state="pause_requested").id)

    assert _recover() == 1

    task, row, events = _state(factory, task_id)
    assert task.status == TaskStatus.PAUSED
    assert task.control_state == "paused"
    assert (row.reason, row.state) == ("user_pause", "manual")
    assert [e.reason for e in events] == ["user_pause"]


def test_run_without_checkpoint_fails_unchanged_and_records_not_recoverable(
    factory,
):
    with factory() as db:
        task_id = int(_task(db, _user(db), checkpoint=None).id)

    assert _recover() == 1

    task, row, events = _state(factory, task_id)
    assert task.status == TaskStatus.FAILED
    assert task.control_state == "failed"
    assert task.error_message == TASK_LEASE_EXPIRED_ERROR
    assert task.output is None
    assert (row.reason, row.state, row.progress_marker) == (
        "not_recoverable",
        "manual",
        None,
    )
    assert row.paused_state_version == task.state_version
    assert [(e.event, e.detail["task_status"]) for e in events] == [
        ("interrupted", "failed")
    ]


def test_unknown_tool_effect_fails_unchanged_and_records_its_reason(
    factory, monkeypatch
):
    with factory() as db:
        task_id = int(_task(db, _user(db), control_state="pause_requested").id)
    monkeypatch.setattr(
        task_lease_recovery,
        "resolve_checkpoint_recovery_with_data",
        lambda db, candidate: CheckpointRecoveryResolution(
            CheckpointRecoveryVerdict.UNKNOWN_TOOL_EFFECT
        ),
    )

    assert _recover() == 1

    task, row, _events = _state(factory, task_id)
    assert task.status == TaskStatus.FAILED
    assert task.error_message == TASK_UNKNOWN_TOOL_EFFECT_ERROR
    # A pause request does not outrank a terminal verdict.
    assert (row.reason, row.state) == ("unknown_tool_effect", "manual")


def test_indeterminate_verdict_writes_nothing(factory, monkeypatch):
    with factory() as db:
        task_id = int(_task(db, _user(db)).id)
    monkeypatch.setattr(
        task_lease_recovery,
        "resolve_checkpoint_recovery_with_data",
        lambda db, candidate: CheckpointRecoveryResolution(
            CheckpointRecoveryVerdict.INDETERMINATE
        ),
    )

    assert _recover() == 0

    task, row, events = _state(factory, task_id)
    assert task.status == TaskStatus.RUNNING
    assert row is None and events == []


def test_lost_recovery_race_writes_nothing(factory, monkeypatch):
    with factory() as db:
        task_id = int(_task(db, _user(db)).id)
    monkeypatch.setattr(
        task_lease_recovery,
        "recover_expired_task_lease_no_commit",
        lambda *args, **kwargs: False,
    )

    assert _recover() == 0

    _task_row, row, events = _state(factory, task_id)
    assert row is None and events == []


def test_infra_failure_pause_switch_does_not_gate_lease_expiry(factory, monkeypatch):
    # Lease expiry paused runs before this switch existed, so its rows stay
    # manual: a later executor may still schedule them (plan §5.10).
    monkeypatch.setenv("XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED", "false")
    with factory() as db:
        task_id = int(_task(db, _user(db)).id)

    assert _recover() == 1

    task, row, events = _state(factory, task_id)
    assert task.status == TaskStatus.PAUSED
    assert task.error_message is None
    assert (row.reason, row.state, row.state_detail) == (
        "lease_expired",
        "manual",
        None,
    )
    assert [e.event for e in events] == ["interrupted"]


@pytest.mark.parametrize(
    ("source", "switch", "expected"),
    [
        (None, "false", ("disabled", None, "interrupted")),
        (None, "true", ("manual", None, "interrupted")),
        # Ineligibility is permanent, so it wins over the temporary switch.
        ("sdk", "false", ("ineligible", "ineligible:source_sdk", "ineligible")),
        ("sdk", "true", ("ineligible", "ineligible:source_sdk", "ineligible")),
    ],
)
def test_gated_settlement_state_precedence(
    factory, monkeypatch, source, switch, expected
):
    monkeypatch.setenv("XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED", switch)
    with factory() as db:
        fields = {"source": source} if source else {}
        task = _task(db, _user(db), checkpoint=None, **fields)
        row = record_interruption_no_commit(
            db,
            task=task,
            reason=InterruptionReason.PERSISTENCE_FAILURE,
            task_status=TaskStatus.FAILED,
            interrupted_at=utc_now(),
            progress_marker=None,
            gated_by_settlement_switches=True,
        )
        assert row is not None
        event = db.scalars(
            sa.select(TaskRecoveryEvent).where(TaskRecoveryEvent.task_id == task.id)
        ).one()
        assert (row.state, row.state_detail, event.event) == expected
        assert event.detail.get("state_detail") == expected[1]


def _channel(db: Session, user: User) -> int:
    channel = UserChannel(
        user_id=user.id, channel_type="slack", channel_name="ops", config={}
    )
    db.add(channel)
    db.flush()
    return int(channel.id)


def _workforce_run(db: Session, user: User, *, is_preview: bool) -> int:
    manager = Agent(user_id=user.id, name="interruption manager")
    db.add(manager)
    db.flush()
    workforce = Workforce(
        owner_user_id=user.id,
        scope_type="user",
        scope_id=str(user.id),
        name=f"Interruption workforce {uuid.uuid4().hex[:6]}",
        manager_agent_id=manager.id,
        status="published",
    )
    db.add(workforce)
    db.flush()
    run = WorkforceRun(
        workforce_id=workforce.id,
        user_id=user.id,
        status="running",
        snapshot={},
        is_preview=is_preview,
    )
    db.add(run)
    db.flush()
    return int(run.id)


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("sdk", "ineligible:source_sdk"),
        ("a2a", "ineligible:source_a2a"),
        ("external", "ineligible:source_external"),
        ("widget", "ineligible:source_widget"),
        ("shared_link", "ineligible:source_shared_link"),
        ("workforce_preview", "ineligible:preview"),
        ("workforce_missing", "ineligible:workforce_run_missing"),
        ("internal_invisible", "ineligible:preview"),
        ("channel_combined", "ineligible:channel_inline"),
    ],
)
def test_ineligible_tasks_recover_unchanged_and_record_why(
    factory, monkeypatch, case, expected
):
    monkeypatch.setenv(
        "XAGENT_SHARED_TASK_EXECUTION_ENABLED",
        "false" if case == "channel_combined" else "true",
    )
    with factory() as db:
        user = _user(db)
        fields: dict[str, Any] = {}
        if case in {"sdk", "a2a", "external", "widget", "shared_link"}:
            fields["source"] = case
        elif case == "workforce_preview":
            fields["agent_config"] = {
                "workforce_run_id": _workforce_run(db, user, is_preview=True)
            }
        elif case == "workforce_missing":
            fields["agent_config"] = {"workforce_run_id": 987654321}
        elif case == "internal_invisible":
            fields["is_visible"] = False
        else:
            fields["channel_id"] = _channel(db, user)
        task_id = int(_task(db, user, **fields).id)

    assert _recover() == 1

    task, row, events = _state(factory, task_id)
    assert task.status == TaskStatus.PAUSED
    assert task.error_message is None
    assert (row.reason, row.state, row.state_detail) == (
        "lease_expired",
        "ineligible",
        expected,
    )
    assert [(e.event, e.detail.get("state_detail")) for e in events] == [
        ("ineligible", expected)
    ]


@pytest.mark.parametrize(
    ("case", "kind"),
    [
        ("normal", "normal"),
        ("trigger", "trigger_webhook"),
        ("trigger_no_type", "trigger"),
        ("trigger_unknown_type", "trigger"),
        ("trigger_unhashable_type", "trigger"),
        ("trigger_with_workforce", "workforce"),
        ("workforce", "workforce"),
        ("channel_shared", "channel"),
        ("channel_shared_invisible", "channel"),
    ],
)
def test_eligible_tasks_record_their_kind(factory, monkeypatch, case, kind):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    with factory() as db:
        user = _user(db)
        fields: dict[str, Any] = {}
        if case.startswith("trigger"):
            fields["source"] = "trigger"
            fields["agent_config"] = {
                "trigger": {"trigger_type": TriggerType.WEBHOOK.value},
                "trigger_no_type": {},
                "trigger_unknown_type": {"trigger_type": "carrier_pigeon"},
                "trigger_unhashable_type": {"trigger_type": ["webhook"]},
                "trigger_with_workforce": {
                    "trigger_type": TriggerType.SCHEDULED.value,
                    "workforce_run_id": _workforce_run(db, user, is_preview=False),
                },
            }[case]
        elif case == "workforce":
            fields["agent_config"] = {
                "workforce_run_id": _workforce_run(db, user, is_preview=False)
            }
        elif case.startswith("channel_shared"):
            fields["channel_id"] = _channel(db, user)
            if case == "channel_shared_invisible":
                # The channel rule precedes the hidden-preview rule.
                fields["is_visible"] = False
        task_id = int(_task(db, user, **fields).id)

    assert _recover() == 1

    _task_row, row, events = _state(factory, task_id)
    assert (row.state, row.state_detail) == ("manual", None)
    assert [(e.event, e.detail.get("kind")) for e in events] == [("interrupted", kind)]


def _rerun(factory: sessionmaker, task_id: int, *, run_id: str | None = None):
    """Resume ``task_id`` by hand and let the resumed run's lease expire."""
    with factory() as db:
        task = db.get(Task, task_id)
        _expire(task, run_id=run_id or task.run_id)
        db.commit()


def _progress(factory: sessionmaker, task_id: int, *, messages: int) -> None:
    with factory() as db:
        task = db.get(Task, task_id)
        _legacy_checkpoint(db, task, messages=messages, iteration=1)
        db.commit()


def _mark_dispatched(factory: sessionmaker, task_id: int, **counters: int) -> None:
    """Stand-in for an automatic dispatch, which is what the counters count."""
    with factory() as db:
        row = db.get(TaskAutoRecovery, task_id)
        row.state = "dispatched"
        row.last_command_id = "auto-resume:previous"
        for name, value in counters.items():
            setattr(row, name, value)
        db.commit()


def test_repeated_interruptions_track_episodes_and_runs(factory):
    with factory() as db:
        task_id = int(_task(db, _user(db), messages=2).id)
    assert _recover() == 1
    _task_row, first, _events = _state(factory, task_id)
    _mark_dispatched(factory, task_id, no_progress_resumes=2, total_resumes=5)

    # Same run, same checkpoint: the no-progress episode continues.
    _rerun(factory, task_id)
    assert _recover() == 1
    task, same, events = _state(factory, task_id)
    assert same.run_id == first.run_id
    assert same.progress_marker == first.progress_marker
    assert _aware(same.episode_started_at) == _aware(first.episode_started_at)
    assert _aware(same.interrupted_at) > _aware(first.interrupted_at)
    assert same.paused_state_version == task.state_version
    assert same.paused_state_version > first.paused_state_version
    assert (same.no_progress_resumes, same.total_resumes) == (2, 5)
    assert same.last_command_id == "auto-resume:previous"
    assert len(events) == 2

    # Same run with a newer checkpoint: progress starts a new episode.
    _mark_dispatched(factory, task_id)
    _rerun(factory, task_id)
    _progress(factory, task_id, messages=4)
    assert _recover() == 1
    _task_row, progressed, _events = _state(factory, task_id)
    assert progressed.progress_marker == "m4:i1:t0:p0:s0"
    assert _aware(progressed.episode_started_at) == _aware(progressed.interrupted_at)
    assert _aware(progressed.episode_started_at) > _aware(same.episode_started_at)
    assert (progressed.no_progress_resumes, progressed.total_resumes) == (0, 5)
    assert progressed.last_command_id == "auto-resume:previous"

    # A new run resets every counter, even at an identical marker.
    _mark_dispatched(factory, task_id, no_progress_resumes=1)
    _rerun(factory, task_id, run_id="run-next")
    with factory() as db:
        task = db.get(Task, task_id)
        _legacy_checkpoint(db, task, messages=4, iteration=1)
        db.commit()
    assert _recover() == 1
    _task_row, new_run, events = _state(factory, task_id)
    assert new_run.run_id == "run-next"
    assert new_run.progress_marker == progressed.progress_marker
    assert _aware(new_run.episode_started_at) == _aware(new_run.interrupted_at)
    assert (new_run.no_progress_resumes, new_run.total_resumes) == (0, 0)
    assert new_run.last_command_id is None
    assert [e.run_id for e in events] == [first.run_id] * 3 + ["run-next"]


def test_missing_marker_continues_the_episode(factory, monkeypatch):
    with factory() as db:
        task_id = int(_task(db, _user(db), messages=2).id)
    assert _recover() == 1
    _task_row, first, _events = _state(factory, task_id)
    assert first.progress_marker is not None
    _mark_dispatched(factory, task_id, no_progress_resumes=2)

    # An undecodable checkpoint is unknown, not progress.
    monkeypatch.setattr(
        task_auto_recovery, "resolution_progress_marker", lambda *_a, **_k: None
    )
    _rerun(factory, task_id)
    assert _recover() == 1

    _task_row, row, _events = _state(factory, task_id)
    assert row.progress_marker == first.progress_marker
    assert _aware(row.episode_started_at) == _aware(first.episode_started_at)
    assert row.no_progress_resumes == 2


_MARKER = "m2:i1:t0:p0:s0"
_DISPATCH_ID = "auto-resume:3:8:run"


def _seed_episode(
    db: Session, *, state: str, reason: str = "lease_expired"
) -> tuple[Task, datetime]:
    """A task whose row has spent resumes in an episode that began earlier."""
    task = _task(db, _user(db), checkpoint=None)
    started = utc_now() - timedelta(minutes=10)
    db.add(
        TaskAutoRecovery(
            task_id=task.id,
            run_id=task.run_id,
            reason=reason,
            state=state,
            paused_state_version=int(task.state_version),
            interrupted_at=started,
            episode_started_at=started,
            no_progress_resumes=2,
            total_resumes=7,
            progress_marker=_MARKER,
            last_command_id=_DISPATCH_ID,
        )
    )
    db.flush()
    return task, started


def _command(db: Session, task: Task, command_id: str, kind: str) -> None:
    db.add(
        TaskExecutionCommand(
            task_id=task.id,
            command_id=command_id,
            kind=kind,
            payload={},
            status="completed",
        )
    )
    db.flush()


def _record_again(
    db: Session,
    task: Task,
    *,
    reason: InterruptionReason = InterruptionReason.LEASE_EXPIRED,
    marker: str | None = _MARKER,
) -> TaskAutoRecovery:
    row = record_interruption_no_commit(
        db,
        task=task,
        reason=reason,
        task_status=TaskStatus.PAUSED,
        interrupted_at=utc_now(),
        progress_marker=marker,
        gated_by_settlement_switches=False,
    )
    assert row is not None
    return row


@pytest.mark.parametrize(
    "state", ["exhausted", "manual", "scheduled", "stale", "dispatch_failed"]
)
def test_interruption_after_a_non_dispatched_state_restarts_every_counter(
    factory, state
):
    # Nothing automatic resumed the run since the row was written: its user
    # (or nobody) did, so the run is not held to the resumes spent before.
    with factory() as db:
        task, started = _seed_episode(db, state=state)
        row = _record_again(db, task)
        assert _aware(row.episode_started_at) == _aware(row.interrupted_at)
        assert _aware(row.episode_started_at) > _aware(started)
        assert (row.no_progress_resumes, row.total_resumes) == (0, 0)
        assert row.last_command_id is None


def test_interruption_of_a_dispatched_resume_continues_the_episode(factory):
    with factory() as db:
        task, started = _seed_episode(db, state="dispatched")
        _command(db, task, _DISPATCH_ID, "resume")
        # A later dispatch of the sweeper's own is not a person acting.
        _command(db, task, "auto-resume:3:9:run", "resume")
        row = _record_again(db, task)
        assert _aware(row.episode_started_at) == _aware(started)
        assert (row.no_progress_resumes, row.total_resumes) == (2, 7)
        assert row.last_command_id == _DISPATCH_ID


def test_new_reason_after_dispatch_starts_an_episode_but_keeps_the_total(factory):
    with factory() as db:
        task, started = _seed_episode(db, state="dispatched")
        row = _record_again(db, task, reason=InterruptionReason.LLM_UNAVAILABLE)
        assert _aware(row.episode_started_at) == _aware(row.interrupted_at)
        assert (row.no_progress_resumes, row.total_resumes) == (0, 7)
        assert row.last_command_id == _DISPATCH_ID


def test_progress_after_dispatch_starts_an_episode_but_keeps_the_total(factory):
    with factory() as db:
        task, _started = _seed_episode(db, state="dispatched")
        row = _record_again(db, task, marker="m5:i2:t0:p0:s0")
        assert _aware(row.episode_started_at) == _aware(row.interrupted_at)
        assert (row.no_progress_resumes, row.total_resumes) == (0, 7)


@pytest.mark.parametrize(
    ("command_id", "kind"),
    [
        ("message-1", "message"),
        ("resume-by-hand", "resume"),
        ("pause-1", "pause"),
        # Only the sweeper's RESUME counts as automatic; a forged id on any
        # other kind is a person's command.
        ("auto-resume:3:9:run", "message"),
    ],
)
def test_user_command_after_dispatch_restarts_every_counter(factory, command_id, kind):
    with factory() as db:
        task, _started = _seed_episode(db, state="dispatched")
        _command(db, task, _DISPATCH_ID, "resume")
        _command(db, task, command_id, kind)
        row = _record_again(db, task)
        assert _aware(row.episode_started_at) == _aware(row.interrupted_at)
        assert (row.no_progress_resumes, row.total_resumes) == (0, 0)
        assert row.last_command_id is None


def test_user_command_before_the_dispatch_does_not_count(factory):
    with factory() as db:
        task, started = _seed_episode(db, state="dispatched")
        _command(db, task, "message-1", "message")
        _command(db, task, _DISPATCH_ID, "resume")
        row = _record_again(db, task)
        assert _aware(row.episode_started_at) == _aware(started)
        assert (row.no_progress_resumes, row.total_resumes) == (2, 7)


def test_missing_dispatch_command_reads_as_no_intervention(factory):
    # The dispatch command is gone; a later command cannot be ordered after
    # it, so the interruption still continues the episode.
    with factory() as db:
        task, started = _seed_episode(db, state="dispatched")
        _command(db, task, "message-1", "message")
        row = _record_again(db, task)
        assert _aware(row.episode_started_at) == _aware(started)
        assert (row.no_progress_resumes, row.total_resumes) == (2, 7)


def _projected_task(factory: sessionmaker, *, verdict: str) -> tuple[int, int, int]:
    """A v2 trigger + workforce task whose run's lease has expired."""
    with factory() as db:
        user = _user(db)
        run_id = _workforce_run(db, user, is_preview=False)
        task = _task(
            db,
            user,
            checkpoint="events" if verdict == "recoverable" else None,
            source="trigger",
            conversation_storage_version=2,
        )
        task.agent_config = {"workforce_run_id": run_id}
        workforce_run = db.get(WorkforceRun, run_id)
        workforce_run.task_id = task.id
        trigger = AgentTrigger(
            user_id=user.id,
            workforce_id=workforce_run.workforce_id,
            type=TriggerType.SCHEDULED.value,
            name="Interruption trigger",
            config={},
        )
        db.add(trigger)
        db.flush()
        trigger_run = TriggerRun(
            trigger_id=trigger.id,
            task_id=task.id,
            status=TriggerRunStatus.RUNNING.value,
            idempotency_key=f"interruption-{task.id}",
        )
        db.add(trigger_run)
        db.commit()
        return int(task.id), run_id, int(trigger_run.id)


def _projections(factory: sessionmaker, ids: tuple[int, int, int]) -> tuple:
    task_id, run_id, trigger_run_id = ids
    with factory() as db:
        task = db.get(Task, task_id)
        workforce_run = db.get(WorkforceRun, run_id)
        trigger_run = db.get(TriggerRun, trigger_run_id)
        facts = [
            (event.payload["status"], event.payload["result"])
            for event in db.scalars(
                sa.select(TaskExecutionEvent).where(
                    TaskExecutionEvent.task_id == task_id,
                    TaskExecutionEvent.kind == "execution_settled",
                )
            )
        ]
        return (
            task.status,
            task.control_state,
            task.error_message,
            task.output,
            task.state_version,
            task.runner_id,
            task.lease_expires_at,
            workforce_run.status,
            workforce_run.completed_at is None,
            trigger_run.status,
            trigger_run.error_message,
            trigger_run.finished_at is not None,
            facts,
        )


def _fail_recording(m: pytest.MonkeyPatch, failure: str) -> None:
    if failure in {"python", "database"}:
        original = task_auto_recovery.record_interruption_no_commit

        def fail(db, **kwargs):
            # Leave real writes behind first: the SAVEPOINT must discard them.
            original(db, **kwargs)
            if failure == "database":
                db.execute(sa.text("SELECT * FROM no_such_table"))
            raise RuntimeError("metadata failed")

        m.setattr(task_auto_recovery, "record_interruption_no_commit", fail)
        return

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(f"{failure} failed")

    target = {
        "marker": "resolution_progress_marker",
        "eligibility": "auto_recovery_eligibility",
    }[failure]
    m.setattr(task_auto_recovery, target, boom)


@pytest.mark.parametrize("verdict", ["recoverable", "not_recoverable"])
@pytest.mark.parametrize(
    "mode", ["recorded", "python", "database", "marker", "eligibility"]
)
def test_recording_never_changes_the_recovered_task(
    factory, monkeypatch, verdict, mode
):
    """Projections equal those of recovery with no metadata at all."""
    with monkeypatch.context() as m:
        m.setattr(
            task_auto_recovery,
            "record_lease_expiry_interruption_no_commit",
            lambda *_args, **_kwargs: None,
        )
        plain = _projected_task(factory, verdict=verdict)
        assert _recover() == 1
    with monkeypatch.context() as m:
        if mode != "recorded":
            _fail_recording(m, mode)
        observed = _projected_task(factory, verdict=verdict)
        assert _recover() == 1

    # Pin the baseline itself too, so both sides cannot drift together.
    baseline = _projections(factory, plain)
    if verdict == "recoverable":
        assert baseline[0] == TaskStatus.PAUSED
        assert baseline[2] is None
        assert baseline[7] == "paused"
        assert baseline[10] == TASK_LEASE_PAUSED_TRIGGER_ERROR
    else:
        assert baseline[0] == TaskStatus.FAILED
        assert baseline[2] == TASK_LEASE_EXPIRED_ERROR
        assert baseline[10] == TASK_LEASE_EXPIRED_ERROR
    assert baseline[9] == TriggerRunStatus.FAILED.value
    assert baseline[12], "a v2 task stages its execution_settled fact"
    assert _projections(factory, observed) == baseline
    _task_row, row, events = _state(factory, observed[0])
    if mode == "recorded":
        assert row is not None
        assert row.reason == (
            "lease_expired" if verdict == "recoverable" else "not_recoverable"
        )
        assert len(events) == 1
    else:
        assert row is None and events == []
