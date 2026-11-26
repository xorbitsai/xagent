"""``_schedule_bg`` announces each lease-backed run with ``task_started``.

The frame must reach clients before the run's first trace event so a client
that filters by run id learns the new id first, and a failed publish must never
fail the run.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from xagent.web.models.database import (
    Base,
    get_db,
    get_engine,
    init_db,
)
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services import task_event_bridge as bridge_module
from xagent.web.services import task_events
from xagent.web.services import task_orchestrator as orchestrator
from xagent.web.services import task_start_consumer as consumer
from xagent.web.services.task_lease_service import (
    TaskLease,
    TaskLeaseRefreshState,
)
from xagent.web.services.task_orchestrator import (
    TaskTurnOrchestrator,
    TaskTurnPayload,
    TurnKind,
    _schedule_bg,
)

FAILED_COUNTER = "xagent.task.started_publish.failed"


@pytest.fixture()
def db_session(tmp_path, monkeypatch):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "false")
    # ``set_task_event_sink`` also resets the audience probe; restore both.
    monkeypatch.setattr(task_events, "_task_event_sink", task_events._task_event_sink)
    monkeypatch.setattr(
        task_events, "_task_audience_probe", task_events._task_audience_probe
    )
    init_db(db_url=f"sqlite:///{tmp_path / 'announce.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


def _make_task(db, status: TaskStatus) -> tuple[int, int]:
    user = User(username="announce-user", password_hash="hash", is_admin=False)
    db.add(user)
    db.commit()
    task = Task(
        user_id=user.id,
        title="announce",
        description="announce",
        status=status,
        execution_mode="auto",
        source="sdk",
    )
    db.add(task)
    db.commit()
    return int(task.id), int(user.id)


class _Run:
    """Records the order of everything the scheduled run does."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.frames: list[dict[str, Any]] = []

    async def sink(self, message: dict[str, Any], task_id: int) -> None:
        self.events.append("published")
        self.frames.append(message)

    def patches(self):
        run = self

        def load_snapshot(*_a, **_kw):
            run.events.append("snapshot")
            return MagicMock()

        async def execute(**_kw):
            run.events.append("execute")

        from xagent.web.services.task_execution import background_task_manager

        return [
            patch.object(orchestrator, "run_task_lease_heartbeat", new=AsyncMock()),
            patch.object(
                orchestrator,
                "load_task_setup_snapshot_sync",
                side_effect=load_snapshot,
            ),
            patch.object(orchestrator, "resolve_execution_scope", return_value=None),
            patch(
                "xagent.web.services.task_execution.execute_task_background",
                new=execute,
            ),
            patch(
                "xagent.web.services.shared_channel_execution.execute_channel_background",
                new=execute,
            ),
            patch.object(orchestrator, "settle_task_lease_isolated"),
            patch.object(orchestrator, "_get_agent_manager", return_value=MagicMock()),
            patch.object(background_task_manager, "register_task"),
        ]


def _enter(stack, patches):
    for p in patches:
        stack.enter_context(p)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "initial_status"),
    [(TurnKind.CREATE, TaskStatus.PENDING), (TurnKind.APPEND, TaskStatus.COMPLETED)],
    ids=["create", "follow-up-turn"],
)
async def test_begin_turn_announces_the_run_before_any_execution(
    db_session, kind, initial_status
) -> None:
    from contextlib import ExitStack

    task_id, user_id = _make_task(db_session, initial_status)
    run = _Run()
    task_events.set_task_event_sink(run.sink)
    with ExitStack() as stack:
        _enter(stack, run.patches())
        started = await TaskTurnOrchestrator.begin_turn(
            task_id=task_id,
            payload=TaskTurnPayload("hello"),
            task_owner_user_id=user_id,
            kind=kind,
            force_fresh=False,
        )
        await asyncio.wait_for(started.background_task, timeout=10)

    assert run.events == ["published", "snapshot", "execute"]
    (frame,) = run.frames
    assert frame["type"] == "task_started"
    assert frame["task_id"] == task_id
    assert frame["run_id"] == started.run_id
    assert frame["state_version"] == started.state_version
    assert frame["status"] == "running"
    assert frame["control_state"] == "running"


@pytest.mark.asyncio
async def test_existing_start_passes_the_claimed_version_to_the_scheduler(
    db_session,
) -> None:
    claimed = SimpleNamespace(
        state_version=7, task_lease=object(), task_source="sdk", agent_config=None
    )
    start = SimpleNamespace(
        kind="existing",
        run_id="run-existing",
        message="m",
        execution_message=None,
        turn_id="turn-1",
        timezone=None,
        existing_context=SimpleNamespace(model_dump=lambda **_kw: {}),
    )
    handoff = SimpleNamespace(
        task_owner_user_id=1, start=start, claimed=claimed, actor_policy=None
    )
    scheduler = Mock(return_value=None)
    command = SimpleNamespace(task_id=1)
    with (
        patch.object(consumer, "_commit_handoff", return_value=handoff),
        patch.object(orchestrator, "_schedule_bg", scheduler),
        patch.object(consumer, "bg_turn_inflight", return_value=False),
    ):
        await consumer._execute_task_start(command)  # type: ignore[arg-type]

    assert scheduler.call_args.kwargs["claimed_state_version"] == 7
    assert scheduler.call_args.kwargs["task_lease"] is claimed.task_lease


@pytest.mark.asyncio
async def test_channel_start_passes_the_claimed_version_to_the_scheduler(
    db_session,
) -> None:
    claimed = SimpleNamespace(
        state_version=9, task_lease=object(), task_source="slack", agent_config=None
    )
    start = SimpleNamespace(
        kind="channel",
        run_id="run-channel",
        message="m",
        execution_message=None,
        file_ids=[],
        turn_id="turn-2",
        before_message_id=None,
        timezone=None,
    )
    handoff = SimpleNamespace(
        task_owner_user_id=1, start=start, claimed=claimed, actor_policy=None
    )
    scheduler = Mock(return_value=None)
    command = SimpleNamespace(task_id=1)
    with (
        patch.object(consumer, "_commit_handoff", return_value=handoff),
        patch.object(orchestrator, "_schedule_bg", scheduler),
        patch.object(consumer, "bg_turn_inflight", return_value=False),
    ):
        await consumer._execute_task_start(command)  # type: ignore[arg-type]

    assert scheduler.call_args.kwargs["claimed_state_version"] == 9
    assert scheduler.call_args.kwargs["task_lease"] is claimed.task_lease


async def _run_scheduled(db, run: _Run, **overrides):
    """Claim a real turn, then drive the real ``_schedule_bg`` for it."""
    from contextlib import ExitStack

    task_id, user_id = _make_task(db, TaskStatus.PENDING)
    payload = TaskTurnPayload("hello")
    claimed = orchestrator._begin_turn_atomic_sync(
        task_id, user_id, payload=payload, kind=TurnKind.CREATE
    )
    kwargs: dict[str, Any] = dict(
        task_id=task_id,
        task_owner_user_id=user_id,
        task_source="sdk",
        run_id=claimed.run_id,
        task_lease=claimed.task_lease,
        payload=payload,
        force_fresh=False,
        context=None,
        claimed_state_version=claimed.state_version,
    )
    kwargs.update(overrides)
    with ExitStack() as stack:
        _enter(stack, run.patches())
        await asyncio.wait_for(_schedule_bg(**kwargs), timeout=10)
    return claimed


@pytest.mark.asyncio
async def test_single_process_publish_failure_counts_and_the_run_completes(
    db_session,
) -> None:
    run = _Run()

    async def failing_sink(message: dict[str, Any], task_id: int) -> None:
        raise RuntimeError("socket write failed")

    task_events.set_task_event_sink(failing_sink)
    counter = Mock()
    with patch.object(orchestrator, "increment_counter", counter):
        await _run_scheduled(db_session, run)

    assert run.events == ["snapshot", "execute"]
    counter.assert_any_call(FAILED_COUNTER)


@pytest.mark.asyncio
async def test_bridge_publish_failure_is_counted_by_the_bridge_and_run_completes(
    db_session, monkeypatch
) -> None:
    monkeypatch.setenv("XAGENT_REDIS_URL", "redis://localhost:6379/0")
    bridge = bridge_module.TaskEventBridge()

    async def failing_publish(channel: str, body: str) -> None:
        raise ConnectionError("redis down")

    monkeypatch.setattr(bridge, "_publish", failing_publish)
    task_events.set_task_event_sink(bridge.publish)
    run = _Run()
    bridge_counter = Mock()
    started_counter = Mock()
    with (
        patch.object(bridge_module, "report_bridge_frame", bridge_counter),
        patch.object(orchestrator, "increment_counter", started_counter),
    ):
        await _run_scheduled(db_session, run)

    assert run.events == ["snapshot", "execute"]
    bridge_counter.assert_called_once_with("publish_failed")
    started_counter.assert_not_called()


@pytest.mark.asyncio
async def test_no_frame_without_a_claimed_version(db_session) -> None:
    run = _Run()
    task_events.set_task_event_sink(run.sink)
    await _run_scheduled(db_session, run, claimed_state_version=None)
    assert run.events == ["snapshot", "execute"]


@pytest.mark.asyncio
async def test_no_frame_without_a_prefetched_lease(db_session) -> None:
    run = _Run()
    task_events.set_task_event_sink(run.sink)
    task_id, user_id = _make_task(db_session, TaskStatus.RUNNING)
    lease = TaskLease(task_id=task_id, runner_id="r", attempt_id="a")
    from contextlib import ExitStack

    with ExitStack() as stack:
        _enter(stack, run.patches())
        stack.enter_context(
            patch.object(
                orchestrator,
                "acquire_task_lease_isolated",
                return_value=lease,
            )
        )
        await asyncio.wait_for(
            _schedule_bg(
                task_id=task_id,
                task_owner_user_id=user_id,
                task_source="sdk",
                payload=TaskTurnPayload("hello"),
                force_fresh=False,
                context=None,
            ),
            timeout=10,
        )
    assert run.events == ["snapshot", "execute"]


@pytest.mark.asyncio
async def test_channel_run_announces_before_the_channel_execution(db_session) -> None:
    run = _Run()
    task_events.set_task_event_sink(run.sink)
    claimed = await _run_scheduled(db_session, run, channel_command=Mock())
    assert run.events == ["published", "snapshot", "execute"]
    assert run.frames[0]["run_id"] == claimed.run_id
    assert run.frames[0]["state_version"] == claimed.state_version


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state",
    [TaskLeaseRefreshState.LOST, TaskLeaseRefreshState.SETTLEMENT_READY],
)
async def test_no_frame_when_the_prefetched_lease_is_not_runnable(
    db_session, state
) -> None:
    run = _Run()
    task_events.set_task_event_sink(run.sink)
    with patch.object(
        orchestrator,
        "validate_preacquired_task_lease_isolated",
        return_value=state,
    ):
        await _run_scheduled(db_session, run)
    assert run.events == []
