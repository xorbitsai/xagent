"""Per-process cap on dispatcher-claimed executions."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from tests.shared.async_waits import DB_PROGRESS_TIMEOUT, eventually
from xagent import config
from xagent.web.models import database
from xagent.web.models.database import Base
from xagent.web.models.task import Task
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.task_command_terminal_event import TaskCommandTerminalEvent
from xagent.web.models.user import User

# Imported up front: modules that bind ``get_session_local`` at import time must
# not first load while a test has replaced it.
from xagent.web.services import (  # noqa: F401
    task_command_execution,
)
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_coordinator_runtime as runtime
from xagent.web.services import (  # noqa: F401
    task_execution,
    task_execution_admission,
    task_orchestrator,
    task_resume,
    task_resume_command,
    task_start,
    task_start_consumer,
)
from xagent.web.services.task_command_transport import (
    TaskCommandDeferred,
    TaskCommandKind,
    TaskCommandRejected,
)
from xagent.web.services.task_dispatch_cap import (
    CONTROL_KINDS,
    EXECUTION_OPENING_KINDS,
    current_dispatch_limit,
)
from xagent.web.services.task_execution import BackgroundTaskManager

LIMIT_ENV = config.DISPATCHER_MAX_INFLIGHT_PER_PROCESS


def set_limit(monkeypatch: pytest.MonkeyPatch, limit: int) -> None:
    monkeypatch.setenv(LIMIT_ENV, str(limit))


# --- limit memo --------------------------------------------------------------


def test_dispatch_limit_memo_warns_once_per_raw_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = Mock()
    monkeypatch.setattr(config, "logger", logger)
    monkeypatch.setenv(LIMIT_ENV, "9")
    assert current_dispatch_limit() == 9

    monkeypatch.setenv(LIMIT_ENV, "abc")
    for _ in range(5):
        assert current_dispatch_limit() == 80
    assert logger.warning.call_count == 1

    monkeypatch.setenv(LIMIT_ENV, "xyz")
    assert current_dispatch_limit() == 80
    assert logger.warning.call_count == 2

    monkeypatch.setenv(LIMIT_ENV, "9")
    assert current_dispatch_limit() == 9
    assert logger.warning.call_count == 2

    monkeypatch.setenv(LIMIT_ENV, "abc")
    assert current_dispatch_limit() == 80
    assert logger.warning.call_count == 3


def test_execution_opening_kinds_are_complete() -> None:
    assert EXECUTION_OPENING_KINDS | CONTROL_KINDS == {
        kind.value for kind in TaskCommandKind
    }
    assert not EXECUTION_OPENING_KINDS & CONTROL_KINDS
    assert EXECUTION_OPENING_KINDS == task_execution_admission._EXECUTION_KINDS


# --- shared-mode dispatcher host ---------------------------------------------


class CapHost:
    """A shared-mode dispatcher over a SQLite queue and a private execution registry."""

    def __init__(
        self,
        factory: sessionmaker,
        registry: runtime.TaskCoordinatorRegistry,
        manager: BackgroundTaskManager,
        user_id: int,
    ) -> None:
        self.factory = factory
        self.registry = registry
        self.manager = manager
        self.user_id = user_id
        self.claimed: list[tuple[int, TaskCommandKind]] = []
        self.held: list[asyncio.Task] = []
        self._identity = 0
        self.registered_active_sizes: list[int] = []

    def new_task(self, **fields: Any) -> int:
        with self.factory() as db:
            task = Task(
                user_id=self.user_id, title="task", description="task", **fields
            )
            db.add(task)
            db.commit()
            return int(task.id)

    def enqueue(self, task_id: int, kind: str) -> int:
        self._identity += 1
        with self.factory() as db:
            row = TaskExecutionCommand(
                task_id=task_id,
                command_id=f"cmd-{self._identity}",
                kind=kind,
                payload={},
            )
            db.add(row)
            db.commit()
            return int(row.id)

    def row(self, command_id: int) -> TaskExecutionCommand:
        with self.factory() as db:
            row = db.get(TaskExecutionCommand, command_id)
            db.expunge(row)
            return row

    def live(self, task_id: int) -> tuple[asyncio.Task, asyncio.Event]:
        """Register an execution that ends when the returned event is set."""

        finish = asyncio.Event()
        handle = asyncio.get_running_loop().create_task(finish.wait())
        self.held.append(handle)
        self.manager.register_task(task_id, handle)
        return handle, finish

    async def handler(self, command: Any) -> dict[str, Any]:
        """Claim sink: execution-opening kinds register an execution, controls do not."""

        self.claimed.append((command.task_id, command.kind))
        if command.kind.value in EXECUTION_OPENING_KINDS:
            handle = asyncio.get_running_loop().create_task(asyncio.Event().wait())
            self.held.append(handle)
            self.manager.register_task(command.task_id, handle)
            self.registered_active_sizes.append(len(self.manager._active_task_ids()))
        return {"ok": True}

    def terminal_events(self) -> int:
        with self.factory() as db:
            return db.query(TaskCommandTerminalEvent).count()


@pytest.fixture
async def cap_host(tmp_path, monkeypatch):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'cap.db'}")

    @sa.event.listens_for(engine, "connect")
    def relax_durability(connection, _record):
        # The queue is scratch state; commit latency is not under test.
        connection.execute("PRAGMA synchronous=OFF")

    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as db:
        user = User(username="cap-user", password_hash="unused")
        db.add(user)
        db.commit()
        user_id = int(user.id)
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(database, "get_session_local", lambda: factory)
    monkeypatch.setattr(transport, "get_runner_id", lambda: "worker")
    monkeypatch.setattr(runtime, "get_runner_id", lambda: "worker")
    monkeypatch.setattr(runtime, "get_task_lease_heartbeat_seconds", lambda: 0.5)
    registry = runtime.TaskCoordinatorRegistry(factory)
    monkeypatch.setattr(runtime, "get_task_coordinator_registry", lambda: registry)
    manager = BackgroundTaskManager()
    monkeypatch.setattr(task_execution, "background_task_manager", manager)
    host = CapHost(factory, registry, manager, user_id)
    try:
        yield host
    finally:
        await transport.stop_task_command_dispatcher()
        for handle in host.held:
            handle.cancel()
        await asyncio.gather(*host.held, return_exceptions=True)
        await registry.close()
        engine.dispose()


async def dispatch(host: CapHost) -> bool:
    return await asyncio.wait_for(
        transport.dispatch_one_task_command(host.handler), DB_PROGRESS_TIMEOUT
    )


async def test_dispatcher_snapshot_feeds_gate_held_seconds(
    cap_host: CapHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = cap_host
    set_limit(monkeypatch, 1)
    clock = SimpleNamespace(now=0.0)
    host.manager._cap_clock = lambda: clock.now
    host.manager._live_executions[999] = {object()}  # type: ignore[assignment]

    assert await dispatch(host) is False

    assert host.manager.drain_gate_held_seconds(5.0) == 5.0


# --- tail and reservation accounting -----------------------------------------


@pytest.mark.parametrize("registration", ["register_task", "reserved_resume"])
async def test_execution_tail_counts_against_cap(
    cap_host: CapHost, monkeypatch: pytest.MonkeyPatch, registration: str
) -> None:
    host = cap_host
    set_limit(monkeypatch, 2)
    manager = host.manager
    finishes: list[asyncio.Event] = []
    handles: list[asyncio.Task] = []

    for task_id in (host.new_task(), host.new_task()):
        finish = asyncio.Event()
        finishes.append(finish)

        async def body(task_id: int = task_id, finish: asyncio.Event = finish) -> None:
            # The execution's own finally has run: it has left every registry
            # table but is still unwinding.
            manager.cleanup_task(task_id)
            await finish.wait()

        handle = asyncio.get_running_loop().create_task(body())
        handles.append(handle)
        if registration == "register_task":
            manager.register_task(task_id, handle)
        else:
            assert manager.reserve_resume(task_id)
            manager.register_reserved_resume(task_id, handle, run_id="run")
            manager.promote_resume_task(task_id, handle)
    await asyncio.sleep(0.05)
    assert manager.running_tasks == {} and manager.resume_tasks == {}

    target = host.new_task()
    command_id = host.enqueue(target, "start")
    assert not await dispatch(host)
    assert host.claimed == []
    assert host.row(command_id).status == "pending"

    finishes[0].set()
    await asyncio.wait_for(handles[0], DB_PROGRESS_TIMEOUT)
    await asyncio.sleep(0)
    assert await dispatch(host)
    assert host.claimed == [(target, TaskCommandKind.START)]
    finishes[1].set()
    await asyncio.gather(*handles)


def pending_start_commands(host: CapHost) -> list[tuple[int, int]]:
    with host.factory() as db:
        rows = (
            db.query(TaskExecutionCommand.id, TaskExecutionCommand.task_id)
            .filter(
                TaskExecutionCommand.kind == "start",
                TaskExecutionCommand.status == "pending",
            )
            .order_by(TaskExecutionCommand.id)
            .all()
        )
    return [(int(command_id), int(task_id)) for command_id, task_id in rows]


def enqueue_starts(host: CapHost, count: int) -> None:
    with host.factory() as db:
        for _ in range(count):
            task = Task(user_id=host.user_id, title="task", description="task")
            db.add(task)
            db.flush()
            host._identity += 1
            db.add(
                TaskExecutionCommand(
                    task_id=task.id,
                    command_id=f"cmd-{host._identity}",
                    kind="start",
                    payload={},
                )
            )
        db.commit()


async def test_one_notify_one_free_slot_admits_exactly_one(
    cap_host: CapHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = cap_host
    set_limit(monkeypatch, 3)
    # Short enough to stay well inside the waits below. A losing attempt for
    # the task that just won can hold its reservation until that execution
    # ends; releasing it signals nothing (see dispatch_one_task_command), so
    # the slot is taken at the next idle poll. What this test proves is the
    # cap admitting exactly one, which no idle poll can change;
    # test_release_wakeups_outcome[1] proves a freed slot wakes the dispatcher.
    monkeypatch.setattr(transport, "DISPATCHER_IDLE_SECONDS", 1.0)
    for _ in range(2):
        host.live(host.new_task())
    rolling_trials, concurrent_trials = 25, 30
    enqueue_starts(host, 6 + rolling_trials + 4 * concurrent_trials)

    # One wake-up admits exactly one start, and each freed slot admits exactly
    # one more, through the real worker loop.
    transport.start_task_command_dispatcher(host.handler)
    await eventually(lambda: transport._dispatcher_wakeup is not None)
    transport.notify_task_command_dispatcher()
    await eventually(lambda: len(host.claimed) >= 1)
    await asyncio.sleep(0.01)
    assert len(host.claimed) == 1
    for trial in range(rolling_trials):
        before = len(host.claimed)
        victim = host.held.pop(next(i for i, h in enumerate(host.held) if not h.done()))
        victim.cancel()
        await asyncio.gather(victim, return_exceptions=True)
        await eventually(lambda: len(host.claimed) >= before + 1)
        await asyncio.sleep(0.01)
        assert len(host.claimed) == before + 1, trial
    assert max(host.registered_active_sizes) <= 3
    await transport.stop_task_command_dispatcher()
    # A losing attempt for the task that just won waits in execute_command for
    # that execution to end. Stopping the dispatcher interrupts it, and the
    # owner may then close and cancel that execution. Let every owner settle,
    # then free slots until exactly one is open, whatever the stop freed.
    await eventually(
        lambda: all(
            coordinator.state == runtime.CoordinatorState.ACTIVE
            and not coordinator._command_tasks
            for coordinator in host.registry._coordinators.values()
        )
    )
    while len(host.manager._active_task_ids()) >= 3:
        victim = host.held.pop(next(i for i, h in enumerate(host.held) if not h.done()))
        victim.cancel()
        await asyncio.gather(victim, return_exceptions=True)
    assert len(host.manager._active_task_ids()) == 2

    # Four attempts that all saw an open cap and hold different candidates
    # race for the one free slot: exactly one may take it.
    real_find = transport._find_command_candidate
    for trial in range(concurrent_trials):
        pending = pending_start_commands(host)[:4]
        barrier = threading.Barrier(4)
        handed_out = iter(pending)
        lock = threading.Lock()

        def find(command_db_id: Any, busy: Any, cap: Any = None) -> Any:
            with lock:
                command_id, task_id = next(handed_out)
            barrier.wait(10)
            return (command_id, task_id, TaskCommandKind.START)

        monkeypatch.setattr(transport, "_find_command_candidate", find)
        before = len(host.claimed)
        results = await asyncio.gather(*(dispatch(host) for _ in range(4)))
        monkeypatch.setattr(transport, "_find_command_candidate", real_find)
        assert results.count(True) == 1, (trial, results)
        assert len(host.claimed) == before + 1, trial
        assert max(host.registered_active_sizes) <= 3, trial
        winner = host.held[-1]
        winner.cancel()
        await asyncio.gather(winner, return_exceptions=True)
        assert not host.manager._dispatch_reservations


async def test_dispatch_reservation_tokens_count_per_attempt(
    cap_host: CapHost,
) -> None:
    manager = cap_host.manager
    first = manager.reserve_dispatch_slot(7, limit=5)
    second = manager.reserve_dispatch_slot(7, limit=5)
    assert first is not None and second is not None
    manager.release_dispatch_slot(first)
    assert 7 in manager._active_task_ids()
    manager.release_dispatch_slot(first)
    assert 7 in manager._active_task_ids()
    manager.release_dispatch_slot(second)
    assert 7 not in manager._active_task_ids()
    assert not manager._dispatch_reservations
    manager.release_dispatch_slot(None)


# --- per-attempt reservation paths -------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "ensure_none",
        "ensure_cancelled",
        "coordinator_busy",
        "claim_matches_no_row",
        "handler_runtime_error",
        "handler_deferred",
        "handler_rejected",
        "coordinator_interrupted",
        "coordinator_closed",
    ],
)
async def test_dispatch_reservation_released_on_every_path(
    cap_host: CapHost, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    host = cap_host
    target = host.new_task()
    host.enqueue(target, "start")
    held_while_handling: list[bool] = []

    if path == "ensure_none":
        monkeypatch.setattr(host.registry, "ensure", AsyncMock(return_value=None))
    elif path == "ensure_cancelled":
        monkeypatch.setattr(
            host.registry, "ensure", AsyncMock(side_effect=asyncio.CancelledError)
        )
    elif path == "coordinator_busy":
        host.enqueue(host.new_task(), "start")
        real_ensure = host.registry.ensure

        async def ensure(task_id: int) -> Any:
            if task_id == target:
                return SimpleNamespace(_command_tasks={object()})
            return await real_ensure(task_id)

        monkeypatch.setattr(host.registry, "ensure", ensure)
    elif path == "claim_matches_no_row":
        from xagent.web.services import task_coordinator_service as lease_service

        real_lock = lease_service.lock_task_lease_no_commit

        def lock_after_competing_update(db: Any, lease: Any) -> bool:
            # A second session makes the command unclaimable between the
            # claim's re-query and its UPDATE.
            with host.factory() as other:
                other.query(TaskExecutionCommand).update(
                    {
                        TaskExecutionCommand.retry_available_at: (
                            transport._utc_now() + transport.timedelta(hours=1)
                        )
                    }
                )
                other.commit()
            return real_lock(db, lease)

        monkeypatch.setattr(
            lease_service, "lock_task_lease_no_commit", lock_after_competing_update
        )
    elif path in ("coordinator_interrupted", "coordinator_closed"):
        error = (
            runtime._CoordinatorInterrupted
            if path == "coordinator_interrupted"
            else runtime._CoordinatorClosed
        )

        async def execute_command(self: Any, command: Any, apply: Any) -> bool:
            held_while_handling.append(
                command.task_id in host.manager._dispatch_reservations
            )
            raise error()

        monkeypatch.setattr(runtime.TaskCoordinator, "execute_command", execute_command)
    else:
        failure = {
            "handler_runtime_error": RuntimeError("boom"),
            "handler_deferred": TaskCommandDeferred("busy"),
            "handler_rejected": TaskCommandRejected("refused"),
        }[path]

        async def failing(command: Any) -> dict[str, Any]:
            held_while_handling.append(
                command.task_id in host.manager._dispatch_reservations
            )
            raise failure

        host.handler = failing  # type: ignore[method-assign]

    if path == "ensure_cancelled":
        with pytest.raises(asyncio.CancelledError):
            await dispatch(host)
    else:
        try:
            await dispatch(host)
        except RuntimeError:
            assert path == "handler_runtime_error"
    if path.startswith(("handler_", "coordinator_i", "coordinator_cl")):
        assert held_while_handling == [True]
    assert not host.manager._dispatch_reservations
    if path == "coordinator_busy":
        # The busy task was skipped; the next candidate claimed and registered.
        assert [task_id for task_id, _ in host.claimed] == [target + 1]
        assert host.manager._active_task_ids() == frozenset({target + 1})
    else:
        assert host.manager._active_task_ids() == frozenset()


@pytest.fixture
async def real_host(tmp_path, monkeypatch):
    """Shared execution over a real database: the production handlers run."""

    from xagent.web.models.agent import Agent
    from xagent.web.models.database import get_engine, get_session_local, init_db
    from xagent.web.services import task_event_bridge, task_lease_service

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'real.db'}")
    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    for module in (runtime, transport, task_lease_service):
        monkeypatch.setattr(module, "get_runner_id", lambda: "worker-1")
    monkeypatch.setattr(runtime, "get_task_lease_heartbeat_seconds", lambda: 0.5)
    monkeypatch.setattr(task_event_bridge, "_bridge", Mock())
    manager = BackgroundTaskManager()
    monkeypatch.setattr(task_execution, "background_task_manager", manager)
    init_db(db_url=f"sqlite:///{tmp_path / 'real.db'}")
    with get_session_local()() as db:
        account = User(username="account", password_hash="unused")
        db.add(account)
        db.flush()
        agent = Agent(user_id=account.id, name="test agent")
        db.add(agent)
        db.commit()
        account_id, agent_id = int(account.id), int(agent.id)
    yield SimpleNamespace(user_id=account_id, agent=agent_id, manager=manager)
    await runtime.close_task_coordinators()
    await manager.shutdown()
    Base.metadata.drop_all(bind=get_engine())
    get_engine().dispose()


@pytest.mark.parametrize("kind", ["start", "resume", "resume_input", "message"])
async def test_dispatch_reservation_released_after_real_handler(
    real_host: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """The reservation hands over to the live handle in the handler's own step."""

    from contextlib import ExitStack
    from unittest.mock import MagicMock, patch

    from xagent.web.api import websocket as websocket_api
    from xagent.web.models.database import get_session_local
    from xagent.web.models.task import TaskStatus

    manager = real_host.manager
    release = asyncio.Event()
    stack = ExitStack()

    async def execute_background(**_kwargs: Any) -> None:
        await release.wait()

    monkeypatch.setattr(task_execution, "execute_task_background", execute_background)
    monkeypatch.setattr(task_execution, "execute_resume_background", execute_background)

    def stage(
        command_kind: TaskCommandKind | None, payload: dict, **task_fields: Any
    ) -> int:
        with get_session_local()() as db:
            task = Task(
                user_id=real_host.user_id,
                agent_id=real_host.agent,
                title="task",
                description="task",
                source="sdk",
                run_id="run-a",
                execution_mode="balanced",
                **task_fields,
            )
            db.add(task)
            db.commit()
            if command_kind is not None:
                transport.enqueue_task_command(
                    db,
                    task_id=int(task.id),
                    actor_user_id=real_host.user_id,
                    command_id=f"{kind}-1",
                    kind=command_kind,
                    payload=payload,
                )
            return int(task.id)

    if kind in ("start", "message"):
        # The turn is scheduled by the real orchestrator; only its runner is stubbed.
        monkeypatch.setattr(
            task_orchestrator,
            "load_task_setup_snapshot_sync",
            lambda *a, **kw: object(),
        )
        monkeypatch.setattr(
            task_orchestrator, "resolve_execution_scope", lambda *a: object()
        )
        monkeypatch.setattr(task_orchestrator, "_get_agent_manager", Mock())
    if kind in ("message", "resume"):
        connections = MagicMock()
        connections.connections_for_task.return_value = []
        connections.send_personal_message = AsyncMock()
        connections.broadcast_to_task = AsyncMock()
        stack.enter_context(patch.object(websocket_api, "manager", connections))
        stack.enter_context(
            patch.object(
                websocket_api._command_origins, "resolve", return_value=MagicMock()
            )
        )

    if kind == "start":
        created = await task_start.create_sdk_task(
            agent_id=real_host.agent,
            task_owner_user_id=real_host.user_id,
            actor_user_id=real_host.user_id,
            message="first",
            file_ids=(),
            connector_runtime_context=(),
            timezone=None,
        )
        target = int(created.task_id)
    elif kind == "message":
        # The previous run ended, so the message opens a new turn.
        target = stage(
            TaskCommandKind.MESSAGE,
            {
                "type": "chat_message",
                "message": "again",
                "client_message_id": "message-1",
                "files": [],
            },
            status=TaskStatus.COMPLETED,
            control_state="completed",
            state_version=3,
        )
    elif kind == "resume":
        from xagent.web.services.task_execution_controller import TaskControlState

        target = stage(
            TaskCommandKind.RESUME,
            {"type": "resume_task"},
            status=TaskStatus.PAUSED,
            control_state=TaskControlState.RESUME_REQUESTED.value,
        )
        agent = MagicMock()
        agent.supports_live_control.return_value = True
        stack.enter_context(
            patch(
                "xagent.web.services.agent_service_manager.get_agent_manager",
                return_value=MagicMock(
                    get_agent_for_task=AsyncMock(return_value=agent)
                ),
            )
        )
        stack.enter_context(
            patch.object(
                task_command_execution,
                "resolve_execution_scope_off_turn",
                return_value=None,
            )
        )
        stack.enter_context(
            patch.object(
                websocket_api.task_execution_controller,
                "transition",
                new=AsyncMock(
                    return_value=SimpleNamespace(
                        run_id="run-a", status=TaskStatus.PAUSED
                    )
                ),
            )
        )
    else:
        from xagent.core.agent.context import ContextManager
        from xagent.core.agent.registry import ExecutionRegistry
        from xagent.core.agent.runner import AgentRunner

        target = stage(
            None,
            {},
            status=TaskStatus.WAITING_FOR_USER,
            control_state="waiting_for_user",
            state_version=2,
        )
        contexts = ContextManager()
        contexts.create_context(str(target)).add_user_message("original")
        tracer = SimpleNamespace(
            load_latest_checkpoint=AsyncMock(return_value=None), checkpoint=AsyncMock()
        )
        runner = AgentRunner(
            SimpleNamespace(llm=None), tracer=tracer, context_manager=contexts
        )
        registry = ExecutionRegistry()
        registry.register(str(target), runner)
        subscription = registry.subscribe(lambda _: registry.unsubscribe(subscription))

        async def inject(execution_id: str, **kwargs: Any) -> Any:
            return (await registry.post_user_message(execution_id, **kwargs)).outcome

        monkeypatch.setattr(
            task_resume.agent_runtime_service,
            "get_agent_manager",
            lambda: SimpleNamespace(
                get_agent_for_task=AsyncMock(
                    return_value=SimpleNamespace(
                        post_user_message=AsyncMock(side_effect=inject)
                    )
                )
            ),
        )
        task_resume_command._admit_reply(
            task_resume.TaskReplyInput(
                task_id=target,
                agent_id=real_host.agent,
                task_owner_user_id=real_host.user_id,
                actor_user_id=real_host.user_id,
                run_id="run-a",
                status=TaskStatus.WAITING_FOR_USER,
                text="answer",
                command_id="reply-1",
            ),
            "sdk",
            "",
            "reply-1",
        )

    held_at_entry: list[bool] = []
    live_at_return: list[bool] = []

    async def handler(command: Any) -> Any:
        held_at_entry.append(command.task_id in manager._dispatch_reservations)
        result = await task_command_execution.execute_durable_task_command(command)
        live_at_return.append(command.task_id in manager._live_executions)
        return result

    try:
        with stack:
            assert await transport.dispatch_one_task_command(handler)
        assert held_at_entry == [True]
        assert live_at_return == [True]
        assert not manager._dispatch_reservations
        assert target in manager._active_task_ids()
    finally:
        release.set()
        await manager.shutdown()


@pytest.mark.parametrize("kind", ["resume", "resume_input"])
async def test_resume_counted_until_promotion(
    cap_host: CapHost, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    host = cap_host
    manager = host.manager
    target = host.new_task()
    host.enqueue(target, kind)
    promote, finish = asyncio.Event(), asyncio.Event()
    seen: dict[str, bool] = {}
    resume_handle: list[asyncio.Task] = []

    async def body() -> None:
        await promote.wait()
        manager.promote_resume_task(target, resume_handle[0])
        seen["promoted"] = True
        await finish.wait()

    async def resume_handler(command: Any) -> dict[str, Any]:
        seen["before_registration"] = target in manager._active_task_ids()
        assert (
            manager.try_reserve_resume(
                target, expected_run_id=task_execution.ANY_RESUME_RUN
            )
            is task_execution.ResumeReservationOutcome.RESERVED
        )
        handle = asyncio.get_running_loop().create_task(body())
        host.held.append(handle)
        resume_handle.append(handle)
        manager.register_reserved_resume(target, handle, run_id="run")
        return {"ok": True}

    host.handler = resume_handler  # type: ignore[method-assign]
    assert await dispatch(host)
    assert seen["before_registration"]
    assert not manager._dispatch_reservations
    assert target in manager._active_task_ids()

    promote.set()
    await eventually(lambda: seen.get("promoted"))
    assert manager.running_tasks[target] is resume_handle[0]
    assert target in manager._active_task_ids()
    finish.set()
    await asyncio.gather(*host.held)
    await eventually(lambda: target not in manager._active_task_ids())


def take_last_slot_after_snapshot(
    host: CapHost, monkeypatch: pytest.MonkeyPatch, other_task_id: int
) -> None:
    """Make the first reservation find the cap full although its snapshot was open."""

    real_reserve = host.manager.reserve_dispatch_slot
    calls: list[int] = []

    def reserve(task_id: int, *, limit: int) -> Any:
        if not calls:
            calls.append(task_id)
            assert real_reserve(other_task_id, limit=limit) is not None
        return real_reserve(task_id, limit=limit)

    monkeypatch.setattr(host.manager, "reserve_dispatch_slot", reserve)


# --- closed cap --------------------------------------------------------------


def assert_untouched(host: CapHost, command_id: int, task_id: int) -> None:
    row = host.row(command_id)
    assert (row.status, row.attempt_count, row.defer_count, row.failure_count) == (
        "pending",
        0,
        0,
        0,
    )
    with host.factory() as db:
        assert db.get(Task, task_id).runner_id is None
    assert host.terminal_events() == 0


@pytest.mark.parametrize(
    "kind", ["start", "resume", "resume_input", "message", "snapshot_open_but_full"]
)
async def test_closed_cap_holds_execution_command_without_side_effects(
    cap_host: CapHost,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    host = cap_host
    ensure_spy = AsyncMock(wraps=host.registry.ensure)
    host.registry.ensure = ensure_spy  # type: ignore[method-assign]
    set_limit(monkeypatch, 1)
    unrelated = host.new_task()
    target = host.new_task()
    if kind == "snapshot_open_but_full":
        command_id = host.enqueue(target, "start")
        take_last_slot_after_snapshot(host, monkeypatch, unrelated)
    else:
        host.live(unrelated)
        command_id = host.enqueue(target, kind)

    assert await dispatch(host) is False

    ensure_spy.assert_not_awaited()
    assert host.claimed == []
    assert_untouched(host, command_id, target)


@pytest.mark.parametrize(
    ("kind", "task_is_active"),
    [
        ("pause", False),
        ("cancel", False),
        ("message", True),
        ("start", True),
        ("resume_input", True),
        ("resume", True),
    ],
)
async def test_closed_cap_claims_controls_and_active_task_commands(
    cap_host: CapHost, monkeypatch: pytest.MonkeyPatch, kind: str, task_is_active: bool
) -> None:
    host = cap_host
    set_limit(monkeypatch, 1)
    target = host.new_task()
    if task_is_active:
        host.live(target)
        held_start = None
    else:
        host.live(host.new_task())
        # An older held start of another task: FIFO would return it first.
        held_start = host.enqueue(host.new_task(), "start")
    command_id = host.enqueue(target, kind)
    snapshot = host.manager.dispatch_cap_snapshot(1)
    assert snapshot.closed

    with host.factory() as db:
        sql = str(
            transport._claimable_query(
                db, runner_id="worker", command_db_id=None, cap=snapshot
            ).statement.compile(compile_kwargs={"literal_binds": True})
        )
    assert "task_execution_commands.task_id IN (" in sql
    for active in sorted(snapshot.active_ids):
        assert f"task_id IN ({active})" in sql

    if held_start is not None:
        assert transport._find_command_candidate(None, (), snapshot) == (
            command_id,
            target,
            TaskCommandKind(kind),
        )
    assert await dispatch(host) is True
    assert host.claimed == [(target, TaskCommandKind(kind))]
    if task_is_active:
        assert len(host.manager._active_task_ids()) == 1
    if held_start is not None:
        assert host.row(held_start).status == "pending"


async def test_candidate_and_claim_use_one_loop_snapshot(
    cap_host: CapHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = cap_host
    host.enqueue(host.new_task(), "message")
    caps: list[Any] = []
    real_query = transport._claimable_query

    def recording_query(*args: Any, **kwargs: Any) -> Any:
        caps.append(kwargs.get("cap"))
        return real_query(*args, **kwargs)

    monkeypatch.setattr(transport, "_claimable_query", recording_query)
    builders: list[tuple[int, Any]] = []
    real_build = transport.build_dispatch_cap_snapshot

    def recording_build() -> Any:
        snapshot = real_build()
        builders.append((threading.get_ident(), snapshot))
        return snapshot

    monkeypatch.setattr(transport, "build_dispatch_cap_snapshot", recording_build)

    assert await dispatch(host)

    assert len(caps) >= 2
    assert all(cap is caps[0] for cap in caps)
    assert caps[0] is not None
    assert [snapshot for _, snapshot in builders] == [caps[0]]
    assert {thread for thread, _ in builders} == {threading.get_ident()}


# --- wake-ups ----------------------------------------------------------------


class Attempts:
    """Counts dispatch attempts made by the real worker loop."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.count = 0
        real = transport.dispatch_one_task_command

        async def counting(*args: Any, **kwargs: Any) -> bool:
            self.count += 1
            return await real(*args, **kwargs)

        monkeypatch.setattr(transport, "dispatch_one_task_command", counting)


@pytest.mark.parametrize("row", [1, 2, 3, 4])
async def test_release_wakeups_outcome(
    cap_host: CapHost, monkeypatch: pytest.MonkeyPatch, row: int
) -> None:
    host = cap_host
    attempts = Attempts(monkeypatch)
    if row == 1:
        # Idle waits never time out within the test, so only the end of an
        # execution can wake the dispatcher. That proves the wake-up without a
        # wall-clock bound, which a loaded CI runner can miss (#3141).
        monkeypatch.setattr(transport, "DISPATCHER_IDLE_SECONDS", 3600.0)
    elif row != 4:
        monkeypatch.setattr(transport, "DISPATCHER_IDLE_SECONDS", 30.0)

    if row == 1:
        set_limit(monkeypatch, 2)
        lives = [host.live(host.new_task()) for _ in range(2)]
        pending = host.enqueue(host.new_task(), "start")
        transport.start_task_command_dispatcher(host.handler)
        # Every worker has taken its cap snapshot, so none can see the freed
        # slot through a first attempt that starts late.
        await eventually(lambda: attempts.count >= transport.DISPATCHER_CONCURRENCY)
        await asyncio.sleep(0.2)
        assert host.row(pending).status == "pending"
        lives[0][1].set()
        await eventually(lambda: len(host.claimed) == 1)
    elif row in (2, 3):
        set_limit(monkeypatch, 80)
        if row == 3:
            for _ in range(79):
                host.live(host.new_task())
        target = host.new_task(runner_id="worker")
        host.enqueue(target, "message")
        transport.start_task_command_dispatcher(host.handler)
        await eventually(lambda: attempts.count >= 1)
        await asyncio.sleep(0.05)
        first = attempts.count
        await asyncio.sleep(0.4)
        assert attempts.count - first <= 2 * transport.DISPATCHER_CONCURRENCY
        assert host.claimed == []
    else:
        set_limit(monkeypatch, 10)
        lives = [host.live(host.new_task()) for _ in range(10)]

        def failing(*args: Any, **kwargs: Any) -> Any:
            raise OperationalError("SELECT 1", {}, Exception("database unavailable"))

        monkeypatch.setattr(transport, "_find_command_candidate", failing)
        transport.start_task_command_dispatcher(host.handler)
        await eventually(lambda: attempts.count >= 1)
        started = time.monotonic()
        await asyncio.sleep(0.2)
        # Ending one execution at a time: each end would cut a back-off wait
        # short if the dispatcher listened to it there.
        for _, finish in lives:
            finish.set()
            await asyncio.sleep(0.1)
        await asyncio.sleep(3.0 - (time.monotonic() - started))
        # Four workers, backoff 0.5 s then 1 s then 2 s: three attempts each.
        assert attempts.count == 3 * transport.DISPATCHER_CONCURRENCY


@pytest.mark.parametrize(
    "row",
    [
        "register_task_done",
        "reserved_resume_done",
        "cancel_with_live_handle",
        "cancel_with_nothing_registered",
        "release_signals_nothing",
    ],
)
async def test_active_set_reductions_notify_dispatcher(
    monkeypatch: pytest.MonkeyPatch, row: str
) -> None:
    manager = BackgroundTaskManager()
    loop = asyncio.get_running_loop()
    wakeup, slot_freed = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr(transport, "_dispatcher_loop", loop)
    monkeypatch.setattr(transport, "_dispatcher_wakeup", wakeup)
    monkeypatch.setattr(transport, "_dispatcher_slot_freed", slot_freed)

    if row == "register_task_done":
        handle = loop.create_task(asyncio.sleep(0))
        manager.register_task(1, handle)
        await handle
    elif row == "reserved_resume_done":
        assert manager.reserve_resume(2)
        handle = loop.create_task(asyncio.sleep(0))
        manager.register_reserved_resume(2, handle, run_id="run")
        await handle
    elif row == "cancel_with_live_handle":
        # Placed in the table directly so that no done callback signals: only
        # cancel_task's own signal can set the event.
        handle = loop.create_task(asyncio.Event().wait())
        manager.running_tasks[3] = handle
        await manager.cancel_task(3)
    elif row == "cancel_with_nothing_registered":
        await manager.cancel_task(4)
    else:
        token = manager.reserve_dispatch_slot(5, limit=3)
        manager.release_dispatch_slot(token)
    for _ in range(3):
        await asyncio.sleep(0)

    assert not wakeup.is_set()
    assert slot_freed.is_set() is (row != "release_signals_nothing")


# --- prompt dispatch and legacy mode -----------------------------------------


@pytest.mark.parametrize("row", ["start", "cancel", "start_after_snapshot"])
async def test_prompt_dispatch_honours_cap(
    cap_host: CapHost, monkeypatch: pytest.MonkeyPatch, row: str
) -> None:
    host = cap_host
    set_limit(monkeypatch, 1)
    target = host.new_task()
    unrelated = host.new_task()
    if row == "start_after_snapshot":
        command_id = host.enqueue(target, "start")
        take_last_slot_after_snapshot(host, monkeypatch, unrelated)
    else:
        host.live(unrelated)
        command_id = host.enqueue(target, row)

    await transport.dispatch_task_command_promptly(
        host.handler, command_db_id=command_id
    )
    await asyncio.gather(*list(transport._prompt_dispatch_tasks))

    if row == "cancel":
        assert host.claimed == [(target, TaskCommandKind.CANCEL)]
    else:
        assert host.claimed == []
        assert host.row(command_id).status == "pending"


async def test_legacy_dispatch_ignores_cap(
    cap_host: CapHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = cap_host
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "false")
    set_limit(monkeypatch, 1)
    host.live(host.new_task())
    message_task, resume_task = host.new_task(), host.new_task()
    host.enqueue(message_task, "message")
    host.enqueue(resume_task, "resume")

    assert await dispatch(host)
    assert await dispatch(host)

    assert sorted(host.claimed, key=lambda item: item[0]) == [
        (message_task, TaskCommandKind.MESSAGE),
        (resume_task, TaskCommandKind.RESUME),
    ]
