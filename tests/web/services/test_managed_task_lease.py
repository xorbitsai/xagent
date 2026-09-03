from __future__ import annotations

import ast
import asyncio
import inspect
import logging
import threading
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError

import xagent.config as config
import xagent.web.services.interaction_rollout as ir
import xagent.web.services.managed_task_lease as managed_task_lease
from xagent.web.models.agent import Agent
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce, WorkforceRun
from xagent.web.services.managed_task_lease import (
    ManagedTaskLease,
    _finalize_managed_task_lease_result_sync,
    claim_managed_task_lease,
    claim_managed_task_lease_isolated,
    finalize_managed_task_lease_result,
    finalize_managed_task_lease_result_isolated,
    start_managed_task_lease,
)
from xagent.web.services.task_lease_service import (
    TaskLease,
    TaskLeaseHeartbeatOutcome,
    acquire_task_lease,
)


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'managed-lease.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


def _create_task(db) -> Task:
    user = User(username="managed-lease-user", password_hash="hash", is_admin=False)
    db.add(user)
    db.commit()
    task = Task(
        user_id=user.id,
        title="Managed lease",
        description="Managed lease",
        status=TaskStatus.PENDING,
        execution_mode="auto",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def test_finalize_managed_result_commits_exact_status_and_transcript(
    db_session,
) -> None:
    task = _create_task(db_session)
    lease = acquire_task_lease(db_session, int(task.id), new_run=True)
    assert lease is not None

    assert (
        finalize_managed_task_lease_result(
            db_session,
            lease,
            status=TaskStatus.COMPLETED,
            assistant_content="completed inline",
            turn_id="accepted-turn",
            interactions=[{"type": "question", "content": "continue?"}],
            message_type="assistant_response",
        )
        is True
    )

    db_session.refresh(task)
    assert task.status == TaskStatus.COMPLETED
    assert task.runner_id is None
    assert task.lease_expires_at is None
    stored = (
        db_session.query(TaskChatMessage)
        .filter(
            TaskChatMessage.task_id == int(task.id),
            TaskChatMessage.role == "assistant",
        )
        .one()
    )
    assert "completed inline" in stored.content
    assert stored.turn_id == "accepted-turn"


def test_finalize_managed_result_rejects_replacement_owner(db_session) -> None:
    task = _create_task(db_session)
    lease = acquire_task_lease(db_session, int(task.id), new_run=True)
    assert lease is not None
    task.runner_id = "replacement-runner"
    task.run_id = "replacement-run"
    db_session.commit()

    assert (
        finalize_managed_task_lease_result(
            db_session,
            lease,
            status=TaskStatus.COMPLETED,
            assistant_content="stale result",
        )
        is False
    )

    db_session.refresh(task)
    assert task.status == TaskStatus.RUNNING
    assert task.runner_id == "replacement-runner"
    assert (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.task_id == int(task.id))
        .count()
        == 0
    )


@pytest.mark.asyncio
async def test_isolated_finalize_rejects_replacement_runner_with_same_run_id(
    db_session,
) -> None:
    task = _create_task(db_session)
    lease = acquire_task_lease(db_session, int(task.id), new_run=True)
    assert lease is not None
    assert lease.run_id is not None

    task.runner_id = "replacement-runner"
    task.run_id = lease.run_id
    db_session.commit()

    assert (
        await finalize_managed_task_lease_result_isolated(
            lease,
            status=TaskStatus.FAILED,
        )
        is False
    )

    db_session.refresh(task)
    assert task.status == TaskStatus.RUNNING
    assert task.runner_id == "replacement-runner"
    assert task.run_id == lease.run_id
    assert task.lease_expires_at is not None


@pytest.mark.asyncio
async def test_isolated_finalize_keeps_event_loop_responsive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_started = threading.Event()
    allow_worker = threading.Event()

    def blocking_finalize(*_args, **_kwargs) -> bool:  # type: ignore[no-untyped-def]
        worker_started.set()
        assert allow_worker.wait(timeout=2)
        return True

    monkeypatch.setattr(
        "xagent.web.services.managed_task_lease."
        "_finalize_managed_task_lease_result_sync",
        blocking_finalize,
    )

    settlement = asyncio.create_task(
        finalize_managed_task_lease_result_isolated(
            TaskLease(task_id=7, runner_id="runner-a", run_id="run-a"),
            status=TaskStatus.FAILED,
        )
    )
    await asyncio.to_thread(worker_started.wait, 2)
    await asyncio.sleep(0)

    assert settlement.done() is False
    allow_worker.set()
    assert await settlement is True


@pytest.mark.asyncio
async def test_managed_finalize_stops_heartbeat_before_exact_settlement() -> None:
    heartbeat_task = asyncio.create_task(asyncio.sleep(0))
    managed = ManagedTaskLease(
        lease=TaskLease(task_id=17, runner_id="runner-a", run_id="run-a"),
        stop_event=asyncio.Event(),
        heartbeat_task=heartbeat_task,  # type: ignore[arg-type]
    )
    events: list[str] = []

    async def stop_heartbeat(*_args, **_kwargs) -> TaskLeaseHeartbeatOutcome:
        events.append("stop-heartbeat")
        return TaskLeaseHeartbeatOutcome()

    async def finalize(*_args, **_kwargs) -> bool:
        events.append("finalize")
        return True

    with (
        patch(
            "xagent.web.services.managed_task_lease.stop_task_lease_heartbeat",
            side_effect=stop_heartbeat,
        ),
        patch(
            "xagent.web.services.managed_task_lease."
            "finalize_managed_task_lease_result_isolated",
            side_effect=finalize,
        ),
    ):
        assert (
            await managed.finalize_result(
                status=TaskStatus.COMPLETED,
                assistant_content="done",
            )
            is True
        )
        assert await managed.close() is False

    assert events == ["stop-heartbeat", "finalize"]


@pytest.mark.asyncio
async def test_claim_projects_workforce_run_to_running_in_claim_transaction(
    db_session,
) -> None:
    user = User(username="workforce-claim-user", password_hash="hash", is_admin=False)
    manager = Agent(user=user, name="Managed claim manager")
    db_session.add_all([user, manager])
    db_session.flush()
    workforce = Workforce(
        owner_user_id=user.id,
        scope_type="user",
        scope_id=str(user.id),
        name="Managed claim workforce",
        manager_agent_id=manager.id,
        status="active",
    )
    db_session.add(workforce)
    db_session.flush()
    task = Task(
        user_id=user.id,
        title="Managed workforce claim",
        description="Managed workforce claim",
        status=TaskStatus.PENDING,
        execution_mode="auto",
        agent_id=manager.id,
        agent_config={},
    )
    db_session.add(task)
    db_session.flush()
    run = WorkforceRun(
        workforce_id=workforce.id,
        task_id=task.id,
        user_id=user.id,
        status="pending",
        snapshot={},
    )
    db_session.add(run)
    db_session.flush()
    task.agent_config = {"workforce_run_id": int(run.id)}
    db_session.commit()

    managed = claim_managed_task_lease(db_session, int(task.id))

    assert managed is not None
    db_session.refresh(run)
    assert run.status == "running"
    assert await managed.close() is True


@pytest.mark.asyncio
async def test_claim_new_run_clears_stale_terminal_snapshot(db_session) -> None:
    task = _create_task(db_session)
    task.status = TaskStatus.COMPLETED
    task.output = "previous answer"
    task.error_message = "previous error"
    db_session.commit()

    managed = claim_managed_task_lease(db_session, int(task.id))

    assert managed is not None
    db_session.refresh(task)
    assert task.status == TaskStatus.RUNNING
    assert task.output is None
    assert task.error_message is None
    assert await managed.close() is True


@pytest.mark.asyncio
async def test_isolated_claim_keeps_event_loop_responsive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_started = threading.Event()
    allow_worker = threading.Event()
    lease = TaskLease(task_id=23, runner_id="runner-a", run_id="run-a")
    managed = object()

    def blocking_claim(_task_id: int) -> TaskLease:
        worker_started.set()
        assert allow_worker.wait(timeout=2)
        return lease

    monkeypatch.setattr(
        "xagent.web.services.managed_task_lease._claim_managed_task_lease_sync",
        blocking_claim,
    )
    monkeypatch.setattr(
        "xagent.web.services.managed_task_lease.start_managed_task_lease",
        lambda claimed: managed if claimed is lease else None,
    )

    claim = asyncio.create_task(claim_managed_task_lease_isolated(23))
    assert await asyncio.to_thread(worker_started.wait, 1)
    await asyncio.sleep(0)
    assert claim.done() is False

    allow_worker.set()
    assert await claim is managed


@pytest.mark.asyncio
async def test_isolated_claim_cancellation_settles_late_committed_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_started = threading.Event()
    allow_worker = threading.Event()
    cleanup_finished = threading.Event()
    lease = TaskLease(task_id=24, runner_id="runner-a", run_id="run-a")

    def blocking_claim(_task_id: int) -> TaskLease:
        worker_started.set()
        assert allow_worker.wait(timeout=2)
        return lease

    def cleanup(claimed: TaskLease) -> bool:
        assert claimed is lease
        cleanup_finished.set()
        return True

    monkeypatch.setattr(
        "xagent.web.services.managed_task_lease._claim_managed_task_lease_sync",
        blocking_claim,
    )
    monkeypatch.setattr(
        "xagent.web.services.managed_task_lease._release_managed_task_lease_sync",
        cleanup,
    )

    claim = asyncio.create_task(claim_managed_task_lease_isolated(24))
    assert await asyncio.to_thread(worker_started.wait, 1)
    claim.cancel()
    await asyncio.sleep(0)
    assert claim.done() is False

    allow_worker.set()
    with pytest.raises(asyncio.CancelledError):
        await claim
    assert cleanup_finished.is_set()


def test_claim_rolls_back_lease_when_running_projection_fails(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _create_task(db_session)

    def fail_projection(*_args, **_kwargs) -> bool:  # type: ignore[no-untyped-def]
        raise RuntimeError("workforce projection failed")

    monkeypatch.setattr(
        "xagent.web.services.managed_task_lease.sync_workforce_run_status",
        fail_projection,
    )

    with pytest.raises(RuntimeError, match="workforce projection failed"):
        claim_managed_task_lease(db_session, int(task.id))

    db_session.refresh(task)
    assert task.status == TaskStatus.PENDING
    assert task.runner_id is None
    assert task.run_id is None


@pytest.mark.asyncio
async def test_managed_lease_releases_terminal_task(db_session) -> None:
    task = _create_task(db_session)
    managed = claim_managed_task_lease(db_session, int(task.id))
    assert managed is not None
    assert claim_managed_task_lease(db_session, int(task.id)) is None
    task.status = TaskStatus.COMPLETED
    task.control_state = "completed"
    db_session.commit()

    assert await managed.close() is True
    assert await managed.close() is False
    db_session.refresh(task)
    assert task.status == TaskStatus.COMPLETED
    assert task.runner_id is None
    assert task.lease_expires_at is None


@pytest.mark.asyncio
async def test_managed_lease_fails_an_unfinished_task(db_session) -> None:
    task = _create_task(db_session)
    lease = acquire_task_lease(db_session, int(task.id), new_run=True)
    assert lease is not None
    managed = start_managed_task_lease(lease)

    assert await managed.close() is True
    db_session.refresh(task)
    assert task.status == TaskStatus.FAILED
    assert task.control_state == "failed"
    assert task.runner_id is None


@pytest.mark.asyncio
async def test_managed_lease_heartbeat_timeout_skips_release_checkout() -> None:
    heartbeat_task = asyncio.create_task(asyncio.sleep(0))
    managed = ManagedTaskLease(
        lease=TaskLease(task_id=7, runner_id="runner-a", run_id="run-a"),
        stop_event=asyncio.Event(),
        heartbeat_task=heartbeat_task,  # type: ignore[arg-type]
    )
    timeout = SQLAlchemyTimeoutError("heartbeat pool timeout")

    with (
        patch(
            "xagent.web.services.managed_task_lease.stop_task_lease_heartbeat",
            new=AsyncMock(return_value=TaskLeaseHeartbeatOutcome(pool_timeout=timeout)),
        ),
        patch(
            "xagent.web.services.managed_task_lease._release_managed_task_lease_sync"
        ) as release,
    ):
        assert await managed.close() is False

    release.assert_not_called()


@pytest.mark.asyncio
async def test_managed_lease_close_drains_release_before_cancellation() -> None:
    heartbeat_task = asyncio.create_task(asyncio.sleep(0))
    managed = ManagedTaskLease(
        lease=TaskLease(task_id=8, runner_id="runner-a", run_id="run-a"),
        stop_event=asyncio.Event(),
        heartbeat_task=heartbeat_task,  # type: ignore[arg-type]
    )
    release_started = threading.Event()
    allow_release = threading.Event()

    def blocking_release(_lease: TaskLease) -> bool:
        release_started.set()
        assert allow_release.wait(timeout=2)
        return True

    with (
        patch(
            "xagent.web.services.managed_task_lease.stop_task_lease_heartbeat",
            new=AsyncMock(return_value=TaskLeaseHeartbeatOutcome()),
        ),
        patch(
            "xagent.web.services.managed_task_lease._release_managed_task_lease_sync",
            side_effect=blocking_release,
        ),
    ):
        closing = asyncio.create_task(managed.close())
        assert await asyncio.to_thread(release_started.wait, 1)
        closing.cancel()
        await asyncio.sleep(0.02)
        assert not closing.done()
        allow_release.set()
        with pytest.raises(asyncio.CancelledError):
            await closing


@pytest.mark.asyncio
async def test_managed_lease_close_drains_heartbeat_before_cancellation() -> None:
    heartbeat_task = asyncio.create_task(asyncio.sleep(0))
    managed = ManagedTaskLease(
        lease=TaskLease(task_id=9, runner_id="runner-a", run_id="run-a"),
        stop_event=asyncio.Event(),
        heartbeat_task=heartbeat_task,  # type: ignore[arg-type]
    )
    heartbeat_stop_started = asyncio.Event()
    allow_heartbeat_stop = asyncio.Event()

    async def blocking_heartbeat_stop(*_args) -> TaskLeaseHeartbeatOutcome:
        heartbeat_stop_started.set()
        await allow_heartbeat_stop.wait()
        return TaskLeaseHeartbeatOutcome()

    with (
        patch(
            "xagent.web.services.managed_task_lease.stop_task_lease_heartbeat",
            side_effect=blocking_heartbeat_stop,
        ),
        patch(
            "xagent.web.services.managed_task_lease._release_managed_task_lease_sync",
            return_value=True,
        ) as release,
    ):
        closing = asyncio.create_task(managed.close())
        await heartbeat_stop_started.wait()
        closing.cancel()
        await asyncio.sleep(0.02)
        assert not closing.done()
        allow_heartbeat_stop.set()
        with pytest.raises(asyncio.CancelledError):
            await closing

    release.assert_called_once_with(managed.lease)


# ---------------------------------------------------------------------------
# execution_result: identity threading through all five finalize layers.
# ---------------------------------------------------------------------------


def test_execution_result_identity_at_finalize_core() -> None:
    """The base layer accepts execution_result and binds it unchanged."""

    sentinel: dict[str, object] = {"marker": object()}
    bound = inspect.signature(finalize_managed_task_lease_result).bind(
        db=object(),
        lease=TaskLease(task_id=201, runner_id="runner-a", run_id="run-a"),
        status=TaskStatus.COMPLETED,
        execution_result=sentinel,
    )
    assert bound.arguments["execution_result"] is sentinel


def test_execution_result_identity_at_finalize_sync(
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel: dict[str, object] = {"marker": object()}
    received: list[object] = []

    def capture_core(_db, _lease, *, execution_result=None, **_kwargs):
        received.append(execution_result)
        return True

    monkeypatch.setattr(
        managed_task_lease, "finalize_managed_task_lease_result", capture_core
    )

    assert (
        _finalize_managed_task_lease_result_sync(
            TaskLease(task_id=202, runner_id="runner-a", run_id="run-a"),
            status=TaskStatus.COMPLETED,
            execution_result=sentinel,
        )
        is True
    )
    assert received == [sentinel]
    assert received[0] is sentinel


@pytest.mark.asyncio
async def test_execution_result_identity_at_finalize_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel: dict[str, object] = {"marker": object()}
    received: list[object] = []

    def capture_sync(_lease, *, execution_result=None, **_kwargs):
        received.append(execution_result)
        return True

    monkeypatch.setattr(
        managed_task_lease,
        "_finalize_managed_task_lease_result_sync",
        capture_sync,
    )

    assert (
        await finalize_managed_task_lease_result_isolated(
            TaskLease(task_id=203, runner_id="runner-a", run_id="run-a"),
            status=TaskStatus.COMPLETED,
            execution_result=sentinel,
        )
        is True
    )
    assert received == [sentinel]
    assert received[0] is sentinel


@pytest.mark.asyncio
async def test_execution_result_identity_at_finalize_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    heartbeat_task = asyncio.create_task(asyncio.sleep(0))
    managed = ManagedTaskLease(
        lease=TaskLease(task_id=204, runner_id="runner-a", run_id="run-a"),
        stop_event=asyncio.Event(),
        heartbeat_task=heartbeat_task,  # type: ignore[arg-type]
    )
    sentinel: dict[str, object] = {"marker": object()}
    received: list[object] = []

    async def capture_finalize_resources(_self, *, execution_result=None, **_kwargs):
        received.append(execution_result)
        return True

    monkeypatch.setattr(
        ManagedTaskLease, "_finalize_resources", capture_finalize_resources
    )

    assert (
        await managed.finalize_result(
            status=TaskStatus.COMPLETED,
            execution_result=sentinel,
        )
        is True
    )
    assert received == [sentinel]
    assert received[0] is sentinel


@pytest.mark.asyncio
async def test_execution_result_identity_at_finalize_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    heartbeat_task = asyncio.create_task(asyncio.sleep(0))
    managed = ManagedTaskLease(
        lease=TaskLease(task_id=205, runner_id="runner-a", run_id="run-a"),
        stop_event=asyncio.Event(),
        heartbeat_task=heartbeat_task,  # type: ignore[arg-type]
    )
    sentinel: dict[str, object] = {"marker": object()}
    received: list[object] = []

    async def stop_heartbeat(*_args, **_kwargs) -> TaskLeaseHeartbeatOutcome:
        return TaskLeaseHeartbeatOutcome()

    async def capture_isolated(_lease, *, execution_result=None, **_kwargs):
        received.append(execution_result)
        return True

    monkeypatch.setattr(managed_task_lease, "stop_task_lease_heartbeat", stop_heartbeat)
    monkeypatch.setattr(
        managed_task_lease,
        "finalize_managed_task_lease_result_isolated",
        capture_isolated,
    )

    assert (
        await managed._finalize_resources(
            status=TaskStatus.COMPLETED,
            assistant_content=None,
            turn_id=None,
            interactions=None,
            message_type="assistant_response",
            error_message=None,
            execution_result=sentinel,
        )
        is True
    )
    assert received == [sentinel]
    assert received[0] is sentinel


# ---------------------------------------------------------------------------
# execution_result: it must never reach a log call or an exception message.
# ---------------------------------------------------------------------------


def test_execution_result_never_reaches_a_log_or_exception_message() -> None:
    """This module must never hand execution_result to a logger call
    or an exception constructor. finalize_managed_task_lease_result_isolated
    hands its call off to a worker thread, so a lambda closure holds this
    mapping across that thread boundary; it may carry large, untruncated
    structures such as file_outputs, so logging it would leak that payload.
    """

    source = inspect.getsource(managed_task_lease)
    tree = ast.parse(source)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_logger_call = (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "logger"
        )
        is_exception_construction = (
            isinstance(func, ast.Name) and func.id.endswith("Error")
        ) or (isinstance(func, ast.Attribute) and func.attr.endswith("Error"))
        if not (is_logger_call or is_exception_construction):
            continue
        for value in list(node.args) + [kw.value for kw in node.keywords]:
            for sub in ast.walk(value):
                if isinstance(sub, ast.Name) and sub.id == "execution_result":
                    offenders.append(ast.dump(node))
    assert offenders == []


def test_finalize_core_never_logs_the_execution_result_payload(
    db_session, caplog: pytest.LogCaptureFixture
) -> None:
    task = _create_task(db_session)
    lease = acquire_task_lease(db_session, int(task.id), new_run=True)
    assert lease is not None
    sentinel_marker = "EXECUTION-RESULT-SENTINEL-4f2c9"

    with caplog.at_level(logging.DEBUG):
        assert (
            finalize_managed_task_lease_result(
                db_session,
                lease,
                status=TaskStatus.COMPLETED,
                assistant_content="done",
                execution_result={"marker": sentinel_marker},
            )
            is True
        )

    assert sentinel_marker not in caplog.text


# ---------------------------------------------------------------------------
# execution_result: draft publication must be dominated by its presence.
# ---------------------------------------------------------------------------


def test_managed_lease_publication_requires_an_execution_result() -> None:
    """Any resolve_publishable_clarification call inside
    finalize_managed_task_lease_result must be dominated by an
    `execution_result is not None` guard.

    This PR adds no such call -- publication is wired up separately -- so
    this pin currently passes on absence. Do not rewrite it to assert the
    call exists: the smallest way to turn this red is adding the call
    without the guard, which is exactly the mistake it exists to catch.

    Two mutations were run against this assertion and both turn it red: a
    mutation adding an unguarded call, and one demoting the guard to
    truthiness (``if execution_result:``).

    One guard shape only is recognized -- an ``if execution_result is not
    None:`` block dominating the call. An early-return ``is None`` guard
    is equally safe at runtime and still turns this red, so the change
    that wires the resolver has to use the block shape.
    """

    source = inspect.getsource(finalize_managed_task_lease_result)
    tree = ast.parse(source)
    func_node = tree.body[0]
    assert isinstance(func_node, ast.FunctionDef)

    def _is_execution_result_not_none_guard(test: ast.expr) -> bool:
        return (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "execution_result"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.IsNot)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
        )

    def _publish_calls(node: ast.AST) -> list[ast.Call]:
        return [
            n
            for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and (
                (
                    isinstance(n.func, ast.Name)
                    and n.func.id == "resolve_publishable_clarification"
                )
                or (
                    isinstance(n.func, ast.Attribute)
                    and n.func.attr == "resolve_publishable_clarification"
                )
            )
        ]

    all_calls = {id(call) for call in _publish_calls(func_node)}
    guarded_calls: set[int] = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.If) and _is_execution_result_not_none_guard(node.test):
            for stmt in node.body:
                guarded_calls.update(id(call) for call in _publish_calls(stmt))

    assert all_calls - guarded_calls == set()


def test_finalize_without_execution_result_creates_no_interaction_row_in_native_mode(
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitting execution_result must not publish a clarification even when
    native-mode rollout would otherwise allow one.
    """

    monkeypatch.delenv(config.INTERACTION_PROTOCOL_MODE, raising=False)
    monkeypatch.delenv(config.INTERACTION_NATIVE_SOURCES, raising=False)
    monkeypatch.setattr(ir, "_policy", None)
    monkeypatch.setenv(config.INTERACTION_PROTOCOL_MODE, "native")
    monkeypatch.setenv(config.INTERACTION_NATIVE_SOURCES, "sdk")
    policy = ir.validate_interaction_rollout_at_startup()
    assert policy.mode == "native"

    task = _create_task(db_session)
    task.source = "sdk"
    db_session.commit()
    lease = acquire_task_lease(db_session, int(task.id), new_run=True)
    assert lease is not None

    assert (
        finalize_managed_task_lease_result(
            db_session,
            lease,
            status=TaskStatus.WAITING_FOR_USER,
            assistant_content="Please confirm",
        )
        is True
    )

    assert db_session.query(TaskInteractionRequest).count() == 0
