"""Tests for ``AgentServiceManager.execute_task`` lease delegation."""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from tests.web.pool_contention_shared import (
    GUARD_TIMEOUT,
    LOOP_LIVENESS_TICKS,
    gated_pool_checkout,
    wait_for_ticks,
)
from xagent.core.model.chat.token_context import TokenUsage
from xagent.web.api.chat import AgentServiceManager, _update_task_title_isolated
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce, WorkforceRun
from xagent.web.services import task_lease_service
from xagent.web.services.task_lease_service import (
    TaskLease,
    TaskLeaseHeartbeatOutcome,
    TaskLeaseLostError,
    current_task_lease,
)
from xagent.web.services.workforce_runtime import (
    sync_workforce_run_status,
    sync_workforce_run_status_for_task_id_isolated,
)
from xagent.web.tracking.task_tracker import _TaskTrackingSeed


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'execute_task_lease.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


class _FakeAgentService:
    async def execute_task(self, **_kwargs):
        return {"success": True}

    def set_interrupt_checker(self, _checker):
        # execute_task_background sets the mid-run quota checker after tracking
        # starts and clears it on completion; the double must accept both.
        pass


def _create_single_connection_runtime_db(tmp_path, filename: str):
    engine = create_engine(
        f"sqlite:///{tmp_path / filename}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.05,
        connect_args={"check_same_thread": False},
    )
    User.__table__.create(bind=engine)
    Task.__table__.create(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    with factory() as setup_db:
        user = User(username=f"{filename}-user", password_hash="hash", is_admin=False)
        setup_db.add(user)
        setup_db.flush()
        task = Task(
            user_id=user.id,
            title="pool test",
            description="test",
            status=TaskStatus.RUNNING,
            execution_mode="auto",
        )
        setup_db.add(task)
        setup_db.commit()
        task_id = int(task.id)
    return engine, factory, task_id


@pytest.mark.asyncio
async def test_execute_task_binds_outer_lease_only_during_agent_execution() -> None:
    manager = AgentServiceManager()
    lease = TaskLease(task_id=42, runner_id="runner-a", run_id="run-a")
    observed_leases: list[TaskLease | None] = []

    class LeaseObservingAgent(_FakeAgentService):
        async def execute_task(self, **_kwargs):
            observed_leases.append(current_task_lease())
            return {"success": True}

    with (
        patch.object(
            manager,
            "_acquire_sandbox_task",
            new=AsyncMock(return_value=None),
        ),
        patch.object(manager, "_release_sandbox_task", new=AsyncMock()),
        patch(
            "xagent.web.api.chat.stop_task_lease_heartbeat",
            new=AsyncMock(),
        ),
    ):
        result = await manager.execute_task(
            agent_service=LeaseObservingAgent(),
            task="hello",
            manage_task_lease=False,
            task_lease=lease,
        )

    assert result["success"] is True
    assert observed_leases == [lease]
    assert current_task_lease() is None


@pytest.mark.asyncio
async def test_execute_task_tracks_usage_for_outer_owned_lease() -> None:
    manager = AgentServiceManager()
    lease = TaskLease(task_id=42, runner_id="runner-a", run_id="run-a")
    tracker = MagicMock(
        start_tracking=AsyncMock(),
        complete_tracking=AsyncMock(),
        stop_periodic_updates=AsyncMock(),
        interrupt_reason_for_quota=AsyncMock(),
        quota_interrupt_reason=None,
    )

    with (
        patch(
            "xagent.web.api.chat._load_task_run_gate_user_id_isolated",
            return_value=7,
        ),
        patch(
            "xagent.web.api.chat._check_task_run_gate_on_event_loop",
            return_value=None,
        ),
        patch(
            "xagent.web.api.chat.sync_workforce_run_status_for_task_id_isolated"
        ) as sync_workforce,
        patch(
            "xagent.web.tracking.task_tracker.TaskTracker",
            return_value=tracker,
        ) as tracker_factory,
        patch.object(
            manager,
            "_acquire_sandbox_task",
            new=AsyncMock(return_value=None),
        ),
        patch.object(manager, "_release_sandbox_task", new=AsyncMock()),
    ):
        result = await manager.execute_task(
            agent_service=_FakeAgentService(),
            task="hello",
            tracking_task_id="42",
            manage_task_lease=False,
            task_lease=lease,
        )

    assert result["success"] is True
    sync_workforce.assert_not_called()
    tracker_factory.assert_called_once_with(
        task_id=42,
        expected_run_id="run-a",
        expected_runner_id="runner-a",
    )
    tracker.start_tracking.assert_awaited_once()
    tracker.complete_tracking.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_task_external_lease_loss_cancels_agent_execution() -> None:
    manager = AgentServiceManager()
    lease = TaskLease(task_id=42, runner_id="runner-a", run_id="run-a")
    execution_started = asyncio.Event()
    execution_cancelled = asyncio.Event()

    class BlockingAgent(_FakeAgentService):
        async def execute_task(self, **_kwargs):
            execution_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                execution_cancelled.set()

    async def lose_lease() -> TaskLeaseHeartbeatOutcome:
        await execution_started.wait()
        return TaskLeaseHeartbeatOutcome(lease_lost=True)

    heartbeat_task = asyncio.create_task(lose_lease())
    with (
        patch.object(
            manager,
            "_acquire_sandbox_task",
            new=AsyncMock(return_value=None),
        ),
        patch.object(manager, "_release_sandbox_task", new=AsyncMock()),
        patch(
            "xagent.web.api.chat.stop_task_lease_heartbeat",
            new=AsyncMock(),
        ),
    ):
        with pytest.raises(TaskLeaseLostError):
            await manager.execute_task(
                agent_service=BlockingAgent(),
                task="hello",
                manage_task_lease=False,
                task_lease=lease,
                task_lease_heartbeat_task=heartbeat_task,
            )

    assert execution_cancelled.is_set()


@pytest.mark.asyncio
async def test_execute_task_managed_lease_loss_skips_usage_and_release() -> None:
    manager = AgentServiceManager()
    lease = TaskLease(task_id=42, runner_id="runner-a", run_id="run-a")
    execution_started = asyncio.Event()
    execution_cancelled = asyncio.Event()
    tracker = MagicMock(
        start_tracking=AsyncMock(),
        complete_tracking=AsyncMock(),
        stop_periodic_updates=AsyncMock(),
        interrupt_reason_for_quota=AsyncMock(),
    )

    class BlockingAgent(_FakeAgentService):
        async def execute_task(self, **_kwargs):
            execution_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                execution_cancelled.set()

    async def lose_lease(
        _lease: TaskLease,
        _stop_event: asyncio.Event,
    ) -> TaskLeaseHeartbeatOutcome:
        await execution_started.wait()
        return TaskLeaseHeartbeatOutcome(lease_lost=True)

    release = MagicMock(return_value=True)
    with (
        patch(
            "xagent.web.api.chat._load_task_run_gate_user_id_isolated",
            return_value=None,
        ),
        patch(
            "xagent.web.api.chat.acquire_task_lease_isolated",
            return_value=lease,
        ),
        patch(
            "xagent.web.api.chat.run_task_lease_heartbeat",
            new=lose_lease,
        ),
        patch(
            "xagent.web.api.chat._release_managed_task_lease_isolated",
            release,
        ),
        patch(
            "xagent.web.api.chat.sync_workforce_run_status_for_task_id_isolated",
            return_value=False,
        ),
        patch(
            "xagent.web.tracking.task_tracker.TaskTracker",
            return_value=tracker,
        ),
        patch.object(
            manager,
            "_acquire_sandbox_task",
            new=AsyncMock(return_value=None),
        ),
        patch.object(manager, "_release_sandbox_task", new=AsyncMock()),
    ):
        with pytest.raises(TaskLeaseLostError):
            await manager.execute_task(
                agent_service=BlockingAgent(),
                task="hello",
                tracking_task_id="42",
                manage_task_lease=True,
            )

    assert execution_cancelled.is_set()
    tracker.complete_tracking.assert_not_awaited()
    tracker.stop_periodic_updates.assert_awaited_once()
    release.assert_not_called()


@pytest.mark.asyncio
async def test_execute_task_lease_loss_during_title_update_cancels_stale_write() -> (
    None
):
    manager = AgentServiceManager()
    lease = TaskLease(task_id=42, runner_id="runner-a", run_id="run-a")
    title_started = asyncio.Event()
    title_cancelled = asyncio.Event()

    async def update_title(*_args, **_kwargs) -> bool:
        title_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            title_cancelled.set()

    async def lose_lease() -> TaskLeaseHeartbeatOutcome:
        await title_started.wait()
        return TaskLeaseHeartbeatOutcome(lease_lost=True)

    heartbeat_task = asyncio.create_task(lose_lease())
    with (
        patch.object(
            manager,
            "_acquire_sandbox_task",
            new=AsyncMock(return_value=None),
        ),
        patch.object(manager, "_release_sandbox_task", new=AsyncMock()),
        patch(
            "xagent.web.api.chat.update_task_title_from_agent",
            new=update_title,
        ),
        patch(
            "xagent.web.api.chat.stop_task_lease_heartbeat",
            new=AsyncMock(),
        ),
    ):
        with pytest.raises(TaskLeaseLostError):
            await manager.execute_task(
                agent_service=_FakeAgentService(),
                task="hello",
                task_id="42",
                manage_task_lease=False,
                task_lease=lease,
                task_lease_heartbeat_task=heartbeat_task,
            )

    assert title_cancelled.is_set()


@pytest.mark.asyncio
async def test_execute_task_quota_pool_timeout_stops_pre_run_checkouts(
    tmp_path,
) -> None:
    """A run-gate context timeout stops later workforce/tracker checkouts."""
    engine, factory, task_id = _create_single_connection_runtime_db(
        tmp_path,
        "quota-pool-timeout.db",
    )
    held_connection = engine.connect()
    manager = AgentServiceManager()

    try:
        with (
            patch(
                "xagent.web.models.database.get_session_local",
                return_value=factory,
            ),
            patch(
                "xagent.web.api.chat.sync_workforce_run_status_for_task_id_isolated",
            ) as workforce_sync,
            patch(
                "xagent.web.tracking.task_tracker.TaskTracker",
            ) as tracker_factory,
        ):
            with pytest.raises(SQLAlchemyTimeoutError, match="QueuePool limit"):
                await manager.execute_task(
                    agent_service=_FakeAgentService(),
                    task="hello",
                    tracking_task_id=str(task_id),
                    manage_task_lease=False,
                )

        workforce_sync.assert_not_called()
        tracker_factory.assert_not_called()
    finally:
        held_connection.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_execute_task_workforce_pool_timeout_stops_tracker_checkout(
    tmp_path,
) -> None:
    """A workforce checkout timeout must not cascade into tracker checkout."""
    engine, factory, task_id = _create_single_connection_runtime_db(
        tmp_path,
        "workforce-pool-timeout.db",
    )
    held_connection = engine.connect()
    manager = AgentServiceManager()
    lease = TaskLease(
        task_id=task_id,
        runner_id="test-runner",
        run_id="test-run",
    )
    release_lease = MagicMock(return_value=True)
    stop_heartbeat = AsyncMock()

    try:
        with (
            patch(
                "xagent.web.models.database.get_session_local",
                return_value=factory,
            ),
            patch(
                "xagent.web.api.chat._load_task_run_gate_user_id_isolated",
                return_value=None,
            ),
            # The share-quota gate (#973) sits between the run gate and the
            # lease acquisition; stub its checkout like the run gate's so the
            # pool timeout under test still lands on the workforce stage.
            patch(
                "xagent.web.api.chat._load_task_public_run_quota_config_isolated",
                return_value=None,
            ),
            patch(
                "xagent.web.api.chat.acquire_task_lease_isolated",
                return_value=lease,
            ),
            patch(
                "xagent.web.api.chat.run_task_lease_heartbeat",
                new=AsyncMock(),
            ),
            patch(
                "xagent.web.api.chat.stop_task_lease_heartbeat",
                new=stop_heartbeat,
            ),
            patch(
                "xagent.web.api.chat._release_managed_task_lease_isolated",
                release_lease,
            ),
            patch(
                "xagent.web.tracking.task_tracker.TaskTracker",
            ) as tracker_factory,
            patch.object(
                manager,
                "_release_sandbox_task",
                new=AsyncMock(),
            ),
        ):
            with pytest.raises(SQLAlchemyTimeoutError, match="QueuePool limit"):
                await manager.execute_task(
                    agent_service=_FakeAgentService(),
                    task="hello",
                    tracking_task_id=str(task_id),
                    manage_task_lease=True,
                )

        tracker_factory.assert_not_called()
        stop_heartbeat.assert_awaited_once()
        release_lease.assert_not_called()
    finally:
        held_connection.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_execute_task_pre_run_timeout_waits_for_shared_heartbeat_batch() -> None:
    """Cleanup shares the heartbeat waiter and never opens a release checkout."""

    task_id = 109
    lease = TaskLease(
        task_id=task_id,
        runner_id="test-runner",
        run_id="test-run",
    )
    manager = AgentServiceManager()
    workforce_started = threading.Event()
    heartbeat_started = threading.Event()
    allow_workforce_timeout = threading.Event()
    allow_heartbeat_timeout = threading.Event()
    workforce_timeout = SQLAlchemyTimeoutError("workforce pool exhausted")
    heartbeat_timeout = SQLAlchemyTimeoutError("heartbeat pool exhausted")
    release_lease = MagicMock(return_value=True)

    def blocking_workforce_sync(*_args, **_kwargs) -> bool:
        workforce_started.set()
        assert allow_workforce_timeout.wait(timeout=2)
        raise workforce_timeout

    def blocking_heartbeat_batch(
        _leases: tuple[TaskLease, ...],
    ) -> dict[tuple[int, str, str | None], object]:
        heartbeat_started.set()
        assert allow_heartbeat_timeout.wait(timeout=2)
        raise heartbeat_timeout

    with (
        patch(
            "xagent.web.api.chat._load_task_run_gate_user_id_isolated",
            return_value=None,
        ),
        patch(
            "xagent.web.api.chat.acquire_task_lease_isolated",
            return_value=lease,
        ),
        patch(
            "xagent.web.api.chat.sync_workforce_run_status_for_task_id_isolated",
            side_effect=blocking_workforce_sync,
        ),
        patch(
            "xagent.web.api.chat._release_managed_task_lease_isolated",
            release_lease,
        ),
        patch(
            "xagent.web.tracking.task_tracker.TaskTracker",
        ) as tracker_factory,
        patch.object(
            task_lease_service,
            "get_task_lease_heartbeat_seconds",
            return_value=0.001,
        ),
        patch.object(
            task_lease_service,
            "refresh_task_leases_isolated",
            side_effect=blocking_heartbeat_batch,
        ),
        patch.object(manager, "_release_sandbox_task", new=AsyncMock()),
    ):
        execution = asyncio.create_task(
            manager.execute_task(
                agent_service=_FakeAgentService(),
                task="hello",
                tracking_task_id=str(task_id),
                manage_task_lease=True,
            )
        )
        try:
            assert await asyncio.to_thread(workforce_started.wait, 1)
            assert await asyncio.to_thread(heartbeat_started.wait, 1)

            allow_workforce_timeout.set()
            await asyncio.sleep(0.02)
            assert not execution.done()
            assert not allow_heartbeat_timeout.is_set()

            allow_heartbeat_timeout.set()
            with pytest.raises(SQLAlchemyTimeoutError) as exc_info:
                await asyncio.wait_for(execution, timeout=1)
            assert exc_info.value is workforce_timeout
        finally:
            allow_workforce_timeout.set()
            allow_heartbeat_timeout.set()
            if not execution.done():
                execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
            await asyncio.wait_for(
                task_lease_service.wait_for_heartbeat_manager_idle(),
                timeout=1,
            )

    tracker_factory.assert_not_called()
    release_lease.assert_not_called()


@pytest.mark.asyncio
async def test_execute_task_waits_for_shared_heartbeat_timeout_and_retains_lease() -> (
    None
):
    """A shared heartbeat timeout must not fan out into a release checkout."""

    task_id = 110
    lease = TaskLease(
        task_id=task_id,
        runner_id="test-runner",
        run_id="test-run",
    )
    manager = AgentServiceManager()
    usage_started = threading.Event()
    heartbeat_started = threading.Event()
    allow_usage = threading.Event()
    allow_heartbeat_timeout = threading.Event()
    heartbeat_timeout = SQLAlchemyTimeoutError("heartbeat pool exhausted")
    release_lease = MagicMock(return_value=True)

    def blocking_usage(*_args, **_kwargs) -> bool:
        usage_started.set()
        assert allow_usage.wait(timeout=2)
        return True

    def blocking_heartbeat_batch(
        _leases: tuple[TaskLease, ...],
    ) -> dict[tuple[int, str, str | None], object]:
        heartbeat_started.set()
        assert allow_heartbeat_timeout.wait(timeout=2)
        raise heartbeat_timeout

    with (
        patch(
            "xagent.web.api.chat._load_task_run_gate_user_id_isolated",
            return_value=None,
        ),
        patch(
            "xagent.web.api.chat.acquire_task_lease_isolated",
            return_value=lease,
        ),
        patch(
            "xagent.web.api.chat.sync_workforce_run_status_for_task_id_isolated",
            return_value=False,
        ),
        patch(
            "xagent.web.api.chat._release_managed_task_lease_isolated",
            release_lease,
        ),
        patch(
            "xagent.web.tracking.task_tracker._load_task_seed_sync",
            return_value=_TaskTrackingSeed(user_id=7, usage=TokenUsage()),
        ),
        patch(
            "xagent.web.tracking.task_tracker._complete_task_usage_sync",
            side_effect=blocking_usage,
        ),
        patch.object(
            task_lease_service,
            "get_task_lease_heartbeat_seconds",
            return_value=0.001,
        ),
        patch.object(
            task_lease_service,
            "refresh_task_leases_isolated",
            side_effect=blocking_heartbeat_batch,
        ),
        patch.object(
            manager,
            "_acquire_sandbox_task",
            new=AsyncMock(return_value=None),
        ),
        patch.object(manager, "_release_sandbox_task", new=AsyncMock()),
    ):
        execution = asyncio.create_task(
            manager.execute_task(
                agent_service=_FakeAgentService(),
                task="hello",
                tracking_task_id=str(task_id),
                manage_task_lease=True,
            )
        )
        try:
            assert await asyncio.to_thread(usage_started.wait, 1)
            assert await asyncio.to_thread(heartbeat_started.wait, 1)

            allow_usage.set()
            await asyncio.sleep(0.02)
            assert not execution.done()
            assert not allow_heartbeat_timeout.is_set()

            allow_heartbeat_timeout.set()
            with pytest.raises(SQLAlchemyTimeoutError) as exc_info:
                await asyncio.wait_for(execution, timeout=1)
            assert exc_info.value is heartbeat_timeout
        finally:
            allow_usage.set()
            allow_heartbeat_timeout.set()
            if not execution.done():
                execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
            await asyncio.wait_for(
                task_lease_service.wait_for_heartbeat_manager_idle(),
                timeout=1,
            )

    release_lease.assert_not_called()


@pytest.mark.asyncio
async def test_execute_task_tracker_pool_timeout_stops_execution_and_release() -> None:
    task_id = 108
    lease = TaskLease(
        task_id=task_id,
        runner_id="test-runner",
        run_id="test-run",
    )
    manager = AgentServiceManager()
    agent_service = _FakeAgentService()
    agent_service.execute_task = AsyncMock()  # type: ignore[method-assign]
    tracker = MagicMock()
    tracker.start_tracking = AsyncMock(
        side_effect=SQLAlchemyTimeoutError("tracker pool exhausted")
    )
    release_lease = MagicMock(return_value=True)

    with (
        patch(
            "xagent.web.api.chat._load_task_run_gate_user_id_isolated",
            return_value=None,
        ),
        patch(
            "xagent.web.api.chat.acquire_task_lease_isolated",
            return_value=lease,
        ),
        patch(
            "xagent.web.api.chat.sync_workforce_run_status_for_task_id_isolated",
            return_value=False,
        ),
        patch(
            "xagent.web.api.chat.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.chat.stop_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.chat._release_managed_task_lease_isolated",
            release_lease,
        ),
        patch(
            "xagent.web.tracking.task_tracker.TaskTracker",
            return_value=tracker,
        ),
        patch.object(manager, "_release_sandbox_task", new=AsyncMock()),
    ):
        with pytest.raises(SQLAlchemyTimeoutError, match="tracker pool exhausted"):
            await manager.execute_task(
                agent_service=agent_service,
                task="hello",
                tracking_task_id=str(task_id),
                manage_task_lease=True,
            )

    agent_service.execute_task.assert_not_awaited()
    release_lease.assert_not_called()


@pytest.mark.asyncio
async def test_execute_task_non_pool_quota_error_remains_fail_open() -> None:
    manager = AgentServiceManager()
    agent_service = _FakeAgentService()

    with (
        patch(
            "xagent.web.api.chat._load_task_run_gate_user_id_isolated",
            return_value=7,
        ),
        patch(
            "xagent.web.api.chat._check_task_run_gate_on_event_loop",
            side_effect=RuntimeError("quota service unavailable"),
        ),
        patch(
            "xagent.web.api.chat.sync_workforce_run_status_for_task_id_isolated",
            return_value=False,
        ),
        patch(
            "xagent.web.tracking.task_tracker.TaskTracker",
            side_effect=RuntimeError("tracking unavailable"),
        ),
        patch.object(
            manager,
            "_acquire_sandbox_task",
            new=AsyncMock(return_value=None),
        ),
        patch.object(manager, "_release_sandbox_task", new=AsyncMock()),
    ):
        result = await manager.execute_task(
            agent_service=agent_service,
            task="hello",
            tracking_task_id="101",
            manage_task_lease=False,
        )

    assert result["success"] is True


@pytest.mark.asyncio
async def test_execute_task_preflight_pool_wait_does_not_block_event_loop(
    tmp_path,
) -> None:
    """Run-gate context/workforce preflight waits off the event loop."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'execute-preflight-pool.db'}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.5,
        connect_args={"check_same_thread": False},
    )
    User.__table__.create(bind=engine)
    Task.__table__.create(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    with factory() as setup_db:
        user = User(username="pool-user", password_hash="hash", is_admin=False)
        setup_db.add(user)
        setup_db.flush()
        task = Task(
            user_id=user.id,
            title="pool test",
            description="test",
            status=TaskStatus.RUNNING,
            execution_mode="auto",
        )
        setup_db.add(task)
        setup_db.commit()
        task_id = int(task.id)

    held_connection = engine.connect()
    caller_db = factory()
    manager = AgentServiceManager()
    ticks = 0
    stop = asyncio.Event()

    async def ticker() -> None:
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    execute = None
    with gated_pool_checkout(engine) as gate:
        ticker_task = asyncio.create_task(ticker())
        try:
            with (
                patch(
                    "xagent.web.models.database.get_session_local",
                    return_value=factory,
                ),
                patch.object(
                    manager, "_acquire_sandbox_task", new=AsyncMock(return_value=None)
                ),
                patch.object(manager, "_release_sandbox_task", new=AsyncMock()),
                patch(
                    "xagent.web.tracking.task_tracker.TaskTracker",
                    side_effect=RuntimeError("skip tracking in pool-boundary test"),
                ),
            ):
                execute = asyncio.create_task(
                    manager.execute_task(
                        agent_service=_FakeAgentService(),
                        task="hello",
                        tracking_task_id=str(task_id),
                        db_session=caller_db,
                        manage_task_lease=False,
                    )
                )
                await gate.wait_until_contending()
                observed = await wait_for_ticks(lambda: ticks)
                assert observed >= LOOP_LIVENESS_TICKS
                assert not execute.done()

                held_connection.close()
                gate.let_through()
                result = await asyncio.wait_for(execute, timeout=GUARD_TIMEOUT)
                assert result["success"] is True
        finally:
            if not held_connection.closed:
                held_connection.close()
                gate.let_through()
            if execute is not None:
                await asyncio.wait_for(
                    asyncio.gather(execute, return_exceptions=True),
                    timeout=GUARD_TIMEOUT,
                )
            stop.set()
            await ticker_task
            caller_db.close()
            engine.dispose()


@pytest.mark.asyncio
async def test_execute_task_releases_read_only_caller_checkout_before_worker_io(
    tmp_path,
) -> None:
    """A legacy caller checkout must not force the worker to need slot two."""
    engine, factory, task_id = _create_single_connection_runtime_db(
        tmp_path,
        "execute-caller-checkout.db",
    )
    caller_db = factory()
    manager = AgentServiceManager()
    gate_checked_out: list[int] = []

    # A SELECT starts a transaction and pins the pool's only connection.
    caller_db.query(Task).filter(Task.id == task_id).one()
    assert engine.pool.checkedout() == 1

    def isolated_gate(worker_task_id: int) -> None:
        gate_checked_out.append(engine.pool.checkedout())
        with factory() as worker_db:
            worker_db.query(Task).filter(Task.id == worker_task_id).one()
        return None

    try:
        with (
            patch(
                "xagent.web.api.chat._load_task_run_gate_user_id_isolated",
                side_effect=isolated_gate,
            ),
            patch(
                "xagent.web.api.chat.sync_workforce_run_status_for_task_id_isolated",
                return_value=False,
            ),
            patch(
                "xagent.web.tracking.task_tracker.TaskTracker",
                side_effect=RuntimeError("skip tracking in pool-boundary test"),
            ),
            patch.object(
                manager,
                "_acquire_sandbox_task",
                new=AsyncMock(return_value=None),
            ),
            patch.object(manager, "_release_sandbox_task", new=AsyncMock()),
        ):
            result = await manager.execute_task(
                agent_service=_FakeAgentService(),
                task="hello",
                tracking_task_id=str(task_id),
                db_session=caller_db,
                manage_task_lease=False,
            )

        assert result["success"] is True
        assert gate_checked_out == [0]
        assert not caller_db.in_transaction()
    finally:
        caller_db.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_execute_task_rejects_dirty_caller_session_before_worker_io(
    tmp_path,
) -> None:
    """Pending caller writes must never be rolled back to free a pool slot."""
    engine, factory, task_id = _create_single_connection_runtime_db(
        tmp_path,
        "execute-dirty-caller.db",
    )
    caller_db = factory()
    manager = AgentServiceManager()
    task = caller_db.query(Task).filter(Task.id == task_id).one()
    task.title = "uncommitted caller title"

    try:
        with (
            patch(
                "xagent.web.api.chat._load_task_run_gate_user_id_isolated",
            ) as quota_gate,
            pytest.raises(
                RuntimeError,
                match="caller database session has pending writes",
            ),
        ):
            await manager.execute_task(
                agent_service=_FakeAgentService(),
                task="hello",
                tracking_task_id=str(task_id),
                db_session=caller_db,
                manage_task_lease=False,
            )

        quota_gate.assert_not_called()
        assert task in caller_db.dirty
        assert task.title == "uncommitted caller title"
    finally:
        caller_db.rollback()
        caller_db.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_execute_task_acquires_and_releases_lease_when_manage_true(
    db_session,
) -> None:
    user = User(username="lease-user", password_hash="hash", is_admin=False)
    db_session.add(user)
    db_session.commit()
    task = Task(
        user_id=user.id,
        title="lease test",
        description="test",
        status=TaskStatus.RUNNING,
        execution_mode="auto",
    )
    db_session.add(task)
    db_session.commit()

    fake_lease = TaskLease(
        task_id=int(task.id),
        runner_id="test-runner",
        run_id="test-run",
    )
    manager = AgentServiceManager()

    with (
        patch(
            "xagent.web.api.chat.acquire_task_lease_isolated",
            return_value=fake_lease,
        ) as mock_acquire,
        patch(
            "xagent.web.api.chat.release_task_lease_with_workforce_sync",
        ) as mock_release,
        patch(
            "xagent.web.api.chat.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.chat.stop_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch.object(
            manager, "_acquire_sandbox_task", new=AsyncMock(return_value=None)
        ),
        patch.object(manager, "_release_sandbox_task", new=AsyncMock()),
        patch(
            "xagent.web.api.chat.sync_workforce_run_status_for_task_id_isolated",
            return_value=False,
        ) as mock_sync,
    ):
        result = await manager.execute_task(
            agent_service=_FakeAgentService(),
            task="hello",
            tracking_task_id=str(task.id),
            db_session=db_session,
            manage_task_lease=True,
        )

    assert result["success"] is True
    mock_acquire.assert_called_once()
    mock_release.assert_called_once()
    mock_sync.assert_called_once_with(
        int(task.id),
        TaskStatus.RUNNING,
        task_lease=fake_lease,
    )


@pytest.mark.asyncio
async def test_execute_task_skips_lease_but_syncs_running_when_manage_false(
    db_session,
) -> None:
    user = User(username="lease-user2", password_hash="hash", is_admin=False)
    db_session.add(user)
    db_session.commit()
    task = Task(
        user_id=user.id,
        title="lease test",
        description="test",
        status=TaskStatus.RUNNING,
        execution_mode="auto",
    )
    db_session.add(task)
    db_session.commit()

    manager = AgentServiceManager()

    with (
        patch(
            "xagent.web.api.chat.acquire_task_lease_isolated",
        ) as mock_acquire,
        patch(
            "xagent.web.api.chat.release_task_lease_with_workforce_sync",
        ) as mock_release,
        patch(
            "xagent.web.api.chat.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.chat.stop_task_lease_heartbeat",
            new=AsyncMock(),
        ) as mock_stop_hb,
        patch.object(
            manager, "_acquire_sandbox_task", new=AsyncMock(return_value=None)
        ),
        patch.object(manager, "_release_sandbox_task", new=AsyncMock()),
        patch(
            "xagent.web.api.chat.sync_workforce_run_status_for_task_id_isolated",
            return_value=False,
        ) as mock_sync,
        patch(
            "xagent.web.tracking.task_tracker.TaskTracker",
            side_effect=RuntimeError("skip tracking in unit test"),
        ),
    ):
        result = await manager.execute_task(
            agent_service=_FakeAgentService(),
            task="hello",
            tracking_task_id=str(task.id),
            db_session=db_session,
            manage_task_lease=False,
        )

    assert result["success"] is True
    mock_acquire.assert_not_called()
    mock_release.assert_not_called()
    mock_sync.assert_called_once_with(
        int(task.id),
        TaskStatus.RUNNING,
        task_lease=None,
    )
    mock_stop_hb.assert_awaited_once_with(None, None)


@pytest.mark.asyncio
async def test_execute_task_surfaces_mid_run_quota_reason(db_session) -> None:
    """When the mid-run quota gate trips, the run result is reshaped to a
    terminal quota_exceeded carrying the reason as output (mirroring the start
    gate) instead of the pattern-interrupt path's silent flip to PAUSED."""
    user = User(username="quota-user", password_hash="hash", is_admin=False)
    db_session.add(user)
    db_session.commit()
    task = Task(
        user_id=user.id,
        title="quota test",
        description="test",
        status=TaskStatus.RUNNING,
        execution_mode="auto",
    )
    db_session.add(task)
    db_session.commit()

    manager = AgentServiceManager()
    reason = "Monthly ai_credits_per_month quota reached. Upgrade your plan."
    tracker = MagicMock()
    tracker.start_tracking = AsyncMock()
    tracker.complete_tracking = AsyncMock()
    tracker.quota_interrupt_reason = reason  # the mid-run gate tripped

    agent_service = _FakeAgentService()
    # The pattern-interrupt path returns a silent "interrupted" result.
    agent_service.execute_task = AsyncMock(  # type: ignore[method-assign]
        return_value={"success": False, "status": "interrupted", "error": "interrupted"}
    )

    with (
        patch("xagent.web.api.chat.run_task_lease_heartbeat", new=AsyncMock()),
        patch("xagent.web.api.chat.stop_task_lease_heartbeat", new=AsyncMock()),
        patch.object(
            manager, "_acquire_sandbox_task", new=AsyncMock(return_value=None)
        ),
        patch.object(manager, "_release_sandbox_task", new=AsyncMock()),
        patch(
            "xagent.web.api.chat.sync_workforce_run_status_for_task_id_isolated",
            return_value=False,
        ),
        patch(
            "xagent.web.tracking.task_tracker.TaskTracker",
            return_value=tracker,
        ),
    ):
        result = await manager.execute_task(
            agent_service=agent_service,
            task="hello",
            tracking_task_id=str(task.id),
            db_session=db_session,
            manage_task_lease=False,
        )

    assert result["status"] == "quota_exceeded"
    assert result["success"] is False
    assert result["output"] == reason
    assert result["error"] == reason
    # A mid-run interrupt is always the quota checker, so the result carries the
    # code (matching the start gate) to drive the app-layer dialog.
    assert result["error_code"] == "quota_exceeded"


@pytest.mark.asyncio
async def test_execute_task_start_gate_forwards_structured_reason(db_session) -> None:
    """When the start gate returns a structured reason (mapping), the run result
    carries its message plus error_code/error_details so the client can localise
    and branch, instead of only a plain string."""
    user = User(username="quota-start-user", password_hash="hash", is_admin=False)
    db_session.add(user)
    db_session.commit()
    task = Task(
        user_id=user.id,
        title="quota start test",
        description="test",
        status=TaskStatus.PENDING,
        execution_mode="auto",
    )
    db_session.add(task)
    db_session.commit()

    manager = AgentServiceManager()
    event_loop_thread = threading.get_ident()
    hook_threads: list[int] = []
    block = {
        "code": "quota_exceeded",
        "metric": "runs_per_month",
        "limit": 0,
        "plan": "basic",
        "message": "Team quota exhausted for this billing period.",
    }

    def run_gate(*_args):
        hook_threads.append(threading.get_ident())
        return block

    # The gate short-circuits before lease/tracker/execution, so a patched
    # check_run_gate returning the structured block is enough.
    with patch(
        "xagent.web.services.quota_hooks.check_run_gate",
        side_effect=run_gate,
    ):
        result = await manager.execute_task(
            agent_service=_FakeAgentService(),
            task="hello",
            tracking_task_id=str(task.id),
            db_session=db_session,
            manage_task_lease=False,
        )

    assert result["status"] == "quota_exceeded"
    assert result["success"] is False
    assert result["output"] == block["message"]
    assert result["error_code"] == "quota_exceeded"
    assert result["error_details"] == block
    assert hook_threads == [event_loop_thread]


@pytest.mark.asyncio
async def test_execute_task_cancellation_during_workforce_sync_releases_lease() -> None:
    """Cancellation after heartbeat start must drain worker I/O then clean up."""
    task_id = 106
    lease = TaskLease(
        task_id=task_id,
        runner_id="test-runner",
        run_id="test-run",
    )
    manager = AgentServiceManager()
    workforce_sync_started = threading.Event()
    allow_workforce_sync = threading.Event()
    release_lease = MagicMock(return_value=True)
    stop_heartbeat = AsyncMock()
    release_sandbox = AsyncMock()

    def blocking_workforce_sync(*_args, **_kwargs) -> bool:
        workforce_sync_started.set()
        if not allow_workforce_sync.wait(timeout=1):
            raise AssertionError("workforce sync test worker was not released")
        return False

    with (
        patch(
            "xagent.web.api.chat._load_task_run_gate_user_id_isolated",
            return_value=None,
        ),
        patch(
            "xagent.web.api.chat.acquire_task_lease_isolated",
            return_value=lease,
        ),
        patch(
            "xagent.web.api.chat.sync_workforce_run_status_for_task_id_isolated",
            side_effect=blocking_workforce_sync,
        ),
        patch(
            "xagent.web.api.chat._release_managed_task_lease_isolated",
            release_lease,
        ),
        patch(
            "xagent.web.api.chat.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.chat.stop_task_lease_heartbeat",
            new=stop_heartbeat,
        ),
        patch(
            "xagent.web.tracking.task_tracker.TaskTracker",
        ) as tracker_factory,
        patch.object(
            manager,
            "_acquire_sandbox_task",
            new=AsyncMock(return_value=None),
        ) as acquire_sandbox,
        patch.object(
            manager,
            "_release_sandbox_task",
            new=release_sandbox,
        ),
    ):
        execution = asyncio.create_task(
            manager.execute_task(
                agent_service=_FakeAgentService(),
                task="hello",
                tracking_task_id=str(task_id),
                manage_task_lease=True,
            )
        )
        started = await asyncio.wait_for(
            asyncio.to_thread(workforce_sync_started.wait, 1),
            timeout=1,
        )
        assert started

        execution.cancel()
        await asyncio.sleep(0)
        assert not execution.done()
        allow_workforce_sync.set()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(execution, timeout=1)

    tracker_factory.assert_not_called()
    acquire_sandbox.assert_not_awaited()
    stop_heartbeat.assert_awaited_once()
    release_lease.assert_called_once()
    assert release_lease.call_args.kwargs["status"] == TaskStatus.FAILED
    release_sandbox.assert_awaited_once_with(None)


@pytest.mark.asyncio
async def test_execute_task_cancellation_during_tracker_start_releases_lease() -> None:
    """Tracker startup is inside the lease owner's cleanup boundary."""
    task_id = 107
    lease = TaskLease(
        task_id=task_id,
        runner_id="test-runner",
        run_id="test-run",
    )
    manager = AgentServiceManager()
    tracker_start_entered = asyncio.Event()
    tracker = MagicMock()
    tracker.quota_interrupt_reason = None

    async def blocking_tracker_start() -> None:
        tracker_start_entered.set()
        await asyncio.Event().wait()

    tracker.start_tracking = AsyncMock(side_effect=blocking_tracker_start)
    tracker.complete_tracking = AsyncMock()
    release_lease = MagicMock(return_value=True)
    stop_heartbeat = AsyncMock()
    release_sandbox = AsyncMock()

    with (
        patch(
            "xagent.web.api.chat._load_task_run_gate_user_id_isolated",
            return_value=None,
        ),
        patch(
            "xagent.web.api.chat.acquire_task_lease_isolated",
            return_value=lease,
        ),
        patch(
            "xagent.web.api.chat.sync_workforce_run_status_for_task_id_isolated",
            return_value=False,
        ),
        patch(
            "xagent.web.api.chat._release_managed_task_lease_isolated",
            release_lease,
        ),
        patch(
            "xagent.web.api.chat.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.chat.stop_task_lease_heartbeat",
            new=stop_heartbeat,
        ),
        patch(
            "xagent.web.tracking.task_tracker.TaskTracker",
            return_value=tracker,
        ),
        patch.object(
            manager,
            "_acquire_sandbox_task",
            new=AsyncMock(return_value=None),
        ) as acquire_sandbox,
        patch.object(
            manager,
            "_release_sandbox_task",
            new=release_sandbox,
        ),
    ):
        execution = asyncio.create_task(
            manager.execute_task(
                agent_service=_FakeAgentService(),
                task="hello",
                tracking_task_id=str(task_id),
                manage_task_lease=True,
            )
        )
        await asyncio.wait_for(tracker_start_entered.wait(), timeout=1)
        execution.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(execution, timeout=1)

    acquire_sandbox.assert_not_awaited()
    tracker.complete_tracking.assert_awaited_once()
    stop_heartbeat.assert_awaited_once()
    release_lease.assert_called_once()
    assert release_lease.call_args.kwargs["status"] == TaskStatus.FAILED
    release_sandbox.assert_awaited_once_with(None)


@pytest.mark.asyncio
async def test_execute_task_cleans_up_when_sandbox_acquire_raises(
    db_session,
) -> None:
    """A reclaimed-sandbox raise from ``_acquire_sandbox_task`` must still
    run the finally cleanup: heartbeat stop, lease release, and tracker
    completion (whose only call site is that finally block)."""
    user = User(username="lease-user3", password_hash="hash", is_admin=False)
    db_session.add(user)
    db_session.commit()
    task = Task(
        user_id=user.id,
        title="lease test",
        description="test",
        status=TaskStatus.RUNNING,
        execution_mode="auto",
    )
    db_session.add(task)
    db_session.commit()

    fake_lease = TaskLease(
        task_id=int(task.id),
        runner_id="test-runner",
        run_id="test-run",
    )
    manager = AgentServiceManager()
    tracker = MagicMock()
    tracker.start_tracking = AsyncMock()
    tracker.complete_tracking = AsyncMock()
    agent_service = _FakeAgentService()
    agent_service.execute_task = AsyncMock()  # type: ignore[method-assign]
    agent_service.set_interrupt_checker = MagicMock()  # type: ignore[method-assign]

    with (
        patch(
            "xagent.web.api.chat.acquire_task_lease_isolated",
            return_value=fake_lease,
        ),
        patch(
            "xagent.web.api.chat.release_task_lease_with_workforce_sync",
        ) as mock_release,
        patch(
            "xagent.web.api.chat.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.chat.stop_task_lease_heartbeat",
            new=AsyncMock(),
        ) as mock_stop_hb,
        patch.object(
            manager,
            "_acquire_sandbox_task",
            new=AsyncMock(side_effect=RuntimeError("sandbox reclaimed")),
        ),
        patch.object(manager, "_release_sandbox_task", new=AsyncMock()) as mock_sbx,
        patch(
            "xagent.web.api.chat.sync_workforce_run_status_for_task_id_isolated",
            return_value=False,
        ),
        patch(
            "xagent.web.tracking.task_tracker.TaskTracker",
            return_value=tracker,
        ),
    ):
        with pytest.raises(RuntimeError, match="sandbox reclaimed"):
            await manager.execute_task(
                agent_service=agent_service,
                task="hello",
                tracking_task_id=str(task.id),
                db_session=db_session,
                manage_task_lease=True,
            )

    agent_service.execute_task.assert_not_awaited()
    mock_stop_hb.assert_awaited_once()
    mock_release.assert_called_once()
    assert mock_release.call_args.kwargs["status"] == TaskStatus.FAILED
    tracker.complete_tracking.assert_awaited_once()
    mock_sbx.assert_awaited_once_with(None)
    # The mid-run quota checker must be cleared in the finally so a reused
    # agent_service can't keep calling this finished run's tracker.
    agent_service.set_interrupt_checker.assert_any_call(None)


@pytest.mark.asyncio
async def test_execute_task_persists_final_usage_before_releasing_lease() -> None:
    """The final usage snapshot belongs to the current run, so it must land
    while that run still owns its lease."""
    task_id = 101
    lease = TaskLease(
        task_id=task_id,
        runner_id="test-runner",
        run_id="test-run",
    )
    manager = AgentServiceManager()
    events: list[str] = []

    tracker = MagicMock()
    tracker.start_tracking = AsyncMock()
    tracker.quota_interrupt_reason = None

    async def complete_tracking() -> None:
        events.append("usage")

    tracker.complete_tracking = AsyncMock(side_effect=complete_tracking)

    async def stop_heartbeat(*_args) -> None:
        events.append("heartbeat")

    def release_lease(*_args, **_kwargs) -> bool:
        events.append("lease")
        return True

    async def release_sandbox(*_args) -> None:
        events.append("sandbox")

    with (
        patch(
            "xagent.web.api.chat._load_task_run_gate_user_id_isolated",
            return_value=None,
        ),
        patch(
            "xagent.web.api.chat.acquire_task_lease_isolated",
            return_value=lease,
        ),
        patch(
            "xagent.web.api.chat.sync_workforce_run_status_for_task_id_isolated",
            return_value=False,
        ),
        patch(
            "xagent.web.api.chat._release_managed_task_lease_isolated",
            side_effect=release_lease,
        ),
        patch(
            "xagent.web.api.chat.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.chat.stop_task_lease_heartbeat",
            new=stop_heartbeat,
        ),
        patch(
            "xagent.web.tracking.task_tracker.TaskTracker",
            return_value=tracker,
        ) as tracker_factory,
        patch.object(
            manager,
            "_acquire_sandbox_task",
            new=AsyncMock(return_value="user:101"),
        ),
        patch.object(manager, "_release_sandbox_task", new=release_sandbox),
    ):
        result = await manager.execute_task(
            agent_service=_FakeAgentService(),
            task="hello",
            tracking_task_id=str(task_id),
            manage_task_lease=True,
        )

    assert result["success"] is True
    tracker_factory.assert_called_once_with(
        task_id=task_id,
        expected_run_id="test-run",
        expected_runner_id="test-runner",
    )
    assert events == ["usage", "heartbeat", "lease", "sandbox"]


@pytest.mark.asyncio
async def test_execute_task_releases_sandbox_when_lease_release_raises() -> None:
    """A lease settlement failure must not strand the independent sandbox."""
    task_id = 105
    lease = TaskLease(
        task_id=task_id,
        runner_id="test-runner",
        run_id="test-run",
    )
    manager = AgentServiceManager()
    tracker = MagicMock()
    tracker.start_tracking = AsyncMock()
    tracker.complete_tracking = AsyncMock()
    tracker.quota_interrupt_reason = None
    release_sandbox = AsyncMock()

    with (
        patch(
            "xagent.web.api.chat._load_task_run_gate_user_id_isolated",
            return_value=None,
        ),
        patch(
            "xagent.web.api.chat.acquire_task_lease_isolated",
            return_value=lease,
        ),
        patch(
            "xagent.web.api.chat.sync_workforce_run_status_for_task_id_isolated",
            return_value=False,
        ),
        patch(
            "xagent.web.api.chat._release_managed_task_lease_isolated",
            side_effect=RuntimeError("lease release failed"),
        ),
        patch(
            "xagent.web.api.chat.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.chat.stop_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.tracking.task_tracker.TaskTracker",
            return_value=tracker,
        ),
        patch.object(
            manager,
            "_acquire_sandbox_task",
            new=AsyncMock(return_value="user:105"),
        ),
        patch.object(manager, "_release_sandbox_task", new=release_sandbox),
    ):
        with pytest.raises(RuntimeError, match="lease release failed"):
            await manager.execute_task(
                agent_service=_FakeAgentService(),
                task="hello",
                tracking_task_id=str(task_id),
                manage_task_lease=True,
            )

    release_sandbox.assert_awaited_once_with("user:105")


@pytest.mark.asyncio
async def test_execute_task_final_usage_pool_timeout_retains_lease() -> None:
    """One exhausted final checkout must not trigger a second lease checkout."""
    task_id = 103
    lease = TaskLease(
        task_id=task_id,
        runner_id="test-runner",
        run_id="test-run",
    )
    manager = AgentServiceManager()
    tracker = MagicMock()
    tracker.start_tracking = AsyncMock()
    tracker.complete_tracking = AsyncMock(
        side_effect=SQLAlchemyTimeoutError("pool exhausted")
    )
    tracker.quota_interrupt_reason = None

    release_lease = MagicMock()
    stop_heartbeat = AsyncMock()
    release_sandbox = AsyncMock()

    with (
        patch(
            "xagent.web.api.chat._load_task_run_gate_user_id_isolated",
            return_value=None,
        ),
        patch(
            "xagent.web.api.chat.acquire_task_lease_isolated",
            return_value=lease,
        ),
        patch(
            "xagent.web.api.chat.sync_workforce_run_status_for_task_id_isolated",
            return_value=False,
        ),
        patch(
            "xagent.web.api.chat._release_managed_task_lease_isolated",
            release_lease,
        ),
        patch(
            "xagent.web.api.chat.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.chat.stop_task_lease_heartbeat",
            new=stop_heartbeat,
        ),
        patch(
            "xagent.web.tracking.task_tracker.TaskTracker",
            return_value=tracker,
        ),
        patch.object(
            manager,
            "_acquire_sandbox_task",
            new=AsyncMock(return_value="user:103"),
        ),
        patch.object(manager, "_release_sandbox_task", new=release_sandbox),
    ):
        with pytest.raises(SQLAlchemyTimeoutError, match="pool exhausted"):
            await manager.execute_task(
                agent_service=_FakeAgentService(),
                task="hello",
                tracking_task_id=str(task_id),
                manage_task_lease=True,
            )

    tracker.complete_tracking.assert_awaited_once()
    stop_heartbeat.assert_awaited_once()
    release_lease.assert_not_called()
    release_sandbox.assert_awaited_once_with("user:103")


@pytest.mark.asyncio
async def test_execute_task_heartbeat_pool_timeout_retains_lease() -> None:
    """Heartbeat pool exhaustion must not be followed by lease release I/O."""
    task_id = 104
    lease = TaskLease(
        task_id=task_id,
        runner_id="test-runner",
        run_id="test-run",
    )
    manager = AgentServiceManager()
    tracker = MagicMock()
    tracker.start_tracking = AsyncMock()
    tracker.complete_tracking = AsyncMock()
    tracker.quota_interrupt_reason = None
    heartbeat_timeout = SQLAlchemyTimeoutError("heartbeat pool exhausted")
    release_lease = MagicMock()
    release_sandbox = AsyncMock()

    with (
        patch(
            "xagent.web.api.chat._load_task_run_gate_user_id_isolated",
            return_value=None,
        ),
        patch(
            "xagent.web.api.chat.acquire_task_lease_isolated",
            return_value=lease,
        ),
        patch(
            "xagent.web.api.chat.sync_workforce_run_status_for_task_id_isolated",
            return_value=False,
        ),
        patch(
            "xagent.web.api.chat._release_managed_task_lease_isolated",
            release_lease,
        ),
        patch(
            "xagent.web.api.chat.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.chat.stop_task_lease_heartbeat",
            new=AsyncMock(
                return_value=TaskLeaseHeartbeatOutcome(pool_timeout=heartbeat_timeout)
            ),
        ),
        patch(
            "xagent.web.tracking.task_tracker.TaskTracker",
            return_value=tracker,
        ),
        patch.object(
            manager,
            "_acquire_sandbox_task",
            new=AsyncMock(return_value="user:104"),
        ),
        patch.object(manager, "_release_sandbox_task", new=release_sandbox),
    ):
        with pytest.raises(SQLAlchemyTimeoutError, match="heartbeat pool exhausted"):
            await manager.execute_task(
                agent_service=_FakeAgentService(),
                task="hello",
                tracking_task_id=str(task_id),
                manage_task_lease=True,
            )

    tracker.complete_tracking.assert_awaited_once()
    release_lease.assert_not_called()
    release_sandbox.assert_awaited_once_with("user:104")


@pytest.mark.asyncio
async def test_execute_task_heartbeat_loss_after_result_rejects_success() -> None:
    task_id = 106
    lease = TaskLease(
        task_id=task_id,
        runner_id="test-runner",
        run_id="test-run",
    )
    manager = AgentServiceManager()
    tracker = MagicMock()
    tracker.start_tracking = AsyncMock()
    tracker.complete_tracking = AsyncMock()
    tracker.stop_periodic_updates = AsyncMock()
    tracker.quota_interrupt_reason = None
    release_lease = MagicMock()

    with (
        patch(
            "xagent.web.api.chat._load_task_run_gate_user_id_isolated",
            return_value=None,
        ),
        patch(
            "xagent.web.api.chat.acquire_task_lease_isolated",
            return_value=lease,
        ),
        patch(
            "xagent.web.api.chat.sync_workforce_run_status_for_task_id_isolated",
            return_value=False,
        ),
        patch(
            "xagent.web.api.chat._release_managed_task_lease_isolated",
            release_lease,
        ),
        patch(
            "xagent.web.api.chat.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.chat.stop_task_lease_heartbeat",
            new=AsyncMock(return_value=TaskLeaseHeartbeatOutcome(lease_lost=True)),
        ),
        patch(
            "xagent.web.tracking.task_tracker.TaskTracker",
            return_value=tracker,
        ),
        patch.object(
            manager,
            "_acquire_sandbox_task",
            new=AsyncMock(return_value="user:106"),
        ),
        patch.object(manager, "_release_sandbox_task", new=AsyncMock()),
    ):
        with pytest.raises(TaskLeaseLostError):
            await manager.execute_task(
                agent_service=_FakeAgentService(),
                task="hello",
                tracking_task_id=str(task_id),
                manage_task_lease=True,
            )

    release_lease.assert_not_called()


def test_task_title_update_is_fenced_by_exact_lease(db_session) -> None:
    user = User(username="title-fence-user", password_hash="hash", is_admin=False)
    db_session.add(user)
    db_session.flush()
    task = Task(
        user_id=user.id,
        title="Original title",
        status=TaskStatus.RUNNING,
        runner_id="current-runner",
        run_id="current-run",
    )
    db_session.add(task)
    db_session.commit()

    stale = TaskLease(
        task_id=int(task.id),
        runner_id="stale-runner",
        run_id="stale-run",
    )
    current = TaskLease(
        task_id=int(task.id),
        runner_id="current-runner",
        run_id="current-run",
    )

    assert (
        _update_task_title_isolated(
            int(task.id),
            "Stale title",
            task_lease=stale,
        )
        is False
    )
    db_session.expire_all()
    assert db_session.get(Task, task.id).title == "Original title"

    assert (
        _update_task_title_isolated(
            int(task.id),
            "Current title",
            task_lease=current,
        )
        is True
    )
    db_session.expire_all()
    assert db_session.get(Task, task.id).title == "Current title"


@pytest.mark.asyncio
async def test_execute_task_cancellation_during_heartbeat_stop_drains_cleanup() -> None:
    """Caller cancellation during heartbeat shutdown must not strand the
    lease or skip tracker and sandbox cleanup."""
    task_id = 102
    lease = TaskLease(
        task_id=task_id,
        runner_id="test-runner",
        run_id="test-run",
    )
    manager = AgentServiceManager()
    events: list[str] = []
    heartbeat_stop_entered = asyncio.Event()
    allow_heartbeat_stop = asyncio.Event()

    tracker = MagicMock()
    tracker.start_tracking = AsyncMock()
    tracker.quota_interrupt_reason = None

    async def complete_tracking() -> None:
        events.append("usage")

    tracker.complete_tracking = AsyncMock(side_effect=complete_tracking)

    async def stop_heartbeat(*_args) -> None:
        events.append("heartbeat-enter")
        heartbeat_stop_entered.set()
        await allow_heartbeat_stop.wait()
        events.append("heartbeat-finish")

    def release_lease(*_args, **_kwargs) -> bool:
        events.append("lease")
        return True

    async def release_sandbox(*_args) -> None:
        events.append("sandbox")

    with (
        patch(
            "xagent.web.api.chat._load_task_run_gate_user_id_isolated",
            return_value=None,
        ),
        patch(
            "xagent.web.api.chat.acquire_task_lease_isolated",
            return_value=lease,
        ),
        patch(
            "xagent.web.api.chat.sync_workforce_run_status_for_task_id_isolated",
            return_value=False,
        ),
        patch(
            "xagent.web.api.chat._release_managed_task_lease_isolated",
            side_effect=release_lease,
        ),
        patch(
            "xagent.web.api.chat.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.chat.stop_task_lease_heartbeat",
            new=stop_heartbeat,
        ),
        patch(
            "xagent.web.tracking.task_tracker.TaskTracker",
            return_value=tracker,
        ),
        patch.object(
            manager,
            "_acquire_sandbox_task",
            new=AsyncMock(return_value="user:102"),
        ),
        patch.object(manager, "_release_sandbox_task", new=release_sandbox),
    ):
        execution = asyncio.create_task(
            manager.execute_task(
                agent_service=_FakeAgentService(),
                task="hello",
                tracking_task_id=str(task_id),
                manage_task_lease=True,
            )
        )
        await asyncio.wait_for(heartbeat_stop_entered.wait(), timeout=1)
        execution.cancel()
        await asyncio.sleep(0)
        allow_heartbeat_stop.set()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(execution, timeout=1)

    assert events == [
        "usage",
        "heartbeat-enter",
        "heartbeat-finish",
        "lease",
        "sandbox",
    ]


def test_sync_workforce_run_status_running_is_idempotent(db_session) -> None:
    """Repeat RUNNING sync is a no-op when WorkforceRun is already running."""
    user = User(username="sync-user", password_hash="hash", is_admin=False)
    db_session.add(user)
    db_session.flush()
    manager = Agent(
        user_id=user.id,
        name="Manager",
        description="desc",
        instructions="instr",
        execution_mode="balanced",
        models={"general": "test-model"},
        knowledge_bases=[],
        skills=[],
        tool_categories=[],
        suggested_prompts=[],
        status=AgentStatus.PUBLISHED,
    )
    db_session.add(manager)
    db_session.flush()
    workforce = Workforce(
        owner_user_id=user.id,
        scope_type="user",
        scope_id=str(user.id),
        name="Team",
        description="desc",
        manager_agent_id=manager.id,
        status="active",
    )
    db_session.add(workforce)
    db_session.flush()
    task = Task(
        user_id=user.id,
        title="sync test",
        description="test",
        status=TaskStatus.RUNNING,
        agent_id=manager.id,
        agent_config={},
        execution_mode="auto",
    )
    db_session.add(task)
    db_session.flush()
    run = WorkforceRun(
        workforce_id=workforce.id,
        task_id=task.id,
        user_id=user.id,
        status="running",
        snapshot={"version": 1},
    )
    db_session.add(run)
    db_session.flush()
    task.agent_config = {"workforce_run_id": run.id}
    db_session.commit()

    assert sync_workforce_run_status(db_session, task, TaskStatus.RUNNING) is False
    assert sync_workforce_run_status(db_session, task, TaskStatus.RUNNING) is False
    db_session.refresh(run)
    assert run.status == "running"
    assert run.completed_at is None


@pytest.mark.parametrize("use_stale_lease", [False, True])
def test_delayed_workforce_running_projection_cannot_resurrect_terminal_run(
    db_session,
    use_stale_lease: bool,
) -> None:
    user = User(username="projection-user", password_hash="hash", is_admin=False)
    db_session.add(user)
    db_session.flush()
    manager = Agent(
        user_id=user.id,
        name="Projection manager",
        description="desc",
        instructions="instr",
        execution_mode="balanced",
        models={"general": "test-model"},
        knowledge_bases=[],
        skills=[],
        tool_categories=[],
        suggested_prompts=[],
        status=AgentStatus.PUBLISHED,
    )
    db_session.add(manager)
    db_session.flush()
    workforce = Workforce(
        owner_user_id=user.id,
        scope_type="user",
        scope_id=str(user.id),
        name="Projection team",
        description="desc",
        manager_agent_id=manager.id,
        status="active",
    )
    db_session.add(workforce)
    db_session.flush()
    task = Task(
        user_id=user.id,
        title="terminal projection",
        description="test",
        status=TaskStatus.COMPLETED,
        runner_id="replacement-b",
        run_id="same-run",
        agent_id=manager.id,
        agent_config={},
        execution_mode="auto",
    )
    db_session.add(task)
    db_session.flush()
    run = WorkforceRun(
        workforce_id=workforce.id,
        task_id=task.id,
        user_id=user.id,
        status="completed",
        completed_at=datetime.now(timezone.utc),
        snapshot={"version": 1},
    )
    db_session.add(run)
    db_session.flush()
    task.agent_config = {"workforce_run_id": int(run.id)}
    db_session.commit()

    stale_lease = (
        TaskLease(
            task_id=int(task.id),
            runner_id="stale-a",
            run_id="same-run",
        )
        if use_stale_lease
        else None
    )
    assert (
        sync_workforce_run_status_for_task_id_isolated(
            int(task.id),
            TaskStatus.RUNNING,
            task_lease=stale_lease,
        )
        is False
    )

    db_session.expire_all()
    persisted_run = db_session.get(WorkforceRun, int(run.id))
    assert persisted_run.status == "completed"
    assert persisted_run.completed_at is not None
