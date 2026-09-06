"""Test cases for TaskTracker and TaskTrackerManager."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from tests.web.pool_contention_shared import (
    GUARD_TIMEOUT,
    LOOP_LIVENESS_TICKS,
    gated_pool_checkout,
    wait_for_ticks,
)
from xagent.core.model.chat.token_context import (
    TokenUsage,
    add_token_usage,
    get_token_usage,
)
from xagent.web.models.task import Task, TaskStatus
from xagent.web.services.db_runtime import drain_async_task_cancellation_safe
from xagent.web.tracking import TaskTracker, TaskTrackerManager


def _create_tracker_concurrency_db(tmp_path, filename):
    engine = create_engine(
        f"sqlite:///{tmp_path / filename}",
        connect_args={"check_same_thread": False},
        isolation_level="AUTOCOMMIT",
    )
    Task.__table__.create(engine)
    session_factory = sessionmaker(
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        bind=engine,
    )
    with session_factory() as db:
        db.add(
            Task(
                id=123,
                user_id=1,
                title="tracker ownership race",
                status=TaskStatus.RUNNING,
                run_id="run-a",
                runner_id="runner-a",
                input_tokens=4,
                output_tokens=2,
                total_tokens=6,
                llm_calls=1,
            )
        )
        db.commit()
    return engine, session_factory


def _run_tracker_ownership_race(
    monkeypatch,
    tmp_path,
    filename,
    operation,
    *,
    pause_after_ownership_read=False,
):
    from xagent.web.models import database
    from xagent.web.tracking import task_tracker as tracker_module

    engine, session_factory = _create_tracker_concurrency_db(tmp_path, filename)
    monkeypatch.setattr(database, "get_session_local", lambda: session_factory)
    fence_ready = threading.Event()
    takeover_done = threading.Event()
    pause_lock = threading.Lock()
    worker_thread_id: int | None = None
    paused = False
    result: dict[str, bool] = {}
    errors: list[BaseException] = []
    original_task_for_run = tracker_module._task_for_run

    def pause_for_takeover() -> None:
        nonlocal paused
        with pause_lock:
            if paused:
                return
            paused = True
        fence_ready.set()
        assert takeover_done.wait(timeout=5)

    def pause_before_usage_update(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ):
        if (
            threading.get_ident() == worker_thread_id
            and statement.lstrip().upper().startswith("UPDATE TASKS SET")
            and "input_tokens" in statement.lower()
        ):
            pause_for_takeover()

    def pause_after_legacy_ownership_read(*args, **kwargs):
        task = original_task_for_run(*args, **kwargs)
        if threading.get_ident() == worker_thread_id:
            pause_for_takeover()
        return task

    event.listen(engine, "before_cursor_execute", pause_before_usage_update)
    if pause_after_ownership_read:
        monkeypatch.setattr(
            tracker_module,
            "_task_for_run",
            pause_after_legacy_ownership_read,
        )

    def run_as_old_runner():
        nonlocal worker_thread_id
        worker_thread_id = threading.get_ident()
        try:
            result["owned"] = operation(tracker_module)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=run_as_old_runner)
    worker.start()
    try:
        assert fence_ready.wait(timeout=5)
        with session_factory() as takeover_db:
            replacement = takeover_db.get(Task, 123)
            assert replacement is not None
            replacement.runner_id = "runner-b"
            replacement.input_tokens = 100
            replacement.output_tokens = 50
            replacement.total_tokens = 150
            takeover_db.commit()
    finally:
        takeover_done.set()
        worker.join(timeout=5)
        event.remove(engine, "before_cursor_execute", pause_before_usage_update)

    assert not worker.is_alive()
    assert errors == []
    with session_factory() as verify_db:
        replacement = verify_db.get(Task, 123)
        assert replacement is not None
        persisted = (
            replacement.runner_id,
            replacement.input_tokens,
            replacement.output_tokens,
            replacement.total_tokens,
        )
    engine.dispose()
    return result["owned"], persisted


def _create_tracker_pool_db(tmp_path, filename):
    engine = create_engine(
        f"sqlite:///{tmp_path / filename}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=1,
    )
    Task.__table__.create(engine)
    session_factory = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )
    with session_factory() as db:
        db.add(
            Task(
                id=123,
                user_id=1,
                title="pool isolation",
                status=TaskStatus.RUNNING,
            )
        )
        db.commit()
    return engine, session_factory


async def _assert_tracker_operation_keeps_loop_responsive_under_pool_pressure(
    monkeypatch,
    tmp_path,
    filename,
    sync_helper_name,
    prepare,
    operation,
    verify,
    *,
    operation_scheduled=None,
):
    """Exercise one tracker DB operation while its QueuePool is exhausted."""
    from xagent.web.models import database
    from xagent.web.tracking import task_tracker as tracker_module

    engine, session_factory = _create_tracker_pool_db(tmp_path, filename)
    monkeypatch.setattr(database, "get_session_local", lambda: session_factory)
    tracker = TaskTracker(task_id=123, update_interval_seconds=60)
    operation_task = None
    ticker_task = None
    held_connection = None
    ticker_stop = threading.Event()
    ticker_ticks = 0
    event_loop_thread_id = threading.get_ident()
    worker_thread_id: int | None = None
    original_helper = getattr(tracker_module, sync_helper_name)

    def traced_helper(*args, **kwargs):
        nonlocal worker_thread_id
        worker_thread_id = threading.get_ident()
        return original_helper(*args, **kwargs)

    async def ticker():
        nonlocal ticker_ticks
        while not ticker_stop.is_set():
            await asyncio.sleep(0.01)
            ticker_ticks += 1

    try:
        await prepare(tracker)
        monkeypatch.setattr(tracker_module, sync_helper_name, traced_helper)
        held_connection = engine.connect()
        with gated_pool_checkout(engine) as gate:
            ticker_task = asyncio.create_task(ticker())
            operation_task = asyncio.create_task(operation(tracker))
            if operation_scheduled is not None:
                operation_scheduled.set()
            try:
                await gate.wait_until_contending()
                observed = await wait_for_ticks(lambda: ticker_ticks)
                assert observed >= LOOP_LIVENESS_TICKS
                assert not operation_task.done()
                assert worker_thread_id is not None
                assert worker_thread_id != event_loop_thread_id
            finally:
                held_connection.close()
                held_connection = None
                gate.let_through()
                # Drain even when cancelled before the worker reaches checkout.
                await drain_async_task_cancellation_safe(operation_task)
            result = operation_task.result()
            verify(tracker, session_factory, result)
    finally:
        ticker_stop.set()
        try:
            if held_connection is not None:
                held_connection.close()
            if ticker_task is not None:
                await drain_async_task_cancellation_safe(ticker_task)
        finally:
            try:
                if tracker.is_tracking:
                    await tracker.stop_periodic_updates()
            finally:
                engine.dispose()


class TestTaskTracker:
    """Test cases for TaskTracker."""

    @pytest.fixture
    def db_session(self, monkeypatch):
        """Fixture providing a mock database session."""
        from xagent.web.models import database

        session = MagicMock()

        # Mock Task object
        mock_task = MagicMock(spec=Task)
        mock_task.id = 123
        mock_task.status = TaskStatus.RUNNING
        mock_task.input_tokens = 0
        mock_task.output_tokens = 0
        mock_task.total_tokens = 0
        mock_task.llm_calls = 0
        mock_task.token_usage_details = None

        # Mock query to return the task
        session.query.return_value.filter.return_value.first.return_value = mock_task

        def apply_atomic_update(values, *, synchronize_session):
            assert synchronize_session is False
            for column, value in values.items():
                setattr(mock_task, column.key, value)
            return 1

        session.query.return_value.filter.return_value.update.side_effect = (
            apply_atomic_update
        )
        monkeypatch.setattr(database, "get_session_local", lambda: lambda: session)

        return session

    @pytest.fixture
    def task_tracker(self, db_session):
        """Fixture providing a TaskTracker instance."""
        return TaskTracker(task_id=123, db_session=db_session)

    @pytest.mark.asyncio
    async def test_init_task_tracker(self, db_session):
        """Test TaskTracker initialization."""
        tracker = TaskTracker(task_id=123, db_session=db_session)

        assert tracker.task_id == 123
        assert not hasattr(tracker, "db_session")
        assert not hasattr(tracker, "task")
        assert tracker.update_interval_seconds == 15  # default
        assert not tracker.is_tracking

    @pytest.mark.asyncio
    async def test_start_tracking_does_not_block_event_loop_when_pool_is_full(
        self, monkeypatch, tmp_path
    ):
        """A pool wait belongs in a worker thread, never on the event loop."""

        async def prepare(_tracker):
            return None

        def verify(tracker, _session_factory, _result):
            assert tracker.is_tracking

        await _assert_tracker_operation_keeps_loop_responsive_under_pool_pressure(
            monkeypatch,
            tmp_path,
            "tracker-start.db",
            "_load_task_seed_sync",
            prepare,
            lambda tracker: tracker.start_tracking(),
            verify,
        )

    @pytest.mark.asyncio
    async def test_periodic_update_does_not_block_event_loop_when_pool_is_full(
        self, monkeypatch, tmp_path
    ):
        """A periodic usage write must wait for the pool off the event loop."""

        async def prepare(tracker):
            await tracker.start_tracking()
            add_token_usage(input_tokens=10, output_tokens=5)

        def verify(tracker, session_factory, _result):
            assert tracker.is_tracking
            with session_factory() as db:
                task = db.get(Task, 123)
                assert task is not None
                assert (task.input_tokens, task.output_tokens, task.total_tokens) == (
                    10,
                    5,
                    15,
                )

        await _assert_tracker_operation_keeps_loop_responsive_under_pool_pressure(
            monkeypatch,
            tmp_path,
            "tracker-periodic.db",
            "_write_task_usage_sync",
            prepare,
            lambda tracker: tracker.periodic_update(),
            verify,
        )

    @pytest.mark.asyncio
    async def test_complete_tracking_does_not_block_event_loop_when_pool_is_full(
        self, monkeypatch, tmp_path
    ):
        """Final usage persistence must wait for the pool off the event loop."""

        async def prepare(tracker):
            await tracker.start_tracking()
            add_token_usage(input_tokens=20, output_tokens=10)

        def verify(tracker, session_factory, result):
            assert not tracker.is_tracking
            assert (result.input_tokens, result.output_tokens, result.total_tokens) == (
                20,
                10,
                30,
            )
            with session_factory() as db:
                task = db.get(Task, 123)
                assert task is not None
                assert (task.input_tokens, task.output_tokens, task.total_tokens) == (
                    20,
                    10,
                    30,
                )

        await _assert_tracker_operation_keeps_loop_responsive_under_pool_pressure(
            monkeypatch,
            tmp_path,
            "tracker-complete.db",
            "_complete_task_usage_sync",
            prepare,
            lambda tracker: tracker.complete_tracking(),
            verify,
        )

    @pytest.mark.parametrize(
        "inject_cleanup_error",
        [False, True],
        ids=["normal-cleanup", "cleanup-error"],
    )
    @pytest.mark.asyncio
    async def test_tracker_pool_helper_drains_delayed_worker(
        self, monkeypatch, tmp_path, inject_cleanup_error
    ):
        """Cancellation must drain a worker queued before it reaches checkout."""
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        original_executor = getattr(loop, "_default_executor")
        blocker_started = asyncio.Event()
        release_worker = threading.Event()
        operation_scheduled = asyncio.Event()
        worker_finished = threading.Event()
        from xagent.web.tracking import task_tracker as tracker_module

        original_seed = tracker_module._load_task_seed_sync

        def observed_seed(*args, **kwargs):
            try:
                return original_seed(*args, **kwargs)
            finally:
                worker_finished.set()

        monkeypatch.setattr(tracker_module, "_load_task_seed_sync", observed_seed)

        def occupy_default_executor():
            loop.call_soon_threadsafe(blocker_started.set)
            assert release_worker.wait(timeout=GUARD_TIMEOUT)

        async def prepare(_tracker):
            return None

        def verify(_tracker, _session_factory, _result):
            pytest.fail("cancelled helper completed normally")

        async def run_probe():
            blocker = loop.run_in_executor(None, occupy_default_executor)
            helper_task = None
            try:
                await asyncio.wait_for(blocker_started.wait(), timeout=GUARD_TIMEOUT)
                helper_task = asyncio.create_task(
                    _assert_tracker_operation_keeps_loop_responsive_under_pool_pressure(
                        monkeypatch,
                        tmp_path,
                        "tracker-delayed-worker.db",
                        "_load_task_seed_sync",
                        prepare,
                        lambda tracker: tracker.start_tracking(),
                        verify,
                        operation_scheduled=operation_scheduled,
                    )
                )
                await asyncio.wait_for(
                    operation_scheduled.wait(), timeout=GUARD_TIMEOUT
                )
                helper_task.cancel()
                release_worker.set()
                await blocker
                with pytest.raises(asyncio.CancelledError):
                    await helper_task

                assert helper_task.cancelled()
                assert worker_finished.is_set()
            finally:
                try:
                    release_worker.set()
                    if helper_task is not None and not helper_task.done():
                        helper_task.cancel()
                        await asyncio.gather(helper_task, return_exceptions=True)
                    if not blocker.done():
                        await blocker
                finally:
                    if inject_cleanup_error:
                        raise RuntimeError("injected cleanup failure")

        monkeypatch.setattr(loop, "_default_executor", executor)
        try:
            if inject_cleanup_error:
                with pytest.raises(RuntimeError, match="injected cleanup failure"):
                    await run_probe()
            else:
                await run_probe()
        finally:
            setattr(loop, "_default_executor", original_executor)
            executor.shutdown(wait=True)

        assert getattr(loop, "_default_executor") is original_executor
        with pytest.raises(RuntimeError):
            executor.submit(lambda: None)

    @pytest.mark.asyncio
    async def test_final_usage_does_not_overwrite_a_replacement_run(
        self, monkeypatch, tmp_path
    ):
        from xagent.web.models import database
        from xagent.web.services import quota_hooks

        engine = create_engine(
            f"sqlite:///{tmp_path / 'tracker-run-fence.db'}",
            connect_args={"check_same_thread": False},
        )
        Task.__table__.create(engine)
        session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine,
        )
        with session_factory() as db:
            db.add(
                Task(
                    id=123,
                    user_id=1,
                    title="run fence",
                    status=TaskStatus.RUNNING,
                    run_id="run-a",
                    input_tokens=4,
                    output_tokens=2,
                    total_tokens=6,
                    llm_calls=1,
                )
            )
            db.commit()

        monkeypatch.setattr(database, "get_session_local", lambda: session_factory)
        usage_hook_calls: list[int] = []
        quota_hooks.set_usage_record_hook(
            lambda _db, _user_id, _details, _actions: usage_hook_calls.append(1)
        )
        try:
            tracker = TaskTracker(task_id=123, expected_run_id="run-a")
            await tracker.start_tracking()
            add_token_usage(input_tokens=10, output_tokens=5)

            with session_factory() as replacement_db:
                replacement = replacement_db.get(Task, 123)
                assert replacement is not None
                replacement.run_id = "run-b"
                replacement.input_tokens = 100
                replacement.output_tokens = 50
                replacement.total_tokens = 150
                replacement_db.commit()

            await tracker.complete_tracking()
        finally:
            quota_hooks.set_usage_record_hook(None)

        with session_factory() as verify_db:
            replacement = verify_db.get(Task, 123)
            assert replacement is not None
            assert replacement.run_id == "run-b"
            assert replacement.input_tokens == 100
            assert replacement.output_tokens == 50
            assert replacement.total_tokens == 150
        assert usage_hook_calls == []
        engine.dispose()

    @pytest.mark.asyncio
    async def test_runner_fence_rejects_same_run_takeover_at_every_db_boundary(
        self, monkeypatch, tmp_path
    ):
        """A stale tracker must not seed or write after runner ownership changes."""
        from xagent.web.models import database
        from xagent.web.services import quota_hooks

        engine = create_engine(
            f"sqlite:///{tmp_path / 'tracker-runner-fence.db'}",
            connect_args={"check_same_thread": False},
        )
        Task.__table__.create(engine)
        session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine,
        )
        with session_factory() as db:
            for task_id in (123, 124, 125):
                db.add(
                    Task(
                        id=task_id,
                        user_id=1,
                        title=f"runner fence {task_id}",
                        status=TaskStatus.RUNNING,
                        run_id="run-a",
                        runner_id="runner-a",
                        input_tokens=4,
                        output_tokens=2,
                        total_tokens=6,
                        llm_calls=1,
                    )
                )
            db.commit()

        monkeypatch.setattr(database, "get_session_local", lambda: session_factory)
        usage_hook_calls: list[int] = []
        quota_hooks.set_usage_record_hook(
            lambda _db, _user_id, _details, _actions: usage_hook_calls.append(1)
        )
        periodic_tracker = None
        try:
            periodic_tracker = TaskTracker(
                task_id=123,
                update_interval_seconds=60,
                expected_run_id="run-a",
                expected_runner_id="runner-a",
            )
            final_tracker = TaskTracker(
                task_id=124,
                update_interval_seconds=60,
                expected_run_id="run-a",
                expected_runner_id="runner-a",
            )
            await periodic_tracker.start_tracking()
            await final_tracker.start_tracking()

            with session_factory() as takeover_db:
                for task_id in (123, 124, 125):
                    replacement = takeover_db.get(Task, task_id)
                    assert replacement is not None
                    replacement.runner_id = "runner-b"
                    replacement.input_tokens = 100
                    replacement.output_tokens = 50
                    replacement.total_tokens = 150
                takeover_db.commit()

            add_token_usage(input_tokens=10, output_tokens=5)
            await periodic_tracker.periodic_update()
            await final_tracker.complete_tracking()

            seed_tracker = TaskTracker(
                task_id=125,
                expected_run_id="run-a",
                expected_runner_id="runner-a",
            )
            with pytest.raises(ValueError, match="Task 125.*not found"):
                await seed_tracker.start_tracking()
        finally:
            quota_hooks.set_usage_record_hook(None)
            if periodic_tracker is not None and periodic_tracker.is_tracking:
                await periodic_tracker.stop_periodic_updates()

        with session_factory() as verify_db:
            for task_id in (123, 124, 125):
                replacement = verify_db.get(Task, task_id)
                assert replacement is not None
                assert replacement.run_id == "run-a"
                assert replacement.runner_id == "runner-b"
                assert replacement.input_tokens == 100
                assert replacement.output_tokens == 50
                assert replacement.total_tokens == 150
        assert not periodic_tracker.is_tracking
        assert usage_hook_calls == []
        engine.dispose()

    def test_periodic_write_cannot_overwrite_runner_takeover_after_ownership_read(
        self, monkeypatch, tmp_path
    ):
        """The ownership predicate and counter write must be one SQL statement."""
        owned, persisted = _run_tracker_ownership_race(
            monkeypatch,
            tmp_path,
            "tracker-periodic-cas.db",
            lambda tracker_module: tracker_module._write_task_usage_sync(
                123,
                TokenUsage(input_tokens=20, output_tokens=10, llm_calls=2),
                "run-a",
                "runner-a",
            ),
        )

        assert owned is False
        assert persisted == ("runner-b", 100, 50, 150)

    def test_final_usage_write_loses_takeover_ownership_race(
        self, monkeypatch, tmp_path
    ):
        """A stale runner must lose the durable fence before final persistence."""
        owned, persisted = _run_tracker_ownership_race(
            monkeypatch,
            tmp_path,
            "tracker-final-cas.db",
            lambda tracker_module: tracker_module._complete_task_usage_sync(
                123,
                TokenUsage(input_tokens=20, output_tokens=10, llm_calls=2),
                "run-a",
                "runner-a",
            ),
            pause_after_ownership_read=True,
        )

        assert owned is False
        assert persisted == ("runner-b", 100, 50, 150)

    @pytest.mark.asyncio
    async def test_usage_hook_pool_timeout_is_best_effort_after_counter_commit(
        self, monkeypatch, tmp_path, caplog
    ):
        """A metering checkout timeout must not reclassify a committed write."""
        import logging

        from xagent.web.models import database
        from xagent.web.services import quota_hooks

        engine = create_engine(
            f"sqlite:///{tmp_path / 'tracker-hook-pool-timeout.db'}",
            connect_args={"check_same_thread": False},
            poolclass=QueuePool,
            pool_size=1,
            max_overflow=0,
            pool_timeout=0.05,
        )
        Task.__table__.create(engine)
        session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine,
        )
        with session_factory() as db:
            db.add(
                Task(
                    id=123,
                    user_id=1,
                    title="best-effort usage hook",
                    status=TaskStatus.RUNNING,
                    run_id="run-a",
                    runner_id="runner-a",
                    input_tokens=4,
                    output_tokens=2,
                    total_tokens=6,
                    llm_calls=1,
                )
            )
            db.commit()

        monkeypatch.setattr(database, "get_session_local", lambda: session_factory)

        def pool_pressured_usage_hook(db, *_args):
            # The fenced counter commit has returned its connection. Occupy that
            # only slot before the hook's compatibility-session read, reproducing
            # a real QueuePool checkout timeout inside best-effort metering.
            held_connection = engine.connect()
            try:
                db.query(Task.id).first()
            finally:
                held_connection.close()

        quota_hooks.set_usage_record_hook(pool_pressured_usage_hook)
        try:
            tracker = TaskTracker(
                task_id=123,
                update_interval_seconds=60,
                expected_run_id="run-a",
                expected_runner_id="runner-a",
            )
            await tracker.start_tracking()
            add_token_usage(input_tokens=16, output_tokens=8)

            with caplog.at_level(
                logging.WARNING,
                logger="xagent.web.tracking.task_tracker",
            ):
                await tracker.complete_tracking()

            assert "Quota usage recording failed for task 123" in caplog.text
            assert "QueuePool limit of size 1 overflow 0 reached" in caplog.text

            with session_factory() as verify_db:
                task = verify_db.get(Task, 123)
                assert task is not None
                assert task.input_tokens == 20
                assert task.output_tokens == 10
                assert task.total_tokens == 30
                assert task.llm_calls == 2
        finally:
            quota_hooks.set_usage_record_hook(None)
            engine.dispose()

    @pytest.mark.asyncio
    async def test_db_writes_run_off_loop_without_moving_quota_hooks(self, monkeypatch):
        """Database writes use workers without changing quota hook affinity."""
        from xagent.web.models import database
        from xagent.web.services import quota_hooks

        event_loop_thread = threading.get_ident()
        session_threads = []
        sessions = []
        hook_threads = []
        hook_sessions = []

        def create_session():
            session_threads.append(threading.get_ident())
            session = MagicMock()
            task = MagicMock(spec=Task)
            task.id = 123
            task.user_id = 7
            task.input_tokens = 0
            task.output_tokens = 0
            task.total_tokens = 0
            task.llm_calls = 0
            task.token_usage_details = None
            session.query.return_value.filter.return_value.first.return_value = task
            sessions.append(session)
            return session

        monkeypatch.setattr(database, "get_session_local", lambda: create_session)

        def progress_hook(db, user_id, delta_details, delta_actions):
            hook_threads.append(threading.get_ident())
            hook_sessions.append(db)
            return None

        def usage_hook(db, user_id, delta_details, delta_actions):
            hook_threads.append(threading.get_ident())
            hook_sessions.append(db)

        quota_hooks.set_run_progress_gate_hook(progress_hook)
        quota_hooks.set_usage_record_hook(usage_hook)
        try:
            tracker = TaskTracker(task_id=123, update_interval_seconds=60)
            await tracker.start_tracking()
            await tracker.periodic_update()
            assert await tracker.interrupt_reason_for_quota() is None
            await tracker.complete_tracking()
        finally:
            quota_hooks.set_run_progress_gate_hook(None)
            quota_hooks.set_usage_record_hook(None)

        assert len(hook_threads) == 2
        assert hook_threads == [event_loop_thread, event_loop_thread]
        assert len(hook_sessions) == 2
        assert len(sessions) == 5
        assert session_threads.count(event_loop_thread) == 2
        hook_session_ids = {id(session) for session in hook_sessions}
        for hook_session in hook_sessions:
            session_index = sessions.index(hook_session)
            assert session_threads[session_index] == event_loop_thread
        for session_index, session in enumerate(sessions):
            if id(session) not in hook_session_ids:
                assert session_threads[session_index] != event_loop_thread
        for session in sessions:
            session.close.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_completion_waits_for_inflight_periodic_worker(
        self, db_session, monkeypatch
    ):
        """An uncancellable thread write must finish before the final write."""
        from xagent.web.tracking import task_tracker as tracker_module

        periodic_started = threading.Event()
        release_periodic = threading.Event()
        writes = []

        def blocking_periodic_write(*_args):
            periodic_started.set()
            assert release_periodic.wait(timeout=1)
            writes.append("periodic")
            return True

        def final_write(*_args):
            writes.append("complete")
            return True

        monkeypatch.setattr(
            tracker_module, "_write_task_usage_sync", blocking_periodic_write
        )
        monkeypatch.setattr(tracker_module, "_complete_task_usage_sync", final_write)

        tracker = TaskTracker(
            task_id=123,
            db_session=db_session,
            update_interval_seconds=0.01,
        )
        await tracker.start_tracking()
        assert await asyncio.to_thread(periodic_started.wait, 1)

        completion = asyncio.create_task(tracker.complete_tracking())
        try:
            await asyncio.sleep(0.02)
            assert not completion.done()
            release_periodic.set()
            await asyncio.wait_for(completion, timeout=1)
        finally:
            release_periodic.set()
            if not completion.done():
                await asyncio.wait_for(completion, timeout=1)

        assert writes == ["periodic", "complete"]

    @pytest.mark.asyncio
    async def test_periodic_pool_timeout_does_not_checkout_again_or_stop_tracking(
        self, db_session, monkeypatch
    ):
        """A transient pool timeout is one failed write, not a second probe."""
        from xagent.web.tracking import task_tracker as tracker_module

        attempts = 0

        def write_with_one_timeout(*_args):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise SQLAlchemyTimeoutError("pool exhausted")
            return True

        tracker = TaskTracker(
            task_id=123,
            db_session=db_session,
            update_interval_seconds=60,
        )
        await tracker.start_tracking()
        monkeypatch.setattr(
            tracker_module,
            "_write_task_usage_sync",
            write_with_one_timeout,
        )
        monkeypatch.setattr(
            tracker_module,
            "_new_short_session",
            lambda: pytest.fail("periodic failure performed a second checkout"),
        )

        try:
            await tracker.periodic_update()
            assert tracker.is_tracking

            await tracker.periodic_update()
            assert tracker.is_tracking
            assert attempts == 2
        finally:
            await tracker.stop_periodic_updates()

    @pytest.mark.asyncio
    async def test_final_pool_timeout_is_reported_to_the_lease_owner(
        self, db_session, monkeypatch
    ):
        """The lease owner must be able to avoid a second checkout after timeout."""
        from xagent.web.tracking import task_tracker as tracker_module

        attempts = 0

        def final_write(*_args):
            nonlocal attempts
            attempts += 1
            raise SQLAlchemyTimeoutError("pool exhausted")

        tracker = TaskTracker(
            task_id=123,
            db_session=db_session,
            update_interval_seconds=60,
        )
        await tracker.start_tracking()
        monkeypatch.setattr(
            tracker_module,
            "_complete_task_usage_sync",
            final_write,
        )

        with pytest.raises(SQLAlchemyTimeoutError, match="pool exhausted"):
            await tracker.complete_tracking()

        assert attempts == 1
        assert not tracker.is_tracking

    @pytest.mark.asyncio
    async def test_periodic_missing_row_stops_tracking_without_a_followup_checkout(
        self, db_session, monkeypatch
    ):
        """The write helper's False result is the only missing-row signal."""
        from xagent.web.tracking import task_tracker as tracker_module

        tracker = TaskTracker(
            task_id=123,
            db_session=db_session,
            update_interval_seconds=60,
        )
        await tracker.start_tracking()
        monkeypatch.setattr(
            tracker_module,
            "_write_task_usage_sync",
            lambda *_args: False,
        )
        monkeypatch.setattr(
            tracker_module,
            "_new_short_session",
            lambda: pytest.fail("missing-row result performed a second checkout"),
        )

        try:
            await tracker.periodic_update()
            assert not tracker.is_tracking
        finally:
            await tracker.stop_periodic_updates()

    @pytest.mark.asyncio
    async def test_cancelled_completion_drains_periodic_then_persists_final_usage(
        self, db_session, monkeypatch
    ):
        """Cancellation is propagated only after stale writes can no longer win."""
        from xagent.web.tracking import task_tracker as tracker_module

        periodic_started = threading.Event()
        release_periodic = threading.Event()
        writes = []

        def blocking_periodic_write(*_args):
            periodic_started.set()
            assert release_periodic.wait(timeout=1)
            writes.append("periodic")
            return True

        def final_write(*_args):
            writes.append("complete")
            return True

        monkeypatch.setattr(
            tracker_module,
            "_write_task_usage_sync",
            blocking_periodic_write,
        )
        monkeypatch.setattr(tracker_module, "_complete_task_usage_sync", final_write)

        tracker = TaskTracker(
            task_id=123,
            db_session=db_session,
            update_interval_seconds=0.01,
        )
        await tracker.start_tracking()
        assert await asyncio.to_thread(periodic_started.wait, 1)

        completion = asyncio.create_task(tracker.complete_tracking())
        try:
            await asyncio.sleep(0.02)
            completion.cancel()
            await asyncio.sleep(0.02)

            assert not completion.done()
            assert writes == []

            release_periodic.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(completion, timeout=1)
        finally:
            release_periodic.set()
            if not completion.done():
                try:
                    await asyncio.wait_for(asyncio.shield(completion), timeout=1)
                except BaseException:
                    pass

        assert completion.cancelled()
        assert writes == ["periodic", "complete"]

    @pytest.mark.asyncio
    async def test_complete_tracking_reports_only_current_turn_delta(self, db_session):
        """The usage-record hook must receive only this turn's delta, not the
        re-seeded prior-turn baseline (multi-turn tasks seed from the DB)."""
        from xagent.core.model.chat.token_context import add_tool_call_usage
        from xagent.web.services import quota_hooks

        task = db_session.query.return_value.filter.return_value.first.return_value
        task.user_id = 42
        # Prior-turn state seeded from the DB row.
        task.input_tokens = 100
        task.output_tokens = 50
        task.llm_calls = 1
        task.token_usage_details = [
            {"type": "input", "tokens": 100, "model": "m", "call_type": "chat"},
            {"type": "output", "tokens": 50, "model": "m", "call_type": "chat"},
        ]

        captured = {}

        def _hook(db, user_id, delta_details, delta_actions):
            captured.update(
                user_id=user_id, details=delta_details, actions=delta_actions
            )

        quota_hooks.set_usage_record_hook(_hook)
        try:
            tracker = TaskTracker(task_id=123, db_session=db_session)
            await tracker.start_tracking()
            # This turn's usage, appended on top of the seeded baseline.
            add_token_usage(
                input_tokens=10, output_tokens=5, model="m", call_type="chat"
            )
            add_tool_call_usage(3)
            await tracker.complete_tracking()
        finally:
            quota_hooks.set_usage_record_hook(None)

        assert captured["user_id"] == 42
        assert captured["actions"] == 3  # only this turn's tool calls (baseline was 0)
        # Only the two entries appended this turn, not the two seeded ones.
        assert len(captured["details"]) == 2
        assert sorted(d["tokens"] for d in captured["details"]) == [5, 10]

    @pytest.mark.asyncio
    async def test_interrupt_reason_for_quota_passes_turn_delta(self, db_session):
        """The per-step quota gate must see the same this-turn delta the metering
        path computes, and surface the gate's reason (or None when open)."""
        from xagent.core.model.chat.token_context import add_tool_call_usage
        from xagent.web.services import quota_hooks

        task = db_session.query.return_value.filter.return_value.first.return_value
        task.user_id = 42
        task.input_tokens = 100
        task.output_tokens = 50
        task.llm_calls = 1
        task.token_usage_details = [
            {"type": "input", "tokens": 100, "model": "m", "call_type": "chat"},
        ]

        captured = {}

        def _gate(db, user_id, delta_details, delta_actions):
            captured.update(
                user_id=user_id, details=delta_details, actions=delta_actions
            )
            return "over credits" if delta_actions >= 2 else None

        quota_hooks.set_run_progress_gate_hook(_gate)
        try:
            tracker = TaskTracker(task_id=123, db_session=db_session)
            # Before tracking starts, the checker is a no-op (fails open).
            assert await tracker.interrupt_reason_for_quota() is None
            await tracker.start_tracking()
            add_token_usage(
                input_tokens=10, output_tokens=5, model="m", call_type="chat"
            )
            add_tool_call_usage(2)
            reason = await tracker.interrupt_reason_for_quota()
        finally:
            quota_hooks.set_run_progress_gate_hook(None)

        assert reason == "over credits"
        # The reason is recorded so the run's caller can surface why it stopped.
        assert tracker.quota_interrupt_reason == "over credits"
        assert captured["user_id"] == 42
        assert captured["actions"] == 2  # only this turn's tool calls
        # Only this turn's input+output entries, not the seeded baseline one.
        assert len(captured["details"]) == 2
        assert sorted(d["tokens"] for d in captured["details"]) == [5, 10]

    @pytest.mark.asyncio
    async def test_quota_gate_caches_user_id_and_logs_once(self, db_session, caplog):
        """The tracker caches user_id and logs gate failures once per run."""
        import logging

        from xagent.web.services import quota_hooks

        task = db_session.query.return_value.filter.return_value.first.return_value
        task.user_id = 7

        def _boom(db, user_id, dd, da):
            raise RuntimeError("gate infra down")

        quota_hooks.set_run_progress_gate_hook(_boom)
        try:
            tracker = TaskTracker(task_id=123, db_session=db_session)
            assert tracker._user_id == 7
            await tracker.start_tracking()
            with caplog.at_level(logging.WARNING):
                assert await tracker.interrupt_reason_for_quota() is None
                assert await tracker.interrupt_reason_for_quota() is None
                assert await tracker.interrupt_reason_for_quota() is None
        finally:
            quota_hooks.set_run_progress_gate_hook(None)

        warnings = [r for r in caplog.records if "failed open" in r.getMessage()]
        assert len(warnings) == 1
        assert tracker.quota_interrupt_reason is None  # never tripped → no reason

    @pytest.mark.asyncio
    async def test_runtime_should_interrupt_drives_tracker_gate(self, db_session):
        """End-to-end seam: the runtime's should_interrupt — the exact call the
        pattern loop makes at each safe point — wired to the tracker's real gate
        method fires and records the reason when a registered hook trips."""
        from xagent.core.agent.runtime import PatternRuntime
        from xagent.web.services import quota_hooks

        task = db_session.query.return_value.filter.return_value.first.return_value
        task.user_id = 5

        quota_hooks.set_run_progress_gate_hook(lambda db, uid, dd, da: "Out of credits")
        try:
            tracker = TaskTracker(task_id=123, db_session=db_session)
            await tracker.start_tracking()
            runtime = PatternRuntime(
                interrupt_checker=tracker.interrupt_reason_for_quota
            )
            assert await runtime.should_interrupt() is True
            assert runtime.interrupt_reason == "Out of credits"
            assert tracker.quota_interrupt_reason == "Out of credits"
        finally:
            quota_hooks.set_run_progress_gate_hook(None)

    @pytest.mark.asyncio
    async def test_init_task_tracker_with_custom_interval(self, db_session):
        """Test TaskTracker with custom update interval."""
        tracker = TaskTracker(
            task_id=123, db_session=db_session, update_interval_seconds=60
        )

        assert tracker.update_interval_seconds == 60

    @pytest.mark.asyncio
    async def test_init_task_tracker_task_not_found(self, db_session):
        """Test TaskTracker with non-existent task."""
        # Mock query to return None (task not found)
        db_session.query.return_value.filter.return_value.first.return_value = None

        with pytest.raises(ValueError, match="Task 123 not found"):
            TaskTracker(task_id=123, db_session=db_session)

    @pytest.mark.asyncio
    async def test_start_tracking(self, task_tracker):
        """Test starting token tracking."""
        # Add some tokens before starting
        add_token_usage(input_tokens=10, output_tokens=5)

        await task_tracker.start_tracking()

        # Should reset token usage
        usage = get_token_usage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert task_tracker.is_tracking

    @pytest.mark.asyncio
    async def test_start_tracking_uses_existing_task_totals(self, db_session):
        mock_task = db_session.query.return_value.filter.return_value.first.return_value
        mock_task.input_tokens = 120
        mock_task.output_tokens = 80
        mock_task.llm_calls = 4
        mock_task.token_usage_details = [{"type": "input", "tokens": 120}]

        tracker = TaskTracker(task_id=123, db_session=db_session)
        await tracker.start_tracking()

        usage = get_token_usage()
        assert usage.input_tokens == 120
        assert usage.output_tokens == 80
        assert usage.llm_calls == 4
        assert usage.details == [{"type": "input", "tokens": 120}]

    @pytest.mark.asyncio
    async def test_start_tracking_already_tracking(self, task_tracker, caplog):
        """Test starting tracking when already tracking."""
        with patch("xagent.web.tracking.task_tracker.logger.warning") as mock_warning:
            await task_tracker.start_tracking()

            # Try to start again
            await task_tracker.start_tracking()

        # Should log warning
        assert mock_warning.called
        assert "already being tracked" in mock_warning.call_args.args[0]

    @pytest.mark.asyncio
    async def test_periodic_update(self, task_tracker, db_session):
        """Test periodic database update."""
        await task_tracker.start_tracking()

        # Add some tokens
        add_token_usage(input_tokens=100, output_tokens=50)

        # Perform periodic update
        await task_tracker.periodic_update()

        # Verify database was updated
        task = db_session.query.return_value.filter.return_value.first.return_value
        assert task.input_tokens == 100
        assert task.output_tokens == 50
        assert task.total_tokens == 150
        assert task.llm_calls == 1
        db_session.commit.assert_called_once()
        db_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_periodic_update_not_tracking(self, task_tracker, caplog):
        """Test periodic update when not tracking."""
        # Don't start tracking

        with patch("xagent.web.tracking.task_tracker.logger.warning") as mock_warning:
            await task_tracker.periodic_update()

        # Should log warning
        assert mock_warning.called
        assert "not being tracked" in mock_warning.call_args.args[0]

    @pytest.mark.asyncio
    async def test_complete_tracking(self, task_tracker, db_session):
        """Test completing tracking."""
        await task_tracker.start_tracking()

        # Add some tokens
        add_token_usage(input_tokens=200, output_tokens=100)
        add_token_usage(input_tokens=50, output_tokens=25)

        # Complete tracking
        usage = await task_tracker.complete_tracking()

        # Verify usage was returned
        assert usage.input_tokens == 250
        assert usage.output_tokens == 125
        assert usage.total_tokens == 375
        assert usage.llm_calls == 2

        # Verify database was updated
        task = db_session.query.return_value.filter.return_value.first.return_value
        assert task.input_tokens == 250
        assert task.output_tokens == 125
        assert task.total_tokens == 375
        assert task.llm_calls == 2
        db_session.commit.assert_called_once()
        # The fenced task write and compatibility quota callback each own one
        # short-lived Session.
        assert db_session.close.call_count == 2

        assert not task_tracker.is_tracking

    @pytest.mark.asyncio
    async def test_complete_tracking_not_started(self, task_tracker):
        """Test completing tracking without starting."""
        # Don't start tracking

        with pytest.raises(RuntimeError, match="not being tracked"):
            await task_tracker.complete_tracking()

    @pytest.mark.asyncio
    async def test_get_current_usage(self, task_tracker):
        """Test getting current usage without stopping."""
        await task_tracker.start_tracking()

        add_token_usage(input_tokens=30, output_tokens=15)

        usage = task_tracker.get_current_usage()

        assert usage.input_tokens == 30
        assert usage.output_tokens == 15
        assert task_tracker.is_tracking  # Should still be tracking

    @pytest.mark.asyncio
    async def test_start_stop_periodic_updates(self, task_tracker):
        """Test starting and stopping periodic background updates."""
        await task_tracker.start_tracking()

        # Start periodic updates
        await task_tracker.start_periodic_updates()

        assert task_tracker.is_tracking

        # Stop periodic updates
        await task_tracker.stop_periodic_updates()

        assert not task_tracker.is_tracking

    @pytest.mark.asyncio
    async def test_start_periodic_updates_already_active(self, task_tracker, caplog):
        """Test starting periodic updates when already active."""
        with patch("xagent.web.tracking.task_tracker.logger.warning") as mock_warning:
            await task_tracker.start_tracking()
            await task_tracker.start_periodic_updates()

            # Try to start again
            await task_tracker.start_periodic_updates()

        # Should log warning
        assert mock_warning.called
        assert "already active" in mock_warning.call_args.args[0]


class TestTaskTrackerManager:
    """Test cases for TaskTrackerManager."""

    @pytest.fixture
    def manager(self):
        """Fixture providing a TaskTrackerManager."""
        return TaskTrackerManager()

    @pytest.fixture
    def mock_session(self, monkeypatch):
        """Fixture providing a mock database session."""
        from xagent.web.models import database

        session = MagicMock()

        # Mock Task object
        mock_task = MagicMock(spec=Task)
        mock_task.id = 1
        mock_task.status = TaskStatus.RUNNING

        session.query.return_value.filter.return_value.first.return_value = mock_task
        monkeypatch.setattr(database, "get_session_local", lambda: lambda: session)

        return session

    @pytest.mark.asyncio
    async def test_get_or_create_tracker_new(self, manager, mock_session):
        """Test creating a new tracker."""
        tracker = manager.get_or_create_tracker(task_id=1, db_session=mock_session)

        assert tracker is not None
        assert tracker.task_id == 1
        assert 1 in manager._trackers

    @pytest.mark.asyncio
    async def test_get_or_create_tracker_existing(self, manager, mock_session):
        """Test getting existing tracker."""
        # Create tracker first
        tracker1 = manager.get_or_create_tracker(task_id=1, db_session=mock_session)

        # Get same tracker again
        tracker2 = manager.get_or_create_tracker(task_id=1, db_session=mock_session)

        # Should return the same instance
        assert tracker1 is tracker2

    @pytest.mark.asyncio
    async def test_get_tracker(self, manager, mock_session):
        """Test getting tracker without creating."""
        # Non-existent tracker
        tracker = manager.get_tracker(task_id=1)
        assert tracker is None

        # Create tracker
        manager.get_or_create_tracker(task_id=1, db_session=mock_session)

        # Now it exists
        tracker = manager.get_tracker(task_id=1)
        assert tracker is not None

    @pytest.mark.asyncio
    async def test_complete_tracker(self, manager, mock_session):
        """Test completing a specific tracker."""
        # Create and start tracker
        tracker = manager.get_or_create_tracker(task_id=1, db_session=mock_session)
        await tracker.start_tracking()

        # Add tokens
        add_token_usage(input_tokens=10, output_tokens=5)

        # Complete the tracker
        usage = await manager.complete_tracker(task_id=1)

        # Verify usage
        assert usage.input_tokens == 10
        assert usage.output_tokens == 5

        # Tracker should be removed
        assert 1 not in manager._trackers

    @pytest.mark.asyncio
    async def test_complete_tracker_nonexistent(self, manager):
        """Test completing non-existent tracker."""
        usage = await manager.complete_tracker(task_id=999)

        # Should return None
        assert usage is None

    @pytest.mark.asyncio
    async def test_complete_all(self, manager, mock_session):
        """Test completing all trackers."""
        # Create multiple trackers with independent token tracking
        # Note: In real usage, each task would have its own execution context
        # Here we simulate by manually tracking tokens per task

        # Trackers are created but tokens accumulate in shared context
        for i in range(1, 4):
            tracker = manager.get_or_create_tracker(task_id=i, db_session=MagicMock())
            await tracker.start_tracking()

        # Complete all (will have 0 tokens since we didn't add any after last reset)
        results = await manager.complete_all()

        # Verify all trackers completed
        assert len(results) == 3
        assert 1 in results
        assert 2 in results
        assert 3 in results

        # All trackers should be removed
        assert len(manager._trackers) == 0

    @pytest.mark.asyncio
    async def test_multiple_tasks_independent(self, manager):
        """Test that multiple task trackers can be created and managed independently."""
        # Mock different sessions
        session1 = MagicMock()
        session2 = MagicMock()

        for i, session in enumerate([session1, session2], 1):
            mock_task = MagicMock(spec=Task)
            mock_task.id = i
            session.query.return_value.filter.return_value.first.return_value = (
                mock_task
            )

        # Create two trackers
        tracker1 = manager.get_or_create_tracker(task_id=1, db_session=session1)
        tracker2 = manager.get_or_create_tracker(task_id=2, db_session=session2)

        # Verify both are tracked independently
        assert tracker1.task_id == 1
        assert tracker2.task_id == 2
        assert len(manager._trackers) == 2


class TestTaskTrackerIntegration:
    """Integration tests for TaskTracker with real token tracking."""

    @pytest.fixture
    def db_session(self, monkeypatch):
        """Fixture providing a mock database session."""
        from xagent.web.models import database

        session = MagicMock()

        mock_task = MagicMock(spec=Task)
        mock_task.id = 123
        mock_task.status = TaskStatus.RUNNING
        mock_task.input_tokens = 0
        mock_task.output_tokens = 0
        mock_task.total_tokens = 0
        mock_task.llm_calls = 0
        mock_task.token_usage_details = None

        session.query.return_value.filter.return_value.first.return_value = mock_task
        monkeypatch.setattr(database, "get_session_local", lambda: lambda: session)

        return session

    @pytest.mark.asyncio
    async def test_full_tracking_workflow(self, db_session):
        """Test complete tracking workflow."""
        tracker = TaskTracker(task_id=123, db_session=db_session)

        # Start tracking
        await tracker.start_tracking()

        # Simulate LLM calls
        add_token_usage(
            input_tokens=100, output_tokens=50, model="gpt-4", call_type="chat"
        )
        add_token_usage(
            input_tokens=200, output_tokens=100, model="gpt-4", call_type="chat"
        )

        # Check current usage
        usage = tracker.get_current_usage()
        assert usage.input_tokens == 300
        assert usage.output_tokens == 150

        # Complete tracking
        final_usage = await tracker.complete_tracking()

        # Verify final stats
        assert final_usage.input_tokens == 300
        assert final_usage.output_tokens == 150
        assert final_usage.llm_calls == 2  # Two add_token_usage calls
        assert len(final_usage.details) == 4  # 2 input + 2 output entries

        # Verify database was updated
        db_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_tracking_with_details(self, db_session):
        """Test tracking with detailed token information."""
        tracker = TaskTracker(task_id=123, db_session=db_session)

        await tracker.start_tracking()

        # Add tokens with details
        add_token_usage(
            input_tokens=100,
            output_tokens=50,
            model="gpt-4",
            call_type="chat",
        )
        add_token_usage(
            input_tokens=50,
            output_tokens=25,
            model="gpt-3.5-turbo",
            call_type="stream_chat",
        )

        final_usage = await tracker.complete_tracking()

        # Verify details are tracked (4 entries: 2 input + 2 output)
        assert len(final_usage.details) == 4

        # Details are accumulated, check both models are present
        models = [d.get("model") for d in final_usage.details]
        assert "gpt-4" in models
        assert "gpt-3.5-turbo" in models

        # Check both input and output types are present
        types = [d.get("type") for d in final_usage.details]
        assert "input" in types
        assert "output" in types


def test_check_run_progress_gate_guards():
    """The quota-gate seam is a no-op when no hook is registered or user_id is
    None, and otherwise forwards to the hook and returns its reason."""
    from xagent.web.services import quota_hooks

    # No hook registered → None regardless of args.
    quota_hooks.set_run_progress_gate_hook(None)
    assert quota_hooks.check_run_progress_gate("db", 1, [], 0) is None

    seen = []
    quota_hooks.set_run_progress_gate_hook(
        lambda db, uid, dd, da: (seen.append(uid), "OVER")[1]
    )
    try:
        # user_id None short-circuits before the hook is called.
        assert quota_hooks.check_run_progress_gate("db", None, [], 0) is None
        assert seen == []
        # Otherwise the hook runs and its reason is returned verbatim.
        assert quota_hooks.check_run_progress_gate("db", 7, [], 3) == "OVER"
        assert seen == [7]
    finally:
        quota_hooks.set_run_progress_gate_hook(None)
