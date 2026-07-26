"""Test that ``execute_task_background`` consumes an off-loop snapshot.

Background:
    Profiling measured the inline ``db.query(Task)`` at the top of
    ``execute_task_background`` at ~3.3s of synchronous DB read under
    contention (the same row had just been queried by
    ``_schedule_bg._runner``). The off-loop snapshot path plumbs a
    ``task_setup_snapshot`` parameter through ``_runner`` →
    ``execute_task_background`` → ``get_agent_for_task`` so the Task
    SELECT happens once, off-loop, in
    ``load_task_setup_snapshot_sync``.

What this test pins:

    * Snapshot path performs no event-loop Task/User query.
    * A caller that omits the snapshot loads the same primitive snapshot in a
      worker thread instead of reintroducing an event-loop Session.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import Counter
from time import monotonic
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from xagent.web.api.websocket import execute_task_background
from xagent.web.models.agent import AgentStatus
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.llm_utils import AgentRuntimeFields
from xagent.web.services.task_lease_service import TaskLease
from xagent.web.services.task_setup_snapshot import (
    RuntimeUserFields,
    TaskSetupSnapshot,
    _TaskFields,
)


def _make_task_orm() -> Task:
    """Fake ORM Task row used only in the legacy / WS fallback path."""
    t = Task(
        id=42,
        user_id=1,
        title="exec-bg test",
        description="x",
        status=TaskStatus.RUNNING,
        agent_id=7,
        agent_type="standard",
    )
    return t


def _make_user_orm() -> User:
    return User(id=1, username="exec-bg-user", password_hash="hash", is_admin=False)


def _make_snapshot(*, source: str | None = "trigger") -> TaskSetupSnapshot:
    return TaskSetupSnapshot(
        task=_TaskFields(
            id=42,
            user_id=1,
            status=TaskStatus.RUNNING,
            source=source,
            agent_id=7,
            agent_config=None,
            model_name=None,
            compact_model_name=None,
            execution_mode="flash",
            agent_type="standard",
        ),
        runtime_user=RuntimeUserFields(id=1, is_admin=False),
        has_reconstructable_history=False,
        task_pattern="single_call",
        task_llm=None,
        task_fast_llm=None,
        task_vision_llm=None,
        task_compact_llm=None,
        agent=AgentRuntimeFields(
            id=7,
            name="snap-agent",
            status=AgentStatus.PUBLISHED,
            instructions="be terse",
        ),
        agent_config={
            "llms": (None, None, None, None),
            "execution_mode": "flash",
            "instructions": "be terse",
            "skills": [],
            "knowledge_bases": [],
            "tool_categories": ["basic"],
        },
        excluded_agent_id=7,
    )


class _QueryCounter:
    """Wrap ``db.query`` so we can count invocations per model class.

    ``Session.query(Model)`` is the SQLAlchemy entry point; later
    ``.filter()`` / ``.first()`` chain calls don't re-enter
    ``Session.query``, so counting at the entry is enough to detect a
    Task or User SELECT.
    """

    def __init__(self) -> None:
        self.calls_by_model: Counter[type] = Counter()
        self._returns: dict[type, Any] = {}

    def set_first(self, model: type, value: Any) -> None:
        self._returns[model] = value

    def __call__(self, model: type) -> Any:
        self.calls_by_model[model] += 1
        result = MagicMock()
        result.filter = MagicMock(return_value=result)
        result.first = MagicMock(return_value=self._returns.get(model))
        result.all = MagicMock(return_value=[])
        result.order_by = MagicMock(return_value=result)
        return result


def _build_db_mock(*, task_row: Any, user_row: Any) -> tuple[MagicMock, _QueryCounter]:
    counter = _QueryCounter()
    counter.set_first(Task, task_row)
    counter.set_first(User, user_row)
    db = MagicMock()
    db.query = counter
    return db, counter


def _common_patches(db: Any, agent_service: Any) -> list[Any]:
    """Stub persistence so this test observes only snapshot handoff."""

    return [
        patch(
            "xagent.web.services.task_setup_snapshot.load_task_setup_snapshot_sync",
            return_value=_make_snapshot(),
        ),
        patch(
            "xagent.web.api.websocket.background_task_manager.wait_for_previous",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.websocket._register_uploaded_files_for_agent",
        ),
        patch(
            "xagent.web.api.websocket._finalize_task_execution_result_isolated",
            return_value=SimpleNamespace(
                normalized_outputs=[],
                ai_response="ok",
                chat_response=None,
                waiting_for_control=False,
                terminal_state_committed=True,
                final_control_snapshot=None,
                final_task_status=TaskStatus.COMPLETED.value,
                broadcast_meta={
                    "id": 42,
                    "title": "exec-bg test",
                    "description": "x",
                    "execution_mode": "flash",
                    "updated_at": None,
                },
                late_result=False,
            ),
        ),
        patch(
            "xagent.web.api.websocket.manager.broadcast_to_task",
            new=AsyncMock(),
        ),
    ]


def _build_fake_agent_service() -> MagicMock:
    """Minimal stand-in for ``AgentService`` covering the methods
    ``execute_task_background`` calls after the queries we're counting.
    Returning a successful run keeps the downstream finalize path
    happy (status update + persist) without a real agent runtime.
    """
    svc = MagicMock()
    svc.set_outbound_message_handler = MagicMock()
    svc.set_conversation_history = MagicMock()
    svc.set_execution_context_messages = MagicMock()
    svc.set_recovered_skill_context = MagicMock()
    svc.execute_task = AsyncMock(
        return_value={"success": True, "output": "ok", "status": "completed"}
    )
    svc.workspace = None
    return svc


@pytest.mark.asyncio
async def test_snapshot_path_skips_task_and_user_queries() -> None:
    """A supplied snapshot must require no task/user checkout on the loop."""
    db, counter = _build_db_mock(task_row=_make_task_orm(), user_row=_make_user_orm())
    snapshot = _make_snapshot()
    agent_service = _build_fake_agent_service()
    agent_manager = MagicMock(
        get_agent_for_task=AsyncMock(return_value=agent_service),
        execute_task=AsyncMock(
            return_value={"success": True, "output": "ok", "status": "completed"}
        ),
    )

    with _Patches(_common_patches(db, agent_service)):
        try:
            await execute_task_background(
                task_id=42,
                user_message="hi",
                context={},
                agent_manager=agent_manager,
                task_owner_user_id=1,
                task_setup_snapshot=snapshot,
            )
        except Exception:
            # Downstream finalize stubs may raise; the query counts
            # are recorded before that point.
            pass

    assert counter.calls_by_model[Task] == 0, (
        f"Task queried {counter.calls_by_model[Task]} time(s) on the request "
        "session with snapshot provided -- expected 0. The snapshot "
        "passthrough exists to skip this re-read."
    )
    assert counter.calls_by_model[User] == 0, (
        f"User queried {counter.calls_by_model[User]} time(s) on the event-loop "
        "session with snapshot provided -- expected 0."
    )
    forwarded_snapshot = agent_manager.get_agent_for_task.await_args.kwargs[
        "task_setup_snapshot"
    ]
    assert forwarded_snapshot is snapshot
    assert forwarded_snapshot.task.source == "trigger"


@pytest.mark.asyncio
@pytest.mark.parametrize("broadcast_fails", [False, True])
async def test_cancellation_during_finalization_broadcasts_committed_result(
    broadcast_fails: bool,
) -> None:
    snapshot = _make_snapshot()
    agent_service = _build_fake_agent_service()
    agent_manager = MagicMock(
        get_agent_for_task=AsyncMock(return_value=agent_service),
        execute_task=AsyncMock(
            return_value={"success": True, "output": "ok", "status": "completed"}
        ),
    )
    finalization_started = threading.Event()
    allow_finalization = threading.Event()
    broadcast = AsyncMock(
        side_effect=RuntimeError("client disconnected") if broadcast_fails else None
    )

    def blocking_finalize(**_kwargs: Any) -> Any:
        finalization_started.set()
        assert allow_finalization.wait(timeout=2)
        return SimpleNamespace(
            normalized_outputs=[],
            ai_response="ok",
            chat_response=None,
            waiting_for_control=False,
            terminal_state_committed=True,
            final_control_snapshot=None,
            final_task_status=TaskStatus.COMPLETED.value,
            broadcast_meta={
                "id": 42,
                "title": "exec-bg test",
                "description": "x",
                "execution_mode": "flash",
                "updated_at": None,
            },
            late_result=False,
        )

    patches = [
        patch(
            "xagent.web.api.websocket.background_task_manager.wait_for_previous",
            new=AsyncMock(),
        ),
        patch("xagent.web.api.websocket._register_uploaded_files_for_agent"),
        patch(
            "xagent.web.api.websocket._finalize_task_execution_result_isolated",
            side_effect=blocking_finalize,
        ),
        patch(
            "xagent.web.api.websocket.manager.broadcast_to_task",
            new=broadcast,
        ),
    ]
    with _Patches(patches):
        execution = asyncio.create_task(
            execute_task_background(
                task_id=42,
                user_message="hi",
                context={},
                agent_manager=agent_manager,
                task_owner_user_id=1,
                task_setup_snapshot=snapshot,
                resolved_execution_scope=None,
            )
        )
        await asyncio.wait_for(
            asyncio.to_thread(finalization_started.wait, 1),
            timeout=1,
        )
        execution.cancel()
        await asyncio.sleep(0)
        assert not execution.done()

        allow_finalization.set()
        with pytest.raises(asyncio.CancelledError):
            await execution

    broadcast.assert_awaited_once()
    assert broadcast.await_args.args[0]["type"] == "task_completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("with_task_lease", [False, True])
async def test_cancellation_after_uncommitted_finalization_always_propagates(
    with_task_lease: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="xagent.web.api.websocket")
    snapshot = _make_snapshot()
    agent_service = _build_fake_agent_service()
    agent_manager = MagicMock(
        get_agent_for_task=AsyncMock(return_value=agent_service),
        execute_task=AsyncMock(
            return_value={"success": True, "output": "ok", "status": "completed"}
        ),
    )
    finalization_started = threading.Event()
    allow_finalization = threading.Event()
    broadcast_error = RuntimeError("client disconnected")
    broadcast = AsyncMock(side_effect=broadcast_error)

    def blocking_finalize(**_kwargs: Any) -> Any:
        finalization_started.set()
        assert allow_finalization.wait(timeout=2)
        return SimpleNamespace(
            normalized_outputs=[],
            ai_response="ok",
            chat_response=None,
            waiting_for_control=False,
            terminal_state_committed=False,
            final_control_snapshot=None,
            final_task_status=TaskStatus.RUNNING.value,
            broadcast_meta={
                "id": 42,
                "title": "exec-bg test",
                "description": "x",
                "execution_mode": "flash",
                "updated_at": None,
            },
            late_result=False,
        )

    patches = [
        patch("xagent.web.api.websocket._register_uploaded_files_for_agent"),
        patch(
            "xagent.web.api.websocket._finalize_task_execution_result_isolated",
            side_effect=blocking_finalize,
        ),
        patch(
            "xagent.web.api.websocket._terminal_task_error_payload",
            return_value=None,
        ),
        patch(
            "xagent.web.api.websocket.manager.broadcast_to_task",
            new=broadcast,
        ),
    ]
    with _Patches(patches):
        execution = asyncio.create_task(
            execute_task_background(
                task_id=42,
                user_message="hi",
                context={},
                agent_manager=agent_manager,
                task_owner_user_id=1,
                task_setup_snapshot=snapshot,
                expected_run_id="run-a",
                task_lease=(
                    TaskLease(task_id=42, runner_id="runner-a", run_id="run-a")
                    if with_task_lease
                    else None
                ),
                resolved_execution_scope=None,
            )
        )
        await asyncio.wait_for(
            asyncio.to_thread(finalization_started.wait, 1),
            timeout=1,
        )
        execution.cancel()
        await asyncio.sleep(0)
        assert not execution.done()

        allow_finalization.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await execution

    assert exc_info.value.__cause__ is broadcast_error
    broadcast.assert_awaited_once()
    assert (
        "Background task 42 cancelled after deferred work failed: client disconnected"
    ) in caplog.text


@pytest.mark.asyncio
async def test_snapshot_execution_start_stays_responsive_with_exhausted_pool(
    tmp_path,
) -> None:
    """The snapshot-to-manager handoff must not checkout on the event loop."""

    engine = create_engine(
        f"sqlite:///{tmp_path / 'execute-startup-pool.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.25,
    )
    SessionLocal = sessionmaker(bind=engine)
    held_connection = engine.connect()
    db = SessionLocal()
    snapshot = _make_snapshot()
    manager_entered = asyncio.Event()
    release_manager = asyncio.Event()
    ticker_stop = asyncio.Event()
    ticks = 0

    async def blocked_get_agent(*_args: Any, **_kwargs: Any) -> None:
        manager_entered.set()
        await release_manager.wait()
        raise RuntimeError("stop after the manager boundary")

    async def ticker() -> None:
        nonlocal ticks
        while not ticker_stop.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    agent_manager = MagicMock(
        get_agent_for_task=AsyncMock(side_effect=blocked_get_agent)
    )

    with _Patches(
        [
            *_common_patches(db, MagicMock()),
            patch(
                "xagent.web.api.websocket._terminal_task_error_payload",
                return_value=None,
            ),
        ]
    ):
        started = monotonic()
        execution_task = asyncio.create_task(
            execute_task_background(
                task_id=42,
                user_message="hi",
                context={},
                agent_manager=agent_manager,
                task_owner_user_id=1,
                task_setup_snapshot=snapshot,
                resolved_execution_scope=None,
            )
        )
        ticker_task = asyncio.create_task(ticker())
        try:
            await asyncio.wait_for(manager_entered.wait(), timeout=0.4)
            elapsed = monotonic() - started
            await asyncio.sleep(0.04)
            assert elapsed < 0.15, (
                "Snapshot handoff waited for the exhausted request pool; "
                f"manager reached after {elapsed:.3f}s."
            )
            assert ticks >= 3, f"Event-loop ticker advanced only {ticks} time(s)."
        finally:
            release_manager.set()
            ticker_stop.set()
            await asyncio.gather(
                execution_task,
                ticker_task,
                return_exceptions=True,
            )
            db.close()
            held_connection.close()
            engine.dispose()


@pytest.mark.asyncio
async def test_execute_task_background_accepts_missing_context() -> None:
    """``context=None`` means no turn metadata, not a setup failure."""

    db, _counter = _build_db_mock(task_row=_make_task_orm(), user_row=_make_user_orm())
    snapshot = _make_snapshot()
    agent_service = _build_fake_agent_service()
    agent_manager = MagicMock(get_agent_for_task=AsyncMock(return_value=agent_service))

    with _Patches(_common_patches(db, agent_service)):
        try:
            await execute_task_background(
                task_id=42,
                user_message="hi",
                context=None,
                agent_manager=agent_manager,
                task_owner_user_id=1,
                task_setup_snapshot=snapshot,
            )
        except Exception:
            # The test only covers setup argument shaping. Downstream finalize
            # stubs may raise after get_agent_for_task has received the args.
            pass

    agent_manager.get_agent_for_task.assert_awaited_once()
    assert (
        agent_manager.get_agent_for_task.await_args.kwargs["connector_runtime_turn_id"]
        is None
    )


@pytest.mark.asyncio
async def test_missing_snapshot_is_loaded_off_loop_without_event_loop_queries() -> None:
    """A legacy caller is upgraded to the worker-owned snapshot boundary."""
    db, counter = _build_db_mock(task_row=_make_task_orm(), user_row=_make_user_orm())
    agent_service = _build_fake_agent_service()
    agent_manager = MagicMock(
        get_agent_for_task=AsyncMock(return_value=agent_service),
        execute_task=AsyncMock(
            return_value={"success": True, "output": "ok", "status": "completed"}
        ),
    )
    loader_threads: list[int] = []

    def load_snapshot(*_args: Any, **_kwargs: Any) -> TaskSetupSnapshot:
        loader_threads.append(threading.get_ident())
        return _make_snapshot()

    patches = _common_patches(db, agent_service)
    patches[0] = patch(
        "xagent.web.services.task_setup_snapshot.load_task_setup_snapshot_sync",
        side_effect=load_snapshot,
    )
    with _Patches(patches):
        await execute_task_background(
            task_id=42,
            user_message="hi",
            context={},
            agent_manager=agent_manager,
            task_owner_user_id=1,
            task_setup_snapshot=None,
        )

    assert counter.calls_by_model[Task] == 0
    assert counter.calls_by_model[User] == 0
    assert loader_threads and loader_threads[0] != threading.get_ident()
    assert agent_manager.get_agent_for_task.await_args.args[1] is None


class _Patches:
    def __init__(self, patches: list[Any]) -> None:
        self._patches = patches

    def __enter__(self) -> None:
        for p in self._patches:
            p.start()

    def __exit__(self, *exc_info: Any) -> None:
        for p in reversed(self._patches):
            p.stop()
