"""Issue #935 regression tests for the WebSocket pause DB boundary."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.orm import Session

from xagent.web.api import chat as chat_api
from xagent.web.api import websocket as websocket_api
from xagent.web.models import database as database_module
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services import task_setup_snapshot as snapshot_module
from xagent.web.services.task_execution_controller import (
    StaleTaskRunError,
    TaskControlState,
)


@pytest.fixture()
def db_session(tmp_path: Path) -> Iterator[Session]:
    init_db(db_url=f"sqlite:///{tmp_path / 'pause-boundary.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


@pytest.mark.asyncio
async def test_pause_handler_keeps_database_work_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_loop_thread = threading.get_ident()
    worker_threads: dict[str, int] = {}
    task_id = 41
    owner_id = 7
    actor = SimpleNamespace(id=owner_id, is_admin=False)
    runtime_user = SimpleNamespace(id=owner_id, is_admin=False)
    snapshot = SimpleNamespace(
        task=SimpleNamespace(
            user_id=owner_id,
            status=TaskStatus.RUNNING,
            run_id="run-1",
        ),
        runtime_user=runtime_user,
    )

    def load_snapshot(*args: object, **kwargs: object) -> object:
        worker_threads["snapshot"] = threading.get_ident()
        assert args == (task_id, None)
        assert kwargs == {
            "actor_user_id": owner_id,
            "actor_is_admin": False,
        }
        return snapshot

    def resolve_scope(resolved_task_id: int) -> None:
        worker_threads["scope"] = threading.get_ident()
        assert resolved_task_id == task_id
        return None

    def finalize_pause(resolved_task_id: int, *, expected_run_id: str | None) -> bool:
        worker_threads["finalize"] = threading.get_ident()
        assert resolved_task_id == task_id
        assert expected_run_id == "run-1"
        return True

    def forbidden_event_loop_db() -> object:
        raise AssertionError("pause handler opened a request Session on the event loop")

    agent_service = MagicMock()
    agent_service.pause_execution = AsyncMock(return_value=True)
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()

    monkeypatch.setattr(snapshot_module, "load_task_setup_snapshot_sync", load_snapshot)
    monkeypatch.setattr(websocket_api, "resolve_execution_scope", resolve_scope)
    monkeypatch.setattr(
        websocket_api,
        "_apply_pause_requested_isolated",
        finalize_pause,
        raising=False,
    )
    monkeypatch.setattr(database_module, "get_db", forbidden_event_loop_db)
    monkeypatch.setattr(chat_api, "get_agent_manager", lambda: agent_manager)
    monkeypatch.setattr(websocket_api, "manager", connection_manager)

    try:
        await websocket_api._handle_pause_task_unserialized(
            MagicMock(),
            task_id,
            {"user": actor},
        )
    finally:
        websocket_api._clear_task_pause_accepted(task_id)

    assert set(worker_threads) == {"snapshot", "scope", "finalize"}
    assert all(thread_id != event_loop_thread for thread_id in worker_threads.values())
    agent_manager.get_agent_for_task.assert_awaited_once()
    call = agent_manager.get_agent_for_task.await_args
    assert call.args == (task_id, None)
    assert call.kwargs == {
        "user": runtime_user,
        "task_setup_snapshot": snapshot,
        "task_owner_user_id": owner_id,
        "resolved_execution_scope": None,
    }
    agent_service.pause_execution.assert_awaited_once_with()
    connection_manager.broadcast_to_task.assert_awaited_once()


def _running_task(db_session: Session, *, run_id: str = "run-1") -> Task:
    user = User(username=f"pause-owner-{run_id}", password_hash="x")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    task = Task(
        user_id=int(user.id),
        title="pause boundary",
        description="pause boundary",
        status=TaskStatus.RUNNING,
        execution_mode="balanced",
        run_id=run_id,
        state_version=3,
        control_state=TaskControlState.RUNNING.value,
    )
    db_session.add(task)
    db_session.commit()
    db_session.refresh(task)
    return task


def test_pause_transition_updates_only_the_expected_running_run(
    db_session: Session,
) -> None:
    task = _running_task(db_session)

    applied = websocket_api._apply_pause_requested_isolated(
        int(task.id),
        expected_run_id="run-1",
    )

    db_session.expire_all()
    stored = db_session.query(Task).filter(Task.id == int(task.id)).one()
    assert applied is True
    assert stored.status == TaskStatus.RUNNING
    assert stored.run_id == "run-1"
    assert stored.control_state == TaskControlState.PAUSE_REQUESTED.value
    assert stored.state_version == 4


def test_pause_transition_rejects_a_replacement_run(db_session: Session) -> None:
    task = _running_task(db_session, run_id="replacement-run")

    with pytest.raises(StaleTaskRunError, match="run changed"):
        websocket_api._apply_pause_requested_isolated(
            int(task.id),
            expected_run_id="original-run",
        )

    db_session.expire_all()
    stored = db_session.query(Task).filter(Task.id == int(task.id)).one()
    assert stored.run_id == "replacement-run"
    assert stored.control_state == TaskControlState.RUNNING.value
    assert stored.state_version == 3


def test_pause_transition_leaves_a_terminal_task_unchanged(
    db_session: Session,
) -> None:
    task = _running_task(db_session)
    setattr(task, "status", TaskStatus.COMPLETED)
    setattr(task, "control_state", TaskControlState.COMPLETED.value)
    db_session.commit()

    applied = websocket_api._apply_pause_requested_isolated(
        int(task.id),
        expected_run_id="run-1",
    )

    db_session.expire_all()
    stored = db_session.query(Task).filter(Task.id == int(task.id)).one()
    assert applied is False
    assert stored.status == TaskStatus.COMPLETED
    assert stored.control_state == TaskControlState.COMPLETED.value
    assert stored.state_version == 3
