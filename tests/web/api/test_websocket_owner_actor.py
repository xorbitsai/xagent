"""Owner/actor regression pins for the WebSocket control handlers.

An admin may operate on another user's task (admin bypass), but the agent
runtime must run as the task OWNER, not the admin. A non-admin who is not the
owner must be refused before any runtime is built. These pin the pause / resume
handlers directly (the focused unit tests cover get_agent_for_task in
isolation; here we exercise the handlers end to end).
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, TypeVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from tests.shared.execution_scope import register_scope_resolver
from xagent.core.agent.checkpoint import (
    CheckpointAccessRefusedError,
    CheckpointCorruptError,
    CheckpointUnavailableError,
)
from xagent.core.agent.runner import UserMessageInjectionOutcome
from xagent.core.execution_scope import (
    ExecutionScope,
)
from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRef
from xagent.web.api import websocket as websocket_api
from xagent.web.api.websocket import (
    ResumeReservationOutcome,
    _claim_user_message_delivery_isolated,
    _execute_durable_task_command,
    _handle_chat_message_unserialized,
    _handle_pause_task_unserialized,
    _handle_resume_task_unserialized,
    _restore_resumed_task_lease_to_prior_status,
    _waiting_or_paused_event_fields,
    background_task_manager,
    execute_resume_background,
    handle_chat_message,
    handle_pause_task,
    handle_resume_task,
    send_message_delivery,
)
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services import connector_runtime as connector_runtime_module
from xagent.web.services import task_orchestrator
from xagent.web.services.chat_history_service import DELIVERY_FAILED, DELIVERY_PENDING
from xagent.web.services.connector_runtime import (
    get_ephemeral_runtime_values,
    pop_ephemeral_runtime_values,
    store_ephemeral_runtime_values,
)
from xagent.web.services.managed_file_ref import (
    DurableObjectIntegrityError,
    DurableStorageOperationError,
)
from xagent.web.services.task_command_transport import (
    COMMAND_COMPLETED,
    ClaimedTaskCommand,
    TaskCommandKind,
    TaskCommandRejected,
)
from xagent.web.services.task_execution_controller import StaleTaskRunError
from xagent.web.services.task_lease_service import (
    TaskLease,
    current_task_lease,
    get_runner_id,
)


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'owner_actor.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


def _user(db, username, *, is_admin=False) -> User:
    u = User(username=username, password_hash="x", is_admin=is_admin)
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def _task(db, owner_id: int, status: TaskStatus = TaskStatus.RUNNING) -> Task:
    t = Task(
        user_id=owner_id,
        title="t",
        description="d",
        status=status,
        execution_mode="balanced",
        source="sdk",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


def _seed_active_interaction_row(
    db: Session, *, task_id: int, run_id: str, idempotency_key: str
) -> int:
    """One legal active TaskInteractionRequest row, for the replay-skips-
    close tests below. Mirrors test_a2a_api.py's local, single-purpose row
    builder of the same name rather than sharing it across test files."""
    anchor = TraceEvent(
        task_id=task_id,
        event_id=f"anchor-{idempotency_key}",
        event_type="agent_execution_checkpoint",
        timestamp=datetime.now(timezone.utc),
        data={},
    )
    db.add(anchor)
    db.flush()
    row = TaskInteractionRequest(
        task_id=task_id,
        run_id=run_id,
        kind="clarification",
        protocol_version=1,
        status="active",
        active_slot=1,
        origin="sdk",
        request_payload={"prompt": "example"},
        request_idempotency_key=idempotency_key,
        resume_trace_event_id=int(anchor.id),
        resume_event_id="resume-event-1",
        resume_execution_id="resume-execution-1",
        resume_locator_format="trace_event_pk_v1",
        resume_checkpoint_type="agent_execution_checkpoint",
        resume_run_partition=run_id,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return int(row.id)


# Anti-hang bounds, not latency assertions. Nothing here is measuring speed:
# these handlers finish in milliseconds, and the budgets only exist so a
# deadlock fails as a test error instead of hanging the run. They are therefore
# set far above any plausible runtime on a contended CI runner. A wait on a
# signal crossing the loop/worker-thread boundary must expire *before* the
# handler deadline, so a stall is reported by the specific assertion that owns
# the signal rather than by a bare TimeoutError on the whole handler.
_THREAD_SIGNAL_DEADLINE_SECONDS = 10.0
_HANDLER_DEADLINE_SECONDS = 30.0

_T = TypeVar("_T")


class _EventLoopLivenessProbe:
    """Deterministic replacement for counting event-loop ticks.

    Wrapping a synchronous step in :meth:`gate` makes that step queue an answer
    on the event loop and then block until the loop delivers it. Off the loop
    (in a worker thread) the loop runs the callback on its next pass. On the
    loop thread the queued callback cannot run, because the loop thread is the
    one blocking, so the wait expires. The verdict therefore depends on where
    the work ran, not on how fast the machine is.
    """

    def __init__(self) -> None:
        # Captured here rather than inside the gate: the gated step may run in
        # a worker thread, where get_running_loop() raises.
        self._loop = asyncio.get_running_loop()
        self._answered = threading.Event()
        self.step_ran = False
        self.loop_ran_during_step = False

    def gate(self, step: Callable[..., _T]) -> Callable[..., _T]:
        """Wrap ``step`` so it holds until the event loop answers."""

        def gated(*args: Any, **kwargs: Any) -> _T:
            assert not self.step_ran, "the probe gates a single call"
            self.step_ran = True
            self._loop.call_soon_threadsafe(self._answered.set)
            self.loop_ran_during_step = self._answered.wait(
                timeout=_THREAD_SIGNAL_DEADLINE_SECONDS
            )
            return step(*args, **kwargs)

        return gated

    def assert_loop_stayed_responsive(self, what: str) -> None:
        assert self.step_ran, f"{what} never ran"
        assert self.loop_ran_during_step, f"{what} blocked the asyncio event loop"


@pytest.mark.asyncio
async def test_rejected_delivery_serializes_an_explicit_outcome() -> None:
    ws_manager = MagicMock(send_personal_message=AsyncMock())
    with patch("xagent.web.api.websocket.manager", ws_manager):
        await send_message_delivery(
            MagicMock(),
            client_message_id="rejected-turn",
            turn_id="rejected-turn",
            accepted=False,
            rejection_outcome="not_accepted",
        )

    payload = ws_manager.send_personal_message.await_args.args[0]
    assert payload["type"] == "message_rejected"
    assert payload["rejection_outcome"] == "not_accepted"

    with pytest.raises(ValueError, match="requires an explicit rejection outcome"):
        await send_message_delivery(
            MagicMock(),
            client_message_id="missing-outcome-turn",
            turn_id="missing-outcome-turn",
            accepted=False,
        )


def _register_current_resume(task_id: int) -> None:
    current = asyncio.current_task()
    assert current is not None
    background_task_manager.resume_tasks[task_id] = current


def _patched_manager_and_agent():
    """Return (patches contextmanagers, captured) wiring get_agent_manager +
    the module ``manager`` so the handler can run without real IO."""
    captured: dict = {}
    agent_service = MagicMock()
    agent_service.pause_execution = AsyncMock(return_value={"status": "paused"})
    agent_service.resume_execution = AsyncMock()
    agent_service.supports_live_control = MagicMock(return_value=False)

    async def _get_agent_for_task(
        task_id, db, *, user=None, task_owner_user_id=None, **_kwargs
    ):
        captured["task_owner_user_id"] = task_owner_user_id
        return agent_service

    mgr = MagicMock()
    mgr.get_agent_for_task = AsyncMock(side_effect=_get_agent_for_task)

    ws_manager = MagicMock()
    ws_manager.send_personal_message = AsyncMock()
    ws_manager.broadcast_to_task = AsyncMock()
    return captured, agent_service, mgr, ws_manager


@pytest.mark.asyncio
async def test_chat_admin_append_to_other_users_task_claims_as_owner(
    db_session,
) -> None:
    """The original #587 regression: an admin appending through
    ``handle_chat_message`` to a task owned by another user. The bug was the
    atomic claim using the actor id, so the owner's appendable task failed with
    ``TaskTurnNotFoundError``. Pin that ``begin_turn`` is invoked with
    ``task_owner_user_id == task.user_id`` (the owner), not the admin actor.
    """
    owner = _user(db_session, "owner")
    admin = _user(db_session, "admin", is_admin=True)
    # COMPLETED -> the WS path treats the follow-up as an APPEND turn.
    task = _task(db_session, owner.id, status=TaskStatus.COMPLETED)

    ws_manager = MagicMock()
    ws_manager.broadcast_to_task = AsyncMock()
    ws_manager.send_personal_message = AsyncMock()
    begin_turn = AsyncMock()

    with (
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch(
            "xagent.web.services.task_orchestrator.TaskTurnOrchestrator.begin_turn",
            new=begin_turn,
        ),
    ):
        await handle_chat_message(
            MagicMock(),
            int(task.id),
            {
                "message": "follow-up",
                "client_message_id": "client-turn-1",
                "user": admin,
                "files": [],
            },
        )
        for _ in range(100):
            if begin_turn.await_count:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("durable admin message was not dispatched in time")

    begin_turn.assert_awaited_once()
    assert begin_turn.await_args.kwargs["task_owner_user_id"] == int(owner.id)
    assert begin_turn.await_args.kwargs["payload"].turn_id == "client-turn-1"
    accepted = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_accepted"
    ]
    assert len(accepted) == 1
    assert accepted[0]["client_message_id"] == "client-turn-1"
    assert accepted[0]["turn_id"] == "client-turn-1"


@pytest.mark.asyncio
async def test_chat_without_an_actor_uses_the_authentication_contract() -> None:
    ws_manager = MagicMock(send_personal_message=AsyncMock())

    with patch("xagent.web.api.websocket.manager", ws_manager):
        await handle_chat_message(
            MagicMock(),
            7,
            {
                "message": "do not echo this request",
                "client_message_id": "unauthenticated-turn",
                "files": [],
            },
        )

    payloads = [
        call.args[0] for call in ws_manager.send_personal_message.call_args_list
    ]
    assert payloads
    assert all(
        payload["error_code"] == "authentication_required" for payload in payloads
    )
    assert all(
        payload["message"] == "Authentication is required to send this message."
        for payload in payloads
    )
    assert "do not echo this request" not in repr(payloads)


@pytest.mark.asyncio
async def test_chat_non_owner_uses_the_neutral_unavailable_contract(db_session) -> None:
    owner = _user(db_session, "chat-access-owner")
    stranger = _user(db_session, "chat-access-stranger")
    task = _task(db_session, owner.id, status=TaskStatus.COMPLETED)
    ws_manager = MagicMock(send_personal_message=AsyncMock())

    with patch("xagent.web.api.websocket.manager", ws_manager):
        await handle_chat_message(
            MagicMock(),
            int(task.id),
            {
                "message": "try another owner's task",
                "client_message_id": "access-denied-turn",
                "user": stranger,
                "files": [],
            },
        )

    payloads = [
        call.args[0] for call in ws_manager.send_personal_message.call_args_list
    ]
    assert payloads
    assert all(payload["error_code"] == "task_unavailable" for payload in payloads)
    assert all(
        payload["message"] == "Task is no longer available." for payload in payloads
    )
    assert f"Task {task.id} does not belong to you" not in repr(payloads)


@pytest.mark.asyncio
async def test_unserialized_chat_non_owner_keeps_the_neutral_contract(
    db_session,
) -> None:
    owner = _user(db_session, "direct-chat-access-owner")
    stranger = _user(db_session, "direct-chat-access-stranger")
    task = _task(db_session, owner.id, status=TaskStatus.COMPLETED)
    ws_manager = MagicMock(send_personal_message=AsyncMock())

    with patch("xagent.web.api.websocket.manager", ws_manager):
        await _handle_chat_message_unserialized(
            MagicMock(),
            int(task.id),
            {
                "message": "try direct execution",
                "client_message_id": "direct-access-denied-turn",
                "user": stranger,
                "files": [],
            },
        )

    payloads = [
        call.args[0] for call in ws_manager.send_personal_message.call_args_list
    ]
    assert payloads
    assert all(payload["error_code"] == "task_unavailable" for payload in payloads)
    assert all(
        payload["message"] == "Task is no longer available." for payload in payloads
    )


@pytest.mark.asyncio
async def test_chat_invalid_shape_uses_the_invalid_message_contract() -> None:
    ws_manager = MagicMock(send_personal_message=AsyncMock())

    with patch("xagent.web.api.websocket.manager", ws_manager):
        await _handle_chat_message_unserialized(
            MagicMock(),
            7,
            {
                "message": {"operator_detail": "must not escape"},
                "client_message_id": "invalid-message-turn",
                "user": SimpleNamespace(id=1, is_admin=False),
                "files": [],
            },
        )

    payloads = [
        call.args[0] for call in ws_manager.send_personal_message.call_args_list
    ]
    assert payloads
    assert all(payload["error_code"] == "invalid_message" for payload in payloads)
    assert all(
        payload["message"] == "The message format is invalid." for payload in payloads
    )
    assert "operator_detail" not in repr(payloads)


@pytest.mark.asyncio
async def test_chat_new_turn_releases_request_transaction_before_orchestrator(
    db_session,
) -> None:
    """Turn preparation must release its worker checkout before the claim."""
    from xagent.web.models.database import get_session_local

    owner = _user(db_session, "new-turn-owner")
    task = _task(db_session, owner.id, status=TaskStatus.COMPLETED)
    owner_id = int(owner.id)
    task_id = int(task.id)
    db_session.close()
    event_loop_thread = threading.get_ident()
    preparation_threads: list[int] = []
    session_closed_before_begin: list[bool] = []
    closed_sessions: list[bool] = []
    SessionLocal = get_session_local()
    prepare_turn = websocket_api._prepare_websocket_turn_sync

    class TrackingSessionContext:
        def __init__(self) -> None:
            self.session = SessionLocal()

        def __enter__(self):
            return self.session

        def __exit__(self, _exc_type, _exc, _traceback) -> None:
            self.session.close()
            closed_sessions.append(True)

    def tracking_session_factory():
        return TrackingSessionContext()

    def tracked_prepare_turn(**kwargs):
        preparation_threads.append(threading.get_ident())
        return prepare_turn(**kwargs)

    async def begin_turn(**_kwargs):
        session_closed_before_begin.append(bool(closed_sessions))

    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    with (
        patch(
            "xagent.web.api.websocket._prepare_websocket_turn_sync",
            side_effect=tracked_prepare_turn,
        ),
        patch(
            "xagent.web.api.websocket.get_session_local",
            return_value=tracking_session_factory,
        ),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch(
            "xagent.web.services.task_orchestrator.TaskTurnOrchestrator.begin_turn",
            side_effect=begin_turn,
        ),
    ):
        await _handle_chat_message_unserialized(
            MagicMock(),
            task_id,
            {
                "message": "follow-up",
                "client_message_id": "pool-boundary-turn",
                "user": SimpleNamespace(id=owner_id, is_admin=False),
                "files": [],
            },
        )

    assert preparation_threads
    assert preparation_threads[0] != event_loop_thread
    assert session_closed_before_begin == [True]


@pytest.mark.asyncio
async def test_chat_turn_releases_one_slot_pool_before_task_info_broadcast(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task-info snapshot checkout must not overlap turn preparation."""

    from xagent.web.models import database as database_module

    engine = create_engine(
        f"sqlite:///{tmp_path / 'websocket-turn-broadcast.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.25,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    with SessionLocal() as db:
        owner = User(
            username="broadcast-pool-owner",
            password_hash="x",
            is_admin=False,
        )
        db.add(owner)
        db.flush()
        task = Task(
            user_id=int(owner.id),
            title="broadcast pool task",
            description="d",
            status=TaskStatus.COMPLETED,
            execution_mode="balanced",
            source="sdk",
        )
        db.add(task)
        db.commit()
        actor_user_id = int(owner.id)
        task_id = int(task.id)

    def local_get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    class RecordingWebSocket:
        def __init__(self) -> None:
            self.messages: list[str] = []

        async def send_text(self, message: str) -> None:
            self.messages.append(message)

    websocket = RecordingWebSocket()
    local_manager = websocket_api.ConnectionManager()
    local_manager.register_connection(websocket, task_id)  # type: ignore[arg-type]
    begin_turn = AsyncMock()

    monkeypatch.setattr(websocket_api, "get_db", local_get_db)
    monkeypatch.setattr(websocket_api, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)
    prepare_turn = websocket_api._prepare_websocket_turn_sync
    monkeypatch.setattr(websocket_api, "manager", local_manager)
    monkeypatch.setattr(
        "xagent.web.services.task_orchestrator.TaskTurnOrchestrator.begin_turn",
        begin_turn,
    )

    # The gate holds preparation open long enough for loop blocking to be
    # observable. Overlap detection is a separate mechanism: the one-slot pool
    # raises on a task-info checkout that collides with preparation.
    probe = _EventLoopLivenessProbe()
    monkeypatch.setattr(
        websocket_api,
        "_prepare_websocket_turn_sync",
        probe.gate(prepare_turn),
    )
    try:
        await asyncio.wait_for(
            _handle_chat_message_unserialized(
                websocket,  # type: ignore[arg-type]
                task_id,
                {
                    "message": "follow-up",
                    "client_message_id": "broadcast-pool-turn",
                    "user": SimpleNamespace(id=actor_user_id, is_admin=False),
                    "files": [],
                },
            ),
            timeout=_HANDLER_DEADLINE_SECONDS,
        )
    finally:
        engine.dispose()

    probe.assert_loop_stayed_responsive("WebSocket preparation")
    begin_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_pause_accepted_wait_releases_one_slot_pool_before_previous_run(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Previous-run settlement must acquire the pool while this handler waits."""

    from xagent.web.models import database as database_module

    engine = create_engine(
        f"sqlite:///{tmp_path / 'websocket-turn-pause.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.25,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    with SessionLocal() as db:
        owner = User(
            username="pause-pool-owner",
            password_hash="x",
            is_admin=False,
        )
        db.add(owner)
        db.flush()
        task = Task(
            user_id=int(owner.id),
            title="pause pool task",
            description="d",
            status=TaskStatus.RUNNING,
            execution_mode="balanced",
            source="sdk",
            run_id="pause-pool-run",
            runner_id="pause-pool-runner",
            control_state="pause_requested",
        )
        db.add(task)
        db.commit()
        actor_user_id = int(owner.id)
        task_id = int(task.id)

    def local_get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    async def settle_previous(_task_id: int) -> None:
        def settle_sync() -> None:
            with SessionLocal() as db:
                current = db.query(Task).filter(Task.id == task_id).one()
                current.status = TaskStatus.PAUSED
                current.control_state = "paused"
                db.commit()

        await asyncio.to_thread(settle_sync)

    background_manager = MagicMock()
    background_manager.wait_for_previous = AsyncMock(side_effect=settle_previous)
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    begin_turn = AsyncMock()

    monkeypatch.setattr(websocket_api, "get_db", local_get_db)
    monkeypatch.setattr(websocket_api, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)
    prepare_turn = websocket_api._prepare_websocket_turn_sync
    monkeypatch.setattr(websocket_api, "manager", ws_manager)
    monkeypatch.setattr(
        websocket_api,
        "background_task_manager",
        background_manager,
    )
    monkeypatch.setattr(
        "xagent.web.services.task_orchestrator.TaskTurnOrchestrator.begin_turn",
        begin_turn,
    )

    websocket_api._mark_task_pause_accepted(task_id)
    probe = _EventLoopLivenessProbe()
    monkeypatch.setattr(
        websocket_api,
        "_prepare_websocket_turn_sync",
        probe.gate(prepare_turn),
    )
    try:
        await asyncio.wait_for(
            _handle_chat_message_unserialized(
                MagicMock(),
                task_id,
                {
                    "message": "continue after pause",
                    "client_message_id": "pause-pool-turn",
                    "user": SimpleNamespace(id=actor_user_id, is_admin=False),
                    "files": [],
                },
            ),
            timeout=_HANDLER_DEADLINE_SECONDS,
        )
    finally:
        websocket_api._clear_task_pause_accepted(task_id)
        engine.dispose()

    probe.assert_loop_stayed_responsive("pause settlement preparation")
    # The settlement itself is guarded by the one-slot pool: it opens its own
    # session, so it can only succeed if the handler released its checkout
    # before awaiting the previous run.
    background_manager.wait_for_previous.assert_awaited_once_with(task_id)
    begin_turn.assert_awaited_once()
    assert begin_turn.await_args.kwargs["kind"].value == "append"


@pytest.mark.asyncio
async def test_missing_task_create_commits_claim_message_file_and_prelease_together(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The replacement task is already a fully claimed CREATE at first send."""

    from xagent.web.models import database as database_module

    engine = create_engine(
        f"sqlite:///{tmp_path / 'websocket-missing-create.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.25,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    file_path = tmp_path / "atomic.txt"
    file_path.write_text("atomic file")
    with SessionLocal() as db:
        owner = User(
            username="missing-create-owner",
            password_hash="x",
            is_admin=False,
        )
        db.add(owner)
        db.flush()
        file_record = UploadedFile(
            file_id="missing-create-file",
            user_id=int(owner.id),
            filename="atomic.txt",
            storage_path=str(file_path),
            mime_type="text/plain",
            file_size=file_path.stat().st_size,
        )
        db.add(file_record)
        db.commit()
        owner_id = int(owner.id)

    observed_task_ids: list[int] = []
    sent_messages: list[dict] = []

    class AtomicCreateManager:
        def move_connection(self, _websocket: object, new_task_id: int) -> None:
            observed_task_ids.append(new_task_id)

        async def send_personal_message(self, event: dict, _websocket: object) -> None:
            sent_messages.append(event)
            if event.get("type") != "task_id_updated":
                return
            assert engine.pool.checkedout() == 0
            new_task_id = int(event["new_task_id"])
            with SessionLocal() as db:
                task = db.query(Task).filter(Task.id == new_task_id).one()
                message = (
                    db.query(TaskChatMessage)
                    .filter(TaskChatMessage.task_id == new_task_id)
                    .one()
                )
                uploaded = (
                    db.query(UploadedFile)
                    .filter(UploadedFile.file_id == "missing-create-file")
                    .one()
                )
                assert task.status == TaskStatus.RUNNING
                assert task.runner_id
                assert task.run_id
                assert task.lease_expires_at is not None
                assert message.content == "start atomically"
                assert message.delivery_status == DELIVERY_PENDING
                assert uploaded.task_id == new_task_id

        async def broadcast_to_task(self, _event: dict, _task_id: int) -> None:
            assert engine.pool.checkedout() == 0

    ws_manager = AtomicCreateManager()
    schedule = AsyncMock()
    begin_turn = AsyncMock()
    monkeypatch.setattr(websocket_api, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(websocket_api, "manager", ws_manager)
    monkeypatch.setattr(
        "xagent.web.services.task_orchestrator.TaskTurnOrchestrator.schedule_claimed_create_turn",
        schedule,
    )
    monkeypatch.setattr(
        "xagent.web.services.task_orchestrator.TaskTurnOrchestrator.begin_turn",
        begin_turn,
    )

    try:
        await _handle_chat_message_unserialized(
            MagicMock(),
            987654,
            {
                "message": "start atomically",
                "client_message_id": "missing-create-turn",
                "user": SimpleNamespace(id=owner_id, is_admin=False),
                "files": [
                    {
                        "file_id": "missing-create-file",
                        "name": "atomic.txt",
                    }
                ],
            },
        )
    finally:
        engine.dispose()

    assert observed_task_ids
    schedule.assert_awaited_once()
    assert schedule.await_args.kwargs["task_id"] == observed_task_ids[0]
    begin_turn.assert_not_awaited()
    assert any(event.get("type") == "message_accepted" for event in sent_messages)


@pytest.mark.asyncio
async def test_missing_task_file_bind_race_rolls_back_the_whole_create(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lost file claim cannot expose a Task without its first turn."""

    from xagent.web.models import database as database_module

    engine = create_engine(
        f"sqlite:///{tmp_path / 'websocket-missing-bind-race.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.25,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    file_path = tmp_path / "raced.txt"
    file_path.write_text("raced file")
    with SessionLocal() as db:
        owner = User(
            username="missing-bind-race-owner",
            password_hash="x",
            is_admin=False,
        )
        db.add(owner)
        db.flush()
        db.add(
            UploadedFile(
                file_id="missing-bind-race-file",
                user_id=int(owner.id),
                filename="raced.txt",
                storage_path=str(file_path),
                mime_type="text/plain",
                file_size=file_path.stat().st_size,
            )
        )
        db.commit()
        owner_id = int(owner.id)

    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    schedule = AsyncMock()
    monkeypatch.setattr(websocket_api, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(websocket_api, "manager", ws_manager)
    monkeypatch.setattr(
        task_orchestrator,
        "bind_turn_files_no_commit",
        MagicMock(return_value=["missing-bind-race-file"]),
    )
    monkeypatch.setattr(
        "xagent.web.services.task_orchestrator.TaskTurnOrchestrator.schedule_claimed_create_turn",
        schedule,
    )

    try:
        await _handle_chat_message_unserialized(
            MagicMock(),
            654321,
            {
                "message": "rollback raced create",
                "client_message_id": "missing-bind-race-turn",
                "user": SimpleNamespace(id=owner_id, is_admin=False),
                "files": [
                    {
                        "file_id": "missing-bind-race-file",
                        "name": "raced.txt",
                    }
                ],
            },
        )
        with SessionLocal() as db:
            assert (
                db.query(Task)
                .filter(
                    Task.user_id == owner_id,
                    Task.title.like("Chat: rollback raced create%"),
                )
                .count()
                == 0
            )
            assert db.query(TaskChatMessage).count() == 0
            uploaded = (
                db.query(UploadedFile)
                .filter(UploadedFile.file_id == "missing-bind-race-file")
                .one()
            )
            assert uploaded.task_id is None
    finally:
        engine.dispose()

    schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_task_send_failure_leaves_only_ttl_owned_running_claim(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.models import database as database_module

    engine = create_engine(
        f"sqlite:///{tmp_path / 'websocket-missing-send-failure.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.25,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    with SessionLocal() as db:
        owner = User(
            username="missing-send-failure-owner",
            password_hash="x",
            is_admin=False,
        )
        db.add(owner)
        db.commit()
        owner_id = int(owner.id)

    ws_manager = MagicMock()
    ws_manager.send_personal_message = AsyncMock(
        side_effect=RuntimeError("socket send failed")
    )
    ws_manager.broadcast_to_task = AsyncMock()
    schedule = AsyncMock()
    monkeypatch.setattr(websocket_api, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(websocket_api, "manager", ws_manager)
    monkeypatch.setattr(
        "xagent.web.services.task_orchestrator.TaskTurnOrchestrator.schedule_claimed_create_turn",
        schedule,
    )

    try:
        with pytest.raises(RuntimeError, match="socket send failed"):
            await _handle_chat_message_unserialized(
                MagicMock(),
                876543,
                {
                    "message": "send may fail",
                    "client_message_id": "missing-send-failure-turn",
                    "user": SimpleNamespace(id=owner_id, is_admin=False),
                    "files": [],
                },
            )
        with SessionLocal() as db:
            task = (
                db.query(Task)
                .filter(
                    Task.user_id == owner_id,
                    Task.title.like("Chat: send may fail%"),
                )
                .one()
            )
            message = (
                db.query(TaskChatMessage)
                .filter(TaskChatMessage.task_id == int(task.id))
                .one()
            )
            assert task.status == TaskStatus.RUNNING
            assert task.runner_id
            assert task.run_id
            assert task.lease_expires_at is not None
            assert message.delivery_status == DELIVERY_FAILED
    finally:
        engine.dispose()

    schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_task_cancel_after_atomic_create_never_leaves_pending(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.models import database as database_module

    engine = create_engine(
        f"sqlite:///{tmp_path / 'websocket-missing-cancel.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.25,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    with SessionLocal() as db:
        owner = User(
            username="missing-cancel-owner",
            password_hash="x",
            is_admin=False,
        )
        db.add(owner)
        db.commit()
        owner_id = int(owner.id)

    worker_finished = threading.Event()
    release_worker = threading.Event()
    prepare_turn = websocket_api._prepare_websocket_turn_sync

    def blocked_prepare_turn(**kwargs):
        preparation = prepare_turn(**kwargs)
        worker_finished.set()
        assert release_worker.wait(timeout=_THREAD_SIGNAL_DEADLINE_SECONDS)
        return preparation

    schedule = AsyncMock()
    monkeypatch.setattr(websocket_api, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(
        websocket_api,
        "_prepare_websocket_turn_sync",
        blocked_prepare_turn,
    )
    monkeypatch.setattr(
        "xagent.web.services.task_orchestrator.TaskTurnOrchestrator.schedule_claimed_create_turn",
        schedule,
    )

    handler = asyncio.create_task(
        _handle_chat_message_unserialized(
            MagicMock(),
            765432,
            {
                "message": "cancel after commit",
                "client_message_id": "missing-cancel-turn",
                "user": SimpleNamespace(id=owner_id, is_admin=False),
                "files": [],
            },
        )
    )
    try:
        assert await asyncio.to_thread(
            worker_finished.wait, _THREAD_SIGNAL_DEADLINE_SECONDS
        )
        handler.cancel()
        release_worker.set()
        with pytest.raises(asyncio.CancelledError):
            await handler
        with SessionLocal() as db:
            task = (
                db.query(Task)
                .filter(
                    Task.user_id == owner_id,
                    Task.title.like("Chat: cancel after commit%"),
                )
                .one()
            )
            message = (
                db.query(TaskChatMessage)
                .filter(TaskChatMessage.task_id == int(task.id))
                .one()
            )
            assert task.status == TaskStatus.RUNNING
            assert task.runner_id
            assert task.run_id
            assert task.lease_expires_at is not None
            assert message.delivery_status == DELIVERY_PENDING
    finally:
        release_worker.set()
        engine.dispose()

    schedule.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "expected_code", "expected_message"),
    [
        (
            "actor_task_reuse_unsupported",
            "message_continuation_unsupported",
            "Task does not support message continuation.",
        ),
        (
            "workforce_archived",
            "workforce_archived",
            "This workforce has been archived. Unarchive and publish it before "
            "starting a new conversation, or select an active workforce.",
        ),
        (
            "workforce_config_changed",
            "workforce_unavailable",
            "This workforce conversation can no longer accept messages; "
            "please start a new conversation.",
        ),
        (
            "workforce_run_not_found",
            "workforce_unavailable",
            "This workforce conversation can no longer accept messages; "
            "please start a new conversation.",
        ),
        (
            "workforce_run_not_active",
            "workforce_unavailable",
            "This workforce conversation can no longer accept messages; "
            "please start a new conversation.",
        ),
    ],
)
async def test_chat_turn_rejection_payload_is_loaded_off_loop(
    db_session,
    reason: str,
    expected_code: str,
    expected_message: str,
) -> None:
    from xagent.web.services.task_orchestrator import TaskTurnError

    owner = _user(db_session, "busy-payload-owner")
    task = _task(db_session, owner.id, status=TaskStatus.COMPLETED)
    event_loop_thread = threading.get_ident()
    payload_threads: list[int] = []
    payload_messages: list[str] = []
    payload_codes: list[str | None] = []

    def read_payload(_task_id, default_message, *_args, **_kwargs):
        payload_threads.append(threading.get_ident())
        payload_messages.append(default_message)
        payload_codes.append(_kwargs.get("error_code"))
        return {
            "type": "agent_error",
            "message": "busy",
            "error_code": _kwargs.get("error_code"),
        }

    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    with (
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch(
            "xagent.web.services.task_orchestrator.TaskTurnOrchestrator.begin_turn",
            side_effect=TaskTurnError(reason),
        ),
        patch(
            "xagent.web.api.websocket._read_task_error_payload_isolated",
            side_effect=read_payload,
        ),
        patch(
            "xagent.web.api.websocket._task_error_payload",
            side_effect=AssertionError("payload query ran on the event loop"),
        ),
    ):
        await _handle_chat_message_unserialized(
            MagicMock(),
            int(task.id),
            {
                "message": "follow-up",
                "client_message_id": "busy-payload-turn",
                "user": owner,
                "files": [],
            },
        )

    assert len(payload_threads) == 1
    assert payload_threads[0] != event_loop_thread
    assert payload_codes == [expected_code]
    assert payload_messages == [expected_message]
    broadcast = ws_manager.broadcast_to_task.await_args.args[0]
    assert broadcast["error_code"] == expected_code
    rejected = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_rejected"
    ]
    assert rejected[0]["error_code"] == expected_code


@pytest.mark.asyncio
async def test_chat_without_client_id_uses_durable_command_id_as_turn_id(
    db_session,
) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.COMPLETED)
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    begin_turn = AsyncMock()

    with (
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch(
            "xagent.web.services.task_orchestrator.TaskTurnOrchestrator.begin_turn",
            new=begin_turn,
        ),
    ):
        await handle_chat_message(
            MagicMock(),
            int(task.id),
            {"message": "server generated identity", "user": owner, "files": []},
        )
        for _ in range(100):
            if begin_turn.await_count:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("durable message was not dispatched in time")

    db_session.expire_all()
    command = (
        db_session.query(TaskExecutionCommand)
        .filter(TaskExecutionCommand.task_id == int(task.id))
        .one()
    )
    assert command.command_id.startswith("message:")
    assert command.payload["client_message_id"] == command.command_id
    assert begin_turn.await_args.kwargs["payload"].turn_id == command.command_id


@pytest.mark.asyncio
async def test_running_chat_message_is_persisted_before_resume(db_session) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    task.runner_id = "live-runner"
    task.run_id = "live-run"
    db_session.commit()
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    observed_leases: list[TaskLease | None] = []

    async def post_user_message(*_args, **_kwargs) -> UserMessageInjectionOutcome:
        observed_leases.append(current_task_lease())
        return UserMessageInjectionOutcome.POSTED_FRESH

    agent.post_user_message = AsyncMock(side_effect=post_user_message)
    mgr = MagicMock(get_agent_for_task=AsyncMock(return_value=agent))
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    resume_bg = AsyncMock()
    bg_mgr = MagicMock()
    bg_mgr.try_reserve_resume.return_value = ResumeReservationOutcome.RESERVED
    bg_mgr.running_tasks.get.return_value = None

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.execute_resume_background", resume_bg),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
    ):
        await handle_chat_message(
            MagicMock(),
            int(task.id),
            {
                "message": "Use the audio tool",
                "client_message_id": "live-turn-1",
                "user": owner,
                "files": [],
            },
        )
        for _ in range(100):
            if bg_mgr.register_reserved_resume.call_count:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("durable live message was not dispatched in time")

    stored = (
        db_session.query(TaskChatMessage)
        .filter(
            TaskChatMessage.task_id == int(task.id),
            TaskChatMessage.role == "user",
        )
        .one()
    )
    assert stored.content == "Use the audio tool"
    assert stored.turn_id == "live-turn-1"
    assert agent.post_user_message.await_args.kwargs["turn_id"] == "live-turn-1"
    assert observed_leases == [
        TaskLease(
            task_id=int(task.id),
            runner_id="live-runner",
            run_id="live-run",
        )
    ]
    bg_mgr.register_reserved_resume.assert_called_once()
    accepted = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_accepted"
    ]
    assert len(accepted) == 1
    assert accepted[0]["client_message_id"] == "live-turn-1"


@pytest.mark.asyncio
async def test_legacy_continuation_runtime_fails_closed_without_side_effects(
    db_session,
) -> None:
    owner = _user(db_session, "legacy-continuation-owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    task.runner_id = "runner-a"
    task.run_id = "run-a"
    db_session.commit()

    legacy_pattern = MagicMock()
    legacy_pattern.request_continuation = MagicMock()
    agent = MagicMock()
    agent.supports_live_control.return_value = False
    agent.get_dag_pattern.return_value = legacy_pattern
    mgr = MagicMock(get_agent_for_task=AsyncMock(return_value=agent))
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch(
            "xagent.web.api.websocket.mark_user_message_delivery_sync",
        ) as mark_delivery,
    ):
        await _handle_chat_message_unserialized(
            MagicMock(),
            int(task.id),
            {
                "message": "must not enter legacy continuation",
                "client_message_id": "legacy-continuation-1",
                "user": owner,
                "files": [],
            },
        )

    legacy_pattern.request_continuation.assert_not_called()
    mark_delivery.assert_not_called()
    sent_messages = [
        call.args[0] for call in ws_manager.send_personal_message.call_args_list
    ]
    assert any(
        message.get("message") == "Task does not support message continuation."
        and message.get("error_code") == "message_continuation_unsupported"
        for message in sent_messages
    )
    rejection = next(
        message
        for message in sent_messages
        if message.get("type") == "message_rejected"
    )
    assert rejection["error_code"] == "message_continuation_unsupported"


@pytest.mark.asyncio
async def test_running_chat_message_uses_one_offloop_scope_and_no_request_session(
    db_session,
) -> None:
    """The MESSAGE -> live resume handoff must not keep the request Session
    while agent construction or resume coordination awaits.
    """

    owner = _user(db_session, "scope-owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    scope = ExecutionScope(
        sandbox_key_suffix="tenant-a", workspace_segments=("tenant-a",)
    )
    main_thread_id = threading.get_ident()
    resolver_threads: list[int] = []

    def resolver(resolved_task_id: str) -> ExecutionScope:
        assert resolved_task_id == str(task.id)
        resolver_threads.append(threading.get_ident())
        return scope

    register_scope_resolver(resolver)
    original_get_db = get_db
    tracked_sessions: list[SimpleNamespace] = []

    class TrackingSession:
        def __init__(self, inner: object) -> None:
            self.inner = inner
            self.state = SimpleNamespace(closed=False)
            tracked_sessions.append(self.state)

        def __getattr__(self, name: str) -> object:
            return getattr(self.inner, name)

        def close(self) -> None:
            self.state.closed = True
            self.inner.close()  # type: ignore[attr-defined]

    def tracked_get_db():
        inner_generator = original_get_db()
        tracked = TrackingSession(next(inner_generator))
        try:
            yield tracked
        finally:
            tracked.close()
            inner_generator.close()

    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    agent.post_user_message = AsyncMock(
        return_value=UserMessageInjectionOutcome.POSTED_FRESH
    )
    manager_calls: list[tuple[object, dict]] = []
    claim_threads: list[int] = []

    def claim_delivery(**kwargs: object):
        claim_threads.append(threading.get_ident())
        return _claim_user_message_delivery_isolated(**kwargs)  # type: ignore[arg-type]

    async def get_agent_for_task(_task_id: int, db: object, **kwargs: object) -> object:
        manager_calls.append(
            (
                db,
                {
                    **kwargs,
                    "request_sessions_closed": all(
                        session.closed for session in tracked_sessions
                    ),
                },
            )
        )
        return agent

    agent_manager = MagicMock(
        get_agent_for_task=AsyncMock(side_effect=get_agent_for_task)
    )
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    resume_bg = AsyncMock()
    bg_manager = MagicMock()
    bg_manager.try_reserve_resume.return_value = ResumeReservationOutcome.RESERVED
    bg_manager.running_tasks.get.return_value = None

    try:
        with (
            patch("xagent.web.api.websocket.get_db", side_effect=tracked_get_db),
            patch(
                "xagent.web.api.websocket._claim_user_message_delivery_isolated",
                side_effect=claim_delivery,
            ),
            patch("xagent.web.api.chat.get_agent_manager", return_value=agent_manager),
            patch("xagent.web.api.websocket.manager", ws_manager),
            patch("xagent.web.api.websocket.execute_resume_background", resume_bg),
            patch("xagent.web.api.websocket.background_task_manager", bg_manager),
            patch(
                "xagent.web.api.websocket.task_execution_controller.transition",
                new=AsyncMock(
                    return_value=SimpleNamespace(
                        run_id="live-run", status=TaskStatus.RUNNING
                    )
                ),
            ),
        ):
            await _handle_chat_message_unserialized(
                MagicMock(),
                int(task.id),
                {
                    "message": "continue safely",
                    "client_message_id": "scope-live-turn",
                    "user": owner,
                    "files": [],
                    "_durable_ack_sent": True,
                },
            )
            await asyncio.sleep(0)
    finally:
        register_scope_resolver(None)

    assert len(manager_calls) == 1
    manager_db, manager_kwargs = manager_calls[0]
    assert manager_db is None
    assert manager_kwargs["request_sessions_closed"] is True
    assert manager_kwargs["task_setup_snapshot"] is not None
    assert manager_kwargs["resolved_execution_scope"] is scope
    assert resolver_threads and resolver_threads[0] != main_thread_id
    assert len(resolver_threads) == 1
    assert claim_threads and claim_threads[0] != main_thread_id
    assert resume_bg.await_args.kwargs["resolved_execution_scope"] is scope


@pytest.mark.asyncio
async def test_deferred_chat_message_is_acked_after_durable_command_commit(
    db_session,
) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    agent.post_user_message = AsyncMock(return_value=False)
    mgr = MagicMock(get_agent_for_task=AsyncMock(return_value=agent))
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    resume_bg = AsyncMock()
    bg_mgr = MagicMock()
    bg_mgr.try_reserve_resume.return_value = ResumeReservationOutcome.RESERVED
    bg_mgr.running_tasks.get.return_value = None
    websocket = MagicMock()

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.execute_resume_background", resume_bg),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
    ):
        await handle_chat_message(
            websocket,
            int(task.id),
            {
                "message": "Wait for the checkpoint",
                "client_message_id": "deferred-turn-1",
                "user": owner,
                "files": [],
            },
        )
        for _ in range(100):
            db_session.expire_all()
            stored_command = (
                db_session.query(TaskExecutionCommand)
                .filter_by(task_id=int(task.id), command_id="deferred-turn-1")
                .one()
            )
            if (
                stored_command.status == "pending"
                and int(stored_command.attempt_count or 0) >= 1
                and resume_bg.await_count == 1
            ):
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("deferred command claim was not released in time")

    accepted = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_accepted"
    ]
    assert len(accepted) == 1
    assert accepted[0]["client_message_id"] == "deferred-turn-1"
    assert stored_command.status == "pending"
    assert not any(
        call.args[0].get("type") == "message_rejected"
        for call in ws_manager.send_personal_message.call_args_list
    )
    kwargs = resume_bg.call_args.kwargs
    assert kwargs["delivery_already_dispatched"] is False
    assert kwargs["delivery_websocket"] is None
    assert kwargs["delivery_client_message_id"] is None


@pytest.mark.asyncio
async def test_live_lease_injection_degrades_to_deferred_on_checkpoint_unavailable(
    db_session,
) -> None:
    """A checkpoint read failure during live injection must fold into the
    same posted=False deferred-delivery path as no exact lease at all --
    not a raw exception and not a rejection."""
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    task.runner_id = "unavailable-runner"
    task.run_id = "unavailable-run"
    db_session.commit()
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    agent.post_user_message = AsyncMock(
        side_effect=CheckpointUnavailableError("checkpoint query failed")
    )
    mgr = MagicMock(get_agent_for_task=AsyncMock(return_value=agent))
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    resume_bg = AsyncMock()
    bg_mgr = MagicMock()
    bg_mgr.try_reserve_resume.return_value = ResumeReservationOutcome.RESERVED
    bg_mgr.running_tasks.get.return_value = None
    websocket = MagicMock()

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.execute_resume_background", resume_bg),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
    ):
        await handle_chat_message(
            websocket,
            int(task.id),
            {
                "message": "Wait for the checkpoint",
                "client_message_id": "unavailable-turn-1",
                "user": owner,
                "files": [],
            },
        )
        for _ in range(100):
            db_session.expire_all()
            stored_command = (
                db_session.query(TaskExecutionCommand)
                .filter_by(task_id=int(task.id), command_id="unavailable-turn-1")
                .one()
            )
            if (
                stored_command.status == "pending"
                and int(stored_command.attempt_count or 0) >= 1
                and resume_bg.await_count == 1
            ):
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("deferred command claim was not released in time")

    accepted = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_accepted"
    ]
    assert len(accepted) == 1
    assert accepted[0]["client_message_id"] == "unavailable-turn-1"
    assert stored_command.status == "pending"
    assert not any(
        call.args[0].get("type") == "message_rejected"
        for call in ws_manager.send_personal_message.call_args_list
    )
    kwargs = resume_bg.call_args.kwargs
    assert kwargs["delivery_already_dispatched"] is False
    assert kwargs["delivery_websocket"] is None
    assert kwargs["delivery_client_message_id"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "sender_error_code", "sender_message"),
    [
        (
            CheckpointCorruptError("all matching rows undecodable"),
            "task_checkpoint_unreadable",
            "The task's saved progress could not be read.",
        ),
        (
            CheckpointAccessRefusedError("active lease is not bound to this reader"),
            "task_checkpoint_unreadable",
            "The task's saved progress could not be read.",
        ),
        (
            DurableObjectIntegrityError(
                "checksum mismatch",
                storage_key="users/1/uploads/corrupt.txt",
            ),
            "message_attachment_corrupt",
            "A stored file for this message failed its integrity check and must be re-uploaded.",
        ),
        (
            DurableStorageOperationError(
                "provider unavailable",
                storage_key="users/1/uploads/unavailable.txt",
            ),
            "message_attachment_unavailable",
            "A stored file for this message could not be read. Please try again.",
        ),
    ],
    ids=[
        "checkpoint-corrupt",
        "checkpoint-access-refused",
        "attachment-corrupt",
        "attachment-unavailable",
    ],
)
@pytest.mark.parametrize(
    "durable_ack_sent",
    [False, True],
    ids=["live-ack", "suppressed-ack"],
)
async def test_durable_failure_keeps_detail_sender_only(
    db_session,
    error: Exception,
    sender_error_code: str,
    sender_message: str,
    durable_ack_sent: bool,
) -> None:
    """Answer a durable turn failure without disclosing it to subscribers."""
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    task.runner_id = "rejected-runner"
    task.run_id = "rejected-run"
    db_session.commit()
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    agent.post_user_message = AsyncMock(side_effect=error)
    mgr = MagicMock(get_agent_for_task=AsyncMock(return_value=agent))
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    resume_bg = AsyncMock()
    bg_mgr = MagicMock()
    bg_mgr.try_reserve_resume.return_value = ResumeReservationOutcome.RESERVED
    bg_mgr.running_tasks.get.return_value = None
    origin_socket = MagicMock(name="origin-socket")
    payload = {
        "message": "Wait for the checkpoint",
        "client_message_id": "rejected-turn-1",
        "user": owner,
        "files": [],
    }
    if durable_ack_sent:
        payload["_durable_ack_sent"] = True

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.execute_resume_background", resume_bg),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
    ):
        await _handle_chat_message_unserialized(
            origin_socket,
            int(task.id),
            payload,
        )

    resume_bg.assert_not_awaited()
    bg_mgr.release_resume_reservation.assert_called_once_with(int(task.id))
    rejected = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_rejected"
    ]
    if durable_ack_sent:
        assert not rejected
        sender_errors = [
            call
            for call in ws_manager.send_personal_message.call_args_list
            if call.args[0].get("type") == "error"
        ]
        assert len(sender_errors) == 1
        assert sender_errors[0].args[0]["error_code"] == sender_error_code
        assert sender_errors[0].args[0]["message"] == sender_message
        assert sender_errors[0].args[1] is origin_socket
    else:
        assert len(rejected) == 1
        assert rejected[0]["client_message_id"] == "rejected-turn-1"
        assert rejected[0]["error_code"] == sender_error_code
        assert rejected[0]["message"] == sender_message
        assert rejected[0]["rejection_outcome"] == "not_accepted"
    task_errors = [
        call.args[0]
        for call in ws_manager.broadcast_to_task.call_args_list
        if call.args[0].get("type") in {"error", "agent_error", "task_error"}
    ]
    assert task_errors
    assert all(
        payload["error_code"] == "task_execution_failed"
        and payload["message"] == websocket_api.CLIENT_SAFE_TASK_FAILURE
        for payload in task_errors
    )
    db_session.expire_all()
    stored = (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.turn_id == "rejected-turn-1")
        .one()
    )
    assert stored.delivery_status == DELIVERY_FAILED


@pytest.mark.asyncio
async def test_attachment_bind_race_keeps_specific_failure_on_origin_lane(
    db_session,
    tmp_path,
) -> None:
    """A post-prepare bind loss must not disclose sender state to subscribers."""

    owner = _user(db_session, "bind-race-owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    task.runner_id = "bind-race-runner"
    task.run_id = "bind-race-run"
    file_path = tmp_path / "bind-race.txt"
    file_path.write_text("raced attachment")
    db_session.add(
        UploadedFile(
            file_id="bind-race-file",
            user_id=int(owner.id),
            task_id=int(task.id),
            filename="bind-race.txt",
            storage_path=str(file_path),
            mime_type="text/plain",
            file_size=file_path.stat().st_size,
        )
    )
    db_session.commit()

    class RecordingSocket:
        def __init__(self) -> None:
            self.messages: list[dict[str, Any]] = []

        async def send_text(self, payload: str) -> None:
            import json

            self.messages.append(json.loads(payload))

    origin = RecordingSocket()
    subscriber = RecordingSocket()
    connection_manager = websocket_api.ConnectionManager()
    connection_manager.register_connection(origin, int(task.id))  # type: ignore[arg-type]
    connection_manager.register_connection(subscriber, int(task.id))  # type: ignore[arg-type]

    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    agent_manager = MagicMock(get_agent_for_task=AsyncMock(return_value=agent))
    bg_manager = MagicMock()
    bg_manager.try_reserve_resume.return_value = ResumeReservationOutcome.RESERVED
    bg_manager.running_tasks.get.return_value = None

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=agent_manager),
        patch("xagent.web.api.websocket.manager", connection_manager),
        patch("xagent.web.api.websocket.background_task_manager", bg_manager),
        patch(
            "xagent.web.api.websocket.bind_turn_files_no_commit",
            return_value=["bind-race-file"],
        ),
    ):
        await _handle_chat_message_unserialized(
            origin,  # type: ignore[arg-type]
            int(task.id),
            {
                "message": "Use my attachment",
                "client_message_id": "bind-race-turn",
                "user": SimpleNamespace(id=int(owner.id), is_admin=False),
                "files": [
                    {
                        "file_id": "bind-race-file",
                        "name": "bind-race.txt",
                    }
                ],
            },
        )

    assert any(
        event.get("type") == "message_rejected"
        and event.get("error_code") == "message_attachment_unavailable"
        for event in origin.messages
    )
    subscriber_failures = [
        event
        for event in subscriber.messages
        if event.get("type") in {"error", "agent_error", "task_error"}
    ]
    assert subscriber_failures
    assert all(
        event.get("error_code") == "task_execution_failed"
        and event.get("message") == websocket_api.CLIENT_SAFE_TASK_FAILURE
        for event in subscriber_failures
    )


@pytest.mark.asyncio
async def test_resume_registration_failure_keeps_injected_delivery_pending(
    db_session,
) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    task.runner_id = "registration-runner"
    task.run_id = "registration-run"
    db_session.commit()
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    agent.post_user_message = AsyncMock(
        return_value=UserMessageInjectionOutcome.POSTED_FRESH
    )
    mgr = MagicMock(get_agent_for_task=AsyncMock(return_value=agent))
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    bg_mgr = MagicMock()
    bg_mgr.try_reserve_resume.return_value = ResumeReservationOutcome.RESERVED
    bg_mgr.running_tasks.get.return_value = None
    bg_mgr.register_reserved_resume.side_effect = RuntimeError("reservation lost")
    bg_handle = MagicMock()

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.execute_resume_background", AsyncMock()),
        patch(
            "xagent.web.api.websocket.asyncio.create_task",
            return_value=bg_handle,
        ),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
    ):
        await handle_chat_message(
            MagicMock(),
            int(task.id),
            {
                "message": "Apply this safely",
                "client_message_id": "registration-failure-turn",
                "user": owner,
                "files": [],
            },
        )
        for _ in range(100):
            if bg_handle.cancel.called:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("resume registration failure was not handled in time")

    bg_handle.cancel.assert_called_once()
    db_session.expire_all()
    stored = (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.turn_id == "registration-failure-turn")
        .one()
    )
    assert stored.delivery_status == DELIVERY_PENDING
    accepted = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_accepted"
    ]
    assert len(accepted) == 1


@pytest.mark.asyncio
async def test_live_marker_failure_after_registered_handoff_is_still_accepted(
    db_session,
) -> None:
    owner = _user(db_session, "marker-failure-owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    task.runner_id = "marker-failure-runner"
    task.run_id = "marker-failure-run"
    db_session.commit()
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    agent.post_user_message = AsyncMock(
        return_value=UserMessageInjectionOutcome.POSTED_FRESH
    )
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    bg_mgr = MagicMock()
    bg_mgr.try_reserve_resume.return_value = ResumeReservationOutcome.RESERVED
    bg_mgr.running_tasks.get.return_value = None

    with (
        patch(
            "xagent.web.api.chat.get_agent_manager",
            return_value=MagicMock(get_agent_for_task=AsyncMock(return_value=agent)),
        ),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
        patch("xagent.web.api.websocket.execute_resume_background", AsyncMock()),
        patch(
            "xagent.web.api.websocket.mark_user_message_delivery_sync",
            side_effect=RuntimeError("marker unavailable"),
        ),
    ):
        await _handle_chat_message_unserialized(
            MagicMock(),
            int(task.id),
            {
                "message": "Apply this once",
                "client_message_id": "marker-failure-turn",
                "user": owner,
                "files": [],
            },
        )

    bg_mgr.register_reserved_resume.assert_called_once()
    accepted = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_accepted"
    ]
    assert len(accepted) == 1


@pytest.mark.asyncio
async def test_live_resume_reads_the_interaction_row_before_injecting(
    db_session,
) -> None:
    """The close is keyed on the row observed *before* the injection, and
    only the ordering makes that true -- see task_interaction_close's
    module docstring. The read also sits before the ``posted`` fork, so the
    deferred branch carries the same observation instead of taking one of
    its own even later."""

    owner = _user(db_session, "close-order-owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    task.runner_id = "close-order-runner"
    task.run_id = "close-order-run"
    db_session.commit()
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None

    order: list[str] = []

    def record_read(_task_id: int) -> int:
        order.append("read")
        return 4321

    async def record_injection(
        *_args: object, **_kwargs: object
    ) -> UserMessageInjectionOutcome:
        order.append("inject")
        return UserMessageInjectionOutcome.POSTED_FRESH

    agent.post_user_message = AsyncMock(side_effect=record_injection)
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    bg_mgr = MagicMock()
    bg_mgr.try_reserve_resume.return_value = ResumeReservationOutcome.RESERVED
    bg_mgr.running_tasks.get.return_value = None

    with (
        patch(
            "xagent.web.api.chat.get_agent_manager",
            return_value=MagicMock(get_agent_for_task=AsyncMock(return_value=agent)),
        ),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
        patch("xagent.web.api.websocket.execute_resume_background", AsyncMock()),
        patch(
            "xagent.web.api.websocket.active_interaction_id_sync",
            side_effect=record_read,
        ),
        patch(
            "xagent.web.api.websocket.close_legacy_resume_interaction_sync",
            return_value=1,
        ) as close_mock,
    ):
        await _handle_chat_message_unserialized(
            MagicMock(),
            int(task.id),
            {
                "message": "Apply this once",
                "client_message_id": "close-order-turn",
                "user": owner,
                "files": [],
            },
        )

    assert order == ["read", "inject"]
    close_mock.assert_called_once_with(
        task_id=int(task.id), run_id="close-order-run", interaction_id=4321
    )


@pytest.mark.asyncio
async def test_live_injection_skips_the_close_on_a_replayed_turn_id(
    db_session,
) -> None:
    """A replayed turn id short-circuits inside AgentRunner.inject_user_message
    and is reported back as POSTED_REPLAY, not the same truthy value a
    fresh write produces. The interaction row this test seeds is not the
    question the replay answered, so the online injection site must leave
    it untouched: still active, uncleared marker, and the close statement
    must not run at all. That is asserted directly on the mock: replacing
    the guard with `if True:` calls the close for real and turns
    assert_not_called red. No DB-state assertion is made here -- with the
    close function mocked out, nothing writes to the row or the marker, so
    "still active" would hold no matter what the guard did."""
    owner = _user(db_session, "close-replay-owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    task.runner_id = "close-replay-runner"
    task.run_id = "close-replay-run"
    task.interaction_protocol_version = 1
    db_session.commit()
    task_id = int(task.id)
    _seed_active_interaction_row(
        db_session,
        task_id=task_id,
        run_id="close-replay-run",
        idempotency_key="close-replay-q1",
    )

    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    agent.post_user_message = AsyncMock(
        return_value=UserMessageInjectionOutcome.POSTED_REPLAY
    )
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    bg_mgr = MagicMock()
    bg_mgr.try_reserve_resume.return_value = ResumeReservationOutcome.RESERVED
    bg_mgr.running_tasks.get.return_value = None

    with (
        patch(
            "xagent.web.api.chat.get_agent_manager",
            return_value=MagicMock(get_agent_for_task=AsyncMock(return_value=agent)),
        ),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
        patch("xagent.web.api.websocket.execute_resume_background", AsyncMock()),
        patch(
            "xagent.web.api.websocket.close_legacy_resume_interaction_sync",
        ) as close_mock,
    ):
        await _handle_chat_message_unserialized(
            MagicMock(),
            task_id,
            {
                "message": "a retried delivery",
                "client_message_id": "close-replay-turn",
                "user": owner,
                "files": [],
            },
        )

    agent.post_user_message.assert_awaited_once()
    close_mock.assert_not_called()


@pytest.mark.asyncio
async def test_live_close_failure_after_registered_handoff_is_still_accepted(
    db_session,
) -> None:
    """A legacy resume interaction close failure must not turn an already
    registered, already accepted resume handoff into a rejected one."""
    owner = _user(db_session, "close-failure-owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    task.runner_id = "close-failure-runner"
    task.run_id = "close-failure-run"
    db_session.commit()
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    agent.post_user_message = AsyncMock(
        return_value=UserMessageInjectionOutcome.POSTED_FRESH
    )
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    bg_mgr = MagicMock()
    bg_mgr.try_reserve_resume.return_value = ResumeReservationOutcome.RESERVED
    bg_mgr.running_tasks.get.return_value = None

    with (
        patch(
            "xagent.web.api.chat.get_agent_manager",
            return_value=MagicMock(get_agent_for_task=AsyncMock(return_value=agent)),
        ),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
        patch("xagent.web.api.websocket.execute_resume_background", AsyncMock()),
        patch(
            "xagent.web.api.websocket.close_legacy_resume_interaction_sync",
            side_effect=RuntimeError("interaction close unavailable"),
        ) as close_mock,
    ):
        await _handle_chat_message_unserialized(
            MagicMock(),
            int(task.id),
            {
                "message": "Apply this once",
                "client_message_id": "close-failure-turn",
                "user": owner,
                "files": [],
            },
        )

    bg_mgr.register_reserved_resume.assert_called_once()
    close_mock.assert_called_once_with(
        task_id=int(task.id), run_id="close-failure-run", interaction_id=None
    )
    accepted = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_accepted"
    ]
    assert len(accepted) == 1


@pytest.mark.asyncio
async def test_live_close_cancellation_does_not_abort_registered_handoff(
    db_session,
) -> None:
    """Same guarantee as the failure case above, for the CancelledError
    branch specifically -- deleting that branch must turn this red."""
    owner = _user(db_session, "close-cancel-owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    task.runner_id = "close-cancel-runner"
    task.run_id = "close-cancel-run"
    db_session.commit()
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    agent.post_user_message = AsyncMock(
        return_value=UserMessageInjectionOutcome.POSTED_FRESH
    )
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    bg_mgr = MagicMock()
    bg_mgr.try_reserve_resume.return_value = ResumeReservationOutcome.RESERVED
    bg_mgr.running_tasks.get.return_value = None

    def raise_cancelled(*_args: object, **_kwargs: object) -> int:
        raise asyncio.CancelledError

    with (
        patch(
            "xagent.web.api.chat.get_agent_manager",
            return_value=MagicMock(get_agent_for_task=AsyncMock(return_value=agent)),
        ),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
        patch("xagent.web.api.websocket.execute_resume_background", AsyncMock()),
        patch(
            "xagent.web.api.websocket.close_legacy_resume_interaction_sync",
            side_effect=raise_cancelled,
        ) as close_mock,
    ):
        await _handle_chat_message_unserialized(
            MagicMock(),
            int(task.id),
            {
                "message": "Apply this once",
                "client_message_id": "close-cancel-turn",
                "user": owner,
                "files": [],
            },
        )

    bg_mgr.register_reserved_resume.assert_called_once()
    close_mock.assert_called_once_with(
        task_id=int(task.id), run_id="close-cancel-run", interaction_id=None
    )
    accepted = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_accepted"
    ]
    assert len(accepted) == 1


def test_lease_restore_fence_miss_leaves_the_marker_untouched(db_session) -> None:
    """When the exact-lease fence excludes every row (this call lost the
    race for the row), the marker clear must not run at all -- not even
    the no-op form. Flipping the ``if restored:`` guard to unconditional
    would clear a marker this call never had authority to touch.
    """
    owner = _user(db_session, "restore-fence-miss-owner")
    task = _task(db_session, owner.id, status=TaskStatus.WAITING_FOR_USER)
    task.runner_id = "current-runner"
    task.run_id = "run-fence-miss"
    task.interaction_protocol_version = 1
    db_session.commit()

    stale_lease = TaskLease(
        task_id=int(task.id), runner_id="a-different-runner", run_id="run-fence-miss"
    )
    restored = _restore_resumed_task_lease_to_prior_status(
        stale_lease, status=TaskStatus.WAITING_FOR_USER
    )

    assert restored is False
    db_session.expire_all()
    refreshed = db_session.query(Task).filter(Task.id == task.id).one()
    assert refreshed.interaction_protocol_version == 1


def test_lease_restore_clears_the_marker_once_no_active_row_remains(
    db_session,
) -> None:
    """The mirror case: the fence matches (this call owns the row) and no
    active interaction row remains, so the marker is stale and gets
    zeroed."""
    owner = _user(db_session, "restore-clears-marker-owner")
    task = _task(db_session, owner.id, status=TaskStatus.WAITING_FOR_USER)
    task.runner_id = "current-runner"
    task.run_id = "run-clears-marker"
    task.interaction_protocol_version = 1
    db_session.commit()

    lease = TaskLease(
        task_id=int(task.id), runner_id="current-runner", run_id="run-clears-marker"
    )
    restored = _restore_resumed_task_lease_to_prior_status(
        lease, status=TaskStatus.WAITING_FOR_USER
    )

    assert restored is True
    db_session.expire_all()
    refreshed = db_session.query(Task).filter(Task.id == task.id).one()
    assert refreshed.interaction_protocol_version is None


def test_live_claim_unique_loser_returns_the_committed_winner(
    db_session,
) -> None:
    owner = _user(db_session, "live-unique-loser-owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(owner.id),
            role="user",
            content="Apply exactly once",
            message_type="user_message",
            turn_id="live-unique-loser-turn",
            delivery_status=DELIVERY_PENDING,
        )
    )
    db_session.commit()

    with patch(
        "xagent.web.api.websocket.claim_user_message_delivery_no_commit",
        side_effect=IntegrityError(
            "INSERT task_chat_messages",
            {},
            RuntimeError("unique constraint"),
        ),
    ):
        claim = _claim_user_message_delivery_isolated(
            task_id=int(task.id),
            task_owner_user_id=int(owner.id),
            content="Apply exactly once",
            attachments=None,
            file_ids=[],
            turn_id="live-unique-loser-turn",
            expected_run_id=None,
            expected_status=TaskStatus.RUNNING,
        )

    assert claim.claimed is False
    assert claim.payload_matches is True
    assert claim.pending is True
    assert claim.failed is False


def test_live_claim_unique_loser_returns_conflicting_winner_with_files(
    db_session,
) -> None:
    owner = _user(db_session, "live-conflicting-unique-loser-owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(owner.id),
            role="user",
            content="Winner payload",
            message_type="user_message",
            turn_id="live-conflicting-unique-loser-turn",
            delivery_status=DELIVERY_PENDING,
        )
    )
    db_session.commit()

    with patch(
        "xagent.web.api.websocket.claim_user_message_delivery_no_commit",
        side_effect=IntegrityError(
            "INSERT task_chat_messages",
            {},
            RuntimeError("unique constraint"),
        ),
    ):
        claim = _claim_user_message_delivery_isolated(
            task_id=int(task.id),
            task_owner_user_id=int(owner.id),
            content="Conflicting loser payload",
            attachments=[{"file_id": "loser-file"}],
            file_ids=["loser-file"],
            turn_id="live-conflicting-unique-loser-turn",
            expected_run_id=None,
            expected_status=TaskStatus.RUNNING,
        )

    assert claim.claimed is False
    assert claim.payload_matches is False
    assert claim.pending is True
    assert claim.failed is False


@pytest.mark.asyncio
async def test_message_handoff_registers_the_minted_run_not_the_stale_one(
    db_session,
) -> None:
    """A legacy NULL-run_id row gets a fresh run at the transition.

    ``apply_task_control_transition`` mints a run id when the row has none,
    so the value read from the routing snapshot before that call is ``None``
    while the task now runs under a uuid. Registering the coordinator under
    the stale value makes a later RESUME asking about the real run read a
    live resume as ``RESERVATION_HELD``, defer sixty times, and terminally
    fail a resume that was already succeeding.
    """

    owner = _user(db_session, "minted-run-owner")
    # WAITING_FOR_USER, because that is what routes a chat message into the
    # live-control resume handoff (``_task_status_uses_live_control``).
    task = _task(db_session, owner.id, status=TaskStatus.WAITING_FOR_USER)
    task.run_id = None
    task.control_state = "waiting_for_user"
    db_session.commit()

    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    # No exact live lease, so the handoff defers the message to the resume
    # owner instead of injecting it -- the path that reaches the transition.
    agent.post_user_message = AsyncMock(return_value=False)
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    bg_mgr = MagicMock()
    bg_mgr.try_reserve_resume.return_value = ResumeReservationOutcome.RESERVED
    bg_mgr.running_tasks.get.return_value = None
    registered_run_ids: list[str | None] = []
    registered_handles: list[asyncio.Task] = []

    def register_resume(
        _task_id: int,
        task_handle: asyncio.Task,
        *,
        run_id: str | None = None,
    ) -> None:
        registered_run_ids.append(run_id)
        registered_handles.append(task_handle)

    async def resume_forever(*_args, **_kwargs) -> None:
        await asyncio.Event().wait()

    bg_mgr.register_reserved_resume.side_effect = register_resume
    with (
        patch(
            "xagent.web.api.chat.get_agent_manager",
            return_value=MagicMock(get_agent_for_task=AsyncMock(return_value=agent)),
        ),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
        patch(
            "xagent.web.api.websocket.execute_resume_background",
            side_effect=resume_forever,
        ) as resume_bg,
    ):
        await _handle_chat_message_unserialized(
            MagicMock(),
            int(task.id),
            {
                "message": "resume a legacy row",
                "client_message_id": "minted-run-turn",
                "user": owner,
                "files": [],
            },
        )
        # The coroutine is recorded at construction, before the loop runs it.
        resume_expected_run_id = resume_bg.call_args.kwargs["expected_run_id"]
        for handle in registered_handles:
            handle.cancel()
        await asyncio.gather(*registered_handles, return_exceptions=True)

    db_session.expire_all()
    stored = db_session.get(Task, int(task.id))
    assert stored is not None
    minted = stored.run_id
    assert minted is not None, "the transition should have minted a run id"

    assert registered_run_ids == [minted], (
        "the coordinator must be registered under the run the task is "
        "actually executing, not the NULL the snapshot carried"
    )
    # And the execution has to agree: a None here would reach
    # acquire_task_lease_no_commit and mint a *second* run, putting the row,
    # the registration and the lease on three different answers.
    assert resume_expected_run_id == minted


@pytest.mark.asyncio
async def test_live_marker_cancellation_does_not_cancel_registered_handoff(
    db_session,
) -> None:
    owner = _user(db_session, "marker-cancellation-owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    task.runner_id = "marker-cancellation-runner"
    task.run_id = "marker-cancellation-run"
    db_session.commit()
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    agent.post_user_message = AsyncMock(
        return_value=UserMessageInjectionOutcome.POSTED_FRESH
    )
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    bg_mgr = MagicMock()
    bg_mgr.try_reserve_resume.return_value = ResumeReservationOutcome.RESERVED
    bg_mgr.running_tasks.get.return_value = None
    marker_started = threading.Event()
    marker_release = threading.Event()
    registered_tasks: list[asyncio.Task] = []
    background_cancelled = asyncio.Event()

    def mark_delivery(*_args, **_kwargs) -> None:
        marker_started.set()
        assert marker_release.wait(timeout=_THREAD_SIGNAL_DEADLINE_SECONDS)

    async def resume_forever(*_args, **_kwargs) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            background_cancelled.set()
            raise

    def register_resume(
        _task_id: int,
        task_handle: asyncio.Task,
        *,
        run_id: str | None = None,
    ) -> None:
        assert run_id == "marker-cancellation-run"
        registered_tasks.append(task_handle)

    bg_mgr.register_reserved_resume.side_effect = register_resume
    with (
        patch(
            "xagent.web.api.chat.get_agent_manager",
            return_value=MagicMock(get_agent_for_task=AsyncMock(return_value=agent)),
        ),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
        patch(
            "xagent.web.api.websocket.execute_resume_background",
            side_effect=resume_forever,
        ),
        patch(
            "xagent.web.api.websocket.mark_user_message_delivery_sync",
            side_effect=mark_delivery,
        ),
    ):
        handling = asyncio.get_running_loop().create_task(
            _handle_chat_message_unserialized(
                MagicMock(),
                int(task.id),
                {
                    "message": "Apply despite disconnect",
                    "client_message_id": "marker-cancellation-turn",
                    "user": owner,
                    "files": [],
                },
            )
        )
        assert await asyncio.to_thread(
            marker_started.wait, _THREAD_SIGNAL_DEADLINE_SECONDS
        )
        handling.cancel()
        marker_release.set()
        with pytest.raises(asyncio.CancelledError):
            await handling

    assert len(registered_tasks) == 1
    assert registered_tasks[0].cancelled() is False
    assert background_cancelled.is_set() is False
    bg_mgr.release_resume_reservation.assert_not_called()
    registered_tasks[0].cancel()
    with pytest.raises(asyncio.CancelledError):
        await registered_tasks[0]


def test_live_claim_reconciles_a_commit_acknowledgement_failure(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _user(db_session, "live-ack-owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    original_commit = Session.commit
    commit_raised = False

    def acknowledge_then_disconnect(session: Session) -> None:
        nonlocal commit_raised
        original_commit(session)
        if session is not db_session and not commit_raised:
            commit_raised = True
            raise ConnectionError("commit acknowledgement lost")

    monkeypatch.setattr(Session, "commit", acknowledge_then_disconnect)

    claim = _claim_user_message_delivery_isolated(
        task_id=int(task.id),
        task_owner_user_id=int(owner.id),
        content="Apply safely",
        attachments=None,
        file_ids=[],
        turn_id="live-ack-turn",
        expected_run_id=None,
        expected_status=TaskStatus.RUNNING,
    )

    assert commit_raised is True
    assert claim.claimed is True
    db_session.expire_all()
    assert (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.turn_id == "live-ack-turn")
        .one()
        .delivery_status
        == DELIVERY_PENDING
    )


def test_waiting_for_user_claim_reconciles_a_commit_acknowledgement_failure(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _user(db_session, "waiting-ack-owner")
    task = _task(db_session, owner.id, status=TaskStatus.WAITING_FOR_USER)
    task.run_id = "waiting-ack-run"
    db_session.commit()
    original_commit = Session.commit
    commit_raised = False

    def acknowledge_then_disconnect(session: Session) -> None:
        nonlocal commit_raised
        original_commit(session)
        if session is not db_session and not commit_raised:
            commit_raised = True
            raise ConnectionError("commit acknowledgement lost")

    monkeypatch.setattr(Session, "commit", acknowledge_then_disconnect)

    claim = _claim_user_message_delivery_isolated(
        task_id=int(task.id),
        task_owner_user_id=int(owner.id),
        content="Answer the pending question",
        attachments=None,
        file_ids=[],
        turn_id="waiting-ack-turn",
        expected_run_id="waiting-ack-run",
        expected_status=TaskStatus.WAITING_FOR_USER,
    )

    assert commit_raised is True
    assert claim.claimed is True
    db_session.expire_all()
    stored = (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.turn_id == "waiting-ack-turn")
        .one()
    )
    assert stored.delivery_status == DELIVERY_PENDING


def test_websocket_commit_reconciliation_read_failure_stays_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = MagicMock()
    session.query.side_effect = RuntimeError("reconciliation database unavailable")
    monkeypatch.setattr(
        websocket_api,
        "get_session_local",
        lambda: lambda: session,
    )
    monkeypatch.setattr(websocket_api.time, "sleep", MagicMock())

    assert (
        websocket_api._reconcile_websocket_acceptance_graph(
            task_id=123,
            task_owner_user_id=456,
            turn_id="read-failure-turn",
            content="Accepted",
            file_ids=[],
            expected_run_id="run",
            expected_status=TaskStatus.RUNNING,
        )
        is False
    )
    assert session.query.call_count == 3


def test_websocket_commit_reconciliation_rejects_failed_delivery(
    db_session,
) -> None:
    owner = _user(db_session, "failed-reconciliation-owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    task.run_id = "failed-reconciliation-run"
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(owner.id),
            role="user",
            content="Accepted",
            message_type="user_message",
            turn_id="failed-reconciliation-turn",
            delivery_status=DELIVERY_FAILED,
        )
    )
    db_session.commit()

    assert (
        websocket_api._reconcile_websocket_acceptance_graph(
            task_id=int(task.id),
            task_owner_user_id=int(owner.id),
            turn_id="failed-reconciliation-turn",
            content="Accepted",
            file_ids=[],
            expected_run_id="failed-reconciliation-run",
            expected_status=TaskStatus.RUNNING,
        )
        is False
    )


def test_missing_task_prepare_reconciles_a_commit_acknowledgement_failure(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _user(db_session, "missing-ack-owner")
    original_commit = Session.commit
    commit_raised = False

    def acknowledge_then_disconnect(session: Session) -> None:
        nonlocal commit_raised
        original_commit(session)
        if session is not db_session and not commit_raised:
            commit_raised = True
            raise ConnectionError("commit acknowledgement lost")

    monkeypatch.setattr(Session, "commit", acknowledge_then_disconnect)

    preparation = websocket_api._prepare_websocket_turn_sync(
        requested_task_id=987654,
        actor_user_id=int(owner.id),
        actor_is_admin=False,
        user_message="start after acknowledgement loss",
        raw_context={},
        raw_files=[],
        client_message_id="missing-ack-turn",
        turn_id="missing-ack-turn",
        durable_attempt_count=1,
        durable_target_run_id=None,
        pause_accepted=False,
    )

    assert commit_raised is True
    assert preparation.task_created is True
    db_session.expire_all()
    assert (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.turn_id == "missing-ack-turn")
        .one()
        .delivery_status
        == DELIVERY_PENDING
    )


def test_missing_task_prepare_keeps_an_absent_commit_outcome_unknown(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _user(db_session, "missing-unknown-owner")

    def lose_commit_before_acknowledgement(_session: Session) -> None:
        raise ConnectionError("commit acknowledgement lost")

    monkeypatch.setattr(Session, "commit", lose_commit_before_acknowledgement)

    with pytest.raises(websocket_api._WebSocketCommitOutcomeUnknown):
        websocket_api._prepare_websocket_turn_sync(
            requested_task_id=987655,
            actor_user_id=int(owner.id),
            actor_is_admin=False,
            user_message="unknown acceptance",
            raw_context={},
            raw_files=[],
            client_message_id="missing-unknown-turn",
            turn_id="missing-unknown-turn",
            durable_attempt_count=1,
            durable_target_run_id=None,
            pause_accepted=False,
        )


@pytest.mark.asyncio
async def test_live_control_delivery_failure_pool_timeout_is_not_retried(
    db_session,
) -> None:
    """One failed DELIVERY_FAILED checkout still produces one rejection ack."""
    owner = _user(db_session, "pool-timeout-owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    task.runner_id = "pool-timeout-runner"
    task.run_id = "pool-timeout-run"
    db_session.commit()
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    agent.post_user_message = AsyncMock(side_effect=RuntimeError("inject failed"))
    mgr = MagicMock(get_agent_for_task=AsyncMock(return_value=agent))
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    bg_mgr = MagicMock()
    bg_mgr.try_reserve_resume.return_value = ResumeReservationOutcome.RESERVED
    mark_delivery = MagicMock(
        side_effect=SQLAlchemyTimeoutError("delivery pool exhausted")
    )
    error_payload_reader = MagicMock()

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
        patch(
            "xagent.web.api.websocket.mark_user_message_delivery_sync",
            mark_delivery,
        ),
        patch(
            "xagent.web.api.websocket._read_task_error_payload_isolated",
            error_payload_reader,
        ),
    ):
        await _handle_chat_message_unserialized(
            MagicMock(),
            int(task.id),
            {
                "message": "apply once",
                "client_message_id": "live-control-pool-timeout",
                "user": owner,
                "files": [],
            },
        )

    mark_delivery.assert_called_once_with(
        int(task.id), "live-control-pool-timeout", DELIVERY_FAILED
    )
    error_payload_reader.assert_not_called()
    rejected = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_rejected"
    ]
    assert len(rejected) == 1
    assert rejected[0]["client_message_id"] == "live-control-pool-timeout"
    assert "inject failed" not in repr(rejected[0])
    assert rejected[0]["message"] == websocket_api.CLIENT_SAFE_VALIDATION_ERROR
    assert rejected[0]["error_code"] == "message_processing_failed"
    assert rejected[0]["rejection_outcome"] == "outcome_unknown"


@pytest.mark.asyncio
async def test_delivery_failure_persistence_drains_before_cancellation(
    db_session,
) -> None:
    owner = _user(db_session, "delivery-cancellation-owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    task.runner_id = "delivery-cancellation-runner"
    task.run_id = "delivery-cancellation-run"
    db_session.commit()
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    agent.post_user_message = AsyncMock(side_effect=RuntimeError("inject failed"))
    agent_manager = MagicMock(get_agent_for_task=AsyncMock(return_value=agent))
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    bg_manager = MagicMock()
    bg_manager.try_reserve_resume.return_value = ResumeReservationOutcome.RESERVED
    persistence_started = threading.Event()
    allow_persistence = threading.Event()
    persistence_finished = threading.Event()

    def blocking_mark_delivery(*_args, **_kwargs) -> None:
        persistence_started.set()
        assert allow_persistence.wait(timeout=_THREAD_SIGNAL_DEADLINE_SECONDS)
        persistence_finished.set()

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=agent_manager),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager", bg_manager),
        patch(
            "xagent.web.api.websocket.mark_user_message_delivery_sync",
            side_effect=blocking_mark_delivery,
        ),
    ):
        handling = asyncio.create_task(
            _handle_chat_message_unserialized(
                MagicMock(),
                int(task.id),
                {
                    "message": "apply once",
                    "client_message_id": "delivery-cancellation-turn",
                    "user": owner,
                    "files": [],
                },
            )
        )
        # Without the assert a slow runner silently proceeds to cancel before
        # persistence started, which changes what the rest of the test proves.
        assert await asyncio.to_thread(
            persistence_started.wait, _THREAD_SIGNAL_DEADLINE_SECONDS
        )
        handling.cancel()
        await asyncio.sleep(0)
        assert not handling.done()

        allow_persistence.set()
        with pytest.raises(asyncio.CancelledError):
            await handling

    assert persistence_finished.is_set()


@pytest.mark.asyncio
async def test_retried_durable_message_is_accepted_without_reexecution(
    db_session,
) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.COMPLETED)
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(owner.id),
            role="user",
            content="Already delivered",
            message_type="user_message",
            turn_id="stable-turn-1",
        )
    )
    db_session.commit()
    agent_manager = MagicMock()
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=agent_manager),
        patch("xagent.web.api.websocket.manager", ws_manager),
    ):
        await handle_chat_message(
            MagicMock(),
            int(task.id),
            {
                "message": "Already delivered",
                "client_message_id": "stable-turn-1",
                "user": owner,
                "files": [],
            },
        )

    agent_manager.get_agent_for_task.assert_not_called()
    stored = (
        db_session.query(TaskChatMessage)
        .filter(
            TaskChatMessage.task_id == int(task.id),
            TaskChatMessage.turn_id == "stable-turn-1",
        )
        .all()
    )
    assert len(stored) == 1
    accepted = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_accepted"
    ]
    assert len(accepted) == 1


@pytest.mark.asyncio
async def test_reusing_client_id_with_different_content_is_rejected(
    db_session,
) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.COMPLETED)
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(owner.id),
            role="user",
            content="Original content",
            message_type="user_message",
            turn_id="stable-turn-1",
        )
    )
    db_session.commit()
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    begin_turn = AsyncMock()

    with (
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch(
            "xagent.web.services.task_orchestrator.TaskTurnOrchestrator.begin_turn",
            new=begin_turn,
        ),
    ):
        await handle_chat_message(
            MagicMock(),
            int(task.id),
            {
                "message": "Different content",
                "client_message_id": "stable-turn-1",
                "user": owner,
                "files": [],
            },
        )

    begin_turn.assert_not_awaited()
    rejected = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_rejected"
    ]
    assert len(rejected) == 1
    assert rejected[0]["error_code"] == "message_id_conflict"
    assert rejected[0]["retry_with_new_id"] is True
    assert rejected[0]["rejection_outcome"] == "not_accepted"


@pytest.mark.asyncio
async def test_failed_durable_delivery_is_not_silently_accepted(db_session) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.FAILED)
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(owner.id),
            role="user",
            content="Retry after checkpoint failure",
            message_type="user_message",
            turn_id="failed-turn-1",
            delivery_status=DELIVERY_FAILED,
        )
    )
    db_session.commit()
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    begin_turn = AsyncMock()

    with (
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch(
            "xagent.web.services.task_orchestrator.TaskTurnOrchestrator.begin_turn",
            new=begin_turn,
        ),
    ):
        await handle_chat_message(
            MagicMock(),
            int(task.id),
            {
                "message": "Retry after checkpoint failure",
                "client_message_id": "failed-turn-1",
                "user": owner,
                "files": [],
            },
        )

    begin_turn.assert_not_awaited()
    rejected = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_rejected"
    ]
    assert len(rejected) == 1
    assert rejected[0]["error_code"] == "message_delivery_failed"
    assert rejected[0]["retry_with_new_id"] is True
    assert rejected[0]["rejection_outcome"] == "not_accepted"


@pytest.mark.asyncio
async def test_pending_same_id_delivery_reports_unknown_outcome(db_session) -> None:
    owner = _user(db_session, "pending-owner")
    task = _task(db_session, owner.id, status=TaskStatus.COMPLETED)
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(owner.id),
            role="user",
            content="Pending guidance",
            message_type="user_message",
            turn_id="pending-turn-1",
            delivery_status=DELIVERY_PENDING,
        )
    )
    db_session.commit()
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )

    with patch("xagent.web.api.websocket.manager", ws_manager):
        await _handle_chat_message_unserialized(
            MagicMock(),
            int(task.id),
            {
                "message": "Pending guidance",
                "client_message_id": "pending-turn-1",
                "user": owner,
                "files": [],
            },
        )

    rejected = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_rejected"
    ]
    assert len(rejected) == 1
    assert rejected[0]["client_message_id"] == "pending-turn-1"
    assert rejected[0]["error_code"] == "guidance_in_progress"
    assert rejected[0]["rejection_outcome"] == "outcome_unknown"


@pytest.mark.asyncio
async def test_pause_admin_on_other_users_task_runs_as_owner(db_session) -> None:
    owner = _user(db_session, "owner")
    admin = _user(db_session, "admin", is_admin=True)
    task = _task(db_session, owner.id)
    captured, agent, mgr, ws_manager = _patched_manager_and_agent()

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
    ):
        await handle_pause_task(MagicMock(), int(task.id), {"user": admin})
        for _ in range(100):
            if "task_owner_user_id" in captured:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("durable pause command was not dispatched in time")

    # Built and paused as the OWNER, not the admin actor.
    assert captured["task_owner_user_id"] == int(owner.id)
    agent.pause_execution.assert_awaited_once()


@pytest.mark.asyncio
async def test_durable_pause_propagates_stale_run_error(db_session) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id)
    _captured, _agent, mgr, ws_manager = _patched_manager_and_agent()

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch(
            "xagent.web.api.websocket._apply_pause_requested_isolated",
            side_effect=StaleTaskRunError("run rotated"),
        ),
        pytest.raises(StaleTaskRunError, match="run rotated"),
    ):
        await _handle_pause_task_unserialized(
            MagicMock(),
            int(task.id),
            {"user": owner, "_durable_ack_sent": True},
        )


@pytest.mark.asyncio
async def test_durable_resume_propagates_stale_run_error(db_session) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.PAUSED)
    _captured, agent, mgr, ws_manager = _patched_manager_and_agent()
    agent.supports_live_control.return_value = True
    bg_mgr = MagicMock()
    bg_mgr.resume_admission_state.return_value = None
    bg_mgr.try_reserve_resume.return_value = ResumeReservationOutcome.RESERVED

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
        patch(
            "xagent.web.api.websocket.task_execution_controller.transition",
            new=AsyncMock(side_effect=StaleTaskRunError("run rotated")),
        ),
        pytest.raises(StaleTaskRunError, match="run rotated"),
    ):
        await _handle_resume_task_unserialized(
            MagicMock(),
            int(task.id),
            {"user": owner, "_durable_ack_sent": True},
        )
    bg_mgr.release_resume_reservation.assert_called_once_with(int(task.id))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "handler_name"),
    [
        (TaskCommandKind.PAUSE, "_handle_pause_task_unserialized"),
        (TaskCommandKind.RESUME, "_handle_resume_task_unserialized"),
    ],
)
async def test_durable_control_converts_handler_stale_run_to_terminal_rejection(
    db_session,
    kind: TaskCommandKind,
    handler_name: str,
) -> None:
    owner = _user(db_session, f"durable-{kind.value}-stale-owner")
    task = _task(db_session, owner.id)
    task.runner_id = None
    task.run_id = "run-a"
    db_session.commit()
    command = ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=int(owner.id),
        command_id=f"{kind.value}-stale-handler",
        kind=kind,
        payload={"type": f"{kind.value}_task"},
        target_run_id="run-a",
        attempt_count=1,
    )

    with (
        patch.object(websocket_api.manager, "connections_for_task", return_value=[]),
        patch(
            f"xagent.web.api.websocket.{handler_name}",
            new=AsyncMock(side_effect=StaleTaskRunError("run rotated")),
        ),
        pytest.raises(TaskCommandRejected) as exc_info,
    ):
        await _execute_durable_task_command(command)

    assert exc_info.value.reason == "stale_run"
    assert "run rotated" in str(exc_info.value)


def _durable_pause_command(task: Task, owner: User) -> ClaimedTaskCommand:
    return ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=int(owner.id),
        command_id="pause-personal-reply-routing",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
        target_run_id="run-a",
        attempt_count=1,
    )


@pytest.mark.asyncio
async def test_durable_command_targets_the_registered_origin_not_list_order(
    db_session,
) -> None:
    """Origin is never inferred from connection order (#1514 round 6).

    This test previously pinned the opposite: "the first real connection is
    picked". That guess could hand a personal reply - including raw error
    text - to whichever socket happened to be first, e.g. an anonymous
    public visitor. A durable command now targets the socket registered as
    its origin at enqueue; without a registration it gets the discarding
    sink even when real connections exist."""
    owner = _user(db_session, "broadcast-skip-owner")
    task = _task(db_session, owner.id)
    task.runner_id = None
    task.run_id = "run-a"
    db_session.commit()
    command = _durable_pause_command(task, owner)

    first_real = SimpleNamespace()  # would have been picked by the old guess
    origin = SimpleNamespace()

    websocket_api._command_origins.register(command.command_id, origin, int(task.id))
    try:
        with (
            patch.object(
                websocket_api.manager,
                "is_connection_registered",
                return_value=True,
            ),
            patch(
                "xagent.web.api.websocket._handle_pause_task_unserialized",
                new=AsyncMock(return_value=None),
            ) as handler,
        ):
            await _execute_durable_task_command(command)
        assert handler.await_args.args[0] is origin

        # Without a registration, real connections in the list are ignored.
        websocket_api._command_origins.discard_command(command.command_id, int(task.id))
        with (
            patch.object(
                websocket_api.manager,
                "connections_for_task",
                return_value=[first_real],
            ),
            patch(
                "xagent.web.api.websocket._handle_pause_task_unserialized",
                new=AsyncMock(return_value=None),
            ) as handler,
        ):
            await _execute_durable_task_command(command)
        assert isinstance(
            handler.await_args.args[0], websocket_api._DiscardingCommandWebSocket
        )
    finally:
        websocket_api._command_origins.discard_command(command.command_id, int(task.id))


@pytest.mark.asyncio
async def test_durable_command_falls_back_to_discarding_websocket_when_all_connections_are_broadcast_only(
    db_session,
) -> None:
    """When every registered connection for the task is broadcast-only
    (e.g. only v1 SSE streams are attached, no real WebSocket), there is
    no valid personal-reply target, so the handler gets the discarding
    sink -- same as when the list is empty."""
    owner = _user(db_session, "broadcast-only-fallback-owner")
    task = _task(db_session, owner.id)
    task.runner_id = None
    task.run_id = "run-a"
    db_session.commit()
    command = _durable_pause_command(task, owner)

    broadcast_only = SimpleNamespace(is_broadcast_only=True)

    with (
        patch.object(
            websocket_api.manager,
            "connections_for_task",
            return_value=[broadcast_only],
        ),
        patch(
            "xagent.web.api.websocket._handle_pause_task_unserialized",
            new=AsyncMock(return_value=None),
        ) as handler,
    ):
        await _execute_durable_task_command(command)

    assert isinstance(
        handler.await_args.args[0], websocket_api._DiscardingCommandWebSocket
    )


@pytest.mark.asyncio
async def test_durable_command_falls_back_to_discarding_websocket_when_no_connections(
    db_session,
) -> None:
    """An empty connection list still routes to the discarding sink,
    unchanged by the broadcast-only filter."""
    owner = _user(db_session, "no-connections-fallback-owner")
    task = _task(db_session, owner.id)
    task.runner_id = None
    task.run_id = "run-a"
    db_session.commit()
    command = _durable_pause_command(task, owner)

    with (
        patch.object(websocket_api.manager, "connections_for_task", return_value=[]),
        patch(
            "xagent.web.api.websocket._handle_pause_task_unserialized",
            new=AsyncMock(return_value=None),
        ) as handler,
    ):
        await _execute_durable_task_command(command)

    assert isinstance(
        handler.await_args.args[0], websocket_api._DiscardingCommandWebSocket
    )


@pytest.mark.asyncio
async def test_pause_non_owner_non_admin_is_refused(db_session) -> None:
    owner = _user(db_session, "owner")
    stranger = _user(db_session, "stranger")  # not admin, not owner
    task = _task(db_session, owner.id)
    captured, agent, mgr, ws_manager = _patched_manager_and_agent()

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
    ):
        # The handler authorizes the task away and handles the denial
        # internally; the point is that no owner runtime is built / paused.
        await handle_pause_task(MagicMock(), int(task.id), {"user": stranger})

    assert "task_owner_user_id" not in captured
    agent.pause_execution.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_admin_on_other_users_task_runs_as_owner(db_session) -> None:
    owner = _user(db_session, "owner")
    admin = _user(db_session, "admin", is_admin=True)
    task = _task(db_session, owner.id, status=TaskStatus.PAUSED)
    captured, agent, mgr, ws_manager = _patched_manager_and_agent()

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
    ):
        await handle_resume_task(MagicMock(), int(task.id), {"user": admin})
        # ``dispatch_task_command_promptly`` may detach the durable resume
        # command after its 50ms deadline, so the captured agent build can
        # land after handle_resume_task returns (same pattern as the pause
        # test above).
        for _ in range(100):
            if "task_owner_user_id" in captured:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("durable resume command was not dispatched in time")

        # Command dispatch may detach after its short prompt deadline. Wait for
        # the worker to reach agent construction instead of racing that
        # documented durable-dispatch boundary.
        for _ in range(100):
            if "task_owner_user_id" in captured:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("resume command did not reach agent construction")

    assert captured["task_owner_user_id"] == int(owner.id)


@pytest.mark.asyncio
async def test_running_resume_completes_as_explicit_idempotent_success(
    db_session,
) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    task.run_id = "run-current"
    task.state_version = 7
    task.control_state = "running"
    task.runner_id = get_runner_id()
    task.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=1)
    db_session.commit()
    _captured, agent, mgr, ws_manager = _patched_manager_and_agent()
    agent.supports_live_control = MagicMock(return_value=True)

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
    ):
        await handle_resume_task(MagicMock(), int(task.id), {"user": owner})

        command = None
        for _ in range(100):
            db_session.expire_all()
            command = (
                db_session.query(TaskExecutionCommand)
                .filter(
                    TaskExecutionCommand.task_id == int(task.id),
                    TaskExecutionCommand.kind == TaskCommandKind.RESUME.value,
                )
                .one_or_none()
            )
            if command is not None and command.status == COMMAND_COMPLETED:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("idempotent resume did not complete in time")

    assert command is not None
    assert command.result["resume_outcome"] == "already_in_progress"

    # The durable row records an idempotent success, but the client that
    # asked still gets the task's state tuple: the resume control only
    # renders while the client believes the task is paused, so a silent
    # completion would leave a stale client clicking resume forever. This is
    # the payload shape a resume correction has always carried.
    payload = None
    for _ in range(100):
        for call in ws_manager.send_personal_message.await_args_list:
            if call.args and "task" in call.args[0]:
                payload = call.args[0]
                break
        if payload is not None:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("idempotent resume correction did not arrive in time")

    # The handler deliberately supplies no state tuple: its own setup
    # snapshot can be stale by the time this branch fires, and
    # ``send_personal_message`` attaches the live row precisely when the
    # producer supplied none. The attachment is pinned by
    # ``test_idempotent_resume_frame_is_enriched_with_the_live_row``.
    assert payload["task"] == {"id": int(task.id)}


@pytest.mark.asyncio
async def test_resume_live_control_admin_runs_background_as_owner(db_session) -> None:
    """Live-control resume schedules ``execute_resume_background``; when an
    admin resumes another user's task it must run with the OWNER's
    UserContext, i.e. ``task_owner_user_id`` is the owner, not the admin."""
    owner = _user(db_session, "owner")
    admin = _user(db_session, "admin", is_admin=True)
    task = _task(db_session, owner.id, status=TaskStatus.PAUSED)
    captured, agent, mgr, ws_manager = _patched_manager_and_agent()
    agent.supports_live_control = MagicMock(return_value=True)

    resume_bg = AsyncMock()
    transition = AsyncMock(
        return_value=SimpleNamespace(
            run_id="run-from-resume-transition",
            status=TaskStatus.PAUSED,
        )
    )
    bg_mgr = MagicMock()
    bg_mgr.running_tasks.get = MagicMock(return_value=None)
    bg_mgr.resume_admission_state.return_value = None
    bg_mgr.try_reserve_resume.return_value = ResumeReservationOutcome.RESERVED

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.execute_resume_background", resume_bg),
        patch(
            "xagent.web.api.websocket.task_execution_controller.transition",
            new=transition,
        ),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
    ):
        await handle_resume_task(MagicMock(), int(task.id), {"user": admin})
        # ``dispatch_task_command_promptly`` may detach the durable resume
        # command after its 50ms deadline, so the captured agent build and
        # the background-resume scheduling can land after handle_resume_task
        # returns (same pattern as the pause test above).
        for _ in range(100):
            if "task_owner_user_id" in captured and resume_bg.call_count:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("durable resume command was not dispatched in time")

        # Durable dispatch intentionally detaches after its short prompt
        # deadline. Keep the patched runtime owners in place until the command
        # reaches agent construction instead of racing that boundary under
        # parallel test load.
        for _ in range(100):
            if "task_owner_user_id" in captured:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("resume command did not reach agent construction")

    # Agent built as owner, and the background resume runs as owner.
    assert captured["task_owner_user_id"] == int(owner.id)
    resume_bg.assert_called_once()
    assert resume_bg.call_args.kwargs["task_owner_user_id"] == int(owner.id)
    assert resume_bg.call_args.kwargs["expected_run_id"] == "run-from-resume-transition"
    bg_mgr.try_reserve_resume.assert_called_once_with(
        int(task.id),
        expected_run_id=None,
    )
    bg_mgr.register_reserved_resume.assert_called_once()


@pytest.mark.asyncio
async def test_resume_registration_failure_cancels_coordinator(db_session) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.PAUSED)
    captured, agent, mgr, ws_manager = _patched_manager_and_agent()
    agent.supports_live_control = MagicMock(return_value=True)
    bg_mgr = MagicMock()
    bg_mgr.resume_admission_state.return_value = None
    bg_mgr.try_reserve_resume.return_value = ResumeReservationOutcome.RESERVED
    bg_mgr.running_tasks.get.return_value = None
    bg_mgr.register_reserved_resume.side_effect = RuntimeError("reservation lost")
    bg_handle = MagicMock()

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.execute_resume_background", MagicMock()),
        patch(
            "xagent.web.api.websocket.asyncio.create_task",
            return_value=bg_handle,
        ),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
    ):
        await handle_resume_task(MagicMock(), int(task.id), {"user": owner})
        for _ in range(100):
            if bg_handle.cancel.called:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("resume command did not finish in time")

    bg_handle.cancel.assert_called_once()


@pytest.mark.asyncio
async def test_execute_resume_background_rejects_owner_mismatch(db_session) -> None:
    """``execute_resume_background`` runs the resume under
    ``UserContext(task_owner_user_id)``. If a caller passes an owner id that
    disagrees with the task row, the symmetric guard (same as
    ``execute_task_background``) must fire before the agent resumes, so the
    runtime never executes as the wrong user."""
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id)

    agent = MagicMock()
    agent.resume_execution_by_id = AsyncMock()
    ws_manager = MagicMock()
    ws_manager.broadcast_to_task = AsyncMock()

    with (
        patch("xagent.web.api.websocket.stop_task_lease_heartbeat", new=AsyncMock()),
        patch("xagent.web.api.websocket.manager", ws_manager),
    ):
        _register_current_resume(int(task.id))
        await execute_resume_background(
            task_id=int(task.id),
            agent_service=agent,
            task_owner_user_id=int(owner.id) + 999,  # != task owner
        )

    # Guard fired before the resume ran -- nothing executed as the wrong user.
    agent.resume_execution_by_id.assert_not_awaited()
    error_types = {
        msg.get("type")
        for (msg, _tid) in (
            call.args for call in ws_manager.broadcast_to_task.call_args_list
        )
        if isinstance(msg, dict)
    }
    assert "task_error" in error_types
    task_errors = [
        call.args[0]
        for call in ws_manager.broadcast_to_task.call_args_list
        if call.args[0].get("type") == "task_error"
    ]
    assert task_errors[0]["message"] == websocket_api.CLIENT_SAFE_TASK_FAILURE
    assert task_errors[0]["error"] == websocket_api.CLIENT_SAFE_TASK_FAILURE
    assert str(int(owner.id) + 999) not in repr(task_errors[0])


@pytest.mark.asyncio
async def test_execute_resume_background_persists_assistant_for_live_turn(
    db_session,
) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.PAUSED)
    agent = MagicMock()
    context = SimpleNamespace(
        messages=[
            SimpleNamespace(
                role="user",
                metadata={"turn_id": "live-turn-1"},
            )
        ]
    )
    agent.resume_execution_by_id = AsyncMock(
        return_value={
            "status": "completed",
            "success": True,
            "output": "Guidance applied",
            "agent_result": {"context": context},
        }
    )
    ws_manager = MagicMock(broadcast_to_task=AsyncMock())

    with patch("xagent.web.api.websocket.manager", ws_manager):
        _register_current_resume(int(task.id))
        await execute_resume_background(
            task_id=int(task.id),
            agent_service=agent,
            task_owner_user_id=int(owner.id),
        )

    db_session.expire_all()
    stored = (
        db_session.query(TaskChatMessage)
        .filter(
            TaskChatMessage.task_id == int(task.id),
            TaskChatMessage.role == "assistant",
        )
        .one()
    )
    assert stored.content == "Guidance applied"
    assert stored.turn_id == "live-turn-1"
    db_session.refresh(task)
    assert task.status == TaskStatus.COMPLETED
    assert task.output == "Guidance applied"


@pytest.mark.asyncio
async def test_execute_resume_background_pops_ephemeral_secrets_after_completion(
    db_session,
) -> None:
    """execute_resume_background settles through its own _finalize_resumed_task,
    a completely separate finalizer from task_orchestrator.finish_turn - so a
    turn that paused on waiting_for_user and then genuinely completes on
    resume must pop its ephemeral secrets from THIS finalizer too, under the
    original pausing turn's id; otherwise they'd sit until the TTL reaper
    eventually reclaims them instead of being freed at the terminal outcome."""
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.WAITING_FOR_USER)
    turn_id = "resume-secrets-turn-completes"
    store_ephemeral_runtime_values(
        turn_id,
        {ConnectorRef("mcp", 1): {"secrets": {"authorization": "Bearer resume-token"}}},
    )
    assert get_ephemeral_runtime_values(turn_id) is not None

    tool_config = MagicMock()
    tool_config.get_connector_runtime_turn_id.return_value = turn_id
    agent = MagicMock(tool_config=tool_config)
    agent.resume_execution_by_id = AsyncMock(
        return_value={
            "status": "completed",
            "success": True,
            "output": "done",
            "agent_result": {},
        }
    )
    ws_manager = MagicMock(broadcast_to_task=AsyncMock())

    with patch("xagent.web.api.websocket.manager", ws_manager):
        _register_current_resume(int(task.id))
        await execute_resume_background(
            task_id=int(task.id),
            agent_service=agent,
            task_owner_user_id=int(owner.id),
        )

    db_session.refresh(task)
    assert task.status == TaskStatus.COMPLETED
    assert get_ephemeral_runtime_values(turn_id) is None
    assert pop_ephemeral_runtime_values(turn_id) is None


@pytest.mark.asyncio
async def test_execute_resume_background_keeps_ephemeral_secrets_when_resume_pauses_again(
    db_session,
) -> None:
    """A resume that itself pauses again on waiting_for_user is the same turn
    continuing under the same turn_id, not a finished one - popping here
    would strand that next resume with nothing to look up its own secrets
    under."""
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.WAITING_FOR_USER)
    turn_id = "resume-secrets-turn-repauses"
    store_ephemeral_runtime_values(
        turn_id,
        {ConnectorRef("mcp", 1): {"secrets": {"authorization": "Bearer resume-token"}}},
    )
    assert get_ephemeral_runtime_values(turn_id) is not None

    tool_config = MagicMock()
    tool_config.get_connector_runtime_turn_id.return_value = turn_id
    agent = MagicMock(tool_config=tool_config)
    agent.resume_execution_by_id = AsyncMock(
        return_value={
            "status": "waiting_for_user",
            "success": False,
            "output": "Please connect another app.",
            "agent_result": {},
        }
    )
    ws_manager = MagicMock(broadcast_to_task=AsyncMock())

    with patch("xagent.web.api.websocket.manager", ws_manager):
        _register_current_resume(int(task.id))
        await execute_resume_background(
            task_id=int(task.id),
            agent_service=agent,
            task_owner_user_id=int(owner.id),
        )

    db_session.refresh(task)
    assert task.status == TaskStatus.WAITING_FOR_USER
    assert get_ephemeral_runtime_values(turn_id) is not None
    assert pop_ephemeral_runtime_values(turn_id) is not None


@pytest.mark.asyncio
async def test_execute_resume_background_renews_ephemeral_secrets_when_resume_pauses_again(
    db_session, monkeypatch
) -> None:
    """A resume that re-pauses carries a fresh interaction lifetime of its
    own - _finalize_resumed_task must actually renew the secrets' TTL, not
    merely leave them alone, or they'd still expire on the ORIGINAL pause's
    clock even though this one is still active."""
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.WAITING_FOR_USER)
    turn_id = "resume-secrets-turn-renews"
    store_ephemeral_runtime_values(
        turn_id,
        {ConnectorRef("mcp", 1): {"secrets": {"authorization": "Bearer resume-token"}}},
    )

    real_monotonic = connector_runtime_module.time.monotonic
    offset = {"value": connector_runtime_module._EPHEMERAL_RUNTIME_TTL_SECONDS - 1}
    monkeypatch.setattr(
        connector_runtime_module.time,
        "monotonic",
        lambda: real_monotonic() + offset["value"],
    )

    tool_config = MagicMock()
    tool_config.get_connector_runtime_turn_id.return_value = turn_id
    agent = MagicMock(tool_config=tool_config)
    agent.resume_execution_by_id = AsyncMock(
        return_value={
            "status": "waiting_for_user",
            "success": False,
            "output": "Please connect another app.",
            "agent_result": {},
        }
    )
    ws_manager = MagicMock(broadcast_to_task=AsyncMock())

    with patch("xagent.web.api.websocket.manager", ws_manager):
        _register_current_resume(int(task.id))
        await execute_resume_background(
            task_id=int(task.id),
            agent_service=agent,
            task_owner_user_id=int(owner.id),
        )

    # Past the ORIGINAL store's TTL window, but well within the renewed one.
    offset["value"] += connector_runtime_module._EPHEMERAL_RUNTIME_TTL_SECONDS - 1
    assert get_ephemeral_runtime_values(turn_id) is not None
    assert pop_ephemeral_runtime_values(turn_id) is not None


@pytest.mark.asyncio
async def test_execute_resume_background_persists_missing_checkpoint_failure(
    db_session,
) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.PAUSED)
    agent = MagicMock(
        resume_execution_by_id=AsyncMock(return_value=None),
    )
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(owner.id),
            role="user",
            content="Deferred guidance",
            message_type="user_message",
            turn_id="deferred-failure-turn",
            delivery_status=DELIVERY_PENDING,
        )
    )
    db_session.commit()
    ws_manager = MagicMock(broadcast_to_task=AsyncMock())

    with patch("xagent.web.api.websocket.manager", ws_manager):
        _register_current_resume(int(task.id))
        await execute_resume_background(
            task_id=int(task.id),
            agent_service=agent,
            task_owner_user_id=int(owner.id),
            delivery_turn_id="deferred-failure-turn",
        )

    db_session.expire_all()
    db_session.refresh(task)
    assert task.status == TaskStatus.FAILED
    assert "No resumable execution checkpoint" in str(task.error_message)
    delivery = (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.turn_id == "deferred-failure-turn")
        .one()
    )
    assert delivery.delivery_status == DELIVERY_FAILED
    failures = [
        call.args[0]
        for call in ws_manager.broadcast_to_task.call_args_list
        if call.args[0].get("type") == "task_error"
    ]
    assert len(failures) == 1
    assert failures[0]["task"]["status"] == TaskStatus.FAILED.value


@pytest.mark.asyncio
async def test_resume_background_settles_running_prior_status_on_checkpoint_unavailable(
    db_session,
) -> None:
    """A RUNNING prior status is never a valid restore target.

    ``release_task_lease_no_commit`` refuses to release a lease back to
    RUNNING. A resume that steals an abandoned lease from a task whose row
    was still RUNNING (no active runner, expired TTL) must therefore fall
    through to the ordinary settle/FAILED path on a checkpoint read failure,
    not attempt a "restore to prior status" that can only dead-end with the
    lease stuck unreleased until TTL recovery.
    """
    owner = _user(db_session, "running-prior-owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    assert task.runner_id is None
    assert task.lease_expires_at is None
    agent = MagicMock(
        resume_execution_by_id=AsyncMock(
            side_effect=CheckpointUnavailableError("checkpoint query failed")
        ),
    )
    ws_manager = MagicMock(broadcast_to_task=AsyncMock())

    with patch("xagent.web.api.websocket.manager", ws_manager):
        _register_current_resume(int(task.id))
        await execute_resume_background(
            task_id=int(task.id),
            agent_service=agent,
            task_owner_user_id=int(owner.id),
        )

    db_session.expire_all()
    db_session.refresh(task)
    assert task.status == TaskStatus.FAILED
    assert task.runner_id is None
    failures = [
        call.args[0]
        for call in ws_manager.broadcast_to_task.call_args_list
        if call.args[0].get("type") == "task_error"
    ]
    assert len(failures) == 1


def test_waiting_or_paused_event_fields_shared_by_both_call_sites() -> None:
    """The live-lease restore broadcast and the historical-replay status
    reassertion both compute their event type/message off this one helper;
    pin its output so a change to either site's vocabulary is caught here
    rather than only in one of the two integration tests."""
    assert _waiting_or_paused_event_fields(TaskStatus.WAITING_FOR_USER) == (
        "task_waiting_for_user",
        "Task waiting for user response",
    )
    assert _waiting_or_paused_event_fields(TaskStatus.PAUSED) == (
        "task_paused",
        "Task paused",
    )


@pytest.mark.parametrize(
    ("prior_status", "expected_event_type"),
    [
        (TaskStatus.PAUSED, "task_paused"),
        (TaskStatus.WAITING_FOR_USER, "task_waiting_for_user"),
    ],
)
@pytest.mark.asyncio
async def test_resume_background_broadcasts_corrective_event_after_restore(
    db_session,
    prior_status: TaskStatus,
    expected_event_type: str,
) -> None:
    """After a checkpoint-unavailable restore, clients that saw the optimistic
    RUNNING transition need the prior status re-asserted -- reusing the same
    event vocabulary the historical-replay path uses for PAUSED/WAITING_FOR_USER."""
    owner = _user(db_session, "restore-broadcast-owner")
    task = _task(db_session, owner.id, status=prior_status)
    task.error_message = "earlier attempt failed"
    task.output = "prior turn answer"
    db_session.commit()
    agent = MagicMock(
        resume_execution_by_id=AsyncMock(
            side_effect=CheckpointUnavailableError("checkpoint query failed")
        ),
    )
    ws_manager = MagicMock(broadcast_to_task=AsyncMock())

    with patch("xagent.web.api.websocket.manager", ws_manager):
        _register_current_resume(int(task.id))
        await execute_resume_background(
            task_id=int(task.id),
            agent_service=agent,
            task_owner_user_id=int(owner.id),
        )

    db_session.expire_all()
    db_session.refresh(task)
    assert task.status == prior_status
    assert task.error_message is None  # named: restore clears stale error
    assert task.output == "prior turn answer"  # named: restore preserves output

    corrective = [
        call.args[0]
        for call in ws_manager.broadcast_to_task.call_args_list
        if call.args[0].get("type") == expected_event_type
    ]
    assert len(corrective) == 1
    assert corrective[0]["type"] == expected_event_type
    assert corrective[0]["status"] == prior_status.value
    assert "state_version" in corrective[0]
    assert not any(
        call.args[0].get("type") == "task_error"
        for call in ws_manager.broadcast_to_task.call_args_list
    )


@pytest.mark.parametrize(
    ("settled", "expected_error_broadcasts"),
    [(True, 1), (False, 0)],
)
@pytest.mark.asyncio
async def test_resume_failure_broadcasts_only_after_exact_settlement(
    db_session,
    settled: bool,
    expected_error_broadcasts: int,
) -> None:
    secret = "resume-provider-secret"
    owner = _user(db_session, "resume-settlement-owner")
    task = _task(db_session, owner.id, status=TaskStatus.PAUSED)
    agent = MagicMock(
        resume_execution_by_id=AsyncMock(
            side_effect=RuntimeError(f"resume failed: {secret}")
        ),
    )
    events: list[str] = []
    error_payloads: list[dict] = []

    def settle(*_args, **_kwargs) -> bool:
        events.append("settle")
        return settled

    async def broadcast(payload, *_args, **_kwargs) -> None:
        if payload.get("type") == "task_error":
            events.append("broadcast")
            error_payloads.append(payload)

    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(side_effect=broadcast),
    )

    with (
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch(
            "xagent.web.api.websocket._settle_resumed_task_lease",
            side_effect=settle,
        ),
    ):
        _register_current_resume(int(task.id))
        await execute_resume_background(
            task_id=int(task.id),
            agent_service=agent,
            task_owner_user_id=int(owner.id),
        )

    assert events == ["settle"] + (["broadcast"] * expected_error_broadcasts)
    if settled:
        assert error_payloads[0]["message"] == websocket_api.CLIENT_SAFE_TASK_FAILURE
        assert error_payloads[0]["error"] == websocket_api.CLIENT_SAFE_TASK_FAILURE
        assert secret not in repr(error_payloads[0])


@pytest.mark.asyncio
async def test_resume_failure_rejection_redacts_exception_text(db_session) -> None:
    secret = "resume-rejection-secret"
    owner = _user(db_session, "resume-rejection-owner")
    task = _task(db_session, owner.id, status=TaskStatus.PAUSED)
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(owner.id),
            role="user",
            content="Deferred guidance",
            message_type="user_message",
            turn_id="resume-rejection-turn",
            delivery_status=DELIVERY_PENDING,
        )
    )
    db_session.commit()
    agent = MagicMock(
        resume_execution_by_id=AsyncMock(
            side_effect=RuntimeError(f"resume failed: {secret}")
        ),
    )
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )

    with patch("xagent.web.api.websocket.manager", ws_manager):
        _register_current_resume(int(task.id))
        await execute_resume_background(
            task_id=int(task.id),
            agent_service=agent,
            task_owner_user_id=int(owner.id),
            delivery_turn_id="resume-rejection-turn",
            delivery_websocket=MagicMock(),
            delivery_client_message_id="resume-rejection-turn",
        )

    rejected = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_rejected"
    ]
    assert len(rejected) == 1
    assert rejected[0]["message"] == websocket_api.CLIENT_SAFE_VALIDATION_ERROR
    assert secret not in repr(rejected[0])


@pytest.mark.asyncio
async def test_deferred_injection_failure_rejects_before_any_acceptance(
    db_session,
) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.PAUSED)
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(owner.id),
            role="user",
            content="Deferred guidance",
            message_type="user_message",
            turn_id="deferred-injection-failure",
            delivery_status=DELIVERY_PENDING,
        )
    )
    db_session.commit()
    observed_leases: list[TaskLease | None] = []

    async def post_user_message(*_args, **_kwargs) -> bool:
        observed_leases.append(current_task_lease())
        return False

    agent = MagicMock(
        post_user_message=AsyncMock(side_effect=post_user_message),
        resume_execution_by_id=AsyncMock(),
    )
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )

    with (
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager.promote_resume_task"),
    ):
        await execute_resume_background(
            task_id=int(task.id),
            agent_service=agent,
            task_owner_user_id=int(owner.id),
            pending_user_message={
                "execution_message": "Deferred guidance",
                "display_message": "Deferred guidance",
                "files": [],
                "turn_id": "deferred-injection-failure",
            },
            delivery_turn_id="deferred-injection-failure",
            delivery_websocket=MagicMock(),
            delivery_client_message_id="deferred-injection-failure",
        )

    delivery_events = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") in {"message_accepted", "message_rejected"}
    ]
    assert [event["type"] for event in delivery_events] == ["message_rejected"]
    assert delivery_events[0]["retry_with_new_id"] is True
    assert len(observed_leases) == 1
    assert observed_leases[0] is not None
    assert observed_leases[0].task_id == int(task.id)
    assert observed_leases[0].runner_id
    assert observed_leases[0].run_id
    db_session.expire_all()
    delivery = (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.turn_id == "deferred-injection-failure")
        .one()
    )
    assert delivery.delivery_status == DELIVERY_FAILED


@pytest.mark.asyncio
async def test_deferred_injection_marker_failure_does_not_abort_resume(
    db_session,
) -> None:
    owner = _user(db_session, "deferred-marker-owner")
    task = _task(db_session, owner.id, status=TaskStatus.PAUSED)
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(owner.id),
            role="user",
            content="Deferred guidance",
            message_type="user_message",
            turn_id="deferred-marker-turn",
            delivery_status=DELIVERY_PENDING,
        )
    )
    db_session.commit()
    context = SimpleNamespace(
        messages=[
            SimpleNamespace(
                role="user",
                metadata={"turn_id": "deferred-marker-turn"},
            )
        ]
    )
    agent = MagicMock(
        post_user_message=AsyncMock(
            return_value=UserMessageInjectionOutcome.POSTED_FRESH
        ),
        resume_execution_by_id=AsyncMock(
            return_value={
                "status": "completed",
                "success": True,
                "output": "Applied",
                "agent_result": {"context": context},
            }
        ),
    )
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )

    def mark_delivery(_task_id: int, _turn_id: str, status: str):
        if status == websocket_api.DELIVERY_DISPATCHED:
            raise RuntimeError("delivery marker unavailable")
        return None

    with (
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager.promote_resume_task"),
        patch(
            "xagent.web.api.websocket.mark_user_message_delivery_sync",
            side_effect=mark_delivery,
        ),
    ):
        await execute_resume_background(
            task_id=int(task.id),
            agent_service=agent,
            task_owner_user_id=int(owner.id),
            pending_user_message={
                "execution_message": "Deferred guidance",
                "display_message": "Deferred guidance",
                "files": [],
                "turn_id": "deferred-marker-turn",
            },
            delivery_turn_id="deferred-marker-turn",
            delivery_websocket=MagicMock(),
            delivery_client_message_id="deferred-marker-turn",
        )

    agent.resume_execution_by_id.assert_awaited_once_with(str(task.id))
    accepted = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_accepted"
    ]
    assert len(accepted) == 1


@pytest.mark.asyncio
async def test_deferred_injection_marker_cancellation_does_not_abort_resume(
    db_session,
) -> None:
    owner = _user(db_session, "deferred-marker-cancel-owner")
    task = _task(db_session, owner.id, status=TaskStatus.PAUSED)
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(owner.id),
            role="user",
            content="Deferred guidance",
            message_type="user_message",
            turn_id="deferred-marker-cancel-turn",
            delivery_status=DELIVERY_PENDING,
        )
    )
    db_session.commit()
    context = SimpleNamespace(
        messages=[
            SimpleNamespace(
                role="user",
                metadata={"turn_id": "deferred-marker-cancel-turn"},
            )
        ]
    )
    agent = MagicMock(
        post_user_message=AsyncMock(
            return_value=UserMessageInjectionOutcome.POSTED_FRESH
        ),
        resume_execution_by_id=AsyncMock(
            return_value={
                "status": "completed",
                "success": True,
                "output": "Applied",
                "agent_result": {"context": context},
            }
        ),
    )
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )

    def mark_delivery(_task_id: int, _turn_id: str, status: str):
        if status == websocket_api.DELIVERY_DISPATCHED:
            raise asyncio.CancelledError
        return None

    with (
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager.promote_resume_task"),
        patch(
            "xagent.web.api.websocket.mark_user_message_delivery_sync",
            side_effect=mark_delivery,
        ),
    ):
        await execute_resume_background(
            task_id=int(task.id),
            agent_service=agent,
            task_owner_user_id=int(owner.id),
            pending_user_message={
                "execution_message": "Deferred guidance",
                "display_message": "Deferred guidance",
                "files": [],
                "turn_id": "deferred-marker-cancel-turn",
            },
            delivery_turn_id="deferred-marker-cancel-turn",
            delivery_websocket=MagicMock(),
            delivery_client_message_id="deferred-marker-cancel-turn",
        )

    agent.resume_execution_by_id.assert_awaited_once_with(str(task.id))
    accepted = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_accepted"
    ]
    assert len(accepted) == 1


@pytest.mark.asyncio
async def test_deferred_injection_close_failure_does_not_abort_resume(
    db_session,
) -> None:
    """Same guarantee as the deferred delivery-marker failure case above,
    for the legacy resume interaction close issued right before it."""
    owner = _user(db_session, "deferred-close-failure-owner")
    task = _task(db_session, owner.id, status=TaskStatus.PAUSED)
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(owner.id),
            role="user",
            content="Deferred guidance",
            message_type="user_message",
            turn_id="deferred-close-failure-turn",
            delivery_status=DELIVERY_PENDING,
        )
    )
    db_session.commit()
    context = SimpleNamespace(
        messages=[
            SimpleNamespace(
                role="user",
                metadata={"turn_id": "deferred-close-failure-turn"},
            )
        ]
    )
    agent = MagicMock(
        post_user_message=AsyncMock(
            return_value=UserMessageInjectionOutcome.POSTED_FRESH
        ),
        resume_execution_by_id=AsyncMock(
            return_value={
                "status": "completed",
                "success": True,
                "output": "Applied",
                "agent_result": {"context": context},
            }
        ),
    )
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    # Recorded, not asserted, inside the side_effect below: the production
    # code that calls close_legacy_resume_interaction_sync wraps it in
    # ``except Exception:``, which would silently swallow an AssertionError
    # raised from inside this callback the same way it swallows the
    # RuntimeError this test is otherwise driving. The real assertions run
    # after the `with` block, outside that handler's reach.
    observed_close_calls: list[tuple[int, str, str | None]] = []

    def fail_close(*, task_id: int, run_id: str, interaction_id: int | None) -> int:
        # A fresh session, not db_session: this runs inside the worker
        # thread run_db_io_cancellation_safe schedules it on, while the
        # test's own db_session sits unused on the main thread -- sharing
        # one Session across that handoff is not a pattern this suite uses
        # elsewhere.
        probe = next(get_db())
        try:
            live_run_id = probe.query(Task).filter(Task.id == task_id).one().run_id
        finally:
            probe.close()
        observed_close_calls.append((task_id, run_id, live_run_id))
        raise RuntimeError("interaction close unavailable")

    with (
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager.promote_resume_task"),
        patch(
            "xagent.web.api.websocket.close_legacy_resume_interaction_sync",
            side_effect=fail_close,
        ),
    ):
        await execute_resume_background(
            task_id=int(task.id),
            agent_service=agent,
            task_owner_user_id=int(owner.id),
            pending_user_message={
                "execution_message": "Deferred guidance",
                "display_message": "Deferred guidance",
                "files": [],
                "turn_id": "deferred-close-failure-turn",
            },
            delivery_turn_id="deferred-close-failure-turn",
            delivery_websocket=MagicMock(),
            delivery_client_message_id="deferred-close-failure-turn",
        )

    agent.resume_execution_by_id.assert_awaited_once_with(str(task.id))
    assert len(observed_close_calls) == 1
    called_task_id, called_run_id, live_run_id = observed_close_calls[0]
    assert called_task_id == int(task.id)
    assert called_run_id
    assert called_run_id == live_run_id
    accepted = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_accepted"
    ]
    assert len(accepted) == 1


@pytest.mark.asyncio
async def test_deferred_injection_closes_the_row_the_online_handler_observed(
    db_session,
) -> None:
    """The deferred path takes no observation of its own. Its injection is
    later still than the online one, so a read taken here would be even
    further past the point where the answered row is identifiable. The
    online handler's pre-injection observation travels in
    pending_user_message and is what the close is keyed on."""

    owner = _user(db_session, "deferred-carry-owner")
    task = _task(db_session, owner.id, status=TaskStatus.PAUSED)
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(owner.id),
            role="user",
            content="Deferred guidance",
            message_type="user_message",
            turn_id="deferred-carry-turn",
            delivery_status=DELIVERY_PENDING,
        )
    )
    db_session.commit()
    context = SimpleNamespace(
        messages=[
            SimpleNamespace(role="user", metadata={"turn_id": "deferred-carry-turn"})
        ]
    )
    agent = MagicMock(
        post_user_message=AsyncMock(
            return_value=UserMessageInjectionOutcome.POSTED_FRESH
        ),
        resume_execution_by_id=AsyncMock(
            return_value={
                "status": "completed",
                "success": True,
                "output": "Applied",
                "agent_result": {"context": context},
            }
        ),
    )
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )

    with (
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager.promote_resume_task"),
        patch(
            "xagent.web.api.websocket.active_interaction_id_sync",
            side_effect=AssertionError("the deferred path must not read its own"),
        ),
        patch(
            "xagent.web.api.websocket.close_legacy_resume_interaction_sync",
            return_value=1,
        ) as close_mock,
    ):
        await execute_resume_background(
            task_id=int(task.id),
            agent_service=agent,
            task_owner_user_id=int(owner.id),
            pending_user_message={
                "execution_message": "Deferred guidance",
                "display_message": "Deferred guidance",
                "files": [],
                "turn_id": "deferred-carry-turn",
                "interaction_id": 9876,
            },
            delivery_turn_id="deferred-carry-turn",
            delivery_websocket=MagicMock(),
            delivery_client_message_id="deferred-carry-turn",
        )

    close_mock.assert_called_once()
    assert close_mock.call_args.kwargs["task_id"] == int(task.id)
    assert close_mock.call_args.kwargs["interaction_id"] == 9876


@pytest.mark.asyncio
async def test_deferred_injection_skips_the_close_on_a_replayed_turn_id(
    db_session,
) -> None:
    """A cross-run retry of this background task re-enters with the same
    pending_user_message. post_user_message short-circuits the repeated
    turn id and reports POSTED_REPLAY, so the close must be skipped
    entirely -- not just matched against the carried (stale) interaction
    id, but never attempted. The row this test seeds is the question the
    resumed agent has staged *since* the first attempt (a different id
    from the one carried in pending_user_message), so a site that
    re-derived its close key by reading the current active row instead of
    using the carried one would retire this live question; asserting
    active_interaction_id_sync is never called pins that the deferred path
    still takes no read of its own, replay or not -- the same technique
    the carried-vs-observed test above uses. close_legacy_resume_interaction_sync
    is asserted uncalled directly: replacing the guard with `if True:`
    calls it for real and turns this assertion red on its own."""
    owner = _user(db_session, "deferred-replay-owner")
    task = _task(db_session, owner.id, status=TaskStatus.PAUSED)
    task.interaction_protocol_version = 1
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(owner.id),
            role="user",
            content="Deferred guidance",
            message_type="user_message",
            turn_id="deferred-replay-turn",
            delivery_status=DELIVERY_PENDING,
        )
    )
    db_session.commit()
    task_id = int(task.id)
    # The question staged since the first attempt -- a different row from
    # the one named in pending_user_message below.
    _seed_active_interaction_row(
        db_session,
        task_id=task_id,
        run_id="deferred-replay-run",
        idempotency_key="deferred-replay-q1",
    )
    context = SimpleNamespace(
        messages=[
            SimpleNamespace(role="user", metadata={"turn_id": "deferred-replay-turn"})
        ]
    )
    agent = MagicMock(
        post_user_message=AsyncMock(
            return_value=UserMessageInjectionOutcome.POSTED_REPLAY
        ),
        resume_execution_by_id=AsyncMock(
            return_value={
                "status": "completed",
                "success": True,
                "output": "Applied",
                "agent_result": {"context": context},
            }
        ),
    )
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )

    with (
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager.promote_resume_task"),
        patch(
            "xagent.web.api.websocket.active_interaction_id_sync",
            side_effect=AssertionError("the deferred path must not read its own"),
        ),
        patch(
            "xagent.web.api.websocket.close_legacy_resume_interaction_sync",
        ) as close_mock,
    ):
        await execute_resume_background(
            task_id=task_id,
            agent_service=agent,
            task_owner_user_id=int(owner.id),
            pending_user_message={
                "execution_message": "Deferred guidance",
                "display_message": "Deferred guidance",
                "files": [],
                "turn_id": "deferred-replay-turn",
                # The stale id carried from the first attempt -- not the
                # row seeded above, which is what a re-read would find.
                "interaction_id": 424242,
            },
            delivery_turn_id="deferred-replay-turn",
            delivery_websocket=MagicMock(),
            delivery_client_message_id="deferred-replay-turn",
        )

    agent.post_user_message.assert_awaited_once()
    close_mock.assert_not_called()


@pytest.mark.asyncio
async def test_deferred_injection_close_cancellation_does_not_abort_resume(
    db_session,
) -> None:
    """CancelledError branch specifically -- deleting it must turn this red."""
    owner = _user(db_session, "deferred-close-cancel-owner")
    task = _task(db_session, owner.id, status=TaskStatus.PAUSED)
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(owner.id),
            role="user",
            content="Deferred guidance",
            message_type="user_message",
            turn_id="deferred-close-cancel-turn",
            delivery_status=DELIVERY_PENDING,
        )
    )
    db_session.commit()
    context = SimpleNamespace(
        messages=[
            SimpleNamespace(
                role="user",
                metadata={"turn_id": "deferred-close-cancel-turn"},
            )
        ]
    )
    agent = MagicMock(
        post_user_message=AsyncMock(
            return_value=UserMessageInjectionOutcome.POSTED_FRESH
        ),
        resume_execution_by_id=AsyncMock(
            return_value={
                "status": "completed",
                "success": True,
                "output": "Applied",
                "agent_result": {"context": context},
            }
        ),
    )
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    # Recorded, not asserted, inside the side_effect below -- see the sibling
    # failure-case test's comment: an AssertionError raised from inside this
    # callback would be swallowed by the production code's own
    # ``except Exception:`` the same way it swallows an ordinary failure.
    observed_close_calls: list[tuple[int, str, str | None]] = []

    def raise_cancelled(
        *, task_id: int, run_id: str, interaction_id: int | None
    ) -> int:
        probe = next(get_db())
        try:
            live_run_id = probe.query(Task).filter(Task.id == task_id).one().run_id
        finally:
            probe.close()
        observed_close_calls.append((task_id, run_id, live_run_id))
        raise asyncio.CancelledError

    with (
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager.promote_resume_task"),
        patch(
            "xagent.web.api.websocket.close_legacy_resume_interaction_sync",
            side_effect=raise_cancelled,
        ),
    ):
        await execute_resume_background(
            task_id=int(task.id),
            agent_service=agent,
            task_owner_user_id=int(owner.id),
            pending_user_message={
                "execution_message": "Deferred guidance",
                "display_message": "Deferred guidance",
                "files": [],
                "turn_id": "deferred-close-cancel-turn",
            },
            delivery_turn_id="deferred-close-cancel-turn",
            delivery_websocket=MagicMock(),
            delivery_client_message_id="deferred-close-cancel-turn",
        )

    agent.resume_execution_by_id.assert_awaited_once_with(str(task.id))
    assert len(observed_close_calls) == 1
    called_task_id, called_run_id, live_run_id = observed_close_calls[0]
    assert called_task_id == int(task.id)
    assert called_run_id
    assert called_run_id == live_run_id
    accepted = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_accepted"
    ]
    assert len(accepted) == 1


@pytest.mark.asyncio
async def test_deferred_injection_rejects_before_post_when_lease_is_denied(
    db_session,
) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(owner.id),
            role="user",
            content="Deferred guidance",
            message_type="user_message",
            turn_id="deferred-lease-denied",
            delivery_status=DELIVERY_PENDING,
        )
    )
    db_session.commit()
    agent = MagicMock(
        post_user_message=AsyncMock(
            return_value=UserMessageInjectionOutcome.POSTED_FRESH
        ),
        resume_execution_by_id=AsyncMock(),
    )
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )

    with (
        patch("xagent.web.api.websocket._acquire_resume_task_lease", return_value=None),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager.promote_resume_task"),
    ):
        await execute_resume_background(
            task_id=int(task.id),
            agent_service=agent,
            task_owner_user_id=int(owner.id),
            pending_user_message={
                "execution_message": "Deferred guidance",
                "display_message": "Deferred guidance",
                "files": [],
                "turn_id": "deferred-lease-denied",
            },
            delivery_turn_id="deferred-lease-denied",
            delivery_websocket=MagicMock(),
            delivery_client_message_id="deferred-lease-denied",
        )

    agent.post_user_message.assert_not_awaited()
    delivery_events = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") in {"message_accepted", "message_rejected"}
    ]
    assert [event["type"] for event in delivery_events] == ["message_rejected"]
    assert delivery_events[0]["retry_with_new_id"] is True
    db_session.expire_all()
    delivery = (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.turn_id == "deferred-lease-denied")
        .one()
    )
    assert delivery.delivery_status == DELIVERY_FAILED


@pytest.mark.asyncio
async def test_resume_non_owner_non_admin_is_refused(db_session) -> None:
    owner = _user(db_session, "owner")
    stranger = _user(db_session, "stranger")
    task = _task(db_session, owner.id)
    captured, agent, mgr, ws_manager = _patched_manager_and_agent()

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
    ):
        await handle_resume_task(MagicMock(), int(task.id), {"user": stranger})

    # Authorized away before any runtime is built; an error is sent back.
    assert "task_owner_user_id" not in captured
    ws_manager.send_personal_message.assert_awaited()


@pytest.mark.asyncio
async def test_durable_attachment_failure_keeps_the_storage_key_off_the_socket(
    db_session,
    caplog,
) -> None:
    """A stored-file fault must not send the storage key to the client.

    Attachment preparation runs in the handler's *outer* scope, so this fault
    surfaces before the inner agent-execution arms and was answered with
    ``str(exc)``, which then read
    ``Failed to restore durable object: users/<id>/uploads/...`` and embedded the
    owning user's id. Same defect as the model-facing leak in #1467, one
    transport over.

    The key has since moved off the message onto ``storage_key``, so ``str(exc)``
    no longer carries it and this arm is no longer the only thing standing
    between the fault and the client. Both still matter: this pins the arm's
    fixed message, and ``test_the_wrap_keeps_the_storage_key_out_of_its_own_message``
    pins the exception. Neither alone would have caught both rounds of this.

    What this proves is the *rejection frame*: it asserts one was sent, so the
    negative assertions below cannot pass over an empty list. The broadcast
    assertion is defence in depth -- this path does not broadcast, so it holds
    vacuously and exists to fail if a future edit starts. The persisted
    rejection is not reached: ``finish_delivery_failure`` only writes when
    ``delivery_claimed`` is set, and that comes from
    ``preparation.delivery_claimed``, which never gets assigned when preparation
    is what raised. Do not read this test as covering all three egresses.
    """
    import logging

    from xagent.web.services.managed_file_ref import DurableStorageOperationError

    owner = _user(db_session, "durable-leak-owner")
    task = _task(db_session, owner.id, status=TaskStatus.COMPLETED)
    owner_id = int(owner.id)
    task_id = int(task.id)
    db_session.close()

    storage_key = f"users/{owner_id}/uploads/8ac1f2/quarterly-report.xlsx"

    class _ProviderThrottled(RuntimeError):
        pass

    def failing_prepare(**_kwargs):
        wrap = DurableStorageOperationError(
            "Failed to restore durable object", storage_key=storage_key
        )
        wrap.__cause__ = _ProviderThrottled("SlowDown: reduce your request rate")
        raise wrap

    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    logger_name = "xagent.web.api.websocket"

    with (
        patch(
            "xagent.web.api.websocket._prepare_websocket_turn_sync",
            side_effect=failing_prepare,
        ),
        patch("xagent.web.api.websocket.manager", ws_manager),
        caplog.at_level(logging.WARNING, logger=logger_name),
    ):
        with pytest.raises(DurableStorageOperationError):
            await _handle_chat_message_unserialized(
                MagicMock(),
                task_id,
                {
                    "message": "with an attachment",
                    "client_message_id": "durable-leak-probe",
                    "user": SimpleNamespace(id=owner_id, is_admin=False),
                    "files": ["8ac1f2"],
                },
            )

    # The rejection frame must actually have gone out, or every negative
    # assertion below would hold over nothing and this test would pass while
    # answering the client with anything at all.
    frames = [str(call) for call in ws_manager.send_personal_message.await_args_list]
    assert any("message_rejected" in frame for frame in frames), frames
    # Every outbound egress: nothing may carry the key or the provider text.
    outbound = frames + [
        str(call) for call in ws_manager.broadcast_to_task.await_args_list
    ]
    for payload in outbound:
        assert storage_key not in payload
        assert f"users/{owner_id}" not in payload
        assert "Failed to restore durable object" not in payload
        assert "SlowDown" not in payload

    # The server-side record keeps the whole chain, exactly once.
    fault_lines = [
        logging.Formatter("%(message)s").format(entry)
        for entry in caplog.records
        if entry.name == logger_name
        and "Durable storage unavailable" in entry.getMessage()
    ]
    assert len(fault_lines) == 1, fault_lines
    assert "during websocket chat turn preparation" in fault_lines[0]
    assert storage_key in fault_lines[0]
    assert "_ProviderThrottled" in fault_lines[0]


@pytest.mark.asyncio
async def test_a_dispatch_fault_is_labelled_with_the_message_that_failed(
    db_session,
    caplog,
) -> None:
    """The endpoint arm must name the message it was applying, not "chat turn".

    That arm guards the whole receive-loop dispatch, so it sees faults from
    every message type whose handler lets one through. It logged a fixed
    ``"websocket chat turn"``, mislabelling all of them in the single line meant
    to be authoritative about what failed.

    ``resume_task`` rather than ``execute_task``, and the difference matters:
    ``handle_execute_task`` ends in ``except RuntimeError`` with no re-raise, so
    a durable fault there never reaches this arm at all. An earlier version of
    this test mocked that handler and asserted a label for a path production
    cannot take -- green, and describing nothing.
    ``handle_resume_task`` propagates, so mocking it to raise stands in for a
    real fault, and
    ``test_a_type_is_unlabelled_only_because_its_handler_swallows`` is what keeps
    that distinction true rather than remembered.

    Driven through the real endpoint, because neither the label map nor its
    agreement with the cascade would notice this arm going back to a constant.
    """
    import json
    import logging

    from fastapi import WebSocketDisconnect

    from xagent.web.services.managed_file_ref import DurableStorageOperationError

    owner = _user(db_session, "dispatch-label-owner")
    task = _task(db_session, owner.id)
    owner_id = int(owner.id)
    task_id = int(task.id)
    db_session.close()

    websocket = MagicMock()
    websocket.accept = AsyncMock()
    websocket.receive_text = AsyncMock(
        side_effect=[json.dumps({"type": "resume_task"}), WebSocketDisconnect()]
    )
    ws_manager = MagicMock(
        register_connection=MagicMock(),
        disconnect=MagicMock(),
        send_personal_message=AsyncMock(),
        broadcast_to_task=AsyncMock(),
    )
    logger_name = "xagent.web.api.websocket"

    with (
        patch.object(websocket_api, "manager", ws_manager),
        patch.object(
            websocket_api,
            "get_authenticated_user",
            AsyncMock(return_value=SimpleNamespace(id=owner_id, is_admin=False)),
        ),
        patch.object(websocket_api, "handle_status_request", AsyncMock()),
        patch.object(
            websocket_api,
            "handle_resume_task",
            AsyncMock(
                side_effect=DurableStorageOperationError(
                    "Failed to restore durable object",
                    storage_key=f"users/{owner_id}/uploads/a/b.txt",
                )
            ),
        ),
        caplog.at_level(logging.WARNING, logger=logger_name),
    ):
        await websocket_api.websocket_chat_endpoint(websocket, task_id, None)

    fault_lines = [
        entry.getMessage()
        for entry in caplog.records
        if entry.name == logger_name
        and "Durable storage unavailable" in entry.getMessage()
    ]
    assert len(fault_lines) == 1, fault_lines
    assert "during websocket resume_task" in fault_lines[0]
    assert "chat turn" not in fault_lines[0]
    # The key still reaches the log, from the attribute rather than the message.
    assert f"storage_key=users/{owner_id}/uploads/a/b.txt" in fault_lines[0]


@pytest.mark.asyncio
async def test_a_durable_integrity_fault_is_answered_as_corruption_not_an_outage(
    db_session,
    caplog,
) -> None:
    """The integrity subclass through the real cascade, not just the parent.

    ``test_durable_attachment_failure_keeps_the_storage_key_off_the_socket``
    injects only ``DurableStorageOperationError``, so it would pass with these
    two arms swapped -- the parent would catch the subclass and tell the client
    to retry something retrying cannot fix, while emitting a transient-outage
    warning over the permanent-corruption ERROR already logged upstream.

    Ordering is checked across all twelve pairs by
    ``test_the_integrity_arm_precedes_its_parent_at_every_site``; this is what
    proves this pair's arms also produce the right answers.
    """
    import logging

    from xagent.web.services.managed_file_ref import (
        FILE_INTEGRITY_REUPLOAD_MESSAGE,
        DurableObjectIntegrityError,
    )

    owner = _user(db_session, "integrity-answer-owner")
    task = _task(db_session, owner.id, status=TaskStatus.COMPLETED)
    owner_id = int(owner.id)
    task_id = int(task.id)
    db_session.close()

    def failing_prepare(**_kwargs):
        raise DurableObjectIntegrityError(
            FILE_INTEGRITY_REUPLOAD_MESSAGE,
            storage_key="users/7/uploads/8ac1f2/corrupt.txt",
        )

    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    logger_name = "xagent.web.api.websocket"

    with (
        patch(
            "xagent.web.api.websocket._prepare_websocket_turn_sync",
            side_effect=failing_prepare,
        ),
        patch("xagent.web.api.websocket.manager", ws_manager),
        caplog.at_level(logging.WARNING, logger=logger_name),
    ):
        with pytest.raises(DurableObjectIntegrityError):
            await _handle_chat_message_unserialized(
                MagicMock(),
                task_id,
                {
                    "message": "with a corrupted attachment",
                    "client_message_id": "integrity-probe",
                    "user": SimpleNamespace(id=owner_id, is_admin=False),
                    "files": ["8ac1f2"],
                },
            )

    frames = [str(call) for call in ws_manager.send_personal_message.await_args_list]
    assert any("message_rejected" in frame for frame in frames), frames
    # Told to re-upload, not to retry: retrying cannot repair corruption.
    assert any("integrity check" in frame for frame in frames), frames
    assert not any("temporarily unavailable" in frame for frame in frames), frames

    assert not [
        entry
        for entry in caplog.records
        if entry.name == logger_name
        and "Durable storage unavailable" in entry.getMessage()
    ], "an integrity fault emitted an outage warning -- the arms are misordered"
