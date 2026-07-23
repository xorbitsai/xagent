import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.web.api import websocket as websocket_api
from xagent.web.api.chat import AgentServiceManager, resolve_agent_service_memory_policy
from xagent.web.api.websocket import (
    ConnectionManager,
    _normalize_file_outputs,
    handle_build_preview_execution,
)
from xagent.web.models.database import Base
from xagent.web.models.task import Task
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.schemas.chat import TaskCreateResponse
from xagent.web.services.managed_file_ref import (
    DurableStorageOperationError,
    build_task_output_storage_key,
)


class _BlockingPreviewWebSocket:
    def __init__(self) -> None:
        self.receive_count = 0
        self.waiting_for_next_message = asyncio.Event()

    async def accept(self) -> None:
        return None

    async def receive_text(self) -> str:
        self.receive_count += 1
        if self.receive_count == 1:
            return json.dumps({"type": "preview"})
        self.waiting_for_next_message.set()
        await asyncio.Future()
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_build_preview_endpoint_disconnects_when_cancelled(monkeypatch) -> None:
    task_id = 123
    websocket = _BlockingPreviewWebSocket()
    connection_manager = ConnectionManager()

    async def register_preview_connection(*args) -> None:
        connection_manager.register_connection(websocket, task_id)

    monkeypatch.setattr(websocket_api, "manager", connection_manager)
    monkeypatch.setattr(
        websocket_api,
        "get_authenticated_user",
        AsyncMock(return_value=SimpleNamespace(id=7)),
    )
    monkeypatch.setattr(
        websocket_api,
        "handle_build_preview_execution",
        AsyncMock(side_effect=register_preview_connection),
    )

    endpoint = asyncio.create_task(
        websocket_api.websocket_build_preview_endpoint(websocket, "token")
    )
    await websocket.waiting_for_next_message.wait()

    assert connection_manager.active_connections == {task_id: [websocket]}

    endpoint.cancel()
    with pytest.raises(asyncio.CancelledError):
        await endpoint

    assert connection_manager.active_connections == {}


@pytest.mark.asyncio
async def test_handle_build_preview_execution_uses_normal_task_flow():
    mock_websocket = AsyncMock()
    mock_user = MagicMock(spec=User)
    mock_user.id = 1
    mock_user.is_admin = False

    message_data = {
        "instructions": "test instructions",
        "execution_mode": "graph",
        "models": {
            "general": 1,
        },
        "tool_categories": [],  # Empty list to trigger the potential issue
        "message": "test message",
    }

    mock_db = MagicMock(spec=Session)
    task_response = TaskCreateResponse(
        task_id=123,
        title="test message",
        status="pending",
        created_at="2026-05-20T00:00:00Z",
    )
    with (
        patch("xagent.web.models.database.get_db", return_value=iter([mock_db])),
        patch(
            "xagent.web.api.chat.create_task",
            new=AsyncMock(return_value=task_response),
        ) as mock_create_task,
        patch(
            "xagent.web.api.websocket.handle_chat_message",
            new=AsyncMock(),
        ) as mock_handle_chat_message,
        patch(
            "xagent.web.api.websocket.manager.register_connection"
        ) as mock_register_connection,
        patch(
            "xagent.web.api.websocket.manager.connect", new=AsyncMock()
        ) as mock_connect,
    ):
        await handle_build_preview_execution(mock_websocket, message_data, mock_user)

    mock_create_task.assert_awaited_once()
    create_request = mock_create_task.await_args.args[0]
    assert create_request.is_visible is False
    assert create_request.agent_id is None
    assert create_request.agent_config["instructions"] == "test instructions"
    assert create_request.llm_ids == ["1", None, None, None]
    assert create_request.files is None
    mock_register_connection.assert_called_once_with(mock_websocket, 123)
    mock_connect.assert_not_awaited()
    mock_handle_chat_message.assert_awaited_once()
    assert mock_handle_chat_message.await_args.args[1] == 123


@pytest.mark.asyncio
async def test_handle_build_preview_execution_does_not_use_preview_sessions():
    mock_websocket = AsyncMock()
    mock_user = MagicMock(spec=User)
    mock_user.id = 1
    mock_user.is_admin = False

    message_data = {
        "type": "preview",
        "instructions": "test instructions",
        "models": {"general": 1},
        "message": "test message",
        "files": [{"file_id": "normal-file-1", "name": "data.csv"}],
    }

    mock_db = MagicMock(spec=Session)
    task_response = TaskCreateResponse(
        task_id=123,
        title="test message",
        status="pending",
        created_at="2026-05-20T00:00:00Z",
    )
    with (
        patch("xagent.web.models.database.get_db", return_value=iter([mock_db])),
        patch(
            "xagent.web.api.chat.create_task",
            new=AsyncMock(return_value=task_response),
        ) as mock_create_task,
        patch(
            "xagent.web.api.websocket.handle_chat_message",
            new=AsyncMock(),
        ) as mock_handle_chat_message,
        patch("xagent.web.api.websocket.manager.register_connection"),
    ):
        await handle_build_preview_execution(mock_websocket, message_data, mock_user)

    create_request = mock_create_task.await_args.args[0]
    assert create_request.is_visible is False
    assert "preview_session_id" not in create_request.agent_config
    assert "preview_session_id" not in vars(mock_websocket.state)
    assert mock_handle_chat_message.await_args.args[2]["context"] == {}


@pytest.mark.asyncio
async def test_handle_build_preview_execution_reuses_existing_preview_task():
    mock_websocket = AsyncMock()
    mock_websocket.state.preview_task_id = 123
    mock_user = MagicMock(spec=User)
    mock_user.id = 1
    mock_user.is_admin = False

    message_data = {
        "type": "preview",
        "instructions": "updated instructions",
        "models": {"general": 1},
        "message": "second turn",
        "files": [{"file_id": "second-turn-file", "name": "data.csv"}],
    }

    with (
        patch(
            "xagent.web.api.chat.create_task",
            new=AsyncMock(),
        ) as mock_create_task,
        patch(
            "xagent.web.api.websocket.handle_chat_message",
            new=AsyncMock(),
        ) as mock_handle_chat_message,
        patch(
            "xagent.web.api.websocket.manager.register_connection"
        ) as mock_register_connection,
    ):
        await handle_build_preview_execution(mock_websocket, message_data, mock_user)

    mock_create_task.assert_not_awaited()
    mock_register_connection.assert_not_called()
    mock_handle_chat_message.assert_awaited_once()
    assert mock_handle_chat_message.await_args.args[1] == 123
    assert mock_handle_chat_message.await_args.args[2]["message"] == "second turn"
    assert mock_handle_chat_message.await_args.args[2]["files"] == [
        {"file_id": "second-turn-file", "name": "data.csv"}
    ]
    mock_websocket.send_text.assert_not_called()


@pytest.mark.asyncio
async def test_handle_build_preview_execution_creates_task_when_no_preview_task_exists():
    mock_websocket = AsyncMock()
    mock_user = MagicMock(spec=User)
    mock_user.id = 1
    mock_user.is_admin = False

    message_data = {
        "type": "preview",
        "instructions": "updated instructions",
        "execution_mode": "balanced",
        "models": {"general": 1},
        "message": "second turn",
    }

    mock_db = MagicMock(spec=Session)
    task_response = TaskCreateResponse(
        task_id=456,
        title="second turn",
        status="pending",
        created_at="2026-05-20T00:00:00Z",
    )
    with (
        patch("xagent.web.models.database.get_db", return_value=iter([mock_db])),
        patch(
            "xagent.web.api.chat.create_task",
            new=AsyncMock(return_value=task_response),
        ) as mock_create_task,
        patch(
            "xagent.web.api.websocket.handle_chat_message",
            new=AsyncMock(),
        ) as mock_handle_chat_message,
        patch("xagent.web.api.websocket.manager.disconnect") as mock_disconnect,
        patch(
            "xagent.web.api.websocket.manager.register_connection"
        ) as mock_register_connection,
    ):
        await handle_build_preview_execution(mock_websocket, message_data, mock_user)

    mock_create_task.assert_awaited_once()
    create_request = mock_create_task.await_args.args[0]
    assert create_request.agent_config["instructions"] == "updated instructions"
    assert create_request.is_visible is False
    assert mock_disconnect.call_args_list == []
    mock_register_connection.assert_called_once_with(mock_websocket, 456)
    mock_handle_chat_message.assert_awaited_once()
    assert mock_handle_chat_message.await_args.args[1] == 456
    assert mock_websocket.state.preview_task_id == 456


def test_preview_agent_config_uses_in_memory_disabled_policy():
    task = MagicMock(spec=Task)
    task.agent_id = None
    task.agent_config = {
        "is_preview": True,
    }

    policy = resolve_agent_service_memory_policy(task=task)

    assert isinstance(policy.memory, InMemoryMemoryStore)
    assert policy.memory_enabled is False


def test_inline_preview_agent_config_uses_in_memory_disabled_policy():
    task = MagicMock(spec=Task)
    task.agent_id = None
    task.execution_mode = "balanced"
    task.agent_config = {
        "instructions": "preview instructions",
        "knowledge_bases": [],
        "skills": [],
        "tool_categories": [],
        "is_preview": True,
    }

    agent_config = AgentServiceManager()._load_task_inline_agent_config(task)
    policy = resolve_agent_service_memory_policy(
        task=task,
        agent_config=agent_config,
    )

    assert isinstance(policy.memory, InMemoryMemoryStore)
    assert policy.memory_enabled is False


def test_normalize_file_outputs_rolls_back_when_durable_storage_fails(
    monkeypatch, tmp_path
):
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)
    db = SessionLocal()
    try:
        user = User(username="ws-output-user", password_hash="hash")
        db.add(user)
        db.flush()
        task = Task(id=321, user_id=user.id, title="Output outage task")
        db.add(task)
        db.commit()

        uploads_dir = tmp_path / "uploads"
        output_path = uploads_dir / "user_1" / "web_task_321" / "output" / "report.txt"
        output_path.parent.mkdir(parents=True)
        output_path.write_text("report", encoding="utf-8")
        monkeypatch.setattr(
            "xagent.web.api.websocket.get_uploads_dir", lambda: uploads_dir
        )

        from xagent.core.file_storage.storage import FsspecFileStorage

        def fail_put_file(self, source, key, content_type=None):
            raise RuntimeError("simulated durable output outage")

        monkeypatch.setattr(FsspecFileStorage, "put_file", fail_put_file)

        with pytest.raises(DurableStorageOperationError):
            _normalize_file_outputs(
                db,
                task_id=321,
                task_user_id=int(user.id),
                file_outputs=[str(output_path)],
            )

        db.rollback()
        assert db.query(UploadedFile).filter_by(task_id=321).all() == []
    finally:
        db.close()
        engine.dispose()


def test_normalize_file_outputs_refreshes_existing_output_row(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)
    db = SessionLocal()
    try:
        user = User(username="ws-refresh-user", password_hash="hash")
        db.add(user)
        db.flush()
        task = Task(id=654, user_id=user.id, title="Output refresh task")
        db.add(task)
        db.commit()

        uploads_dir = tmp_path / "uploads"
        output_path = uploads_dir / "user_1" / "web_task_654" / "output" / "report.txt"
        output_path.parent.mkdir(parents=True)
        output_path.write_text("old-data", encoding="utf-8")
        monkeypatch.setattr(
            "xagent.web.api.websocket.get_uploads_dir", lambda: uploads_dir
        )

        _normalize_file_outputs(
            db,
            task_id=654,
            task_user_id=int(user.id),
            file_outputs=[str(output_path)],
        )
        record = db.query(UploadedFile).filter_by(task_id=654).one()
        storage_key = build_task_output_storage_key(
            int(user.id),
            654,
            str(record.file_id),
            "output/report.txt",
        )

        output_path.write_text("new-data", encoding="utf-8")
        _normalize_file_outputs(
            db,
            task_id=654,
            task_user_id=int(user.id),
            file_outputs=[str(output_path)],
        )

        db.refresh(record)
        assert record.checksum is not None
        with get_unscoped_file_storage().open_read(storage_key) as handle:
            assert handle.read() == b"new-data"
    finally:
        db.close()
        engine.dispose()
        get_unscoped_file_storage.cache_clear()


@pytest.mark.asyncio
async def test_websocket_build_preview_endpoint_clear_context():
    """
    Test that websocket_build_preview_endpoint handles 'clear_context' message correctly.
    """
    from xagent.web.api.websocket import websocket_build_preview_endpoint

    mock_websocket = AsyncMock()
    mock_websocket.state = MagicMock()
    mock_websocket.state.preview_task_id = 123

    # Mock user
    mock_user = MagicMock(spec=User)
    mock_user.id = 1

    # Setup sequence of events: receive 'clear_context', then raise WebSocketDisconnect to exit loop
    from fastapi import WebSocketDisconnect

    mock_websocket.receive_text.side_effect = [
        json.dumps({"type": "clear_context"}),
        WebSocketDisconnect(),
    ]

    with patch(
        "xagent.web.api.websocket.get_authenticated_user", return_value=mock_user
    ):
        await websocket_build_preview_endpoint(mock_websocket)

    # Verify accept was called
    mock_websocket.accept.assert_called_once()

    assert mock_websocket.state.preview_task_id is None

    # Verify a response was sent
    send_text_calls = mock_websocket.send_text.call_args_list
    assert len(send_text_calls) == 1
    sent_data = json.loads(send_text_calls[0][0][0])
    assert sent_data["type"] == "context_cleared"
    assert "timestamp" in sent_data


@pytest.mark.asyncio
async def test_websocket_build_preview_endpoint_pause_resume():
    """
    Test that websocket_build_preview_endpoint handles 'pause' and 'resume' messages correctly.
    """
    from xagent.web.api.websocket import websocket_build_preview_endpoint

    mock_websocket = AsyncMock()
    mock_websocket.state = MagicMock()

    mock_websocket.state.preview_task_id = 123

    mock_user = MagicMock(spec=User)
    mock_user.id = 1

    from fastapi import WebSocketDisconnect

    mock_websocket.receive_text.side_effect = [
        json.dumps({"type": "pause"}),
        json.dumps({"type": "resume"}),
        WebSocketDisconnect(),
    ]

    with (
        patch(
            "xagent.web.api.websocket.get_authenticated_user", return_value=mock_user
        ),
        patch("xagent.web.api.websocket.handle_pause_task", new=AsyncMock()) as pause,
        patch("xagent.web.api.websocket.handle_resume_task", new=AsyncMock()) as resume,
    ):
        await websocket_build_preview_endpoint(mock_websocket)

    pause.assert_awaited_once()
    assert pause.await_args.args[1] == 123
    resume.assert_awaited_once()
    assert resume.await_args.args[1] == 123
