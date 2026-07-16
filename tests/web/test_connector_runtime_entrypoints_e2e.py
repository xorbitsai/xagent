from __future__ import annotations

import os
import secrets
import shutil
import tempfile
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from xagent.web.api.auth import auth_router
from xagent.web.api.chat import AgentServiceManager, chat_router
from xagent.web.api.share import share_router
from xagent.web.api.websocket import handle_chat_message
from xagent.web.api.widget import widget_router
from xagent.web.channels.feishu.bot import FeishuBotInstance
from xagent.web.channels.telegram.bot import TelegramBotInstance
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base, get_db, get_engine
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.task import Task, TaskConnectorRuntimeContext, TaskStatus
from xagent.web.models.user import User
from xagent.web.models.user_channel import UserChannel


def _override_get_db() -> Iterator[Session]:
    db = None
    try:
        db = next(get_db())
        yield db
    finally:
        if db is not None:
            db.close()


app = FastAPI()
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(widget_router)
app.include_router(share_router)
app.dependency_overrides[get_db] = _override_get_db
client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def e2e_db() -> Iterator[None]:
    from xagent.web.models.database import init_db

    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test.db")
    init_db(db_url=f"sqlite:///{temp_db_path}")
    try:
        yield
    finally:
        Base.metadata.drop_all(bind=get_engine())
        shutil.rmtree(temp_dir, ignore_errors=True)


def _setup_admin_headers() -> dict[str, str]:
    status = client.get("/api/auth/setup-status")
    assert status.status_code == 200, status.text
    if status.json().get("needs_setup", True):
        setup = client.post(
            "/api/auth/setup-admin",
            json={
                "username": "admin",
                "email": "admin@example.com",
                "password": "admin123",
            },
        )
        assert setup.status_code == 200, setup.text
    login = client.post(
        "/api/auth/login", json={"username": "admin", "password": "admin123"}
    )
    assert login.status_code == 200, login.text
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


def _db_session() -> Session:
    return next(get_db())


def _admin_user(db: Session) -> User:
    user = db.query(User).filter(User.username == "admin").one()
    return user


def _create_agent(
    db: Session,
    user: User,
    *,
    name: str,
    tool_categories: list[str] | None = None,
    widget_enabled: bool = False,
    share_enabled: bool = False,
    share_token: str | None = None,
) -> Agent:
    agent = Agent(
        user_id=user.id,
        name=name,
        description=f"{name} description",
        instructions=f"{name} instructions",
        execution_mode="balanced",
        status=AgentStatus.PUBLISHED,
        tool_categories=tool_categories or [],
        widget_enabled=widget_enabled,
        widget_key=f"wk-{secrets.token_urlsafe(24)}" if widget_enabled else None,
        allowed_domains=["example.com"] if widget_enabled else [],
        share_enabled=share_enabled,
        share_token=share_token,
    )
    db.add(agent)
    db.flush()
    return agent


def _create_mcp_server(
    db: Session,
    user: User,
    *,
    name: str,
    with_runtime_declaration: bool,
) -> MCPServer:
    kwargs: dict[str, Any] = {}
    if with_runtime_declaration:
        kwargs = {
            "runtime_input_schema": {
                "context": {"account_id": {"type": "string", "required": False}}
            },
            "runtime_bindings": [
                {
                    "source": {"input_type": "context", "key": "account_id"},
                    "target": {"target_type": "mcp_meta", "key": "account_id"},
                }
            ],
        }
    server = MCPServer(
        name=name,
        description=f"{name} description",
        managed="external",
        transport="streamable_http",
        url=f"https://example.com/{name}/mcp",
        **kwargs,
    )
    db.add(server)
    db.flush()
    db.add(
        UserMCPServer(
            user_id=user.id,
            mcpserver_id=server.id,
            is_owner=True,
            can_edit=True,
            can_delete=True,
            is_active=True,
        )
    )
    db.flush()
    return server


def _task(task_id: int) -> Task:
    db = _db_session()
    try:
        return db.query(Task).filter(Task.id == task_id).one()
    finally:
        db.close()


@pytest.mark.parametrize(
    ("scenario", "expected_message"),
    [
        (
            "no_asr",
            "I couldn't understand that voice message because no speech "
            "recognition model is configured. Configure an ASR model or send "
            "the request as text.",
        ),
        (
            "missing_download",
            "I couldn't transcribe that voice message. Please try again or send "
            "the request as text.",
        ),
    ],
)
@pytest.mark.asyncio
async def test_telegram_voice_errors_are_reported_to_user(
    e2e_db: None,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    expected_message: str,
) -> None:
    _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        channel = UserChannel(
            user_id=user.id,
            channel_type="telegram",
            channel_name=f"Telegram voice {scenario}",
            config={},
            is_active=True,
        )
        db.add(channel)
        db.commit()
        db.refresh(channel)

        agent_manager = _FakeAgentManager()
        monkeypatch.setattr(
            "xagent.web.channels.telegram.bot.get_agent_manager",
            lambda: agent_manager,
        )

        async def _restore_telegram_task_context(
            *_args: Any,
            **_kwargs: Any,
        ) -> None:
            return None

        monkeypatch.setattr(
            "xagent.web.channels.telegram.bot.restore_telegram_task_context",
            _restore_telegram_task_context,
        )

        class _FakeASR:
            def __init__(self) -> None:
                self.closed = False

            async def aclose(self) -> None:
                self.closed = True

        asr_model = _FakeASR()
        voice = SimpleNamespace(file_id="telegram-voice-id")
        bot = object.__new__(TelegramBotInstance)
        bot.channel_id = int(channel.id)
        bot.channel_name = f"Telegram voice {scenario}"
        bot.active_tasks = {}
        bot.bot = object()
        bot.user_preparing_executions = set()
        bot.user_stop_events = {}
        bot.user_active_executions = {}
        bot._save_active_tasks = lambda: None
        bot._clear_user_stop_request = lambda _user_id: None
        bot._consume_user_stop_request = lambda _user_id: False
        bot._resolve_voice_asr_model = (
            (lambda _db, _user: None)
            if scenario == "no_asr"
            else (lambda _db, _user: asr_model)
        )

        async def _extract_message_content(_message: Any) -> tuple[str, list[Any]]:
            return "", [voice]

        async def _download_and_register_files(**_kwargs: Any) -> list[dict[str, Any]]:
            return []

        bot._extract_message_content = _extract_message_content
        bot._download_and_register_files = _download_and_register_files

        answers: list[str] = []

        class _TelegramMessage:
            from_user = SimpleNamespace(id=123)
            chat = SimpleNamespace(id=456)
            voice = SimpleNamespace(file_id="telegram-voice-id")

            async def answer(self, text: str, **_kwargs: Any) -> SimpleNamespace:
                answers.append(text)
                return SimpleNamespace(message_id=1)

        await bot._process_user_messages_batch(123, [_TelegramMessage()])

        assert answers == [expected_message]
        if scenario == "missing_download":
            assert asr_model.closed is True
    finally:
        db.close()


def _context_row_count(task_id: int) -> int:
    db = _db_session()
    try:
        return (
            db.query(TaskConnectorRuntimeContext)
            .filter(TaskConnectorRuntimeContext.task_id == task_id)
            .count()
        )
    finally:
        db.close()


def _smuggled_payload(connector_id: int = 999999) -> list[dict[str, Any]]:
    return [
        {
            "connector_ref": {"connector_type": "mcp", "connector_id": connector_id},
            "context": {"account_id": "should-not-bind"},
            "secrets": {"authorization": "Bearer should-not-persist"},
        }
    ]


class _FakeWebSocket:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send_text(self, message: str) -> None:
        self.messages.append(message)


class _FakeTracer:
    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def add_handler(self, handler: Any) -> None:
        self.handlers.append(handler)


class _FakeAgentService:
    def __init__(self) -> None:
        self.tracer = _FakeTracer()

    def set_execution_context_messages(self, _messages: list[Any]) -> None:
        pass

    def set_recovered_skill_context(self, _skill_context: Any) -> None:
        pass


class _FakeAgentManager:
    def __init__(self, execution_result: dict[str, Any] | None = None) -> None:
        self.service = _FakeAgentService()
        self.execute_calls: list[dict[str, Any]] = []
        self.execution_result = execution_result or {"success": True, "output": "done"}

    async def get_agent_for_task(
        self,
        _task_id: int,
        _db: Session,
        *,
        user: User,
    ) -> _FakeAgentService:
        return self.service

    async def execute_task(self, **_kwargs: Any) -> dict[str, Any]:
        self.execute_calls.append(_kwargs)
        return dict(self.execution_result)


def test_web_chat_create_filters_runtime_declared_connectors_and_ignores_payload(
    e2e_db: None,
) -> None:
    headers = _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        runtime_server = _create_mcp_server(
            db,
            user,
            name="runtime-web-chat",
            with_runtime_declaration=True,
        )
        plain_server = _create_mcp_server(
            db,
            user,
            name="plain-web-chat",
            with_runtime_declaration=False,
        )
        agent = _create_agent(
            db,
            user,
            name="Runtime Web Chat Agent",
            tool_categories=["mcp"],
        )
        db.commit()
        db.refresh(agent)
        db.refresh(runtime_server)
        db.refresh(plain_server)
    finally:
        db.close()

    response = client.post(
        "/api/chat/task/create",
        headers=headers,
        json={
            "title": "runtime web chat",
            "description": "create task",
            "agent_id": int(agent.id),
            "connector_runtime_context": _smuggled_payload(int(runtime_server.id)),
        },
    )
    assert response.status_code == 200, response.text

    task = _task(int(response.json()["task_id"]))
    assert task.connector_runtime_selected_refs == [
        {"connector_type": "mcp", "connector_id": int(runtime_server.id)}
    ]
    assert {"connector_type": "mcp", "connector_id": int(plain_server.id)} not in (
        task.connector_runtime_selected_refs or []
    )
    assert _context_row_count(int(task.id)) == 0


def test_web_chat_preview_placeholder_snapshot_is_empty_and_payload_is_ignored(
    e2e_db: None,
) -> None:
    headers = _setup_admin_headers()
    response = client.post(
        "/api/chat/task/create",
        headers=headers,
        json={
            "title": "preview placeholder",
            "description": "preview",
            "is_preview": True,
            "connector_runtime_context": _smuggled_payload(),
        },
    )
    assert response.status_code == 200, response.text

    task = _task(int(response.json()["task_id"]))
    assert task.connector_runtime_selected_refs == []
    assert _context_row_count(int(task.id)) == 0


def test_visible_connector_without_runtime_declaration_snapshots_empty(
    e2e_db: None,
) -> None:
    headers = _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        plain_server = _create_mcp_server(
            db,
            user,
            name="plain-only-web-chat",
            with_runtime_declaration=False,
        )
        agent = _create_agent(
            db,
            user,
            name="Plain Connector Agent",
            tool_categories=["mcp"],
        )
        db.commit()
        db.refresh(agent)
        db.refresh(plain_server)
    finally:
        db.close()

    response = client.post(
        "/api/chat/task/create",
        headers=headers,
        json={
            "title": "plain connector",
            "description": "plain connector",
            "agent_id": int(agent.id),
        },
    )
    assert response.status_code == 200, response.text

    task = _task(int(response.json()["task_id"]))
    assert task.connector_runtime_selected_refs == []
    assert {"connector_type": "mcp", "connector_id": int(plain_server.id)} not in (
        task.connector_runtime_selected_refs or []
    )


def test_widget_and_share_create_snapshot_and_ignore_smuggled_payload(
    e2e_db: None,
) -> None:
    _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        runtime_server = _create_mcp_server(
            db,
            user,
            name="runtime-public-chat",
            with_runtime_declaration=True,
        )
        agent = _create_agent(
            db,
            user,
            name="Public Runtime Agent",
            tool_categories=["mcp"],
            widget_enabled=True,
            share_enabled=True,
            share_token="share-runtime-token",
        )
        db.commit()
        db.refresh(agent)
        db.refresh(runtime_server)
    finally:
        db.close()

    widget_auth = client.post(
        "/api/widget/auth",
        json={"widget_key": agent.widget_key, "guest_id": "guest-runtime"},
    )
    assert widget_auth.status_code == 200, widget_auth.text
    widget_headers = {"Authorization": f"Bearer {widget_auth.json()['access_token']}"}
    widget_response = client.post(
        "/api/widget/chat/task/create",
        headers=widget_headers,
        json={
            "title": "widget runtime",
            "description": "widget",
            "agent_id": int(agent.id),
            "connector_runtime_context": _smuggled_payload(int(runtime_server.id)),
        },
    )
    assert widget_response.status_code == 200, widget_response.text
    widget_task = _task(int(widget_response.json()["task_id"]))
    assert widget_task.connector_runtime_selected_refs == [
        {"connector_type": "mcp", "connector_id": int(runtime_server.id)}
    ]
    assert _context_row_count(int(widget_task.id)) == 0

    share_auth = client.post("/api/share/auth", json={"share_token": agent.share_token})
    assert share_auth.status_code == 200, share_auth.text
    share_headers = {"Authorization": f"Bearer {share_auth.json()['access_token']}"}
    share_response = client.post(
        "/api/share/chat/task/create",
        headers=share_headers,
        json={
            "title": "share runtime",
            "description": "share",
            "agent_id": int(agent.id),
            "connector_runtime_context": _smuggled_payload(int(runtime_server.id)),
        },
    )
    assert share_response.status_code == 200, share_response.text
    share_task = _task(int(share_response.json()["task_id"]))
    assert share_task.connector_runtime_selected_refs == [
        {"connector_type": "mcp", "connector_id": int(runtime_server.id)}
    ]
    assert _context_row_count(int(share_task.id)) == 0


@pytest.mark.asyncio
async def test_agent_service_auto_create_fallback_snapshots_empty(
    e2e_db: None,
) -> None:
    _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        task_id = 987654
        await AgentServiceManager().get_agent_for_task(
            task_id=task_id,
            db=db,
            user=user,
        )
        task = (
            db.query(Task)
            .filter(Task.user_id == user.id, Task.title == f"Task {task_id}")
            .one_or_none()
        )
        assert task is not None
        assert task.connector_runtime_selected_refs == []
    finally:
        db.close()


@pytest.mark.asyncio
async def test_websocket_context_payload_does_not_persist_runtime_context(
    e2e_db: None,
) -> None:
    _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        task = Task(
            user_id=user.id,
            title="websocket smuggling",
            description="websocket smuggling",
            status=TaskStatus.PENDING,
            connector_runtime_selected_refs=[],
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)

        websocket = _FakeWebSocket()
        await handle_chat_message(
            websocket,  # type: ignore[arg-type]
            task_id,
            {
                "message": "hello",
                "context": {"connector_runtime_context": _smuggled_payload()},
                "user": user,
            },
        )
        assert _context_row_count(task_id) == 0
        db.refresh(task)
        assert task.connector_runtime_selected_refs == []
    finally:
        db.close()


@pytest.mark.asyncio
async def test_websocket_missing_task_auto_create_fallback_always_snapshots_empty(
    e2e_db: None,
) -> None:
    _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        missing_task_id = 246802

        websocket = _FakeWebSocket()
        await handle_chat_message(
            websocket,  # type: ignore[arg-type]
            missing_task_id,
            {
                "message": "hello from websocket",
                "user": user,
            },
        )

        task = (
            db.query(Task)
            .filter(
                Task.user_id == user.id,
                Task.title.like("Chat: hello from websocket%"),
            )
            .one_or_none()
        )
        assert task is not None
        assert task.connector_runtime_selected_refs == []
        assert _context_row_count(int(task.id)) == 0
    finally:
        db.close()


@pytest.mark.asyncio
async def test_feishu_new_task_fallback_snapshots_empty(
    e2e_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        channel = UserChannel(
            user_id=user.id,
            channel_type="feishu",
            channel_name="Feishu test",
            config={},
            is_active=True,
        )
        db.add(channel)
        db.commit()
        db.refresh(channel)

        monkeypatch.setattr(
            "xagent.web.channels.feishu.bot.get_agent_manager",
            lambda: _FakeAgentManager(),
        )

        bot = object.__new__(FeishuBotInstance)
        bot.channel_id = int(channel.id)
        bot.channel_name = "Feishu test"
        bot.active_tasks = {}
        bot.api_client = object()
        bot._save_active_tasks = lambda: None

        async def _send_text(_chat_id: str, _text: str) -> None:
            return None

        bot._send_text = _send_text

        message = SimpleNamespace(
            event=SimpleNamespace(
                message=SimpleNamespace(
                    chat_id="chat-1",
                    message_id="msg-1",
                    message_type="text",
                    content='{"text": "hello from feishu"}',
                )
            )
        )
        await bot._process_messages_batch("open-id-1", [message])

        task = (
            db.query(Task)
            .filter(Task.user_id == user.id, Task.title == "hello from feishu")
            .one_or_none()
        )
        assert task is not None
        assert task.connector_runtime_selected_refs == []
        assert _context_row_count(int(task.id)) == 0
    finally:
        db.close()


@pytest.mark.parametrize(
    ("execution_result", "expected_persisted_turns"),
    [
        ({"success": True, "output": "done"}, 1),
        (
            {
                "status": "interrupted",
                "success": False,
                "output": "ReActPattern interrupted.",
            },
            0,
        ),
    ],
    ids=["completed", "interrupted"],
)
@pytest.mark.asyncio
async def test_telegram_new_task_fallback_snapshots_empty(
    e2e_db: None,
    monkeypatch: pytest.MonkeyPatch,
    execution_result: dict[str, Any],
    expected_persisted_turns: int,
) -> None:
    _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        channel = UserChannel(
            user_id=user.id,
            channel_type="telegram",
            channel_name="Telegram test",
            config={},
            is_active=True,
        )
        db.add(channel)
        db.commit()
        db.refresh(channel)

        async def _restore_telegram_task_context(*_args: Any, **_kwargs: Any) -> None:
            return None

        monkeypatch.setattr(
            "xagent.web.channels.telegram.bot.restore_telegram_task_context",
            _restore_telegram_task_context,
        )
        monkeypatch.setattr(
            "xagent.web.channels.telegram.bot.persist_user_message",
            lambda **_kwargs: None,
        )

        agent_manager = _FakeAgentManager(execution_result)
        monkeypatch.setattr(
            "xagent.web.channels.telegram.bot.get_agent_manager",
            lambda: agent_manager,
        )

        bot = object.__new__(TelegramBotInstance)
        bot.channel_id = int(channel.id)
        bot.channel_name = "Telegram test"
        bot.active_tasks = {}
        bot.bot = object()
        bot.user_preparing_executions = set()
        bot.user_stop_events = {}
        bot.user_active_executions = {}
        bot._save_active_tasks = lambda: None
        bot._clear_user_stop_request = lambda _user_id: None
        bot._consume_user_stop_request = lambda _user_id: False

        async def _extract_message_content(_message: Any) -> tuple[str, list[Any]]:
            return "hello from telegram", []

        async def _await_execution(_user_id: int, execution, *, reason: str) -> dict:
            return await execution

        bot._extract_message_content = _extract_message_content
        bot._await_execution_with_stop_monitor = _await_execution

        class _LoadingMessage:
            message_id = 33

            async def edit_text(self, _text: str, **_kwargs: Any) -> None:
                pass

        class _TelegramMessage:
            from_user = SimpleNamespace(id=123)
            chat = SimpleNamespace(id=456)

            async def answer(self, _text: str, **_kwargs: Any) -> _LoadingMessage:
                return _LoadingMessage()

        await bot._process_user_messages_batch(123, [_TelegramMessage()])

        task = (
            db.query(Task)
            .filter(Task.user_id == user.id, Task.title == "hello from telegram")
            .one_or_none()
        )
        assert task is not None
        assert task.connector_runtime_selected_refs == []
        assert _context_row_count(int(task.id)) == 0
        persisted_turns = (
            db.query(TaskChatMessage)
            .filter(
                TaskChatMessage.task_id == task.id,
                TaskChatMessage.role == "assistant",
            )
            .count()
        )
        assert persisted_turns == expected_persisted_turns
    finally:
        db.close()


@pytest.mark.asyncio
async def test_telegram_voice_is_transcribed_as_prompt_and_kept_as_input_file(
    e2e_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        channel = UserChannel(
            user_id=user.id,
            channel_type="telegram",
            channel_name="Telegram voice test",
            config={},
            is_active=True,
        )
        db.add(channel)
        db.commit()
        db.refresh(channel)

        agent_manager = _FakeAgentManager()
        monkeypatch.setattr(
            "xagent.web.channels.telegram.bot.get_agent_manager",
            lambda: agent_manager,
        )

        async def _restore_telegram_task_context(*_args: Any, **_kwargs: Any) -> None:
            return None

        monkeypatch.setattr(
            "xagent.web.channels.telegram.bot.restore_telegram_task_context",
            _restore_telegram_task_context,
        )
        monkeypatch.setattr(
            "xagent.web.channels.telegram.bot.persist_telegram_assistant_turn",
            lambda **_kwargs: None,
        )

        class _FakeASR:
            def __init__(self) -> None:
                self.closed = False

            async def transcribe(self, *, audio: str, format: str | None = None) -> str:
                assert audio == "/workspace/input/voice.oga"
                assert format == "ogg"
                return "今晚有世界杯比赛吗？"

            async def aclose(self) -> None:
                self.closed = True

        asr_model = _FakeASR()
        voice = SimpleNamespace(file_id="telegram-voice-id")
        bot = object.__new__(TelegramBotInstance)
        bot.channel_id = int(channel.id)
        bot.channel_name = "Telegram voice test"
        bot.active_tasks = {}
        bot.bot = object()
        bot.user_preparing_executions = set()
        bot.user_stop_events = {}
        bot.user_active_executions = {}
        bot._save_active_tasks = lambda: None
        bot._clear_user_stop_request = lambda _user_id: None
        bot._consume_user_stop_request = lambda _user_id: False
        bot._resolve_voice_asr_model = lambda _db, _user: asr_model

        async def _extract_message_content(_message: Any) -> tuple[str, list[Any]]:
            return "", [voice]

        async def _download_and_register_files(**_kwargs: Any) -> list[dict[str, Any]]:
            return [
                {
                    "file_id": "workspace-file-id",
                    "telegram_file_id": "telegram-voice-id",
                    "name": "voice.oga",
                    "path": "/workspace/input/voice.oga",
                    "type": "audio/ogg",
                    "size": 123,
                }
            ]

        async def _await_execution(_user_id: int, execution, *, reason: str) -> dict:
            return await execution

        bot._extract_message_content = _extract_message_content
        bot._download_and_register_files = _download_and_register_files
        bot._await_execution_with_stop_monitor = _await_execution

        class _LoadingMessage:
            message_id = 33

            async def edit_text(self, _text: str, **_kwargs: Any) -> None:
                pass

        class _TelegramMessage:
            from_user = SimpleNamespace(id=123)
            chat = SimpleNamespace(id=456)

            def __init__(self, voice_input: Any) -> None:
                self.voice = voice_input

            async def answer(self, _text: str, **_kwargs: Any) -> _LoadingMessage:
                return _LoadingMessage()

        await bot._process_user_messages_batch(123, [_TelegramMessage(voice)])

        assert len(agent_manager.execute_calls) == 1
        execute_call = agent_manager.execute_calls[0]
        assert execute_call["task"].startswith("今晚有世界杯比赛吗？")
        assert "voice.oga: file_id=workspace-file-id" in execute_call["task"]
        assert execute_call["context"]["file_info"] == [
            {
                "file_id": "workspace-file-id",
                "telegram_file_id": "telegram-voice-id",
                "name": "voice.oga",
                "path": "/workspace/input/voice.oga",
                "type": "audio/ogg",
                "size": 123,
            }
        ]
        assert execute_call["context"]["uploaded_files"] == [
            "/workspace/input/voice.oga"
        ]
        expected_attachments = [
            {
                "file_id": "workspace-file-id",
                "name": "voice.oga",
                "size": 123,
                "type": "audio/ogg",
            }
        ]
        assert execute_call["context"]["files"] == expected_attachments
        assert execute_call["context"]["display_message"] == "今晚有世界杯比赛吗？"
        assert asr_model.closed is True

        task = (
            db.query(Task)
            .filter(Task.user_id == user.id, Task.title == "今晚有世界杯比赛吗？")
            .one_or_none()
        )
        assert task is not None
        user_message = (
            db.query(TaskChatMessage)
            .filter(
                TaskChatMessage.task_id == task.id,
                TaskChatMessage.role == "user",
            )
            .one()
        )
        assert user_message.content == "今晚有世界杯比赛吗？"
        assert user_message.attachments == expected_attachments
    finally:
        db.close()
