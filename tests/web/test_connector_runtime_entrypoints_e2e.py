from __future__ import annotations

import asyncio
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
from sqlalchemy import event
from sqlalchemy.orm import Session

from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from xagent.web.api.auth import auth_router, create_access_token
from xagent.web.api.chat import AgentServiceManager, chat_router
from xagent.web.api.public_chat_access import create_public_chat_access_token
from xagent.web.api.share import share_router
from xagent.web.api.websocket import handle_chat_message
from xagent.web.api.widget import widget_router
from xagent.web.channels.feishu.bot import FeishuBotInstance
from xagent.web.channels.telegram import bot as telegram_bot_module
from xagent.web.channels.telegram.bot import TelegramBotInstance
from xagent.web.models.agent import Agent, AgentOrigin, AgentStatus
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.custom_api import CustomApi, UserCustomApi
from xagent.web.models.database import (
    Base,
    get_db,
    get_engine,
    get_session_local,
)
from xagent.web.models.deployment import Deployment, DeploymentOwnerType
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.task import Task, TaskConnectorRuntimeContext, TaskStatus
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.models.user_channel import UserChannel
from xagent.web.models.workforce import Workforce
from xagent.web.services import connector_team_scope
from xagent.web.services.agent_team_scope import (
    AgentTeamScope,
    set_agent_team_scope_hook,
)


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


def _create_user(db: Session, username: str) -> User:
    user = User(username=username, password_hash="hash", is_admin=False)
    db.add(user)
    db.flush()
    return user


def _auth_headers_for_user(user: User) -> dict[str, str]:
    """Mint an access token for an already-created user, bypassing the
    HTTP login round trip. Drives the same ``get_current_user`` dependency
    every endpoint under test uses -- only the token minting is shortcut.
    """
    token = create_access_token(
        data={"sub": str(user.username), "user_id": int(user.id)}
    )
    return {"Authorization": f"Bearer {token}"}


def _mcp_server_with_context_schema(
    db: Session,
    user: User,
    *,
    name: str,
    context_schema: dict[str, Any],
    url: str = "https://example.com/mcp",
) -> MCPServer:
    server = MCPServer(
        name=name,
        description=f"{name} description",
        managed="external",
        transport="streamable_http",
        url=url,
        runtime_input_schema={"context": context_schema},
        runtime_bindings=[
            {
                "source": {"input_type": "context", "key": key},
                "target": {"target_type": "mcp_meta", "key": key},
            }
            for key in context_schema
        ],
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


class _TelegramVoiceMessage:
    from_user = SimpleNamespace(id=123)
    chat = SimpleNamespace(id=456)
    voice = SimpleNamespace(file_id="telegram-voice-id")

    def __init__(self) -> None:
        self.answers: list[str] = []

    async def answer(self, text: str, **_kwargs: Any) -> SimpleNamespace:
        self.answers.append(text)
        return SimpleNamespace(message_id=1)


def _telegram_voice_error_bot(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
    *,
    channel_name: str,
    asr_model: Any | None,
) -> tuple[TelegramBotInstance, _TelegramVoiceMessage]:
    user = _admin_user(db)
    channel = UserChannel(
        user_id=user.id,
        channel_type="telegram",
        channel_name=channel_name,
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

    voice = SimpleNamespace(file_id="telegram-voice-id")
    bot = object.__new__(TelegramBotInstance)
    bot.channel_id = int(channel.id)
    bot.channel_name = channel_name
    bot.active_tasks = {}
    bot.bot = object()
    bot.user_preparing_executions = set()
    bot.user_stop_events = {}
    bot.user_active_executions = {}
    bot.user_conversation_generations = {}
    bot.user_active_trace_handlers = {}
    bot.user_switch_locks = {}
    bot.selected_agents = {}
    bot._save_selected_agents = lambda: True
    bot._save_active_tasks = lambda: True
    bot._clear_user_stop_request = lambda _user_id: None
    bot._consume_user_stop_request = lambda _user_id: False
    bot._resolve_voice_asr_model_isolated = lambda _user_id: asr_model

    async def _extract_message_content(_message: Any) -> tuple[str, list[Any]]:
        return "", [voice]

    async def _download_and_register_files(**_kwargs: Any) -> list[dict[str, Any]]:
        return []

    bot._extract_message_content = _extract_message_content
    bot._download_and_register_files = _download_and_register_files
    return bot, _TelegramVoiceMessage()


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

        class _FakeASR:
            def __init__(self) -> None:
                self.closed = False

            async def aclose(self) -> None:
                self.closed = True

        asr_model = _FakeASR()
        bot, message = _telegram_voice_error_bot(
            db,
            monkeypatch,
            channel_name=f"Telegram voice {scenario}",
            asr_model=None if scenario == "no_asr" else asr_model,
        )

        await bot._process_user_messages_batch(123, [message])

        assert message.answers == [expected_message]
        if scenario == "missing_download":
            assert asr_model.closed is True
    finally:
        db.close()


@pytest.mark.asyncio
async def test_telegram_cancellation_during_voice_cleanup_closes_managed_lease(
    e2e_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_admin_headers()
    db = _db_session()
    managed_lease = None
    process_task: asyncio.Task[None] | None = None
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    try:
        original_prepare = telegram_bot_module.prepare_channel_task
        captured_leases: list[Any] = []

        async def _capture_prepared_task(**kwargs: Any) -> Any:
            prepared = await original_prepare(**kwargs)
            assert prepared is not None
            captured_leases.append(prepared.managed_lease)
            return prepared

        monkeypatch.setattr(
            telegram_bot_module,
            "prepare_channel_task",
            _capture_prepared_task,
        )

        class _BlockingASR:
            def __init__(self) -> None:
                self.closed = False

            async def aclose(self) -> None:
                close_started.set()
                await allow_close.wait()
                self.closed = True

        asr_model = _BlockingASR()
        bot, message = _telegram_voice_error_bot(
            db,
            monkeypatch,
            channel_name="Telegram voice cancellation",
            asr_model=asr_model,
        )

        process_task = asyncio.create_task(
            bot._process_user_messages_batch(123, [message])
        )
        await asyncio.wait_for(close_started.wait(), timeout=10)
        managed_lease = captured_leases[0]

        process_task.cancel()
        allow_close.set()
        with pytest.raises(asyncio.CancelledError):
            await process_task

        assert asr_model.closed is True
        assert managed_lease._closed is True
        assert 123 not in bot.user_preparing_executions
    finally:
        allow_close.set()
        try:
            if process_task is not None:
                if not process_task.done():
                    process_task.cancel()
                await asyncio.gather(process_task, return_exceptions=True)
        finally:
            try:
                if managed_lease is not None and not managed_lease._closed:
                    await managed_lease.close()
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

    def remove_handler(self, handler: Any) -> None:
        if handler in self.handlers:
            self.handlers.remove(handler)


class _FakeAgentService:
    def __init__(self) -> None:
        self.tracer = _FakeTracer()

    def set_execution_context_messages(self, _messages: list[Any]) -> None:
        pass

    def set_conversation_history(
        self, _messages: list[Any], *, watermark: int | None = None
    ) -> None:
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
        _db: Session | None = None,
        *,
        user: Any = None,
        **_kwargs: Any,
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


def test_web_chat_create_surfaces_typed_503_without_leaking_hook_message(
    e2e_db: None,
) -> None:
    """A team hook that raises while resolving the new task's connector
    selection snapshot must surface here as a typed 503, with the hook's
    raw message absent from the response body. Without the endpoint-side
    ``ConnectorRuntimeError`` mapping, ``create_task``'s blanket
    ``except Exception`` handler would return HTTP 500 with ``str(exc)`` as
    the response ``detail`` -- which for the already-wrapped error is the
    typed safe message, not the raw hook text, but with the wrong status.
    """
    from xagent.web.services import connector_team_scope

    headers = _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        agent = _create_agent(
            db, user, name="Raising Hook Web Chat Agent", tool_categories=["mcp"]
        )
        agent.team_id = 101
        db.commit()
        db.refresh(agent)
        agent_id = int(agent.id)
    finally:
        db.close()

    def _raising_hook(db: Session, *, team_id: int) -> dict[str, set[int]]:
        raise RuntimeError(
            "Bearer planted-hook-secret-must-not-leak: password authentication "
            "failed for 'svc'"
        )

    connector_team_scope.set_connector_team_hooks(team_visibility=_raising_hook)
    try:
        response = client.post(
            "/api/chat/task/create",
            headers=headers,
            json={
                "title": "raising hook web chat",
                "description": "create task",
                "agent_id": agent_id,
            },
        )
    finally:
        connector_team_scope.set_connector_team_hooks()

    assert response.status_code == 503, response.text
    assert "planted-hook-secret-must-not-leak" not in response.text


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
        bot._save_active_tasks = lambda: True

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


@pytest.mark.asyncio
async def test_feishu_existing_task_commits_registered_attachment_before_execution(
    e2e_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The channel must settle attachment writes before runtime Session handoff."""
    _setup_admin_headers()
    setup_db = _db_session()
    try:
        user = _admin_user(setup_db)
        channel = UserChannel(
            user_id=user.id,
            channel_type="feishu",
            channel_name="Feishu attachment test",
            config={},
            is_active=True,
        )
        task = Task(
            user_id=user.id,
            title="existing Feishu task",
            description="existing task",
            status=TaskStatus.PENDING,
            execution_mode="balanced",
            channel_name="Feishu attachment test",
            connector_runtime_selected_refs=[],
        )
        setup_db.add_all([channel, task])
        setup_db.commit()
        setup_db.refresh(channel)
        setup_db.refresh(task)
        channel_id = int(channel.id)
        task_id = int(task.id)
        user_id = int(user.id)
    finally:
        setup_db.close()

    class _BoundaryObservingAgentManager(_FakeAgentManager):
        caller_session_is_none: bool | None = None
        attachment_visible_at_entry: bool | None = None

        async def execute_task(self, **kwargs: Any) -> dict[str, Any]:
            self.caller_session_is_none = kwargs["db_session"] is None
            verification_db = _db_session()
            try:
                self.attachment_visible_at_entry = (
                    verification_db.query(UploadedFile)
                    .filter(UploadedFile.file_id == "feishu-existing-file")
                    .one_or_none()
                    is not None
                )
            finally:
                verification_db.close()
            return await super().execute_task(**kwargs)

    agent_manager = _BoundaryObservingAgentManager()
    monkeypatch.setattr(
        "xagent.web.channels.feishu.bot.get_agent_manager",
        lambda: agent_manager,
    )

    bot = object.__new__(FeishuBotInstance)
    bot.channel_id = channel_id
    bot.channel_name = "Feishu attachment test"
    bot.active_tasks = {"open-id-existing": str(task_id)}
    bot.api_client = object()
    bot._save_active_tasks = lambda: True

    async def _send_text(_chat_id: str, _text: str) -> None:
        return None

    async def _download_and_register_files(**kwargs: Any) -> list[dict[str, Any]]:
        assert "db" not in kwargs
        file_db = _db_session()
        try:
            file_db.add(
                UploadedFile(
                    file_id="feishu-existing-file",
                    user_id=user_id,
                    task_id=task_id,
                    filename="existing.txt",
                    storage_path="/tmp/feishu-existing.txt",
                    storage_status="pending",
                    mime_type="text/plain",
                    file_size=7,
                )
            )
            file_db.commit()
        finally:
            file_db.close()
        return [
            {
                "file_id": "feishu-existing-file",
                "name": "existing.txt",
                "path": "/tmp/feishu-existing.txt",
                "type": "text/plain",
                "size": 7,
            }
        ]

    bot._send_text = _send_text
    bot._download_and_register_files = _download_and_register_files

    message = SimpleNamespace(
        event=SimpleNamespace(
            message=SimpleNamespace(
                chat_id="chat-existing",
                message_id="msg-existing",
                message_type="file",
                content='{"file_key": "file-key-existing"}',
            )
        )
    )
    await bot._process_messages_batch("open-id-existing", [message])

    assert agent_manager.caller_session_is_none is True
    assert agent_manager.attachment_visible_at_entry is True
    assert len(agent_manager.execute_calls) == 1


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
        bot.user_conversation_generations = {}
        bot.user_active_trace_handlers = {}
        bot.user_switch_locks = {}
        bot.selected_agents = {}
        bot._save_selected_agents = lambda: True
        bot._save_active_tasks = lambda: True
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
        assert task.telegram_user_id == "123"
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

        async def _finalize_managed_result(*_args: Any, **_kwargs: Any) -> bool:
            return True

        monkeypatch.setattr(
            "xagent.web.services.managed_task_lease.ManagedTaskLease.finalize_result",
            _finalize_managed_result,
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
        bot.user_conversation_generations = {}
        bot.user_active_trace_handlers = {}
        bot.user_switch_locks = {}
        bot.selected_agents = {}
        bot._save_selected_agents = lambda: True
        bot._save_active_tasks = lambda: True
        bot._clear_user_stop_request = lambda _user_id: None
        bot._consume_user_stop_request = lambda _user_id: False
        bot._resolve_voice_asr_model_isolated = lambda _user_id: asr_model

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


# ---------------------------------------------------------------------------
# Connector-runtime-requirements read endpoints
# (GET /agent/{agent_id}/connector-runtime-requirements,
#  GET /task/{task_id}/connector-runtime-requirements).
# ---------------------------------------------------------------------------


def test_agent_requirements_hides_connection_config_and_normalizes_type(
    e2e_db: None,
) -> None:
    """The agent-keyed report never leaks a connector's transport or
    authentication configuration, a declared ``type`` other than the raw
    string ``"object"`` normalizes to ``"string"``, and the report is
    untouched by any task's stored values -- not even a task created
    against the same agent and connector.
    """
    headers = _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        server = MCPServer(
            name="leaky-server",
            description="leaky-server description",
            managed="external",
            transport="streamable_http",
            url="https://leak.example/probe",
            headers={"Authorization": "Bearer leak-header-secret"},
            env={"SECRET": "leak-env-secret"},
            auth={"type": "oauth", "client_secret": "leak-auth-secret"},
            runtime_input_schema={
                "context": {
                    "auth_token": {"type": "string", "required": True},
                    "profile": {"type": {"$ref": "leak"}, "required": False},
                }
            },
            runtime_bindings=[
                {
                    "source": {"input_type": "context", "key": "auth_token"},
                    "target": {"target_type": "mcp_meta", "key": "auth_token"},
                },
                {
                    "source": {"input_type": "context", "key": "profile"},
                    "target": {"target_type": "mcp_meta", "key": "profile"},
                },
            ],
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
        agent = _create_agent(
            db, user, name="Leaky Requirements Agent", tool_categories=["mcp"]
        )
        db.commit()
        db.refresh(agent)
        db.refresh(server)
        agent_id = int(agent.id)
        server_id = int(server.id)
    finally:
        db.close()

    response = client.get(
        f"/api/chat/agent/{agent_id}/connector-runtime-requirements",
        headers=headers,
    )
    assert response.status_code == 200, response.text
    for leaked in (
        "leak.example",
        "leak-header-secret",
        "leak-env-secret",
        "leak-auth-secret",
    ):
        assert leaked not in response.text

    payload = response.json()
    assert payload["secrets_expires_at"] is None
    connectors = payload["connectors"]
    assert len(connectors) == 1
    inputs_by_key = {item["key"]: item for item in connectors[0]["inputs"]}
    assert inputs_by_key["auth_token"]["type"] == "string"
    # Declared as {"$ref": "leak"}, not the literal string "object" -- must
    # normalize to "string", not pass through unnormalized.
    assert inputs_by_key["profile"]["type"] == "string"
    assert inputs_by_key["auth_token"]["satisfied"] is False
    assert inputs_by_key["auth_token"]["expired"] is False

    # A task created from this same agent, with this same connector's
    # required key filled directly in storage, must not move this report's
    # numbers: it has no task in scope.
    create_response = client.post(
        "/api/chat/task/create",
        headers=headers,
        json={
            "title": "leak isolation task",
            "description": "d",
            "agent_id": agent_id,
        },
    )
    assert create_response.status_code == 200, create_response.text
    task_id = int(create_response.json()["task_id"])
    db = _db_session()
    try:
        db.add(
            TaskConnectorRuntimeContext(
                task_id=task_id,
                connector_type="mcp",
                connector_id=server_id,
                context={"auth_token": "filled"},
            )
        )
        db.commit()
    finally:
        db.close()

    second_response = client.get(
        f"/api/chat/agent/{agent_id}/connector-runtime-requirements",
        headers=headers,
    )
    assert second_response.status_code == 200, second_response.text
    second_payload = second_response.json()
    assert second_payload["secrets_expires_at"] is None
    second_inputs = {
        item["key"]: item for item in second_payload["connectors"][0]["inputs"]
    }
    assert second_inputs["auth_token"]["satisfied"] is False


@pytest.mark.parametrize(
    "scenario",
    [
        "different_user",
        "admins_only_team",
        "unpublished_other_user",
        "workforce_manager_owned",
    ],
)
def test_agent_requirements_hides_non_visible_agents(
    e2e_db: None, scenario: str
) -> None:
    """Four identities that must all see a uniform 404, with the agent's
    name absent from the response body.
    """
    team_hook_installed = False
    db = _db_session()
    try:
        owner = _create_user(db, "agent-owner")
        caller = _create_user(db, "agent-caller")
        db.flush()
        caller_id = int(caller.id)

        if scenario == "different_user":
            agent = Agent(
                user_id=owner.id,
                name="Secret Agent",
                instructions="i",
                execution_mode="balanced",
                status=AgentStatus.DRAFT,
                tool_categories=[],
            )
        elif scenario == "admins_only_team":
            agent = Agent(
                user_id=owner.id,
                name="Secret Agent",
                instructions="i",
                execution_mode="balanced",
                status=AgentStatus.PUBLISHED,
                tool_categories=[],
                team_id=101,
                visibility="admins",
            )
            set_agent_team_scope_hook(
                lambda db, user_id: (
                    AgentTeamScope(team_id=101, is_team_admin=False)
                    if user_id == caller_id
                    else None
                )
            )
            team_hook_installed = True
        elif scenario == "unpublished_other_user":
            agent = Agent(
                user_id=owner.id,
                name="Secret Agent",
                instructions="i",
                execution_mode="balanced",
                status=AgentStatus.DRAFT,
                tool_categories=[],
            )
        elif scenario == "workforce_manager_owned":
            # Owned by the caller: proves the workforce-manager check runs
            # before -- not as part of -- the ownership check.
            agent = Agent(
                user_id=caller.id,
                name="Secret Agent",
                instructions="i",
                execution_mode="balanced",
                status=AgentStatus.PUBLISHED,
                tool_categories=[],
                origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
            )
        else:
            raise AssertionError(scenario)
        db.add(agent)
        db.commit()
        db.refresh(agent)
        agent_id = int(agent.id)
        caller_headers = _auth_headers_for_user(caller)
    finally:
        db.close()

    try:
        response = client.get(
            f"/api/chat/agent/{agent_id}/connector-runtime-requirements",
            headers=caller_headers,
        )
        assert response.status_code == 404, response.text
        assert "Secret Agent" not in response.text
    finally:
        if team_hook_installed:
            set_agent_team_scope_hook(None)


def test_custom_api_requirements_report_lists_context_only(e2e_db: None) -> None:
    """A custom_api connector's declared ``context`` key is listed, and its
    declared ``auth_selector`` key is not: the ``auth_selector`` section is
    only ever emitted for an MCP connector, so a non-MCP connector's
    declaration of it is never surfaced in a report.
    """
    headers = _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        api = CustomApi(
            name="custom-runtime-api",
            description="custom-runtime-api description",
            url="https://example.com/custom",
            method="GET",
            runtime_input_schema={
                "context": {"tenant_id": {"type": "string", "required": True}},
                "auth_selector": {"profile": {"type": "string", "required": False}},
            },
            runtime_bindings=[
                {
                    "source": {"input_type": "context", "key": "tenant_id"},
                    "target": {"target_type": "headers", "key": "X-Tenant-Id"},
                }
            ],
        )
        db.add(api)
        db.flush()
        api_id = int(api.id)
        db.add(
            UserCustomApi(
                user_id=user.id,
                custom_api_id=api_id,
                is_owner=True,
                can_edit=True,
                can_delete=True,
                is_active=True,
            )
        )
        agent = _create_agent(
            db,
            user,
            name="Custom API Agent",
            tool_categories=["mcp:custom-runtime-api"],
        )
        db.commit()
        db.refresh(agent)
        agent_id = int(agent.id)
    finally:
        db.close()

    response = client.get(
        f"/api/chat/agent/{agent_id}/connector-runtime-requirements",
        headers=headers,
    )
    assert response.status_code == 200, response.text
    assert "https://example.com/custom" not in response.text
    payload = response.json()
    assert len(payload["connectors"]) == 1
    connector = payload["connectors"][0]
    assert connector["connector_ref"] == {
        "connector_type": "custom_api",
        "connector_id": api_id,
    }
    section_keys = {(item["section"], item["key"]) for item in connector["inputs"]}
    assert section_keys == {("context", "tenant_id")}
    tenant_id_input = next(
        item for item in connector["inputs"] if item["key"] == "tenant_id"
    )
    assert tenant_id_input["satisfied"] is False
    assert tenant_id_input["expired"] is False
    assert tenant_id_input["required"] is True


def test_mcp_auth_selector_key_is_listed_as_its_own_section(e2e_db: None) -> None:
    """The mirror of the custom_api case above: on an MCP connector the
    declared ``auth_selector`` key is listed, under its own section, and is
    unsatisfied because no store holds such a value at this phase -- so a
    required one keeps the whole report unsatisfied. The key carries no
    runtime binding: binding an ``auth_selector`` key to a connector target
    is rejected at declaration time, unlike a ``context`` key.
    """
    headers = _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        server = MCPServer(
            name="auth-selector-server",
            description="auth-selector-server description",
            managed="external",
            transport="streamable_http",
            url="https://example.com/auth-selector/mcp",
            runtime_input_schema={
                "context": {"account_id": {"type": "string", "required": False}},
                "auth_selector": {
                    "resource_owner_key": {"type": "string", "required": True}
                },
            },
            runtime_bindings=[
                {
                    "source": {"input_type": "context", "key": "account_id"},
                    "target": {"target_type": "mcp_meta", "key": "account_id"},
                }
            ],
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
        agent = _create_agent(
            db, user, name="Auth Selector Agent", tool_categories=["mcp"]
        )
        db.commit()
        db.refresh(agent)
        agent_id = int(agent.id)
    finally:
        db.close()

    response = client.get(
        f"/api/chat/agent/{agent_id}/connector-runtime-requirements",
        headers=headers,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["connectors"]) == 1
    inputs = payload["connectors"][0]["inputs"]
    assert {(item["section"], item["key"]) for item in inputs} == {
        ("context", "account_id"),
        ("auth_selector", "resource_owner_key"),
    }
    auth_selector_input = next(
        item for item in inputs if item["section"] == "auth_selector"
    )
    assert auth_selector_input["key"] == "resource_owner_key"
    assert auth_selector_input["type"] == "string"
    assert auth_selector_input["required"] is True
    assert auth_selector_input["satisfied"] is False
    assert auth_selector_input["expired"] is False
    assert payload["satisfied"] is False


def test_agent_requirements_are_satisfied_when_every_declared_key_is_optional(
    e2e_db: None,
) -> None:
    """A non-vacuous top-level ``satisfied: true`` on the agent-keyed
    report: the agent does select a connector that declares a runtime
    input, the input is listed and unsatisfied -- that report can store no
    value -- and the flag still reads true, because it answers "would a
    task created from this agent right now need anything else" and nothing
    declared is required.
    """
    headers = _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        _create_mcp_server(
            db, user, name="optional-only-server", with_runtime_declaration=True
        )
        agent = _create_agent(
            db, user, name="Optional Only Agent", tool_categories=["mcp"]
        )
        db.commit()
        db.refresh(agent)
        agent_id = int(agent.id)
    finally:
        db.close()

    response = client.get(
        f"/api/chat/agent/{agent_id}/connector-runtime-requirements",
        headers=headers,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["connectors"]) == 1
    inputs = payload["connectors"][0]["inputs"]
    assert [item["key"] for item in inputs] == ["account_id"]
    assert inputs[0]["required"] is False
    assert inputs[0]["satisfied"] is False
    assert payload["satisfied"] is True


def test_team_shared_connector_visible_across_read_endpoints(e2e_db: None) -> None:
    """A connector shared only through the agent's team is listed by both
    read endpoints for a non-owning team member, and the task-keyed read
    endpoint returns 200
    (not 400) while the connector's one required key is still unfilled.
    Reversing the connector-team hook to withhold sharing removes it from
    both endpoints for the same caller and agent.
    """
    db = _db_session()
    try:
        owner = _create_user(db, "team-connector-owner")
        member = _create_user(db, "team-connector-member")
        db.flush()
        member_id = int(member.id)

        server = MCPServer(
            name="team-shared-server",
            description="team-shared-server description",
            managed="external",
            transport="streamable_http",
            url="https://example.com/mcp",
            runtime_input_schema={
                "context": {"auth_token": {"type": "string", "required": True}}
            },
            runtime_bindings=[
                {
                    "source": {"input_type": "context", "key": "auth_token"},
                    "target": {"target_type": "mcp_meta", "key": "auth_token"},
                }
            ],
        )
        db.add(server)
        db.flush()
        # No UserMCPServer link for `member` at all -- reachable only
        # through the team hook below.
        server_id = int(server.id)

        agent = Agent(
            user_id=owner.id,
            name="Team Shared Agent",
            instructions="i",
            execution_mode="balanced",
            status=AgentStatus.PUBLISHED,
            tool_categories=["mcp"],
            team_id=101,
            visibility="team",
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        agent_id = int(agent.id)
        member_headers = _auth_headers_for_user(member)
    finally:
        db.close()

    def _connector_ids_for_team(shared: bool):
        def _hook(db: Session, *, team_id: int) -> dict[str, set[int]]:
            if shared and team_id == 101:
                return {"mcp": {server_id}, "custom_api": set()}
            return {"mcp": set(), "custom_api": set()}

        return _hook

    set_agent_team_scope_hook(
        lambda db, user_id: (
            AgentTeamScope(team_id=101, is_team_admin=False)
            if user_id == member_id
            else None
        )
    )
    connector_team_scope.set_connector_team_hooks(
        team_visibility=_connector_ids_for_team(shared=True)
    )
    try:
        agent_response = client.get(
            f"/api/chat/agent/{agent_id}/connector-runtime-requirements",
            headers=member_headers,
        )
        assert agent_response.status_code == 200, agent_response.text
        agent_refs = [
            item["connector_ref"] for item in agent_response.json()["connectors"]
        ]
        assert {"connector_type": "mcp", "connector_id": server_id} in agent_refs

        create_response = client.post(
            "/api/chat/task/create",
            headers=member_headers,
            json={
                "title": "team shared task",
                "description": "d",
                "agent_id": agent_id,
            },
        )
        assert create_response.status_code == 200, create_response.text
        task_id = int(create_response.json()["task_id"])

        task_response = client.get(
            f"/api/chat/task/{task_id}/connector-runtime-requirements",
            headers=member_headers,
        )
        assert task_response.status_code == 200, task_response.text
        task_payload = task_response.json()
        assert task_payload["satisfied"] is False
        task_refs = [item["connector_ref"] for item in task_payload["connectors"]]
        assert {"connector_type": "mcp", "connector_id": server_id} in task_refs
    finally:
        set_agent_team_scope_hook(None)
        connector_team_scope.set_connector_team_hooks()

    # Reverse: withhold team sharing for the same agent/caller pair. Uses a
    # fresh task (the earlier one already persisted its selected refs) so
    # this is purely a visibility check on the read endpoints.
    set_agent_team_scope_hook(
        lambda db, user_id: (
            AgentTeamScope(team_id=101, is_team_admin=False)
            if user_id == member_id
            else None
        )
    )
    connector_team_scope.set_connector_team_hooks(
        team_visibility=_connector_ids_for_team(shared=False)
    )
    try:
        agent_response = client.get(
            f"/api/chat/agent/{agent_id}/connector-runtime-requirements",
            headers=member_headers,
        )
        assert agent_response.status_code == 200, agent_response.text
        assert agent_response.json()["connectors"] == []

        create_response = client.post(
            "/api/chat/task/create",
            headers=member_headers,
            json={
                "title": "team unshared task",
                "description": "d",
                "agent_id": agent_id,
            },
        )
        assert create_response.status_code == 200, create_response.text
        task_id = int(create_response.json()["task_id"])
        task_response = client.get(
            f"/api/chat/task/{task_id}/connector-runtime-requirements",
            headers=member_headers,
        )
        assert task_response.status_code == 200, task_response.text
        assert task_response.json()["connectors"] == []
    finally:
        set_agent_team_scope_hook(None)
        connector_team_scope.set_connector_team_hooks()


def test_agent_requirements_lists_connector_shared_only_through_agent_team(
    e2e_db: None,
) -> None:
    """A connector the caller can reach only through the agent's team --
    no personal link of their own -- is listed by the agent-keyed report.
    The team id the report resolves connectors under comes from the agent,
    not from the caller, so this is the case that would silently vanish if
    that scope were ever taken from the caller instead.
    """
    db = _db_session()
    try:
        owner = _create_user(db, "team-connector-owner-2")
        member = _create_user(db, "team-connector-member-2")
        db.flush()
        member_id = int(member.id)
        server = MCPServer(
            name="team-shared-server-2",
            description="team-shared-server-2 description",
            managed="external",
            transport="streamable_http",
            url="https://example.com/mcp",
            runtime_input_schema={
                "context": {"auth_token": {"type": "string", "required": True}}
            },
            runtime_bindings=[
                {
                    "source": {"input_type": "context", "key": "auth_token"},
                    "target": {"target_type": "mcp_meta", "key": "auth_token"},
                }
            ],
        )
        db.add(server)
        db.flush()
        server_id = int(server.id)
        agent = Agent(
            user_id=owner.id,
            name="Team Shared Agent 2",
            instructions="i",
            execution_mode="balanced",
            status=AgentStatus.PUBLISHED,
            tool_categories=["mcp"],
            team_id=202,
            visibility="team",
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        agent_id = int(agent.id)
        member_headers = _auth_headers_for_user(member)
    finally:
        db.close()

    set_agent_team_scope_hook(
        lambda db, user_id: (
            AgentTeamScope(team_id=202, is_team_admin=False)
            if user_id == member_id
            else None
        )
    )
    connector_team_scope.set_connector_team_hooks(
        team_visibility=lambda db, *, team_id: (
            {"mcp": {server_id}, "custom_api": set()}
            if team_id == 202
            else {"mcp": set(), "custom_api": set()}
        )
    )
    try:
        response = client.get(
            f"/api/chat/agent/{agent_id}/connector-runtime-requirements",
            headers=member_headers,
        )
        assert response.status_code == 200, response.text
        refs = [item["connector_ref"] for item in response.json()["connectors"]]
        assert {"connector_type": "mcp", "connector_id": server_id} in refs
    finally:
        set_agent_team_scope_hook(None)
        connector_team_scope.set_connector_team_hooks()


def test_task_requirements_reports_stored_context_key_as_satisfied(
    e2e_db: None,
) -> None:
    """A ``context`` key that already has a stored value is reported
    ``satisfied`` on the task-keyed read endpoint, and the top-level
    ``satisfied`` follows it when it is the connector's only required key.
    """
    headers = _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        server = _mcp_server_with_context_schema(
            db,
            user,
            name="satisfied-server",
            context_schema={"auth_token": {"type": "string", "required": True}},
        )
        agent = _create_agent(
            db, user, name="Satisfied Context Agent", tool_categories=["mcp"]
        )
        db.commit()
        db.refresh(agent)
        db.refresh(server)
        agent_id = int(agent.id)
        server_id = int(server.id)
    finally:
        db.close()

    create_response = client.post(
        "/api/chat/task/create",
        headers=headers,
        json={"title": "satisfied task", "description": "d", "agent_id": agent_id},
    )
    assert create_response.status_code == 200, create_response.text
    task_id = int(create_response.json()["task_id"])

    db = _db_session()
    try:
        db.add(
            TaskConnectorRuntimeContext(
                task_id=task_id,
                connector_type="mcp",
                connector_id=server_id,
                context={"auth_token": "stored"},
            )
        )
        db.commit()
    finally:
        db.close()

    response = client.get(
        f"/api/chat/task/{task_id}/connector-runtime-requirements",
        headers=headers,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    inputs_by_key = {
        item["key"]: item
        for connector in payload["connectors"]
        for item in connector["inputs"]
    }
    assert inputs_by_key["auth_token"]["satisfied"] is True
    assert inputs_by_key["auth_token"]["expired"] is False
    assert payload["satisfied"] is True


def test_task_requirements_uses_runtime_agent_resolution_for_team_scope(
    e2e_db: None,
) -> None:
    """A task whose agent is a workforce-generated manager agent, but for
    which no matching ``WorkforceRun`` exists, gets its connector scope
    from ``_load_agent_for_task_runtime`` returning ``None`` -- the same
    outcome a turn would get for this task -- so a connector reachable
    only through the agent's team is omitted while a personally linked
    connector is still reported.

    The task is built directly rather than through ``POST /task/create``:
    that path 404s for a workforce-generated manager agent
    (``_load_agent_for_task_create`` returns ``None`` for one), so a task
    with this agent can only exist by way of a row the workforce side
    creates directly, and a direct row is the only reachable construction
    for exercising the read endpoint against one.
    """
    db = _db_session()
    try:
        caller = _create_user(db, "wf-task-owner")
        db.flush()
        personal = _mcp_server_with_context_schema(
            db,
            caller,
            name="wf-personal-server",
            context_schema={"auth_token": {"type": "string", "required": True}},
            url="https://example.com/personal/mcp",
        )
        team_shared = MCPServer(
            name="wf-team-shared-server",
            description="wf-team-shared-server description",
            managed="external",
            transport="streamable_http",
            url="https://example.com/team/mcp",
            runtime_input_schema={
                "context": {"auth_token": {"type": "string", "required": True}}
            },
            runtime_bindings=[
                {
                    "source": {"input_type": "context", "key": "auth_token"},
                    "target": {"target_type": "mcp_meta", "key": "auth_token"},
                }
            ],
        )
        db.add(team_shared)
        db.flush()
        # No UserMCPServer link for `caller` at all -- reachable only
        # through the team hook below.
        personal_id = int(personal.id)
        team_shared_id = int(team_shared.id)
        agent = Agent(
            user_id=caller.id,
            name="WF Manager Agent",
            instructions="i",
            execution_mode="balanced",
            status=AgentStatus.PUBLISHED,
            tool_categories=["mcp"],
            team_id=303,
            origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
        )
        db.add(agent)
        db.flush()
        agent_id = int(agent.id)
        ordered_ids = sorted([personal_id, team_shared_id])
        task = Task(
            user_id=caller.id,
            agent_id=agent_id,
            title="workforce manager task",
            description="d",
            status=TaskStatus.PENDING,
            connector_runtime_selected_refs=[
                {"connector_type": "mcp", "connector_id": ref_id}
                for ref_id in ordered_ids
            ],
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
        caller_headers = _auth_headers_for_user(caller)
    finally:
        db.close()

    def _hook(db: Session, *, team_id: int) -> dict[str, set[int]]:
        if team_id == 303:
            return {"mcp": {team_shared_id}, "custom_api": set()}
        return {"mcp": set(), "custom_api": set()}

    connector_team_scope.set_connector_team_hooks(team_visibility=_hook)
    try:
        response = client.get(
            f"/api/chat/task/{task_id}/connector-runtime-requirements",
            headers=caller_headers,
        )
    finally:
        connector_team_scope.set_connector_team_hooks()

    assert response.status_code == 200, response.text
    payload = response.json()
    refs_seen = [item["connector_ref"] for item in payload["connectors"]]
    assert refs_seen == [{"connector_type": "mcp", "connector_id": personal_id}]
    auth_token_input = next(
        item
        for item in payload["connectors"][0]["inputs"]
        if item["key"] == "auth_token"
    )
    assert auth_token_input["satisfied"] is False


def test_task_requirements_reports_unfillable_declared_key_as_unsatisfied(
    e2e_db: None,
) -> None:
    """A declared ``context`` key whose name the per-turn gate rejects as
    malformed is still listed in the report, but is reported unsatisfied
    unconditionally -- even with a stored value under that name -- so the
    top-level ``satisfied`` stays false while a required key of that kind
    is declared. The report itself never raises on this key; only the
    per-turn gate does.
    """
    headers = _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        server = _mcp_server_with_context_schema(
            db,
            user,
            name="malformed-key-server",
            context_schema={
                "auth_token": {"type": "string", "required": True},
                "bad.key": {"type": "string", "required": True},
            },
            url="https://example.com/malformed/mcp",
        )
        agent = _create_agent(
            db, user, name="Malformed Key Agent", tool_categories=["mcp"]
        )
        db.commit()
        db.refresh(agent)
        db.refresh(server)
        agent_id = int(agent.id)
        server_id = int(server.id)
    finally:
        db.close()

    create_response = client.post(
        "/api/chat/task/create",
        headers=headers,
        json={
            "title": "malformed key task",
            "description": "d",
            "agent_id": agent_id,
        },
    )
    assert create_response.status_code == 200, create_response.text
    task_id = int(create_response.json()["task_id"])

    db = _db_session()
    try:
        db.add(
            TaskConnectorRuntimeContext(
                task_id=task_id,
                connector_type="mcp",
                connector_id=server_id,
                context={"auth_token": "stored", "bad.key": "stored"},
            )
        )
        db.commit()
    finally:
        db.close()

    response = client.get(
        f"/api/chat/task/{task_id}/connector-runtime-requirements",
        headers=headers,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    inputs_by_key = {
        item["key"]: item
        for connector in payload["connectors"]
        for item in connector["inputs"]
    }
    assert set(inputs_by_key) == {"auth_token", "bad.key"}
    assert inputs_by_key["bad.key"]["satisfied"] is False
    assert inputs_by_key["auth_token"]["satisfied"] is True
    assert payload["satisfied"] is False


def test_optional_unfillable_declared_key_holds_report_unsatisfied(
    e2e_db: None,
) -> None:
    """A malformed declared key holds the top-level ``satisfied`` at false
    even when nothing the connector declares is required. The per-turn gate
    validates the syntax of every declared key before it looks at
    ``required``, so this task cannot run; a report that read only the
    ``required`` keys would call it ready. The key stays listed, its own
    ``satisfied`` is false, and the read still answers 200 -- reporting the
    problem is this endpoint's job, raising on it is the gate's.
    """
    headers = _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        server = _mcp_server_with_context_schema(
            db,
            user,
            name="optional-malformed-key-server",
            context_schema={
                "account_id": {"type": "string", "required": False},
                "bad.key": {"type": "string", "required": False},
            },
            url="https://example.com/optional-malformed/mcp",
        )
        agent = _create_agent(
            db, user, name="Optional Malformed Key Agent", tool_categories=["mcp"]
        )
        db.commit()
        db.refresh(agent)
        db.refresh(server)
        agent_id = int(agent.id)
        server_id = int(server.id)
    finally:
        db.close()

    create_response = client.post(
        "/api/chat/task/create",
        headers=headers,
        json={
            "title": "optional malformed key task",
            "description": "d",
            "agent_id": agent_id,
        },
    )
    assert create_response.status_code == 200, create_response.text
    task_id = int(create_response.json()["task_id"])

    db = _db_session()
    try:
        db.add(
            TaskConnectorRuntimeContext(
                task_id=task_id,
                connector_type="mcp",
                connector_id=server_id,
                context={"account_id": "stored"},
            )
        )
        db.commit()
    finally:
        db.close()

    response = client.get(
        f"/api/chat/task/{task_id}/connector-runtime-requirements",
        headers=headers,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    inputs_by_key = {
        item["key"]: item
        for connector in payload["connectors"]
        for item in connector["inputs"]
    }
    assert set(inputs_by_key) == {"account_id", "bad.key"}
    assert inputs_by_key["bad.key"]["required"] is False
    assert inputs_by_key["bad.key"]["satisfied"] is False
    assert inputs_by_key["account_id"]["required"] is False
    assert inputs_by_key["account_id"]["satisfied"] is True
    assert payload["satisfied"] is False


def test_task_requirements_reports_empty_for_a_task_with_no_agent(
    e2e_db: None,
) -> None:
    """A task created with no agent has no connector selection to report:
    the agent-keyed producer that fills the task's selection snapshot was
    called with no agent, so the snapshot is empty, and the task-keyed read
    resolves the same absent agent and answers 200 with an empty,
    satisfied report rather than failing on the missing row.
    """
    headers = _setup_admin_headers()
    create_response = client.post(
        "/api/chat/task/create",
        headers=headers,
        json={"title": "agentless task", "description": "d"},
    )
    assert create_response.status_code == 200, create_response.text
    task_id = int(create_response.json()["task_id"])
    assert _task(task_id).agent_id is None

    response = client.get(
        f"/api/chat/task/{task_id}/connector-runtime-requirements",
        headers=headers,
    )
    assert response.status_code == 200, response.text
    assert response.json() == {
        "satisfied": True,
        "secrets_expires_at": None,
        "connectors": [],
    }


@pytest.mark.parametrize(
    ("endpoint_kind", "expected_detail"),
    [
        ("agent", "Agent not found or access denied"),
        ("task", "Task not found"),
    ],
)
def test_read_endpoints_report_an_unknown_id_as_not_found(
    e2e_db: None, endpoint_kind: str, expected_detail: str
) -> None:
    """An id that exists on neither endpoint is a 404 on both -- the same
    answer each gives for an id the caller may not read, so neither
    endpoint tells an unauthorized caller that the id exists.
    """
    headers = _setup_admin_headers()
    response = client.get(
        f"/api/chat/{endpoint_kind}/987654321/connector-runtime-requirements",
        headers=headers,
    )
    assert response.status_code == 404, response.text
    assert response.json()["detail"] == expected_detail


def test_task_requirements_endpoint_gives_an_admin_no_read_of_another_task(
    e2e_db: None,
) -> None:
    """Being an admin buys no read of another user's task here: the
    task-keyed read filters on the caller's own user id with no admin
    exception, unlike ``/task/{task_id}/runtime-extensions``. The owner's
    own read of the same task answers 200, so the admin's 404 is about the
    caller, not about the id.
    """
    admin_headers = _setup_admin_headers()
    db = _db_session()
    try:
        owner = _create_user(db, "task-requirements-owner")
        agent = _create_agent(
            db, owner, name="Owner Only Agent", tool_categories=["mcp"]
        )
        db.commit()
        db.refresh(agent)
        agent_id = int(agent.id)
        owner_headers = _auth_headers_for_user(owner)
    finally:
        db.close()

    create_response = client.post(
        "/api/chat/task/create",
        headers=owner_headers,
        json={"title": "owner task", "description": "d", "agent_id": agent_id},
    )
    assert create_response.status_code == 200, create_response.text
    task_id = int(create_response.json()["task_id"])

    url = f"/api/chat/task/{task_id}/connector-runtime-requirements"
    owner_response = client.get(url, headers=owner_headers)
    assert owner_response.status_code == 200, owner_response.text

    admin_response = client.get(url, headers=admin_headers)
    assert admin_response.status_code == 404, admin_response.text
    assert admin_response.json()["detail"] == "Task not found"


def test_task_requirements_endpoint_requires_task_ownership_by_caller(
    e2e_db: None,
) -> None:
    """A task belonging to another logged-in user is a uniform 404 on the
    task-keyed read endpoint, per plain task ownership with no admin
    exception.
    """
    headers = _setup_admin_headers()
    db = _db_session()
    try:
        owner = _admin_user(db)
        other = _create_user(db, "task-requirements-other")
        agent = _create_agent(
            db, owner, name="Task Owner Agent", tool_categories=["mcp"]
        )
        db.commit()
        db.refresh(agent)
        agent_id = int(agent.id)
        other_headers = _auth_headers_for_user(other)
    finally:
        db.close()

    create_response = client.post(
        "/api/chat/task/create",
        headers=headers,
        json={"title": "owner only task", "description": "d", "agent_id": agent_id},
    )
    assert create_response.status_code == 200, create_response.text
    task_id = int(create_response.json()["task_id"])

    response = client.get(
        f"/api/chat/task/{task_id}/connector-runtime-requirements",
        headers=other_headers,
    )
    assert response.status_code == 404, response.text


@pytest.mark.parametrize("endpoint_kind", ["agent", "task"])
def test_read_endpoints_reject_anonymous_and_widget_credentials(
    e2e_db: None, endpoint_kind: str
) -> None:
    """No ``Authorization`` header is a bare 403
    (``HTTPBearer`` itself, ``auto_error=True``), and a well-formed widget
    guest token is a 401 ``"Invalid token type"`` -- the same two doors
    every other authenticated-only endpoint in this module is gated by
    (``get_current_user``'s ``type: "access"`` check, ``auth_dependencies.py``).
    """
    headers = _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        agent = _create_agent(
            db, user, name="Anon Guard Agent", tool_categories=["mcp"]
        )
        db.commit()
        db.refresh(agent)
        agent_id = int(agent.id)
    finally:
        db.close()

    if endpoint_kind == "agent":
        url = f"/api/chat/agent/{agent_id}/connector-runtime-requirements"
    else:
        create_response = client.post(
            "/api/chat/task/create",
            headers=headers,
            json={"title": "anon guard task", "description": "d", "agent_id": agent_id},
        )
        assert create_response.status_code == 200, create_response.text
        task_id = int(create_response.json()["task_id"])
        url = f"/api/chat/task/{task_id}/connector-runtime-requirements"

    no_auth_response = client.get(url)
    assert no_auth_response.status_code == 403, no_auth_response.text

    widget_token = create_public_chat_access_token(
        {"guest_id": "anon-guard-guest", "widget_agent_id": agent_id}
    )
    widget_response = client.get(
        url, headers={"Authorization": f"Bearer {widget_token}"}
    )
    assert widget_response.status_code == 401, widget_response.text
    assert widget_response.json()["detail"] == "Invalid token type"


# ---------------------------------------------------------------------------
# The connector_runtime_requirements field on the task-create response.
# ---------------------------------------------------------------------------


def test_create_task_reports_missing_context_requirement(e2e_db: None) -> None:
    """An agent with an unmet required ``context`` key reports it on the
    create response without writing anything, and the task starts PENDING.
    """
    headers = _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        server = _mcp_server_with_context_schema(
            db,
            user,
            name="a3-context-server",
            context_schema={"auth_token": {"type": "string", "required": True}},
        )
        agent = _create_agent(
            db, user, name="Context Requirement Agent", tool_categories=["mcp"]
        )
        db.commit()
        db.refresh(agent)
        db.refresh(server)
        agent_id = int(agent.id)
    finally:
        db.close()

    response = client.post(
        "/api/chat/task/create",
        headers=headers,
        json={"title": "a3 context task", "description": "d", "agent_id": agent_id},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    requirements = payload["connector_runtime_requirements"]
    assert requirements["satisfied"] is False
    keys = {
        item["key"]
        for connector in requirements["connectors"]
        for item in connector["inputs"]
    }
    assert "auth_token" in keys
    task_id = int(payload["task_id"])
    assert _context_row_count(task_id) == 0
    assert _task(task_id).status == TaskStatus.PENDING


def test_create_task_reports_missing_secret_requirement_without_reading_any_column(
    e2e_db: None,
) -> None:
    """A required ``secrets`` key makes the top-level ``satisfied`` false
    purely from the phase-2 constant, with no secret store or column read
    anywhere in this phase.
    """
    headers = _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        server = MCPServer(
            name="a3-secret-server",
            description="a3-secret-server description",
            managed="external",
            transport="streamable_http",
            url="https://example.com/mcp",
            runtime_input_schema={
                "secrets": {"authorization": {"type": "string", "required": True}}
            },
            runtime_bindings=[
                {
                    "source": {"input_type": "secrets", "key": "authorization"},
                    "target": {
                        "target_type": "transport_headers",
                        "key": "Authorization",
                    },
                }
            ],
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
        agent = _create_agent(
            db, user, name="Secret Requirement Agent", tool_categories=["mcp"]
        )
        db.commit()
        db.refresh(agent)
        agent_id = int(agent.id)
    finally:
        db.close()

    response = client.post(
        "/api/chat/task/create",
        headers=headers,
        json={"title": "a3 secret task", "description": "d", "agent_id": agent_id},
    )
    assert response.status_code == 200, response.text
    requirements = response.json()["connector_runtime_requirements"]
    assert requirements["satisfied"] is False
    secret_input = next(
        item
        for connector in requirements["connectors"]
        for item in connector["inputs"]
        if item["section"] == "secrets"
    )
    assert secret_input["satisfied"] is False
    assert requirements["secrets_expires_at"] is None


def test_create_task_reports_empty_requirements_when_nothing_is_declared(
    e2e_db: None,
) -> None:
    """On the logged-in web chat create path, the field always appears, and
    with no declared connectors it is the empty, always-satisfied report --
    never absent, never ``null`` (``null`` is reserved for the public/share
    paths, which never evaluate this at all; see the public-path tests
    below).
    """
    headers = _setup_admin_headers()
    response = client.post(
        "/api/chat/task/create",
        headers=headers,
        json={"title": "a3 empty task", "description": "d"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "connector_runtime_requirements" in payload
    assert payload["connector_runtime_requirements"] == {
        "satisfied": True,
        "secrets_expires_at": None,
        "connectors": [],
    }


def _create_workforce_with_deployment(
    db: Session,
    user: User,
    *,
    name: str,
    widget_enabled: bool = False,
    share_enabled: bool = False,
) -> tuple[Workforce, Deployment]:
    """A minimal published workforce with a deployment row, for the two
    workforce-backed public create paths (widget and share). The workforce's
    own manager agent is a bystander here -- only its FK needs to resolve --
    so it is created with no runtime declarations of its own."""
    manager = _create_agent(db, user, name=f"{name} Manager", tool_categories=[])
    workforce = Workforce(
        owner_user_id=user.id,
        scope_type="user",
        scope_id=str(user.id),
        name=name,
        manager_agent_id=manager.id,
        status="active",
    )
    db.add(workforce)
    db.flush()
    deployment = Deployment(
        owner_type=DeploymentOwnerType.WORKFORCE.value,
        owner_id=workforce.id,
        widget_enabled=widget_enabled,
        widget_key=f"wfwk-{secrets.token_urlsafe(24)}" if widget_enabled else None,
        share_enabled=share_enabled,
        share_token=f"wfst-{secrets.token_urlsafe(24)}" if share_enabled else None,
    )
    db.add(deployment)
    db.flush()
    return workforce, deployment


@pytest.mark.parametrize(
    "producer",
    ["widget_agent", "workforce_widget", "share_agent", "workforce_share"],
)
def test_public_create_paths_all_report_null_requirements(
    e2e_db: None, monkeypatch: pytest.MonkeyPatch, producer: str
) -> None:
    """Every ``TaskCreateResponse`` producer in ``public_chat_access.py``
    that serves an anonymous widget or share guest sets
    ``connector_runtime_requirements`` to ``None``, never a real report: the
    widget-agent, workforce-widget, share-agent and workforce-share paths
    are four separate call sites with four separate explicit ``None``
    literals, all guarding the same decision that a guest never sees a
    connector's declared key names.

    The two workforce producers are reached through the real auth and route
    layers; only ``create_workforce_run`` -- the heavy collaborator that
    snapshots agent config and starts the first turn -- is stubbed to return
    an already-created task, which is all a response-shape assertion needs.
    """
    is_widget = producer in ("widget_agent", "workforce_widget")
    is_workforce = producer in ("workforce_widget", "workforce_share")

    _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        if not is_workforce:
            owner_agent = _create_agent(
                db,
                user,
                name=f"Null {producer} Agent",
                tool_categories=["mcp"],
                widget_enabled=is_widget,
                share_enabled=not is_widget,
                share_token=(
                    None if is_widget else f"null-share-{secrets.token_urlsafe(16)}"
                ),
            )
            db.commit()
            db.refresh(owner_agent)
            credential = (
                owner_agent.widget_key if is_widget else owner_agent.share_token
            )
        else:
            _workforce, deployment = _create_workforce_with_deployment(
                db,
                user,
                name=f"Null {producer} Workforce",
                widget_enabled=is_widget,
                share_enabled=not is_widget,
            )
            db.commit()
            db.refresh(deployment)
            credential = deployment.widget_key if is_widget else deployment.share_token

            stub_task = Task(
                user_id=user.id,
                title="stub workforce task",
                status=TaskStatus.PENDING,
                source="widget" if is_widget else "shared_link",
            )
            db.add(stub_task)
            db.commit()
            db.refresh(stub_task)

            from xagent.web.api import public_chat_access as public_chat_access_module

            async def _fake_create_workforce_run(*_args: Any, **_kwargs: Any) -> Any:
                return SimpleNamespace(task=stub_task)

            monkeypatch.setattr(
                public_chat_access_module,
                "create_workforce_run",
                _fake_create_workforce_run,
            )
    finally:
        db.close()

    if is_widget:
        auth_response = client.post(
            "/api/widget/auth",
            json={"guest_id": f"null-{producer}-guest", "widget_key": credential},
        )
        assert auth_response.status_code == 200, auth_response.text
        guest_token = auth_response.json()["access_token"]
        create_response = client.post(
            "/api/widget/chat/task/create",
            headers={"Authorization": f"Bearer {guest_token}"},
            json={"title": f"null {producer} task", "description": "d"},
        )
    else:
        auth_response = client.post("/api/share/auth", json={"share_token": credential})
        assert auth_response.status_code == 200, auth_response.text
        guest_token = auth_response.json()["access_token"]
        create_response = client.post(
            "/api/share/chat/task/create",
            headers={"Authorization": f"Bearer {guest_token}"},
            json={"title": f"null {producer} task", "description": "d"},
        )

    assert create_response.status_code == 200, create_response.text
    payload = create_response.json()
    assert "connector_runtime_requirements" in payload
    assert payload["connector_runtime_requirements"] is None


def test_create_task_calls_connector_resolution_exactly_once(
    e2e_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The create response's requirements report costs no extra query --
    ``resolve_agent_selected_connectors`` is called exactly once per task
    creation, not once for the snapshot and again for the report.
    """
    headers = _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        server = _mcp_server_with_context_schema(
            db,
            user,
            name="a3-call-count-server",
            context_schema={"auth_token": {"type": "string", "required": False}},
        )
        agent = _create_agent(
            db, user, name="Call Count Agent", tool_categories=["mcp"]
        )
        db.commit()
        db.refresh(agent)
        db.refresh(server)
        agent_id = int(agent.id)
    finally:
        db.close()

    from xagent.web.services import connector_runtime as connector_runtime_service

    original = connector_runtime_service.resolve_agent_selected_connectors
    calls: list[int] = []

    def _counting_resolver(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        connector_runtime_service,
        "resolve_agent_selected_connectors",
        _counting_resolver,
    )

    response = client.post(
        "/api/chat/task/create",
        headers=headers,
        json={
            "title": "a3 call count task",
            "description": "d",
            "agent_id": agent_id,
        },
    )
    assert response.status_code == 200, response.text
    assert len(calls) == 1


def test_create_task_persists_same_selected_refs_as_legacy_snapshot(
    e2e_db: None,
) -> None:
    """The persisted ``Task.connector_runtime_selected_refs`` column -- which
    the per-turn gate and ``load_connector_runtime_view`` both read -- is
    unchanged in content and order by task creation's switch to
    ``resolve_agent_runtime_requirements``.
    Compared against the legacy ``prepare_connector_runtime_selection_snapshot``
    (the column's pre-existing source of truth) on the same agent, with
    list equality (not set equality) so a reordering would fail this too.
    """
    from xagent.web.services.connector_runtime import (
        prepare_connector_runtime_selection_snapshot,
    )

    headers = _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        declared_one = _mcp_server_with_context_schema(
            db,
            user,
            name="a3-refs-declared-1",
            context_schema={"account_id": {"type": "string", "required": False}},
        )
        declared_two = _mcp_server_with_context_schema(
            db,
            user,
            name="a3-refs-declared-2",
            context_schema={"account_id": {"type": "string", "required": False}},
        )
        undeclared = _create_mcp_server(
            db, user, name="a3-refs-undeclared", with_runtime_declaration=False
        )
        agent = _create_agent(
            db, user, name="Refs Order Agent", tool_categories=["mcp"]
        )
        db.commit()
        db.refresh(agent)
        db.refresh(declared_one)
        db.refresh(declared_two)
        db.refresh(undeclared)
        agent_id = int(agent.id)
        agent_row = db.query(Agent).filter(Agent.id == agent_id).one()
        expected_refs = list(
            prepare_connector_runtime_selection_snapshot(
                db=db, agent=agent_row, connector_user_id=int(user.id)
            )
        )
    finally:
        db.close()

    response = client.post(
        "/api/chat/task/create",
        headers=headers,
        json={
            "title": "a3 refs order task",
            "description": "d",
            "agent_id": agent_id,
        },
    )
    assert response.status_code == 200, response.text
    task_id = int(response.json()["task_id"])
    persisted_refs = _task(task_id).connector_runtime_selected_refs
    expected_wire = [ref.to_wire() for ref in expected_refs]
    assert persisted_refs == expected_wire


# ---------------------------------------------------------------------------
# POST /task/{task_id}/connector-runtime-values.
# ---------------------------------------------------------------------------


def _values_url(task_id: int) -> str:
    return f"/api/chat/task/{task_id}/connector-runtime-values"


def _setup_context_task(
    *,
    required: bool = True,
    key_type: str = "string",
    server_name: str = "a4-server",
) -> tuple[dict[str, str], int, int]:
    """Owner-only setup shared by most of the values-endpoint tests below:
    one MCP connector with a single declared ``context`` key, one task that
    selected it. Returns (owner headers, task_id, server_id).
    """
    headers = _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        server = _mcp_server_with_context_schema(
            db,
            user,
            name=server_name,
            context_schema={"auth_token": {"type": key_type, "required": required}},
        )
        agent = _create_agent(
            db, user, name=f"{server_name}-agent", tool_categories=["mcp"]
        )
        db.commit()
        db.refresh(agent)
        db.refresh(server)
        agent_id = int(agent.id)
        server_id = int(server.id)
    finally:
        db.close()

    create_response = client.post(
        "/api/chat/task/create",
        headers=headers,
        json={"title": server_name, "description": "d", "agent_id": agent_id},
    )
    assert create_response.status_code == 200, create_response.text
    task_id = int(create_response.json()["task_id"])
    return headers, task_id, server_id


def test_values_endpoint_requires_task_ownership_by_caller(e2e_db: None) -> None:
    """Three identities that must all see a uniform 404, matching the read
    endpoints' ownership predicate -- including a non-owner whose request
    body is itself oversized, so that the ownership check cannot be
    inferred to run after the size check from a caller watching only the
    status code (a non-owner would then get 400 for a huge body and 404 for
    a small one, which leaks whether the task exists)."""
    _, task_id, server_id = _setup_context_task()
    db = _db_session()
    try:
        other = _create_user(db, "values-other-user")
        db.commit()
        other_headers = _auth_headers_for_user(other)
    finally:
        db.close()

    body = {
        "items": [
            {
                "connector_ref": {"connector_type": "mcp", "connector_id": server_id},
                "context": {"auth_token": "x"},
            }
        ]
    }

    other_response = client.post(_values_url(task_id), headers=other_headers, json=body)
    assert other_response.status_code == 404, other_response.text

    oversized_body = {
        "items": [
            {
                "connector_ref": {"connector_type": "mcp", "connector_id": server_id},
                "context": {"auth_token": "x" * (64 * 1024 + 1)},
            }
        ]
    }
    oversized_other_response = client.post(
        _values_url(task_id), headers=other_headers, json=oversized_body
    )
    assert oversized_other_response.status_code == 404, oversized_other_response.text

    anon_response = client.post(_values_url(task_id), json=body)
    assert anon_response.status_code == 403, anon_response.text

    widget_token = create_public_chat_access_token(
        {"guest_id": "values-anon-guest", "widget_agent_id": 1}
    )
    widget_response = client.post(
        _values_url(task_id),
        headers={"Authorization": f"Bearer {widget_token}"},
        json=body,
    )
    assert widget_response.status_code == 401, widget_response.text
    assert _context_row_count(task_id) == 0


def test_values_endpoint_gives_an_admin_no_write_to_another_task(
    e2e_db: None,
) -> None:
    """Being an admin buys no write to another user's task here either:
    this endpoint filters on the caller's own user id with no admin
    exception, and answers the same "Task not found" 404 the task-keyed
    read endpoint answers for the same caller. The owner's own write of
    the same body answers 200, so the admin's 404 is about the caller, not
    about the id or the body.
    """
    admin_headers = _setup_admin_headers()
    db = _db_session()
    try:
        owner = _create_user(db, "values-admin-probe-owner")
        db.flush()
        server = _mcp_server_with_context_schema(
            db,
            owner,
            name="values-admin-probe-server",
            context_schema={"auth_token": {"type": "string", "required": True}},
            url="https://example.com/values-admin-probe/mcp",
        )
        agent = _create_agent(
            db, owner, name="Values Admin Probe Agent", tool_categories=["mcp"]
        )
        db.commit()
        db.refresh(agent)
        db.refresh(server)
        agent_id = int(agent.id)
        server_id = int(server.id)
        owner_headers = _auth_headers_for_user(owner)
    finally:
        db.close()

    create_response = client.post(
        "/api/chat/task/create",
        headers=owner_headers,
        json={
            "title": "values admin probe task",
            "description": "d",
            "agent_id": agent_id,
        },
    )
    assert create_response.status_code == 200, create_response.text
    task_id = int(create_response.json()["task_id"])

    body = {
        "items": [
            {
                "connector_ref": {"connector_type": "mcp", "connector_id": server_id},
                "context": {"auth_token": "x"},
            }
        ]
    }

    admin_response = client.post(_values_url(task_id), headers=admin_headers, json=body)
    assert admin_response.status_code == 404, admin_response.text
    assert admin_response.json()["detail"] == "Task not found"
    assert _context_row_count(task_id) == 0

    owner_response = client.post(_values_url(task_id), headers=owner_headers, json=body)
    assert owner_response.status_code == 200, owner_response.text
    assert _context_row_count(task_id) == 1


def test_values_endpoint_surfaces_team_scope_failure_as_typed_503(
    e2e_db: None,
) -> None:
    """A team-scope resolution failure surfaces as 503 with
    ``reason == "team_scope_resolution_failed"``, not a 400 -- setup must
    install the team hook first (``_load_visible_runtime_connectors`` only
    calls ``resolve_team_connector_ids_or_raise``, which is what raises
    this, when the hook is installed; without it the branch that produces
    this 503 is unreachable and the assertion below would be probing a
    no-op setup instead of the endpoint).
    """
    headers = _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        agent = _create_agent(
            db, user, name="Team Scope Failure Agent", tool_categories=["mcp"]
        )
        agent.team_id = 303
        db.commit()
        db.refresh(agent)
        agent_id = int(agent.id)
    finally:
        db.close()

    create_response = client.post(
        "/api/chat/task/create",
        headers=headers,
        json={"title": "i10b task", "description": "d", "agent_id": agent_id},
    )
    assert create_response.status_code == 200, create_response.text
    task_id = int(create_response.json()["task_id"])

    def _raising_hook(db: Session, *, team_id: int) -> dict[str, set[int]]:
        raise RuntimeError("team-scope-hook-failure-must-not-leak")

    connector_team_scope.set_connector_team_hooks(team_visibility=_raising_hook)
    try:
        response = client.post(
            _values_url(task_id),
            headers=headers,
            json={
                "items": [
                    {
                        "connector_ref": {"connector_type": "mcp", "connector_id": 1},
                        "context": {"auth_token": "x"},
                    }
                ]
            },
        )
    finally:
        connector_team_scope.set_connector_team_hooks()

    assert response.status_code == 503, response.text
    assert "team-scope-hook-failure-must-not-leak" not in response.text
    body = response.json()
    assert body["error"]["details"]["reason"] == "team_scope_resolution_failed"


@pytest.mark.parametrize(
    ("payload_builder", "expected_reason_prefix"),
    [
        (
            lambda server_id: {"other_key": "x"},
            "undeclared_context_key",
        ),
        (
            lambda server_id: {"auth_token": {"nested": "object"}},
            "type_mismatch.context.auth_token",
        ),
        (
            lambda server_id: {"auth_token": 5},
            "type_mismatch.context.auth_token",
        ),
    ],
)
def test_values_endpoint_rejects_invalid_values_and_writes_nothing(
    e2e_db: None, payload_builder: Any, expected_reason_prefix: str
) -> None:
    """An undeclared key, a string-typed key given an object, and a
    string-typed key given an int are each rejected, and each rejection
    leaves zero rows behind."""
    headers, task_id, server_id = _setup_context_task()
    response = client.post(
        _values_url(task_id),
        headers=headers,
        json={
            "items": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": server_id,
                    },
                    "context": payload_builder(server_id),
                }
            ]
        },
    )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["details"]["reason"] == expected_reason_prefix
    assert _context_row_count(task_id) == 0


def test_values_endpoint_rejects_object_key_given_wrong_shapes(e2e_db: None) -> None:
    """An object-typed key given a string, then given an array, is rejected
    both times."""
    headers, task_id, server_id = _setup_context_task(key_type="object")
    for bad_value in ("not-an-object", ["also", "not", "an", "object"]):
        response = client.post(
            _values_url(task_id),
            headers=headers,
            json={
                "items": [
                    {
                        "connector_ref": {
                            "connector_type": "mcp",
                            "connector_id": server_id,
                        },
                        "context": {"auth_token": bad_value},
                    }
                ]
            },
        )
        assert response.status_code == 400, response.text
        assert (
            response.json()["error"]["details"]["reason"]
            == "type_mismatch.context.auth_token"
        )
        assert _context_row_count(task_id) == 0


def test_values_endpoint_rejects_malformed_key_name(e2e_db: None) -> None:
    """A key name that fails ``validate_runtime_source_key`` (e.g.
    containing a dot) is rejected."""
    headers, task_id, server_id = _setup_context_task()
    response = client.post(
        _values_url(task_id),
        headers=headers,
        json={
            "items": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": server_id,
                    },
                    "context": {"auth.token": "x"},
                }
            ]
        },
    )
    assert response.status_code == 400, response.text
    assert _context_row_count(task_id) == 0


def test_values_endpoint_response_stays_unsatisfied_for_a_malformed_declared_key(
    e2e_db: None,
) -> None:
    """A successful write does not make the response's top-level
    ``satisfied`` true while the connector still declares a key whose name
    the per-turn gate rejects -- even an optional one. The write itself
    succeeds and the key it wrote reads satisfied, but the malformed key
    stays listed and unsatisfied and holds the top-level flag at false, so
    a caller cannot read a 200 here as "this task can now run".
    """
    headers = _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        server = _mcp_server_with_context_schema(
            db,
            user,
            name="values-malformed-declaration-server",
            context_schema={
                "auth_token": {"type": "string", "required": True},
                "bad.key": {"type": "string", "required": False},
            },
            url="https://example.com/values-malformed-declaration/mcp",
        )
        agent = _create_agent(
            db,
            user,
            name="Values Malformed Declaration Agent",
            tool_categories=["mcp"],
        )
        db.commit()
        db.refresh(agent)
        db.refresh(server)
        agent_id = int(agent.id)
        server_id = int(server.id)
    finally:
        db.close()

    create_response = client.post(
        "/api/chat/task/create",
        headers=headers,
        json={
            "title": "values malformed declaration task",
            "description": "d",
            "agent_id": agent_id,
        },
    )
    assert create_response.status_code == 200, create_response.text
    task_id = int(create_response.json()["task_id"])

    response = client.post(
        _values_url(task_id),
        headers=headers,
        json={
            "items": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": server_id,
                    },
                    "context": {"auth_token": "x"},
                }
            ]
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    inputs_by_key = {
        item["key"]: item
        for connector in payload["connectors"]
        for item in connector["inputs"]
    }
    assert set(inputs_by_key) == {"auth_token", "bad.key"}
    assert inputs_by_key["auth_token"]["satisfied"] is True
    assert inputs_by_key["bad.key"]["required"] is False
    assert inputs_by_key["bad.key"]["satisfied"] is False
    assert payload["satisfied"] is False


def test_values_endpoint_rejects_oversized_single_key_and_batch(e2e_db: None) -> None:
    """The two size-cap cases: a single key over 64KB, and a batch over
    256KB in aggregate even though no single key trips the per-key cap.
    Neither writes anything, and which key was oversized never
    appears in the response ``reason`` -- only the fixed literal
    ``payload_too_large``."""
    headers, task_id, server_id = _setup_context_task()
    oversized_value = "x" * (64 * 1024 + 1)
    response = client.post(
        _values_url(task_id),
        headers=headers,
        json={
            "items": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": server_id,
                    },
                    "context": {"auth_token": oversized_value},
                }
            ]
        },
    )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["details"]["reason"] == "payload_too_large"
    assert _context_row_count(task_id) == 0

    # A batch that trips the aggregate cap without any single key tripping
    # the per-key cap: one connector declaring five keys, each just under
    # 64KB (5 * ~64KB > 256KB in total).
    headers2 = _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        key_names = [f"key_{i}" for i in range(5)]
        server2 = _mcp_server_with_context_schema(
            db,
            user,
            name="a4-batch-server",
            context_schema={
                key: {"type": "string", "required": False} for key in key_names
            },
        )
        agent2 = _create_agent(db, user, name="a4-batch-agent", tool_categories=["mcp"])
        db.commit()
        db.refresh(agent2)
        db.refresh(server2)
        agent2_id = int(agent2.id)
        server2_id = int(server2.id)
    finally:
        db.close()

    create_response2 = client.post(
        "/api/chat/task/create",
        headers=headers2,
        json={"title": "a4-batch-task", "description": "d", "agent_id": agent2_id},
    )
    assert create_response2.status_code == 200, create_response2.text
    task_id2 = int(create_response2.json()["task_id"])

    just_under_per_key_cap = "x" * (64 * 1024 - 200)
    response2 = client.post(
        _values_url(task_id2),
        headers=headers2,
        json={
            "items": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": server2_id,
                    },
                    "context": {key: just_under_per_key_cap for key in key_names},
                },
            ]
        },
    )
    assert response2.status_code == 400, response2.text
    assert response2.json()["error"]["details"]["reason"] == "payload_too_large"
    assert _context_row_count(task_id2) == 0


def test_values_endpoint_rejects_duplicate_ref_in_batch(e2e_db: None) -> None:
    """The same connector ref submitted twice in one batch is rejected as a
    duplicate."""
    headers, task_id, server_id = _setup_context_task()
    item = {
        "connector_ref": {"connector_type": "mcp", "connector_id": server_id},
        "context": {"auth_token": "x"},
    }
    response = client.post(
        _values_url(task_id), headers=headers, json={"items": [item, item]}
    )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["details"]["reason"] == "duplicate_ref"
    assert _context_row_count(task_id) == 0


def _stored_context(task_id: int, server_id: int) -> dict[str, Any] | None:
    db = _db_session()
    try:
        row = (
            db.query(TaskConnectorRuntimeContext)
            .filter(
                TaskConnectorRuntimeContext.task_id == task_id,
                TaskConnectorRuntimeContext.connector_type == "mcp",
                TaskConnectorRuntimeContext.connector_id == server_id,
            )
            .one_or_none()
        )
        return dict(row.context) if row is not None else None
    finally:
        db.close()


def test_values_endpoint_merges_keys_and_never_replaces_a_stored_value(
    e2e_db: None,
) -> None:
    """Five requests against the same connector row, in sequence -- a
    stored value is never replaced, a new key can be added, and the
    response reflects the write it just made (not a stale pre-write read).
    """
    headers = _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        server = _mcp_server_with_context_schema(
            db,
            user,
            name="i12-server",
            context_schema={
                "a": {"type": "string", "required": False},
                "b": {"type": "string", "required": False},
                "c": {"type": "string", "required": False},
            },
        )
        agent = _create_agent(db, user, name="i12-agent", tool_categories=["mcp"])
        db.commit()
        db.refresh(agent)
        db.refresh(server)
        agent_id = int(agent.id)
        server_id = int(server.id)
    finally:
        db.close()

    create_response = client.post(
        "/api/chat/task/create",
        headers=headers,
        json={"title": "i12 task", "description": "d", "agent_id": agent_id},
    )
    assert create_response.status_code == 200, create_response.text
    task_id = int(create_response.json()["task_id"])
    ref = {"connector_type": "mcp", "connector_id": server_id}

    def _post(context: dict[str, Any]) -> Any:
        return client.post(
            _values_url(task_id),
            headers=headers,
            json={"items": [{"connector_ref": ref, "context": context}]},
        )

    # (1) write "a", then resend the same value -- 200, one row, unchanged.
    r1 = _post({"a": "1"})
    assert r1.status_code == 200, r1.text
    r1b = _post({"a": "1"})
    assert r1b.status_code == 200, r1b.text
    assert _stored_context(task_id, server_id) == {"a": "1"}
    assert _context_row_count(task_id) == 1

    # (2) resend "a" with a different value -- 409, unchanged.
    r2 = _post({"a": "2"})
    assert r2.status_code == 409, r2.text
    assert r2.json()["error"]["details"]["reason"] == "conflict.context.a"
    assert _stored_context(task_id, server_id) == {"a": "1"}

    # (3) add "b" -- 200, merged, and this response's own inputs for both
    # "a" and "b" already read satisfied=true: this response is assembled
    # from the write it just made, not a stale pre-write read.
    r3 = _post({"b": "2"})
    assert r3.status_code == 200, r3.text
    assert _stored_context(task_id, server_id) == {"a": "1", "b": "2"}
    r3_body = r3.json()
    r3_inputs = {item["key"]: item for item in r3_body["connectors"][0]["inputs"]}
    assert r3_inputs["a"]["satisfied"] is True
    assert r3_inputs["b"]["satisfied"] is True
    # The wire fields this phase emits as constants, pinned on the 200 body
    # of the third producer of this model: `expired` is false for every
    # input and `secrets_expires_at` is null, because no expiry-bearing
    # value can be stored yet, and every input a `context`-only connector
    # declares lands in the `context` section.
    assert all(item["expired"] is False for item in r3_inputs.values())
    assert all(item["section"] == "context" for item in r3_inputs.values())
    assert r3_body["secrets_expires_at"] is None

    # (4) add "c" alongside the already-stored "a" -- 200, all three present.
    r4 = _post({"a": "1", "c": "3"})
    assert r4.status_code == 200, r4.text
    assert _stored_context(task_id, server_id) == {"a": "1", "b": "2", "c": "3"}

    # (5) conflict on "a" again, this time alongside a currently-valid "c"
    # -- the whole request still fails, not a byte written.
    r5 = _post({"a": "2", "c": "3"})
    assert r5.status_code == 409, r5.text
    assert _stored_context(task_id, server_id) == {"a": "1", "b": "2", "c": "3"}


@pytest.mark.parametrize("override_key", ["if_absent", "overwrite", "force"])
def test_values_endpoint_rejects_any_override_switch(
    e2e_db: None, override_key: str
) -> None:
    """No override switch is accepted at the request-shape level --
    ``extra="forbid"`` turns any of them into 422 before the merge logic
    ever runs."""
    headers, task_id, server_id = _setup_context_task(required=False)
    response = client.post(
        _values_url(task_id),
        headers=headers,
        json={
            "items": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": server_id,
                    },
                    "context": {"auth_token": "x"},
                    override_key: True,
                }
            ]
        },
    )
    assert response.status_code == 422, response.text
    assert _context_row_count(task_id) == 0


@pytest.mark.parametrize(
    ("section", "payload"),
    [("secrets", {"api_key": "x"}), ("auth_selector", {"choice": "a"})],
)
def test_values_endpoint_rejects_secret_bearing_sections(
    e2e_db: None, section: str, payload: dict[str, str]
) -> None:
    """``secrets`` and ``auth_selector`` are deliberately absent from the
    request shape at this phase, so an item carrying either section is
    422 before the merge logic runs and nothing is stored."""
    headers, task_id, server_id = _setup_context_task(required=False)
    response = client.post(
        _values_url(task_id),
        headers=headers,
        json={
            "items": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": server_id,
                    },
                    "context": {"auth_token": "x"},
                    section: payload,
                }
            ]
        },
    )
    assert response.status_code == 422, response.text
    assert _context_row_count(task_id) == 0


def test_values_endpoint_conflict_details_carry_reason_and_connector_ref(
    e2e_db: None,
) -> None:
    """A 409's ``details`` carries both ``reason`` and ``connector_ref`` --
    the latter is a documented part of the contract, not an accidental leak
    of ``ConnectorRuntimeError``'s internals."""
    headers, task_id, server_id = _setup_context_task(required=False)
    ref = {"connector_type": "mcp", "connector_id": server_id}
    first = client.post(
        _values_url(task_id),
        headers=headers,
        json={"items": [{"connector_ref": ref, "context": {"auth_token": "1"}}]},
    )
    assert first.status_code == 200, first.text
    second = client.post(
        _values_url(task_id),
        headers=headers,
        json={"items": [{"connector_ref": ref, "context": {"auth_token": "2"}}]},
    )
    assert second.status_code == 409, second.text
    details = second.json()["error"]["details"]
    assert details["reason"] == "conflict.context.auth_token"
    assert details["connector_ref"] == ref


def test_values_endpoint_first_item_atomic_when_second_item_fails_validation(
    e2e_db: None,
) -> None:
    """Multi-item atomicity on the 400 path -- the first item alone would
    have succeeded, but the whole batch fails together with the second, and
    neither lands."""
    headers = _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        good_server = _mcp_server_with_context_schema(
            db,
            user,
            name="i15-good-server",
            context_schema={"auth_token": {"type": "string", "required": False}},
        )
        bad_server = _mcp_server_with_context_schema(
            db,
            user,
            name="i15-bad-server",
            context_schema={"auth_token": {"type": "string", "required": False}},
        )
        agent = _create_agent(db, user, name="i15-agent", tool_categories=["mcp"])
        db.commit()
        db.refresh(agent)
        db.refresh(good_server)
        db.refresh(bad_server)
        agent_id = int(agent.id)
        good_id = int(good_server.id)
        bad_id = int(bad_server.id)
    finally:
        db.close()

    create_response = client.post(
        "/api/chat/task/create",
        headers=headers,
        json={"title": "i15 task", "description": "d", "agent_id": agent_id},
    )
    assert create_response.status_code == 200, create_response.text
    task_id = int(create_response.json()["task_id"])

    response = client.post(
        _values_url(task_id),
        headers=headers,
        json={
            "items": [
                {
                    "connector_ref": {"connector_type": "mcp", "connector_id": good_id},
                    "context": {"auth_token": "ok"},
                },
                {
                    "connector_ref": {"connector_type": "mcp", "connector_id": bad_id},
                    "context": {"undeclared_key": "x"},
                },
            ]
        },
    )
    assert response.status_code == 400, response.text
    assert _context_row_count(task_id) == 0


def test_values_endpoint_first_item_atomic_when_second_item_conflicts(
    e2e_db: None,
) -> None:
    """Multi-item atomicity on the 409 path. Ref A has no row yet; ref B
    already has a different value stored under another session. One
    request submitting A+B fails together (A gets zero residue too), then
    resubmitting A alone (dropping B, the frontend rule after a 409)
    succeeds."""
    headers = _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        server_a = _mcp_server_with_context_schema(
            db,
            user,
            name="i11ba-server-a",
            context_schema={"auth_token": {"type": "string", "required": False}},
        )
        server_b = _mcp_server_with_context_schema(
            db,
            user,
            name="i11ba-server-b",
            context_schema={"auth_token": {"type": "string", "required": False}},
        )
        agent = _create_agent(db, user, name="i11ba-agent", tool_categories=["mcp"])
        db.commit()
        db.refresh(agent)
        db.refresh(server_a)
        db.refresh(server_b)
        agent_id = int(agent.id)
        a_id = int(server_a.id)
        b_id = int(server_b.id)
    finally:
        db.close()

    create_response = client.post(
        "/api/chat/task/create",
        headers=headers,
        json={"title": "i11ba task", "description": "d", "agent_id": agent_id},
    )
    assert create_response.status_code == 200, create_response.text
    task_id = int(create_response.json()["task_id"])

    ref_a = {"connector_type": "mcp", "connector_id": a_id}
    ref_b = {"connector_type": "mcp", "connector_id": b_id}
    pre_fill = client.post(
        _values_url(task_id),
        headers=headers,
        json={
            "items": [{"connector_ref": ref_b, "context": {"auth_token": "original"}}]
        },
    )
    assert pre_fill.status_code == 200, pre_fill.text

    combined = client.post(
        _values_url(task_id),
        headers=headers,
        json={
            "items": [
                {"connector_ref": ref_a, "context": {"auth_token": "new-a"}},
                {"connector_ref": ref_b, "context": {"auth_token": "conflicting"}},
            ]
        },
    )
    assert combined.status_code == 409, combined.text
    assert (
        combined.json()["error"]["details"]["reason"] == "conflict.context.auth_token"
    )
    assert _context_row_count(task_id) == 1  # only B's pre-fill row
    assert _stored_context(task_id, a_id) is None

    only_a = client.post(
        _values_url(task_id),
        headers=headers,
        json={"items": [{"connector_ref": ref_a, "context": {"auth_token": "new-a"}}]},
    )
    assert only_a.status_code == 200, only_a.text
    assert _stored_context(task_id, a_id) == {"auth_token": "new-a"}


def test_values_endpoint_partial_fill_returns_200_and_persists(e2e_db: None) -> None:
    """Partial fill is 200, not a rejection -- including when the
    connector's declared schema grows a new required key between when the
    caller last read the requirements and when it submits."""
    headers = _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        server = _mcp_server_with_context_schema(
            db,
            user,
            name="i32b-server",
            context_schema={
                "a": {"type": "string", "required": True},
                "b": {"type": "string", "required": True},
            },
        )
        agent = _create_agent(db, user, name="i32b-agent", tool_categories=["mcp"])
        db.commit()
        db.refresh(agent)
        db.refresh(server)
        agent_id = int(agent.id)
        server_id = int(server.id)
    finally:
        db.close()

    create_response = client.post(
        "/api/chat/task/create",
        headers=headers,
        json={"title": "i32b task", "description": "d", "agent_id": agent_id},
    )
    assert create_response.status_code == 200, create_response.text
    task_id = int(create_response.json()["task_id"])
    ref = {"connector_type": "mcp", "connector_id": server_id}

    response = client.post(
        _values_url(task_id),
        headers=headers,
        json={"items": [{"connector_ref": ref, "context": {"a": "1"}}]},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["satisfied"] is False
    inputs = {item["key"]: item for item in payload["connectors"][0]["inputs"]}
    assert inputs["a"]["satisfied"] is True
    assert inputs["b"]["satisfied"] is False
    assert _stored_context(task_id, server_id) == {"a": "1"}

    # Declaration grows a new required key after this read; the caller
    # (having computed "only a was missing" from a stale read) submits
    # exactly what it originally intended. Still 200; the new key is simply
    # still unsatisfied afterward, not a rejection of this request.
    db = _db_session()
    try:
        row = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        row.runtime_input_schema = {
            "context": {
                "a": {"type": "string", "required": True},
                "b": {"type": "string", "required": True},
                "d": {"type": "string", "required": True},
            }
        }
        db.commit()
    finally:
        db.close()

    response2 = client.post(
        _values_url(task_id),
        headers=headers,
        json={"items": [{"connector_ref": ref, "context": {"b": "2"}}]},
    )
    assert response2.status_code == 200, response2.text
    assert _stored_context(task_id, server_id) == {"a": "1", "b": "2"}


def _count_context_table_writes(fn: Any) -> int:
    """Run ``fn`` while counting INSERT/UPDATE/DELETE statements issued
    against ``task_connector_runtime_contexts`` on the app's engine."""
    from xagent.web.models.database import get_engine

    engine = get_engine()
    counted = {"n": 0}

    def _listener(
        conn: Any, cursor: Any, statement: str, *_args: Any, **_kwargs: Any
    ) -> None:
        upper = statement.upper()
        if "TASK_CONNECTOR_RUNTIME_CONTEXTS" in upper and (
            "INSERT" in upper or "UPDATE" in upper or "DELETE" in upper
        ):
            counted["n"] += 1

    event.listen(engine, "before_cursor_execute", _listener)
    try:
        fn()
    finally:
        event.remove(engine, "before_cursor_execute", _listener)
    return counted["n"]


def test_values_endpoint_same_value_rewrite_issues_no_write_statement(
    e2e_db: None,
) -> None:
    """Writing the same value twice issues zero INSERT/UPDATE statements on
    the second request -- the short circuit at "same content already on the
    row" actually skips the write, not just the response change it would
    otherwise have caused."""
    headers, task_id, server_id = _setup_context_task(required=False)
    ref = {"connector_type": "mcp", "connector_id": server_id}
    body = {"items": [{"connector_ref": ref, "context": {"auth_token": "same"}}]}

    first = client.post(_values_url(task_id), headers=headers, json=body)
    assert first.status_code == 200, first.text
    assert _context_row_count(task_id) == 1

    second_writes = _count_context_table_writes(
        lambda: client.post(_values_url(task_id), headers=headers, json=body)
    )
    assert second_writes == 0
    assert _context_row_count(task_id) == 1


@pytest.mark.parametrize(
    ("key_type", "context_value"),
    [
        ("string", "SENTINEL-CONTEXT-VALUE-DO-NOT-LOG"),
        ("object", {"nested": "SENTINEL-CONTEXT-VALUE-DO-NOT-LOG"}),
    ],
    ids=["top_level_string", "nested_in_object"],
)
def test_values_endpoint_never_logs_submitted_values(
    e2e_db: None,
    caplog: pytest.LogCaptureFixture,
    key_type: str,
    context_value: Any,
) -> None:
    """A submitted value never reaches a log line, checked at DEBUG across
    the whole request, both in the rendered message and in any
    lazily-formatted ``%s`` args -- for a plain top-level string value and
    for one nested inside an object."""
    import logging

    headers, task_id, server_id = _setup_context_task(required=False, key_type=key_type)
    sentinel = "SENTINEL-CONTEXT-VALUE-DO-NOT-LOG"
    ref = {"connector_type": "mcp", "connector_id": server_id}

    with caplog.at_level(logging.DEBUG):
        response = client.post(
            _values_url(task_id),
            headers=headers,
            json={
                "items": [
                    {
                        "connector_ref": ref,
                        "context": {"auth_token": context_value},
                    }
                ]
            },
        )
    assert response.status_code == 200, response.text
    for record in caplog.records:
        assert sentinel not in record.getMessage()
        for arg in record.args or ():
            assert sentinel not in str(arg)
    # Positive half: a fill is still recorded, just without the value -- the
    # key name alone shows up in the one log line the write path does emit.
    assert any("auth_token" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("blank_value", ["", "   ", "\t\n"])
def test_values_endpoint_rejects_a_blank_string_value(
    e2e_db: None, blank_value: str
) -> None:
    """A required key submitted empty, or with nothing but whitespace, is
    rejected before anything is written. Accepting it would mark the key
    filled forever -- a stored value is never replaced -- while carrying
    nothing a connector can use."""
    headers, task_id, server_id = _setup_context_task()
    response = client.post(
        _values_url(task_id),
        headers=headers,
        json={
            "items": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": server_id,
                    },
                    "context": {"auth_token": blank_value},
                }
            ]
        },
    )
    assert response.status_code == 400, response.text
    body = response.json()
    assert body["error"]["code"] == "invalid_runtime_context"
    assert body["error"]["details"]["reason"] == "empty_value.context.auth_token"
    assert _context_row_count(task_id) == 0


def test_values_endpoint_rejects_a_blank_value_for_an_optional_key(
    e2e_db: None,
) -> None:
    """The blank rule ignores ``required``: an optional key's value is just
    as permanent once stored, so a blank one is refused there too."""
    headers, task_id, server_id = _setup_context_task(required=False)
    response = client.post(
        _values_url(task_id),
        headers=headers,
        json={
            "items": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": server_id,
                    },
                    "context": {"auth_token": ""},
                }
            ]
        },
    )
    assert response.status_code == 400, response.text
    assert (
        response.json()["error"]["details"]["reason"]
        == "empty_value.context.auth_token"
    )
    assert _context_row_count(task_id) == 0


def test_values_endpoint_rejects_an_empty_object_value(e2e_db: None) -> None:
    """An object-typed key given ``{}`` passes the type check and is still
    refused: an empty object is a value with no content, and ``{}`` is not
    "a value" for the purpose of satisfying the key."""
    headers, task_id, server_id = _setup_context_task(key_type="object")
    response = client.post(
        _values_url(task_id),
        headers=headers,
        json={
            "items": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": server_id,
                    },
                    "context": {"auth_token": {}},
                }
            ]
        },
    )
    assert response.status_code == 400, response.text
    assert (
        response.json()["error"]["details"]["reason"]
        == "empty_value.context.auth_token"
    )
    assert _context_row_count(task_id) == 0


def test_values_endpoint_accepts_a_real_value_after_a_blank_one_was_refused(
    e2e_db: None,
) -> None:
    """The refusal writes nothing, so the caller can simply submit again --
    this is the whole point of refusing at the door rather than storing a
    blank that could never be replaced."""
    headers, task_id, server_id = _setup_context_task()
    ref = {"connector_type": "mcp", "connector_id": server_id}

    refused = client.post(
        _values_url(task_id),
        headers=headers,
        json={"items": [{"connector_ref": ref, "context": {"auth_token": ""}}]},
    )
    assert refused.status_code == 400, refused.text
    assert _stored_context(task_id, server_id) is None

    accepted = client.post(
        _values_url(task_id),
        headers=headers,
        json={
            "items": [{"connector_ref": ref, "context": {"auth_token": "real-token"}}]
        },
    )
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["satisfied"] is True
    assert _stored_context(task_id, server_id) == {"auth_token": "real-token"}


def test_values_endpoint_treats_a_json_reshaped_value_as_a_conflict(
    e2e_db: None,
) -> None:
    """``{"a": 1}`` already stored, then ``{"a": 1.0}`` or ``{"a": true}``
    submitted: Python calls those equal, but they store different JSON, so
    the immutability rule must fire rather than answering 200 and writing
    nothing. Resubmitting the identical value is still the no-op it was,
    with no write statement at all."""
    headers, task_id, server_id = _setup_context_task(key_type="object")
    ref = {"connector_type": "mcp", "connector_id": server_id}

    def _post(value: Any) -> Any:
        return client.post(
            _values_url(task_id),
            headers=headers,
            json={"items": [{"connector_ref": ref, "context": {"auth_token": value}}]},
        )

    first = _post({"a": 1})
    assert first.status_code == 200, first.text
    assert _stored_context(task_id, server_id) == {"auth_token": {"a": 1}}

    for reshaped in ({"a": 1.0}, {"a": True}):
        conflict = _post(reshaped)
        assert conflict.status_code == 409, conflict.text
        assert (
            conflict.json()["error"]["details"]["reason"]
            == "conflict.context.auth_token"
        )
        assert _stored_context(task_id, server_id) == {"auth_token": {"a": 1}}

    writes = _count_context_table_writes(lambda: _post({"a": 1}))
    assert writes == 0
    assert _stored_context(task_id, server_id) == {"auth_token": {"a": 1}}


def test_values_endpoint_reports_satisfied_once_every_required_key_is_filled(
    e2e_db: None,
) -> None:
    """The endpoint exists to turn this flag true, so one test asserts it
    does -- on the write response itself, and again on the task-keyed read
    endpoint afterwards."""
    headers = _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        server = _mcp_server_with_context_schema(
            db,
            user,
            name="satisfied-server",
            context_schema={
                "a": {"type": "string", "required": True},
                "b": {"type": "string", "required": True},
            },
        )
        agent = _create_agent(db, user, name="satisfied-agent", tool_categories=["mcp"])
        db.commit()
        db.refresh(agent)
        db.refresh(server)
        agent_id = int(agent.id)
        server_id = int(server.id)
    finally:
        db.close()

    create_response = client.post(
        "/api/chat/task/create",
        headers=headers,
        json={"title": "satisfied task", "description": "d", "agent_id": agent_id},
    )
    assert create_response.status_code == 200, create_response.text
    task_id = int(create_response.json()["task_id"])
    ref = {"connector_type": "mcp", "connector_id": server_id}

    partial = client.post(
        _values_url(task_id),
        headers=headers,
        json={"items": [{"connector_ref": ref, "context": {"a": "1"}}]},
    )
    assert partial.status_code == 200, partial.text
    assert partial.json()["satisfied"] is False

    final = client.post(
        _values_url(task_id),
        headers=headers,
        json={"items": [{"connector_ref": ref, "context": {"b": "2"}}]},
    )
    assert final.status_code == 200, final.text
    final_body = final.json()
    assert final_body["satisfied"] is True
    final_inputs = {item["key"]: item for item in final_body["connectors"][0]["inputs"]}
    assert final_inputs["a"]["satisfied"] is True
    assert final_inputs["b"]["satisfied"] is True

    report = client.get(
        f"/api/chat/task/{task_id}/connector-runtime-requirements", headers=headers
    )
    assert report.status_code == 200, report.text
    assert report.json()["satisfied"] is True


def test_values_endpoint_returns_503_when_the_conditional_update_keeps_missing(
    e2e_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retry exhaustion on the conditional-update branch, asserted at the
    HTTP layer: exactly two attempts, then the typed 503 whose ``details``
    carry no reason -- the shape that tells a caller this was a collision
    with another writer and is safe to retry. Driven by forcing every
    conditional update to match no row, which is what a lost race looks
    like from inside this request."""
    task_id, server_id = _setup_concurrency_task(key_names=["a", "b"])
    headers = _setup_admin_headers()
    ref = {"connector_type": "mcp", "connector_id": server_id}

    seed = client.post(
        _values_url(task_id),
        headers=headers,
        json={"items": [{"connector_ref": ref, "context": {"a": "1"}}]},
    )
    assert seed.status_code == 200, seed.text

    from xagent.web.services import connector_runtime as connector_runtime_service

    calls = {"n": 0}

    def _always_missing_update(
        db: Session, *, row_id: int, context_text_as_read: str, merged: dict[str, Any]
    ) -> bool:
        calls["n"] += 1
        return False

    monkeypatch.setattr(
        connector_runtime_service, "_update_context_row", _always_missing_update
    )

    response = client.post(
        _values_url(task_id),
        headers=headers,
        json={"items": [{"connector_ref": ref, "context": {"b": "2"}}]},
    )

    assert response.status_code == 503, response.text
    assert response.json() == {
        "error": {
            "code": "connector_runtime_unavailable",
            "message": "Connector runtime context is unavailable.",
            "details": {},
        }
    }
    assert calls["n"] == 2
    assert _stored_context(task_id, server_id) == {"a": "1"}


def test_values_endpoint_returns_503_when_the_insert_keeps_colliding(
    e2e_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sibling branch, same assertion at the same layer: a snapshot
    read that never sees the row already in the table sends this request
    down the insert path twice, and the second ``IntegrityError`` ends it
    as the same typed 503 with the same empty ``details``."""
    task_id, server_id = _setup_concurrency_task(key_names=["a", "b"])
    headers = _setup_admin_headers()
    ref = {"connector_type": "mcp", "connector_id": server_id}

    seed = client.post(
        _values_url(task_id),
        headers=headers,
        json={"items": [{"connector_ref": ref, "context": {"a": "1"}}]},
    )
    assert seed.status_code == 200, seed.text

    from xagent.web.services import connector_runtime as connector_runtime_service

    calls = {"n": 0}

    def _always_stale_read(db: Session, *, task_id: int) -> dict[str, Any]:
        calls["n"] += 1
        return {}

    monkeypatch.setattr(
        connector_runtime_service,
        "_load_task_context_row_snapshots",
        _always_stale_read,
    )

    response = client.post(
        _values_url(task_id),
        headers=headers,
        json={"items": [{"connector_ref": ref, "context": {"b": "2"}}]},
    )

    assert response.status_code == 503, response.text
    assert response.json() == {
        "error": {
            "code": "connector_runtime_unavailable",
            "message": "Connector runtime context is unavailable.",
            "details": {},
        }
    }
    assert calls["n"] == 2
    assert _stored_context(task_id, server_id) == {"a": "1"}


def test_values_endpoint_rolls_back_when_the_commit_itself_fails(
    e2e_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``db.commit()`` that raises is rolled back explicitly and the
    exception is left to travel: a bare 500, not the typed envelope, and
    nothing of the batch in the table. The envelope is reserved for a
    ``ConnectorRuntimeError``, whose ``code`` promises the caller a
    condition they can act on; an unclassified commit fault is not one.

    ``Session.commit``/``Session.rollback`` are patched only for the
    duration of the request, so the setup above and the table read below
    both run against the real methods.
    """
    headers, task_id, server_id = _setup_context_task(required=False)
    ref = {"connector_type": "mcp", "connector_id": server_id}
    body = {"items": [{"connector_ref": ref, "context": {"auth_token": "v"}}]}

    real_rollback = Session.rollback
    rollbacks = {"n": 0}

    def _failing_commit(self: Session) -> None:
        raise RuntimeError("commit-failure-must-not-leak")

    def _counting_rollback(self: Session) -> None:
        rollbacks["n"] += 1
        real_rollback(self)

    monkeypatch.setattr(Session, "commit", _failing_commit)
    monkeypatch.setattr(Session, "rollback", _counting_rollback)
    try:
        response = client.post(_values_url(task_id), headers=headers, json=body)
    finally:
        monkeypatch.undo()

    assert response.status_code == 500, response.text
    assert "commit-failure-must-not-leak" not in response.text
    assert rollbacks["n"] == 1
    assert _context_row_count(task_id) == 0


def _setup_two_connector_context_task() -> tuple[dict[str, str], int, int, int]:
    """One task whose agent selects two MCP connectors, each declaring the
    same two optional context keys. Returns (owner headers, task_id, and
    the two connector ids in canonical order -- ascending, which is what
    ``_sort_connector_refs`` produces for two refs of the same type).
    """
    headers = _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        schema = {
            "a": {"type": "string", "required": False},
            "b": {"type": "string", "required": False},
        }
        first = _mcp_server_with_context_schema(
            db, user, name="order-server-1", context_schema=schema
        )
        second = _mcp_server_with_context_schema(
            db, user, name="order-server-2", context_schema=schema
        )
        agent = _create_agent(db, user, name="order-agent", tool_categories=["mcp"])
        db.commit()
        db.refresh(agent)
        db.refresh(first)
        db.refresh(second)
        agent_id = int(agent.id)
        server_ids = sorted([int(first.id), int(second.id)])
    finally:
        db.close()

    create_response = client.post(
        "/api/chat/task/create",
        headers=headers,
        json={"title": "order task", "description": "d", "agent_id": agent_id},
    )
    assert create_response.status_code == 200, create_response.text
    task_id = int(create_response.json()["task_id"])
    return headers, task_id, server_ids[0], server_ids[1]


def test_values_endpoint_updates_rows_in_canonical_ref_order(
    e2e_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch naming two connectors in reverse canonical order still
    updates their rows in canonical order. Two concurrent requests listing
    the same rows in opposite order would otherwise be able to take those
    rows' locks in opposite order and deadlock on PostgreSQL, so the
    request's own ordering must not reach the write.
    """
    headers, task_id, low_id, high_id = _setup_two_connector_context_task()
    low_ref = {"connector_type": "mcp", "connector_id": low_id}
    high_ref = {"connector_type": "mcp", "connector_id": high_id}

    # Both rows must already exist, so the second request drives every ref
    # down the conditional-update branch rather than the insert branch.
    seed = client.post(
        _values_url(task_id),
        headers=headers,
        json={
            "items": [
                {"connector_ref": low_ref, "context": {"a": "1"}},
                {"connector_ref": high_ref, "context": {"a": "1"}},
            ]
        },
    )
    assert seed.status_code == 200, seed.text

    db = _db_session()
    try:
        row_id_by_connector = {
            int(row.connector_id): int(row.id)
            for row in db.query(TaskConnectorRuntimeContext)
            .filter(TaskConnectorRuntimeContext.task_id == task_id)
            .all()
        }
    finally:
        db.close()
    assert set(row_id_by_connector) == {low_id, high_id}

    from xagent.web.services import connector_runtime as connector_runtime_service

    real_update = connector_runtime_service._update_context_row
    updated_row_ids: list[int] = []

    def _recording_update(
        db: Session, *, row_id: int, context_text_as_read: str, merged: dict[str, Any]
    ) -> bool:
        updated_row_ids.append(row_id)
        return bool(
            real_update(
                db,
                row_id=row_id,
                context_text_as_read=context_text_as_read,
                merged=merged,
            )
        )

    monkeypatch.setattr(
        connector_runtime_service, "_update_context_row", _recording_update
    )

    response = client.post(
        _values_url(task_id),
        headers=headers,
        json={
            "items": [
                {"connector_ref": high_ref, "context": {"b": "2"}},
                {"connector_ref": low_ref, "context": {"b": "2"}},
            ]
        },
    )

    assert response.status_code == 200, response.text
    assert updated_row_ids == [
        row_id_by_connector[low_id],
        row_id_by_connector[high_id],
    ]
    assert _stored_context(task_id, low_id) == {"a": "1", "b": "2"}
    assert _stored_context(task_id, high_id) == {"a": "1", "b": "2"}


# ---------------------------------------------------------------------------
# Values-endpoint concurrency: two sessions racing the write. Driven
# below the HTTP layer, directly against
# ``apply_task_connector_runtime_context_values``, because the race needs
# precise control over when each session reads versus when the other
# commits -- an ordinary sequential HTTP call cannot express "A read before
# B committed, then B committed, then A wrote".
# ---------------------------------------------------------------------------


def _setup_concurrency_task(*, key_names: list[str]) -> tuple[int, int]:
    """One connector declaring the given (all optional) context keys, one
    task that selected it. Returns (task_id, server_id)."""
    headers = _setup_admin_headers()
    db = _db_session()
    try:
        user = _admin_user(db)
        server = _mcp_server_with_context_schema(
            db,
            user,
            name=f"conc-server-{'-'.join(key_names)}-{secrets.token_hex(4)}",
            context_schema={
                key: {"type": "string", "required": False} for key in key_names
            },
        )
        agent = _create_agent(
            db,
            user,
            name=f"conc-agent-{secrets.token_hex(4)}",
            tool_categories=["mcp"],
        )
        db.commit()
        db.refresh(agent)
        db.refresh(server)
        agent_id = int(agent.id)
        server_id = int(server.id)
    finally:
        db.close()

    create_response = client.post(
        "/api/chat/task/create",
        headers=headers,
        json={"title": "conc task", "description": "d", "agent_id": agent_id},
    )
    assert create_response.status_code == 200, create_response.text
    return int(create_response.json()["task_id"]), server_id


def _direct_apply(
    session: Session, task_id: int, server_id: int, context: dict[str, Any]
) -> Any:
    from xagent.web.models.agent import Agent as AgentModel
    from xagent.web.schemas.connector_runtime import ConnectorRuntimeValueItem
    from xagent.web.services.connector_runtime import (
        apply_task_connector_runtime_context_values,
    )

    task = session.query(Task).filter(Task.id == task_id).one()
    agent = (
        session.query(AgentModel).filter(AgentModel.id == task.agent_id).one()
        if task.agent_id is not None
        else None
    )
    item = ConnectorRuntimeValueItem(
        connector_ref={"connector_type": "mcp", "connector_id": server_id},
        context=context,
    )
    return apply_task_connector_runtime_context_values(
        db=session, task=task, agent=agent, payload_items=[item]
    )


@pytest.mark.parametrize(
    ("a_context", "b_context", "expect_conflict"),
    [
        ({"a": "1"}, {"b": "2"}, False),
        ({"a": "1"}, {"a": "1"}, False),
        ({"a": "1"}, {"a": "2"}, True),
    ],
)
def test_concurrent_first_write_has_one_winner(
    e2e_db: None,
    monkeypatch: pytest.MonkeyPatch,
    a_context: dict[str, str],
    b_context: dict[str, str],
    expect_conflict: bool,
) -> None:
    """Two sessions both find no row for a ref. B writes and commits
    first; A -- still holding its stale "no row" read -- then attempts its
    own insert, hits the real unique-constraint ``IntegrityError``, and
    must recover: roll back, reread (now sees B's committed row), remerge,
    and either succeed or conflict depending on whether the two writers'
    keys collide. A's session must stay usable after the ``IntegrityError``
    -- proven here by querying through it once the call returns.
    """
    task_id, server_id = _setup_concurrency_task(key_names=["a", "b"])

    from xagent.web.services import connector_runtime as connector_runtime_service

    real_read = connector_runtime_service._load_task_context_row_snapshots

    session_a = get_session_local()()
    try:
        # B runs and commits first, through the real (unpatched) read --
        # this is genuinely the first write for this ref, no simulation
        # needed here.
        session_b = get_session_local()()
        try:
            _direct_apply(session_b, task_id, server_id, b_context)
            session_b.commit()
        finally:
            session_b.close()

        # Now install the stale read for A's *first* attempt only: A is
        # simulated as having looked before B committed, i.e. still empty,
        # even though B's row already exists in the database by the time
        # A's insert actually runs. The second attempt (the retry) uses
        # the real reader, which does see B's row.
        state = {"calls": 0}

        def _stale_first_read(db: Session, *, task_id: int) -> dict[str, Any]:
            state["calls"] += 1
            if state["calls"] == 1:
                return {}
            return real_read(db, task_id=task_id)

        monkeypatch.setattr(
            connector_runtime_service,
            "_load_task_context_row_snapshots",
            _stale_first_read,
        )

        if expect_conflict:
            with pytest.raises(ConnectorRuntimeError) as exc_info:
                _direct_apply(session_a, task_id, server_id, a_context)
            assert exc_info.value.code == "runtime_context_immutable"
            session_a.rollback()
        else:
            result = _direct_apply(session_a, task_id, server_id, a_context)
            merged_keys = {
                item.key: item.satisfied
                for connector in result.connectors
                for item in connector.inputs
            }
            for key in {**a_context, **b_context}:
                assert merged_keys[key] is True
            session_a.commit()

        # The session survived the IntegrityError-triggered rollback and
        # retry -- prove it is not left in SQLAlchemy's
        # ``PendingRollbackError`` state by issuing one more query on it.
        assert session_a.query(Task).filter(Task.id == task_id).one() is not None
    finally:
        session_a.close()

    assert _context_row_count(task_id) == 1


def test_concurrent_first_write_retry_is_bounded(
    e2e_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write that keeps colliding does not retry forever -- exactly two
    attempts, then 503."""
    task_id, server_id = _setup_concurrency_task(key_names=["a"])

    from xagent.web.services import connector_runtime as connector_runtime_service

    # B commits a row for this ref up front, unconditionally.
    session_b = get_session_local()()
    try:
        _direct_apply(session_b, task_id, server_id, {"a": "occupied"})
        session_b.commit()
    finally:
        session_b.close()

    calls = {"n": 0}

    def _always_stale_read(db: Session, *, task_id: int) -> dict[str, Any]:
        calls["n"] += 1
        return {}

    monkeypatch.setattr(
        connector_runtime_service,
        "_load_task_context_row_snapshots",
        _always_stale_read,
    )

    session_a = get_session_local()()
    try:
        with pytest.raises(ConnectorRuntimeError) as exc_info:
            _direct_apply(session_a, task_id, server_id, {"a": "occupied"})
        assert exc_info.value.status_code == 503
        session_a.rollback()
    finally:
        session_a.close()

    assert calls["n"] == 2


@pytest.mark.parametrize(
    ("a_context", "expect_conflict"),
    [
        ({"b": "2"}, False),
        ({"x": "0"}, False),
        ({"x": "9"}, True),
    ],
)
def test_concurrent_overwrite_of_an_existing_row(
    e2e_db: None,
    monkeypatch: pytest.MonkeyPatch,
    a_context: dict[str, str],
    expect_conflict: bool,
) -> None:
    """SQLite half of this compare-and-swap proof; the PostgreSQL half,
    which exercises the database's own text rendering of the JSON column,
    is a separate change. A row already has
    ``{"x": "0"}``. A read it before B committed a change to
    the same row; B commits (adding "a"); A's conditional UPDATE, built on
    the text it read before B's write, cannot match the row's current text
    -- rowcount 0, not an error -- and must retry: reread, remerge, and
    either succeed (A's own key doesn't collide with B's) or conflict
    (A resubmits "x" with a different value than what's on the row now).
    """
    task_id, server_id = _setup_concurrency_task(key_names=["x", "a", "b"])
    ref = {"connector_type": "mcp", "connector_id": server_id}
    seed = client.post(
        _values_url(task_id),
        headers=_setup_admin_headers(),
        json={"items": [{"connector_ref": ref, "context": {"x": "0"}}]},
    )
    assert seed.status_code == 200, seed.text

    from xagent.web.services import connector_runtime as connector_runtime_service

    real_read = connector_runtime_service._load_task_context_row_snapshots

    session_a = get_session_local()()
    try:
        stale_snapshot = real_read(session_a, task_id=task_id)

        session_b = get_session_local()()
        try:
            _direct_apply(session_b, task_id, server_id, {"a": "1"})
            session_b.commit()
        finally:
            session_b.close()

        state = {"calls": 0}
        update_results: list[bool] = []
        real_update = connector_runtime_service._update_context_row

        def _stale_first_read(db: Session, *, task_id: int) -> dict[str, Any]:
            state["calls"] += 1
            if state["calls"] == 1:
                return stale_snapshot
            return real_read(db, task_id=task_id)

        def _counting_update(*args: Any, **kwargs: Any) -> bool:
            result = real_update(*args, **kwargs)
            update_results.append(result)
            return result

        monkeypatch.setattr(
            connector_runtime_service,
            "_load_task_context_row_snapshots",
            _stale_first_read,
        )
        monkeypatch.setattr(
            connector_runtime_service, "_update_context_row", _counting_update
        )

        if expect_conflict:
            with pytest.raises(ConnectorRuntimeError) as exc_info:
                _direct_apply(session_a, task_id, server_id, a_context)
            assert exc_info.value.details["reason"] == "conflict.context.x"
            session_a.rollback()
        else:
            _direct_apply(session_a, task_id, server_id, a_context)
            session_a.commit()
            assert _stored_context(task_id, server_id)["x"] == "0"
            for key, value in a_context.items():
                assert _stored_context(task_id, server_id)[key] == value
            # The counting format: a genuinely new key (the "b" case) has
            # something to write against both the stale and the fresh
            # view, so its first (stale-text) UPDATE attempt hits 0 rows
            # and only the retry, built on a fresh read, succeeds.
            # Resubmitting a value the row already has (the "x" case)
            # merges to the *same* content under the stale view too, so
            # tier 8's same-content skip fires before any UPDATE is even
            # attempted -- zero calls, not one that fails and one that
            # succeeds.
            if "b" in a_context:
                assert update_results == [False, True]
            else:
                assert update_results == []
    finally:
        session_a.close()

    if expect_conflict:
        # No byte moved: the row still shows only what existed before A's
        # failed request, plus B's unrelated write.
        assert _stored_context(task_id, server_id) == {"x": "0", "a": "1"}


def _overwrite_context_row_text(task_id: int, server_id: int, raw_text: str) -> None:
    """Write ``raw_text`` into a context row's ``context`` column via a bare
    SQL UPDATE -- bypassing the JSON column's bind processor entirely, so
    the row holds exactly ``raw_text``, byte for byte, whatever its
    canonical form would have been.
    """
    from sqlalchemy import text as sa_text

    db = _db_session()
    try:
        row = (
            db.query(TaskConnectorRuntimeContext)
            .filter(
                TaskConnectorRuntimeContext.task_id == task_id,
                TaskConnectorRuntimeContext.connector_type == "mcp",
                TaskConnectorRuntimeContext.connector_id == server_id,
            )
            .one()
        )
        db.execute(
            sa_text(
                "UPDATE task_connector_runtime_contexts SET context = :t WHERE id = :i"
            ),
            {"t": raw_text, "i": row.id},
        )
        db.commit()
    finally:
        db.close()


@pytest.mark.parametrize(
    "raw_text",
    [
        # Non-canonical rendering: different key order plus extra
        # whitespace than SQLAlchemy's default `json.dumps` would produce.
        '{"b": 1,   "a": 2}',
        # Non-ASCII, unescaped -- `ensure_ascii=True` (json.dumps's
        # default) would have turned this into \\uXXXX escapes.
        '{"name": "会议室预订"}',
    ],
)
def test_values_endpoint_expected_old_value_is_the_database_rendering(
    e2e_db: None, raw_text: str
) -> None:
    """SQLite half of this compare-and-swap proof; the PostgreSQL half,
    which exercises the database's own text rendering of the JSON column,
    is a separate change. A row whose stored
    text was never produced by this codebase's own
    ``json.dumps`` call (a non-canonical rendering, or unescaped non-ASCII)
    must still accept a new key. If the expected-old-value comparison ever
    re-serializes the read value in Python instead of using the database's
    own ``CAST(context AS TEXT)`` rendering, this row's CAS predicate can
    never match -- every write to it fails forever, not just once.
    """
    headers, task_id, server_id = _setup_context_task(required=False)
    db = _db_session()
    try:
        db.add(
            TaskConnectorRuntimeContext(
                task_id=task_id,
                connector_type="mcp",
                connector_id=server_id,
                context={"placeholder": "will be overwritten"},
            )
        )
        db.commit()
    finally:
        db.close()
    _overwrite_context_row_text(task_id, server_id, raw_text)

    response = client.post(
        _values_url(task_id),
        headers=headers,
        json={
            "items": [
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": server_id,
                    },
                    "context": {"auth_token": "new-value"},
                }
            ]
        },
    )
    assert response.status_code == 200, response.text
    assert _stored_context(task_id, server_id)["auth_token"] == "new-value"


def test_values_endpoint_accepts_team_shared_connector(e2e_db: None) -> None:
    """The values-endpoint half of the team-visibility check: a connector
    visible only through the agent's team is writable through the values
    endpoint for a non-owning team member -- not a 404, which is what
    happens if this
    endpoint's own ``agent_team_id`` derivation is ever dropped back to
    ``None``."""
    db = _db_session()
    try:
        owner = _create_user(db, "values-team-owner")
        member = _create_user(db, "values-team-member")
        db.flush()
        member_id = int(member.id)
        server = MCPServer(
            name="values-team-shared-server",
            description="values-team-shared-server description",
            managed="external",
            transport="streamable_http",
            url="https://example.com/mcp",
            runtime_input_schema={
                "context": {"auth_token": {"type": "string", "required": False}}
            },
            runtime_bindings=[
                {
                    "source": {"input_type": "context", "key": "auth_token"},
                    "target": {"target_type": "mcp_meta", "key": "auth_token"},
                }
            ],
        )
        db.add(server)
        db.flush()
        server_id = int(server.id)
        agent = Agent(
            user_id=owner.id,
            name="Values Team Shared Agent",
            instructions="i",
            execution_mode="balanced",
            status=AgentStatus.PUBLISHED,
            tool_categories=["mcp"],
            team_id=404,
            visibility="team",
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        agent_id = int(agent.id)
        member_headers = _auth_headers_for_user(member)
    finally:
        db.close()

    set_agent_team_scope_hook(
        lambda db, user_id: (
            AgentTeamScope(team_id=404, is_team_admin=False)
            if user_id == member_id
            else None
        )
    )
    connector_team_scope.set_connector_team_hooks(
        team_visibility=lambda db, *, team_id: (
            {"mcp": {server_id}, "custom_api": set()}
            if team_id == 404
            else {"mcp": set(), "custom_api": set()}
        )
    )
    try:
        create_response = client.post(
            "/api/chat/task/create",
            headers=member_headers,
            json={
                "title": "values team shared task",
                "description": "d",
                "agent_id": agent_id,
            },
        )
        assert create_response.status_code == 200, create_response.text
        task_id = int(create_response.json()["task_id"])

        response = client.post(
            _values_url(task_id),
            headers=member_headers,
            json={
                "items": [
                    {
                        "connector_ref": {
                            "connector_type": "mcp",
                            "connector_id": server_id,
                        },
                        "context": {"auth_token": "x"},
                    }
                ]
            },
        )
        assert response.status_code == 200, response.text
    finally:
        set_agent_team_scope_hook(None)
        connector_team_scope.set_connector_team_hooks()


def test_values_endpoint_uses_runtime_agent_resolution_for_team_scope(
    e2e_db: None,
) -> None:
    """The values-endpoint half of the runtime agent resolution: a task
    whose agent is a workforce-generated manager agent, but for which no
    matching ``WorkforceRun`` exists, gets its connector scope from
    ``_load_agent_for_task_runtime`` returning ``None`` -- the same outcome
    a turn would get -- so a connector reachable only through the agent's
    team is neither writable here nor listed in the response, while a
    personally linked connector is both. Resolving the agent from
    ``task.agent_id`` directly would open this endpoint to a connector the
    turn itself will never resolve, and make its report disagree with the
    task-keyed read endpoint's for the same task.

    The task is built directly for the same reason the read endpoint's
    counterpart is: ``POST /task/create`` 404s for a workforce-generated
    manager agent, so a task with this agent can only exist as a row the
    workforce side creates.
    """
    db = _db_session()
    try:
        caller = _create_user(db, "wf-values-owner")
        db.flush()
        personal = _mcp_server_with_context_schema(
            db,
            caller,
            name="wf-values-personal-server",
            context_schema={"auth_token": {"type": "string", "required": True}},
            url="https://example.com/wf-values-personal/mcp",
        )
        team_shared = MCPServer(
            name="wf-values-team-shared-server",
            description="wf-values-team-shared-server description",
            managed="external",
            transport="streamable_http",
            url="https://example.com/wf-values-team/mcp",
            runtime_input_schema={
                "context": {"auth_token": {"type": "string", "required": True}}
            },
            runtime_bindings=[
                {
                    "source": {"input_type": "context", "key": "auth_token"},
                    "target": {"target_type": "mcp_meta", "key": "auth_token"},
                }
            ],
        )
        db.add(team_shared)
        db.flush()
        # No UserMCPServer link for `caller` at all -- reachable only
        # through the team hook below.
        personal_id = int(personal.id)
        team_shared_id = int(team_shared.id)
        agent = Agent(
            user_id=caller.id,
            name="WF Values Manager Agent",
            instructions="i",
            execution_mode="balanced",
            status=AgentStatus.PUBLISHED,
            tool_categories=["mcp"],
            team_id=505,
            origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
        )
        db.add(agent)
        db.flush()
        agent_id = int(agent.id)
        ordered_ids = sorted([personal_id, team_shared_id])
        task = Task(
            user_id=caller.id,
            agent_id=agent_id,
            title="workforce manager values task",
            description="d",
            status=TaskStatus.PENDING,
            connector_runtime_selected_refs=[
                {"connector_type": "mcp", "connector_id": ref_id}
                for ref_id in ordered_ids
            ],
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
        caller_headers = _auth_headers_for_user(caller)
    finally:
        db.close()

    def _hook(db: Session, *, team_id: int) -> dict[str, set[int]]:
        if team_id == 505:
            return {"mcp": {team_shared_id}, "custom_api": set()}
        return {"mcp": set(), "custom_api": set()}

    connector_team_scope.set_connector_team_hooks(team_visibility=_hook)
    try:
        personal_response = client.post(
            _values_url(task_id),
            headers=caller_headers,
            json={
                "items": [
                    {
                        "connector_ref": {
                            "connector_type": "mcp",
                            "connector_id": personal_id,
                        },
                        "context": {"auth_token": "x"},
                    }
                ]
            },
        )
        team_response = client.post(
            _values_url(task_id),
            headers=caller_headers,
            json={
                "items": [
                    {
                        "connector_ref": {
                            "connector_type": "mcp",
                            "connector_id": team_shared_id,
                        },
                        "context": {"auth_token": "x"},
                    }
                ]
            },
        )
    finally:
        connector_team_scope.set_connector_team_hooks()

    assert personal_response.status_code == 200, personal_response.text
    refs_seen = [
        item["connector_ref"] for item in personal_response.json()["connectors"]
    ]
    assert refs_seen == [{"connector_type": "mcp", "connector_id": personal_id}]

    # The team-shared connector is outside the scope a turn would resolve,
    # so it is not writable here either -- the same uniform "not found"
    # the endpoint answers for any ref the caller cannot see.
    assert team_response.status_code == 404, team_response.text
    assert _context_row_count(task_id) == 1
