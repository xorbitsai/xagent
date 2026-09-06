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
from sqlalchemy.orm import Session

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
