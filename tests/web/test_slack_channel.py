from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import pytest
from fastapi import BackgroundTasks, HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

from xagent.web.api.auth import create_access_token, verify_token
from xagent.web.api.channel import (
    SLACK_OAUTH_SCOPES,
    create_user_channel,
    slack_oauth_callback,
    start_slack_oauth,
    trigger_slack_sync,
)
from xagent.web.channels.slack.bot import (
    SlackBotInstance,
    SlackChannelManager,
    SlackOAuthSocketGateway,
)
from xagent.web.channels.slack.utils import markdown_to_slack, strip_slack_file_refs
from xagent.web.models.database import Base
from xagent.web.models.task import TaskStatus
from xagent.web.models.user import User
from xagent.web.models.user_channel import UserChannel
from xagent.web.schemas.user_channel import UserChannelCreate
from xagent.web.services.channel_runtime import ChannelConfigSnapshot
from xagent.web.services.task_execution_context_service import (
    TaskExecutionRecoverySnapshot,
)
from xagent.web.services.task_lease_service import TaskLease


def make_bot() -> SlackBotInstance:
    bot = object.__new__(SlackBotInstance)
    bot.channel_id = 7
    bot.channel_name = "Support Slack"
    bot.bot_user_id = "U_BOT"
    bot.web_client = object()  # type: ignore[assignment]
    bot.active_tasks = {}
    bot.event_queues = {}
    bot.event_tasks = {}
    bot._recent_event_ids = []
    bot._recent_event_id_set = set()
    bot._accepting = True
    return bot


def test_slack_markdown_and_file_refs_are_projected_for_transport() -> None:
    text, refs = strip_slack_file_refs(
        "Done\n\n![chart](file:image-1)\n[report](file:report-1)"
    )

    assert text == "Done"
    assert [(ref.file_id, ref.label, ref.is_image) for ref in refs] == [
        ("image-1", "chart", True),
        ("report-1", "report", False),
    ]
    assert markdown_to_slack("**Done** [docs](https://example.com?a=1&b=2)") == (
        "*Done* <https://example.com?a=1&b=2|docs>"
    )


def test_create_slack_channel_encrypts_tokens_and_schedules_sync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'slack-channel.db'}")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as db:
        user = User(username="slack-owner", password_hash="hash")
        db.add(user)
        db.commit()
        db.refresh(user)
        background_tasks = BackgroundTasks()
        monkeypatch.setattr(
            "xagent.web.api.channel.get_slack_bot_name_sync",
            lambda _token: "Xagent Slack",
        )

        channel = create_user_channel(
            UserChannelCreate(
                channel_type="slack",
                channel_name=None,
                config={
                    "bot_token": "xoxb-plain",
                    "app_token": "xapp-plain",
                    "allowed_users": ["U1"],
                },
                is_active=True,
            ),
            background_tasks,
            user,
            db,
        )

        assert channel.channel_name == "Xagent Slack"
        assert channel.config["bot_token"] == "xoxb-plain"
        assert channel.config["app_token"] == "xapp-plain"
        assert channel._config["bot_token"] != "xoxb-plain"
        assert channel._config["app_token"] != "xapp-plain"
        assert len(background_tasks.tasks) == 1
        assert background_tasks.tasks[0].func is trigger_slack_sync
    engine.dispose()


def test_slack_oauth_channel_cannot_be_created_through_manual_api() -> None:
    with pytest.raises(HTTPException, match="authorization flow"):
        create_user_channel(
            UserChannelCreate(
                channel_type="slack",
                channel_name="Forged OAuth Channel",
                config={
                    "installation_mode": "oauth",
                    "bot_token": "xoxb-not-from-callback",
                },
                is_active=True,
            ),
            BackgroundTasks(),
            SimpleNamespace(id=1),
            None,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_socket_listener_acks_immediately_and_deduplicates_events() -> None:
    bot = make_bot()
    processed: list[tuple[str, str]] = []

    async def process_event(
        conversation_key: str,
        _payload: dict[str, Any],
        event: dict[str, Any],
    ) -> None:
        processed.append((conversation_key, str(event["text"])))

    bot._process_event = process_event  # type: ignore[method-assign]

    class FakeSocketClient:
        def __init__(self) -> None:
            self.responses: list[Any] = []

        async def send_socket_mode_response(self, response: Any) -> None:
            self.responses.append(response)

    client = FakeSocketClient()
    request = SimpleNamespace(
        envelope_id="envelope-1",
        type="events_api",
        payload={
            "event_id": "event-1",
            "team_id": "T1",
            "event": {
                "type": "message",
                "channel_type": "im",
                "channel": "D1",
                "user": "U1",
                "ts": "1.0",
                "text": "hello",
            },
        },
    )

    await bot._handle_socket_request(client, request)  # type: ignore[arg-type]
    await bot._handle_socket_request(client, request)  # type: ignore[arg-type]
    await asyncio.gather(*bot.event_tasks.values())

    assert len(client.responses) == 2
    assert processed == [("T1:D1:U1:direct", "hello")]


@pytest.mark.asyncio
async def test_slack_manager_starts_active_configured_bots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SlackChannelManager()
    snapshot = ChannelConfigSnapshot(
        channel_id=9,
        channel_name="Workspace Slack",
        config_items=(
            ("bot_token", "xoxb-test"),
            ("app_token", "xapp-test"),
        ),
    )

    async def load_configs(**kwargs: Any) -> tuple[ChannelConfigSnapshot, ...]:
        assert kwargs["channel_type"] == "slack"
        if kwargs["required_config_keys"] == ("bot_token", "app_token"):
            return (snapshot,)
        assert kwargs["required_config_keys"] == (
            "bot_token",
            "team_id",
            "bot_user_id",
            "installation_mode",
        )
        return ()

    started: list[dict[str, Any]] = []

    async def start_bot(**kwargs: Any) -> None:
        started.append(kwargs)

    monkeypatch.setattr(
        "xagent.web.channels.slack.bot.load_active_channel_configs",
        load_configs,
    )
    manager._start_bot_for_token = start_bot  # type: ignore[method-assign]

    await manager._sync_bots_async()

    assert started == [
        {
            "bot_token": "xoxb-test",
            "app_token": "xapp-test",
            "channel_id": 9,
            "channel_name": "Workspace Slack",
        }
    ]


def _request(
    *,
    path: str,
    query: dict[str, str] | None = None,
    origin: str = "https://app.example.com",
) -> Request:
    query_string = urlencode(query or {}).encode()
    return Request(
        {
            "type": "http",
            "method": "GET",
            "scheme": "https",
            "server": ("api.example.com", 443),
            "path": path,
            "query_string": query_string,
            "headers": [(b"origin", origin.encode())],
        }
    )


def test_start_slack_oauth_returns_signed_workspace_authorization_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "xagent.web.api.channel.get_slack_client_id",
        lambda: "client-id",
    )
    monkeypatch.setattr(
        "xagent.web.api.channel.get_slack_client_secret",
        lambda: "client-secret",
    )
    monkeypatch.setattr(
        "xagent.web.api.channel.get_slack_app_token",
        lambda: "xapp-shared",
    )
    monkeypatch.setattr(
        "xagent.web.api.channel.get_slack_oauth_redirect_uri",
        lambda: "https://api.example.com/api/channels/slack/oauth/callback",
    )
    monkeypatch.setattr(
        "xagent.web.api.channel.get_app_base_url",
        lambda: "https://app.example.com",
    )

    result = start_slack_oauth(
        _request(path="/api/channels/slack/oauth/start"),
        SimpleNamespace(id=17),
    )

    parsed = urlparse(result["authorize_url"])
    query = parse_qs(parsed.query)
    assert parsed.scheme == "https"
    assert parsed.netloc == "slack.com"
    assert result["callback_origin"] == "https://api.example.com"
    assert query["client_id"] == ["client-id"]
    assert query["scope"] == [",".join(SLACK_OAUTH_SCOPES)]
    state = verify_token(query["state"][0])
    assert state is not None
    assert state["type"] == "slack_oauth_state"
    assert state["user_id"] == 17
    assert state["frontend_origin"] == "https://app.example.com"


@pytest.mark.asyncio
async def test_slack_oauth_callback_creates_encrypted_workspace_channel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'slack-oauth.db'}")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    monkeypatch.setattr(
        "xagent.web.api.channel.get_slack_client_id",
        lambda: "client-id",
    )
    monkeypatch.setattr(
        "xagent.web.api.channel.get_slack_client_secret",
        lambda: "client-secret",
    )
    monkeypatch.setattr(
        "xagent.web.api.channel.get_slack_oauth_redirect_uri",
        lambda: "https://api.example.com/api/channels/slack/oauth/callback",
    )

    token_data = {
        "ok": True,
        "access_token": "xoxb-oauth-secret",
        "scope": ",".join(SLACK_OAUTH_SCOPES),
        "bot_user_id": "U_BOT",
        "app_id": "A_APP",
        "team": {"id": "T_WORKSPACE", "name": "Acme"},
        "enterprise": None,
    }

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return token_data

    class FakeAsyncClient:
        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        async def post(self, *_args: Any, **_kwargs: Any) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(
        "xagent.web.api.channel.httpx.AsyncClient",
        FakeAsyncClient,
    )

    with SessionLocal() as db:
        user = User(username="slack-oauth-owner", password_hash="hash")
        db.add(user)
        db.commit()
        db.refresh(user)
        state = create_access_token(
            data={
                "type": "slack_oauth_state",
                "user_id": int(user.id),
                "frontend_origin": "https://app.example.com",
            },
            expires_delta=timedelta(minutes=10),
        )
        background_tasks = BackgroundTasks()

        response = await slack_oauth_callback(
            _request(
                path="/api/channels/slack/oauth/callback",
                query={"code": "oauth-code", "state": state},
            ),
            background_tasks,
            db,
        )

        channel = db.query(UserChannel).one()
        assert response.status_code == 200
        assert "slack-oauth-success" in response.body.decode()
        assert channel.channel_name == "Acme"
        assert channel.config["installation_mode"] == "oauth"
        assert channel.config["team_id"] == "T_WORKSPACE"
        assert channel.config["bot_token"] == "xoxb-oauth-secret"
        assert channel._config["bot_token"] != "xoxb-oauth-secret"
        assert len(background_tasks.tasks) == 1
        assert background_tasks.tasks[0].func is trigger_slack_sync
    engine.dispose()


@pytest.mark.asyncio
async def test_shared_slack_gateway_acks_and_routes_by_workspace() -> None:
    handled: list[dict[str, Any]] = []

    class FakeBot:
        async def handle_events_api_payload(self, payload: dict[str, Any]) -> None:
            handled.append(payload)

    bot = FakeBot()
    gateway = SlackOAuthSocketGateway(
        app_token="xapp-shared",
        bot_lookup=lambda payload: bot if payload.get("team_id") == "T1" else None,  # type: ignore[arg-type]
    )

    class FakeSocketClient:
        def __init__(self) -> None:
            self.responses: list[Any] = []

        async def send_socket_mode_response(self, response: Any) -> None:
            self.responses.append(response)

    client = FakeSocketClient()
    request = SimpleNamespace(
        envelope_id="envelope-1",
        type="events_api",
        payload={
            "team_id": "T1",
            "event": {
                "type": "message",
                "channel_type": "im",
                "channel": "D1",
                "user": "U1",
                "text": "hello",
            },
        },
    )

    await gateway._handle_socket_request(client, request)  # type: ignore[arg-type]

    assert len(client.responses) == 1
    assert handled == [request.payload]


@pytest.mark.asyncio
async def test_slack_manager_registers_oauth_workspace_on_shared_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SlackChannelManager()
    oauth_snapshot = ChannelConfigSnapshot(
        channel_id=12,
        channel_name="Acme",
        config_items=(
            ("bot_token", "xoxb-acme"),
            ("team_id", "T_ACME"),
            ("bot_user_id", "U_BOT"),
            ("installation_mode", "oauth"),
        ),
    )

    async def load_configs(**kwargs: Any) -> tuple[ChannelConfigSnapshot, ...]:
        if kwargs["required_config_keys"] == ("bot_token", "app_token"):
            return ()
        return (oauth_snapshot,)

    gateway_tokens: list[str] = []

    async def ensure_gateway(app_token: str) -> None:
        gateway_tokens.append(app_token)

    monkeypatch.setattr(
        "xagent.web.channels.slack.bot.load_active_channel_configs",
        load_configs,
    )
    monkeypatch.setattr(
        "xagent.web.channels.slack.bot.get_slack_app_token",
        lambda: "xapp-shared",
    )
    manager._ensure_oauth_gateway = ensure_gateway  # type: ignore[method-assign]

    await manager._sync_bots_async()

    assert set(manager.oauth_bots) == {"T_ACME"}
    assert manager.oauth_bots["T_ACME"].bot_token == "xoxb-acme"
    assert gateway_tokens == ["xapp-shared"]


def test_slack_only_handles_mentions_in_shared_channels() -> None:
    bot = make_bot()

    assert bot._should_handle_event(
        {
            "type": "app_mention",
            "channel": "C1",
            "user": "U1",
            "text": "<@U_BOT> hello",
        }
    )
    assert not bot._should_handle_event(
        {
            "type": "message",
            "channel": "C1",
            "channel_type": "channel",
            "user": "U1",
            "text": "hello",
        }
    )
    bot.active_tasks["T1:C1:U1:1.0"] = 42
    assert bot._should_handle_event(
        {
            "type": "message",
            "channel": "C1",
            "channel_type": "channel",
            "user": "U1",
            "thread_ts": "1.0",
            "text": "thread follow-up",
        }
    )
    assert not bot._should_handle_event(
        {
            "type": "message",
            "channel": "D1",
            "channel_type": "im",
            "bot_id": "B1",
            "text": "loop",
        }
    )
    assert (
        bot._message_text({"text": "<@U_BOT> first line\nsecond line"})
        == "first line\nsecond line"
    )


@pytest.mark.asyncio
async def test_successful_slack_turn_reuses_channel_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = make_bot()
    bot._save_active_tasks = lambda: None  # type: ignore[method-assign]
    lease = TaskLease(task_id=45, runner_id="runner-a", run_id="run-a")
    finalized: list[tuple[TaskStatus, str]] = []

    class FakeManagedLease:
        heartbeat_task = None

        def __init__(self) -> None:
            self.lease = lease
            self.closed = False

        async def finalize_result(
            self,
            *,
            status: TaskStatus,
            assistant_content: str = "",
            **_kwargs: Any,
        ) -> bool:
            finalized.append((status, assistant_content))
            return True

        async def close(self) -> bool:
            self.closed = True
            return True

    managed = FakeManagedLease()

    async def prepare(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            user_id=5,
            task_id=45,
            is_new_task=True,
            managed_lease=managed,
        )

    class FakeTracer:
        def __init__(self) -> None:
            self.handlers: list[Any] = []

        def add_handler(self, handler: Any) -> None:
            self.handlers.append(handler)

    agent_service = SimpleNamespace(
        workspace=None,
        tracer=FakeTracer(),
        set_conversation_history=lambda _messages: None,
        set_execution_context_messages=lambda _messages: None,
        set_recovered_skill_context=lambda _context: None,
    )

    class FakeAgentManager:
        async def get_agent_for_task(self, *_args: Any, **_kwargs: Any) -> Any:
            return agent_service

        async def execute_task(self, **_kwargs: Any) -> dict[str, Any]:
            return {"success": True, "output": "Slack reply"}

    persisted: list[dict[str, Any]] = []
    final_messages: list[dict[str, Any]] = []

    async def persist(**kwargs: Any) -> None:
        persisted.append(kwargs)

    async def send_text(
        _channel_id: str,
        _text: str,
        *,
        thread_ts: str | None,
    ) -> str:
        assert thread_ts is None
        return "loading-ts"

    async def send_final_text(**kwargs: Any) -> None:
        final_messages.append(kwargs)

    monkeypatch.setattr(
        "xagent.web.channels.slack.bot.prepare_channel_task",
        prepare,
    )
    monkeypatch.setattr(
        "xagent.web.channels.slack.bot.load_task_setup_snapshot_sync",
        lambda *_args: SimpleNamespace(
            runtime_user=None,
            conversation_history=(),
            execution_recovery=TaskExecutionRecoverySnapshot(),
        ),
    )
    monkeypatch.setattr(
        "xagent.web.channels.slack.bot.get_agent_manager",
        lambda: FakeAgentManager(),
    )
    monkeypatch.setattr(
        "xagent.web.channels.slack.bot.persist_channel_user_message",
        persist,
    )
    bot._send_text = send_text  # type: ignore[method-assign]
    bot._send_final_text = send_final_text  # type: ignore[method-assign]

    await bot._process_event(
        "T1:D1:U1:direct",
        {},
        {
            "type": "message",
            "channel_type": "im",
            "channel": "D1",
            "user": "U1",
            "ts": "1.0",
            "text": "hello",
        },
    )

    assert bot.active_tasks == {"T1:D1:U1:direct": 45}
    assert persisted[0]["content"] == "hello"
    assert finalized == [(TaskStatus.COMPLETED, "Slack reply")]
    assert final_messages == [
        {
            "channel_id": "D1",
            "thread_ts": None,
            "loading_ts": "loading-ts",
            "text": "Slack reply",
        }
    ]
    assert managed.closed is True
