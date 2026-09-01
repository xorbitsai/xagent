from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import pytest
from fastapi import BackgroundTasks, HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

from xagent.web.api.auth import verify_token
from xagent.web.api.channel import (
    SLACK_OAUTH_SCOPES,
    _script_safe_json,
    create_user_channel,
    slack_oauth_callback,
    start_slack_oauth,
    trigger_slack_sync,
    update_user_channel,
)
from xagent.web.channels.slack.bot import (
    SlackBotInstance,
    SlackChannelManager,
    SlackOAuthSocketGateway,
    _RetryBackoff,
)
from xagent.web.channels.slack.utils import (
    SlackFileUrlError,
    markdown_to_slack,
    strip_slack_file_refs,
    validate_slack_file_url,
)
from xagent.web.models.database import Base
from xagent.web.models.task import TaskStatus
from xagent.web.models.user import User
from xagent.web.models.user_channel import SlackOAuthFlowState, UserChannel
from xagent.web.schemas.user_channel import UserChannelCreate, UserChannelUpdate
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
    assert [(ref.file_id, ref.label) for ref in refs] == [
        ("image-1", "chart"),
        ("report-1", "report"),
    ]
    assert markdown_to_slack("**Done** [docs](https://example.com?a=1&b=2)") == (
        "*Done* <https://example.com?a=1&b=2|docs>"
    )


@pytest.mark.parametrize(
    ("payload", "forbidden"),
    [
        # A raw '>' inside the link URL would close the mrkdwn token early and
        # let the rest be parsed as a live @channel broadcast.
        ("[x](https://a.com|hack><!channel>)", "<!channel>"),
        ("[x](https://a.com|hack><!here>)", "<!here>"),
        ("[x](https://a.com|hack><!everyone>)", "<!everyone>"),
        # A raw '|' would forge the label separator.
        ("[x](https://a.com|<@U123>)", "<@U123>"),
    ],
)
def test_markdown_link_cannot_inject_mrkdwn_control_tokens(
    payload: str, forbidden: str
) -> None:
    converted = markdown_to_slack(payload)

    assert forbidden not in converted
    # Exactly one mrkdwn token: the link itself, with no stray delimiters.
    assert converted.count("<") == 1
    assert converted.count(">") == 1


@pytest.mark.asyncio
async def test_final_text_chunks_never_split_a_mrkdwn_token() -> None:
    bot = make_bot()
    sent: list[str] = []

    async def fake_send(channel_id: str, text: str, *, thread_ts: str | None) -> None:
        sent.append(text)

    bot._send_mrkdwn = fake_send  # type: ignore[method-assign]

    # A long body whose links sit near every chunk boundary; converting after
    # splitting keeps each <url|label> token intact.
    link = "[docs](https://example.com/a?x=1&y=2)"
    body = "\n".join(f"line {i} {link}" for i in range(400))

    await bot._send_final_text(
        channel_id="C1", thread_ts=None, loading_ts=None, text=body
    )

    assert len(sent) > 1
    for chunk in sent:
        assert len(chunk) <= 3900
        # Balanced tokens: no chunk ends mid-<url|label>.
        assert chunk.count("<") == chunk.count(">")
        # Every link in the chunk is a complete, converted mrkdwn token.
        assert "[docs](" not in chunk
        assert chunk.count("<https://example.com/a?x=1&y=2|docs>") == chunk.count("<")


def test_markdown_bare_broadcast_text_stays_inert() -> None:
    assert markdown_to_slack("ping <!channel> now") == "ping &lt;!channel&gt; now"


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


@pytest.mark.parametrize(
    "hostile_key",
    ["team_id", "bot_user_id", "slack_app_id", "enterprise_id", "scope"],
)
def test_create_slack_channel_rejects_server_managed_config(hostile_key: str) -> None:
    # The update path already refuses these; POST must match, or a manual row
    # could be created carrying an attacker-chosen workspace identity.
    with pytest.raises(HTTPException, match="managed by the authorization flow"):
        create_user_channel(
            UserChannelCreate(
                channel_type="slack",
                channel_name="Forged",
                config={
                    "bot_token": "xoxb-x",
                    "app_token": "xapp-x",
                    hostile_key: "injected",
                },
                is_active=True,
            ),
            BackgroundTasks(),
            SimpleNamespace(id=1),
            None,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "config",
    [
        {"bot_token": "xoxb-only"},
        {"app_token": "xapp-only"},
    ],
)
def test_create_slack_channel_requires_both_tokens(config: dict[str, Any]) -> None:
    # _sync_manual_bots skips rows missing either token, so such a row would be
    # written active and then silently never start.
    with pytest.raises(HTTPException, match="bot token and an app token"):
        create_user_channel(
            UserChannelCreate(
                channel_type="slack",
                channel_name="Half configured",
                config=config,
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
async def test_event_queue_survives_a_failing_event() -> None:
    bot = make_bot()
    processed: list[str] = []

    async def process_event(
        _conversation_key: str,
        _payload: dict[str, Any],
        event: dict[str, Any],
    ) -> None:
        text = str(event["text"])
        if text == "boom":
            raise RuntimeError("slack api failed")
        processed.append(text)

    bot._process_event = process_event  # type: ignore[method-assign]

    conversation_key = "T1:D1:U1:direct"
    bot.event_queues[conversation_key] = [
        ({}, {"text": "boom"}),
        ({}, {"text": "still delivered"}),
    ]

    await bot._process_event_queue(conversation_key)

    assert processed == ["still delivered"]
    assert conversation_key not in bot.event_queues
    assert conversation_key not in bot.event_tasks


@pytest.mark.asyncio
async def test_slack_manager_starts_active_configured_bots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SlackChannelManager()
    manual_snapshot = ChannelConfigSnapshot(
        channel_id=9,
        channel_name="Workspace Slack",
        config_items=(
            ("app_token", "xapp-test"),
            ("bot_token", "xoxb-test"),
            ("installation_mode", "manual"),
        ),
    )
    # A hostile or corrupted row carrying BOTH manual and OAuth markers must
    # resolve to exactly one runtime path (its declared installation_mode).
    ambiguous_oauth_snapshot = ChannelConfigSnapshot(
        channel_id=10,
        channel_name="Ambiguous OAuth",
        config_items=(
            ("app_token", "xapp-smuggled"),
            ("bot_token", "xoxb-oauth"),
            ("bot_user_id", "U_BOT"),
            ("installation_mode", "oauth"),
            ("team_id", "T1"),
        ),
    )

    async def load_configs(**kwargs: Any) -> tuple[ChannelConfigSnapshot, ...]:
        assert kwargs["channel_type"] == "slack"
        assert kwargs["required_config_keys"] == ("bot_token",)
        return (manual_snapshot, ambiguous_oauth_snapshot)

    started: list[dict[str, Any]] = []

    async def start_bot(**kwargs: Any) -> None:
        started.append(kwargs)

    monkeypatch.setattr(
        "xagent.web.channels.slack.bot.load_active_channel_configs",
        load_configs,
    )
    manager._start_bot_for_token = start_bot  # type: ignore[method-assign]

    await manager._sync_bots_async()

    # Only the manual row starts a manual bot; the ambiguous row is routed
    # exclusively to the OAuth path (which stays idle without a shared
    # XAGENT_SLACK_APP_TOKEN) instead of starting a second manual instance.
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


def _patch_slack_oauth_config(monkeypatch: pytest.MonkeyPatch) -> None:
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
        "xagent.web.api.channel.has_production_channel_encryption_key",
        lambda: True,
    )


def _oauth_test_session(tmp_path: Path, name: str) -> tuple[Any, Any]:
    engine = create_engine(f"sqlite:///{tmp_path / name}")
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine)


def test_start_slack_oauth_returns_signed_workspace_authorization_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_slack_oauth_config(monkeypatch)
    monkeypatch.setattr(
        "xagent.web.api.channel.get_app_base_url",
        lambda: "https://app.example.com",
    )
    engine, SessionLocal = _oauth_test_session(tmp_path, "slack-oauth-start.db")

    with SessionLocal() as db:
        result = start_slack_oauth(
            _request(path="/api/channels/slack/oauth/start"),
            SimpleNamespace(id=17),
            db,
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

        flow_state = db.query(SlackOAuthFlowState).one()
        assert flow_state.nonce == state["nonce"]
        assert flow_state.consumed_at is None
    engine.dispose()


@pytest.mark.asyncio
async def test_slack_oauth_callback_creates_encrypted_workspace_channel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine, SessionLocal = _oauth_test_session(tmp_path, "slack-oauth.db")
    _patch_slack_oauth_config(monkeypatch)

    token_data = {
        "ok": True,
        "access_token": "xoxb-oauth-secret",
        "scope": ",".join(SLACK_OAUTH_SCOPES),
        "bot_user_id": "U_BOT",
        "app_id": "A_APP",
        "team": {"id": "T_WORKSPACE", "name": "Acme"},
        "authed_user": {"id": "U_INSTALLER"},
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
        start_result = start_slack_oauth(
            _request(path="/api/channels/slack/oauth/start"),
            SimpleNamespace(id=int(user.id)),
            db,
        )
        state = parse_qs(urlparse(start_result["authorize_url"]).query)["state"][0]
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
        assert channel.config["allowed_users"] == ["U_INSTALLER"]
        assert channel._config["bot_token"] != "xoxb-oauth-secret"
        assert len(background_tasks.tasks) == 1
        assert background_tasks.tasks[0].func is trigger_slack_sync
    engine.dispose()


@pytest.mark.parametrize(
    "url",
    [
        "https://files.slack.com/files-pri/T1-F1/report.pdf",
        "https://slack.com/files-pri/T1-F1/report.pdf",
        "https://a.b.slack-edge.com/x.png",
        "https://downloads.slack-files.com/x.png",
    ],
)
def test_validate_slack_file_url_accepts_slack_hosts(url: str) -> None:
    assert validate_slack_file_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example.com/steal",
        # Suffix-confusion: must not match on a bare substring.
        "https://slack.com.evil.example.com/steal",
        "https://notslack-files.com/x",
        # Non-public targets reachable from the backend.
        "https://169.254.169.254/latest/meta-data/",
        "https://127.0.0.1/admin",
        # Plaintext and credential-bearing URLs.
        "http://files.slack.com/x",
        "https://user:pass@files.slack.com/x",
    ],
)
def test_validate_slack_file_url_rejects_non_slack_targets(url: str) -> None:
    with pytest.raises(SlackFileUrlError):
        validate_slack_file_url(url)


@pytest.mark.asyncio
async def test_slack_file_download_rejects_off_slack_url() -> None:
    bot = make_bot()
    bot.bot_token = "xoxb-secret"

    # A hostile url_private_download in an inbound event must never be
    # fetched, because the request would carry the workspace bot token.
    with pytest.raises(SlackFileUrlError):
        await bot._download_slack_file(
            {"id": "F1", "url_private_download": "https://evil.example.com/steal"},
            Path("/tmp"),
        )


@pytest.mark.asyncio
async def test_slack_file_download_rejects_off_slack_redirect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bot = make_bot()
    bot.bot_token = "xoxb-secret"
    requested: list[str] = []

    class FakeStream:
        def __init__(self, url: str) -> None:
            self.url = url
            self.is_redirect = True
            self.headers = {"location": "https://evil.example.com/steal"}

        async def __aenter__(self) -> "FakeStream":
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

    class FakeClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        def stream(self, _method: str, url: str) -> FakeStream:
            requested.append(url)
            return FakeStream(url)

    monkeypatch.setattr("xagent.web.channels.slack.bot.httpx.AsyncClient", FakeClient)
    target_dir = tmp_path / "slack-input"

    with pytest.raises(SlackFileUrlError):
        await bot._download_slack_file(
            {
                "id": "F1",
                "name": "report.pdf",
                "url_private_download": "https://files.slack.com/files-pri/T1-F1/r.pdf",
            },
            target_dir,
        )

    # The off-Slack redirect target was never requested, and the partial
    # download was cleaned up.
    assert requested == ["https://files.slack.com/files-pri/T1-F1/r.pdf"]
    assert list(target_dir.iterdir()) == []


def test_retry_backoff_grows_and_caps() -> None:
    backoff = _RetryBackoff(initial_seconds=10.0, max_seconds=300.0)

    delays = []
    for _ in range(8):
        delays.append(backoff.delay)
        backoff.attempts += 1

    assert delays[:5] == [10.0, 20.0, 40.0, 80.0, 160.0]
    # Capped, so a permanently broken config stops logging every 10 seconds.
    assert delays[5:] == [300.0, 300.0, 300.0]
    assert backoff.exhausted_quiet_threshold is True

    backoff.reset()
    assert backoff.delay == 10.0
    assert backoff.exhausted_quiet_threshold is False


@pytest.mark.asyncio
async def test_oauth_claim_released_when_token_exchange_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine, SessionLocal = _oauth_test_session(tmp_path, "slack-oauth-retry.db")
    _patch_slack_oauth_config(monkeypatch)

    class FailingClient:
        async def __aenter__(self) -> "FailingClient":
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        async def post(self, *_args: Any, **_kwargs: Any) -> Any:
            raise httpx.ConnectError("slack unreachable")

    monkeypatch.setattr("xagent.web.api.channel.httpx.AsyncClient", FailingClient)

    with SessionLocal() as db:
        user = User(username="retry-owner", password_hash="hash")
        db.add(user)
        db.commit()
        db.refresh(user)
        start_result = start_slack_oauth(
            _request(path="/api/channels/slack/oauth/start"),
            SimpleNamespace(id=int(user.id)),
            db,
        )
        state = parse_qs(urlparse(start_result["authorize_url"]).query)["state"][0]

        response = await slack_oauth_callback(
            _request(
                path="/api/channels/slack/oauth/callback",
                query={"code": "oauth-code", "state": state},
            ),
            BackgroundTasks(),
            db,
        )

        assert response.status_code == 400
        # A Slack-side outage must not burn the nonce: the same link retries.
        flow_state = db.query(SlackOAuthFlowState).one()
        assert flow_state.consumed_at is None
    engine.dispose()


def test_script_safe_json_neutralizes_script_breakout() -> None:
    encoded = _script_safe_json({"message": "</script><script>alert(1)</script>"})
    assert "<" not in encoded
    assert ">" not in encoded
    assert "\\u003c" in encoded


@pytest.mark.asyncio
async def test_oauth_callback_error_param_cannot_inject_script(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine, SessionLocal = _oauth_test_session(tmp_path, "slack-oauth-xss.db")
    _patch_slack_oauth_config(monkeypatch)

    with SessionLocal() as db:
        start_result = start_slack_oauth(
            _request(path="/api/channels/slack/oauth/start"),
            SimpleNamespace(id=1),
            db,
        )
        state = parse_qs(urlparse(start_result["authorize_url"]).query)["state"][0]

        response = await slack_oauth_callback(
            _request(
                path="/api/channels/slack/oauth/callback",
                query={
                    "state": state,
                    "error": "</script><script>alert(1)</script>",
                },
            ),
            BackgroundTasks(),
            db,
        )

        body = response.body.decode()
        assert response.status_code == 400
        assert "slack-oauth-error" in body
        assert "<script>alert" not in body
        assert "alert(1)" not in body
        assert "Content-Security-Policy" in response.headers
    engine.dispose()


@pytest.mark.asyncio
async def test_oauth_state_is_single_use(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine, SessionLocal = _oauth_test_session(tmp_path, "slack-oauth-replay.db")
    _patch_slack_oauth_config(monkeypatch)

    with SessionLocal() as db:
        start_result = start_slack_oauth(
            _request(path="/api/channels/slack/oauth/start"),
            SimpleNamespace(id=1),
            db,
        )
        state = parse_qs(urlparse(start_result["authorize_url"]).query)["state"][0]

        first = await slack_oauth_callback(
            _request(
                path="/api/channels/slack/oauth/callback",
                query={"state": state, "error": "access_denied"},
            ),
            BackgroundTasks(),
            db,
        )
        replay = await slack_oauth_callback(
            _request(
                path="/api/channels/slack/oauth/callback",
                query={"state": state, "error": "access_denied"},
            ),
            BackgroundTasks(),
            db,
        )

        assert "access_denied" in first.body.decode()
        assert replay.status_code == 400
        assert "invalid or expired" in replay.body.decode()
    engine.dispose()


def test_update_channel_rejects_oauth_identity_rewrites(tmp_path: Path) -> None:
    engine, SessionLocal = _oauth_test_session(tmp_path, "slack-put-guard.db")

    with SessionLocal() as db:
        user = User(username="slack-put-owner", password_hash="hash")
        db.add(user)
        db.commit()
        db.refresh(user)
        channel = UserChannel(
            user_id=int(user.id),
            channel_type="slack",
            channel_name="Acme",
            config={
                "installation_mode": "oauth",
                "bot_token": "xoxb-oauth",
                "team_id": "T_MINE",
                "bot_user_id": "U_BOT",
                "slack_app_id": "A_APP",
                "allowed_users": None,
            },
            is_active=True,
        )
        db.add(channel)
        db.commit()
        db.refresh(channel)

        for hostile_config in (
            {"team_id": "T_VICTIM"},
            {"bot_user_id": "U_EVIL"},
            {"installation_mode": "manual"},
            {"bot_token": "xoxb-attacker"},
        ):
            with pytest.raises(HTTPException) as exc_info:
                update_user_channel(
                    int(channel.id),
                    UserChannelUpdate(config=hostile_config),
                    BackgroundTasks(),
                    SimpleNamespace(id=int(user.id)),
                    db,
                )
            assert exc_info.value.status_code == 400

        updated = update_user_channel(
            int(channel.id),
            UserChannelUpdate(config={"allowed_users": ["U123"], "team_id": "T_MINE"}),
            BackgroundTasks(),
            SimpleNamespace(id=int(user.id)),
            db,
        )
        assert updated.config["allowed_users"] == ["U123"]
        assert updated.config["team_id"] == "T_MINE"
    engine.dispose()


@pytest.mark.asyncio
async def test_mention_and_message_pair_processed_once() -> None:
    bot = make_bot()
    processed: list[str] = []

    async def process_event(
        _conversation_key: str,
        _payload: dict[str, Any],
        event: dict[str, Any],
    ) -> None:
        processed.append(str(event["type"]))

    bot._process_event = process_event  # type: ignore[method-assign]
    bot.active_tasks["T1:C1:U1:1.0"] = 42

    base_event = {
        "channel": "C1",
        "channel_type": "channel",
        "user": "U1",
        "ts": "2.0",
        "thread_ts": "1.0",
        "text": "<@U_BOT> continue",
    }
    # Slack delivers one physical in-thread mention as two envelopes with
    # distinct event ids: an app_mention and a message event.
    await bot.handle_events_api_payload(
        {
            "event_id": "Ev-mention",
            "team_id": "T1",
            "event": {**base_event, "type": "app_mention"},
        }
    )
    await bot.handle_events_api_payload(
        {
            "event_id": "Ev-message",
            "team_id": "T1",
            "event": {**base_event, "type": "message"},
        }
    )
    await asyncio.gather(*bot.event_tasks.values())

    assert processed == ["app_mention"]


@pytest.mark.asyncio
async def test_app_uninstalled_deactivates_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = make_bot()
    deactivated: list[dict[str, Any]] = []

    def fake_deactivate(**kwargs: Any) -> bool:
        deactivated.append(kwargs)
        return True

    async def fake_db_io(func: Any) -> Any:
        return func()

    monkeypatch.setattr(
        "xagent.web.channels.slack.bot.deactivate_channel_sync",
        fake_deactivate,
    )
    monkeypatch.setattr(
        "xagent.web.channels.slack.bot.run_db_io_cancellation_safe",
        fake_db_io,
    )

    async def fake_sync() -> None:
        return None

    monkeypatch.setattr("xagent.web.api.channel.trigger_slack_sync", fake_sync)

    await bot.handle_events_api_payload(
        {"event_id": "Ev-1", "team_id": "T1", "event": {"type": "app_uninstalled"}}
    )
    await asyncio.sleep(0)

    assert bot._accepting is False
    # team_id is cleared too, so the dead row cannot block a later reinstall
    # of the same workspace by any user.
    assert deactivated == [
        {"channel_id": 7, "clear_config_keys": ("bot_token", "team_id")}
    ]


@pytest.mark.asyncio
async def test_uninstall_keeps_accepting_when_deactivation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = make_bot()

    def failing_deactivate(**_kwargs: Any) -> bool:
        raise RuntimeError("database unavailable")

    async def fake_db_io(func: Any) -> Any:
        return func()

    monkeypatch.setattr(
        "xagent.web.channels.slack.bot.deactivate_channel_sync",
        failing_deactivate,
    )
    monkeypatch.setattr(
        "xagent.web.channels.slack.bot.run_db_io_cancellation_safe",
        fake_db_io,
    )

    await bot.handle_events_api_payload(
        {"event_id": "Ev-1", "team_id": "T1", "event": {"type": "app_uninstalled"}}
    )

    # The row is still active, so a non-accepting instance would look healthy
    # to the manager's sync diff and never be replaced short of a restart.
    assert bot._accepting is True


@pytest.mark.asyncio
async def test_tokens_revoked_without_bot_token_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = make_bot()
    deactivated: list[dict[str, Any]] = []

    def fake_deactivate(**kwargs: Any) -> bool:
        deactivated.append(kwargs)
        return True

    async def fake_db_io(func: Any) -> Any:
        return func()

    monkeypatch.setattr(
        "xagent.web.channels.slack.bot.deactivate_channel_sync",
        fake_deactivate,
    )
    monkeypatch.setattr(
        "xagent.web.channels.slack.bot.run_db_io_cancellation_safe",
        fake_db_io,
    )

    # Slack sends tokens_revoked for a user-token revocation as well; the bot
    # token still works, so tearing the integration down would be wrong.
    await bot.handle_events_api_payload(
        {
            "event_id": "Ev-1",
            "team_id": "T1",
            "event": {"type": "tokens_revoked", "tokens": {"oauth": ["U1"]}},
        }
    )

    assert deactivated == []
    assert bot._accepting is True


def test_deactivated_row_does_not_block_reinstall_by_other_user(
    tmp_path: Path,
) -> None:
    from xagent.web.api.channel import _upsert_slack_oauth_channel

    engine, SessionLocal = _oauth_test_session(tmp_path, "slack-reinstall.db")

    with SessionLocal() as db:
        first = User(username="first-owner", password_hash="hash")
        second = User(username="second-owner", password_hash="hash")
        db.add_all([first, second])
        db.commit()
        db.refresh(first)
        db.refresh(second)

        # An uninstall left this row deactivated with its token cleared.
        dead_row = UserChannel(
            user_id=int(first.id),
            channel_type="slack",
            channel_name="Acme",
            config={
                "installation_mode": "oauth",
                "team_id": "T_ACME",
                "bot_user_id": "U_BOT",
            },
            is_active=False,
        )
        db.add(dead_row)
        db.commit()

        channel = _upsert_slack_oauth_channel(
            db,
            user_id=int(second.id),
            token_data={
                "access_token": "xoxb-new",
                "team": {"id": "T_ACME", "name": "Acme"},
                "bot_user_id": "U_BOT2",
                "authed_user": {"id": "U_INSTALLER2"},
            },
        )

        assert int(channel.user_id) == int(second.id)
        assert bool(channel.is_active) is True
        assert channel.config["bot_token"] == "xoxb-new"
    engine.dispose()


def test_missing_authed_user_rejects_install(tmp_path: Path) -> None:
    from xagent.web.api.channel import _upsert_slack_oauth_channel

    engine, SessionLocal = _oauth_test_session(tmp_path, "slack-no-installer.db")

    with SessionLocal() as db:
        user = User(username="installer-missing", password_hash="hash")
        db.add(user)
        db.commit()
        db.refresh(user)

        # Falling back to allowed_users=None here would open the channel to
        # the entire workspace.
        with pytest.raises(ValueError, match="installing user"):
            _upsert_slack_oauth_channel(
                db,
                user_id=int(user.id),
                token_data={
                    "access_token": "xoxb-new",
                    "team": {"id": "T_NEW", "name": "NewCo"},
                    "bot_user_id": "U_BOT",
                },
            )
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
    finalized_execution_results: list[Any] = []

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
            execution_result: Any = None,
            **_kwargs: Any,
        ) -> bool:
            finalized.append((status, assistant_content))
            finalized_execution_results.append(execution_result)
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

    refresh_calls: list[int] = []
    execution_result = {"success": True, "output": "Slack reply"}

    class FakeAgentManager:
        async def get_agent_for_task(self, *_args: Any, **_kwargs: Any) -> Any:
            return agent_service

        def refresh_connector_runtime_tools(self, task_id: int) -> None:
            refresh_calls.append(task_id)

        async def execute_task(self, **_kwargs: Any) -> dict[str, Any]:
            return execution_result

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
    assert refresh_calls == [45]
    assert finalized == [(TaskStatus.COMPLETED, "Slack reply")]
    assert finalized_execution_results == [execution_result]
    assert final_messages == [
        {
            "channel_id": "D1",
            "thread_ts": None,
            "loading_ts": "loading-ts",
            "text": "Slack reply",
        }
    ]
    assert managed.closed is True
