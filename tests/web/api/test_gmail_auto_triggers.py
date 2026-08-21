from __future__ import annotations

import asyncio
import base64
import itertools
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from google.auth.exceptions import RefreshError, TransportError

from xagent.config import (
    get_gmail_watch_enabled,
    get_gmail_watch_renewal_interval_seconds,
    get_gmail_watch_renewal_lead_seconds,
)
from xagent.web.models.agent import Agent
from xagent.web.models.gmail_watch import GmailWatchState
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.trigger import (
    AgentTrigger,
    TriggerAudit,
    TriggerProvisioningStatus,
    TriggerRun,
    TriggerType,
)
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services import ops_signals
from xagent.web.services.gmail_provisioning import (
    GMAIL_WATCH_DISABLED_ERROR,
    gmail_topic_path,
    reconcile_gmail_trigger_provisioning,
)
from xagent.web.services.gmail_triggers import (
    GmailPubsubNotification,
    GmailTriggerError,
    GmailWatchConfigurationError,
    _credentials_expiry,
    _get_google_oauth_config,
    build_gmail_service,
    collect_gmail_pubsub_events,
    scan_due_gmail_watch_renewals,
)
from xagent.web.services.trigger_providers import (
    GmailProvider,
    register_trigger_provider,
)

from .conftest import _direct_db_session, client

pytestmark = pytest.mark.usefixtures("_test_db")


@pytest.fixture(autouse=True)
def _isolate_gmail_push_service_account(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize the developer .env for OIDC verification defaults.

    tests/conftest.py loads the project .env with override=True; a push
    service account configured there makes verify() demand a matching email
    claim, flipping outcomes for every test whose fake OIDC verifier returns
    no email claim. Tests that exercise the service-account check set the
    variable explicitly via monkeypatch.setenv.

    Set to "" rather than deleted: fixtures running later (e.g. _test_db ->
    init_db -> migrations/env.py) call load_dotenv(override=False), which
    would re-populate a deleted variable but leaves an empty one alone, and
    the config getter normalizes "" to None.
    """
    monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT", "")


class _FakeHttpError(Exception):
    def __init__(self, status_code: int, message: str = "gmail error") -> None:
        super().__init__(message)
        self.response = type("Response", (), {"status_code": status_code})()


class _FakeExecutable:
    def __init__(
        self,
        payload: dict[str, object],
        *,
        exception: Exception | None = None,
    ) -> None:
        self.payload = payload
        self.exception = exception

    def execute(self) -> dict[str, object]:
        if self.exception is not None:
            raise self.exception
        return self.payload


class _FakeHistoryResource:
    def __init__(
        self,
        history_response: dict[str, object],
        *,
        exception: Exception | None = None,
    ) -> None:
        self._history_response = history_response
        self._exception = exception
        self.calls: list[dict[str, object]] = []

    def list(self, **kwargs: object) -> _FakeExecutable:
        self.calls.append(dict(kwargs))
        return _FakeExecutable(self._history_response, exception=self._exception)


class _FakeMessagesResource:
    def __init__(self, messages: dict[str, dict[str, object] | Exception]) -> None:
        self._messages = messages
        self.calls: list[dict[str, object]] = []

    def get(self, **kwargs: object) -> _FakeExecutable:
        self.calls.append(dict(kwargs))
        message = self._messages[str(kwargs["id"])]
        if isinstance(message, Exception):
            return _FakeExecutable({}, exception=message)
        return _FakeExecutable(message)


class _FakeUsersResource:
    def __init__(
        self,
        response: dict[str, object],
        calls: list[dict[str, object]],
        history: _FakeHistoryResource,
        messages: _FakeMessagesResource,
    ):
        self._watch_response = response
        self._calls = calls
        self._history = history
        self._messages = messages

    def watch(self, *, userId: str, body: dict[str, object]) -> _FakeExecutable:
        self._calls.append({"userId": userId, "body": body})
        return _FakeExecutable(self._watch_response)

    def history(self) -> _FakeHistoryResource:
        return self._history

    def messages(self) -> _FakeMessagesResource:
        return self._messages


class _FakeGmailService:
    def __init__(
        self,
        response: dict[str, object] | None = None,
        *,
        history_response: dict[str, object] | None = None,
        history_exception: Exception | None = None,
        messages: dict[str, dict[str, object] | Exception] | None = None,
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self._response = response or {}
        self.history_resource = _FakeHistoryResource(
            history_response or {},
            exception=history_exception,
        )
        self.messages_resource = _FakeMessagesResource(messages or {})

    def users(self) -> _FakeUsersResource:
        return _FakeUsersResource(
            self._response,
            self.calls,
            self.history_resource,
            self.messages_resource,
        )


@pytest.fixture
def mock_bg_scheduler():
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ) as mocked:
        yield mocked


class _FakePubsubPublisher:
    def __init__(self) -> None:
        self.topics: set[str] = set()

    def create_topic(self, *, request: dict[str, str]) -> None:
        self.topics.add(request["name"])

    def get_iam_policy(self, *, request: dict[str, str]):
        from types import SimpleNamespace

        class _Bindings(list):
            def add(self, *, role: str, members: list[str]) -> None:
                self.append(SimpleNamespace(role=role, members=members))

        return SimpleNamespace(bindings=_Bindings())

    def set_iam_policy(self, *, request: dict[str, object]) -> None:
        return None

    def delete_topic(self, *, request: dict[str, str]) -> None:
        self.topics.discard(request["topic"])


class _FakePubsubSubscriber:
    def __init__(self) -> None:
        self.subscriptions: dict[str, dict[str, object]] = {}

    def create_subscription(self, *, request: dict[str, object]) -> None:
        self.subscriptions[str(request["name"])] = request

    def delete_subscription(self, *, request: dict[str, str]) -> None:
        self.subscriptions.pop(request["subscription"], None)


@pytest.fixture()
def per_mailbox_pubsub_env(monkeypatch: pytest.MonkeyPatch):
    """Per-mailbox Gmail provisioning config with fake Pub/Sub clients."""
    from xagent.web.services import gmail_provisioning

    monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_PROJECT_ID", "demo-project")
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.example.com")
    monkeypatch.setenv(
        "XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT",
        "pubsub-push@demo-project.iam.gserviceaccount.com",
    )
    publisher = _FakePubsubPublisher()
    subscriber = _FakePubsubSubscriber()
    monkeypatch.setattr(gmail_provisioning, "_default_publisher", lambda: publisher)
    monkeypatch.setattr(gmail_provisioning, "_default_subscriber", lambda: subscriber)
    return publisher, subscriber


def _create_user(db, username: str = "gmail-watch-user") -> User:
    user = User(
        username=username,
        email=f"{username}@example.com",
        password_hash="hash",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _create_gmail_oauth(db, user: User) -> UserOAuth:
    oauth = UserOAuth(
        user_id=int(user.id),
        provider="gmail",
        access_token="access-token",
        refresh_token="refresh-token",
        provider_user_id="provider-user",
        email="codeacme17@gmail.com",
    )
    db.add(oauth)
    db.commit()
    db.refresh(oauth)
    return oauth


_gmail_trigger_agent_names = itertools.count(1)


def _create_gmail_trigger(
    db,
    user: User,
    *,
    enabled: bool = True,
    config: dict[str, object] | None = None,
) -> AgentTrigger:
    agent = Agent(
        user_id=int(user.id),
        name=f"Gmail trigger agent {next(_gmail_trigger_agent_names)}",
        description="test",
        instructions="Handle Gmail.",
        execution_mode="balanced",
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)

    trigger = AgentTrigger(
        user_id=int(user.id),
        agent_id=int(agent.id),
        type=TriggerType.GMAIL.value,
        name="Gmail inbox",
        enabled=enabled,
        config=config or {"watch_label": "INBOX"},
        prompt_template="Handle {{payload}}",
    )
    db.add(trigger)
    db.commit()
    db.refresh(trigger)
    return trigger


def _gmail_message(
    message_id: str,
    *,
    label_ids: list[str] | None = None,
    sender: str = "boss@company.com",
    subject: str = "urgent: e2e",
) -> dict[str, object]:
    return {
        "id": message_id,
        "threadId": f"thread-{message_id}",
        "labelIds": label_ids or ["INBOX"],
        "snippet": "Please reply exactly GMAIL_TRIGGER_OK",
        "payload": {
            "headers": [
                {"name": "From", "value": sender},
                {"name": "Subject", "value": subject},
            ]
        },
    }


def _mark_unified_gmail_trigger(
    db,
    trigger: AgentTrigger,
    *,
    resource_id: str = "codeacme17@gmail.com",
    callback_id: str | None = "legacy-trigger-callback",
) -> AgentTrigger:
    trigger.provider = TriggerType.GMAIL.value
    trigger.resource_id = resource_id
    trigger.callback_id = callback_id
    db.add(trigger)
    db.commit()
    db.refresh(trigger)
    return trigger


def _create_gmail_watch_state(
    db,
    user: User,
    oauth: UserOAuth,
    *,
    callback_id: str = "cb-gmail-watch",
    email: str = "CodeAcme17@Gmail.com",
) -> GmailWatchState:
    state = GmailWatchState(
        user_id=int(user.id),
        oauth_account_id=int(oauth.id),
        email=email,
        history_id="100",
        topic_name="projects/demo/topics/xagent-gmail",
        callback_id=callback_id,
        push_audience=f"https://stored.example.test/api/triggers/callback/gmail/{callback_id}",
    )
    db.add(state)
    db.commit()
    db.refresh(state)
    return state


def _gmail_pubsub_push_body(
    *,
    claimed_email: str,
    history_id: str = "222",
    message_id: str = "pubsub-1",
) -> bytes:
    data = base64.urlsafe_b64encode(
        json.dumps(
            {"emailAddress": claimed_email, "historyId": history_id},
            separators=(",", ":"),
        ).encode("utf-8")
    ).decode("ascii")
    return json.dumps(
        {"message": {"data": data.rstrip("="), "messageId": message_id}},
        separators=(",", ":"),
    ).encode("utf-8")


def test_gmail_watch_state_persists_previous_callback_audience_grace() -> None:
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-audience-grace-state-user")
        oauth = _create_gmail_oauth(db, user)
        state = _create_gmail_watch_state(db, user, oauth, callback_id="cb-grace-state")
        previous_audience = str(state.push_audience)
        grace_expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

        setattr(state, "previous_push_audience", previous_audience)
        setattr(state, "previous_push_audience_expires_at", grace_expires_at)
        db.add(state)
        db.commit()
        state_id = int(state.id)
        db.expunge_all()

        persisted = (
            db.query(GmailWatchState).filter(GmailWatchState.id == state_id).one()
        )
        assert getattr(persisted, "previous_push_audience", None) == previous_audience
        persisted_expiry = getattr(persisted, "previous_push_audience_expires_at", None)
        assert persisted_expiry is not None
        assert persisted_expiry.replace(tzinfo=timezone.utc) == grace_expires_at
    finally:
        db.close()


def test_gmail_provider_verifies_oidc_with_stored_audience_and_service_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT",
        "push-sa@example.iam.gserviceaccount.com",
    )
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-oidc-audience-user")
        oauth = _create_gmail_oauth(db, user)
        trigger = _mark_unified_gmail_trigger(db, _create_gmail_trigger(db, user))
        state = _create_gmail_watch_state(db, user, oauth, callback_id="cb-oidc")
        seen: dict[str, str] = {}

        def fake_verify(token: str, audience: str) -> dict[str, object]:
            seen["token"] = token
            seen["audience"] = audience
            return {
                "iss": "https://accounts.google.com",
                "aud": audience,
                "email": "push-sa@example.iam.gserviceaccount.com",
                "email_verified": True,
            }

        provider = GmailProvider(oidc_verifier=fake_verify)

        result = asyncio.run(
            provider.verify(
                type(
                    "Context",
                    (),
                    {
                        "callback_id": "cb-oidc",
                        "url_path": "/api/triggers/callback/gmail/cb-oidc",
                        "header": lambda _self, name: (
                            "Bearer oidc-token"
                            if name.lower() == "authorization"
                            else None
                        ),
                    },
                )(),
                db=db,
                trigger=trigger,
                raw_body=b"{}",
            )
        )

        assert result.verified is True
        assert result.attested_resource_id == "codeacme17@gmail.com"
        assert seen == {"token": "oidc-token", "audience": state.push_audience}
    finally:
        db.close()


def test_gmail_provider_accepts_configured_audience_during_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT",
        "push-sa@example.iam.gserviceaccount.com",
    )
    monkeypatch.setenv(
        "XAGENT_S2S_API_BASE_URL",
        "https://new-origin.example.test",
    )
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-oidc-transition-user")
        oauth = _create_gmail_oauth(db, user)
        trigger = _mark_unified_gmail_trigger(db, _create_gmail_trigger(db, user))
        state = _create_gmail_watch_state(db, user, oauth, callback_id="cb-transition")
        configured_audience = (
            "https://new-origin.example.test/api/triggers/callback/gmail/cb-transition"
        )
        seen_audiences: list[str] = []

        def fake_verify(_token: str, audience: str) -> dict[str, object]:
            seen_audiences.append(audience)
            if audience == state.push_audience:
                raise ValueError("token uses the newly configured audience")
            return {
                "iss": "https://accounts.google.com",
                "aud": configured_audience,
                "email": "push-sa@example.iam.gserviceaccount.com",
                "email_verified": True,
            }

        async def run_verifier_inline(function, *args):
            """Keep this unit test independent of the default executor lifecycle."""

            return function(*args)

        monkeypatch.setattr(
            "xagent.web.services.trigger_providers.gmail.asyncio.to_thread",
            run_verifier_inline,
        )
        provider = GmailProvider(oidc_verifier=fake_verify)
        result = asyncio.run(
            provider.verify(
                type(
                    "Context",
                    (),
                    {
                        "callback_id": "cb-transition",
                        "url_path": "/api/triggers/callback/gmail/cb-transition",
                        "header": lambda _self, name: (
                            "Bearer oidc-token"
                            if name.lower() == "authorization"
                            else None
                        ),
                    },
                )(),
                db=db,
                trigger=trigger,
                raw_body=b"{}",
            )
        )

        assert result.verified is True
        assert seen_audiences == [state.push_audience, configured_audience]
    finally:
        db.close()


def test_gmail_provider_accepts_previous_audience_after_reconciliation_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT",
        "push-sa@example.iam.gserviceaccount.com",
    )
    monkeypatch.setenv(
        "XAGENT_S2S_API_BASE_URL",
        "https://new-origin.example.test",
    )
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-oidc-post-commit-user")
        oauth = _create_gmail_oauth(db, user)
        trigger = _mark_unified_gmail_trigger(db, _create_gmail_trigger(db, user))
        state = _create_gmail_watch_state(db, user, oauth, callback_id="cb-post-commit")
        previous_audience = str(state.push_audience)
        configured_audience = (
            "https://new-origin.example.test/api/triggers/callback/gmail/cb-post-commit"
        )
        setattr(state, "push_audience", configured_audience)
        setattr(state, "previous_push_audience", previous_audience)
        setattr(
            state,
            "previous_push_audience_expires_at",
            datetime.now(timezone.utc) + timedelta(minutes=10),
        )
        db.add(state)
        db.commit()
        seen_audiences: list[str] = []

        def fake_verify(_token: str, audience: str) -> dict[str, object]:
            seen_audiences.append(audience)
            if audience != previous_audience:
                raise ValueError("token was signed for the previous audience")
            return {
                "iss": "https://accounts.google.com",
                "aud": previous_audience,
                "email": "push-sa@example.iam.gserviceaccount.com",
                "email_verified": True,
            }

        async def run_verifier_inline(function, *args):
            """Keep this unit test independent of the default executor lifecycle."""

            return function(*args)

        monkeypatch.setattr(
            "xagent.web.services.trigger_providers.gmail.asyncio.to_thread",
            run_verifier_inline,
        )
        provider = GmailProvider(oidc_verifier=fake_verify)
        result = asyncio.run(
            provider.verify(
                type(
                    "Context",
                    (),
                    {
                        "callback_id": "cb-post-commit",
                        "url_path": ("/api/triggers/callback/gmail/cb-post-commit"),
                        "header": lambda _self, name: (
                            "Bearer oidc-token"
                            if name.lower() == "authorization"
                            else None
                        ),
                    },
                )(),
                db=db,
                trigger=trigger,
                raw_body=b"{}",
            )
        )

        assert result.verified is True
        assert seen_audiences == [configured_audience, previous_audience]
    finally:
        db.close()


def test_gmail_provider_rejects_previous_audience_after_grace_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT",
        "push-sa@example.iam.gserviceaccount.com",
    )
    monkeypatch.setenv(
        "XAGENT_S2S_API_BASE_URL",
        "https://new-origin.example.test",
    )
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-oidc-expired-grace-user")
        oauth = _create_gmail_oauth(db, user)
        trigger = _mark_unified_gmail_trigger(db, _create_gmail_trigger(db, user))
        state = _create_gmail_watch_state(
            db, user, oauth, callback_id="cb-expired-grace"
        )
        previous_audience = str(state.push_audience)
        configured_audience = (
            "https://new-origin.example.test/api/triggers/callback/gmail/"
            "cb-expired-grace"
        )
        setattr(state, "push_audience", configured_audience)
        setattr(state, "previous_push_audience", previous_audience)
        setattr(
            state,
            "previous_push_audience_expires_at",
            datetime.now(timezone.utc) - timedelta(seconds=1),
        )
        db.add(state)
        db.commit()
        seen_audiences: list[str] = []

        def fake_verify(_token: str, audience: str) -> dict[str, object]:
            seen_audiences.append(audience)
            if audience != previous_audience:
                raise ValueError("token was signed for the previous audience")
            return {
                "iss": "https://accounts.google.com",
                "aud": previous_audience,
                "email": "push-sa@example.iam.gserviceaccount.com",
                "email_verified": True,
            }

        async def run_verifier_inline(function, *args):
            """Keep this unit test independent of the default executor lifecycle."""

            return function(*args)

        monkeypatch.setattr(
            "xagent.web.services.trigger_providers.gmail.asyncio.to_thread",
            run_verifier_inline,
        )
        provider = GmailProvider(oidc_verifier=fake_verify)
        result = asyncio.run(
            provider.verify(
                type(
                    "Context",
                    (),
                    {
                        "callback_id": "cb-expired-grace",
                        "url_path": ("/api/triggers/callback/gmail/cb-expired-grace"),
                        "header": lambda _self, name: (
                            "Bearer oidc-token"
                            if name.lower() == "authorization"
                            else None
                        ),
                    },
                )(),
                db=db,
                trigger=trigger,
                raw_body=b"{}",
            )
        )

        assert result.verified is False
        assert seen_audiences == [configured_audience]
    finally:
        db.close()


def test_gmail_provider_rejects_unverified_push_service_account_email(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT",
        "push-sa@example.iam.gserviceaccount.com",
    )
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-oidc-unverified-user")
        oauth = _create_gmail_oauth(db, user)
        trigger = _mark_unified_gmail_trigger(db, _create_gmail_trigger(db, user))
        _create_gmail_watch_state(db, user, oauth, callback_id="cb-unverified")

        provider = GmailProvider(
            oidc_verifier=lambda _token, audience: {
                "iss": "https://accounts.google.com",
                "aud": audience,
                "email": "push-sa@example.iam.gserviceaccount.com",
                "email_verified": False,
            }
        )

        result = asyncio.run(
            provider.verify(
                type(
                    "Context",
                    (),
                    {
                        "callback_id": "cb-unverified",
                        "header": lambda _self, name: (
                            "Bearer oidc-token"
                            if name.lower() == "authorization"
                            else None
                        ),
                    },
                )(),
                db=db,
                trigger=trigger,
                raw_body=b"{}",
            )
        )

        assert result.verified is False
        assert "email_verified" in str(result.reason)
    finally:
        db.close()


def test_gmail_provider_verify_tracks_oidc_degradation_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """verify() keeps the ops-signal registry current: the degradation
    registers when the push service account is unset and clears again once
    verification runs with it configured."""
    from xagent.web.services.ops_signals import (
        GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED,
        active_degradations,
        clear_degradation,
    )

    clear_degradation(GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED)
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-oidc-signal-user")
        oauth = _create_gmail_oauth(db, user)
        trigger = _mark_unified_gmail_trigger(db, _create_gmail_trigger(db, user))
        _create_gmail_watch_state(db, user, oauth, callback_id="cb-signal")

        provider = GmailProvider(
            oidc_verifier=lambda _token, audience: {
                "iss": "https://accounts.google.com",
                "aud": audience,
                "email": "push-sa@example.iam.gserviceaccount.com",
                "email_verified": True,
            }
        )
        context = type(
            "Context",
            (),
            {
                "callback_id": "cb-signal",
                "header": lambda _self, name: (
                    "Bearer oidc-token" if name.lower() == "authorization" else None
                ),
            },
        )()

        monkeypatch.delenv("XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT", raising=False)
        degraded = asyncio.run(
            provider.verify(context, db=db, trigger=trigger, raw_body=b"{}")
        )
        assert degraded.verified is True
        assert GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED in active_degradations()

        monkeypatch.setenv(
            "XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT",
            "push-sa@example.iam.gserviceaccount.com",
        )
        healthy = asyncio.run(
            provider.verify(context, db=db, trigger=trigger, raw_body=b"{}")
        )
        assert healthy.verified is True
        assert GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED not in active_degradations()
    finally:
        clear_degradation(GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED)
        db.close()


def test_gmail_unified_callback_ingests_history_filters_and_deduplicates(
    mock_bg_scheduler,
) -> None:
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-unified-user")
        oauth = _create_gmail_oauth(db, user)
        matching_trigger = _mark_unified_gmail_trigger(
            db,
            _create_gmail_trigger(
                db,
                user,
                config={
                    "watch_label": "INBOX",
                    "sender_filter": "boss@company.com",
                    "subject_keyword": "urgent",
                },
            ),
        )
        filtered_trigger = _mark_unified_gmail_trigger(
            db,
            _create_gmail_trigger(
                db,
                user,
                config={"watch_label": "INBOX", "sender_filter": "finance@example.com"},
            ),
            callback_id="legacy-trigger-callback-2",
        )
        state = _create_gmail_watch_state(db, user, oauth, callback_id="cb-unified")
        fake_service = _FakeGmailService(
            history_response={
                "history": [{"messagesAdded": [{"message": {"id": "msg-unified"}}]}]
            },
            messages={"msg-unified": _gmail_message("msg-unified")},
        )
        seen_audiences: list[str] = []

        def fake_verify(_token: str, audience: str) -> dict[str, object]:
            seen_audiences.append(audience)
            return {"iss": "https://accounts.google.com", "aud": audience}

        register_trigger_provider(
            GmailProvider(
                service_factory=lambda _db, _oauth: fake_service,
                oidc_verifier=fake_verify,
            ),
            replace=True,
        )
        raw_body = _gmail_pubsub_push_body(
            claimed_email="attacker@example.com",
            message_id="pubsub-unified",
        )

        first = client.post(
            "/api/triggers/callback/gmail/cb-unified",
            headers={"Authorization": "Bearer oidc-token"},
            content=raw_body,
        )
        second = client.post(
            "/api/triggers/callback/gmail/cb-unified",
            headers={"Authorization": "Bearer oidc-token"},
            content=raw_body,
        )

        assert first.status_code == 200, first.text
        assert first.json()["outcome"] == "accepted"
        assert len(first.json()["trigger_run_ids"]) == 1
        assert second.status_code == 200, second.text
        assert second.json()["duplicates"] == 1
        assert seen_audiences == [state.push_audience, state.push_audience]
        run = (
            db.query(TriggerRun)
            .filter(TriggerRun.trigger_id == int(matching_trigger.id))
            .one()
        )
        assert run.source_event_id == "gmail:msg-unified"
        assert run.payload_snapshot["metadata"]["resource_id"] == "codeacme17@gmail.com"
        assert "attacker@example.com" not in str(run.payload_snapshot)
        assert (
            db.query(TriggerRun)
            .filter(TriggerRun.trigger_id == int(filtered_trigger.id))
            .count()
            == 0
        )
        db.refresh(state)
        assert state.history_id == "222"
        assert mock_bg_scheduler.call_count == 1
    finally:
        register_trigger_provider(GmailProvider(), replace=True)
        db.close()


def test_gmail_unified_callback_holds_history_cursor_when_execution_fails() -> None:
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-unified-failure-user")
        oauth = _create_gmail_oauth(db, user)
        _mark_unified_gmail_trigger(db, _create_gmail_trigger(db, user))
        state = _create_gmail_watch_state(db, user, oauth, callback_id="cb-failure")
        fake_service = _FakeGmailService(
            history_response={
                "history": [{"messagesAdded": [{"message": {"id": "msg-failure"}}]}]
            },
            messages={"msg-failure": _gmail_message("msg-failure")},
        )
        register_trigger_provider(
            GmailProvider(
                service_factory=lambda _db, _oauth: fake_service,
                oidc_verifier=lambda _token, audience: {
                    "iss": "https://accounts.google.com",
                    "aud": audience,
                },
            ),
            replace=True,
        )

        async def fail_to_start(*_args, **_kwargs):
            raise RuntimeError("task start failed")

        with patch(
            "xagent.web.services.triggers.start_prepared_trigger_run",
            new=fail_to_start,
        ):
            response = client.post(
                "/api/triggers/callback/gmail/cb-failure",
                headers={"Authorization": "Bearer oidc-token"},
                content=_gmail_pubsub_push_body(
                    claimed_email="codeacme17@gmail.com",
                    message_id="pubsub-failure",
                ),
            )

        assert response.status_code == 500, response.text
        db.refresh(state)
        assert state.history_id == "100"
    finally:
        register_trigger_provider(GmailProvider(), replace=True)
        db.close()


def test_gmail_oidc_transport_error_returns_500_for_redelivery() -> None:
    """A JWKS fetch failure must not be ACKed like a forged token.

    Gmail's ack policy maps rejections to 200 (stop redelivery), so treating
    a transient network error as a rejection would drop the push permanently.
    """
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-oidc-transport-user")
        oauth = _create_gmail_oauth(db, user)
        trigger = _mark_unified_gmail_trigger(db, _create_gmail_trigger(db, user))
        state = _create_gmail_watch_state(db, user, oauth, callback_id="cb-transport")

        def transport_failure(_token: str, _audience: str) -> dict[str, object]:
            raise TransportError("failed to fetch Google JWKS certs")

        register_trigger_provider(
            GmailProvider(oidc_verifier=transport_failure), replace=True
        )

        response = client.post(
            "/api/triggers/callback/gmail/cb-transport",
            headers={"Authorization": "Bearer oidc-token"},
            content=_gmail_pubsub_push_body(claimed_email="codeacme17@gmail.com"),
        )

        assert response.status_code == 500, response.text
        db.refresh(state)
        assert state.history_id == "100"
        assert db.query(TriggerRun).count() == 0
        audit = (
            db.query(TriggerAudit)
            .filter(TriggerAudit.outcome == "execution_failure")
            .one()
        )
        assert audit.trigger_id == trigger.id
        assert audit.detail["stage"] == "verify"
        assert "TransportError" in audit.detail["error"]
    finally:
        register_trigger_provider(GmailProvider(), replace=True)
        db.close()


def test_gmail_oidc_invalid_token_is_rejected_and_acked() -> None:
    """Signature/claims failures stay rejections: audited and ACKed with 200."""
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-oidc-invalid-user")
        oauth = _create_gmail_oauth(db, user)
        _mark_unified_gmail_trigger(db, _create_gmail_trigger(db, user))
        state = _create_gmail_watch_state(db, user, oauth, callback_id="cb-invalid")

        def invalid_token(_token: str, _audience: str) -> dict[str, object]:
            raise ValueError("Token has wrong audience")

        register_trigger_provider(
            GmailProvider(oidc_verifier=invalid_token), replace=True
        )

        response = client.post(
            "/api/triggers/callback/gmail/cb-invalid",
            headers={"Authorization": "Bearer oidc-token"},
            content=_gmail_pubsub_push_body(claimed_email="codeacme17@gmail.com"),
        )

        assert response.status_code == 200, response.text
        assert response.json()["outcome"] == "rejected_signature"
        db.refresh(state)
        assert state.history_id == "100"
        audit = (
            db.query(TriggerAudit)
            .filter(TriggerAudit.outcome == "rejected_signature")
            .one()
        )
        assert "ValueError" in str(audit.detail["reason"])
    finally:
        register_trigger_provider(GmailProvider(), replace=True)
        db.close()


def test_gmail_oidc_verifier_allows_clock_skew() -> None:
    from xagent.web.services.trigger_providers.gmail import verify_google_oidc_token

    with patch(
        "xagent.web.services.trigger_providers.gmail.id_token.verify_oauth2_token",
        return_value={"iss": "https://accounts.google.com"},
    ) as mock_verify:
        claims = verify_google_oidc_token("token", "audience")

    assert claims == {"iss": "https://accounts.google.com"}
    assert mock_verify.call_args.kwargs["clock_skew_in_seconds"] > 0


def test_gmail_unified_callback_rejects_resource_mismatch_without_trusting_payload_email(
    mock_bg_scheduler,
) -> None:
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-resource-mismatch-user")
        oauth = _create_gmail_oauth(db, user)
        trigger = _mark_unified_gmail_trigger(
            db,
            _create_gmail_trigger(db, user),
            resource_id="other@example.com",
        )
        _create_gmail_watch_state(db, user, oauth, callback_id="cb-mismatch")
        fake_service = _FakeGmailService(
            history_response={
                "history": [{"messagesAdded": [{"message": {"id": "msg-mismatch"}}]}]
            },
            messages={"msg-mismatch": _gmail_message("msg-mismatch")},
        )
        register_trigger_provider(
            GmailProvider(
                service_factory=lambda _db, _oauth: fake_service,
                oidc_verifier=lambda _token, audience: {
                    "iss": "https://accounts.google.com",
                    "aud": audience,
                },
            ),
            replace=True,
        )

        response = client.post(
            "/api/triggers/callback/gmail/cb-mismatch",
            headers={"Authorization": "Bearer oidc-token"},
            content=_gmail_pubsub_push_body(claimed_email="other@example.com"),
        )

        assert response.status_code == 200, response.text
        assert response.json()["outcome"] == "rejected_resource"
        assert db.query(TriggerRun).count() == 0
        audit = (
            db.query(TriggerAudit)
            .filter(TriggerAudit.outcome == "rejected_resource")
            .one()
        )
        assert audit.trigger_id == trigger.id
        assert audit.detail["attested_resource_id"] == "codeacme17@gmail.com"
        assert audit.detail["trigger_resource_id"] == "other@example.com"
        assert mock_bg_scheduler.call_count == 0
    finally:
        register_trigger_provider(GmailProvider(), replace=True)
        db.close()


def test_gmail_watch_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XAGENT_GMAIL_WATCH_ENABLED", raising=False)
    monkeypatch.delenv("XAGENT_GMAIL_WATCH_RENEWAL_INTERVAL_SECONDS", raising=False)
    monkeypatch.delenv("XAGENT_GMAIL_WATCH_RENEWAL_LEAD_SECONDS", raising=False)

    assert get_gmail_watch_enabled() is False
    assert get_gmail_watch_renewal_interval_seconds() == 3600
    assert get_gmail_watch_renewal_lead_seconds() == 24 * 60 * 60


def test_gmail_watch_config_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAGENT_GMAIL_WATCH_ENABLED", "true")
    monkeypatch.setenv("XAGENT_GMAIL_WATCH_RENEWAL_INTERVAL_SECONDS", "120")
    monkeypatch.setenv("XAGENT_GMAIL_WATCH_RENEWAL_LEAD_SECONDS", "60")

    assert get_gmail_watch_enabled() is True
    assert get_gmail_watch_renewal_interval_seconds() == 120
    assert get_gmail_watch_renewal_lead_seconds() == 60


def test_legacy_gmail_shared_token_config_is_removed() -> None:
    """The shared push token and global topic helpers are gone for good."""
    import xagent.config as config

    assert not hasattr(config, "get_gmail_pubsub_push_token")
    assert not hasattr(config, "get_gmail_pubsub_topic_name")
    assert not hasattr(config, "GMAIL_PUBSUB_PUSH_TOKEN")
    assert not hasattr(config, "GMAIL_PUBSUB_TOPIC")


def test_legacy_gmail_pubsub_route_is_removed() -> None:
    """The shared-token push endpoint is no longer a supported API."""
    data = base64.b64encode(
        json.dumps({"emailAddress": "codeacme17@gmail.com", "historyId": "1"}).encode(
            "utf-8"
        )
    ).decode("ascii")
    payload = {"message": {"data": data, "messageId": "pubsub-legacy"}}

    response = client.post(
        "/api/triggers/gmail/pubsub?token=push-secret",
        headers={"x-xagent-gmail-pubsub-token": "push-secret"},
        json=payload,
    )
    assert response.status_code == 404


def test_gmail_oauth_config_falls_back_to_env_when_db_provider_is_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "env-client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "env-client-secret")
    db = _direct_db_session()
    try:
        provider = (
            db.query(OAuthProvider)
            .filter(OAuthProvider.provider_name == "google")
            .one()
        )
        provider.client_id = ""
        provider.client_secret = ""
        db.add(provider)
        db.commit()

        assert _get_google_oauth_config(db) == ("env-client-id", "env-client-secret")
    finally:
        db.close()


def test_build_gmail_service_passes_persisted_token_expiry_to_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "env-client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "env-client-secret")
    captured_kwargs: dict[str, object] = {}

    class FakeCredentials:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)
            self.expired = False
            self.refresh_token = kwargs.get("refresh_token")

    monkeypatch.setattr(
        "xagent.web.services.gmail_triggers.Credentials",
        FakeCredentials,
    )
    monkeypatch.setattr(
        "xagent.web.services.gmail_triggers.AuthorizedSession",
        lambda creds: object(),
    )

    db = _direct_db_session()
    try:
        provider = (
            db.query(OAuthProvider)
            .filter(OAuthProvider.provider_name == "google")
            .one()
        )
        provider.client_id = ""
        provider.client_secret = ""
        user = _create_user(db, "gmail-expiring-token-user")
        oauth = _create_gmail_oauth(db, user)
        oauth.expires_at = datetime(2026, 6, 29, 12, tzinfo=timezone.utc)
        db.add_all([provider, oauth])
        db.commit()
        db.refresh(oauth)

        build_gmail_service(db, oauth)

        assert captured_kwargs["expiry"] == oauth.expires_at
    finally:
        db.close()


def test_credentials_expiry_converts_aware_expiry_to_naive_utc() -> None:
    """Timezone-aware expires_at (PostgreSQL) must reach google-auth as naive UTC."""
    aware = datetime(2026, 6, 29, 20, tzinfo=timezone(timedelta(hours=8)))

    converted = _credentials_expiry(aware)

    assert converted is not None
    assert converted.tzinfo is None
    assert converted == datetime(2026, 6, 29, 12)
    # Naive values (SQLite) and None pass through untouched.
    assert _credentials_expiry(datetime(2026, 6, 29, 12)) == datetime(2026, 6, 29, 12)
    assert _credentials_expiry(None) is None


def test_build_gmail_service_accepts_aware_expiry_with_real_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for #807: aware expiry made creds.expired raise TypeError."""
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "env-client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "env-client-secret")
    monkeypatch.setattr(
        "xagent.web.services.gmail_triggers.AuthorizedSession",
        lambda creds: object(),
    )

    db = _direct_db_session()
    try:
        provider = (
            db.query(OAuthProvider)
            .filter(OAuthProvider.provider_name == "google")
            .one()
        )
        provider.client_id = ""
        provider.client_secret = ""
        user = _create_user(db, "gmail-real-creds-user")
        oauth = _create_gmail_oauth(db, user)
        db.add(provider)
        db.commit()
        oauth.expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

        service = build_gmail_service(db, oauth)

        assert service is not None
    finally:
        db.close()


def test_build_gmail_service_persists_refreshed_expiry_as_utc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refreshed naive-UTC expiry from google-auth is stored UTC-normalized."""
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "env-client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "env-client-secret")
    refreshed_expiry = datetime(2026, 6, 29, 13)

    class FakeCredentials:
        def __init__(self, **kwargs: object) -> None:
            self.expired = True
            self.refresh_token = kwargs.get("refresh_token")
            self.token = kwargs.get("token")
            self.expiry: datetime | None = None

        def refresh(self, _request: object) -> None:
            self.token = "refreshed-access-token"
            self.expiry = refreshed_expiry

    monkeypatch.setattr(
        "xagent.web.services.gmail_triggers.Credentials",
        FakeCredentials,
    )
    monkeypatch.setattr(
        "xagent.web.services.gmail_triggers.AuthorizedSession",
        lambda creds: object(),
    )

    db = _direct_db_session()
    try:
        provider = (
            db.query(OAuthProvider)
            .filter(OAuthProvider.provider_name == "google")
            .one()
        )
        provider.client_id = ""
        provider.client_secret = ""
        user = _create_user(db, "gmail-refresh-persist-user")
        oauth = _create_gmail_oauth(db, user)
        oauth.expires_at = datetime(2026, 6, 29, 12, tzinfo=timezone.utc)
        db.add_all([provider, oauth])
        db.commit()
        db.refresh(oauth)

        build_gmail_service(db, oauth)

        assert str(oauth.access_token) == "refreshed-access-token"
        stored = oauth.expires_at
        assert stored is not None
        if stored.tzinfo is None:  # SQLite drops tzinfo on the round-trip
            stored = stored.replace(tzinfo=timezone.utc)
        assert stored == refreshed_expiry.replace(tzinfo=timezone.utc)
    finally:
        db.close()


def test_build_gmail_service_raises_configuration_error_when_refresh_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "env-client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "env-client-secret")

    class FakeCredentials:
        def __init__(self, **kwargs: object) -> None:
            self.expired = True
            self.refresh_token = kwargs.get("refresh_token")

        def refresh(self, _request: object) -> None:
            raise RefreshError("revoked credentials")

    monkeypatch.setattr(
        "xagent.web.services.gmail_triggers.Credentials",
        FakeCredentials,
    )
    db = _direct_db_session()
    try:
        provider = (
            db.query(OAuthProvider)
            .filter(OAuthProvider.provider_name == "google")
            .one()
        )
        provider.client_id = ""
        provider.client_secret = ""
        user = _create_user(db, "gmail-refresh-fails-user")
        oauth = _create_gmail_oauth(db, user)
        oauth.expires_at = datetime(2026, 6, 29, 12, tzinfo=timezone.utc)
        db.add_all([provider, oauth])
        db.commit()
        db.refresh(oauth)

        with pytest.raises(
            GmailWatchConfigurationError,
            match="Gmail credential refresh failed",
        ):
            build_gmail_service(db, oauth)
    finally:
        db.close()


def test_gmail_watch_state_model_can_persist() -> None:
    db = _direct_db_session()
    try:
        user = _create_user(db)
        oauth = _create_gmail_oauth(db, user)

        state = GmailWatchState(
            user_id=int(user.id),
            oauth_account_id=int(oauth.id),
            email="codeacme17@gmail.com",
            history_id="100",
            watch_expiration=datetime(2026, 7, 1, tzinfo=timezone.utc),
            topic_name="projects/demo/topics/xagent-gmail",
        )
        db.add(state)
        db.commit()
        db.refresh(state)

        assert state.id is not None
        assert state.history_id == "100"
        assert state.oauth_account_id == int(oauth.id)
    finally:
        db.close()


def test_best_effort_provisioning_targets_only_referenced_mailboxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After OAuth, only mailboxes bound to enabled Gmail triggers are provisioned."""
    from xagent.web.services import gmail_provisioning

    monkeypatch.setenv("XAGENT_GMAIL_WATCH_ENABLED", "true")
    calls: list[str] = []

    def fake_ensure(_db, account, **_kwargs):
        calls.append(str(account.email).lower())
        return None

    monkeypatch.setattr(
        gmail_provisioning, "ensure_gmail_mailbox_provisioned", fake_ensure
    )
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-watch-needed")
        _create_gmail_oauth(db, user)

        gmail_provisioning.best_effort_provision_gmail_watches_for_user(
            db, user_id=int(user.id), context="test"
        )
        assert calls == []

        _mark_unified_gmail_trigger(db, _create_gmail_trigger(db, user))
        gmail_provisioning.best_effort_provision_gmail_watches_for_user(
            db, user_id=int(user.id), context="test"
        )
        assert calls == ["codeacme17@gmail.com"]
    finally:
        db.close()


def test_collect_gmail_pubsub_events_warns_for_nonordinary_watch_account(
    caplog: pytest.LogCaptureFixture,
) -> None:
    signal_name = "gmail_oauth_ownership_mismatch"
    ops_signals.clear_degradation(signal_name)
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-owner-mismatch")
        oauth = _create_gmail_oauth(db, user)
        state = GmailWatchState(
            user_id=int(user.id),
            oauth_account_id=int(oauth.id),
            email="codeacme17@gmail.com",
            history_id="100",
            topic_name="projects/demo/topics/xagent-gmail",
        )
        setattr(oauth, "resource_owner_key", "actor:alice")
        db.add_all([oauth, state])
        db.commit()

        with caplog.at_level("WARNING"):
            result = asyncio.run(
                collect_gmail_pubsub_events(
                    db,
                    GmailPubsubNotification(
                        email_address="codeacme17@gmail.com",
                        history_id="222",
                        pubsub_message_id="pubsub-owner-mismatch",
                    ),
                    state=state,
                )
            )

        assert result.events == []
        assert result.skipped == 1
        assert "Skipping Gmail callback" in caplog.text
        assert str(state.id) in caplog.text
        assert str(oauth.id) in caplog.text
        assert signal_name in ops_signals.active_degradations()
    finally:
        ops_signals.clear_degradation(signal_name)
        db.close()


def test_collect_gmail_pubsub_events_collects_matching_trigger_events() -> None:
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-history-user")
        oauth = _create_gmail_oauth(db, user)
        trigger = _create_gmail_trigger(db, user)
        state = GmailWatchState(
            user_id=int(user.id),
            oauth_account_id=int(oauth.id),
            email="codeacme17@gmail.com",
            history_id="100",
            topic_name="projects/demo/topics/xagent-gmail",
        )
        db.add(state)
        db.commit()
        fake_service = _FakeGmailService(
            history_response={
                "history": [
                    {
                        "messagesAdded": [
                            {"message": {"id": "msg-1", "threadId": "thread-1"}}
                        ]
                    }
                ]
            },
            messages={
                "msg-1": {
                    "id": "msg-1",
                    "threadId": "thread-1",
                    "labelIds": ["INBOX"],
                    "snippet": "Please reply exactly GMAIL_TRIGGER_OK",
                    "payload": {
                        "headers": [
                            {"name": "From", "value": "boss@company.com"},
                            {"name": "Subject", "value": "urgent: e2e"},
                        ]
                    },
                }
            },
        )

        result = asyncio.run(
            collect_gmail_pubsub_events(
                db,
                GmailPubsubNotification(
                    email_address="codeacme17@gmail.com",
                    history_id="222",
                    pubsub_message_id="pubsub-1",
                ),
                state=state,
                service_factory=lambda _db, _oauth: fake_service,
            )
        )

        assert len(result.events) == 1
        event = result.events[0]
        assert event.trigger_id == int(trigger.id)
        assert event.source_event_id == "gmail:msg-1"
        assert event.event_type == "gmail.message"
        assert event.resource_id == "codeacme17@gmail.com"
        assert event.payload["from"] == "boss@company.com"
        assert event.payload["subject"] == "urgent: e2e"
        assert fake_service.history_resource.calls == [
            {
                "userId": "me",
                "startHistoryId": "100",
                "historyTypes": ["messageAdded"],
            }
        ]
        # Collection never advances the cursor; GmailProvider.finalize_callback
        # does, only after all events fired.
        db.refresh(state)
        assert state.history_id == "100"
    finally:
        db.close()


def test_collect_gmail_pubsub_events_targets_exact_bound_account() -> None:
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-exact-bound-account-user")
        watched_account = _create_gmail_oauth(db, user)
        other_account = UserOAuth(
            user_id=int(user.id),
            provider="gmail",
            access_token="other-access-token",
            refresh_token="other-refresh-token",
            provider_user_id="other-provider-user",
            email="codeacme17@gmail.com",
        )
        db.add(other_account)
        db.commit()
        db.refresh(other_account)
        watched_trigger = _mark_unified_gmail_trigger(
            db,
            _create_gmail_trigger(
                db,
                user,
                config={
                    "watch_label": "INBOX",
                    "oauth_account_id": int(watched_account.id),
                },
            ),
        )
        _mark_unified_gmail_trigger(
            db,
            _create_gmail_trigger(
                db,
                user,
                config={
                    "watch_label": "INBOX",
                    "oauth_account_id": int(other_account.id),
                },
            ),
            callback_id="other-trigger-callback",
        )
        state = GmailWatchState(
            user_id=int(user.id),
            oauth_account_id=int(watched_account.id),
            email="codeacme17@gmail.com",
            history_id="100",
            topic_name="projects/demo/topics/xagent-gmail",
        )
        db.add(state)
        db.commit()
        fake_service = _FakeGmailService(
            history_response={
                "history": [{"messagesAdded": [{"message": {"id": "msg-bound"}}]}]
            },
            messages={"msg-bound": _gmail_message("msg-bound")},
        )

        result = asyncio.run(
            collect_gmail_pubsub_events(
                db,
                GmailPubsubNotification(
                    email_address="codeacme17@gmail.com",
                    history_id="222",
                    pubsub_message_id="pubsub-bound",
                ),
                state=state,
                service_factory=lambda _db, _oauth: fake_service,
            )
        )

        assert [event.trigger_id for event in result.events] == [
            int(watched_trigger.id)
        ]
    finally:
        db.close()


def test_collect_gmail_pubsub_events_skips_label_mismatch() -> None:
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-label-mismatch-user")
        oauth = _create_gmail_oauth(db, user)
        _create_gmail_trigger(db, user, config={"watch_label": "INBOX"})
        state = GmailWatchState(
            user_id=int(user.id),
            oauth_account_id=int(oauth.id),
            email="codeacme17@gmail.com",
            history_id="100",
            topic_name="projects/demo/topics/xagent-gmail",
        )
        db.add(state)
        db.commit()
        fake_service = _FakeGmailService(
            history_response={
                "history": [{"messagesAdded": [{"message": {"id": "msg-label"}}]}]
            },
            messages={
                "msg-label": _gmail_message(
                    "msg-label",
                    label_ids=["CATEGORY_PROMOTIONS"],
                )
            },
        )

        result = asyncio.run(
            collect_gmail_pubsub_events(
                db,
                GmailPubsubNotification(
                    email_address="codeacme17@gmail.com",
                    history_id="222",
                    pubsub_message_id="pubsub-label",
                ),
                state=state,
                service_factory=lambda _db, _oauth: fake_service,
            )
        )

        assert result.events == []
        assert result.skipped == 1
        db.refresh(state)
        assert state.history_id == "100"
    finally:
        db.close()


def test_collect_gmail_pubsub_events_accepts_case_insensitive_all_label() -> None:
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-all-label-user")
        oauth = _create_gmail_oauth(db, user)
        trigger = _create_gmail_trigger(db, user, config={"watch_label": "All"})
        state = GmailWatchState(
            user_id=int(user.id),
            oauth_account_id=int(oauth.id),
            email="codeacme17@gmail.com",
            history_id="100",
            topic_name="projects/demo/topics/xagent-gmail",
        )
        db.add(state)
        db.commit()
        fake_service = _FakeGmailService(
            history_response={
                "history": [{"messagesAdded": [{"message": {"id": "msg-all"}}]}]
            },
            messages={
                "msg-all": _gmail_message(
                    "msg-all",
                    label_ids=["CATEGORY_PROMOTIONS"],
                )
            },
        )

        result = asyncio.run(
            collect_gmail_pubsub_events(
                db,
                GmailPubsubNotification(
                    email_address="codeacme17@gmail.com",
                    history_id="222",
                    pubsub_message_id="pubsub-all",
                ),
                state=state,
                service_factory=lambda _db, _oauth: fake_service,
            )
        )

        assert len(result.events) == 1
        assert result.skipped == 0
        assert result.events[0].trigger_id == int(trigger.id)
        assert result.events[0].source_event_id == "gmail:msg-all"
    finally:
        db.close()


def test_collect_gmail_pubsub_events_accepts_wildcard_star_label() -> None:
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-star-label-user")
        oauth = _create_gmail_oauth(db, user)
        trigger = _create_gmail_trigger(db, user, config={"watch_label": "*"})
        state = GmailWatchState(
            user_id=int(user.id),
            oauth_account_id=int(oauth.id),
            email="codeacme17@gmail.com",
            history_id="100",
            topic_name="projects/demo/topics/xagent-gmail",
        )
        db.add(state)
        db.commit()
        fake_service = _FakeGmailService(
            history_response={
                "history": [{"messagesAdded": [{"message": {"id": "msg-star"}}]}]
            },
            messages={
                "msg-star": _gmail_message(
                    "msg-star",
                    label_ids=["CATEGORY_PROMOTIONS"],
                )
            },
        )

        result = asyncio.run(
            collect_gmail_pubsub_events(
                db,
                GmailPubsubNotification(
                    email_address="codeacme17@gmail.com",
                    history_id="222",
                    pubsub_message_id="pubsub-star",
                ),
                state=state,
                service_factory=lambda _db, _oauth: fake_service,
            )
        )

        assert len(result.events) == 1
        assert result.skipped == 0
        assert result.events[0].trigger_id == int(trigger.id)
        assert result.events[0].source_event_id == "gmail:msg-star"
    finally:
        db.close()


def test_collect_gmail_pubsub_events_wildcard_label_excludes_non_incoming_mail() -> (
    None
):
    # PR #1051 review: `history.list` has no labelId filter, so it returns
    # messagesAdded events for EVERY label (sent, drafts, spam, trash, not
    # just the inbox). A blank/wildcard watch label must mean "watch all
    # INCOMING email" (matching the UI copy) — not literally every message
    # of any kind, which would newly match mail the user themselves sent.
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-star-excludes-sent-user")
        oauth = _create_gmail_oauth(db, user)
        trigger = _create_gmail_trigger(db, user, config={"watch_label": "*"})
        state = GmailWatchState(
            user_id=int(user.id),
            oauth_account_id=int(oauth.id),
            email="codeacme17@gmail.com",
            history_id="100",
            topic_name="projects/demo/topics/xagent-gmail",
        )
        db.add(state)
        db.commit()
        fake_service = _FakeGmailService(
            history_response={
                "history": [
                    {
                        "messagesAdded": [
                            {"message": {"id": "msg-sent"}},
                            {"message": {"id": "msg-inbox"}},
                        ]
                    }
                ]
            },
            messages={
                "msg-sent": _gmail_message("msg-sent", label_ids=["SENT"]),
                "msg-inbox": _gmail_message("msg-inbox", label_ids=["INBOX"]),
            },
        )

        result = asyncio.run(
            collect_gmail_pubsub_events(
                db,
                GmailPubsubNotification(
                    email_address="codeacme17@gmail.com",
                    history_id="222",
                    pubsub_message_id="pubsub-mixed",
                ),
                state=state,
                service_factory=lambda _db, _oauth: fake_service,
            )
        )

        assert [event.source_event_id for event in result.events] == ["gmail:msg-inbox"]
        assert result.events[0].trigger_id == int(trigger.id)
        assert result.skipped == 1
    finally:
        db.close()


def test_collect_gmail_pubsub_events_specific_label_also_excludes_non_incoming_mail() -> (
    None
):
    # PR #1051 review, F10: the SENT/DRAFT/SPAM/TRASH exclusion previously
    # guarded only the wildcard branch. A message can carry a custom label
    # AND "SENT" at once (e.g. a user manually labels a sent message) — the
    # specific-label branch must exclude it too, not just literally-everything
    # watch labels.
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-specific-label-excludes-sent-user")
        oauth = _create_gmail_oauth(db, user)
        trigger = _create_gmail_trigger(db, user, config={"watch_label": "Support"})
        state = GmailWatchState(
            user_id=int(user.id),
            oauth_account_id=int(oauth.id),
            email="codeacme18@gmail.com",
            history_id="100",
            topic_name="projects/demo/topics/xagent-gmail",
        )
        db.add(state)
        db.commit()
        fake_service = _FakeGmailService(
            history_response={
                "history": [
                    {
                        "messagesAdded": [
                            {"message": {"id": "msg-sent-labeled"}},
                            {"message": {"id": "msg-incoming-labeled"}},
                        ]
                    }
                ]
            },
            messages={
                "msg-sent-labeled": _gmail_message(
                    "msg-sent-labeled", label_ids=["SENT", "Support"]
                ),
                "msg-incoming-labeled": _gmail_message(
                    "msg-incoming-labeled", label_ids=["INBOX", "Support"]
                ),
            },
        )

        result = asyncio.run(
            collect_gmail_pubsub_events(
                db,
                GmailPubsubNotification(
                    email_address="codeacme18@gmail.com",
                    history_id="222",
                    pubsub_message_id="pubsub-specific-label",
                ),
                state=state,
                service_factory=lambda _db, _oauth: fake_service,
            )
        )

        assert [event.source_event_id for event in result.events] == [
            "gmail:msg-incoming-labeled"
        ]
        assert result.events[0].trigger_id == int(trigger.id)
        assert result.skipped == 1
    finally:
        db.close()


def test_collect_gmail_pubsub_events_whitespace_only_label_falls_back_to_inbox() -> (
    None
):
    # PR #1051 review: `config.get("watch_label") or "INBOX"` treats a
    # whitespace-only stored value as truthy (skipping the INBOX fallback),
    # then strips it to "" — matching neither the wildcard branch nor a real
    # label, so filtering was silently disabled entirely (worse than doing
    # nothing: it also bypassed the SENT/DRAFT/SPAM/TRASH exclusion).
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-whitespace-label-user")
        oauth = _create_gmail_oauth(db, user)
        trigger = _create_gmail_trigger(db, user, config={"watch_label": "   "})
        state = GmailWatchState(
            user_id=int(user.id),
            oauth_account_id=int(oauth.id),
            email="codeacme17@gmail.com",
            history_id="100",
            topic_name="projects/demo/topics/xagent-gmail",
        )
        db.add(state)
        db.commit()
        fake_service = _FakeGmailService(
            history_response={
                "history": [
                    {
                        "messagesAdded": [
                            {"message": {"id": "msg-sent"}},
                            {"message": {"id": "msg-inbox"}},
                        ]
                    }
                ]
            },
            messages={
                "msg-sent": _gmail_message("msg-sent", label_ids=["SENT"]),
                "msg-inbox": _gmail_message("msg-inbox", label_ids=["INBOX"]),
            },
        )

        result = asyncio.run(
            collect_gmail_pubsub_events(
                db,
                GmailPubsubNotification(
                    email_address="codeacme17@gmail.com",
                    history_id="222",
                    pubsub_message_id="pubsub-whitespace",
                ),
                state=state,
                service_factory=lambda _db, _oauth: fake_service,
            )
        )

        # Falls back to INBOX-only, same as an absent/blank watch_label —
        # not "match everything" (which the pre-fix code silently did).
        assert [event.source_event_id for event in result.events] == ["gmail:msg-inbox"]
        assert result.events[0].trigger_id == int(trigger.id)
        assert result.skipped == 1
    finally:
        db.close()


def test_collect_gmail_pubsub_events_skips_sender_mismatch() -> None:
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-sender-mismatch-user")
        oauth = _create_gmail_oauth(db, user)
        trigger = _create_gmail_trigger(
            db,
            user,
            config={"watch_label": "INBOX", "sender_filter": "boss@company.com"},
        )
        state = GmailWatchState(
            user_id=int(user.id),
            oauth_account_id=int(oauth.id),
            email="codeacme17@gmail.com",
            history_id="100",
            topic_name="projects/demo/topics/xagent-gmail",
        )
        db.add(state)
        db.commit()
        fake_service = _FakeGmailService(
            history_response={
                "history": [{"messagesAdded": [{"message": {"id": "msg-sender"}}]}]
            },
            messages={
                "msg-sender": _gmail_message(
                    "msg-sender",
                    sender="teammate@company.com",
                )
            },
        )

        result = asyncio.run(
            collect_gmail_pubsub_events(
                db,
                GmailPubsubNotification(
                    email_address="codeacme17@gmail.com",
                    history_id="222",
                    pubsub_message_id="pubsub-sender",
                ),
                state=state,
                service_factory=lambda _db, _oauth: fake_service,
            )
        )

        assert result.events == []
        assert result.skipped == 1
        assert trigger.enabled is True
    finally:
        db.close()


def test_collect_gmail_pubsub_events_skips_subject_mismatch() -> None:
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-subject-mismatch-user")
        oauth = _create_gmail_oauth(db, user)
        trigger = _create_gmail_trigger(
            db,
            user,
            config={"watch_label": "INBOX", "subject_keyword": "urgent"},
        )
        state = GmailWatchState(
            user_id=int(user.id),
            oauth_account_id=int(oauth.id),
            email="codeacme17@gmail.com",
            history_id="100",
            topic_name="projects/demo/topics/xagent-gmail",
        )
        db.add(state)
        db.commit()
        fake_service = _FakeGmailService(
            history_response={
                "history": [{"messagesAdded": [{"message": {"id": "msg-subject"}}]}]
            },
            messages={
                "msg-subject": _gmail_message(
                    "msg-subject",
                    subject="newsletter",
                )
            },
        )

        result = asyncio.run(
            collect_gmail_pubsub_events(
                db,
                GmailPubsubNotification(
                    email_address="codeacme17@gmail.com",
                    history_id="222",
                    pubsub_message_id="pubsub-subject",
                ),
                state=state,
                service_factory=lambda _db, _oauth: fake_service,
            )
        )

        assert result.events == []
        assert result.skipped == 1
        assert trigger.enabled is True
    finally:
        db.close()


def test_collect_gmail_pubsub_events_reregisters_expired_history_id(
    monkeypatch: pytest.MonkeyPatch, per_mailbox_pubsub_env
) -> None:
    monkeypatch.setenv("XAGENT_GMAIL_WATCH_ENABLED", "true")
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-expired-history-user")
        oauth = _create_gmail_oauth(db, user)
        state = GmailWatchState(
            user_id=int(user.id),
            oauth_account_id=int(oauth.id),
            email="codeacme17@gmail.com",
            history_id="100",
            topic_name="projects/demo/topics/xagent-gmail",
        )
        db.add(state)
        db.commit()
        expired_history_service = _FakeGmailService(
            history_exception=_FakeHttpError(404, "history expired")
        )
        renewed_watch_service = _FakeGmailService(
            {"historyId": "333", "expiration": "1782864000000"}
        )
        services = iter([expired_history_service, renewed_watch_service])

        result = asyncio.run(
            collect_gmail_pubsub_events(
                db,
                GmailPubsubNotification(
                    email_address="codeacme17@gmail.com",
                    history_id="222",
                    pubsub_message_id="pubsub-expired",
                ),
                state=state,
                service_factory=lambda _db, _oauth: next(services),
            )
        )

        assert result.events == []
        assert result.skipped == 1
        db.refresh(state)
        assert state.history_id == "333"
        assert state.last_error is None
        expected_topic = gmail_topic_path("demo-project", "codeacme17@gmail.com")
        assert renewed_watch_service.calls == [
            {
                "userId": "me",
                "body": {
                    "topicName": expected_topic,
                    "labelIds": ["INBOX"],
                },
            }
        ]
        assert state.topic_name == expected_topic
    finally:
        db.close()


def test_collect_gmail_pubsub_events_does_not_reregister_while_watch_disabled(
    monkeypatch: pytest.MonkeyPatch, per_mailbox_pubsub_env
) -> None:
    """With the flag off, a stale-history callback must not re-register a
    watch: that would recreate the silently-expiring-watch bug (#1231)."""
    monkeypatch.delenv("XAGENT_GMAIL_WATCH_ENABLED", raising=False)
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-expired-history-disabled-user")
        oauth = _create_gmail_oauth(db, user)
        trigger = _mark_unified_gmail_trigger(
            db,
            _create_gmail_trigger(db, user),
        )
        trigger.provisioning_status = TriggerProvisioningStatus.ACTIVE.value
        db.add(trigger)
        db.commit()
        state = GmailWatchState(
            user_id=int(user.id),
            oauth_account_id=int(oauth.id),
            email="codeacme17@gmail.com",
            history_id="100",
            topic_name="projects/demo/topics/xagent-gmail",
        )
        db.add(state)
        db.commit()
        expired_history_service = _FakeGmailService(
            history_exception=_FakeHttpError(404, "history expired")
        )
        renewed_watch_service = _FakeGmailService(
            {"historyId": "333", "expiration": "1782864000000"}
        )
        services = iter([expired_history_service, renewed_watch_service])

        collection = asyncio.run(
            collect_gmail_pubsub_events(
                db,
                GmailPubsubNotification(
                    email_address="codeacme17@gmail.com",
                    history_id="222",
                    pubsub_message_id="pubsub-expired-disabled",
                ),
                state=state,
                service_factory=lambda _db, _oauth: next(services),
            )
        )

        # A permanent condition acks (200-equivalent: no exception, no
        # events) instead of driving Pub/Sub redelivery-with-backoff for the
        # retention window; the renewal scan retries by expiration once the
        # flag is re-enabled.
        assert collection.events == []
        assert collection.skipped == 1

        db.refresh(state)
        # The watch endpoint was never called: history_id stays at the
        # original value instead of adopting the renewal service's.
        assert state.history_id == "100"
        assert state.status == TriggerProvisioningStatus.FAILED.value
        assert state.last_error == GMAIL_WATCH_DISABLED_ERROR
        assert renewed_watch_service.calls == []
        assert reconcile_gmail_trigger_provisioning(db, [trigger]) == 1
        db.refresh(trigger)
        assert trigger.provisioning_status == TriggerProvisioningStatus.FAILED.value
        assert trigger.provisioning_error == GMAIL_WATCH_DISABLED_ERROR
    finally:
        db.close()


def test_collect_gmail_pubsub_events_retains_reregistration_failure_detail(
    monkeypatch: pytest.MonkeyPatch, per_mailbox_pubsub_env
) -> None:
    monkeypatch.setenv("XAGENT_GMAIL_WATCH_ENABLED", "true")
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-expired-history-reregistration-failure-user")
        oauth = _create_gmail_oauth(db, user)
        state = GmailWatchState(
            user_id=int(user.id),
            oauth_account_id=int(oauth.id),
            email="codeacme17@gmail.com",
            history_id="100",
            topic_name="projects/demo/topics/xagent-gmail",
        )
        db.add(state)
        db.commit()
        expired_history_service = _FakeGmailService(
            history_exception=_FakeHttpError(404, "history expired")
        )
        service_calls = 0

        def service_factory(_db, _oauth):
            nonlocal service_calls
            service_calls += 1
            if service_calls == 1:
                return expired_history_service
            raise RuntimeError("renewal backend unavailable")

        with pytest.raises(GmailTriggerError) as exc_info:
            asyncio.run(
                collect_gmail_pubsub_events(
                    db,
                    GmailPubsubNotification(
                        email_address="codeacme17@gmail.com",
                        history_id="222",
                        pubsub_message_id="pubsub-expired-renewal-failed",
                    ),
                    state=state,
                    service_factory=service_factory,
                )
            )

        assert "Gmail history expired and re-registration failed" in str(exc_info.value)
        assert "renewal backend unavailable" in str(exc_info.value)
        db.refresh(state)
        # FAILED here comes from ensure_gmail_mailbox_provisioned's own error
        # handling for the genuine RuntimeError (a real provisioning
        # failure), not from collect_gmail_pubsub_events's mark_failed flag -
        # that flag is only set for the disabled condition (see the
        # ...disabled test above) and is False on this flag-on transient
        # path, so it does not redundantly re-flip an already-converged
        # status here.
        assert state.status == TriggerProvisioningStatus.FAILED.value
        assert "renewal backend unavailable" in str(state.last_error)
    finally:
        db.close()


def test_gmail_unified_callback_preserves_cursor_for_nonordinary_watch_account() -> (
    None
):
    """Acknowledged ownership mismatches must not consume Gmail history."""
    signal_name = "gmail_oauth_ownership_mismatch"
    ops_signals.clear_degradation(signal_name)
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-owner-mismatch-route-user")
        oauth = _create_gmail_oauth(db, user)
        _mark_unified_gmail_trigger(db, _create_gmail_trigger(db, user))
        state = _create_gmail_watch_state(
            db, user, oauth, callback_id="cb-owner-mismatch-route"
        )
        setattr(oauth, "resource_owner_key", "actor:alice")
        db.add(oauth)
        db.commit()

        def fake_verify(_token: str, audience: str) -> dict[str, object]:
            return {"iss": "https://accounts.google.com", "aud": audience}

        register_trigger_provider(
            GmailProvider(oidc_verifier=fake_verify),
            replace=True,
        )

        response = client.post(
            "/api/triggers/callback/gmail/cb-owner-mismatch-route",
            headers={"Authorization": "Bearer oidc-token"},
            content=_gmail_pubsub_push_body(
                claimed_email="codeacme17@gmail.com",
                message_id="pubsub-owner-mismatch-route",
            ),
        )

        assert response.status_code == 200, response.text
        db.refresh(state)
        assert state.history_id == "100"
        assert signal_name in ops_signals.active_degradations()
    finally:
        register_trigger_provider(GmailProvider(), replace=True)
        ops_signals.clear_degradation(signal_name)
        db.close()


def test_gmail_unified_callback_acks_and_stays_disabled_when_watch_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale-history push through the real callback route, with watch
    registration disabled, must ack (200) instead of retrying, and finalize
    must not erase the disabled marking collect_gmail_pubsub_events just
    recorded (#1231 follow-up)."""
    monkeypatch.delenv("XAGENT_GMAIL_WATCH_ENABLED", raising=False)
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-disabled-route-user")
        oauth = _create_gmail_oauth(db, user)
        _mark_unified_gmail_trigger(db, _create_gmail_trigger(db, user))
        state = _create_gmail_watch_state(
            db, user, oauth, callback_id="cb-disabled-route"
        )

        fake_service = _FakeGmailService(
            history_exception=_FakeHttpError(404, "history expired")
        )

        def fake_verify(_token: str, audience: str) -> dict[str, object]:
            return {"iss": "https://accounts.google.com", "aud": audience}

        register_trigger_provider(
            GmailProvider(
                service_factory=lambda _db, _oauth: fake_service,
                oidc_verifier=fake_verify,
            ),
            replace=True,
        )
        raw_body = _gmail_pubsub_push_body(
            claimed_email="attacker@example.com",
            message_id="pubsub-disabled-route",
        )

        response = client.post(
            "/api/triggers/callback/gmail/cb-disabled-route",
            headers={"Authorization": "Bearer oidc-token"},
            content=raw_body,
        )

        assert response.status_code == 200, response.text
        db.refresh(state)
        assert state.status == TriggerProvisioningStatus.FAILED.value
        assert state.last_error == GMAIL_WATCH_DISABLED_ERROR
        # finalize_callback must not have advanced the cursor or cleared the
        # marking despite the callback being acked as a whole.
        assert state.history_id == "100"
    finally:
        register_trigger_provider(GmailProvider(), replace=True)
        db.close()


def test_gmail_unified_callback_advances_transient_failure_when_flag_off(
    monkeypatch: pytest.MonkeyPatch,
    mock_bg_scheduler,
) -> None:
    """A row failed for a transient reason (not the disabled marking) while
    the flag happens to be off must keep advancing its cursor once a valid
    push arrives - only the exact disabled marking should freeze finalize
    (#1231 review follow-up, F2)."""
    monkeypatch.delenv("XAGENT_GMAIL_WATCH_ENABLED", raising=False)
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-transient-failure-flag-off-user")
        oauth = _create_gmail_oauth(db, user)
        _mark_unified_gmail_trigger(db, _create_gmail_trigger(db, user))
        state = _create_gmail_watch_state(
            db, user, oauth, callback_id="cb-transient-flag-off"
        )
        setattr(state, "status", TriggerProvisioningStatus.FAILED.value)
        setattr(state, "last_error", "quota exceeded")
        db.add(state)
        db.commit()

        fake_service = _FakeGmailService(
            history_response={
                "history": [{"messagesAdded": [{"message": {"id": "msg-transient"}}]}]
            },
            messages={"msg-transient": _gmail_message("msg-transient")},
        )

        def fake_verify(_token: str, audience: str) -> dict[str, object]:
            return {"iss": "https://accounts.google.com", "aud": audience}

        register_trigger_provider(
            GmailProvider(
                service_factory=lambda _db, _oauth: fake_service,
                oidc_verifier=fake_verify,
            ),
            replace=True,
        )
        raw_body = _gmail_pubsub_push_body(
            claimed_email="attacker@example.com",
            message_id="pubsub-transient-flag-off",
        )

        response = client.post(
            "/api/triggers/callback/gmail/cb-transient-flag-off",
            headers={"Authorization": "Bearer oidc-token"},
            content=raw_body,
        )

        assert response.status_code == 200, response.text
        assert response.json()["outcome"] == "accepted"
        db.refresh(state)
        assert state.history_id == "222"
        assert state.last_error is None
    finally:
        register_trigger_provider(GmailProvider(), replace=True)
        db.close()


def test_collect_gmail_pubsub_events_records_service_configuration_error() -> None:
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-service-error-user")
        oauth = _create_gmail_oauth(db, user)
        state = GmailWatchState(
            user_id=int(user.id),
            oauth_account_id=int(oauth.id),
            email="codeacme17@gmail.com",
            history_id="100",
            topic_name="projects/demo/topics/xagent-gmail",
        )
        db.add(state)
        db.commit()

        def service_factory(_db, _oauth):
            raise GmailWatchConfigurationError("Gmail credential refresh failed")

        with pytest.raises(
            GmailWatchConfigurationError,
            match="Gmail credential refresh failed",
        ):
            asyncio.run(
                collect_gmail_pubsub_events(
                    db,
                    GmailPubsubNotification(
                        email_address="codeacme17@gmail.com",
                        history_id="222",
                        pubsub_message_id="pubsub-refresh-error",
                    ),
                    state=state,
                    service_factory=service_factory,
                )
            )

        db.refresh(state)
        assert "Gmail credential refresh failed" in str(state.last_error)
    finally:
        db.close()


def test_collect_gmail_pubsub_events_skips_deleted_message_without_failing_batch() -> (
    None
):
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-message-error-user")
        oauth = _create_gmail_oauth(db, user)
        trigger = _create_gmail_trigger(db, user)
        state = GmailWatchState(
            user_id=int(user.id),
            oauth_account_id=int(oauth.id),
            email="codeacme17@gmail.com",
            history_id="100",
            topic_name="projects/demo/topics/xagent-gmail",
        )
        db.add(state)
        db.commit()
        fake_service = _FakeGmailService(
            history_response={
                "history": [
                    {
                        "messagesAdded": [
                            {"message": {"id": "deleted-msg"}},
                            {"message": {"id": "msg-2", "threadId": "thread-2"}},
                        ]
                    }
                ]
            },
            messages={
                "deleted-msg": _FakeHttpError(404, "message deleted"),
                "msg-2": {
                    "id": "msg-2",
                    "threadId": "thread-2",
                    "labelIds": ["INBOX"],
                    "snippet": "Please reply exactly GMAIL_TRIGGER_OK",
                    "payload": {
                        "headers": [
                            {"name": "From", "value": "boss@company.com"},
                            {"name": "Subject", "value": "urgent: e2e"},
                        ]
                    },
                },
            },
        )

        result = asyncio.run(
            collect_gmail_pubsub_events(
                db,
                GmailPubsubNotification(
                    email_address="codeacme17@gmail.com",
                    history_id="222",
                    pubsub_message_id="pubsub-batch",
                ),
                state=state,
                service_factory=lambda _db, _oauth: fake_service,
            )
        )

        assert len(result.events) == 1
        assert result.skipped == 1
        assert result.events[0].trigger_id == int(trigger.id)
        assert result.events[0].source_event_id == "gmail:msg-2"
        db.refresh(state)
        assert state.history_id == "100"
        assert state.last_error is None
    finally:
        db.close()


def test_collect_gmail_pubsub_events_skips_forbidden_message_without_failing_batch() -> (
    None
):
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-forbidden-message-user")
        oauth = _create_gmail_oauth(db, user)
        trigger = _create_gmail_trigger(db, user)
        state = GmailWatchState(
            user_id=int(user.id),
            oauth_account_id=int(oauth.id),
            email="codeacme17@gmail.com",
            history_id="100",
            topic_name="projects/demo/topics/xagent-gmail",
        )
        db.add(state)
        db.commit()
        fake_service = _FakeGmailService(
            history_response={
                "history": [
                    {
                        "messagesAdded": [
                            {"message": {"id": "forbidden-msg"}},
                            {"message": {"id": "msg-2", "threadId": "thread-2"}},
                        ]
                    }
                ]
            },
            messages={
                "forbidden-msg": _FakeHttpError(403, "insufficient scope"),
                "msg-2": _gmail_message("msg-2"),
            },
        )

        result = asyncio.run(
            collect_gmail_pubsub_events(
                db,
                GmailPubsubNotification(
                    email_address="codeacme17@gmail.com",
                    history_id="222",
                    pubsub_message_id="pubsub-forbidden",
                ),
                state=state,
                service_factory=lambda _db, _oauth: fake_service,
            )
        )

        assert len(result.events) == 1
        assert result.skipped == 1
        assert result.events[0].trigger_id == int(trigger.id)
        assert result.events[0].source_event_id == "gmail:msg-2"
        db.refresh(state)
        assert state.history_id == "100"
        assert state.last_error is None
    finally:
        db.close()


def test_collect_gmail_pubsub_events_fails_batch_on_transient_message_error_and_holds_cursor() -> (
    None
):
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-transient-message-error-user")
        oauth = _create_gmail_oauth(db, user)
        _create_gmail_trigger(db, user)
        state = GmailWatchState(
            user_id=int(user.id),
            oauth_account_id=int(oauth.id),
            email="codeacme17@gmail.com",
            history_id="100",
            topic_name="projects/demo/topics/xagent-gmail",
            status=TriggerProvisioningStatus.ACTIVE.value,
        )
        db.add(state)
        db.commit()
        fake_service = _FakeGmailService(
            history_response={
                "history": [
                    {
                        "messagesAdded": [
                            {"message": {"id": "transient-msg"}},
                            {"message": {"id": "msg-2", "threadId": "thread-2"}},
                        ]
                    }
                ]
            },
            messages={
                "transient-msg": _FakeHttpError(503, "gmail unavailable"),
                "msg-2": _gmail_message("msg-2"),
            },
        )

        with pytest.raises(Exception, match="Failed to process Gmail message"):
            asyncio.run(
                collect_gmail_pubsub_events(
                    db,
                    GmailPubsubNotification(
                        email_address="codeacme17@gmail.com",
                        history_id="222",
                        pubsub_message_id="pubsub-transient",
                    ),
                    state=state,
                    service_factory=lambda _db, _oauth: fake_service,
                )
            )

        db.refresh(state)
        assert state.history_id == "100"
        assert state.status == TriggerProvisioningStatus.ACTIVE.value
        assert "transient-msg" in str(state.last_error)
    finally:
        db.close()


def test_collect_gmail_pubsub_events_holds_cursor_on_rate_limited_message() -> None:
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-rate-limited-message-user")
        oauth = _create_gmail_oauth(db, user)
        _create_gmail_trigger(db, user)
        state = GmailWatchState(
            user_id=int(user.id),
            oauth_account_id=int(oauth.id),
            email="codeacme17@gmail.com",
            history_id="100",
            topic_name="projects/demo/topics/xagent-gmail",
        )
        db.add(state)
        db.commit()
        fake_service = _FakeGmailService(
            history_response={
                "history": [
                    {
                        "messagesAdded": [
                            {"message": {"id": "rate-limited-msg"}},
                            {"message": {"id": "msg-2", "threadId": "thread-2"}},
                        ]
                    }
                ]
            },
            messages={
                "rate-limited-msg": _FakeHttpError(429, "rate limited"),
                "msg-2": _gmail_message("msg-2"),
            },
        )

        with pytest.raises(Exception, match="Failed to process Gmail message"):
            asyncio.run(
                collect_gmail_pubsub_events(
                    db,
                    GmailPubsubNotification(
                        email_address="codeacme17@gmail.com",
                        history_id="222",
                        pubsub_message_id="pubsub-rate-limited",
                    ),
                    state=state,
                    service_factory=lambda _db, _oauth: fake_service,
                )
            )

        db.refresh(state)
        assert state.history_id == "100"
        assert "rate-limited-msg" in str(state.last_error)
    finally:
        db.close()


def test_scan_due_gmail_watch_renewals_respects_enabled_flag_and_expiration(
    monkeypatch: pytest.MonkeyPatch, per_mailbox_pubsub_env
) -> None:
    monkeypatch.delenv("XAGENT_GMAIL_WATCH_ENABLED", raising=False)
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-renewal-user")
        oauth = _create_gmail_oauth(db, user)
        _create_gmail_trigger(db, user)
        fake_service = _FakeGmailService(
            {"historyId": "789", "expiration": "1782864000000"}
        )

        assert (
            scan_due_gmail_watch_renewals(
                db,
                now=datetime(2026, 6, 29, tzinfo=timezone.utc),
                service_factory=lambda _db, _oauth: fake_service,
            )
            == 0
        )

        monkeypatch.setenv("XAGENT_GMAIL_WATCH_ENABLED", "true")
        assert (
            scan_due_gmail_watch_renewals(
                db,
                now=datetime(2026, 6, 29, tzinfo=timezone.utc),
                service_factory=lambda _db, _oauth: fake_service,
            )
            == 1
        )
        state = (
            db.query(GmailWatchState)
            .filter(GmailWatchState.oauth_account_id == int(oauth.id))
            .one()
        )
        assert state.history_id == "789"

        assert (
            scan_due_gmail_watch_renewals(
                db,
                now=datetime(2026, 6, 29, tzinfo=timezone.utc),
                service_factory=lambda _db, _oauth: fake_service,
            )
            == 0
        )

        state.watch_expiration = datetime(2026, 6, 29, 12, tzinfo=timezone.utc)
        db.add(state)
        db.commit()
        assert (
            scan_due_gmail_watch_renewals(
                db,
                now=datetime(2026, 6, 29, tzinfo=timezone.utc),
                service_factory=lambda _db, _oauth: fake_service,
            )
            == 1
        )
    finally:
        db.close()


def test_scan_due_gmail_watch_renewals_applies_batch_limit(
    monkeypatch: pytest.MonkeyPatch, per_mailbox_pubsub_env
) -> None:
    monkeypatch.setenv("XAGENT_GMAIL_WATCH_ENABLED", "true")
    db = _direct_db_session()
    try:
        users = [_create_user(db, f"gmail-renewal-limit-{idx}") for idx in range(3)]
        oauth_accounts = [_create_gmail_oauth(db, user) for user in users]
        for user in users:
            _create_gmail_trigger(db, user)
        fake_service = _FakeGmailService(
            {"historyId": "789", "expiration": "1782864000000"}
        )

        renewed = scan_due_gmail_watch_renewals(
            db,
            now=datetime(2026, 6, 29, tzinfo=timezone.utc),
            service_factory=lambda _db, _oauth: fake_service,
            limit=2,
        )

        assert renewed == 2
        watched_oauth_ids = {
            state.oauth_account_id for state in db.query(GmailWatchState).all()
        }
        assert watched_oauth_ids == {
            int(oauth_accounts[0].id),
            int(oauth_accounts[1].id),
        }
        assert len(fake_service.calls) == 2
    finally:
        db.close()


def test_scan_due_gmail_watch_renewals_prioritizes_missing_and_earliest_expiration(
    monkeypatch: pytest.MonkeyPatch, per_mailbox_pubsub_env
) -> None:
    monkeypatch.setenv("XAGENT_GMAIL_WATCH_ENABLED", "true")
    db = _direct_db_session()
    try:
        later_user = _create_user(db, "gmail-renewal-later-user")
        later_oauth = _create_gmail_oauth(db, later_user)
        later_oauth.email = "later@example.com"
        _create_gmail_trigger(db, later_user)
        later_state = GmailWatchState(
            user_id=int(later_user.id),
            oauth_account_id=int(later_oauth.id),
            email="later@example.com",
            history_id="100",
            watch_expiration=datetime(2026, 6, 29, 20, tzinfo=timezone.utc),
            topic_name="projects/demo/topics/xagent-gmail",
        )

        missing_expiration_user = _create_user(db, "gmail-renewal-missing-user")
        missing_expiration_oauth = _create_gmail_oauth(db, missing_expiration_user)
        missing_expiration_oauth.email = "missing@example.com"
        _create_gmail_trigger(db, missing_expiration_user)
        missing_expiration_state = GmailWatchState(
            user_id=int(missing_expiration_user.id),
            oauth_account_id=int(missing_expiration_oauth.id),
            email="missing@example.com",
            history_id="100",
            watch_expiration=None,
            topic_name="projects/demo/topics/xagent-gmail",
        )
        db.add_all(
            [
                later_oauth,
                later_state,
                missing_expiration_oauth,
                missing_expiration_state,
            ]
        )
        db.commit()
        fake_service = _FakeGmailService(
            {"historyId": "789", "expiration": "1782864000000"}
        )

        renewed = scan_due_gmail_watch_renewals(
            db,
            now=datetime(2026, 6, 29, tzinfo=timezone.utc),
            service_factory=lambda _db, _oauth: fake_service,
            limit=1,
        )

        assert renewed == 1
        db.refresh(later_state)
        db.refresh(missing_expiration_state)
        assert later_state.history_id == "100"
        assert missing_expiration_state.history_id == "789"
    finally:
        db.close()


def test_scan_due_gmail_watch_renewals_records_failure_and_continues(
    monkeypatch: pytest.MonkeyPatch, per_mailbox_pubsub_env
) -> None:
    monkeypatch.setenv("XAGENT_GMAIL_WATCH_ENABLED", "true")
    db = _direct_db_session()
    try:
        first_user = _create_user(db, "gmail-renewal-fails-user")
        first_oauth = _create_gmail_oauth(db, first_user)
        first_oauth.email = "first@example.com"
        _create_gmail_trigger(db, first_user)
        first_state = GmailWatchState(
            user_id=int(first_user.id),
            oauth_account_id=int(first_oauth.id),
            email="first@example.com",
            history_id="100",
            watch_expiration=datetime(2026, 6, 29, 12, tzinfo=timezone.utc),
            topic_name="projects/demo/topics/xagent-gmail",
        )

        second_user = _create_user(db, "gmail-renewal-continues-user")
        second_oauth = _create_gmail_oauth(db, second_user)
        second_oauth.email = "second@example.com"
        _create_gmail_trigger(db, second_user)
        second_state = GmailWatchState(
            user_id=int(second_user.id),
            oauth_account_id=int(second_oauth.id),
            email="second@example.com",
            history_id="200",
            watch_expiration=datetime(2026, 6, 29, 12, tzinfo=timezone.utc),
            topic_name="projects/demo/topics/xagent-gmail",
        )
        db.add_all([first_oauth, first_state, second_oauth, second_state])
        db.commit()
        renewed_service = _FakeGmailService(
            {"historyId": "999", "expiration": "1782864000000"}
        )

        def service_factory(_db, oauth_account):
            if int(oauth_account.id) == int(first_oauth.id):
                raise RuntimeError("revoked credentials")
            return renewed_service

        renewed = scan_due_gmail_watch_renewals(
            db,
            now=datetime(2026, 6, 29, tzinfo=timezone.utc),
            service_factory=service_factory,
        )

        assert renewed == 1
        db.refresh(first_state)
        db.refresh(second_state)
        assert "revoked credentials" in str(first_state.last_error)
        assert second_state.history_id == "999"
    finally:
        db.close()


def test_gmail_finalize_never_rolls_history_cursor_backwards() -> None:
    """Stale or redelivered notifications must not rewind the watch cursor.

    Guards the expired-history recovery path: re-registration resets the
    cursor to a fresh watch historyId, and the stale notification that
    triggered recovery must not clobber it afterwards.
    """
    from xagent.web.services.trigger_providers import CallbackRequestContext

    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-cursor-guard-user")
        oauth = _create_gmail_oauth(db, user)
        state = _create_gmail_watch_state(db, user, oauth, callback_id="cb-cursor")
        setattr(state, "history_id", "300")
        db.add(state)
        db.commit()

        provider = GmailProvider()
        context = CallbackRequestContext(provider="gmail", callback_id="cb-cursor")

        stale_body = _gmail_pubsub_push_body(
            claimed_email="codeacme17@gmail.com", history_id="222"
        )
        asyncio.run(
            provider.finalize_callback(
                db=db,
                context=context,
                trigger=None,
                events=[],
                raw_body=stale_body,
            )
        )
        db.refresh(state)
        assert state.history_id == "300"

        advancing_body = _gmail_pubsub_push_body(
            claimed_email="codeacme17@gmail.com", history_id="400"
        )
        asyncio.run(
            provider.finalize_callback(
                db=db,
                context=context,
                trigger=None,
                events=[],
                raw_body=advancing_body,
            )
        )
        db.refresh(state)
        assert state.history_id == "400"
    finally:
        db.close()


def test_gmail_unified_callback_ingestion_failure_is_controlled_and_audited(
    mock_bg_scheduler,
) -> None:
    """Transient ingestion errors return the provider failure status with an
    audit row instead of an unhandled 500, so Pub/Sub redelivery can retry."""
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-ingest-failure-user")
        oauth = _create_gmail_oauth(db, user)
        trigger = _mark_unified_gmail_trigger(db, _create_gmail_trigger(db, user))
        _create_gmail_watch_state(db, user, oauth, callback_id="cb-ingest-fail")

        def broken_service_factory(_db, _oauth):
            raise GmailWatchConfigurationError("Gmail credential refresh failed")

        register_trigger_provider(
            GmailProvider(
                service_factory=broken_service_factory,
                oidc_verifier=lambda _token, audience: {
                    "iss": "https://accounts.google.com",
                    "aud": audience,
                },
            ),
            replace=True,
        )

        response = client.post(
            "/api/triggers/callback/gmail/cb-ingest-fail",
            headers={"Authorization": "Bearer oidc-token"},
            content=_gmail_pubsub_push_body(claimed_email="codeacme17@gmail.com"),
        )

        assert response.status_code == 500
        assert response.json()["outcome"] == "execution_failure"
        assert db.query(TriggerRun).count() == 0
        audit = (
            db.query(TriggerAudit)
            .filter(TriggerAudit.outcome == "execution_failure")
            .one()
        )
        assert audit.trigger_id == trigger.id
        assert audit.detail["stage"] == "ingest"
        assert mock_bg_scheduler.call_count == 0
    finally:
        register_trigger_provider(GmailProvider(), replace=True)
        db.close()


def test_gmail_malformed_pubsub_message_is_acked_not_redelivered() -> None:
    """A permanently malformed Pub/Sub message must not loop for the
    retention window: Gmail maps parse failures to a 200 ack while the
    execution_failure audit row keeps the outcome visible."""
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-malformed-user")
        oauth = _create_gmail_oauth(db, user)
        _mark_unified_gmail_trigger(db, _create_gmail_trigger(db, user))
        _create_gmail_watch_state(db, user, oauth, callback_id="cb-malformed")

        register_trigger_provider(
            GmailProvider(
                oidc_verifier=lambda _token, audience: {
                    "iss": "https://accounts.google.com",
                    "aud": audience,
                },
            ),
            replace=True,
        )

        response = client.post(
            "/api/triggers/callback/gmail/cb-malformed",
            headers={"Authorization": "Bearer oidc-token"},
            json={"message": {"data": "!!!not-base64!!!", "messageId": "m-1"}},
        )

        assert response.status_code == 200, response.text
        assert response.json()["outcome"] == "execution_failure"
        audit = (
            db.query(TriggerAudit)
            .filter(TriggerAudit.outcome == "execution_failure")
            .one()
        )
        assert audit.detail["stage"] == "parse"
    finally:
        register_trigger_provider(GmailProvider(), replace=True)
        db.close()


def test_collect_uses_the_callback_watch_state_not_email_lookup() -> None:
    """Two accounts can watch the same mailbox; ingestion must use the
    cursor and trigger set of the watch state the callback addressed."""
    db = _direct_db_session()
    try:
        user_a = _create_user(db, "gmail-shared-mailbox-a")
        oauth_a = _create_gmail_oauth(db, user_a)
        state_a = _create_gmail_watch_state(db, user_a, oauth_a, callback_id="cb-a")
        setattr(state_a, "history_id", "100")

        user_b = _create_user(db, "gmail-shared-mailbox-b")
        oauth_b = UserOAuth(
            user_id=int(user_b.id),
            provider="gmail",
            access_token="access-token-b",
            provider_user_id="provider-user-b",
            email="codeacme17@gmail.com",
        )
        db.add(oauth_b)
        db.commit()
        db.refresh(oauth_b)
        state_b = _create_gmail_watch_state(db, user_b, oauth_b, callback_id="cb-b")
        setattr(state_b, "history_id", "500")
        db.add_all([state_a, state_b])
        db.commit()

        fake_service = _FakeGmailService(history_response={"history": []})
        result = asyncio.run(
            collect_gmail_pubsub_events(
                db,
                GmailPubsubNotification(
                    email_address="codeacme17@gmail.com",
                    history_id="600",
                    pubsub_message_id="pubsub-shared",
                ),
                state=state_b,
                service_factory=lambda _db, oauth: fake_service,
            )
        )

        assert result.events == []
        # History listing started from B's cursor, not A's.
        assert fake_service.history_resource.calls == [
            {
                "userId": "me",
                "startHistoryId": "500",
                "historyTypes": ["messageAdded"],
            }
        ]
        db.refresh(state_a)
        assert state_a.history_id == "100"
    finally:
        db.close()
