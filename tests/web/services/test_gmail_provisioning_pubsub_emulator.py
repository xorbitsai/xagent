"""Per-mailbox Gmail provisioning against a real Pub/Sub emulator.

These tests run the actual google-cloud-pubsub SDK (no fakes) against the
Pub/Sub emulator, covering resource creation, push-endpoint resync, and
reference-counted teardown. They are skipped unless PUBSUB_EMULATOR_HOST is
set, e.g.:

    docker run -d -p 8681:8681 thekevjames/gcloud-pubsub-emulator
    PUBSUB_EMULATOR_HOST=localhost:8681 pytest tests/web/services/test_gmail_provisioning_pubsub_emulator.py
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from sqlalchemy.orm import Session

from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.gmail_watch import GmailWatchState
from xagent.web.models.trigger import TriggerProvisioningStatus
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services.gmail_provisioning import (
    ensure_gmail_mailbox_provisioned,
    gmail_subscription_path,
    gmail_topic_path,
    release_gmail_mailbox_if_unused,
)

pytestmark = pytest.mark.skipif(
    not os.getenv("PUBSUB_EMULATOR_HOST"),
    reason="requires a Pub/Sub emulator (set PUBSUB_EMULATOR_HOST)",
)

PROJECT = "e2e-emulator-project"
MAILBOX = "emulator.owner@gmail.example"


class _FakeExec:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def execute(self) -> dict[str, Any]:
        return self._payload


class _FakeGmailUsers:
    def watch(self, *, userId: str, body: dict[str, Any]) -> _FakeExec:
        return _FakeExec({"historyId": "emu-1", "expiration": "4102444800000"})

    def stop(self, *, userId: str) -> _FakeExec:
        return _FakeExec({})


class _FakeGmailService:
    def users(self) -> _FakeGmailUsers:
        return _FakeGmailUsers()


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'pubsub_emulator.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


@pytest.fixture(autouse=True)
def _gmail_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_PROJECT_ID", PROJECT)
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.emulator.example")
    monkeypatch.setenv(
        "XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT",
        "pubsub-push@e2e-emulator-project.iam.gserviceaccount.com",
    )
    monkeypatch.setenv("XAGENT_GMAIL_WATCH_ENABLED", "true")


@pytest.fixture()
def account(db_session: Session) -> UserOAuth:
    user = User(username="emulator-user", password_hash="hash")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    Agent(  # keep parity with real data shape; not otherwise used
        user_id=int(user.id),
        name="Emu agent",
        description="d",
        instructions="i",
        status=AgentStatus.DRAFT,
    )
    oauth = UserOAuth(
        user_id=int(user.id), provider="gmail", access_token="tok", email=MAILBOX
    )
    db_session.add(oauth)
    db_session.commit()
    db_session.refresh(oauth)
    return oauth


def _cleanup_cloud_resources() -> None:
    from google.api_core.exceptions import NotFound
    from google.cloud import pubsub_v1

    subscriber = pubsub_v1.SubscriberClient()
    publisher = pubsub_v1.PublisherClient()
    try:
        subscriber.delete_subscription(
            request={"subscription": gmail_subscription_path(PROJECT, MAILBOX)}
        )
    except NotFound:
        pass
    try:
        publisher.delete_topic(request={"topic": gmail_topic_path(PROJECT, MAILBOX)})
    except NotFound:
        pass


@pytest.fixture(autouse=True)
def _clean_emulator():
    _cleanup_cloud_resources()
    yield
    _cleanup_cloud_resources()


def _provision(db: Session, oauth: UserOAuth) -> GmailWatchState:
    # Real SDK clients via the default factories (emulator-backed).
    return ensure_gmail_mailbox_provisioned(
        db,
        oauth,
        service_factory=lambda _db, _oauth: _FakeGmailService(),
    )


def test_provisioning_creates_real_pubsub_resources(db_session, account) -> None:
    from google.cloud import pubsub_v1

    state = _provision(db_session, account)

    assert state.status == TriggerProvisioningStatus.ACTIVE.value
    assert state.last_error is None

    subscriber = pubsub_v1.SubscriberClient()
    subscription = subscriber.get_subscription(
        request={"subscription": gmail_subscription_path(PROJECT, MAILBOX)}
    )
    assert subscription.topic == gmail_topic_path(PROJECT, MAILBOX)
    assert subscription.push_config.push_endpoint == state.push_audience
    assert state.push_audience.startswith(
        "https://api.emulator.example/api/triggers/callback/gmail/"
    )


def test_repeated_provisioning_is_idempotent(db_session, account) -> None:
    first = _provision(db_session, account)
    second = _provision(db_session, account)
    assert second.status == TriggerProvisioningStatus.ACTIVE.value
    assert second.callback_id == first.callback_id
    assert second.push_audience == first.push_audience


def test_push_endpoint_resyncs_after_base_url_change(
    db_session, account, monkeypatch
) -> None:
    from google.cloud import pubsub_v1

    _provision(db_session, account)
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api-v2.emulator.example")
    state = _provision(db_session, account)

    assert state.push_audience.startswith("https://api-v2.emulator.example/")
    subscriber = pubsub_v1.SubscriberClient()
    subscription = subscriber.get_subscription(
        request={"subscription": gmail_subscription_path(PROJECT, MAILBOX)}
    )
    assert subscription.push_config.push_endpoint == state.push_audience


def test_release_deletes_real_resources_when_unreferenced(db_session, account) -> None:
    from google.api_core.exceptions import NotFound
    from google.cloud import pubsub_v1

    _provision(db_session, account)

    released = release_gmail_mailbox_if_unused(
        db_session,
        int(account.id),
        service_factory=lambda _db, _oauth: _FakeGmailService(),
    )

    assert released is True
    subscriber = pubsub_v1.SubscriberClient()
    with pytest.raises(NotFound):
        subscriber.get_subscription(
            request={"subscription": gmail_subscription_path(PROJECT, MAILBOX)}
        )
    publisher = pubsub_v1.PublisherClient()
    with pytest.raises(NotFound):
        publisher.get_topic(request={"topic": gmail_topic_path(PROJECT, MAILBOX)})
    assert db_session.query(GmailWatchState).count() == 0
