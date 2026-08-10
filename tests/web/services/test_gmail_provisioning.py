from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, NoReturn

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy import text as sql_text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.pool import QueuePool

from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.database import (
    Base,
    get_db,
    get_engine,
    get_session_local,
    init_db,
)
from xagent.web.models.gmail_watch import GmailWatchState
from xagent.web.models.trigger import (
    AgentTrigger,
    TriggerProvisioningStatus,
    TriggerType,
)
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services import gmail_provisioning
from xagent.web.services.gmail_provisioning import (
    GMAIL_PUSH_PUBLISHER,
    ensure_gmail_mailbox_provisioned,
    gmail_subscription_path,
    gmail_topic_path,
    reconcile_gmail_push_endpoints,
    reconcile_gmail_trigger_provisioning,
    release_gmail_mailbox_if_unused,
    sweep_gmail_provisioning,
)
from xagent.web.services.trigger_providers import gmail as gmail_trigger_provider


def test_gmail_callback_url_builds_the_canonical_callback_contract() -> None:
    helper = getattr(gmail_provisioning, "gmail_callback_url", None)

    assert callable(helper)
    assert helper("https://api.example.com", "callback-id") == (
        "https://api.example.com/api/triggers/callback/gmail/callback-id"
    )


def test_callback_audience_accepts_token_minted_before_recent_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the previous callback audience valid through Pub/Sub token reuse."""
    monkeypatch.setenv(
        "XAGENT_S2S_API_BASE_URL",
        "https://sg-origin.cloud.example.test",
    )
    previous_audience = (
        "https://legacy-origin.cloud.example.test/"
        "api/triggers/callback/gmail/callback-id"
    )
    rotated_at = datetime.now(timezone.utc) - timedelta(minutes=55)
    state = GmailWatchState(
        callback_id="callback-id",
        push_audience=(
            "https://sg-origin.cloud.example.test/"
            "api/triggers/callback/gmail/callback-id"
        ),
        previous_push_audience=previous_audience,
        previous_push_audience_expires_at=(
            rotated_at + gmail_provisioning.GMAIL_CALLBACK_AUDIENCE_GRACE_PERIOD
        ),
    )

    audiences = gmail_trigger_provider._accepted_callback_audiences(
        state,
        "callback-id",
    )

    assert previous_audience in audiences


def test_transition_lock_yields_the_database_session(
    db_session: Session,
) -> None:
    with gmail_provisioning._gmail_watch_transition_lock(
        db_session,
        oauth_account_id=7,
    ) as transition_db:
        assert transition_db is db_session


def test_postgresql_transition_uses_only_the_lock_owning_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_engine("sqlite://", poolclass=QueuePool)
    engine.dialect.name = "postgresql"
    monkeypatch.setattr(
        gmail_provisioning,
        "text",
        lambda _statement: sql_text("SELECT 1"),
    )

    with Session(bind=engine) as db:
        db.execute(sql_text("SELECT 1"))
        assert engine.pool.checkedout() == 1

        with gmail_provisioning._gmail_watch_transition_lock(
            db,
            oauth_account_id=7,
        ) as transition_db:
            assert transition_db is not db
            assert engine.pool.checkedout() == 1
            transition_db.execute(sql_text("SELECT 1"))
            transition_db.commit()
            assert engine.pool.checkedout() == 1

    assert engine.pool.checkedout() == 0


class FakeExecutable:
    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.response = response or {}

    def execute(self) -> dict[str, Any]:
        return self.response


class FakeGmailUsers:
    def __init__(self, service: FakeGmailService) -> None:
        self.service = service

    def watch(self, *, userId: str, body: dict[str, Any]) -> FakeExecutable:
        self.service.watch_calls.append({"userId": userId, "body": body})
        return FakeExecutable(
            {
                "historyId": self.service.history_id,
                "expiration": self.service.expiration,
            }
        )

    def stop(self, *, userId: str) -> FakeExecutable:
        self.service.stop_calls.append({"userId": userId})
        return FakeExecutable()


class FakeGmailService:
    def __init__(
        self, *, history_id: str = "hist-1", expiration: str = "4102444800000"
    ) -> None:
        self.history_id = history_id
        self.expiration = expiration
        self.watch_calls: list[dict[str, Any]] = []
        self.stop_calls: list[dict[str, Any]] = []

    def users(self) -> FakeGmailUsers:
        return FakeGmailUsers(self)


class FakeBinding:
    def __init__(self, *, role: str, members: list[str]) -> None:
        self.role = role
        self.members = members


class FakeBindings(list[FakeBinding]):
    def add(self, *, role: str, members: list[str]) -> FakeBinding:
        binding = FakeBinding(role=role, members=members)
        self.append(binding)
        return binding


class FakePolicy:
    def __init__(self) -> None:
        self.bindings = FakeBindings()


class FakePublisher:
    def __init__(self) -> None:
        self.topics: set[str] = set()
        self.policies: dict[str, FakePolicy] = {}
        self.deleted_topics: list[str] = []

    def create_topic(self, *, request: dict[str, str]) -> None:
        self.topics.add(request["name"])

    def get_iam_policy(self, *, request: dict[str, str]) -> FakePolicy:
        return self.policies.setdefault(request["resource"], FakePolicy())

    def set_iam_policy(self, *, request: dict[str, Any]) -> None:
        self.policies[request["resource"]] = request["policy"]

    def delete_topic(self, *, request: dict[str, str]) -> None:
        self.deleted_topics.append(request["topic"])
        self.topics.discard(request["topic"])


class FakeSubscriber:
    def __init__(self) -> None:
        self.subscriptions: dict[str, dict[str, Any]] = {}
        self.deleted_subscriptions: list[str] = []

    def create_subscription(self, *, request: dict[str, Any]) -> None:
        self.subscriptions[request["name"]] = request

    def delete_subscription(self, *, request: dict[str, str]) -> None:
        self.deleted_subscriptions.append(request["subscription"])
        self.subscriptions.pop(request["subscription"], None)


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'gmail_provisioning.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


@pytest.fixture(autouse=True)
def gmail_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_PROJECT_ID", "demo-project")
    monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_TOPIC_PREFIX", "xagent-gmail")
    monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_SUBSCRIPTION_PREFIX", "xagent-gmail-push")
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.example.com/")
    monkeypatch.delenv("XAGENT_S2S_API_BASE_URL", raising=False)
    monkeypatch.setenv(
        "XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT",
        "pubsub-push@demo-project.iam.gserviceaccount.com",
    )


def _create_user(db: Session) -> User:
    user = User(
        username="owner",
        email="owner@example.com",
        password_hash="hash",
        is_admin=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _create_agent(db: Session, user: User) -> Agent:
    agent = Agent(
        user_id=int(user.id),
        name="Gmail agent",
        description="test",
        instructions="test",
        status=AgentStatus.DRAFT,
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return agent


def _create_oauth(db: Session, user: User, *, email: str = "Owner@Gmail.Example"):
    account = UserOAuth(
        user_id=int(user.id),
        provider="gmail",
        access_token="access-token",
        email=email,
    )
    db.add(account)
    db.commit()
    db.refresh(account)
    return account


def _create_gmail_trigger(
    db: Session,
    user: User,
    agent: Agent,
    account: UserOAuth,
    *,
    enabled: bool = True,
) -> AgentTrigger:
    trigger = AgentTrigger(
        user_id=int(user.id),
        agent_id=int(agent.id),
        type=TriggerType.GMAIL.value,
        name="Gmail inbox",
        enabled=enabled,
        provider=TriggerType.GMAIL.value,
        resource_id=str(account.email).lower(),
        config={"watch_label": "INBOX", "oauth_account_id": int(account.id)},
    )
    db.add(trigger)
    db.commit()
    db.refresh(trigger)
    return trigger


def test_provisioning_creates_deterministic_resources_and_active_state(
    db_session: Session,
) -> None:
    user = _create_user(db_session)
    account = _create_oauth(db_session, user)
    publisher = FakePublisher()
    subscriber = FakeSubscriber()
    gmail = FakeGmailService()

    state = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: gmail,
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: subscriber,
    )

    email = "owner@gmail.example"
    expected_topic = gmail_topic_path("demo-project", email)
    expected_subscription = gmail_subscription_path("demo-project", email)
    expected_audience = (
        f"https://api.example.com/api/triggers/callback/gmail/{state.callback_id}"
    )
    assert state.status == TriggerProvisioningStatus.ACTIVE.value
    assert state.last_error is None
    assert state.topic_name == expected_topic
    assert state.subscription_name == expected_subscription
    assert state.push_audience == expected_audience
    assert state.history_id == "hist-1"
    assert publisher.topics == {expected_topic}
    assert set(subscriber.subscriptions) == {expected_subscription}
    assert subscriber.subscriptions[expected_subscription]["push_config"] == {
        "push_endpoint": expected_audience,
        "oidc_token": {
            "service_account_email": "pubsub-push@demo-project.iam.gserviceaccount.com",
            "audience": expected_audience,
        },
    }
    assert gmail.watch_calls == [
        {
            "userId": "me",
            "body": {"topicName": expected_topic, "labelIds": ["INBOX"]},
        }
    ]


def test_s2s_base_url_overrides_public_api_base(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAGENT_S2S_API_BASE_URL", "https://callbacks.example.com/")
    user = _create_user(db_session)
    account = _create_oauth(db_session, user)
    subscriber = FakeSubscriber()

    state = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: FakeGmailService(),
        publisher_factory=lambda: FakePublisher(),
        subscriber_factory=lambda: subscriber,
    )

    expected_audience = (
        f"https://callbacks.example.com/api/triggers/callback/gmail/{state.callback_id}"
    )
    assert state.status == TriggerProvisioningStatus.ACTIVE.value
    assert state.push_audience == expected_audience
    stored = subscriber.subscriptions[str(state.subscription_name)]
    assert stored["push_config"]["push_endpoint"] == expected_audience
    assert stored["push_config"]["oidc_token"]["audience"] == expected_audience


def test_legacy_callback_base_url_remains_a_gmail_fallback(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "XAGENT_TRIGGER_CALLBACK_BASE_URL",
        "https://legacy-callback.example.com/",
    )
    user = _create_user(db_session)
    account = _create_oauth(db_session, user)

    state = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: FakeGmailService(),
        publisher_factory=lambda: FakePublisher(),
        subscriber_factory=lambda: FakeSubscriber(),
    )

    assert state.push_audience == (
        "https://legacy-callback.example.com/api/triggers/callback/gmail/"
        f"{state.callback_id}"
    )


def test_missing_callback_base_urls_record_failed_state_without_app_base_fallback(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("XAGENT_PUBLIC_API_BASE_URL", raising=False)
    monkeypatch.setenv("XAGENT_APP_BASE_URL", "https://frontend.example.com")
    user = _create_user(db_session)
    account = _create_oauth(db_session, user)

    state = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: FakeGmailService(),
        publisher_factory=lambda: FakePublisher(),
        subscriber_factory=lambda: FakeSubscriber(),
    )

    assert state.status == TriggerProvisioningStatus.FAILED.value
    assert "XAGENT_S2S_API_BASE_URL" in str(state.last_error)
    assert "XAGENT_PUBLIC_API_BASE_URL" in str(state.last_error)
    assert state.push_audience is None


def test_sweep_retries_stale_failed_referenced_mailbox(db_session: Session) -> None:
    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    account = _create_oauth(db_session, user)
    _create_gmail_trigger(db_session, user, agent, account)
    state = GmailWatchState(
        user_id=int(user.id),
        oauth_account_id=int(account.id),
        email="owner@gmail.example",
        history_id="",
        topic_name="",
        status=TriggerProvisioningStatus.FAILED.value,
        last_error="old failure",
        updated_at=datetime.now(timezone.utc) - timedelta(minutes=10),
    )
    db_session.add(state)
    db_session.commit()
    publisher = FakePublisher()
    subscriber = FakeSubscriber()
    gmail = FakeGmailService(history_id="hist-retry")

    attempts = sweep_gmail_provisioning(
        db_session,
        service_factory=lambda _db, _account: gmail,
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: subscriber,
    )

    refreshed = (
        db_session.query(GmailWatchState)
        .filter(GmailWatchState.oauth_account_id == int(account.id))
        .one()
    )
    assert attempts == 1
    assert refreshed.status == TriggerProvisioningStatus.ACTIVE.value
    assert refreshed.history_id == "hist-retry"
    assert refreshed.last_error is None


def test_sweep_matches_duplicate_mailboxes_by_oauth_account_id(
    db_session: Session,
) -> None:
    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    first = _create_oauth(db_session, user, email="shared@gmail.example")
    second = _create_oauth(db_session, user, email="shared@gmail.example")
    _create_gmail_trigger(db_session, user, agent, first)
    for account in (first, second):
        db_session.add(
            GmailWatchState(
                user_id=int(user.id),
                oauth_account_id=int(account.id),
                email="shared@gmail.example",
                history_id="",
                topic_name="",
                status=TriggerProvisioningStatus.FAILED.value,
                last_error="old failure",
                updated_at=datetime.now(timezone.utc) - timedelta(minutes=10),
            )
        )
    db_session.commit()

    attempts = sweep_gmail_provisioning(
        db_session,
        service_factory=lambda _db, _account: FakeGmailService(),
        publisher_factory=lambda: FakePublisher(),
        subscriber_factory=lambda: FakeSubscriber(),
    )

    assert attempts == 1
    states = {
        int(state.oauth_account_id): state.status
        for state in db_session.query(GmailWatchState).all()
    }
    assert states == {
        int(first.id): TriggerProvisioningStatus.ACTIVE.value,
        int(second.id): TriggerProvisioningStatus.FAILED.value,
    }


def test_best_effort_provision_matches_duplicate_mailboxes_by_account_id(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    first = _create_oauth(db_session, user, email="shared@gmail.example")
    _create_oauth(db_session, user, email="shared@gmail.example")
    _create_gmail_trigger(db_session, user, agent, first)
    provisioned: list[int] = []

    monkeypatch.setattr(
        gmail_provisioning,
        "ensure_gmail_mailbox_provisioned",
        lambda _db, account: provisioned.append(int(account.id)),
    )

    gmail_provisioning.best_effort_provision_gmail_watches_for_user(
        db_session,
        user_id=int(user.id),
        context="test",
    )

    assert provisioned == [int(first.id)]


def test_best_effort_provision_swallows_account_lookup_failure(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failing account lookup must not escape the best-effort contract."""
    # No trigger fixture here: the patched query raises on the service's first
    # statement, so the binding data would never be read.
    user = _create_user(db_session)
    provisioned: list[int] = []

    monkeypatch.setattr(
        gmail_provisioning,
        "ensure_gmail_mailbox_provisioned",
        lambda _db, account: provisioned.append(int(account.id)),
    )

    def raising_query(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise OperationalError("SELECT user_oauth", {}, Exception("connection lost"))

    monkeypatch.setattr(db_session, "query", raising_query)
    caplog.set_level(logging.WARNING, logger=gmail_provisioning.__name__)

    gmail_provisioning.best_effort_provision_gmail_watches_for_user(
        db_session,
        user_id=int(user.id),
        context="test",
    )

    assert provisioned == []
    assert "Failed to resolve Gmail accounts for user" in caplog.text
    assert "connection lost" in caplog.text


def test_best_effort_provision_swallows_binding_resolution_failure(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failing trigger-binding lookup must not escape either."""
    # The account lookup runs for real here, so the mailbox fixture matters;
    # the binding resolution it feeds is patched out.
    user = _create_user(db_session)
    _create_oauth(db_session, user)
    provisioned: list[int] = []

    monkeypatch.setattr(
        gmail_provisioning,
        "ensure_gmail_mailbox_provisioned",
        lambda _db, account: provisioned.append(int(account.id)),
    )

    def raising_resolution(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise OperationalError(
            "SELECT agent_triggers", {}, Exception("statement timeout")
        )

    monkeypatch.setattr(
        gmail_provisioning,
        "_referenced_gmail_oauth_account_ids",
        raising_resolution,
    )
    caplog.set_level(logging.WARNING, logger=gmail_provisioning.__name__)

    gmail_provisioning.best_effort_provision_gmail_watches_for_user(
        db_session,
        user_id=int(user.id),
        context="test",
    )

    assert provisioned == []
    assert "Failed to resolve Gmail accounts for user" in caplog.text
    assert "statement timeout" in caplog.text


def test_unregister_releases_mailbox_only_after_last_enabled_trigger_is_deleted(
    db_session: Session,
) -> None:
    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    account = _create_oauth(db_session, user)
    first = _create_gmail_trigger(db_session, user, agent, account)
    second = _create_gmail_trigger(db_session, user, agent, account)
    publisher = FakePublisher()
    subscriber = FakeSubscriber()
    gmail = FakeGmailService()
    state = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: gmail,
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: subscriber,
    )

    assert (
        release_gmail_mailbox_if_unused(
            db_session,
            int(account.id),
            service_factory=lambda _db, _account: gmail,
            publisher_factory=lambda: publisher,
            subscriber_factory=lambda: subscriber,
        )
        is False
    )
    db_session.delete(first)
    db_session.commit()
    assert (
        release_gmail_mailbox_if_unused(
            db_session,
            int(account.id),
            service_factory=lambda _db, _account: gmail,
            publisher_factory=lambda: publisher,
            subscriber_factory=lambda: subscriber,
        )
        is False
    )
    db_session.delete(second)
    db_session.commit()

    assert (
        release_gmail_mailbox_if_unused(
            db_session,
            int(account.id),
            service_factory=lambda _db, _account: gmail,
            publisher_factory=lambda: publisher,
            subscriber_factory=lambda: subscriber,
        )
        is True
    )
    assert gmail.stop_calls == [{"userId": "me"}]
    assert subscriber.deleted_subscriptions == [state.subscription_name]
    assert publisher.deleted_topics == [state.topic_name]
    assert db_session.query(GmailWatchState).count() == 0


async def test_gmail_provider_register_unregister_offload_sync_sdk_work(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xagent.web.services.trigger_providers.gmail import GmailProvider

    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    account = _create_oauth(db_session, user)
    trigger = _create_gmail_trigger(db_session, user, agent, account)
    calls: list[str] = []

    def fake_provision(_db: Session, provisioned_trigger: AgentTrigger) -> str:
        calls.append(f"provision:{provisioned_trigger.id}")
        setattr(
            provisioned_trigger,
            "provisioning_status",
            TriggerProvisioningStatus.ACTIVE.value,
        )
        setattr(provisioned_trigger, "provisioning_error", None)
        _db.add(provisioned_trigger)
        _db.commit()
        return TriggerProvisioningStatus.ACTIVE.value

    def fake_release(_db: Session, oauth_account_id: int) -> bool:
        calls.append(f"release:{oauth_account_id}")
        return True

    async def fake_to_thread(fn, *args, **kwargs):
        calls.append(f"to_thread:{fn.__name__}")
        return fn(*args, **kwargs)

    monkeypatch.setattr(
        "xagent.web.services.trigger_providers.gmail.provision_gmail_trigger",
        fake_provision,
    )
    monkeypatch.setattr(
        "xagent.web.services.trigger_providers.gmail.release_gmail_mailbox_if_unused",
        fake_release,
    )
    monkeypatch.setattr(
        "xagent.web.services.trigger_providers.gmail.asyncio.to_thread",
        fake_to_thread,
    )

    provider = GmailProvider()
    result = await provider.register(db_session, trigger, object())
    # unregister resolves the binding from config alone; the trigger row may
    # already be rebound or deleted when CRUD dispatches it.
    await provider.unregister(
        db_session, trigger, {"oauth_account_id": int(account.id)}
    )

    assert result.status == TriggerProvisioningStatus.ACTIVE
    assert calls == [
        "to_thread:fake_provision",
        f"provision:{trigger.id}",
        "to_thread:fake_release",
        f"release:{account.id}",
    ]


def test_slow_registration_returns_pending_then_reconciles_to_active(
    db_session: Session,
) -> None:
    """Slow cloud provisioning yields pending; the thread converges later."""
    import threading

    from xagent.web.services.gmail_provisioning import provision_gmail_trigger

    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    account = _create_oauth(db_session, user)
    trigger = _create_gmail_trigger(db_session, user, agent, account)

    release = threading.Event()
    publisher = FakePublisher()
    subscriber = FakeSubscriber()
    gmail = FakeGmailService(history_id="hist-slow")
    threads: list[threading.Thread] = []

    def slow_provision(account_id: int) -> None:
        release.wait(timeout=10)
        from xagent.web.models.database import get_session_local

        db2 = get_session_local()()
        try:
            slow_account = db2.query(UserOAuth).filter(UserOAuth.id == account_id).one()
            ensure_gmail_mailbox_provisioned(
                db2,
                slow_account,
                service_factory=lambda _db, _account: gmail,
                publisher_factory=lambda: publisher,
                subscriber_factory=lambda: subscriber,
            )
        finally:
            db2.close()

    def run_in_thread(account_id: int) -> threading.Thread:
        thread = threading.Thread(target=slow_provision, args=(account_id,))
        thread.start()
        threads.append(thread)
        return thread

    status = provision_gmail_trigger(
        db_session,
        trigger,
        timeout_seconds=0,
        run_in_thread=run_in_thread,
    )
    assert status == TriggerProvisioningStatus.PENDING.value
    assert trigger.provisioning_status == TriggerProvisioningStatus.PENDING.value

    release.set()
    threads[0].join(timeout=10)
    assert not threads[0].is_alive()

    db_session.expire_all()
    state = (
        db_session.query(GmailWatchState)
        .filter(GmailWatchState.oauth_account_id == int(account.id))
        .one()
    )
    assert state.status == TriggerProvisioningStatus.ACTIVE.value
    assert state.history_id == "hist-slow"

    # The periodic sweep - not another user-initiated create/update - must
    # surface the converged state on the trigger the API serves.
    attempts = sweep_gmail_provisioning(
        db_session,
        service_factory=lambda _db, _account: gmail,
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: subscriber,
    )
    assert attempts == 0  # state is active; nothing to re-register
    db_session.refresh(trigger)
    assert trigger.provisioning_status == TriggerProvisioningStatus.ACTIVE.value
    assert trigger.provisioning_error is None


def test_reconcile_copies_watch_state_status_onto_triggers(
    db_session: Session,
) -> None:
    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    account = _create_oauth(db_session, user)
    trigger = _create_gmail_trigger(db_session, user, agent, account)
    disabled = _create_gmail_trigger(db_session, user, agent, account, enabled=False)
    setattr(trigger, "provisioning_status", TriggerProvisioningStatus.PENDING.value)
    setattr(disabled, "provisioning_status", TriggerProvisioningStatus.PENDING.value)
    db_session.add_all([trigger, disabled])
    state = GmailWatchState(
        user_id=int(user.id),
        oauth_account_id=int(account.id),
        email="owner@gmail.example",
        history_id="hist-1",
        topic_name="projects/demo-project/topics/xagent-gmail-abc",
        status=TriggerProvisioningStatus.ACTIVE.value,
    )
    db_session.add(state)
    db_session.commit()

    updated = reconcile_gmail_trigger_provisioning(db_session)

    assert updated == 1
    db_session.refresh(trigger)
    db_session.refresh(disabled)
    assert trigger.provisioning_status == TriggerProvisioningStatus.ACTIVE.value
    assert trigger.provisioning_error is None
    # Disabled triggers hold no watch reference; their status is not touched.
    assert disabled.provisioning_status == TriggerProvisioningStatus.PENDING.value

    # Failures propagate too, including the error message.
    setattr(state, "status", TriggerProvisioningStatus.FAILED.value)
    setattr(state, "last_error", "watch registration denied")
    db_session.add(state)
    db_session.commit()

    assert reconcile_gmail_trigger_provisioning(db_session) == 1
    db_session.refresh(trigger)
    assert trigger.provisioning_status == TriggerProvisioningStatus.FAILED.value
    assert trigger.provisioning_error == "watch registration denied"

    # Idempotent: nothing to update on a second pass.
    assert reconcile_gmail_trigger_provisioning(db_session) == 0


@pytest.mark.parametrize(
    ("candidate_count", "expected_page_queries"),
    [
        # Short final page (5 = 2+2+1): the len(page) < page_size branch
        # terminates without an extra query.
        (5, 3),
        # Exact multiple of the page size (4 = 2+2): termination needs one
        # extra empty-page query, taking the `if not page` branch.
        (4, 3),
    ],
)
def test_reconcile_full_scan_pages_candidates_with_bounded_queries(
    db_session: Session,
    candidate_count: int,
    expected_page_queries: int,
) -> None:
    """The sweep-path reconcile (triggers=None) walks candidates in keyset
    pages: every diverged trigger still reconciles, and every candidate
    query carries the page bound instead of scanning system-wide."""
    from sqlalchemy import event as sa_event

    from xagent.web.models.database import get_engine

    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    triggers = []
    for index in range(candidate_count):
        account = _create_oauth(db_session, user, email=f"owner{index}@gmail.example")
        trigger = _create_gmail_trigger(db_session, user, agent, account)
        setattr(trigger, "provisioning_status", TriggerProvisioningStatus.PENDING.value)
        db_session.add(trigger)
        db_session.add(
            GmailWatchState(
                user_id=int(user.id),
                oauth_account_id=int(account.id),
                email=str(account.email).lower(),
                history_id=f"hist-{index}",
                topic_name=f"projects/demo-project/topics/xagent-gmail-{index}",
                status=TriggerProvisioningStatus.ACTIVE.value,
            )
        )
        triggers.append(trigger)
    db_session.commit()

    statements: list[str] = []

    def _track(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    engine = get_engine()
    sa_event.listen(engine, "before_cursor_execute", _track)
    try:
        updated = reconcile_gmail_trigger_provisioning(db_session, batch_size=2)
    finally:
        sa_event.remove(engine, "before_cursor_execute", _track)

    assert updated == candidate_count
    for trigger in triggers:
        db_session.refresh(trigger)
        assert trigger.provisioning_status == TriggerProvisioningStatus.ACTIVE.value

    candidate_queries = [
        s for s in statements if "FROM agent_triggers" in s and "SELECT" in s
    ]
    assert candidate_queries, "expected candidate page queries"
    assert all("LIMIT" in s for s in candidate_queries)
    assert len(candidate_queries) == expected_page_queries


class ResyncFakeSubscriber(FakeSubscriber):
    """FakeSubscriber that behaves like Pub/Sub for existing subscriptions."""

    def __init__(self) -> None:
        super().__init__()
        self.modify_calls: list[dict[str, Any]] = []

    def create_subscription(self, *, request: dict[str, Any]) -> None:
        if request["name"] in self.subscriptions:
            from google.api_core.exceptions import AlreadyExists

            raise AlreadyExists("subscription exists")
        super().create_subscription(request=request)

    def get_subscription(self, *, request: dict[str, str]) -> Any:
        from types import SimpleNamespace

        stored = self.subscriptions[request["subscription"]]
        push_config = stored["push_config"]
        oidc = push_config.get("oidc_token")
        return SimpleNamespace(
            push_config=SimpleNamespace(
                push_endpoint=push_config["push_endpoint"],
                oidc_token=(
                    SimpleNamespace(
                        service_account_email=oidc.get("service_account_email", ""),
                        audience=oidc.get("audience", ""),
                    )
                    if oidc is not None
                    else None
                ),
            )
        )

    def modify_push_config(self, *, request: dict[str, Any]) -> None:
        self.modify_calls.append(request)
        self.subscriptions[request["subscription"]]["push_config"] = request[
            "push_config"
        ]


def test_existing_subscription_endpoint_resyncs_after_base_url_change(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = _create_user(db_session)
    account = _create_oauth(db_session, user)
    publisher = FakePublisher()
    subscriber = ResyncFakeSubscriber()
    gmail = FakeGmailService()

    first = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: gmail,
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: subscriber,
    )
    old_audience = str(first.push_audience)
    assert subscriber.modify_calls == []

    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api-v2.example.com")
    second = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: gmail,
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: subscriber,
    )

    new_audience = (
        f"https://api-v2.example.com/api/triggers/callback/gmail/{second.callback_id}"
    )
    assert second.push_audience == new_audience
    assert new_audience != old_audience
    assert len(subscriber.modify_calls) == 1
    stored = subscriber.subscriptions[str(second.subscription_name)]
    assert stored["push_config"]["push_endpoint"] == new_audience
    assert stored["push_config"]["oidc_token"]["audience"] == new_audience


def test_provisioning_keeps_stored_audience_until_pubsub_accepts_transition(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class TransitionInspectingSubscriber(ResyncFakeSubscriber):
        watch_id: int | None = None

        def modify_push_config(self, *, request: dict[str, Any]) -> None:
            assert self.watch_id is not None
            with get_session_local()() as observer:
                persisted = (
                    observer.query(GmailWatchState)
                    .filter(GmailWatchState.id == self.watch_id)
                    .one()
                )
                observed.update(
                    {
                        "push_audience": persisted.push_audience,
                        "previous_push_audience": persisted.previous_push_audience,
                        "previous_push_audience_expires_at": (
                            persisted.previous_push_audience_expires_at
                        ),
                    }
                )
            super().modify_push_config(request=request)

    user = _create_user(db_session)
    account = _create_oauth(db_session, user)
    publisher = FakePublisher()
    subscriber = TransitionInspectingSubscriber()
    first = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: FakeGmailService(),
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: subscriber,
    )
    subscriber.watch_id = int(first.id)
    previous_audience = str(first.push_audience)
    monkeypatch.setenv("XAGENT_S2S_API_BASE_URL", "https://sg-origin.cloud.xagent.co")

    ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: FakeGmailService(),
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: subscriber,
    )

    assert observed["push_audience"] == previous_audience
    assert observed["previous_push_audience"] is None
    assert observed["previous_push_audience_expires_at"] is None


def test_existing_subscription_resyncs_a_stale_oidc_audience(
    db_session: Session,
) -> None:
    user = _create_user(db_session)
    account = _create_oauth(db_session, user)
    subscriber = ResyncFakeSubscriber()

    first = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: FakeGmailService(),
        publisher_factory=lambda: FakePublisher(),
        subscriber_factory=lambda: subscriber,
    )
    subscription = subscriber.subscriptions[str(first.subscription_name)]
    subscription["push_config"]["oidc_token"]["audience"] = (
        "https://stale.example.com/callback"
    )

    second = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: FakeGmailService(),
        publisher_factory=lambda: FakePublisher(),
        subscriber_factory=lambda: subscriber,
    )

    assert second.status == TriggerProvisioningStatus.ACTIVE.value
    assert len(subscriber.modify_calls) == 1
    assert subscription["push_config"]["oidc_token"]["audience"] == second.push_audience


def test_reconcile_push_endpoint_uses_s2s_base_without_reregistering_watch(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Endpoint migration must preserve Gmail's existing history cursor."""
    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    account = _create_oauth(db_session, user)
    _create_gmail_trigger(db_session, user, agent, account)
    publisher = FakePublisher()
    subscriber = ResyncFakeSubscriber()
    gmail = FakeGmailService(history_id="cursor-before-migration")
    state = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: gmail,
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: subscriber,
    )
    previous_audience = str(state.push_audience)
    preserved = {
        "callback_id": state.callback_id,
        "history_id": state.history_id,
        "watch_expiration": state.watch_expiration,
        "status": state.status,
    }
    monkeypatch.setenv("XAGENT_S2S_API_BASE_URL", "https://sg-origin.cloud.xagent.co/")

    result = reconcile_gmail_push_endpoints(
        db_session,
        execute=True,
        subscriber_factory=lambda: subscriber,
    )

    db_session.refresh(state)
    expected_audience = (
        "https://sg-origin.cloud.xagent.co/api/triggers/callback/gmail/"
        f"{state.callback_id}"
    )
    assert result.scanned == 1
    assert result.changed == 1
    assert result.unchanged == 0
    assert result.failed == 0
    assert state.push_audience == expected_audience
    assert getattr(state, "previous_push_audience", None) == previous_audience
    grace_expires_at = getattr(state, "previous_push_audience_expires_at", None)
    assert grace_expires_at is not None
    if grace_expires_at.tzinfo is None:
        grace_expires_at = grace_expires_at.replace(tzinfo=timezone.utc)
    assert grace_expires_at > datetime.now(timezone.utc)
    assert {
        "callback_id": state.callback_id,
        "history_id": state.history_id,
        "watch_expiration": state.watch_expiration,
        "status": state.status,
    } == preserved
    assert gmail.watch_calls == [
        {
            "userId": "me",
            "body": {
                "topicName": str(state.topic_name),
                "labelIds": ["INBOX"],
            },
        }
    ]
    assert subscriber.modify_calls == [
        {
            "subscription": str(state.subscription_name),
            "push_config": {
                "push_endpoint": expected_audience,
                "oidc_token": {
                    "service_account_email": (
                        "pubsub-push@demo-project.iam.gserviceaccount.com"
                    ),
                    "audience": expected_audience,
                },
            },
        }
    ]

    second = reconcile_gmail_push_endpoints(
        db_session,
        execute=True,
        subscriber_factory=lambda: subscriber,
    )
    assert second.scanned == 1
    assert second.changed == 0
    assert second.unchanged == 1
    assert second.failed == 0
    assert len(subscriber.modify_calls) == 1


def test_reconcile_keeps_stored_audience_until_pubsub_accepts_transition(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    account = _create_oauth(db_session, user)
    _create_gmail_trigger(db_session, user, agent, account)
    observed: dict[str, object] = {}

    class TransitionInspectingSubscriber(ResyncFakeSubscriber):
        watch_id: int | None = None

        def modify_push_config(self, *, request: dict[str, Any]) -> None:
            assert self.watch_id is not None
            with get_session_local()() as observer:
                persisted = (
                    observer.query(GmailWatchState)
                    .filter(GmailWatchState.id == self.watch_id)
                    .one()
                )
                observed.update(
                    {
                        "push_audience": persisted.push_audience,
                        "previous_push_audience": (persisted.previous_push_audience),
                        "previous_push_audience_expires_at": (
                            persisted.previous_push_audience_expires_at
                        ),
                    }
                )
            super().modify_push_config(request=request)

    subscriber = TransitionInspectingSubscriber()
    state = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: FakeGmailService(),
        publisher_factory=lambda: FakePublisher(),
        subscriber_factory=lambda: subscriber,
    )
    subscriber.watch_id = int(state.id)
    previous_audience = str(state.push_audience)
    monkeypatch.setenv("XAGENT_S2S_API_BASE_URL", "https://sg-origin.cloud.xagent.co")

    result = reconcile_gmail_push_endpoints(
        db_session,
        execute=True,
        subscriber_factory=lambda: subscriber,
    )

    assert result.changed == 1
    assert observed["push_audience"] == previous_audience
    assert observed["previous_push_audience"] is None
    assert observed["previous_push_audience_expires_at"] is None


def test_reconcile_unchanged_outcome_releases_the_row_lock_transaction(
    db_session: Session,
) -> None:
    user = _create_user(db_session)
    account = _create_oauth(db_session, user)
    subscriber = ResyncFakeSubscriber()
    state = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: FakeGmailService(),
        publisher_factory=lambda: FakePublisher(),
        subscriber_factory=lambda: subscriber,
    )
    state_row = (
        int(state.id),
        int(state.oauth_account_id),
        str(state.email),
        str(state.callback_id),
        str(state.subscription_name),
        str(state.push_audience),
    )
    db_session.rollback()

    outcome = gmail_provisioning._reconcile_gmail_push_endpoint(
        db_session,
        state_row=state_row,
        base_url="https://api.example.com",
        push_service_account="pubsub-push@demo-project.iam.gserviceaccount.com",
        subscriber=subscriber,
        execute=True,
    )

    assert outcome == "unchanged"
    assert db_session.in_transaction() is False


def test_reconcile_skipped_outcome_releases_the_row_lock_transaction(
    db_session: Session,
) -> None:
    user = _create_user(db_session)
    account = _create_oauth(db_session, user)
    subscriber = ResyncFakeSubscriber()
    state = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: FakeGmailService(),
        publisher_factory=lambda: FakePublisher(),
        subscriber_factory=lambda: subscriber,
    )
    state_row = (
        int(state.id),
        int(state.oauth_account_id),
        str(state.email),
        str(state.callback_id),
        str(state.subscription_name),
        str(state.push_audience),
    )
    db_session.delete(state)
    db_session.commit()

    outcome = gmail_provisioning._reconcile_gmail_push_endpoint(
        db_session,
        state_row=state_row,
        base_url="https://api.example.com",
        push_service_account="pubsub-push@demo-project.iam.gserviceaccount.com",
        subscriber=subscriber,
        execute=True,
    )

    assert outcome == "skipped"
    assert db_session.in_transaction() is False


def test_reconcile_audit_detects_cloud_drift_when_database_is_current(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    account = _create_oauth(db_session, user)
    _create_gmail_trigger(db_session, user, agent, account)
    subscriber = ResyncFakeSubscriber()
    state = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: FakeGmailService(),
        publisher_factory=lambda: FakePublisher(),
        subscriber_factory=lambda: subscriber,
    )
    monkeypatch.setenv("XAGENT_S2S_API_BASE_URL", "https://sg-origin.cloud.xagent.co")
    expected_audience = (
        "https://sg-origin.cloud.xagent.co/api/triggers/callback/gmail/"
        f"{state.callback_id}"
    )
    setattr(state, "push_audience", expected_audience)
    db_session.commit()

    result = reconcile_gmail_push_endpoints(
        db_session,
        subscriber_factory=lambda: subscriber,
    )

    assert result.scanned == 1
    assert result.changed == 1
    assert result.unchanged == 0
    assert result.failed == 0
    assert subscriber.modify_calls == []


def test_reconcile_execute_repairs_cloud_drift_when_database_is_current(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    account = _create_oauth(db_session, user)
    _create_gmail_trigger(db_session, user, agent, account)
    subscriber = ResyncFakeSubscriber()
    state = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: FakeGmailService(),
        publisher_factory=lambda: FakePublisher(),
        subscriber_factory=lambda: subscriber,
    )
    monkeypatch.setenv("XAGENT_S2S_API_BASE_URL", "https://sg-origin.cloud.xagent.co")
    expected_audience = (
        "https://sg-origin.cloud.xagent.co/api/triggers/callback/gmail/"
        f"{state.callback_id}"
    )
    setattr(state, "push_audience", expected_audience)
    db_session.commit()

    result = reconcile_gmail_push_endpoints(
        db_session,
        execute=True,
        subscriber_factory=lambda: subscriber,
    )

    stored = subscriber.subscriptions[str(state.subscription_name)]
    assert result.scanned == 1
    assert result.changed == 1
    assert result.unchanged == 0
    assert result.failed == 0
    assert stored["push_config"]["push_endpoint"] == expected_audience
    assert stored["push_config"]["oidc_token"]["audience"] == expected_audience


def test_reconcile_cloud_drift_refreshes_durable_previous_audience_grace(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preserve the old audience when repairing a legacy DB-first transition."""
    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    account = _create_oauth(db_session, user)
    _create_gmail_trigger(db_session, user, agent, account)
    subscriber = ResyncFakeSubscriber()
    state = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: FakeGmailService(),
        publisher_factory=lambda: FakePublisher(),
        subscriber_factory=lambda: subscriber,
    )
    previous_audience = str(state.push_audience)
    frozen_now = datetime.now(timezone.utc)
    monkeypatch.setattr(gmail_provisioning, "_now", lambda: frozen_now)
    monkeypatch.setenv("XAGENT_S2S_API_BASE_URL", "https://sg-origin.cloud.xagent.co")
    expected_audience = (
        "https://sg-origin.cloud.xagent.co/api/triggers/callback/gmail/"
        f"{state.callback_id}"
    )
    setattr(state, "push_audience", expected_audience)
    setattr(state, "previous_push_audience", previous_audience)
    setattr(
        state,
        "previous_push_audience_expires_at",
        frozen_now - timedelta(minutes=1),
    )
    db_session.commit()

    result = reconcile_gmail_push_endpoints(
        db_session,
        execute=True,
        subscriber_factory=lambda: subscriber,
    )

    db_session.refresh(state)
    stored = subscriber.subscriptions[str(state.subscription_name)]
    grace_expires_at = state.previous_push_audience_expires_at
    assert grace_expires_at is not None
    if grace_expires_at.tzinfo is None:
        grace_expires_at = grace_expires_at.replace(tzinfo=timezone.utc)
    assert result.changed == 1
    assert stored["push_config"]["push_endpoint"] == expected_audience
    assert stored["push_config"]["oidc_token"]["audience"] == expected_audience
    assert state.previous_push_audience == previous_audience
    assert grace_expires_at == (
        frozen_now + gmail_provisioning.GMAIL_CALLBACK_AUDIENCE_GRACE_PERIOD
    )
    assert previous_audience in gmail_trigger_provider._accepted_callback_audiences(
        state,
        str(state.callback_id),
    )


def test_reconcile_execute_supports_update_only_pubsub_permissions(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    account = _create_oauth(db_session, user)
    _create_gmail_trigger(db_session, user, agent, account)
    subscriber = ResyncFakeSubscriber()
    state = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: FakeGmailService(),
        publisher_factory=lambda: FakePublisher(),
        subscriber_factory=lambda: subscriber,
    )
    monkeypatch.setenv("XAGENT_S2S_API_BASE_URL", "https://sg-origin.cloud.xagent.co")

    def deny_inspection(*, request: dict[str, str]) -> Any:
        raise PermissionError(f"cannot inspect {request['subscription']}")

    monkeypatch.setattr(subscriber, "get_subscription", deny_inspection)

    result = reconcile_gmail_push_endpoints(
        db_session,
        execute=True,
        subscriber_factory=lambda: subscriber,
    )

    db_session.refresh(state)
    expected_audience = (
        "https://sg-origin.cloud.xagent.co/api/triggers/callback/gmail/"
        f"{state.callback_id}"
    )
    assert result.changed == 1
    assert result.failed == 0
    assert state.push_audience == expected_audience
    assert subscriber.modify_calls[-1]["push_config"]["push_endpoint"] == (
        expected_audience
    )


def test_reconcile_push_endpoint_dry_run_does_not_change_cloud_or_database(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    account = _create_oauth(db_session, user)
    _create_gmail_trigger(db_session, user, agent, account)
    subscriber = ResyncFakeSubscriber()
    state = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: FakeGmailService(),
        publisher_factory=lambda: FakePublisher(),
        subscriber_factory=lambda: subscriber,
    )
    old_audience = str(state.push_audience)
    monkeypatch.setenv("XAGENT_S2S_API_BASE_URL", "https://sg-origin.cloud.xagent.co")

    result = reconcile_gmail_push_endpoints(
        db_session,
        execute=False,
        subscriber_factory=lambda: subscriber,
    )

    db_session.refresh(state)
    assert result.scanned == 1
    assert result.changed == 1
    assert result.failed == 0
    assert state.push_audience == old_audience
    assert subscriber.modify_calls == []


def test_reconcile_matches_enabled_triggers_by_oauth_account_id(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    account_a = _create_oauth(db_session, user, email="shared@gmail.example")
    account_b = _create_oauth(db_session, user, email="shared@gmail.example")
    _create_gmail_trigger(db_session, user, agent, account_a)
    subscriber = ResyncFakeSubscriber()

    for account in (account_a, account_b):
        ensure_gmail_mailbox_provisioned(
            db_session,
            account,
            service_factory=lambda _db, _account: FakeGmailService(),
            publisher_factory=lambda: FakePublisher(),
            subscriber_factory=lambda: subscriber,
        )

    monkeypatch.setenv("XAGENT_S2S_API_BASE_URL", "https://sg-origin.cloud.xagent.co")
    result = reconcile_gmail_push_endpoints(
        db_session,
        subscriber_factory=lambda: subscriber,
    )

    assert result.scanned == 1


@pytest.mark.parametrize("oauth_account_id", [None, "malformed"])
def test_reconcile_legacy_binding_falls_back_to_mailbox_email(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    oauth_account_id: object,
) -> None:
    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    account = _create_oauth(db_session, user)
    trigger = _create_gmail_trigger(db_session, user, agent, account)
    subscriber = ResyncFakeSubscriber()
    ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: FakeGmailService(),
        publisher_factory=lambda: FakePublisher(),
        subscriber_factory=lambda: subscriber,
    )
    trigger.config = {
        "watch_label": "INBOX",
        **(
            {"oauth_account_id": oauth_account_id}
            if oauth_account_id is not None
            else {}
        ),
    }
    db_session.commit()
    monkeypatch.setenv("XAGENT_S2S_API_BASE_URL", "https://sg-origin.cloud.xagent.co")

    result = reconcile_gmail_push_endpoints(
        db_session,
        subscriber_factory=lambda: subscriber,
    )

    assert result.scanned == 1


def test_reconcile_bounds_gmail_trigger_lookup_to_each_watch_page(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each trigger query is bounded by one keyset-paged watch batch."""
    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    subscriber = ResyncFakeSubscriber()
    for email in ("first@gmail.example", "second@gmail.example"):
        account = _create_oauth(db_session, user, email=email)
        _create_gmail_trigger(db_session, user, agent, account)
        ensure_gmail_mailbox_provisioned(
            db_session,
            account,
            service_factory=lambda _db, _account: FakeGmailService(),
            publisher_factory=lambda: FakePublisher(),
            subscriber_factory=lambda: subscriber,
        )

    monkeypatch.setenv("XAGENT_S2S_API_BASE_URL", "https://sg-origin.cloud.xagent.co")
    trigger_selects: list[str] = []
    watch_selects: list[str] = []
    engine = get_engine()

    def capture_trigger_select(
        _connection: Any,
        _cursor: Any,
        statement: str,
        _parameters: Any,
        _context: Any,
        _executemany: bool,
    ) -> None:
        if "FROM agent_triggers" in statement:
            trigger_selects.append(statement)
        if "FROM gmail_watch_states" in statement and "SELECT" in statement:
            watch_selects.append(statement)

    event.listen(engine, "before_cursor_execute", capture_trigger_select)
    try:
        result = reconcile_gmail_push_endpoints(
            db_session,
            subscriber_factory=lambda: subscriber,
            batch_size=1,
        )
    finally:
        event.remove(engine, "before_cursor_execute", capture_trigger_select)

    assert result.scanned == 2
    assert result.changed == 2
    assert len(trigger_selects) == 2
    assert all(" IN (" in statement for statement in trigger_selects)
    # The projection includes the ``json`` config column, which PostgreSQL
    # cannot compare, so the lookup must never be a SELECT DISTINCT.
    assert not any("DISTINCT" in statement.upper() for statement in trigger_selects)
    assert len(watch_selects) == 3
    assert all("LIMIT" in statement for statement in watch_selects)


def test_reconcile_continues_when_a_snapshotted_watch_is_deleted(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = _create_user(db_session)
    agent = _create_agent(db_session, user)

    class ConcurrentDeleteSubscriber(ResyncFakeSubscriber):
        delete_watch_id: int | None = None
        deleted = False

        def modify_push_config(self, *, request: dict[str, Any]) -> None:
            super().modify_push_config(request=request)
            if self.deleted or self.delete_watch_id is None:
                return
            self.deleted = True
            with get_session_local()() as concurrent_db:
                concurrent_db.query(GmailWatchState).filter(
                    GmailWatchState.id == self.delete_watch_id
                ).delete(synchronize_session=False)
                concurrent_db.commit()

    subscriber = ConcurrentDeleteSubscriber()
    states: list[GmailWatchState] = []
    for email in ("first@gmail.example", "second@gmail.example"):
        account = _create_oauth(db_session, user, email=email)
        _create_gmail_trigger(db_session, user, agent, account)
        states.append(
            ensure_gmail_mailbox_provisioned(
                db_session,
                account,
                service_factory=lambda _db, _account: FakeGmailService(),
                publisher_factory=lambda: FakePublisher(),
                subscriber_factory=lambda: subscriber,
            )
        )

    second_watch_id = int(states[1].id)
    subscriber.delete_watch_id = second_watch_id
    monkeypatch.setenv("XAGENT_S2S_API_BASE_URL", "https://sg-origin.cloud.xagent.co")

    result = reconcile_gmail_push_endpoints(
        db_session,
        execute=True,
        subscriber_factory=lambda: subscriber,
    )

    assert result.scanned == 2
    assert result.changed == 1
    assert result.skipped == 1
    assert result.failed == 0
    assert result.errors == ()
    assert len(subscriber.modify_calls) == 1
    assert (
        db_session.query(GmailWatchState)
        .filter(GmailWatchState.id == second_watch_id)
        .first()
        is None
    )


def test_reconcile_reports_watch_deleted_after_its_cloud_update(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    account = _create_oauth(db_session, user)
    _create_gmail_trigger(db_session, user, agent, account)

    class DeleteAfterModifySubscriber(ResyncFakeSubscriber):
        delete_watch_id: int | None = None

        def modify_push_config(self, *, request: dict[str, Any]) -> None:
            super().modify_push_config(request=request)
            assert self.delete_watch_id is not None
            with get_session_local()() as concurrent_db:
                concurrent_db.query(GmailWatchState).filter(
                    GmailWatchState.id == self.delete_watch_id
                ).delete(synchronize_session=False)
                concurrent_db.commit()

    subscriber = DeleteAfterModifySubscriber()
    state = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: FakeGmailService(),
        publisher_factory=lambda: FakePublisher(),
        subscriber_factory=lambda: subscriber,
    )
    state_id = int(state.id)
    subscriber.delete_watch_id = state_id
    monkeypatch.setenv("XAGENT_S2S_API_BASE_URL", "https://sg-origin.cloud.xagent.co")

    result = reconcile_gmail_push_endpoints(
        db_session,
        execute=True,
        subscriber_factory=lambda: subscriber,
    )

    assert result.scanned == 1
    assert result.changed == 0
    assert result.failed == 1
    assert str(state_id) in result.errors[0]


def test_reconciliation_serializes_with_concurrent_provisioning(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cloud and database audience changes must form one mailbox transition."""
    from xagent.web.models.database import get_session_local
    from xagent.web.services import gmail_provisioning

    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    account = _create_oauth(db_session, user)
    _create_gmail_trigger(db_session, user, agent, account)
    publisher = FakePublisher()
    reconcile_modified_cloud = threading.Event()
    allow_reconcile_to_persist = threading.Event()
    provisioning_finished = threading.Event()

    class PausingSubscriber(ResyncFakeSubscriber):
        def modify_push_config(self, *, request: dict[str, Any]) -> None:
            super().modify_push_config(request=request)
            if threading.current_thread().name != "gmail-reconcile":
                return
            reconcile_modified_cloud.set()
            assert allow_reconcile_to_persist.wait(timeout=30)

    subscriber = PausingSubscriber()
    state = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: FakeGmailService(),
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: subscriber,
    )
    account_id = int(account.id)
    state_id = int(state.id)
    previous_base_url = "https://api.example.com"
    next_base_url = "https://sg-origin.cloud.xagent.co"
    monkeypatch.setattr(
        gmail_provisioning,
        "get_gmail_callback_base_url",
        lambda: (
            previous_base_url
            if threading.current_thread().name == "gmail-provision"
            else next_base_url
        ),
    )
    errors: dict[str, BaseException] = {}

    def reconcile() -> None:
        session = get_session_local()()
        try:
            reconcile_gmail_push_endpoints(
                session,
                execute=True,
                subscriber_factory=lambda: subscriber,
            )
        except BaseException as exc:  # noqa: BLE001 - asserted below
            errors["reconcile"] = exc
        finally:
            session.close()

    def provision() -> None:
        session = get_session_local()()
        try:
            current_account = (
                session.query(UserOAuth).filter(UserOAuth.id == account_id).one()
            )
            ensure_gmail_mailbox_provisioned(
                session,
                current_account,
                service_factory=lambda _db, _account: FakeGmailService(),
                publisher_factory=lambda: publisher,
                subscriber_factory=lambda: subscriber,
            )
        except BaseException as exc:  # noqa: BLE001 - asserted below
            errors["provision"] = exc
        finally:
            session.close()
            provisioning_finished.set()

    reconciler = threading.Thread(target=reconcile, name="gmail-reconcile")
    reconciler.start()
    assert reconcile_modified_cloud.wait(timeout=30)

    provisioner = threading.Thread(target=provision, name="gmail-provision")
    provisioner.start()
    provisioner_finished_while_reconcile_paused = provisioning_finished.wait(timeout=1)
    allow_reconcile_to_persist.set()

    reconciler.join(timeout=30)
    provisioner.join(timeout=30)
    assert not reconciler.is_alive() and not provisioner.is_alive()
    assert errors == {}
    assert provisioner_finished_while_reconcile_paused is False

    db_session.expire_all()
    persisted = (
        db_session.query(GmailWatchState).filter(GmailWatchState.id == state_id).one()
    )
    cloud_audience = subscriber.subscriptions[str(persisted.subscription_name)][
        "push_config"
    ]["oidc_token"]["audience"]
    assert persisted.push_audience == cloud_audience


def test_existing_subscription_resyncs_after_service_account_change(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = _create_user(db_session)
    account = _create_oauth(db_session, user)
    publisher = FakePublisher()
    subscriber = ResyncFakeSubscriber()
    gmail = FakeGmailService()

    first = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: gmail,
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: subscriber,
    )
    assert subscriber.modify_calls == []

    # Only the OIDC push service account changes; the callback base URL (and
    # therefore push_endpoint/audience) stays the same.
    monkeypatch.setenv(
        "XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT",
        "rotated-push@demo-project.iam.gserviceaccount.com",
    )
    second = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: gmail,
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: subscriber,
    )

    assert second.push_audience == first.push_audience
    assert len(subscriber.modify_calls) == 1
    stored = subscriber.subscriptions[str(second.subscription_name)]
    assert (
        stored["push_config"]["oidc_token"]["service_account_email"]
        == "rotated-push@demo-project.iam.gserviceaccount.com"
    )


def test_existing_subscription_not_resynced_when_config_unchanged(
    db_session: Session,
) -> None:
    user = _create_user(db_session)
    account = _create_oauth(db_session, user)
    publisher = FakePublisher()
    subscriber = ResyncFakeSubscriber()
    gmail = FakeGmailService()

    kwargs = {
        "service_factory": lambda _db, _account: gmail,
        "publisher_factory": lambda: publisher,
        "subscriber_factory": lambda: subscriber,
    }
    ensure_gmail_mailbox_provisioned(db_session, account, **kwargs)
    ensure_gmail_mailbox_provisioned(db_session, account, **kwargs)

    assert subscriber.modify_calls == []


class InspectFailsFakeSubscriber(ResyncFakeSubscriber):
    """Existing subscription whose push config cannot be inspected."""

    def get_subscription(self, *, request: dict[str, str]) -> Any:
        raise RuntimeError("pubsub get_subscription unavailable")


class ModifyFailsFakeSubscriber(ResyncFakeSubscriber):
    """Existing subscription whose push config cannot be patched."""

    def modify_push_config(self, *, request: dict[str, Any]) -> None:
        raise RuntimeError("pubsub modify_push_config unavailable")


def test_inspect_failure_falls_through_to_unconditional_resync(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = _create_user(db_session)
    account = _create_oauth(db_session, user)
    publisher = FakePublisher()
    subscriber = InspectFailsFakeSubscriber()
    gmail = FakeGmailService()

    first = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: gmail,
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: subscriber,
    )
    assert first.status == TriggerProvisioningStatus.ACTIVE.value

    # A base-URL change forces a resync, but inspecting the existing
    # subscription fails. Inspection is only an optimization, so provisioning
    # degrades to an unconditional (idempotent) modify_push_config rather than
    # aborting: the watch still converges to ACTIVE with the new audience.
    monkeypatch.setenv("XAGENT_S2S_API_BASE_URL", "https://api-v2.example.com")
    second = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: gmail,
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: subscriber,
    )

    new_audience = (
        f"https://api-v2.example.com/api/triggers/callback/gmail/{second.callback_id}"
    )
    assert second.status == TriggerProvisioningStatus.ACTIVE.value
    assert second.push_audience == new_audience
    assert len(subscriber.modify_calls) == 1
    stored = subscriber.subscriptions[str(second.subscription_name)]
    assert stored["push_config"]["push_endpoint"] == new_audience


def test_patch_failure_marks_failed_not_active(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = _create_user(db_session)
    account = _create_oauth(db_session, user)
    publisher = FakePublisher()
    subscriber = ModifyFailsFakeSubscriber()
    gmail = FakeGmailService()

    first = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: gmail,
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: subscriber,
    )
    assert first.status == TriggerProvisioningStatus.ACTIVE.value
    old_audience = str(first.push_audience)

    # A base-URL change forces a resync, and the patch itself fails. The watch
    # must land FAILED without starting a bounded grace period for a cloud
    # transition that never happened. Callback verification independently
    # accepts the configured retry target alongside this stored cloud audience.
    monkeypatch.setenv("XAGENT_S2S_API_BASE_URL", "https://api-v2.example.com")
    second = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: gmail,
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: subscriber,
    )

    assert second.status == TriggerProvisioningStatus.FAILED.value
    assert "pubsub modify_push_config unavailable" in str(second.last_error)
    assert second.push_audience == old_audience
    assert second.previous_push_audience is None
    assert second.previous_push_audience_expires_at is None


class GetIamPolicyFailsFakePublisher(FakePublisher):
    """Topic whose IAM policy cannot be read."""

    def get_iam_policy(self, *, request: dict[str, str]) -> FakePolicy:
        raise RuntimeError("pubsub get_iam_policy unavailable")


class SetIamPolicyFailsFakePublisher(FakePublisher):
    """Topic whose IAM policy cannot be written."""

    def set_iam_policy(self, *, request: dict[str, Any]) -> None:
        raise RuntimeError("pubsub set_iam_policy unavailable")


@pytest.mark.parametrize(
    ("publisher_cls", "expected_error"),
    [
        (GetIamPolicyFailsFakePublisher, "pubsub get_iam_policy unavailable"),
        (SetIamPolicyFailsFakePublisher, "pubsub set_iam_policy unavailable"),
    ],
)
def test_iam_policy_failure_marks_failed_not_active(
    db_session: Session,
    publisher_cls: type[FakePublisher],
    expected_error: str,
) -> None:
    user = _create_user(db_session)
    account = _create_oauth(db_session, user)
    gmail = FakeGmailService()

    # If the roles/pubsub.publisher grant for the Gmail push identity cannot
    # be verified or applied, Gmail may be unable to publish to the topic and
    # the trigger would silently never fire. The failure must propagate so the
    # watch lands FAILED for the retry sweep instead of recording ACTIVE.
    state = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: gmail,
        publisher_factory=lambda: publisher_cls(),
        subscriber_factory=lambda: FakeSubscriber(),
    )

    assert state.status == TriggerProvisioningStatus.FAILED.value
    assert expected_error in str(state.last_error)
    # Provisioning must stop at the IAM failure: no watch was registered.
    assert gmail.watch_calls == []


def test_iam_grant_appends_to_existing_publisher_binding(
    db_session: Session,
) -> None:
    user = _create_user(db_session)
    account = _create_oauth(db_session, user)
    publisher = FakePublisher()
    topic_path = gmail_topic_path("demo-project", "owner@gmail.example")
    existing_policy = FakePolicy()
    existing_policy.bindings.add(
        role="roles/pubsub.publisher", members=["serviceAccount:other@example.iam"]
    )
    publisher.policies[topic_path] = existing_policy

    state = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: FakeGmailService(),
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: FakeSubscriber(),
    )

    assert state.status == TriggerProvisioningStatus.ACTIVE.value
    # The Gmail push identity joins the existing roles/pubsub.publisher
    # binding instead of creating a duplicate binding for the same role.
    bindings = publisher.policies[topic_path].bindings
    assert len(bindings) == 1
    assert bindings[0].members == [
        "serviceAccount:other@example.iam",
        f"serviceAccount:{GMAIL_PUSH_PUBLISHER}",
    ]


def test_renewal_scan_uses_per_mailbox_provisioning_when_project_configured(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Watch renewal must not point per-mailbox states back at the global topic."""
    from xagent.web.services import gmail_provisioning
    from xagent.web.services.gmail_triggers import scan_due_gmail_watch_renewals

    monkeypatch.setenv("XAGENT_GMAIL_WATCH_ENABLED", "true")
    monkeypatch.delenv("XAGENT_GMAIL_PUBSUB_TOPIC", raising=False)

    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    account = _create_oauth(db_session, user)
    _create_gmail_trigger(db_session, user, agent, account)
    stale = GmailWatchState(
        user_id=int(user.id),
        oauth_account_id=int(account.id),
        email="owner@gmail.example",
        history_id="old",
        topic_name="projects/demo-project/topics/legacy-global",
        watch_expiration=datetime.now(timezone.utc) - timedelta(hours=1),
        status=TriggerProvisioningStatus.ACTIVE.value,
    )
    db_session.add(stale)
    db_session.commit()

    publisher = FakePublisher()
    subscriber = FakeSubscriber()
    monkeypatch.setattr(gmail_provisioning, "_default_publisher", lambda: publisher)
    monkeypatch.setattr(gmail_provisioning, "_default_subscriber", lambda: subscriber)
    gmail = FakeGmailService(history_id="hist-renewed")

    renewed = scan_due_gmail_watch_renewals(
        db_session,
        service_factory=lambda _db, _account: gmail,
    )

    assert renewed == 1
    refreshed = (
        db_session.query(GmailWatchState)
        .filter(GmailWatchState.oauth_account_id == int(account.id))
        .one()
    )
    expected_topic = gmail_topic_path("demo-project", "owner@gmail.example")
    assert refreshed.topic_name == expected_topic
    assert refreshed.status == TriggerProvisioningStatus.ACTIVE.value
    assert refreshed.history_id == "hist-renewed"
    assert gmail.watch_calls[-1]["body"]["topicName"] == expected_topic


@pytest.fixture()
def pg_session():
    """Session against a real Postgres, where SELECT ... FOR UPDATE locks.

    SQLite silently no-ops row locks, so the provisioning/release contention
    path can only be exercised here. Set XAGENT_TEST_POSTGRES_URL to run
    (CI provides it in the PostgreSQL job).
    """
    url = os.getenv("XAGENT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    init_db(db_url=url)
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.mark.postgresql
def test_postgresql_transition_lock_contends_per_oauth_account(
    pg_session: Session,
) -> None:
    """Exercise the real PostgreSQL advisory-lock scope and release behavior.

    A transition for the same OAuth account must wait, while a transition for
    a different account must remain independent. Releasing the first lock must
    then let its waiter complete. This complements the SQLite pool fake above:
    only PostgreSQL can prove the actual advisory-lock semantics.
    """
    same_lock_attempted = threading.Event()
    same_lock_acquired = threading.Event()
    other_lock_acquired = threading.Event()
    errors: list[BaseException] = []

    def acquire_lock(
        oauth_account_id: int,
        attempted: threading.Event,
        acquired: threading.Event,
    ) -> None:
        session = get_session_local()()
        try:
            attempted.set()
            with gmail_provisioning._gmail_watch_transition_lock(
                session,
                oauth_account_id=oauth_account_id,
            ):
                acquired.set()
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assert below
            errors.append(exc)
        finally:
            session.close()

    same_account = threading.Thread(
        target=acquire_lock,
        args=(7001, same_lock_attempted, same_lock_acquired),
    )
    other_account_attempted = threading.Event()
    other_account = threading.Thread(
        target=acquire_lock,
        args=(7002, other_account_attempted, other_lock_acquired),
    )

    with gmail_provisioning._gmail_watch_transition_lock(
        pg_session,
        oauth_account_id=7001,
    ):
        same_account.start()
        assert same_lock_attempted.wait(timeout=10)
        assert not same_lock_acquired.wait(timeout=0.5)

        other_account.start()
        assert other_account_attempted.wait(timeout=10)
        assert other_lock_acquired.wait(timeout=10)
        other_account.join(timeout=10)
        assert not other_account.is_alive()

    assert same_lock_acquired.wait(timeout=10)
    same_account.join(timeout=10)
    assert not same_account.is_alive()
    assert errors == []


@pytest.mark.postgresql
def test_release_and_reprovision_contend_on_the_watch_state_lock(
    pg_session: Session,
) -> None:
    """Unregister-of-last-trigger and a concurrent provision serialize.

    While release_gmail_mailbox_if_unused holds the watch-state row lock,
    _get_or_create_watch_state must block instead of updating a row that is
    about to be deleted (which strands the new trigger at PENDING via
    StaleDataError on the losing commit).
    """
    from xagent.web.models.database import get_session_local

    db = pg_session
    user = _create_user(db)
    account = _create_oauth(db, user)
    publisher = FakePublisher()
    subscriber = FakeSubscriber()
    gmail = FakeGmailService()
    state = ensure_gmail_mailbox_provisioned(
        db,
        account,
        service_factory=lambda _db, _account: gmail,
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: subscriber,
    )
    old_callback_id = str(state.callback_id)
    account_id = int(account.id)

    release_holds_lock = threading.Event()
    release_may_finish = threading.Event()
    results: dict[str, Any] = {}

    def blocking_service_factory(_db: Session, _account: UserOAuth) -> FakeGmailService:
        # Called by release after it has taken FOR UPDATE on the state row.
        release_holds_lock.set()
        release_may_finish.wait(timeout=30)
        return gmail

    def do_release() -> None:
        db_a = get_session_local()()
        try:
            results["released"] = release_gmail_mailbox_if_unused(
                db_a,
                account_id,
                service_factory=blocking_service_factory,
                publisher_factory=lambda: publisher,
                subscriber_factory=lambda: subscriber,
            )
        finally:
            db_a.close()

    def do_provision() -> None:
        db_b = get_session_local()()
        try:
            account_b = db_b.query(UserOAuth).filter(UserOAuth.id == account_id).one()
            fresh = ensure_gmail_mailbox_provisioned(
                db_b,
                account_b,
                service_factory=lambda _db, _account: gmail,
                publisher_factory=lambda: publisher,
                subscriber_factory=lambda: subscriber,
            )
            results["status"] = str(fresh.status)
            results["callback_id"] = str(fresh.callback_id)
        finally:
            db_b.close()

    releaser = threading.Thread(target=do_release)
    releaser.start()
    assert release_holds_lock.wait(timeout=30)

    provisioner = threading.Thread(target=do_provision)
    provisioner.start()
    provisioner.join(timeout=1.0)
    # Provisioning is parked on the row lock, not racing the delete.
    assert provisioner.is_alive()

    release_may_finish.set()
    releaser.join(timeout=30)
    provisioner.join(timeout=30)
    assert not releaser.is_alive() and not provisioner.is_alive()

    assert results["released"] is True
    assert results["status"] == TriggerProvisioningStatus.ACTIVE.value
    # The mailbox was fully released first, then provisioned from scratch.
    assert results["callback_id"] != old_callback_id
    rows = (
        db.query(GmailWatchState)
        .filter(GmailWatchState.oauth_account_id == account_id)
        .all()
    )
    assert len(rows) == 1
    assert rows[0].status == TriggerProvisioningStatus.ACTIVE.value


def test_first_time_creation_race_adopts_winner_row(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FOR UPDATE takes no lock when the watch-state row does not exist yet,
    so two concurrent first-time enables can both reach the insert path. The
    loser's IntegrityError must adopt the winner's committed row instead of
    propagating a spurious error into the background thread."""
    from sqlalchemy.exc import IntegrityError

    from xagent.web.services.gmail_provisioning import _get_or_create_watch_state

    user = _create_user(db_session)
    account = _create_oauth(db_session, user)

    real_commit = db_session.commit
    real_rollback = db_session.rollback
    calls = {"count": 0}

    def racing_commit() -> None:
        calls["count"] += 1
        if calls["count"] == 1:
            # Simulate a concurrent winner committing between this session's
            # empty FOR UPDATE select and its own insert commit.
            real_rollback()
            winner = GmailWatchState(
                user_id=int(user.id),
                oauth_account_id=int(account.id),
                email="owner@gmail.example",
                history_id="hist-winner",
                topic_name="projects/demo-project/topics/winner",
                callback_id="winner-callback-id",
                status=TriggerProvisioningStatus.ACTIVE.value,
            )
            db_session.add(winner)
            real_commit()
            raise IntegrityError(
                "UNIQUE constraint failed: gmail_watch_states.oauth_account_id",
                params=None,
                orig=Exception("simulated concurrent insert"),
            )
        real_commit()

    monkeypatch.setattr(db_session, "commit", racing_commit)
    state = _get_or_create_watch_state(db_session, account, "owner@gmail.example")

    assert state.callback_id == "winner-callback-id"
    assert state.history_id == "hist-winner"
    assert state.status == TriggerProvisioningStatus.PENDING.value
    assert db_session.query(GmailWatchState).count() == 1


@pytest.mark.postgresql
def test_concurrent_first_time_creations_race_on_the_unique_constraint(
    pg_session: Session,
) -> None:
    """Two sessions both pass the empty FOR UPDATE select and insert; the
    loser must recover from the genuine unique-constraint violation by
    rolling back and adopting the winner's committed row. Unlike the mocked
    variant above, the loser's session really is in a failed transaction, so
    this fails if _get_or_create_watch_state drops its rollback."""
    from xagent.web.models.database import get_session_local
    from xagent.web.services.gmail_provisioning import _get_or_create_watch_state

    db = pg_session
    user = _create_user(db)
    account = _create_oauth(db, user)
    account_id = int(account.id)

    both_past_the_empty_select = threading.Barrier(2)
    results: dict[str, str] = {}
    errors: dict[str, BaseException] = {}

    def do_enable(name: str) -> None:
        session = get_session_local()()
        real_commit = session.commit
        insert_commit_pending = True

        def synchronized_commit() -> None:
            # Hold the insert commit until both sessions have run the empty
            # FOR UPDATE select, so both take the insert path; the loser's
            # adoption retry commit passes straight through.
            nonlocal insert_commit_pending
            if insert_commit_pending:
                insert_commit_pending = False
                both_past_the_empty_select.wait(timeout=30)
            real_commit()

        session.commit = synchronized_commit  # type: ignore[method-assign]
        try:
            acct = session.query(UserOAuth).filter(UserOAuth.id == account_id).one()
            state = _get_or_create_watch_state(session, acct, "owner@gmail.example")
            results[name] = str(state.callback_id)
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assert below
            errors[name] = exc
        finally:
            session.close()

    threads = [threading.Thread(target=do_enable, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads)

    assert errors == {}
    # The loser adopted the winner's row, so both report the same identity.
    assert results["a"] == results["b"]
    rows = (
        db.query(GmailWatchState)
        .filter(GmailWatchState.oauth_account_id == account_id)
        .all()
    )
    assert len(rows) == 1
    assert str(rows[0].callback_id) == results["a"]
    assert rows[0].status == TriggerProvisioningStatus.PENDING.value
    assert rows[0].email == "owner@gmail.example"


@pytest.mark.postgresql
def test_gmail_trigger_lookup_resolves_bindings_on_postgresql(
    pg_session: Session,
) -> None:
    """Resolve trigger bindings on the engine that rejects ``json`` equality.

    The lookup projects ``agent_triggers.config``, a ``JSON`` column. SQLite
    compares it happily, so only PostgreSQL can prove the query never asks for
    an equality operator the type does not have: a SELECT DISTINCT here fails
    with "could not identify an equality operator for type json", which broke
    every Gmail OAuth (re)connect, the retry sweep, and push reconciliation.
    Two triggers share one binding so deduplication is still covered.
    """
    db = pg_session
    user = _create_user(db)
    agent = _create_agent(db, user)
    account = _create_oauth(db, user)
    _create_gmail_trigger(db, user, agent, account)
    _create_gmail_trigger(db, user, agent, account)

    referenced = gmail_provisioning._referenced_gmail_oauth_account_ids(
        db,
        [(int(account.id), str(account.email or ""))],
    )

    assert referenced == {int(account.id)}


def test_provisioning_requires_account_email(db_session: Session) -> None:
    from xagent.web.services.gmail_provisioning import GmailProvisioningError

    user = _create_user(db_session)
    account = _create_oauth(db_session, user)
    setattr(account, "email", None)
    db_session.add(account)
    db_session.commit()

    with pytest.raises(GmailProvisioningError, match="email is required"):
        ensure_gmail_mailbox_provisioned(
            db_session,
            account,
            service_factory=lambda _db, _account: FakeGmailService(),
            publisher_factory=lambda: FakePublisher(),
            subscriber_factory=lambda: FakeSubscriber(),
        )
    assert db_session.query(GmailWatchState).count() == 0


def test_default_clients_select_rest_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """XAGENT_GMAIL_PUBSUB_TRANSPORT=rest must pick the GAPIC REST clients."""
    import google.pubsub_v1

    from xagent.web.services import gmail_provisioning

    captured: list[tuple[str, object]] = []

    class FakePublisher:
        def __init__(self, transport: str | None = None) -> None:
            captured.append(("publisher", transport))

    class FakeSubscriber:
        def __init__(self, transport: str | None = None) -> None:
            captured.append(("subscriber", transport))

    monkeypatch.setattr(google.pubsub_v1, "PublisherClient", FakePublisher)
    monkeypatch.setattr(google.pubsub_v1, "SubscriberClient", FakeSubscriber)
    monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_TRANSPORT", "rest")

    assert isinstance(gmail_provisioning._default_publisher(), FakePublisher)
    assert isinstance(gmail_provisioning._default_subscriber(), FakeSubscriber)
    assert captured == [("publisher", "rest"), ("subscriber", "rest")]


def test_is_already_exists_matches_both_transports() -> None:
    """gRPC raises AlreadyExists; REST maps HTTP 409 to its parent Conflict."""
    from google.api_core.exceptions import Aborted, AlreadyExists, Conflict

    from xagent.web.services.gmail_provisioning import _is_already_exists

    assert _is_already_exists(AlreadyExists("grpc duplicate topic"))
    assert _is_already_exists(Conflict("409 PUT .../topics/x: already exists"))
    assert not _is_already_exists(RuntimeError("unrelated"))
    # Aborted is also a Conflict subclass but signals a transient concurrency
    # error, not "already exists" — it must propagate, not be swallowed.
    assert not _is_already_exists(Aborted("transaction aborted"))
