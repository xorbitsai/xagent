"""Stub-provider tests for the unified trigger callback pipeline.

These tests exercise the provider protocol, registry, callback pipeline, and
audit trail without depending on webhook or Gmail implementation details.
"""

from __future__ import annotations

import json
from typing import Any, Mapping
from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session

from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.trigger import (
    AgentTrigger,
    TriggerAudit,
    TriggerAuditOutcome,
    TriggerProvisioningStatus,
    TriggerRun,
)
from xagent.web.models.user import User
from xagent.web.services.trigger_providers import (
    AckPolicy,
    CallbackRequestContext,
    ChallengeResponse,
    NormalizedEvent,
    RegistrationResult,
    TriggerEventParseError,
    UnknownTriggerProviderError,
    VerificationResult,
    get_trigger_provider,
    process_trigger_callback,
    register_trigger_provider,
    registered_trigger_provider_names,
    unregister_trigger_provider,
)
from xagent.web.services.trigger_providers.schemas import parse_trigger_config
from xagent.web.services.triggers import delete_agent_trigger

STUB_PROVIDER = "stub"
GOOD_TOKEN = "good-token"


class StubProvider:
    """Minimal provider used to exercise the pipeline contract."""

    name = STUB_PROVIDER

    def __init__(self, ack_policy: AckPolicy | None = None) -> None:
        self.ack_policy = ack_policy or AckPolicy()
        self.register_calls: list[int] = []
        self.unregister_calls: list[int] = []

    def validate_config(self, config: Mapping[str, Any]) -> Any:
        return parse_trigger_config("webhook", dict(config))

    def locate_trigger(self, db: Session, callback_id: str) -> AgentTrigger | None:
        return (
            db.query(AgentTrigger)
            .filter(
                AgentTrigger.callback_id == callback_id,
                AgentTrigger.provider == self.name,
            )
            .first()
        )

    def handle_challenge(
        self, context: CallbackRequestContext, raw_body: bytes
    ) -> ChallengeResponse | None:
        challenge = context.header("x-stub-challenge")
        if challenge:
            return ChallengeResponse(status_code=200, body=challenge)
        return None

    def authorize_resource(
        self,
        trigger: AgentTrigger,
        attested_resource_id: str | None,
        event: NormalizedEvent,
    ) -> bool:
        if trigger.resource_id is None:
            return True
        return (
            attested_resource_id is not None
            and attested_resource_id.lower() == str(trigger.resource_id).lower()
        )

    async def verify(
        self,
        context: CallbackRequestContext,
        *,
        db: Session,
        trigger: AgentTrigger | None,
        raw_body: bytes,
    ) -> VerificationResult:
        if context.header("x-stub-token") != GOOD_TOKEN:
            return VerificationResult.reject("bad stub token")
        return VerificationResult.ok(
            attested_resource_id=context.header("x-stub-resource")
        )

    async def register(
        self, db: Session, trigger: AgentTrigger, config: Any
    ) -> RegistrationResult:
        self.register_calls.append(int(trigger.id))
        return RegistrationResult(status=TriggerProvisioningStatus.ACTIVE)

    async def unregister(
        self,
        db: Session,
        trigger: AgentTrigger,
        config: Any,
        *,
        resource_id: str | None = None,
    ) -> None:
        self.unregister_calls.append(int(trigger.id))

    async def finalize_callback(
        self,
        *,
        db: Session,
        context: CallbackRequestContext,
        trigger: AgentTrigger | None,
        events: list[NormalizedEvent],
        raw_body: bytes,
    ) -> None:
        """Stub delivery has no cursor state to advance."""

    async def parse_events(
        self,
        context: CallbackRequestContext,
        *,
        db: Session,
        trigger: AgentTrigger | None,
        raw_body: bytes,
    ) -> list[NormalizedEvent]:
        try:
            decoded = json.loads(raw_body.decode("utf-8"))
        except ValueError as exc:
            raise TriggerEventParseError("stub body is not JSON") from exc
        events = decoded if isinstance(decoded, list) else [decoded]
        return [
            NormalizedEvent(
                event_type=str(item.get("type", "stub.event")),
                source_event_id=item.get("id"),
                target_trigger_id=item.get("target_trigger_id"),
                resource_id=item.get("resource"),
                payload=item,
            )
            for item in events
        ]


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'trigger_pipeline.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


@pytest.fixture()
def stub_provider():
    provider = StubProvider()
    register_trigger_provider(provider, replace=True)
    try:
        yield provider
    finally:
        unregister_trigger_provider(STUB_PROVIDER)


@pytest.fixture(autouse=True)
def _fast_run_start():
    """Skip real agent execution; run/task rows are still created."""

    async def _noop_start(db, *, run, wait_for_completion=False):
        return True

    with patch(
        "xagent.web.services.triggers.start_prepared_trigger_run",
        new=_noop_start,
    ):
        yield


def _create_user(db: Session) -> User:
    user = User(username="pipeline-user", password_hash="hash", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _create_agent(db: Session, user: User) -> Agent:
    agent = Agent(
        user_id=user.id,
        name="Pipeline Agent",
        description="pipeline",
        instructions="pipeline agent",
        execution_mode="balanced",
        models={"general": "test-model"},
        knowledge_bases=[],
        skills=[],
        tool_categories=[],
        suggested_prompts=[],
        status=AgentStatus.PUBLISHED,
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return agent


def _create_stub_trigger(
    db: Session,
    user: User,
    agent: Agent,
    *,
    callback_id: str = "cb-stub-1",
    resource_id: str | None = "res-1",
    enabled: bool = True,
    config: dict[str, Any] | None = None,
) -> AgentTrigger:
    trigger = AgentTrigger(
        user_id=user.id,
        agent_id=agent.id,
        type="webhook",
        name="Stub trigger",
        enabled=enabled,
        config=config or {},
        provider=STUB_PROVIDER,
        callback_id=callback_id,
        resource_id=resource_id,
    )
    db.add(trigger)
    db.commit()
    db.refresh(trigger)
    return trigger


def _context(
    *,
    callback_id: str = "cb-stub-1",
    headers: dict[str, str] | None = None,
    provider: str = STUB_PROVIDER,
) -> CallbackRequestContext:
    return CallbackRequestContext(
        provider=provider,
        callback_id=callback_id,
        headers=headers or {},
        remote_ip="203.0.113.9",
    )


def _good_headers(resource: str | None = "res-1") -> dict[str, str]:
    headers = {"X-Stub-Token": GOOD_TOKEN}
    if resource is not None:
        headers["X-Stub-Resource"] = resource
    return headers


def _event_body(event_id: str = "evt-1", event_type: str = "stub.event") -> bytes:
    return json.dumps({"id": event_id, "type": event_type, "n": 1}).encode()


def _audits(db: Session) -> list[TriggerAudit]:
    return db.query(TriggerAudit).order_by(TriggerAudit.id.asc()).all()


class TestProviderRegistry:
    def test_lookup_and_unknown_provider(self, stub_provider):
        assert get_trigger_provider(STUB_PROVIDER) is stub_provider
        assert STUB_PROVIDER in registered_trigger_provider_names()
        with pytest.raises(UnknownTriggerProviderError):
            get_trigger_provider("nope")

    def test_duplicate_registration_is_rejected(self, stub_provider):
        with pytest.raises(ValueError):
            register_trigger_provider(StubProvider())
        replacement = StubProvider()
        register_trigger_provider(replacement, replace=True)
        assert get_trigger_provider(STUB_PROVIDER) is replacement


class TestScheduledTriggerConfigValidation:
    """Negative coverage for the scheduled discriminated-union variant."""

    def test_non_positive_interval_is_rejected(self):
        from pydantic import ValidationError

        for bad_interval in (0, -60):
            with pytest.raises(ValidationError, match="interval_seconds"):
                parse_trigger_config("scheduled", {"interval_seconds": bad_interval})

    def test_schedule_source_is_required(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="interval_seconds or next_run_at"):
            parse_trigger_config("scheduled", {})
        with pytest.raises(ValidationError, match="interval_seconds or next_run_at"):
            parse_trigger_config("scheduled", {"next_run_at": "   "})

    def test_valid_schedules_are_accepted(self):
        by_interval = parse_trigger_config("scheduled", {"interval_seconds": 300})
        assert by_interval.interval_seconds == 300
        by_moment = parse_trigger_config(
            "scheduled", {"next_run_at": "2026-07-03T00:00:00+00:00"}
        )
        assert by_moment.next_run_at == "2026-07-03T00:00:00+00:00"


class TestGmailTriggerConfigValidation:
    @pytest.mark.parametrize(
        "oauth_account_id",
        [None, True, 1.5, "not-an-account-id", 0, "0"],
    )
    def test_rejects_invalid_account_binding(self, oauth_account_id: object):
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as error:
            parse_trigger_config(
                "gmail",
                {
                    "watch_label": "INBOX",
                    "oauth_account_id": oauth_account_id,
                },
            )

        errors = error.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("config", "gmail", "oauth_account_id")
        assert errors[0]["msg"] == (
            "Value error, oauth_account_id must be a positive integer"
        )
        assert errors[0]["input"] == oauth_account_id

    def test_accepts_missing_account_binding_for_legacy_records(self):
        config = parse_trigger_config("gmail", {"watch_label": "INBOX"})

        assert config.oauth_account_id is None

    def test_accepts_numeric_string_account_binding(self):
        config = parse_trigger_config(
            "gmail",
            {"watch_label": "INBOX", "oauth_account_id": "42"},
        )

        assert config.oauth_account_id == 42


class TestCallbackPipeline:
    async def test_unknown_provider_is_audited(self, db_session):
        result = await process_trigger_callback(
            db_session,
            context=_context(provider="ghost"),
            raw_body=b"{}",
        )
        assert result.status_code == 404
        assert result.outcome == TriggerAuditOutcome.UNKNOWN_PROVIDER
        audits = _audits(db_session)
        assert [a.outcome for a in audits] == ["unknown_provider"]
        assert audits[0].provider == "ghost"
        assert audits[0].callback_id == "cb-stub-1"
        assert audits[0].trigger_id is None
        assert audits[0].remote_ip == "203.0.113.9"

    async def test_challenge_short_circuits_before_verification(
        self, db_session, stub_provider
    ):
        result = await process_trigger_callback(
            db_session,
            context=_context(headers={"X-Stub-Challenge": "pong"}),
            raw_body=b"",
        )
        assert result.challenge is not None
        assert result.challenge.body == "pong"
        assert result.status_code == 200
        assert _audits(db_session) == []
        assert db_session.query(TriggerRun).count() == 0

    async def test_unknown_callback_id_is_audited(self, db_session, stub_provider):
        result = await process_trigger_callback(
            db_session,
            context=_context(callback_id="cb-missing", headers=_good_headers()),
            raw_body=_event_body(),
        )
        assert result.status_code == 404
        assert result.outcome == TriggerAuditOutcome.UNKNOWN_CALLBACK
        audits = _audits(db_session)
        assert [a.outcome for a in audits] == ["unknown_callback"]
        assert audits[0].callback_id == "cb-missing"

    async def test_ack_policy_separates_http_ack_from_audit_outcome(self, db_session):
        provider = StubProvider(ack_policy=AckPolicy(not_found_status=200))
        register_trigger_provider(provider, replace=True)
        try:
            result = await process_trigger_callback(
                db_session,
                context=_context(callback_id="cb-missing", headers=_good_headers()),
                raw_body=_event_body(),
            )
        finally:
            unregister_trigger_provider(STUB_PROVIDER)
        assert result.status_code == 200
        assert result.outcome == TriggerAuditOutcome.UNKNOWN_CALLBACK
        assert [a.outcome for a in _audits(db_session)] == ["unknown_callback"]

    async def test_failed_verification_is_audited(self, db_session, stub_provider):
        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        trigger = _create_stub_trigger(db_session, user, agent)

        result = await process_trigger_callback(
            db_session,
            context=_context(headers={"X-Stub-Token": "wrong"}),
            raw_body=_event_body(),
        )
        assert result.status_code == 401
        assert result.outcome == TriggerAuditOutcome.REJECTED_SIGNATURE
        audits = _audits(db_session)
        assert [a.outcome for a in audits] == ["rejected_signature"]
        assert audits[0].trigger_id == trigger.id
        assert audits[0].detail == {"reason": "bad stub token"}
        assert db_session.query(TriggerRun).count() == 0

    async def test_verification_error_keeps_redelivery_semantics(
        self, db_session, stub_provider
    ):
        """A verify() exception is a transient failure, not a rejection.

        Rejections can be ACKed by providers whose ack policy maps them to
        2xx (Gmail); an errored verification must instead answer with the
        failure status so the source redelivers.
        """
        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        trigger = _create_stub_trigger(db_session, user, agent)

        async def _broken_verify(context, *, db, trigger, raw_body):
            raise ConnectionError("JWKS fetch timed out")

        stub_provider.verify = _broken_verify

        result = await process_trigger_callback(
            db_session,
            context=_context(headers=_good_headers()),
            raw_body=_event_body(),
        )
        assert result.status_code == stub_provider.ack_policy.failure_status
        assert result.outcome == TriggerAuditOutcome.EXECUTION_FAILURE
        audits = _audits(db_session)
        assert [a.outcome for a in audits] == ["execution_failure"]
        assert audits[0].trigger_id == trigger.id
        assert audits[0].detail["stage"] == "verify"
        assert "ConnectionError" in audits[0].detail["error"]
        assert db_session.query(TriggerRun).count() == 0

    async def test_disabled_trigger_is_rejected_after_verification(
        self, db_session, stub_provider
    ):
        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        _create_stub_trigger(db_session, user, agent, enabled=False)

        result = await process_trigger_callback(
            db_session,
            context=_context(headers=_good_headers()),
            raw_body=_event_body(),
        )
        assert result.status_code == 409
        assert result.outcome == TriggerAuditOutcome.REJECTED_DISABLED
        assert [a.outcome for a in _audits(db_session)] == ["rejected_disabled"]
        assert db_session.query(TriggerRun).count() == 0

    async def test_unparseable_body_is_a_controlled_failure(
        self, db_session, stub_provider
    ):
        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        _create_stub_trigger(db_session, user, agent)

        result = await process_trigger_callback(
            db_session,
            context=_context(headers=_good_headers()),
            raw_body=b"\x00not-json",
        )
        assert result.status_code == 400
        assert result.outcome == TriggerAuditOutcome.EXECUTION_FAILURE
        audits = _audits(db_session)
        assert [a.outcome for a in audits] == ["execution_failure"]
        assert audits[0].detail["stage"] == "parse"

    async def test_accepted_event_creates_run_and_audit(
        self, db_session, stub_provider
    ):
        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        trigger = _create_stub_trigger(db_session, user, agent)

        result = await process_trigger_callback(
            db_session,
            context=_context(headers=_good_headers()),
            raw_body=_event_body("evt-42"),
        )
        assert result.status_code == 200
        assert result.outcome == TriggerAuditOutcome.ACCEPTED
        assert len(result.runs) == 1
        run = db_session.query(TriggerRun).one()
        assert run.trigger_id == trigger.id
        assert run.source_event_id == "evt-42"
        assert run.task_id is not None
        audits = _audits(db_session)
        assert [a.outcome for a in audits] == ["accepted"]
        assert audits[0].detail["run_ids"] == [run.id]

    async def test_redelivery_is_idempotent(self, db_session, stub_provider):
        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        _create_stub_trigger(db_session, user, agent)

        first = await process_trigger_callback(
            db_session,
            context=_context(headers=_good_headers()),
            raw_body=_event_body("evt-dup"),
        )
        second = await process_trigger_callback(
            db_session,
            context=_context(headers=_good_headers()),
            raw_body=_event_body("evt-dup"),
        )
        assert len(first.runs) == 1
        assert second.status_code == 200
        assert second.runs == []
        assert second.duplicates == 1
        assert db_session.query(TriggerRun).count() == 1

    async def test_event_type_allow_list_filters_events(
        self, db_session, stub_provider
    ):
        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        _create_stub_trigger(
            db_session, user, agent, config={"event_types": ["stub.allowed"]}
        )

        result = await process_trigger_callback(
            db_session,
            context=_context(headers=_good_headers()),
            raw_body=_event_body("evt-x", event_type="stub.other"),
        )
        assert result.status_code == 200
        assert result.runs == []
        assert result.filtered_events == 1
        assert db_session.query(TriggerRun).count() == 0

    async def test_resource_mismatch_is_rejected_and_audited(
        self, db_session, stub_provider
    ):
        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        trigger = _create_stub_trigger(db_session, user, agent, resource_id="res-1")

        raw_body = json.dumps(
            {"id": "evt-1", "type": "stub.event", "resource": "res-claimed"}
        ).encode()
        result = await process_trigger_callback(
            db_session,
            context=_context(headers=_good_headers(resource="res-other")),
            raw_body=raw_body,
        )
        assert result.status_code == 403
        assert result.outcome == TriggerAuditOutcome.REJECTED_RESOURCE
        audits = _audits(db_session)
        assert [a.outcome for a in audits] == ["rejected_resource"]
        assert audits[0].trigger_id == trigger.id
        assert audits[0].detail["reason"] == "mismatch"
        assert audits[0].detail["claimed_resource_id"] == "res-claimed"
        assert audits[0].detail["attested_resource_id"] == "res-other"
        assert audits[0].detail["trigger_resource_id"] == "res-1"
        assert db_session.query(TriggerRun).count() == 0

    async def test_permissive_provider_cannot_bypass_resource_matching(
        self, db_session
    ):
        """The pipeline owns persisted-vs-attested matching, not providers."""

        class PermissiveProvider(StubProvider):
            def authorize_resource(self, trigger, attested_resource_id, event):
                return True

        register_trigger_provider(PermissiveProvider(), replace=True)
        try:
            user = _create_user(db_session)
            agent = _create_agent(db_session, user)
            _create_stub_trigger(db_session, user, agent, resource_id="res-1")

            result = await process_trigger_callback(
                db_session,
                context=_context(headers=_good_headers(resource="res-other")),
                raw_body=_event_body(),
            )
        finally:
            unregister_trigger_provider(STUB_PROVIDER)

        assert result.status_code == 403
        assert result.outcome == TriggerAuditOutcome.REJECTED_RESOURCE
        assert db_session.query(TriggerRun).count() == 0
        assert [a.outcome for a in _audits(db_session)] == ["rejected_resource"]

    async def test_resource_matching_is_case_insensitive(
        self, db_session, stub_provider
    ):
        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        _create_stub_trigger(db_session, user, agent, resource_id="mailbox@example.com")

        result = await process_trigger_callback(
            db_session,
            context=_context(headers=_good_headers(resource="Mailbox@Example.COM")),
            raw_body=_event_body(),
        )
        assert result.status_code == 200
        assert result.outcome == TriggerAuditOutcome.ACCEPTED
        assert len(result.runs) == 1

    async def test_missing_attested_identity_is_rejected_for_bound_trigger(
        self, db_session, stub_provider
    ):
        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        _create_stub_trigger(db_session, user, agent, resource_id="res-1")

        result = await process_trigger_callback(
            db_session,
            context=_context(headers=_good_headers(resource=None)),
            raw_body=_event_body(),
        )
        assert result.status_code == 403
        assert result.outcome == TriggerAuditOutcome.REJECTED_RESOURCE
        audits = _audits(db_session)
        assert audits[-1].detail["reason"] == "no attested identity"
        assert db_session.query(TriggerRun).count() == 0

    async def test_cross_user_target_trigger_is_rejected(
        self, db_session, stub_provider
    ):
        """Events may only be re-targeted at triggers of the same user."""
        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        _create_stub_trigger(db_session, user, agent, resource_id="res-1")

        other_user = User(
            username="pipeline-other", password_hash="hash", is_admin=False
        )
        db_session.add(other_user)
        db_session.commit()
        db_session.refresh(other_user)
        other_agent = _create_agent(db_session, other_user)
        other_trigger = _create_stub_trigger(
            db_session,
            other_user,
            other_agent,
            callback_id="cb-stub-other",
            resource_id="res-1",
        )

        raw_body = json.dumps(
            {
                "id": "evt-cross",
                "type": "stub.event",
                "target_trigger_id": int(other_trigger.id),
            }
        ).encode()
        result = await process_trigger_callback(
            db_session,
            context=_context(headers=_good_headers()),
            raw_body=raw_body,
        )

        assert result.status_code == 403
        assert result.outcome == TriggerAuditOutcome.REJECTED_RESOURCE
        assert db_session.query(TriggerRun).count() == 0
        audits = _audits(db_session)
        assert audits[-1].outcome == "rejected_resource"
        assert audits[-1].detail["reason"] == "unknown_event_trigger"
        assert audits[-1].detail["target_trigger_id"] == other_trigger.id

    async def test_same_user_target_trigger_is_resolved(
        self, db_session, stub_provider
    ):
        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        _create_stub_trigger(db_session, user, agent, resource_id="res-1")
        sibling = _create_stub_trigger(
            db_session,
            user,
            agent,
            callback_id="cb-stub-sibling",
            resource_id="res-1",
        )

        raw_body = json.dumps(
            {
                "id": "evt-sibling",
                "type": "stub.event",
                "target_trigger_id": int(sibling.id),
            }
        ).encode()
        result = await process_trigger_callback(
            db_session,
            context=_context(headers=_good_headers()),
            raw_body=raw_body,
        )

        assert result.status_code == 200
        assert result.outcome == TriggerAuditOutcome.ACCEPTED
        assert len(result.runs) == 1
        assert result.runs[0].trigger_id == sibling.id

    async def test_disabled_sibling_only_callback_reports_disabled_outcome(
        self, db_session, stub_provider
    ):
        """When every event targets a disabled sibling trigger, the terminal
        result mirrors the per-event rejected_disabled audit rather than
        mislabeling the callback as a resource rejection (whose ack status
        can differ)."""
        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        _create_stub_trigger(db_session, user, agent, resource_id="res-1")
        sibling = _create_stub_trigger(
            db_session,
            user,
            agent,
            callback_id="cb-stub-disabled-sibling",
            resource_id="res-1",
            enabled=False,
        )

        raw_body = json.dumps(
            {
                "id": "evt-disabled-sibling",
                "type": "stub.event",
                "target_trigger_id": int(sibling.id),
            }
        ).encode()
        result = await process_trigger_callback(
            db_session,
            context=_context(headers=_good_headers()),
            raw_body=raw_body,
        )

        assert result.status_code == stub_provider.ack_policy.disabled_status == 409
        assert result.outcome == TriggerAuditOutcome.REJECTED_DISABLED
        assert db_session.query(TriggerRun).count() == 0
        audits = _audits(db_session)
        assert [a.outcome for a in audits] == ["rejected_disabled"]
        assert audits[0].trigger_id == sibling.id

    async def test_mixed_target_batch_keeps_per_event_outcomes(
        self, db_session, stub_provider
    ):
        """Batched trigger resolution preserves per-event behavior: valid,
        missing, and disabled targets in one callback land the same outcomes
        as they did with per-event lookups."""
        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        default_trigger = _create_stub_trigger(
            db_session, user, agent, resource_id="res-1"
        )
        sibling = _create_stub_trigger(
            db_session,
            user,
            agent,
            callback_id="cb-stub-sibling",
            resource_id="res-1",
        )
        disabled_sibling = _create_stub_trigger(
            db_session,
            user,
            agent,
            callback_id="cb-stub-disabled",
            resource_id="res-1",
            enabled=False,
        )

        raw_body = json.dumps(
            [
                {"id": "evt-default", "type": "stub.event"},
                {
                    "id": "evt-sibling",
                    "type": "stub.event",
                    "target_trigger_id": int(sibling.id),
                },
                {
                    "id": "evt-missing",
                    "type": "stub.event",
                    "target_trigger_id": 999_999,
                },
                {
                    "id": "evt-disabled",
                    "type": "stub.event",
                    "target_trigger_id": int(disabled_sibling.id),
                },
            ]
        ).encode()
        result = await process_trigger_callback(
            db_session,
            context=_context(headers=_good_headers()),
            raw_body=raw_body,
        )

        assert result.status_code == 200
        assert result.outcome == TriggerAuditOutcome.ACCEPTED
        assert sorted(int(run.trigger_id) for run in result.runs) == sorted(
            [int(default_trigger.id), int(sibling.id)]
        )
        assert result.rejected_events == 2
        audits = _audits(db_session)
        assert [a.outcome for a in audits] == [
            "rejected_resource",
            "rejected_disabled",
            "accepted",
        ]
        assert audits[0].detail["target_trigger_id"] == 999_999
        assert audits[1].trigger_id == disabled_sibling.id

    async def test_trigger_resolution_issues_one_query_per_callback(
        self, db_session, stub_provider
    ):
        """Target-trigger resolution is batched into a single IN() query per
        callback regardless of event count."""
        from sqlalchemy import event as sa_event

        from xagent.web.models.database import get_engine

        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        _create_stub_trigger(db_session, user, agent, resource_id="res-1")
        sibling_a = _create_stub_trigger(
            db_session, user, agent, callback_id="cb-stub-a", resource_id="res-1"
        )
        sibling_b = _create_stub_trigger(
            db_session, user, agent, callback_id="cb-stub-b", resource_id="res-1"
        )

        raw_body = json.dumps(
            [
                {"id": "evt-1", "type": "stub.event"},
                {
                    "id": "evt-2",
                    "type": "stub.event",
                    "target_trigger_id": int(sibling_a.id),
                },
                {
                    "id": "evt-3",
                    "type": "stub.event",
                    "target_trigger_id": int(sibling_a.id),
                },
                {
                    "id": "evt-4",
                    "type": "stub.event",
                    "target_trigger_id": int(sibling_b.id),
                },
            ]
        ).encode()

        statements: list[str] = []

        def _track(conn, cursor, statement, parameters, context, executemany):
            statements.append(statement)

        engine = get_engine()
        sa_event.listen(engine, "before_cursor_execute", _track)
        try:
            result = await process_trigger_callback(
                db_session,
                context=_context(headers=_good_headers()),
                raw_body=raw_body,
            )
        finally:
            sa_event.remove(engine, "before_cursor_execute", _track)

        assert result.outcome == TriggerAuditOutcome.ACCEPTED
        assert len(result.runs) == 4

        # Per-event resolution filtered on agent_triggers.user_id; the batch
        # query is the only statement with that predicate and must use IN().
        resolution_queries = [
            s
            for s in statements
            if "agent_triggers.user_id =" in s or "agent_triggers.id IN" in s
        ]
        assert len(resolution_queries) == 1
        assert "agent_triggers.id IN" in resolution_queries[0]

    async def test_audit_rows_survive_trigger_deletion(self, db_session, stub_provider):
        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        trigger = _create_stub_trigger(db_session, user, agent)

        await process_trigger_callback(
            db_session,
            context=_context(headers=_good_headers()),
            raw_body=_event_body("evt-audit"),
        )
        assert len(_audits(db_session)) == 1

        delete_agent_trigger(
            db_session,
            user_id=int(user.id),
            agent_id=int(agent.id),
            trigger_id=int(trigger.id),
        )
        assert db_session.query(AgentTrigger).count() == 0
        audits = _audits(db_session)
        assert len(audits) == 1
        assert audits[0].trigger_id is None
        assert audits[0].outcome == "accepted"


class TestProviderRegistration:
    async def test_register_and_unregister_receive_trigger_context(
        self, db_session, stub_provider
    ):
        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        trigger = _create_stub_trigger(db_session, user, agent)

        config = stub_provider.validate_config({"event_types": ["stub.event"]})
        registration = await stub_provider.register(db_session, trigger, config)
        assert registration.status == TriggerProvisioningStatus.ACTIVE
        await stub_provider.unregister(db_session, trigger, config)
        assert stub_provider.register_calls == [trigger.id]
        assert stub_provider.unregister_calls == [trigger.id]


class TestPartialFailureAcknowledgement:
    async def test_mixed_success_and_failure_returns_failure_status(
        self, db_session, stub_provider, monkeypatch
    ):
        """A partially failed batch must not be acked as accepted: the
        provider failure status keeps redelivery semantics, and idempotency
        keys protect the runs that already succeeded."""
        from types import SimpleNamespace

        from xagent.web.services.trigger_providers import pipeline as pipeline_module

        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        _create_stub_trigger(db_session, user, agent)

        fired: list[str] = []

        async def flaky_fire(db, *, trigger, event):
            fired.append(event.source_event_id)
            if event.source_event_id == "evt-bad":
                raise RuntimeError("transient enqueue failure")
            return SimpleNamespace(id=101), True

        monkeypatch.setattr(pipeline_module, "_fire_event", flaky_fire)

        body = json.dumps(
            [
                {"id": "evt-good", "type": "stub.event"},
                {"id": "evt-bad", "type": "stub.event"},
            ]
        ).encode()
        result = await process_trigger_callback(
            db_session,
            context=_context(headers=_good_headers()),
            raw_body=body,
        )

        assert fired == ["evt-good", "evt-bad"]
        assert result.status_code == 500
        assert result.outcome == TriggerAuditOutcome.EXECUTION_FAILURE
        assert len(result.runs) == 1
        failure_audits = [
            a for a in _audits(db_session) if a.outcome == "execution_failure"
        ]
        assert len(failure_audits) == 1
        assert failure_audits[0].detail["stage"] == "fire"
        # No accepted audit row: the delivery as a whole was not acknowledged.
        assert all(a.outcome != "accepted" for a in _audits(db_session))

    async def test_task_attach_failure_blocks_ack_and_allows_redelivery(
        self, db_session, stub_provider, monkeypatch
    ):
        """A run created without a task is a preparation failure, not a
        success: the callback must return the failure status (so the source
        redelivers) and finalize must not run (so provider cursors do not
        advance past the lost event). Redelivery then repairs the run via
        its idempotency key."""
        import xagent.web.services.triggers as triggers_module

        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        _create_stub_trigger(db_session, user, agent)

        finalized: list[bytes] = []

        async def tracking_finalize(*, db, context, trigger, events, raw_body):
            finalized.append(raw_body)

        monkeypatch.setattr(stub_provider, "finalize_callback", tracking_finalize)

        original_attach = triggers_module._attach_task_to_trigger_run

        def broken_attach(*args, **kwargs):
            raise RuntimeError("task table unavailable")

        monkeypatch.setattr(
            triggers_module, "_attach_task_to_trigger_run", broken_attach
        )

        result = await process_trigger_callback(
            db_session,
            context=_context(headers=_good_headers()),
            raw_body=_event_body("evt-lost"),
        )

        assert result.status_code == 500
        assert result.outcome == TriggerAuditOutcome.EXECUTION_FAILURE
        assert result.runs == []
        assert finalized == []
        run = db_session.query(TriggerRun).one()
        assert run.status == "failed"
        assert run.task_id is None
        failure_audits = [
            a for a in _audits(db_session) if a.outcome == "execution_failure"
        ]
        assert len(failure_audits) == 1
        assert failure_audits[0].detail["stage"] == "fire"

        # Redelivery of the same event repairs the run once attachment works.
        monkeypatch.setattr(
            triggers_module, "_attach_task_to_trigger_run", original_attach
        )
        retry = await process_trigger_callback(
            db_session,
            context=_context(headers=_good_headers()),
            raw_body=_event_body("evt-lost"),
        )
        assert retry.status_code == 200
        assert retry.outcome == TriggerAuditOutcome.ACCEPTED
        assert retry.duplicates == 1
        assert finalized  # the provider may now advance its cursor
        repaired = db_session.query(TriggerRun).one()
        assert repaired.task_id is not None
        assert repaired.status == "pending"
