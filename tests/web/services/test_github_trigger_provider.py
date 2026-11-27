"""GitHub trigger provider: hook lifecycle, signed delivery, dedup, audit."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from xagent.web.api.triggers import _read_callback_body
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
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services.trigger_providers import (
    CallbackRequestContext,
    GithubProvider,
    process_trigger_callback,
    sign_github_payload,
)
from xagent.web.services.trigger_providers.github import MAX_BODY_BYTES
from xagent.web.services.triggers import (
    TriggerServiceError,
    create_agent_trigger,
    delete_agent_trigger,
    update_agent_trigger,
)

BASE_URL = "https://api.example.test"


class FakeGithub:
    """Records hook API calls; behaviour is scripted per test."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.fail_status: int | None = None
        self.hooks: dict[int, dict[str, Any]] = {}
        self._next_id = 1000
        self.RequestException = Exception

    def _response(self, status: int, body: dict[str, Any] | None = None):
        return SimpleNamespace(status_code=status, json=lambda: body or {})

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(("POST", url, json or {}))
        assert headers["Authorization"] == "Bearer gho_test"
        if self.fail_status:
            return self._response(self.fail_status, {"message": "nope"})
        self._next_id += 1
        self.hooks[self._next_id] = json
        return self._response(201, {"id": self._next_id})

    def patch(self, url, headers=None, json=None, timeout=None):
        self.calls.append(("PATCH", url, json or {}))
        hook_id = int(url.rsplit("/", 1)[1])
        if hook_id not in self.hooks:
            return self._response(404)
        self.hooks[hook_id] = json
        return self._response(200, {"id": hook_id})

    def delete(self, url, headers=None, timeout=None):
        self.calls.append(("DELETE", url, {}))
        self.hooks.pop(int(url.rsplit("/", 1)[1]), None)
        return self._response(204)

    def methods(self) -> list[str]:
        return [call[0] for call in self.calls]


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'github_trigger.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


@pytest.fixture()
def github():
    fake = FakeGithub()
    with (
        patch("xagent.web.services.github_provisioning.requests", fake),
        patch(
            "xagent.web.services.trigger_providers.github.get_trigger_callback_base_url",
            return_value=BASE_URL,
        ),
    ):
        yield fake


@pytest.fixture(autouse=True)
def _fast_run_start():
    async def _noop_start(db, *, run, wait_for_completion=False):
        return True

    with patch(
        "xagent.web.services.triggers.start_prepared_trigger_run", new=_noop_start
    ):
        yield


@pytest.fixture()
def owner(db_session):
    user = User(username="gh-owner", password_hash="hash", is_admin=False)
    db_session.add(user)
    db_session.commit()
    agent = Agent(
        user_id=user.id,
        name="GH Agent",
        description="gh",
        instructions="gh",
        execution_mode="balanced",
        models={"general": "test-model"},
        knowledge_bases=[],
        skills=[],
        tool_categories=[],
        suggested_prompts=[],
        status=AgentStatus.PUBLISHED,
    )
    db_session.add(agent)
    db_session.commit()
    return user, agent


def _connect_github(db, user, *, token_type: str | None = None):
    account = UserOAuth(
        user_id=user.id,
        provider="github",
        access_token="gho_test",
        provider_user_id="42",
        token_type=token_type,
    )
    db.add(account)
    db.commit()
    return account


def _create(db, user, agent, **config):
    return create_agent_trigger(
        db,
        user_id=user.id,
        agent_id=agent.id,
        trigger_type="github",
        config={"repository": "Acme/Widgets", **config},
    )[0]


def _secret(trigger: AgentTrigger) -> str:
    from xagent.core.utils.encryption import decrypt_value

    return decrypt_value(str(trigger.secret_encrypted))


def _delivery(
    trigger: AgentTrigger,
    event: str,
    payload: dict[str, Any],
    *,
    delivery: str = "d-1",
    secret: str | None = None,
    hook_id: str | None = None,
    signature: str | None = None,
) -> tuple[CallbackRequestContext, bytes]:
    body = json.dumps(payload).encode()
    _, _, bound = str(trigger.resource_id).partition("#")
    headers = {
        "x-github-event": event,
        "x-github-delivery": delivery,
        "x-github-hook-id": hook_id or bound,
        "x-hub-signature-256": signature
        or sign_github_payload(secret or _secret(trigger), body),
    }
    context = CallbackRequestContext(
        provider="github",
        callback_id=str(trigger.callback_id),
        headers=headers,
        remote_ip="203.0.113.7",
    )
    return context, body


def _pr_payload(action: str = "opened") -> dict[str, Any]:
    return {
        "action": action,
        "repository": {"full_name": "Acme/Widgets", "private": True},
        "sender": {"login": "octo"},
        "pull_request": {
            "number": 7,
            "title": "Add widgets",
            "html_url": "https://github.com/Acme/Widgets/pull/7",
            "user": {"login": "octo"},
            "base": {"ref": "main"},
            "head": {"ref": "feat/w"},
        },
    }


def _run(coro):
    return asyncio.run(coro)


def _audits(db) -> list[TriggerAudit]:
    return db.query(TriggerAudit).order_by(TriggerAudit.id).all()


def test_create_registers_hook_with_owners_account(db_session, owner, github):
    user, agent = owner
    _connect_github(db_session, user)
    trigger = _create(db_session, user, agent)

    assert github.methods() == ["POST"]
    _, url, body = github.calls[0]
    assert url == "https://api.github.com/repos/Acme/Widgets/hooks"
    assert body["config"]["url"] == (
        f"{BASE_URL}/api/triggers/callback/github/{trigger.callback_id}"
    )
    assert body["config"]["secret"] == _secret(trigger)
    assert set(body["events"]) == {
        "pull_request",
        "issues",
        "issue_comment",
        "pull_request_review",
        "push",
        "create",
        "delete",
    }
    assert trigger.provisioning_status == TriggerProvisioningStatus.ACTIVE.value
    assert trigger.resource_id == "acme/widgets#1001"


def test_signed_event_fires_exactly_once_and_dedups_delivery(db_session, owner, github):
    user, agent = owner
    _connect_github(db_session, user)
    trigger = _create(db_session, user, agent)

    context, body = _delivery(trigger, "pull_request", _pr_payload())
    first = _run(process_trigger_callback(db_session, context=context, raw_body=body))
    again = _run(process_trigger_callback(db_session, context=context, raw_body=body))

    assert first.status_code == 200
    assert len(first.runs) == 1
    assert again.status_code == 200
    assert again.runs == [] and again.duplicates == 1
    assert db_session.query(TriggerRun).count() == 1
    run = db_session.query(TriggerRun).one()
    assert run.source_event_id == "d-1"


def test_normalized_pull_request_event_shape(db_session, owner, github):
    user, agent = owner
    _connect_github(db_session, user)
    trigger = _create(db_session, user, agent)
    context, body = _delivery(trigger, "pull_request", _pr_payload())
    events = _run(
        GithubProvider().parse_events(
            context, db=db_session, trigger=trigger, raw_body=body
        )
    )
    assert len(events) == 1
    assert events[0].event_type == "pull_request.opened"
    assert events[0].source_event_id == "d-1"
    assert events[0].payload["pull_request"]["number"] == 7
    assert events[0].payload["repository"]["full_name"] == "Acme/Widgets"


@pytest.mark.parametrize(
    ("event", "payload", "expected"),
    [
        ("issues", {"action": "opened", "issue": {"number": 1}}, "issues.opened"),
        (
            "issue_comment",
            {"action": "created", "issue": {"number": 1}, "comment": {"id": 2}},
            "issue_comment.created",
        ),
        (
            "pull_request_review",
            {"action": "submitted", "review": {"id": 3}, "pull_request": {}},
            "pull_request_review.submitted",
        ),
        ("push", {"ref": "refs/heads/main", "commits": [{"id": "a"}]}, "push"),
        ("create", {"ref": "v1", "ref_type": "tag"}, "create"),
        ("delete", {"ref": "old", "ref_type": "branch"}, "delete"),
    ],
)
def test_supported_events_normalize(
    db_session, owner, github, event, payload, expected
):
    user, agent = owner
    _connect_github(db_session, user)
    trigger = _create(db_session, user, agent)
    payload = {"repository": {"full_name": "Acme/Widgets"}, **payload}
    context, body = _delivery(trigger, event, payload)
    events = _run(
        GithubProvider().parse_events(
            context, db=db_session, trigger=trigger, raw_body=body
        )
    )
    assert [e.event_type for e in events] == [expected]


def test_bad_signature_is_rejected_and_audited(db_session, owner, github):
    user, agent = owner
    _connect_github(db_session, user)
    trigger = _create(db_session, user, agent)

    context, body = _delivery(trigger, "pull_request", _pr_payload(), secret="wrong")
    result = _run(process_trigger_callback(db_session, context=context, raw_body=body))

    assert result.status_code == 401
    assert db_session.query(TriggerRun).count() == 0
    assert _audits(db_session)[-1].outcome == (
        TriggerAuditOutcome.REJECTED_SIGNATURE.value
    )


def test_ping_is_acknowledged_only_when_signed(db_session, owner, github):
    user, agent = owner
    _connect_github(db_session, user)
    trigger = _create(db_session, user, agent)

    good, body = _delivery(trigger, "ping", {"zen": "hi", "hook_id": 1001})
    ok = _run(process_trigger_callback(db_session, context=good, raw_body=body))
    bad, body = _delivery(trigger, "ping", {"zen": "hi"}, secret="wrong")
    rejected = _run(process_trigger_callback(db_session, context=bad, raw_body=body))

    assert ok.status_code == 200 and ok.runs == []
    assert rejected.status_code == 401


def test_delivery_from_a_different_hook_is_rejected(db_session, owner, github):
    user, agent = owner
    _connect_github(db_session, user)
    trigger = _create(db_session, user, agent)

    context, body = _delivery(trigger, "pull_request", _pr_payload(), hook_id="999")
    result = _run(process_trigger_callback(db_session, context=context, raw_body=body))

    assert result.status_code == 403
    assert db_session.query(TriggerRun).count() == 0
    assert _audits(db_session)[-1].outcome == (
        TriggerAuditOutcome.REJECTED_RESOURCE.value
    )


def test_event_types_scope_the_hook_and_filter_deliveries(db_session, owner, github):
    user, agent = owner
    _connect_github(db_session, user)
    trigger = _create(db_session, user, agent, event_types=["pull_request.opened"])

    assert github.calls[0][2]["events"] == ["pull_request"]
    wanted, body = _delivery(trigger, "pull_request", _pr_payload("opened"))
    other, body2 = _delivery(
        trigger, "pull_request", _pr_payload("closed"), delivery="d-2"
    )
    fired = _run(process_trigger_callback(db_session, context=wanted, raw_body=body))
    skipped = _run(process_trigger_callback(db_session, context=other, raw_body=body2))

    assert len(fired.runs) == 1
    assert skipped.runs == [] and skipped.filtered_events == 1


def test_delete_trigger_removes_the_hook(db_session, owner, github):
    user, agent = owner
    _connect_github(db_session, user)
    trigger = _create(db_session, user, agent)

    delete_agent_trigger(
        db_session, user_id=user.id, agent_id=agent.id, trigger_id=trigger.id
    )

    assert github.calls[-1][:2] == (
        "DELETE",
        "https://api.github.com/repos/acme/widgets/hooks/1001",
    )
    assert github.hooks == {}


def test_unchanged_update_reuses_hook_and_rotation_pushes_new_secret(
    db_session, owner, github
):
    user, agent = owner
    _connect_github(db_session, user)
    trigger = _create(db_session, user, agent)
    old_secret = _secret(trigger)

    update_agent_trigger(
        db_session,
        user_id=user.id,
        agent_id=agent.id,
        trigger_id=trigger.id,
        updates={"name": "renamed", "config": {"repository": "Acme/Widgets"}},
    )
    assert github.methods() == ["POST", "PATCH"]
    assert trigger.resource_id == "acme/widgets#1001"

    _, plain = update_agent_trigger(
        db_session,
        user_id=user.id,
        agent_id=agent.id,
        trigger_id=trigger.id,
        updates={"rotate_secret": True},
    )
    assert plain is None
    assert _secret(trigger) != old_secret
    assert github.calls[-1][2]["config"]["secret"] == _secret(trigger)


def test_changing_repository_moves_the_hook(db_session, owner, github):
    user, agent = owner
    _connect_github(db_session, user)
    trigger = _create(db_session, user, agent)

    update_agent_trigger(
        db_session,
        user_id=user.id,
        agent_id=agent.id,
        trigger_id=trigger.id,
        updates={"config": {"repository": "Acme/Gadgets"}},
    )

    assert github.methods() == ["POST", "DELETE", "POST"]
    assert github.calls[1][1].endswith("/repos/acme/widgets/hooks/1001")
    assert trigger.resource_id == "acme/gadgets#1002"


def test_permission_failure_is_a_provisioning_error(db_session, owner, github):
    user, agent = owner
    _connect_github(db_session, user)
    github.fail_status = 404
    trigger = _create(db_session, user, agent)

    assert trigger.provisioning_status == TriggerProvisioningStatus.FAILED.value
    assert "admin access" in str(trigger.provisioning_error)
    assert trigger.resource_id == "acme/widgets"


def test_missing_github_connection_is_a_provisioning_error(db_session, owner, github):
    user, agent = owner
    trigger = _create(db_session, user, agent)

    assert trigger.provisioning_status == TriggerProvisioningStatus.FAILED.value
    assert "connect GitHub" in str(trigger.provisioning_error)
    assert github.calls == []


def test_composio_grants_cannot_manage_hooks(db_session, owner, github):
    user, agent = owner
    _connect_github(db_session, user, token_type="composio")
    trigger = _create(db_session, user, agent)

    assert trigger.provisioning_status == TriggerProvisioningStatus.FAILED.value
    assert "Composio" in str(trigger.provisioning_error)


@pytest.mark.parametrize(
    "config",
    [
        {"repository": "not-a-repo"},
        {"repository": "a/b/c"},
        {"repository": "../x"},
        {"repository": "a/b", "event_types": ["pull_request"]},
        {"repository": "a/b", "event_types": ["star.created"]},
        {"repository": "a/b", "event_types": ["push.created"]},
    ],
)
def test_invalid_config_is_rejected(db_session, owner, github, config):
    user, agent = owner
    with pytest.raises(TriggerServiceError):
        create_agent_trigger(
            db_session,
            user_id=user.id,
            agent_id=agent.id,
            trigger_type="github",
            config=config,
        )


def test_caller_cannot_supply_the_hook_secret(db_session, owner, github):
    user, agent = owner
    with pytest.raises(TriggerServiceError):
        create_agent_trigger(
            db_session,
            user_id=user.id,
            agent_id=agent.id,
            trigger_type="github",
            config={"repository": "a/b"},
            secret="mine",
        )


class _FakeRequest:
    def __init__(self, chunks: list[bytes], content_length: str | None = None):
        self.headers = {"content-length": content_length} if content_length else {}
        self._chunks = chunks

    async def stream(self):
        for chunk in self._chunks:
            yield chunk

    async def body(self):  # pragma: no cover - must not be used for github
        raise AssertionError("unbounded read")


def test_body_cap_matches_githubs_25mb_limit(github):
    assert MAX_BODY_BYTES == 25 * 1024 * 1024
    assert _run(_read_callback_body(_FakeRequest([b"ab", b"cd"]), "github")) == b"abcd"
    with pytest.raises(HTTPException) as declared:
        _run(
            _read_callback_body(
                _FakeRequest([], content_length=str(MAX_BODY_BYTES + 1)), "github"
            )
        )
    assert declared.value.status_code == 413
    with pytest.raises(HTTPException) as streamed:
        _run(_read_callback_body(_FakeRequest([b"x" * MAX_BODY_BYTES, b"x"]), "github"))
    assert streamed.value.status_code == 413
