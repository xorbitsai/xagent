"""Tests for the HMAC-verified end-user identity in the widget auth flow.

Covers /api/widget/auth's end_user_id/end_user_signature verification and
the /api/widget/tasks/latest resume lookup added alongside it. The most
important case here is isolation: a guest who merely knows or guesses
another guest's end_user_id must not be able to forge a valid signature or
read that guest's conversation history.
"""

import hashlib
import hmac
import secrets

import pytest

from xagent.web.models.agent import Agent, AgentStatus

from .conftest import _admin_headers, _direct_db_session, client

pytestmark = pytest.mark.usefixtures("_test_db")


def _create_widget_agent(user_id: int, name: str = "Widget Agent") -> int:
    db = _direct_db_session()
    try:
        agent = Agent(
            user_id=user_id,
            name=name,
            description=f"{name} description",
            instructions=f"{name} instructions",
            execution_mode="balanced",
            status=AgentStatus.PUBLISHED,
            widget_enabled=True,
            allowed_domains=["*"],
            widget_key=f"wk-{secrets.token_urlsafe(24)}",
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        return int(agent.id)
    finally:
        db.close()


def _widget_key_for(agent_id: int) -> str:
    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        assert agent is not None and agent.widget_key
        return str(agent.widget_key)
    finally:
        db.close()


def _get_end_user_secret(headers: dict[str, str], agent_id: int) -> str:
    resp = client.get(f"/api/agents/{agent_id}/widget-end-user-secret", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()["widget_end_user_secret"]


def _sign(secret: str, end_user_id: str) -> str:
    return hmac.new(
        secret.encode("utf-8"), end_user_id.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _widget_auth(*, agent_id: int, **payload: object) -> dict:
    return client.post(
        "/api/widget/auth",
        json={"widget_key": _widget_key_for(agent_id), **payload},
    )


def _widget_headers(*, agent_id: int, **payload: object) -> dict[str, str]:
    resp = _widget_auth(agent_id=agent_id, **payload)
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _create_widget_task(headers: dict[str, str], title: str = "hello") -> int:
    resp = client.post(
        "/api/widget/chat/task/create",
        json={"title": title, "description": title},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    return int(resp.json()["task_id"])


def _latest_task_id(headers: dict[str, str]) -> int | None:
    resp = client.get("/api/widget/tasks/latest", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()["task_id"]


def test_valid_signature_resumes_the_signed_in_guests_own_task() -> None:
    """The core happy path: a signed end_user_id can find its own history."""
    owner_headers = _admin_headers()
    owner_id = _user_id_from(owner_headers)
    agent_id = _create_widget_agent(owner_id)
    secret = _get_end_user_secret(owner_headers, agent_id)

    device_a = _widget_headers(
        agent_id=agent_id,
        end_user_id="tenant_42:user_007",
        end_user_signature=_sign(secret, "tenant_42:user_007"),
    )
    assert _latest_task_id(device_a) is None

    task_id = _create_widget_task(device_a)

    # A second "device" authenticating with the same signed identity resumes
    # the same task -- this is the cross-device continuity the feature exists
    # to provide.
    device_b = _widget_headers(
        agent_id=agent_id,
        end_user_id="tenant_42:user_007",
        end_user_signature=_sign(secret, "tenant_42:user_007"),
    )
    assert _latest_task_id(device_b) == task_id


def test_guest_cannot_retrieve_a_different_guests_task_via_tasks_latest() -> None:
    """The isolation property /tasks/latest depends on: two different signed
    identities on the same agent never see each other's history."""
    owner_headers = _admin_headers()
    owner_id = _user_id_from(owner_headers)
    agent_id = _create_widget_agent(owner_id)
    secret = _get_end_user_secret(owner_headers, agent_id)

    victim = _widget_headers(
        agent_id=agent_id,
        end_user_id="victim@example.com",
        end_user_signature=_sign(secret, "victim@example.com"),
    )
    _create_widget_task(victim, title="victim's private conversation")

    attacker = _widget_headers(
        agent_id=agent_id,
        end_user_id="attacker@example.com",
        end_user_signature=_sign(secret, "attacker@example.com"),
    )
    assert _latest_task_id(attacker) is None


def test_guessed_end_user_id_without_the_matching_signature_is_rejected() -> None:
    """An attacker who knows/guesses the victim's end_user_id (emails and
    customer ids are often not secret) still cannot authenticate as them
    without the signature only the embedding site's own server can produce.
    """
    owner_headers = _admin_headers()
    owner_id = _user_id_from(owner_headers)
    agent_id = _create_widget_agent(owner_id)
    secret = _get_end_user_secret(owner_headers, agent_id)

    victim_id = "victim@example.com"
    victim = _widget_headers(
        agent_id=agent_id,
        end_user_id=victim_id,
        end_user_signature=_sign(secret, victim_id),
    )
    task_id = _create_widget_task(victim, title="victim's private conversation")

    # No signature at all.
    resp = _widget_auth(agent_id=agent_id, end_user_id=victim_id)
    assert resp.status_code == 403, resp.text

    # A forged/garbage signature.
    resp = _widget_auth(
        agent_id=agent_id, end_user_id=victim_id, end_user_signature="0" * 64
    )
    assert resp.status_code == 403, resp.text

    # A signature that is valid, but for a different end_user_id (replay
    # across identities): must not authenticate as the victim.
    resp = _widget_auth(
        agent_id=agent_id,
        end_user_id=victim_id,
        end_user_signature=_sign(secret, "attacker@example.com"),
    )
    assert resp.status_code == 403, resp.text

    # Sending the victim's raw id as the *unverified* guest_id field (the
    # anonymous-flow field, not end_user_id) must not resolve to the victim's
    # signed task either -- otherwise signing would be pointless, since an
    # attacker could just skip end_user_id entirely and pass the same string
    # as guest_id to reach the same stored identity.
    unverified = _widget_headers(agent_id=agent_id, guest_id=victim_id)
    assert _latest_task_id(unverified) is None

    # Confirm the victim's task really is retrievable via the real signed
    # flow, so the negative assertions above are meaningful and not just
    # "nothing exists yet".
    resp = _widget_auth(
        agent_id=agent_id,
        end_user_id=victim_id,
        end_user_signature=_sign(secret, victim_id),
    )
    assert resp.status_code == 200, resp.text
    real_headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}
    assert _latest_task_id(real_headers) == task_id


def test_auth_requires_either_guest_id_or_end_user_id() -> None:
    owner_headers = _admin_headers()
    owner_id = _user_id_from(owner_headers)
    agent_id = _create_widget_agent(owner_id)

    resp = client.post(
        "/api/widget/auth", json={"widget_key": _widget_key_for(agent_id)}
    )
    assert resp.status_code == 422, resp.text


def test_raw_guest_id_cannot_forge_into_the_verified_namespace() -> None:
    """A client cannot bypass signing by directly setting guest_id to the
    reserved "verified_end_user:" prefix the backend uses internally to mark
    a signature-verified identity."""
    owner_headers = _admin_headers()
    owner_id = _user_id_from(owner_headers)
    agent_id = _create_widget_agent(owner_id)

    resp = _widget_auth(agent_id=agent_id, guest_id="verified_end_user:someone")
    assert resp.status_code == 403, resp.text


def test_end_user_id_without_a_configured_secret_is_rejected() -> None:
    """An agent that never had its end-user secret generated must reject
    signed-identity attempts rather than silently accepting them."""
    owner_headers = _admin_headers()
    owner_id = _user_id_from(owner_headers)
    agent_id = _create_widget_agent(owner_id)
    # Note: no call to _get_end_user_secret, so widget_end_user_secret is
    # still NULL on this agent.

    resp = _widget_auth(
        agent_id=agent_id, end_user_id="user-1", end_user_signature="a" * 64
    )
    assert resp.status_code == 403, resp.text


def test_rotating_the_end_user_secret_invalidates_old_signatures() -> None:
    owner_headers = _admin_headers()
    owner_id = _user_id_from(owner_headers)
    agent_id = _create_widget_agent(owner_id)
    old_secret = _get_end_user_secret(owner_headers, agent_id)
    old_signature = _sign(old_secret, "user-1")

    rotate_resp = client.post(
        f"/api/agents/{agent_id}/widget-end-user-secret/rotate", headers=owner_headers
    )
    assert rotate_resp.status_code == 200, rotate_resp.text
    assert rotate_resp.json()["widget_end_user_secret"] != old_secret

    resp = _widget_auth(
        agent_id=agent_id, end_user_id="user-1", end_user_signature=old_signature
    )
    assert resp.status_code == 403, resp.text


def _user_id_from(headers: dict[str, str]) -> int:
    from xagent.web.models.user import User

    db = _direct_db_session()
    try:
        # The admin bootstrap fixture always creates a single "admin" user;
        # resolving it this way (rather than parsing the JWT) keeps this
        # helper independent of the token's internal shape.
        user = db.query(User).filter(User.username == "admin").first()
        assert user is not None
        return int(user.id)
    finally:
        db.close()
