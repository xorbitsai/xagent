"""Integration tests for agent management endpoints."""

import asyncio
import base64
import io
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from jose import jwt as jose_jwt
from sqlalchemy import event
from sqlalchemy.exc import IntegrityError

from xagent.config import get_uploads_dir
from xagent.web.api import agents as agents_api
from xagent.web.api.public_chat_access import create_public_chat_access_token
from xagent.web.api.widget import (
    EMBED_TICKET_TYPE,
    WIDGET_CREDENTIAL_REQUIRED_DETAIL,
    WIDGET_KEY_REQUIRED_DETAIL,
)
from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY
from xagent.web.models.agent import Agent, AgentOrigin, AgentStatus
from xagent.web.models.agent_api_key import AgentApiKey
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce, WorkforceAgent, WorkforceRun
from xagent.web.services.agent_management import (
    AgentManagementService,
    AgentWorkforceConflictError,
    AgentWorkforceReference,
)
from xagent.web.services.workforce_access import WorkforcePolicy, set_workforce_policy
from xagent.web.services.workforce_lifecycle import discard_draft_workforce

from .conftest import (
    _admin_headers,
    _direct_db_session,
    _register_second_user,
    client,
)

pytestmark = pytest.mark.usefixtures("_test_db")


@pytest.fixture(autouse=True)
def _reset_workforce_policy() -> None:
    set_workforce_policy(WorkforcePolicy())
    yield
    set_workforce_policy(WorkforcePolicy())


def test_delete_agent_uses_sync_route_boundary() -> None:
    delete_route = next(
        route
        for route in agents_api.router.routes
        if route.path == "/api/agents/{agent_id}" and "DELETE" in route.methods
    )

    assert not asyncio.iscoroutinefunction(delete_route.endpoint)


def _create_agent(headers: dict[str, str], name: str = "Test Agent") -> int:
    resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": name,
            "description": "test",
            "instructions": "You are a test agent.",
            "execution_mode": "balanced",
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


def _create_agent_row(
    *,
    user_id: int,
    name: str,
    status: AgentStatus = AgentStatus.DRAFT,
    origin: str = AgentOrigin.USER.value,
    widget_enabled: bool = True,
    allowed_domains: list[str] | None = None,
    share_enabled: bool = False,
    share_token: str | None = None,
    widget_key: str | None = None,
    generate_widget_key: bool = True,
) -> int:
    db = _direct_db_session()
    try:
        if widget_key is None and widget_enabled and generate_widget_key:
            widget_key = f"wk-{secrets.token_urlsafe(24)}"
        agent = Agent(
            user_id=user_id,
            name=name,
            description=f"{name} description",
            instructions=f"{name} instructions",
            execution_mode="balanced",
            origin=origin,
            status=status,
            widget_enabled=widget_enabled,
            allowed_domains=allowed_domains or [],
            share_enabled=share_enabled,
            share_token=share_token,
            widget_key=widget_key,
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


def _user_id(username: str) -> int:
    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.username == username).first()
        assert user is not None
        return int(user.id)
    finally:
        db.close()


def _authenticate_share_guest(
    share_token: str,
) -> dict[str, str]:
    response = client.post(
        "/api/share/auth",
        json={"share_token": share_token},
    )
    assert response.status_code == 200, response.text
    access_token = response.json()["access_token"]
    return {"Authorization": f"Bearer {access_token}"}


def _authenticate_widget_guest(
    *,
    agent_id: int,
    guest_id: str = "guest-1",
    origin: str = "https://example.com",
) -> dict[str, str]:
    # Authenticate via the direct-visit widget-key flow, which needs no
    # embedding origin. The origin arg is accepted for call-site compatibility
    # but no longer gates auth.
    del origin
    response = client.post(
        "/api/widget/auth",
        json={"widget_key": _widget_key_for(agent_id), "guest_id": guest_id},
    )
    assert response.status_code == 200, response.text
    access_token = response.json()["access_token"]
    return {"Authorization": f"Bearer {access_token}"}


def _create_public_task_file(
    *,
    owner_id: int,
    agent_id: int,
    filename: str = "shared-note.txt",
    content: bytes = b"hello from public task",
) -> str:
    uploads_root = get_uploads_dir() / f"user_{owner_id}"
    uploads_root.mkdir(parents=True, exist_ok=True)
    file_path = uploads_root / filename
    file_path.write_bytes(content)

    db = _direct_db_session()
    try:
        task = Task(
            user_id=owner_id,
            title="Public task",
            description="Public task",
            status=TaskStatus.PENDING,
            agent_id=agent_id,
            channel_id=None,
            channel_name="Shared Agent",
            agent_config={
                "auth_mode": "share",
                "share_agent_id": agent_id,
            },
        )
        db.add(task)
        db.commit()
        db.refresh(task)

        uploaded_file = UploadedFile(
            user_id=owner_id,
            task_id=int(task.id),
            filename=filename,
            storage_path=str(file_path),
            mime_type="text/plain",
            file_size=len(content),
        )
        db.add(uploaded_file)
        db.commit()
        db.refresh(uploaded_file)
        return str(uploaded_file.file_id)
    finally:
        db.close()


class _VisibleAgentPolicy(WorkforcePolicy):
    def __init__(self, visible_agent_ids: set[int]) -> None:
        self.visible_agent_ids = visible_agent_ids

    def get_visible_agent_ids(
        self,
        db: Any,
        user: User,
        purpose: str,
    ) -> set[int]:
        del db, user, purpose
        return self.visible_agent_ids


def test_list_agents_includes_owned_agents_and_policy_visible_agents() -> None:
    _admin_headers()
    bob_headers = _register_second_user()
    admin_id = _user_id("admin")
    bob_id = _user_id("bob")

    bob_draft_id = _create_agent_row(user_id=bob_id, name="Bob Draft")
    bob_published_id = _create_agent_row(
        user_id=bob_id,
        name="Bob Published",
        status=AgentStatus.PUBLISHED,
    )
    shared_published_id = _create_agent_row(
        user_id=admin_id,
        name="Shared Published",
        status=AgentStatus.PUBLISHED,
    )
    shared_draft_id = _create_agent_row(
        user_id=admin_id,
        name="Shared Draft",
        status=AgentStatus.DRAFT,
    )
    set_workforce_policy(_VisibleAgentPolicy({shared_published_id, shared_draft_id}))

    response = client.get("/api/agents", headers=bob_headers)
    assert response.status_code == 200, response.text
    items_by_id = {item["id"]: item for item in response.json()}

    assert {
        bob_draft_id,
        bob_published_id,
        shared_published_id,
        shared_draft_id,
    }.issubset(items_by_id)

    assert items_by_id[bob_draft_id]["access"] == "owner"
    assert items_by_id[bob_draft_id]["readonly"] is False
    assert items_by_id[bob_draft_id]["can_edit"] is True
    assert items_by_id[bob_draft_id]["can_publish"] is True
    assert items_by_id[bob_draft_id]["can_delete"] is True

    assert items_by_id[shared_published_id]["access"] == "policy"
    assert items_by_id[shared_published_id]["readonly"] is True
    assert items_by_id[shared_published_id]["can_edit"] is False
    assert items_by_id[shared_published_id]["can_publish"] is False
    assert items_by_id[shared_published_id]["can_delete"] is False
    assert items_by_id[shared_draft_id]["access"] == "policy"
    assert items_by_id[shared_draft_id]["status"] == "draft"
    assert items_by_id[shared_draft_id]["readonly"] is True


def test_agent_lists_keep_reusable_managers_and_hide_generated_managers() -> None:
    headers = _admin_headers()
    owner_id = _user_id("admin")
    reusable_manager_id = _create_agent_row(
        user_id=owner_id,
        name="Reusable Manager",
        status=AgentStatus.PUBLISHED,
    )
    generated_manager_id = _create_agent_row(
        user_id=owner_id,
        name="Generated Manager",
        status=AgentStatus.PUBLISHED,
        origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
    )
    worker_id = _create_agent_row(
        user_id=owner_id,
        name="Reusable Worker",
        status=AgentStatus.PUBLISHED,
    )

    db = _direct_db_session()
    try:
        workforce = Workforce(
            owner_user_id=owner_id,
            scope_type="user",
            scope_id=str(owner_id),
            name="Reusable Manager Workforce",
            manager_agent_id=reusable_manager_id,
            status="draft",
        )
        db.add(workforce)
        db.commit()
    finally:
        db.close()

    response = client.get("/api/agents", headers=headers)
    assert response.status_code == 200, response.text
    agent_ids = {item["id"] for item in response.json()}
    assert reusable_manager_id in agent_ids
    assert generated_manager_id not in agent_ids
    assert worker_id in agent_ids

    options_response = client.get("/api/workforces/agent-options", headers=headers)
    assert options_response.status_code == 200, options_response.text
    option_ids = {item["id"] for item in options_response.json()}
    assert reusable_manager_id in option_ids
    assert generated_manager_id not in option_ids
    assert worker_id in option_ids


def test_agent_name_conflicts_ignore_generated_workforce_managers() -> None:
    headers = _admin_headers()
    owner_id = _user_id("admin")
    generated_name = "Generated Manager Name"
    generated_manager_id = _create_agent_row(
        user_id=owner_id,
        name=generated_name,
        status=AgentStatus.PUBLISHED,
        origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
    )

    create_response = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": generated_name,
            "description": "Reusable agent",
            "instructions": "You are reusable.",
            "execution_mode": "balanced",
        },
    )

    assert create_response.status_code == 200, create_response.text
    created_agent_id = create_response.json()["id"]
    assert created_agent_id != generated_manager_id

    update_target_id = _create_agent_row(user_id=owner_id, name="Update Target")
    hidden_update_name = "Hidden Update Manager Name"
    _create_agent_row(
        user_id=owner_id,
        name=hidden_update_name,
        status=AgentStatus.PUBLISHED,
        origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
    )

    update_response = client.put(
        f"/api/agents/{update_target_id}",
        headers=headers,
        json={"name": hidden_update_name},
    )

    assert update_response.status_code == 200, update_response.text
    assert update_response.json()["name"] == hidden_update_name


def test_generated_workforce_manager_agents_cannot_authenticate_widget() -> None:
    _admin_headers()
    owner_id = _user_id("admin")
    generated_manager_id = _create_agent_row(
        user_id=owner_id,
        name="Generated Widget Manager",
        status=AgentStatus.PUBLISHED,
        origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
        widget_enabled=True,
        allowed_domains=["*"],
    )

    response = client.post(
        "/api/widget/auth",
        json={
            "widget_key": _widget_key_for(generated_manager_id),
            "guest_id": "guest-1",
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid widget key"


def test_embed_ticket_origin_matches_allowed_domains_case_insensitively() -> None:
    _admin_headers()
    owner_id = _user_id("admin")
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Case Insensitive Widget Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=True,
        allowed_domains=["Example.com"],
    )

    response = _issue_embed_ticket(agent_id, "https://EXAMPLE.com")

    assert response.status_code == 200, response.text


def _issue_embed_ticket(
    agent_id: int, origin: str, widget_key: str | None = None
) -> Any:
    return client.post(
        "/api/widget/embed-ticket",
        json={"widget_key": widget_key or _widget_key_for(agent_id)},
        headers={"origin": origin},
    )


def test_widget_embed_ticket_flow_validates_embedding_origin() -> None:
    _admin_headers()
    owner_id = _user_id("admin")
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Embed Ticket Widget Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=True,
        allowed_domains=["trusted-site.com"],
    )

    # widget.js requests the ticket from the top-level embedding page, so
    # the browser-enforced Origin header carries the real embedding site.
    ticket_response = _issue_embed_ticket(agent_id, "https://trusted-site.com")
    assert ticket_response.status_code == 200, ticket_response.text
    ticket = ticket_response.json()["ticket"]

    # The backend signs the validated embedding origin into the ticket, not
    # a caller-supplied value.
    ticket_claims = jose_jwt.decode(ticket, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    assert ticket_claims["type"] == EMBED_TICKET_TYPE
    assert ticket_claims["embed_origin"] == "trusted-site.com"

    # The auth fetch runs inside the iframe: its Origin is the xagent host,
    # which is NOT in allowed_domains — the ticket alone must carry trust.
    auth_response = client.post(
        "/api/widget/auth",
        json={"agent_id": agent_id, "guest_id": "guest-1", "embed_ticket": ticket},
        headers={"origin": "https://xagent-host.example"},
    )
    assert auth_response.status_code == 200, auth_response.text


def test_widget_embed_ticket_rejected_for_disallowed_origin() -> None:
    _admin_headers()
    owner_id = _user_id("admin")
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Embed Ticket Rejection Widget Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=True,
        allowed_domains=["trusted-site.com"],
    )

    response = _issue_embed_ticket(agent_id, "https://evil-attacker.com")

    assert response.status_code == 403
    assert response.json()["detail"] == "Domain not allowed: evil-attacker.com"


def test_embed_ticket_requires_valid_widget_key_despite_spoofed_origin() -> None:
    """#742 (embed-ticket path): a forged allowlisted Origin is worthless
    without the unguessable per-agent widget key."""
    _admin_headers()
    owner_id = _user_id("admin")
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Key Gated Embed Ticket Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=True,
        allowed_domains=["trusted-site.com"],
    )

    ok = _issue_embed_ticket(agent_id, "https://trusted-site.com")
    assert ok.status_code == 200, ok.text
    assert ok.json()["agent_id"] == agent_id

    forged = client.post(
        "/api/widget/embed-ticket",
        json={"widget_key": "wk-not-the-real-key"},
        headers={"origin": "https://trusted-site.com"},
    )
    assert forged.status_code == 403
    assert forged.json()["detail"] == "Invalid widget key"


def test_embed_ticket_rejects_numeric_agent_id_without_key() -> None:
    """The enumerable numeric agent_id can no longer obtain a ticket, and a
    key-less (legacy) request gets an actionable 403 rather than a bare 422."""
    _admin_headers()
    owner_id = _user_id("admin")
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Agent Id Only Embed Ticket Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=True,
        allowed_domains=["trusted-site.com"],
    )

    response = client.post(
        "/api/widget/embed-ticket",
        json={"agent_id": agent_id},
        headers={"origin": "https://trusted-site.com"},
    )
    assert response.status_code == 403
    assert response.json()["detail"] == WIDGET_KEY_REQUIRED_DETAIL


def test_embed_ticket_rejects_stale_key_after_rotation() -> None:
    headers = _admin_headers()
    owner_id = _user_id("admin")
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Rotated Key Embed Ticket Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=True,
        allowed_domains=["trusted-site.com"],
    )
    old_key = _widget_key_for(agent_id)

    rotate = client.post(f"/api/agents/{agent_id}/widget-key/rotate", headers=headers)
    assert rotate.status_code == 200, rotate.text
    new_key = rotate.json()["widget_key"]
    assert new_key != old_key

    stale = client.post(
        "/api/widget/embed-ticket",
        json={"widget_key": old_key},
        headers={"origin": "https://trusted-site.com"},
    )
    assert stale.status_code == 403
    assert stale.json()["detail"] == "Invalid widget key"

    fresh = _issue_embed_ticket(
        agent_id, "https://trusted-site.com", widget_key=new_key
    )
    assert fresh.status_code == 200, fresh.text


def test_embed_ticket_rejects_disabled_widget_like_an_unknown_key() -> None:
    """Disabled-widget and unknown-key both return the same 403 so callers
    cannot tell agents apart."""
    _admin_headers()
    owner_id = _user_id("admin")
    _create_agent_row(
        user_id=owner_id,
        name="Disabled Widget Embed Ticket Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=False,
        allowed_domains=["trusted-site.com"],
        widget_key="wk-disabled-agent-key",
        generate_widget_key=False,
    )

    response = client.post(
        "/api/widget/embed-ticket",
        json={"widget_key": "wk-disabled-agent-key"},
        headers={"origin": "https://trusted-site.com"},
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid widget key"


def test_widget_auth_rejects_spoofed_origin_without_credential() -> None:
    """#742 reproduction: a spoofed allowlisted Origin header plus an
    enumerable agent_id, with no ticket and no widget key, authenticates
    nothing now that the bare-header fallback is removed."""
    _admin_headers()
    owner_id = _user_id("admin")
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Spoofed Origin Widget Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=True,
        allowed_domains=["trusted-site.com"],
    )

    response = client.post(
        "/api/widget/auth",
        json={"agent_id": agent_id, "guest_id": "guest-1"},
        headers={"origin": "https://trusted-site.com"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == WIDGET_CREDENTIAL_REQUIRED_DETAIL


def test_widget_auth_ignores_client_supplied_embed_origin() -> None:
    """A client self-reporting an allowed origin in the body (the pre-ticket
    design's flaw) still gets nothing without a verifiable credential."""
    _admin_headers()
    owner_id = _user_id("admin")
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Self Reported Origin Widget Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=True,
        allowed_domains=["trusted-site.com"],
    )

    response = client.post(
        "/api/widget/auth",
        json={
            "agent_id": agent_id,
            "guest_id": "guest-1",
            "embed_origin": "https://trusted-site.com",
        },
        headers={"origin": "https://evil-attacker.com"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == WIDGET_CREDENTIAL_REQUIRED_DETAIL


def test_widget_auth_with_no_origin_and_no_credential_is_rejected() -> None:
    """The absent-Origin row from #742's table: still 403 with no credential."""
    _admin_headers()
    owner_id = _user_id("admin")
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="No Origin Widget Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=True,
        allowed_domains=["trusted-site.com"],
    )

    response = client.post(
        "/api/widget/auth",
        json={"agent_id": agent_id, "guest_id": "guest-1"},
    )

    assert response.status_code == 403


def test_widget_auth_direct_visit_with_widget_key_succeeds() -> None:
    """Direct (non-embedded) visits authenticate with the widget key alone;
    no origin allowlist applies because a direct visit is not embedded."""
    _admin_headers()
    owner_id = _user_id("admin")
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Direct Visit Widget Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=True,
        allowed_domains=["trusted-site.com"],
    )
    widget_key = _widget_key_for(agent_id)

    response = client.post(
        "/api/widget/auth",
        json={"widget_key": widget_key, "guest_id": "guest-1"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["agent_id"] == agent_id

    stale = client.post(
        "/api/widget/auth",
        json={"widget_key": "wk-not-a-real-key", "guest_id": "guest-1"},
    )
    assert stale.status_code == 403
    assert stale.json()["detail"] == "Invalid widget key"


def test_widget_auth_rejects_tampered_or_mismatched_ticket() -> None:
    _admin_headers()
    owner_id = _user_id("admin")
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Tampered Ticket Widget Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=True,
        allowed_domains=["trusted-site.com"],
    )
    other_agent_id = _create_agent_row(
        user_id=owner_id,
        name="Other Ticket Widget Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=True,
        allowed_domains=["trusted-site.com"],
    )

    # Garbage ticket: fails signature verification.
    response = client.post(
        "/api/widget/auth",
        json={
            "agent_id": agent_id,
            "guest_id": "guest-1",
            "embed_ticket": "not-a-real-ticket",
        },
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid or expired embed ticket"

    # A ticket authenticates only as the agent it was issued for; a spoofed
    # agent_id in the body cannot redirect it to another agent.
    other_ticket = _issue_embed_ticket(
        other_agent_id, "https://trusted-site.com"
    ).json()["ticket"]
    response = client.post(
        "/api/widget/auth",
        json={
            "agent_id": agent_id,
            "guest_id": "guest-1",
            "embed_ticket": other_ticket,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["agent_id"] == other_agent_id


def test_widget_auth_rejects_expired_embed_ticket() -> None:
    _admin_headers()
    owner_id = _user_id("admin")
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Expired Ticket Widget Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=True,
        allowed_domains=["trusted-site.com"],
    )

    expired_ticket = jose_jwt.encode(
        {
            "type": EMBED_TICKET_TYPE,
            "agent_id": agent_id,
            "embed_origin": "trusted-site.com",
            "exp": datetime.now(timezone.utc) - timedelta(seconds=1),
        },
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )

    response = client.post(
        "/api/widget/auth",
        json={
            "agent_id": agent_id,
            "guest_id": "guest-1",
            "embed_ticket": expired_ticket,
        },
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid or expired embed ticket"


def test_widget_auth_rejects_valid_signed_token_of_wrong_type() -> None:
    """All token classes share one signing key, so the type claim is the only
    thing separating them. A validly-signed guest access token with the right
    agent_id must not be replayable as an embed ticket."""
    _admin_headers()
    owner_id = _user_id("admin")
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Wrong Type Ticket Widget Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=True,
        allowed_domains=["trusted-site.com"],
    )

    wrong_type_ticket = jose_jwt.encode(
        {
            "type": "widget",
            "agent_id": agent_id,
            "embed_origin": "trusted-site.com",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        },
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )

    response = client.post(
        "/api/widget/auth",
        json={
            "agent_id": agent_id,
            "guest_id": "guest-1",
            "embed_ticket": wrong_type_ticket,
        },
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid or expired embed ticket"


def test_widget_auth_rechecks_ticket_origin_against_current_allowlist() -> None:
    _admin_headers()
    owner_id = _user_id("admin")
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Shrinking Allowlist Widget Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=True,
        allowed_domains=["trusted-site.com"],
    )

    ticket = _issue_embed_ticket(agent_id, "https://trusted-site.com").json()["ticket"]

    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        assert agent is not None
        agent.allowed_domains = ["another-site.com"]
        db.commit()
    finally:
        db.close()

    response = client.post(
        "/api/widget/auth",
        json={"agent_id": agent_id, "guest_id": "guest-1", "embed_ticket": ticket},
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Domain not allowed: trusted-site.com"


def test_widget_embed_ticket_matches_domain_regardless_of_scheme() -> None:
    """allowed_domains matches on host[:port] only; the scheme is not part
    of the comparison, so an http origin matches an entry with no scheme."""
    _admin_headers()
    owner_id = _user_id("admin")
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Scheme Agnostic Widget Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=True,
        allowed_domains=["trusted-site.com"],
    )

    response = _issue_embed_ticket(agent_id, "http://trusted-site.com")

    assert response.status_code == 200, response.text


def test_widget_embed_ticket_wildcard_allows_any_origin() -> None:
    _admin_headers()
    owner_id = _user_id("admin")
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Wildcard Widget Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=True,
        allowed_domains=["*"],
    )

    response = _issue_embed_ticket(agent_id, "https://anything-goes.example")

    assert response.status_code == 200, response.text


def test_widget_embed_ticket_matches_subdomain_of_allowed_entry() -> None:
    _admin_headers()
    owner_id = _user_id("admin")
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Subdomain Widget Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=True,
        allowed_domains=["trusted-site.com"],
    )

    allowed = _issue_embed_ticket(agent_id, "https://app.trusted-site.com")
    assert allowed.status_code == 200, allowed.text

    # A domain that merely ends with the allowed string but is not a subdomain
    # (no dot boundary) must not match.
    spoofed = _issue_embed_ticket(agent_id, "https://eviltrusted-site.com")
    assert spoofed.status_code == 403
    assert spoofed.json()["detail"] == "Domain not allowed: eviltrusted-site.com"


def test_embed_ticket_endpoint_rejects_unknown_key() -> None:
    _admin_headers()
    response = client.post(
        "/api/widget/embed-ticket",
        json={"widget_key": "wk-does-not-exist"},
        headers={"origin": "https://trusted-site.com"},
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid widget key"


def test_embed_ticket_endpoint_rejects_generated_manager_agent() -> None:
    """A workforce-manager agent's key is treated the same as an unknown key."""
    _admin_headers()
    owner_id = _user_id("admin")
    manager_id = _create_agent_row(
        user_id=owner_id,
        name="Generated Manager Ticket Agent",
        status=AgentStatus.PUBLISHED,
        origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
        widget_enabled=True,
        allowed_domains=["*"],
    )
    response = _issue_embed_ticket(manager_id, "https://trusted-site.com")
    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid widget key"


def test_widget_public_tokens_cannot_create_tasks_for_other_agents() -> None:
    _admin_headers()
    owner_id = _user_id("admin")
    allowed_agent_id = _create_agent_row(
        user_id=owner_id,
        name="Allowed Widget Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=True,
        allowed_domains=["example.com"],
    )
    other_agent_id = _create_agent_row(
        user_id=owner_id,
        name="Other Widget Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=True,
        allowed_domains=["example.com"],
    )

    guest_headers = _authenticate_widget_guest(agent_id=allowed_agent_id)

    create_task_response = client.post(
        "/api/widget/chat/task/create",
        json={
            "title": "cross agent",
            "description": "cross agent",
            "agent_id": other_agent_id,
        },
        headers=guest_headers,
    )
    assert create_task_response.status_code == 403, create_task_response.text
    assert create_task_response.json()["detail"] == "Widget access is unavailable"


def test_widget_task_create_persists_connector_runtime_selection_snapshot() -> None:
    _admin_headers()
    owner_id = _user_id("admin")
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Widget Snapshot Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=True,
        allowed_domains=["example.com"],
    )
    guest_headers = _authenticate_widget_guest(agent_id=agent_id)

    create_task_response = client.post(
        "/api/widget/chat/task/create",
        json={
            "title": "hello from widget",
            "description": "hello from widget",
            "agent_id": agent_id,
        },
        headers=guest_headers,
    )
    assert create_task_response.status_code == 200, create_task_response.text

    db = _direct_db_session()
    try:
        task = (
            db.query(Task)
            .filter(Task.id == create_task_response.json()["task_id"])
            .one()
        )
        assert task.connector_runtime_selected_refs == []
    finally:
        db.close()


def _create_widget_task(guest_headers: dict[str, str], agent_id: int) -> Any:
    return client.post(
        "/api/widget/chat/task/create",
        json={
            "title": "hello",
            "description": "hello",
            "agent_id": agent_id,
        },
        headers=guest_headers,
    )


def test_disabled_widget_invalidates_existing_guest_tokens() -> None:
    """Disabling an agent's widget must invalidate already-issued guest JWTs
    on their next request (mirrors the workforce/share widget path)."""
    headers = _admin_headers()
    owner_id = _user_id("admin")
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Disable Invalidate Widget Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=True,
    )
    guest_headers = _authenticate_widget_guest(agent_id=agent_id)

    # Sanity: the guest token works while the widget is enabled.
    assert _create_widget_task(guest_headers, agent_id).status_code == 200

    disabled = client.put(
        f"/api/agents/{agent_id}",
        headers=headers,
        json={"widget_enabled": False},
    )
    assert disabled.status_code == 200, disabled.text

    response = _create_widget_task(guest_headers, agent_id)
    assert response.status_code == 403, response.text
    assert response.json()["detail"] == "Widget is unavailable"


def test_rotated_widget_key_invalidates_existing_guest_tokens() -> None:
    """Rotating an agent's widget key must invalidate already-issued guest
    JWTs on their next request."""
    headers = _admin_headers()
    owner_id = _user_id("admin")
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Rotate Invalidate Widget Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=True,
    )
    guest_headers = _authenticate_widget_guest(agent_id=agent_id)

    assert _create_widget_task(guest_headers, agent_id).status_code == 200

    rotate = client.post(f"/api/agents/{agent_id}/widget-key/rotate", headers=headers)
    assert rotate.status_code == 200, rotate.text

    response = _create_widget_task(guest_headers, agent_id)
    assert response.status_code == 403, response.text
    assert response.json()["detail"] == "Widget is unavailable"


def test_widget_guest_token_rejected_when_agent_becomes_generated_manager() -> None:
    """``ensure_widget_agent_available`` must reject a live guest token once the
    backing agent is a workforce-generated manager, tripping the check at the
    task-create site (not just at ``/api/widget/auth``)."""
    _admin_headers()
    owner_id = _user_id("admin")
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Becomes Manager Widget Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=True,
    )
    guest_headers = _authenticate_widget_guest(agent_id=agent_id)
    assert _create_widget_task(guest_headers, agent_id).status_code == 200

    # Flip the agent into a generated manager after the token was issued.
    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        agent.origin = AgentOrigin.WORKFORCE_GENERATED_MANAGER.value
        db.commit()
    finally:
        db.close()

    response = _create_widget_task(guest_headers, agent_id)
    assert response.status_code == 403, response.text
    assert response.json()["detail"] == "Widget is unavailable"


def _mint_legacy_widget_guest_token(
    *,
    user_id: int,
    agent_id: int,
    guest_id: str = "guest-legacy",
) -> dict[str, str]:
    """Mint a widget guest JWT the way it was minted before the ``widget_key``
    claim shipped (issue #988): no key embedded. Exercises the backward-compat
    branch in ``ensure_widget_agent_available``."""
    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.id == user_id).one()
        username = str(user.username)
    finally:
        db.close()
    token = create_public_chat_access_token(
        {
            "sub": username,
            "user_id": user_id,
            "channel_id": None,
            "guest_id": guest_id,
            "auth_mode": "widget",
            "widget_agent_id": agent_id,
        }
    )
    return {"Authorization": f"Bearer {token}"}


def test_legacy_guest_token_without_widget_key_stays_gated_on_widget_enabled() -> None:
    """Guest tokens minted before the ``widget_key`` claim shipped carry no key:
    they still work while the widget is enabled, survive key rotation (no key to
    compare), but are revoked when the widget is disabled. Locks in the
    transitional contract documented on ``ensure_widget_agent_available``."""
    headers = _admin_headers()
    owner_id = _user_id("admin")
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Legacy Token Widget Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=True,
    )
    guest_headers = _mint_legacy_widget_guest_token(user_id=owner_id, agent_id=agent_id)

    # (a) A keyless legacy token works while the widget is enabled.
    assert _create_widget_task(guest_headers, agent_id).status_code == 200

    # (c) Rotating the key does not revoke a keyless token: there is no key
    # claim to compare, so it stays gated on widget_enabled only.
    rotate = client.post(f"/api/agents/{agent_id}/widget-key/rotate", headers=headers)
    assert rotate.status_code == 200, rotate.text
    assert _create_widget_task(guest_headers, agent_id).status_code == 200

    # (b) Disabling the widget still revokes it.
    disabled = client.put(
        f"/api/agents/{agent_id}",
        headers=headers,
        json={"widget_enabled": False},
    )
    assert disabled.status_code == 200, disabled.text
    response = _create_widget_task(guest_headers, agent_id)
    assert response.status_code == 403, response.text
    assert response.json()["detail"] == "Widget is unavailable"


def test_share_link_requires_published_agent() -> None:
    headers = _admin_headers()
    owner_id = _user_id("admin")
    draft_agent_id = _create_agent_row(
        user_id=owner_id,
        name="Draft Share Agent",
        status=AgentStatus.DRAFT,
    )

    response = client.post(f"/api/agents/{draft_agent_id}/share-link", headers=headers)

    assert response.status_code == 400
    assert response.json()["detail"] == "Only published agents can be shared"


def test_generic_agent_responses_do_not_expose_share_token() -> None:
    headers = _admin_headers()
    owner_id = _user_id("admin")
    share_token = "hidden-from-generic-agent-response"
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Hidden Share Token Agent",
        status=AgentStatus.PUBLISHED,
        share_enabled=True,
        share_token=share_token,
    )

    detail_response = client.get(f"/api/agents/{agent_id}", headers=headers)
    assert detail_response.status_code == 200, detail_response.text
    assert "share_token" not in detail_response.json()

    list_response = client.get("/api/agents", headers=headers)
    assert list_response.status_code == 200, list_response.text
    list_item = next(item for item in list_response.json() if item["id"] == agent_id)
    assert "share_token" not in list_item

    share_link_response = client.get(
        f"/api/agents/{agent_id}/share-link", headers=headers
    )
    assert share_link_response.status_code == 200, share_link_response.text
    assert share_link_response.json()["share_token"] == share_token


def test_share_link_can_be_enabled_rotated_and_disabled() -> None:
    headers = _admin_headers()
    owner_id = _user_id("admin")
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Published Share Agent",
        status=AgentStatus.PUBLISHED,
    )

    enable_response = client.post(f"/api/agents/{agent_id}/share-link", headers=headers)
    assert enable_response.status_code == 200, enable_response.text
    enabled_agent = enable_response.json()
    assert enabled_agent["agent_id"] == agent_id
    assert enabled_agent["share_enabled"] is True
    assert isinstance(enabled_agent["share_token"], str)
    first_token = enabled_agent["share_token"]

    rotate_response = client.post(
        f"/api/agents/{agent_id}/share-link/rotate", headers=headers
    )
    assert rotate_response.status_code == 200, rotate_response.text
    rotated_agent = rotate_response.json()
    assert rotated_agent["share_enabled"] is True
    assert isinstance(rotated_agent["share_token"], str)
    assert rotated_agent["share_token"] != first_token

    disable_response = client.delete(
        f"/api/agents/{agent_id}/share-link", headers=headers
    )
    assert disable_response.status_code == 200, disable_response.text
    disabled_agent = disable_response.json()
    assert disabled_agent["share_enabled"] is False
    assert disabled_agent["share_token"] is None


def test_created_agents_get_a_widget_key() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers, name="Widget Key Agent")

    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        assert agent is not None
        assert isinstance(agent.widget_key, str) and len(agent.widget_key) >= 32
    finally:
        db.close()


def test_generic_agent_responses_do_not_expose_widget_key() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers, name="Hidden Widget Key Agent")

    detail_response = client.get(f"/api/agents/{agent_id}", headers=headers)
    assert detail_response.status_code == 200, detail_response.text
    assert "widget_key" not in detail_response.json()

    list_response = client.get("/api/agents", headers=headers)
    assert list_response.status_code == 200, list_response.text
    list_item = next(item for item in list_response.json() if item["id"] == agent_id)
    assert "widget_key" not in list_item


def test_widget_key_endpoint_is_owner_only_and_heals_missing_keys() -> None:
    headers = _admin_headers()
    owner_id = _user_id("admin")
    # Row created directly, predating the widget_key column (no key).
    agent_id = _create_agent_row(user_id=owner_id, name="Legacy Widget Agent")

    response = client.get(f"/api/agents/{agent_id}/widget-key", headers=headers)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["agent_id"] == agent_id
    assert isinstance(payload["widget_key"], str) and len(payload["widget_key"]) >= 32

    # Stable across reads once generated.
    second = client.get(f"/api/agents/{agent_id}/widget-key", headers=headers)
    assert second.json()["widget_key"] == payload["widget_key"]

    other_headers = _register_second_user()
    for method, url in (
        ("get", f"/api/agents/{agent_id}/widget-key"),
        ("post", f"/api/agents/{agent_id}/widget-key/rotate"),
    ):
        other_response = getattr(client, method)(url, headers=other_headers)
        assert other_response.status_code == 404


def test_widget_key_can_be_rotated() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers, name="Rotating Widget Agent")

    first = client.get(f"/api/agents/{agent_id}/widget-key", headers=headers)
    assert first.status_code == 200, first.text
    first_key = first.json()["widget_key"]

    rotate = client.post(f"/api/agents/{agent_id}/widget-key/rotate", headers=headers)
    assert rotate.status_code == 200, rotate.text
    rotated_key = rotate.json()["widget_key"]
    assert isinstance(rotated_key, str) and rotated_key != first_key

    after = client.get(f"/api/agents/{agent_id}/widget-key", headers=headers)
    assert after.json()["widget_key"] == rotated_key


def test_enabling_widget_generates_missing_key() -> None:
    headers = _admin_headers()
    owner_id = _user_id("admin")
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Disabled Widget Agent",
        widget_enabled=False,
    )

    response = client.put(
        f"/api/agents/{agent_id}",
        headers=headers,
        json={"widget_enabled": True},
    )
    assert response.status_code == 200, response.text

    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        assert agent is not None
        assert isinstance(agent.widget_key, str) and len(agent.widget_key) >= 32
    finally:
        db.close()


def test_share_link_authenticates_public_chat_for_published_agent() -> None:
    _admin_headers()
    owner_id = _user_id("admin")
    share_token = "public-share-token"
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Public Share Agent",
        status=AgentStatus.PUBLISHED,
        share_enabled=True,
        share_token=share_token,
    )

    response = client.post(
        "/api/share/auth",
        json={"share_token": share_token},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["agent_id"] == agent_id
    assert payload["agent_name"] == "Public Share Agent"
    assert isinstance(payload["access_token"], str)


def test_disabled_share_link_invalidates_existing_public_tokens() -> None:
    headers = _admin_headers()
    owner_id = _user_id("admin")
    share_token = "revoked-share-token"
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Revoked Share Agent",
        status=AgentStatus.PUBLISHED,
        share_enabled=True,
        share_token=share_token,
    )

    guest_headers = _authenticate_share_guest(share_token)

    disable_response = client.delete(
        f"/api/agents/{agent_id}/share-link", headers=headers
    )
    assert disable_response.status_code == 200, disable_response.text

    create_task_response = client.post(
        "/api/share/chat/task/create",
        json={
            "title": "hello",
            "description": "hello",
            "agent_id": agent_id,
        },
        headers=guest_headers,
    )
    assert create_task_response.status_code == 403, create_task_response.text
    assert create_task_response.json()["detail"] == "Share link is unavailable"

    upload_response = client.post(
        "/api/share/files/upload",
        headers=guest_headers,
        data={"task_type": "task"},
        files={"file": ("note.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert upload_response.status_code == 403, upload_response.text
    assert upload_response.json()["detail"] == "Share link is unavailable"


def test_rotated_share_link_invalidates_existing_public_tokens() -> None:
    headers = _admin_headers()
    owner_id = _user_id("admin")
    original_share_token = "rotating-share-token"
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Rotating Share Agent",
        status=AgentStatus.PUBLISHED,
        share_enabled=True,
        share_token=original_share_token,
    )

    guest_headers = _authenticate_share_guest(original_share_token)

    rotate_response = client.post(
        f"/api/agents/{agent_id}/share-link/rotate", headers=headers
    )
    assert rotate_response.status_code == 200, rotate_response.text
    rotated_share_token = rotate_response.json()["share_token"]
    assert isinstance(rotated_share_token, str)
    assert rotated_share_token != original_share_token

    create_task_response = client.post(
        "/api/share/chat/task/create",
        json={
            "title": "hello after rotate",
            "description": "hello after rotate",
            "agent_id": agent_id,
        },
        headers=guest_headers,
    )
    assert create_task_response.status_code == 403, create_task_response.text
    assert create_task_response.json()["detail"] == "Share link is unavailable"

    refreshed_guest_headers = _authenticate_share_guest(rotated_share_token)
    refreshed_create_task_response = client.post(
        "/api/share/chat/task/create",
        json={
            "title": "hello with new token",
            "description": "hello with new token",
            "agent_id": agent_id,
        },
        headers=refreshed_guest_headers,
    )
    assert refreshed_create_task_response.status_code == 200, (
        refreshed_create_task_response.text
    )
    db = _direct_db_session()
    try:
        task = (
            db.query(Task)
            .filter(Task.id == refreshed_create_task_response.json()["task_id"])
            .one()
        )
        assert task.connector_runtime_selected_refs == []
    finally:
        db.close()


def test_reenabled_share_link_invalidates_existing_public_tokens() -> None:
    headers = _admin_headers()
    owner_id = _user_id("admin")
    original_share_token = "reenable-share-token"
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Re-enabled Share Agent",
        status=AgentStatus.PUBLISHED,
        share_enabled=True,
        share_token=original_share_token,
    )

    guest_headers = _authenticate_share_guest(original_share_token)

    disable_response = client.delete(
        f"/api/agents/{agent_id}/share-link", headers=headers
    )
    assert disable_response.status_code == 200, disable_response.text

    enable_response = client.post(f"/api/agents/{agent_id}/share-link", headers=headers)
    assert enable_response.status_code == 200, enable_response.text
    new_share_token = enable_response.json()["share_token"]
    assert isinstance(new_share_token, str)
    assert new_share_token != original_share_token

    create_task_response = client.post(
        "/api/share/chat/task/create",
        json={
            "title": "hello after re-enable",
            "description": "hello after re-enable",
            "agent_id": agent_id,
        },
        headers=guest_headers,
    )
    assert create_task_response.status_code == 403, create_task_response.text
    assert create_task_response.json()["detail"] == "Share link is unavailable"

    refreshed_guest_headers = _authenticate_share_guest(new_share_token)
    refreshed_create_task_response = client.post(
        "/api/share/chat/task/create",
        json={
            "title": "hello after reauth",
            "description": "hello after reauth",
            "agent_id": agent_id,
        },
        headers=refreshed_guest_headers,
    )
    assert refreshed_create_task_response.status_code == 200, (
        refreshed_create_task_response.text
    )


def test_share_upload_requires_task_id() -> None:
    _admin_headers()
    owner_id = _user_id("admin")
    share_token = "upload-needs-task-token"
    _create_agent_row(
        user_id=owner_id,
        name="Upload Requires Task Agent",
        status=AgentStatus.PUBLISHED,
        share_enabled=True,
        share_token=share_token,
    )

    guest_headers = _authenticate_share_guest(share_token)

    upload_response = client.post(
        "/api/share/files/upload",
        headers=guest_headers,
        data={"task_type": "task"},
        files={"file": ("note.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert upload_response.status_code == 400, upload_response.text
    assert upload_response.json()["detail"] == "task_id is required"


def test_share_public_file_preview_requires_valid_share_token() -> None:
    headers = _admin_headers()
    owner_id = _user_id("admin")
    share_token = "share-preview-token"
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Preview Share Agent",
        status=AgentStatus.PUBLISHED,
        share_enabled=True,
        share_token=share_token,
    )
    file_id = _create_public_task_file(
        owner_id=owner_id,
        agent_id=agent_id,
    )

    preview_without_token = client.get(f"/api/files/public/preview/{file_id}")
    assert preview_without_token.status_code == 403, preview_without_token.text

    guest_headers = _authenticate_share_guest(share_token)
    access_token = guest_headers["Authorization"].replace("Bearer ", "", 1)
    preview_with_token = client.get(
        f"/api/files/public/preview/{file_id}",
        params={"token": access_token},
    )
    assert preview_with_token.status_code == 200, preview_with_token.text
    assert preview_with_token.content == b"hello from public task"

    disable_response = client.delete(
        f"/api/agents/{agent_id}/share-link", headers=headers
    )
    assert disable_response.status_code == 200, disable_response.text

    preview_after_disable = client.get(
        f"/api/files/public/preview/{file_id}",
        params={"token": access_token},
    )
    assert preview_after_disable.status_code == 403, preview_after_disable.text
    assert preview_after_disable.json()["detail"] == "Share link is unavailable"


def test_share_public_file_download_requires_valid_share_token() -> None:
    _admin_headers()
    owner_id = _user_id("admin")
    share_token = "share-download-token"
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Download Share Agent",
        status=AgentStatus.PUBLISHED,
        share_enabled=True,
        share_token=share_token,
    )
    file_id = _create_public_task_file(
        owner_id=owner_id,
        agent_id=agent_id,
        filename="download-note.txt",
    )

    guest_headers = _authenticate_share_guest(share_token)
    access_token = guest_headers["Authorization"].replace("Bearer ", "", 1)

    download_with_token = client.get(
        f"/api/files/public/download/{file_id}",
        params={"token": access_token},
    )
    assert download_with_token.status_code == 200, download_with_token.text
    assert download_with_token.content == b"hello from public task"

    download_without_token = client.get(f"/api/files/public/download/{file_id}")
    assert download_without_token.status_code == 403, download_without_token.text


def _logo_data_url(payload: bytes = b"fake-logo-bytes") -> str:
    return "data:image/png;base64," + base64.b64encode(payload).decode()


@pytest.fixture
def logo_uploads_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Isolate agent logo reads/writes to a temp dir instead of the real uploads root."""
    monkeypatch.setattr(agents_api, "get_uploads_dir", lambda: tmp_path)
    return tmp_path


def test_logo_reupload_gets_a_fresh_url_and_removes_the_old_file(
    logo_uploads_dir: Path,
) -> None:
    """A re-uploaded logo must change logo_url (#975): the old fix saved every
    logo under the same deterministic filename, so browsers kept serving the
    stale cached image after an update even though the file on disk changed."""
    headers = _admin_headers()
    agent_id = _create_agent(headers)

    first = client.put(
        f"/api/agents/{agent_id}",
        headers=headers,
        json={"logo_base64": _logo_data_url(b"first")},
    )
    assert first.status_code == 200, first.text
    first_logo_url = first.json()["logo_url"]
    assert first_logo_url
    first_path = logo_uploads_dir / first_logo_url.removeprefix("/uploads/")
    assert first_path.is_file()

    second = client.put(
        f"/api/agents/{agent_id}",
        headers=headers,
        json={"logo_base64": _logo_data_url(b"second")},
    )
    assert second.status_code == 200, second.text
    second_logo_url = second.json()["logo_url"]

    assert second_logo_url and second_logo_url != first_logo_url
    assert not first_path.exists(), "old logo file should be cleaned up"
    assert (logo_uploads_dir / second_logo_url.removeprefix("/uploads/")).is_file()


def test_clearing_logo_with_empty_string_nulls_url_and_deletes_file(
    logo_uploads_dir: Path,
) -> None:
    """logo_base64="" (vs. omitted, meaning "no change") must clear an
    existing logo end-to-end (#976)."""
    headers = _admin_headers()
    agent_id = _create_agent(headers)

    uploaded = client.put(
        f"/api/agents/{agent_id}",
        headers=headers,
        json={"logo_base64": _logo_data_url()},
    )
    assert uploaded.status_code == 200, uploaded.text
    logo_url = uploaded.json()["logo_url"]
    logo_path = logo_uploads_dir / logo_url.removeprefix("/uploads/")
    assert logo_path.is_file()

    cleared = client.put(
        f"/api/agents/{agent_id}",
        headers=headers,
        json={"logo_base64": ""},
    )
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["logo_url"] is None
    assert not logo_path.exists()


def test_delete_logo_removes_a_file_inside_the_uploads_root(
    logo_uploads_dir: Path,
) -> None:
    logo_dir = logo_uploads_dir / "agent_logos"
    logo_dir.mkdir(parents=True)
    logo_file = logo_dir / "agent_1_abcd1234.png"
    logo_file.write_bytes(b"fake")

    agents_api._delete_logo("/uploads/agent_logos/agent_1_abcd1234.png")

    assert not logo_file.exists()


def test_delete_logo_refuses_a_path_traversal_attempt(
    logo_uploads_dir: Path,
) -> None:
    """A logo_url escaping the uploads root (e.g. via "..") must be refused
    rather than deleting whatever it resolves to outside that root."""
    outside_file = logo_uploads_dir.parent / "outside-secret.txt"
    outside_file.write_text("do not delete me")

    try:
        agents_api._delete_logo("/uploads/../" + outside_file.name)
        assert outside_file.exists()
    finally:
        outside_file.unlink(missing_ok=True)


class TestDeleteAgent:
    """DELETE /api/agents/{agent_id} - remove an agent."""

    def test_rejects_visible_manager_and_worker_references(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        headers = _admin_headers()
        owner_id = _user_id("admin")
        target_id = _create_agent_row(
            user_id=owner_id,
            name="Referenced Agent",
            status=AgentStatus.PUBLISHED,
        )
        alternate_manager_id = _create_agent_row(
            user_id=owner_id,
            name="Alternate Manager",
            status=AgentStatus.PUBLISHED,
        )

        db = _direct_db_session()
        try:
            manager_workforce = Workforce(
                owner_user_id=owner_id,
                scope_type="user",
                scope_id=str(owner_id),
                name="Manager Reference",
                manager_agent_id=target_id,
                status="draft",
            )
            worker_workforce = Workforce(
                owner_user_id=owner_id,
                scope_type="user",
                scope_id=str(owner_id),
                name="Worker Reference",
                manager_agent_id=alternate_manager_id,
                status="draft",
            )
            archived_workforce = Workforce(
                owner_user_id=owner_id,
                scope_type="user",
                scope_id=str(owner_id),
                name="Archived Reference",
                manager_agent_id=target_id,
                status="archived",
            )
            db.add_all([manager_workforce, worker_workforce, archived_workforce])
            db.flush()
            db.add(
                WorkforceAgent(
                    workforce_id=worker_workforce.id,
                    agent_id=target_id,
                    assignment_instructions="Handle delegated work.",
                    source_type="existing",
                    enabled=True,
                    sort_order=0,
                )
            )
            db.add(
                WorkforceRun(
                    workforce_id=worker_workforce.id,
                    user_id=owner_id,
                    status="completed",
                    snapshot={"version": 1},
                )
            )
            db.commit()
            manager_workforce_id = int(manager_workforce.id)
            worker_workforce_id = int(worker_workforce.id)
            archived_workforce_id = int(archived_workforce.id)
        finally:
            db.close()

        logo_calls: list[str] = []
        cache_calls: list[tuple[object, ...]] = []
        monkeypatch.setattr("xagent.web.api.agents._delete_logo", logo_calls.append)
        monkeypatch.setattr(
            "xagent.web.services.agent_management.invalidate_agent_cache",
            lambda *args: cache_calls.append(args),
        )
        monkeypatch.setattr(
            "xagent.web.services.agent_store.invalidate_agent_cache",
            lambda *args: cache_calls.append(args),
        )

        response = client.delete(f"/api/agents/{target_id}", headers=headers)

        assert response.status_code == 409, response.text
        assert response.json() == {
            "detail": {
                "code": "agent_in_use_by_workforce",
                "message": "Agent is used by one or more workforces.",
                "references": [
                    {
                        "workforce_id": manager_workforce_id,
                        "name": "Manager Reference",
                        "status": "draft",
                        "roles": ["manager"],
                        "can_edit": True,
                        "can_discard": True,
                    },
                    {
                        "workforce_id": worker_workforce_id,
                        "name": "Worker Reference",
                        "status": "draft",
                        "roles": ["worker"],
                        "can_edit": True,
                        "can_discard": False,
                    },
                    {
                        "workforce_id": archived_workforce_id,
                        "name": "Archived Reference",
                        "status": "archived",
                        "roles": ["manager"],
                        "can_edit": False,
                        "can_discard": False,
                    },
                ],
                "has_hidden_references": False,
            }
        }
        assert logo_calls == []
        assert cache_calls == []

        db = _direct_db_session()
        try:
            assert db.get(Agent, target_id) is not None
            assert (
                db.query(WorkforceAgent)
                .filter(
                    WorkforceAgent.workforce_id == worker_workforce_id,
                    WorkforceAgent.agent_id == target_id,
                )
                .one_or_none()
                is not None
            )
        finally:
            db.close()

    def test_hidden_only_reference_returns_sanitized_conflict(self) -> None:
        _admin_headers()
        bob_headers = _register_second_user("bob", "bobpass1")
        admin_id = _user_id("admin")
        bob_id = _user_id("bob")
        target_id = _create_agent_row(
            user_id=bob_id,
            name="Bob Referenced Agent",
            status=AgentStatus.PUBLISHED,
        )

        db = _direct_db_session()
        try:
            hidden_workforce = Workforce(
                owner_user_id=admin_id,
                scope_type="user",
                scope_id=str(admin_id),
                name="Hidden Workforce Name",
                manager_agent_id=target_id,
                status="draft",
            )
            db.add(hidden_workforce)
            db.commit()
            hidden_workforce_id = int(hidden_workforce.id)
        finally:
            db.close()

        response = client.delete(f"/api/agents/{target_id}", headers=bob_headers)

        assert response.status_code == 409, response.text
        assert response.json() == {
            "detail": {
                "code": "agent_in_use_by_workforce",
                "message": "Agent is used by one or more workforces.",
                "references": [],
                "has_hidden_references": True,
            }
        }
        assert "Hidden Workforce Name" not in response.text
        assert str(hidden_workforce_id) not in response.text

        db = _direct_db_session()
        try:
            assert db.get(Agent, target_id) is not None
        finally:
            db.close()

    def test_empty_workforce_reference_state_has_no_conflict(self) -> None:
        _admin_headers()
        owner_id = _user_id("admin")

        db = _direct_db_session()
        try:
            actor = db.query(User).filter(User.id == owner_id).one()

            conflict = AgentManagementService(db)._workforce_conflict(
                actor=actor,
                agent_id=999_999,
            )

            assert conflict is None
        finally:
            db.close()

    def test_reference_snapshot_classifies_policy_visibility_in_one_statement(
        self,
    ) -> None:
        _admin_headers()
        owner_id = _user_id("admin")
        target_id = _create_agent_row(user_id=owner_id, name="Snapshot Target")

        db = _direct_db_session()
        try:
            actor = db.query(User).filter(User.id == owner_id).one()
            visible_workforce = Workforce(
                owner_user_id=owner_id,
                scope_type="user",
                scope_id=str(owner_id),
                name="Policy Visible",
                manager_agent_id=target_id,
                status="draft",
            )
            hidden_workforce = Workforce(
                owner_user_id=owner_id,
                scope_type="user",
                scope_id=str(owner_id),
                name="Policy Hidden",
                manager_agent_id=target_id,
                status="draft",
            )
            db.add_all([visible_workforce, hidden_workforce])
            db.commit()
            visible_workforce_id = int(visible_workforce.id)
            hidden_workforce_id = int(hidden_workforce.id)

            class NameVisibilityPolicy(WorkforcePolicy):
                filter_calls = 0

                def filter_visible_workforces(
                    self,
                    db: Any,
                    user: User,
                    query: Any,
                ) -> Any:
                    del db, user
                    self.filter_calls += 1
                    return query.filter(Workforce.name == "Policy Visible")

            policy = NameVisibilityPolicy()
            set_workforce_policy(policy)
            statements: list[str] = []
            engine = db.get_bind()

            def record_statement(
                _connection: Any,
                _cursor: Any,
                statement: str,
                _parameters: Any,
                _context: Any,
                _executemany: bool,
            ) -> None:
                statements.append(statement)

            event.listen(engine, "before_cursor_execute", record_statement)
            try:
                snapshot = AgentManagementService(db)._workforce_reference_snapshot(
                    actor=actor,
                    agent_id=target_id,
                )
            finally:
                event.remove(engine, "before_cursor_execute", record_statement)

            snapshot_by_id = {
                reference.workforce_id: reference for reference in snapshot
            }
            assert policy.filter_calls == 1
            assert len(statements) == 1
            assert snapshot_by_id[visible_workforce_id].is_visible is True
            assert snapshot_by_id[hidden_workforce_id].is_visible is False
        finally:
            db.close()

    def test_workforce_conflict_requires_blocker_evidence(self) -> None:
        with pytest.raises(ValueError, match="blocker evidence"):
            AgentWorkforceConflictError((), has_hidden_references=False)

    def test_visible_workforce_conflict_requires_at_least_one_role(self) -> None:
        with pytest.raises(ValueError, match="at least one role"):
            AgentWorkforceConflictError(
                (
                    AgentWorkforceReference(
                        workforce_id=7,
                        name="Invalid reference",
                        status="draft",
                        roles=(),
                        can_edit=True,
                        can_discard=False,
                    ),
                ),
                has_hidden_references=False,
            )

    def test_reference_marks_draft_undiscardable_when_generated_manager_is_shared(
        self,
    ) -> None:
        headers = _admin_headers()
        owner_id = _user_id("admin")
        target_id = _create_agent_row(
            user_id=owner_id,
            name="Referenced Worker",
            status=AgentStatus.PUBLISHED,
        )
        generated_manager_id = _create_agent_row(
            user_id=owner_id,
            name="Shared Generated Manager",
            status=AgentStatus.PUBLISHED,
            origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
        )

        db = _direct_db_session()
        try:
            referenced_workforce = Workforce(
                owner_user_id=owner_id,
                scope_type="user",
                scope_id=str(owner_id),
                name="Referenced Draft",
                manager_agent_id=generated_manager_id,
                status="draft",
            )
            other_workforce = Workforce(
                owner_user_id=owner_id,
                scope_type="user",
                scope_id=str(owner_id),
                name="Other Manager Reference",
                manager_agent_id=generated_manager_id,
                status="draft",
            )
            db.add_all([referenced_workforce, other_workforce])
            db.flush()
            db.add(
                WorkforceAgent(
                    workforce_id=int(referenced_workforce.id),
                    agent_id=target_id,
                    assignment_instructions="Handle delegated work.",
                    source_type="existing",
                    enabled=True,
                    sort_order=0,
                )
            )
            db.commit()
            referenced_workforce_id = int(referenced_workforce.id)
        finally:
            db.close()

        response = client.delete(f"/api/agents/{target_id}", headers=headers)

        assert response.status_code == 409, response.text
        references = response.json()["detail"]["references"]
        assert references == [
            {
                "workforce_id": referenced_workforce_id,
                "name": "Referenced Draft",
                "status": "draft",
                "roles": ["worker"],
                "can_edit": True,
                "can_discard": False,
            }
        ]

    def test_disappearing_worker_reference_is_gone_before_conflict_classification(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        headers = _admin_headers()
        owner_id = _user_id("admin")
        target_id = _create_agent_row(user_id=owner_id, name="Racing Worker")
        manager_id = _create_agent_row(user_id=owner_id, name="Race Manager")

        setup_db = _direct_db_session()
        try:
            workforce = Workforce(
                owner_user_id=owner_id,
                scope_type="user",
                scope_id=str(owner_id),
                name="Disappearing Worker Reference",
                manager_agent_id=manager_id,
                status="draft",
            )
            setup_db.add(workforce)
            setup_db.flush()
            setup_db.add(
                WorkforceAgent(
                    workforce_id=int(workforce.id),
                    agent_id=target_id,
                    assignment_instructions="Race with Agent deletion.",
                    source_type="existing",
                    enabled=True,
                    sort_order=0,
                )
            )
            setup_db.commit()
            workforce_id = int(workforce.id)
        finally:
            setup_db.close()

        original_snapshot = AgentManagementService._workforce_reference_snapshot

        def snapshot_then_discard(
            service: AgentManagementService,
            *,
            actor: User,
            agent_id: int,
        ):
            snapshot = original_snapshot(service, actor=actor, agent_id=agent_id)
            discard_db = _direct_db_session()
            try:
                discard_actor = discard_db.query(User).filter(User.id == owner_id).one()
                discard_draft_workforce(
                    discard_db,
                    discard_actor,
                    discard_db.get(Workforce, workforce_id),
                )
            finally:
                discard_db.close()
            return snapshot

        monkeypatch.setattr(
            AgentManagementService,
            "_workforce_reference_snapshot",
            snapshot_then_discard,
        )

        response = client.delete(f"/api/agents/{target_id}", headers=headers)

        assert response.status_code == 200, response.text
        assert response.json() == {"message": "Agent deleted successfully"}

        verify_db = _direct_db_session()
        try:
            assert verify_db.get(Workforce, workforce_id) is None
            assert verify_db.get(Agent, target_id) is None
        finally:
            verify_db.close()

    def test_maps_commit_time_fk_race_after_rollback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _admin_headers()
        owner_id = _user_id("admin")
        target_id = _create_agent_row(user_id=owner_id, name="Racing Agent")

        db = _direct_db_session()
        try:
            actor = db.query(User).filter(User.id == owner_id).one()
            service = AgentManagementService(db)
            conflicts = iter(
                [
                    None,
                    AgentWorkforceConflictError(
                        (),
                        has_hidden_references=True,
                    ),
                ]
            )
            monkeypatch.setattr(
                service,
                "_workforce_conflict",
                lambda **_kwargs: next(conflicts),
            )
            monkeypatch.setattr(
                db,
                "commit",
                lambda: (_ for _ in ()).throw(
                    IntegrityError(
                        "DELETE FROM agents WHERE id = ?",
                        {"id": target_id},
                        RuntimeError("raw fk constraint details"),
                    )
                ),
            )

            with pytest.raises(AgentWorkforceConflictError) as exc_info:
                service.delete_agent(actor=actor, agent_id=target_id)

            assert "raw fk constraint details" not in str(exc_info.value)
            assert db.get(Agent, target_id) is not None
        finally:
            db.close()

    def test_unrelated_integrity_error_is_not_mapped_to_workforce_conflict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _admin_headers()
        owner_id = _user_id("admin")
        target_id = _create_agent_row(user_id=owner_id, name="Unrelated Error Agent")

        db = _direct_db_session()
        try:
            actor = db.query(User).filter(User.id == owner_id).one()
            service = AgentManagementService(db)
            monkeypatch.setattr(
                service,
                "_workforce_conflict",
                lambda **_kwargs: None,
            )
            raw_error = IntegrityError(
                "DELETE FROM agents WHERE id = ?",
                {"id": target_id},
                RuntimeError("unrelated integrity failure"),
            )
            monkeypatch.setattr(
                db,
                "commit",
                lambda: (_ for _ in ()).throw(raw_error),
            )

            with pytest.raises(IntegrityError) as exc_info:
                service.delete_agent(actor=actor, agent_id=target_id)

            assert exc_info.value is raw_error
            assert db.get(Agent, target_id) is not None
        finally:
            db.close()

    def test_unexpected_delete_failure_returns_structured_sanitized_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        headers = _admin_headers()
        owner_id = _user_id("admin")
        target_id = _create_agent_row(user_id=owner_id, name="Failing Agent")

        def fail_delete(
            _service: AgentManagementService,
            *,
            actor: User,
            agent_id: int,
        ) -> None:
            del actor, agent_id
            raise RuntimeError("sensitive database detail")

        monkeypatch.setattr(AgentManagementService, "delete_agent", fail_delete)

        response = client.delete(f"/api/agents/{target_id}", headers=headers)

        assert response.status_code == 500, response.text
        assert response.json() == {
            "detail": {
                "code": "agent_delete_failed",
                "message": "Failed to delete agent",
            }
        }
        assert "sensitive database detail" not in response.text

    def test_logo_and_cache_cleanup_observe_committed_delete(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        headers = _admin_headers()
        owner_id = _user_id("admin")
        target_id = _create_agent_row(user_id=owner_id, name="Cleanup Agent")
        logo_url = f"/uploads/agent_logos/agent_{target_id}.png"

        db = _direct_db_session()
        try:
            agent = db.get(Agent, target_id)
            assert agent is not None
            agent.logo_url = logo_url
            db.commit()
        finally:
            db.close()

        cleanup_events: list[str] = []

        def assert_committed(event: str) -> None:
            check_db = _direct_db_session()
            try:
                assert check_db.get(Agent, target_id) is None
            finally:
                check_db.close()
            cleanup_events.append(event)

        monkeypatch.setattr(
            "xagent.web.api.agents._delete_logo",
            lambda value: (
                assert_committed("logo")
                if value == logo_url
                else pytest.fail("unexpected logo URL")
            ),
        )
        monkeypatch.setattr(
            "xagent.web.services.agent_management.invalidate_agent_cache",
            lambda *_args: assert_committed("cache"),
        )
        monkeypatch.setattr(
            "xagent.web.services.agent_store.invalidate_agent_cache",
            lambda *_args: assert_committed("cache"),
        )

        response = client.delete(f"/api/agents/{target_id}", headers=headers)

        assert response.status_code == 200, response.text
        assert sorted(cleanup_events) == ["cache", "logo"]

    def test_with_tasks_keeps_tasks_and_nulls_agent_id(self):
        headers = _admin_headers()
        agent_id = _create_agent(headers)
        client.post(f"/api/agents/{agent_id}/api-key", headers=headers)

        db = _direct_db_session()
        try:
            admin_user = db.query(User).filter(User.username == "admin").first()
            assert admin_user is not None
            task = Task(
                user_id=admin_user.id,
                title="task tied to agent",
                description="task tied to agent",
                status=TaskStatus.PENDING,
                agent_id=agent_id,
            )
            db.add(task)
            db.commit()
            db.refresh(task)
            task_id = task.id
        finally:
            db.close()

        delete_resp = client.delete(f"/api/agents/{agent_id}", headers=headers)
        assert delete_resp.status_code == 200, delete_resp.text

        db = _direct_db_session()
        try:
            task = db.query(Task).filter(Task.id == task_id).first()
            assert task is not None
            assert task.agent_id is None
            assert (
                db.query(AgentApiKey).filter(AgentApiKey.agent_id == agent_id).all()
                == []
            )
        finally:
            db.close()


def test_admin_can_read_other_users_agent_detail():
    """管理员只读查看普通用户的 agent 详情。"""
    admin = _admin_headers()
    bob = _register_second_user("bob", "bobpass1")

    created = client.post(
        "/api/agents",
        headers=bob,
        json={
            "name": "Bob Agent",
            "instructions": "hi",
            "tool_categories": ["mcp:foo"],
        },
    )
    assert created.status_code == 200, created.text
    agent_id = created.json()["id"]

    resp = client.get(f"/api/agents/{agent_id}", headers=admin)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "mcp:foo" in body["tool_categories"]
    # Admin sees it read-only: writes stay owner-only, so the builder must lock
    # instead of letting a save fail with "Agent not found".
    assert body["readonly"] is True
    assert body["can_edit"] is False

    # The owner still gets an editable detail.
    owner_resp = client.get(f"/api/agents/{agent_id}", headers=bob)
    assert owner_resp.status_code == 200, owner_resp.text
    assert owner_resp.json()["readonly"] is False
    assert owner_resp.json()["can_edit"] is True


def test_non_admin_cannot_read_other_users_agent_detail():
    """普通用户读别人 agent 仍 404，不泄露存在性。"""
    _admin_headers()
    bob = _register_second_user("bob", "bobpass1")
    carol = _register_second_user("carol", "carolpass1")

    created = client.post(
        "/api/agents", headers=bob, json={"name": "Bob Agent", "instructions": "hi"}
    )
    agent_id = created.json()["id"]

    resp = client.get(f"/api/agents/{agent_id}", headers=carol)
    assert resp.status_code == 404


def test_policy_shared_non_admin_reads_agent_detail_read_only():
    """非管理员通过 policy 只读共享，可读详情且被锁为只读（不再 404）。"""
    _admin_headers()
    _register_second_user("bob", "bobpass1")
    bob_id = _user_id("bob")

    agent_id = _create_agent_row(user_id=bob_id, name="Bob Shared")
    # carol 不拥有该 agent，但 policy 把它列进了 carol 的可见列表。
    carol = _register_second_user("carol", "carolpass1")
    set_workforce_policy(_VisibleAgentPolicy({agent_id}))

    resp = client.get(f"/api/agents/{agent_id}", headers=carol)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["readonly"] is True
    assert body["can_edit"] is False
