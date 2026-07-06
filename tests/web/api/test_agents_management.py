"""Integration tests for agent management endpoints."""

import io
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from jose import jwt as jose_jwt

from xagent.config import get_uploads_dir
from xagent.web.api.widget import EMBED_TICKET_TYPE
from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY
from xagent.web.models.agent import Agent, AgentOrigin, AgentStatus
from xagent.web.models.agent_api_key import AgentApiKey
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce
from xagent.web.services.workforce_access import WorkforcePolicy, set_workforce_policy

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
) -> int:
    db = _direct_db_session()
    try:
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
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        return int(agent.id)
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
    response = client.post(
        "/api/widget/auth",
        json={"agent_id": agent_id, "guest_id": guest_id},
        headers={"origin": origin},
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
        json={"agent_id": generated_manager_id, "guest_id": "guest-1"},
        headers={"origin": "https://example.com"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Widget owner not found or invalid agent_id"


def test_widget_auth_matches_allowed_domains_case_insensitively() -> None:
    _admin_headers()
    owner_id = _user_id("admin")
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Case Insensitive Widget Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=True,
        allowed_domains=["Example.com"],
    )

    response = client.post(
        "/api/widget/auth",
        json={"agent_id": agent_id, "guest_id": "guest-1"},
        headers={"origin": "https://EXAMPLE.com"},
    )

    assert response.status_code == 200, response.text


def _issue_embed_ticket(agent_id: int, origin: str) -> Any:
    return client.post(
        "/api/widget/embed-ticket",
        json={"agent_id": agent_id},
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


def test_widget_auth_ignores_client_supplied_embed_origin() -> None:
    """A browser-JS attacker must not pass by self-reporting an allowed
    origin in the request body (the pre-ticket design's flaw)."""
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
    assert response.json()["detail"] == "Domain not allowed: evil-attacker.com"


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

    # Valid ticket, but issued for a different agent.
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
    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid or expired embed ticket"


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


def test_widget_auth_falls_back_to_origin_header_without_embed_ticket() -> None:
    _admin_headers()
    owner_id = _user_id("admin")
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Header Fallback Widget Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=True,
        allowed_domains=["trusted-site.com"],
    )

    response = client.post(
        "/api/widget/auth",
        json={"agent_id": agent_id, "guest_id": "guest-1"},
        headers={"origin": "https://trusted-site.com"},
    )

    assert response.status_code == 200, response.text


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


def test_widget_auth_rejects_when_ticket_and_origin_both_absent() -> None:
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
    assert response.json()["detail"] == "Domain not allowed: "


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


def test_embed_ticket_endpoint_rejects_unknown_agent() -> None:
    _admin_headers()
    response = _issue_embed_ticket(999999, "https://trusted-site.com")
    assert response.status_code == 401
    assert response.json()["detail"] == "Widget owner not found or invalid agent_id"


def test_embed_ticket_endpoint_rejects_generated_manager_agent() -> None:
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
    assert response.status_code == 401
    assert response.json()["detail"] == "Widget owner not found or invalid agent_id"


def test_embed_ticket_endpoint_rejects_widget_disabled_agent() -> None:
    _admin_headers()
    owner_id = _user_id("admin")
    agent_id = _create_agent_row(
        user_id=owner_id,
        name="Widget Disabled Ticket Agent",
        status=AgentStatus.PUBLISHED,
        widget_enabled=False,
        allowed_domains=["trusted-site.com"],
    )
    response = _issue_embed_ticket(agent_id, "https://trusted-site.com")
    assert response.status_code == 403
    assert response.json()["detail"] == "Widget is disabled for this agent"


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


class TestDeleteAgent:
    """DELETE /api/agents/{agent_id} - remove an agent."""

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
    assert "mcp:foo" in resp.json()["tool_categories"]


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
