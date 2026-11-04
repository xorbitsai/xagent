"""Agent-config write paths silently strip the ``agent`` tool category.

Issue #802: multi-agent delegation is a Workforce concern. An ordinary
agent must not be configurable with the ``agent`` category (which would
inject one delegation tool per published agent in the account), so every
create/update surface — the web agents API and the public v1 agents
API — strips it instead of rejecting the request.
"""

import pytest

from xagent.web.models.agent import Agent

from .conftest import _admin_headers, _direct_db_session, client

pytestmark = pytest.mark.usefixtures("_test_db")


AGENT_BASE = {
    "name": "Strip Test Agent",
    "description": "test",
    "instructions": "You are a test agent.",
    "execution_mode": "balanced",
}


def _stored_tool_categories(agent_id: int) -> list[str]:
    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        return list(agent.tool_categories or [])
    finally:
        db.close()


def _personal_key() -> str:
    headers = _admin_headers()
    resp = client.post("/api/me/personal-keys", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()["full_key"]


class TestWebAgentsApiStripsAgentCategory:
    """POST/PUT /api/agents — ``agent`` never reaches the DB row."""

    def test_create_strips_agent_and_keeps_other_categories(self):
        headers = _admin_headers()
        resp = client.post(
            "/api/agents",
            headers=headers,
            json={
                **AGENT_BASE,
                "tool_categories": ["basic", "agent", "mcp:my_server"],
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["tool_categories"] == ["basic", "mcp:my_server"]
        assert _stored_tool_categories(data["id"]) == ["basic", "mcp:my_server"]

    def test_create_with_only_agent_persists_empty_selection(self):
        headers = _admin_headers()
        resp = client.post(
            "/api/agents",
            headers=headers,
            json={**AGENT_BASE, "tool_categories": ["agent"]},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["tool_categories"] == []
        assert _stored_tool_categories(data["id"]) == []

    def test_update_strips_agent_and_keeps_other_categories(self):
        headers = _admin_headers()
        create = client.post(
            "/api/agents",
            headers=headers,
            json={**AGENT_BASE, "tool_categories": ["basic"]},
        )
        assert create.status_code == 200, create.text
        agent_id = create.json()["id"]

        update = client.put(
            f"/api/agents/{agent_id}",
            headers=headers,
            json={"tool_categories": ["web_search", "agent"]},
        )
        assert update.status_code == 200, update.text
        assert update.json()["tool_categories"] == ["web_search"]
        assert _stored_tool_categories(agent_id) == ["web_search"]

    def test_legacy_agent_row_reads_back_without_agent_category(self):
        """A pre-#802 row with ``agent`` saved is not surfaced on reads."""
        headers = _admin_headers()
        create = client.post(
            "/api/agents",
            headers=headers,
            json={**AGENT_BASE, "tool_categories": ["basic"]},
        )
        assert create.status_code == 200, create.text
        agent_id = create.json()["id"]

        db = _direct_db_session()
        try:
            agent = db.query(Agent).filter(Agent.id == agent_id).one()
            agent.tool_categories = ["basic", "agent"]
            db.commit()
        finally:
            db.close()

        get = client.get(f"/api/agents/{agent_id}", headers=headers)
        assert get.status_code == 200, get.text
        assert get.json()["tool_categories"] == ["basic"]


class TestV1AgentsApiStripsAgentCategory:
    """POST /v1/agents — same silent strip as the web API."""

    def test_v1_create_strips_agent_and_keeps_other_categories(self):
        key = _personal_key()
        resp = client.post(
            "/v1/agents",
            headers={"Authorization": f"Bearer {key}"},
            json={
                **AGENT_BASE,
                "tool_categories": ["basic", "agent"],
                "generate_runtime_key": False,
            },
        )
        assert resp.status_code == 200, resp.text
        agent = resp.json()["agent"]
        assert agent["tool_categories"] == ["basic"]
        assert _stored_tool_categories(agent["id"]) == ["basic"]
