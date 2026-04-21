"""Tests for knowledge-base + tool-category validation and KB system prompt enhancement."""

import os
import shutil
import tempfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from xagent.web.api.agents import enhance_system_prompt_with_kb
from xagent.web.api.agents import router as agents_router
from xagent.web.api.auth import auth_router
from xagent.web.models.database import Base, get_db, get_engine


def _override_get_db():
    db = None
    try:
        db = next(get_db())
        yield db
    finally:
        if db is not None:
            db.close()


app_for_tests = FastAPI()
app_for_tests.include_router(auth_router)
app_for_tests.include_router(agents_router)
app_for_tests.dependency_overrides[get_db] = _override_get_db
client = TestClient(app_for_tests)


@pytest.fixture(autouse=True)
def _test_db():
    """Create and tear down a temporary SQLite database for each test."""
    from xagent.web.models.database import init_db

    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test.db")
    db_url = f"sqlite:///{temp_db_path}"

    # Note: Previously mocked try_upgrade_db to skip db migrations.
    # For new databases, try_upgrade_db only stamps the latest revision,
    # which is safe for tests and provides better coverage.
    init_db(db_url=db_url)

    yield

    Base.metadata.drop_all(bind=get_engine())
    try:
        shutil.rmtree(temp_dir)
    except OSError:
        pass


def _setup_admin() -> None:
    status = client.get("/api/auth/setup-status")
    assert status.status_code == 200
    if status.json().get("needs_setup", True):
        resp = client.post(
            "/api/auth/setup-admin",
            json={"username": "admin", "password": "admin123"},
        )
        assert resp.status_code == 200


def _login(username: str = "admin", password: str = "admin123") -> dict[str, str]:
    resp = client.post(
        "/api/auth/login", json={"username": username, "password": password}
    )
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _headers() -> dict[str, str]:
    _setup_admin()
    return _login()


AGENT_BASE = {
    "name": "Test Agent",
    "description": "test",
    "instructions": "You are a test agent.",
    "execution_mode": "react",
}


# ── Create ──────────────────────────────────────────────────────────


class TestCreateAgentKbValidation:
    """POST /api/agents — knowledge-base + tool-category validation."""

    def test_create_with_kb_without_knowledge_tool_returns_400(self):
        headers = _headers()
        resp = client.post(
            "/api/agents",
            headers=headers,
            json={
                **AGENT_BASE,
                "knowledge_bases": ["my_kb"],
                "tool_categories": ["basic"],
            },
        )
        assert resp.status_code == 400
        assert "Knowledge" in resp.json()["detail"]

    def test_create_with_kb_and_knowledge_tool_succeeds(self):
        headers = _headers()
        resp = client.post(
            "/api/agents",
            headers=headers,
            json={
                **AGENT_BASE,
                "knowledge_bases": ["my_kb"],
                "tool_categories": ["basic", "knowledge"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["knowledge_bases"] == ["my_kb"]
        assert "knowledge" in data["tool_categories"]

    def test_create_without_kb_without_knowledge_tool_succeeds(self):
        headers = _headers()
        resp = client.post(
            "/api/agents",
            headers=headers,
            json={
                **AGENT_BASE,
                "knowledge_bases": [],
                "tool_categories": ["basic"],
            },
        )
        assert resp.status_code == 200

    def test_create_with_empty_kb_no_tools_succeeds(self):
        headers = _headers()
        resp = client.post(
            "/api/agents",
            headers=headers,
            json={**AGENT_BASE},
        )
        assert resp.status_code == 200

    def test_create_with_multiple_kbs_without_knowledge_tool_returns_400(self):
        headers = _headers()
        resp = client.post(
            "/api/agents",
            headers=headers,
            json={
                **AGENT_BASE,
                "name": "Multi KB Agent",
                "knowledge_bases": ["kb1", "kb2", "kb3"],
                "tool_categories": ["file", "browser"],
            },
        )
        assert resp.status_code == 400


# ── Update ──────────────────────────────────────────────────────────


class TestUpdateAgentKbValidation:
    """PUT /api/agents/{id} — knowledge-base + tool-category validation."""

    def _create_agent(self, headers: dict[str, str], **overrides) -> int:
        payload = {**AGENT_BASE, **overrides}
        resp = client.post("/api/agents", headers=headers, json=payload)
        assert resp.status_code == 200
        return int(resp.json()["id"])

    def test_update_add_kb_without_knowledge_tool_returns_400(self):
        headers = _headers()
        agent_id = self._create_agent(headers, tool_categories=["basic"])
        resp = client.put(
            f"/api/agents/{agent_id}",
            headers=headers,
            json={"knowledge_bases": ["new_kb"]},
        )
        assert resp.status_code == 400
        assert "Knowledge" in resp.json()["detail"]

    def test_update_remove_knowledge_tool_with_existing_kb_returns_400(self):
        headers = _headers()
        agent_id = self._create_agent(
            headers,
            knowledge_bases=["my_kb"],
            tool_categories=["basic", "knowledge"],
        )
        resp = client.put(
            f"/api/agents/{agent_id}",
            headers=headers,
            json={"tool_categories": ["basic"]},
        )
        assert resp.status_code == 400

    def test_update_add_kb_with_knowledge_tool_succeeds(self):
        headers = _headers()
        agent_id = self._create_agent(headers, tool_categories=["basic", "knowledge"])
        resp = client.put(
            f"/api/agents/{agent_id}",
            headers=headers,
            json={"knowledge_bases": ["new_kb"]},
        )
        assert resp.status_code == 200
        assert resp.json()["knowledge_bases"] == ["new_kb"]

    def test_update_add_kb_and_knowledge_tool_together_succeeds(self):
        headers = _headers()
        agent_id = self._create_agent(headers)
        resp = client.put(
            f"/api/agents/{agent_id}",
            headers=headers,
            json={
                "knowledge_bases": ["new_kb"],
                "tool_categories": ["knowledge"],
            },
        )
        assert resp.status_code == 200

    def test_update_remove_kb_and_knowledge_tool_together_succeeds(self):
        headers = _headers()
        agent_id = self._create_agent(
            headers,
            knowledge_bases=["my_kb"],
            tool_categories=["basic", "knowledge"],
        )
        resp = client.put(
            f"/api/agents/{agent_id}",
            headers=headers,
            json={
                "knowledge_bases": [],
                "tool_categories": ["basic"],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["knowledge_bases"] == []

    def test_update_unrelated_field_preserves_valid_state(self):
        headers = _headers()
        agent_id = self._create_agent(
            headers,
            knowledge_bases=["my_kb"],
            tool_categories=["basic", "knowledge"],
        )
        resp = client.put(
            f"/api/agents/{agent_id}",
            headers=headers,
            json={"description": "updated description"},
        )
        assert resp.status_code == 200
        assert resp.json()["knowledge_bases"] == ["my_kb"]
        assert "knowledge" in resp.json()["tool_categories"]

    def test_update_unrelated_field_on_no_kb_agent_succeeds(self):
        headers = _headers()
        agent_id = self._create_agent(headers, tool_categories=["basic"])
        resp = client.put(
            f"/api/agents/{agent_id}",
            headers=headers,
            json={"description": "no kb, just updating desc"},
        )
        assert resp.status_code == 200


# ── enhance_system_prompt_with_kb ───────────────────────────────────


class TestEnhanceSystemPromptWithKb:
    """Unit tests for enhance_system_prompt_with_kb helper."""

    def test_no_kb_returns_original_prompt(self):
        assert enhance_system_prompt_with_kb("Hello", None) == "Hello"
        assert enhance_system_prompt_with_kb("Hello", []) == "Hello"

    def test_no_kb_returns_none_when_prompt_is_none(self):
        assert enhance_system_prompt_with_kb(None, None) is None
        assert enhance_system_prompt_with_kb(None, []) is None

    def test_with_kb_appends_priority_prompt(self):
        result = enhance_system_prompt_with_kb("Be helpful.", ["my_kb"])
        assert result is not None
        assert result.startswith("Be helpful.")
        assert "Available knowledge bases:" in result
        assert "my_kb" in result
        assert "Search them directly for answers" in result

    def test_with_kb_no_system_prompt_returns_priority_only(self):
        result = enhance_system_prompt_with_kb(None, ["kb1"])
        assert result is not None
        assert not result.startswith("\n")
        assert "Available knowledge bases:" in result
        assert "kb1" in result

    def test_with_multiple_kbs(self):
        result = enhance_system_prompt_with_kb("Assist user.", ["kb1", "kb2", "kb3"])
        assert result is not None
        assert result.startswith("Assist user.")
        assert "knowledge base" in result

    def test_empty_string_prompt_with_kb(self):
        result = enhance_system_prompt_with_kb("", ["kb1"])
        assert result is not None
        assert "Available knowledge bases:" in result
        assert "kb1" in result
