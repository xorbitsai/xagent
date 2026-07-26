"""Integration tests for /v1 management endpoints."""

from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy.exc import IntegrityError

from xagent.templates.manager import TemplateManager
from xagent.web.models.agent import Agent
from xagent.web.models.model import Model as DBModel
from xagent.web.models.user import User, UserModel

from ..conftest import _admin_headers, _direct_db_session, app_for_tests, client

pytestmark = pytest.mark.usefixtures("_test_db")


def _personal_key() -> str:
    headers = _admin_headers()
    resp = client.post("/api/me/personal-keys", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()["full_key"]


def _bearer(full_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {full_key}"}


def _write_template(root: Path) -> None:
    (root / "qa.yaml").write_text(
        """
id: qa
name: Q&A Assistant
category: General
descriptions:
  en: Answers questions from provided context.
features:
  en:
    - Ask questions
connections: []
agent_config:
  instructions: Answer clearly.
  skills:
    - retrieval
  tool_categories:
    - web_search
  suggested_prompts:
    - Ask anything
  execution_mode: balanced
""".strip(),
        encoding="utf-8",
    )
    # Pre-sets a knowledge base without the knowledge tool category, so
    # from-template must reject it -- exercises that template-sourced KB
    # fields go through the same validation as user input.
    (root / "kb-no-tool.yaml").write_text(
        """
id: kb-no-tool
name: KB without tool
category: General
descriptions:
  en: Pre-sets a knowledge base but not the knowledge tool category.
connections: []
agent_config:
  instructions: Answer.
  tool_categories:
    - web_search
  knowledge_bases:
    - template-kb
  execution_mode: balanced
""".strip(),
        encoding="utf-8",
    )


@pytest.fixture
def template_manager(tmp_path):
    _write_template(tmp_path)
    manager = TemplateManager(templates_root=tmp_path)
    app_for_tests.state.template_manager = manager
    return manager


def test_personal_key_management_round_trip():
    headers = _admin_headers()
    create = client.post("/api/me/personal-keys", headers=headers)
    assert create.status_code == 200, create.text
    body = create.json()
    assert body["full_key"].startswith("xag_personal_")

    list_resp = client.get("/api/me/personal-keys", headers=headers)
    assert list_resp.status_code == 200
    assert any(row["key_prefix"] == body["key_prefix"] for row in list_resp.json())

    revoke = client.delete(f"/api/me/personal-keys/{body['id']}", headers=headers)
    assert revoke.status_code == 200
    assert revoke.json()["revoked"] is True


def test_v1_me_uses_personal_key():
    key = _personal_key()
    resp = client.get("/v1/me", headers=_bearer(key))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["principal_type"] == "user"
    assert body["username"] == "admin"
    assert body["email"] == "admin@example.com"


def test_v1_create_agent_defaults_to_runtime_key():
    key = _personal_key()
    resp = client.post(
        "/v1/agents",
        headers=_bearer(key),
        json={
            "name": "SDK-created agent",
            "description": "created from management SDK",
            "instructions": "Be useful.",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["agent"]["name"] == "SDK-created agent"
    assert body["api_key"]["full_key"].startswith("xag_")


def test_v1_create_agent_can_skip_runtime_key():
    key = _personal_key()
    resp = client.post(
        "/v1/agents",
        headers=_bearer(key),
        json={
            "name": "No key agent",
            "instructions": "Be useful.",
            "generate_runtime_key": False,
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["api_key"] is None


def test_v1_list_agents_returns_detached_summary_contract():
    key = _personal_key()
    create = client.post(
        "/v1/agents",
        headers=_bearer(key),
        json={
            "name": "Listed SDK agent",
            "instructions": "Be useful.",
            "generate_runtime_key": False,
        },
    )
    assert create.status_code == 200, create.text

    response = client.get("/v1/agents", headers=_bearer(key))

    assert response.status_code == 200, response.text
    listed = next(
        item for item in response.json() if item["name"] == "Listed SDK agent"
    )
    assert listed["id"] == create.json()["agent"]["id"]
    assert listed["status"] == "draft"
    assert listed["allowed_domains"] == []


def test_v1_rotate_agent_runtime_key_returns_one_shot_key():
    key = _personal_key()
    create = client.post(
        "/v1/agents",
        headers=_bearer(key),
        json={
            "name": "Rotated SDK agent",
            "instructions": "Be useful.",
            "generate_runtime_key": False,
        },
    )
    assert create.status_code == 200, create.text
    agent_id = create.json()["agent"]["id"]

    response = client.post(
        f"/v1/agents/{agent_id}/api-key",
        headers=_bearer(key),
    )

    assert response.status_code == 200, response.text
    assert response.json()["full_key"].startswith("xag_")
    assert response.json()["key_prefix"]
    assert response.json()["created_at"]


def test_v1_create_agent_rolls_back_agent_when_key_step_fails():
    """Agent + first runtime key commit atomically: if the key step
    fails, the agent row must not persist, so a client retry with the
    same name succeeds instead of hitting a stale duplicate-name."""
    key = _personal_key()
    name = "atomic-create agent"

    with patch(
        "xagent.web.services.agent_management.AgentApiKeyService.stage_rotated_key",
        side_effect=RuntimeError("staged key write blew up"),
    ):
        resp = client.post(
            "/v1/agents",
            headers=_bearer(key),
            json={"name": name, "instructions": "Be useful."},
        )
    assert resp.status_code == 500, resp.text

    # The aborted create must leave no row behind.
    db = _direct_db_session()
    try:
        leftover = db.query(Agent).filter(Agent.name == name).count()
    finally:
        db.close()
    assert leftover == 0

    # A clean retry with the same name now goes through.
    retry = client.post(
        "/v1/agents",
        headers=_bearer(key),
        json={"name": name, "instructions": "Be useful."},
    )
    assert retry.status_code == 200, retry.text
    assert retry.json()["agent"]["name"] == name


def test_v1_create_agent_maps_commit_conflict_to_409():
    """A unique-constraint IntegrityError at commit is translated to a
    409 rotation conflict, not a 500."""
    key = _personal_key()
    with patch(
        "sqlalchemy.orm.Session.commit",
        side_effect=IntegrityError(
            "INSERT", {}, Exception("uq_agent_api_keys_agent_active")
        ),
    ):
        resp = client.post(
            "/v1/agents",
            headers=_bearer(key),
            json={"name": "commit conflict agent", "instructions": "x"},
        )
    assert resp.status_code == 409, resp.text


def test_v1_create_agent_rejects_kb_without_knowledge_category():
    """KB selected but the knowledge tool category is not enabled -> 400,
    same invariant /api/agents enforces."""
    key = _personal_key()
    resp = client.post(
        "/v1/agents",
        headers=_bearer(key),
        json={
            "name": "kb no tool",
            "instructions": "x",
            "knowledge_bases": ["some-kb"],
            "tool_categories": ["web_search"],
        },
    )
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["code"] == "invalid_input"


def test_v1_create_agent_rejects_invisible_kb():
    """KB not visible to the user -> 400."""
    key = _personal_key()
    resp = client.post(
        "/v1/agents",
        headers=_bearer(key),
        json={
            "name": "kb invisible",
            "instructions": "x",
            "knowledge_bases": ["nonexistent-kb"],
            "tool_categories": ["knowledge"],
        },
    )
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["code"] == "invalid_input"


def test_v1_create_agent_rejects_string_model_ids():
    key = _personal_key()
    resp = client.post(
        "/v1/agents",
        headers=_bearer(key),
        json={
            "name": "Bad model shape",
            "instructions": "Be useful.",
            "models": {"general": "deepseek-v4-flash"},
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_input"


def test_v1_create_agent_rejects_unknown_model_ids():
    key = _personal_key()
    resp = client.post(
        "/v1/agents",
        headers=_bearer(key),
        json={
            "name": "Unknown model",
            "instructions": "Be useful.",
            "models": {"general": 999999},
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_input"


def test_v1_create_agent_rejects_inaccessible_model_ids():
    key = _personal_key()
    db = _direct_db_session()
    try:
        other = User(username="model-owner", password_hash="hash")
        db.add(other)
        db.flush()
        model = DBModel(
            model_id="other-private-model",
            category="llm",
            model_provider="openai",
            model_name="gpt-4",
            api_key="test-api-key",
            base_url="https://api.openai.com/v1",
            is_active=True,
        )
        db.add(model)
        db.flush()
        db.add(
            UserModel(
                user_id=other.id,
                model_id=model.id,
                is_owner=True,
                is_shared=False,
            )
        )
        db.commit()
        model_id = model.id
    finally:
        db.close()

    resp = client.post(
        "/v1/agents",
        headers=_bearer(key),
        json={
            "name": "Inaccessible model",
            "instructions": "Be useful.",
            "models": {"general": model_id},
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_input"


def test_runtime_key_cannot_call_management_endpoint():
    personal = _personal_key()
    create = client.post(
        "/v1/agents",
        headers=_bearer(personal),
        json={"name": "Runtime only", "instructions": "hi"},
    )
    assert create.status_code == 200, create.text
    runtime_key = create.json()["api_key"]["full_key"]

    resp = client.get("/v1/agents", headers=_bearer(runtime_key))
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "invalid_api_key"


def test_personal_key_cannot_call_runtime_endpoint():
    personal = _personal_key()
    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(personal),
        json={
            "agent_id": 1,
            "message": {"role": "user", "content": "hello"},
        },
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "invalid_api_key"


def test_from_template_creates_agent(template_manager):
    key = _personal_key()
    resp = client.post(
        "/v1/agents/from-template",
        headers=_bearer(key),
        json={"template_id": "qa", "name": "Template agent"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["agent"]["name"] == "Template agent"
    assert body["agent"]["instructions"] == "Answer clearly."
    assert body["agent"]["skills"] == ["retrieval"]
    assert body["agent"]["tool_categories"] == ["web_search"]
    assert body["agent"]["knowledge_bases"] == []
    assert body["agent"]["suggested_prompts"] == ["Ask anything"]
    assert body["api_key"]["full_key"].startswith("xag_")


def test_from_template_strips_agent_tool_category(template_manager):
    """A tool_categories override containing ``agent`` is silently
    stripped, same as the plain create path (issue #802)."""
    key = _personal_key()
    resp = client.post(
        "/v1/agents/from-template",
        headers=_bearer(key),
        json={
            "template_id": "qa",
            "name": "Template agent stripped",
            "tool_categories": ["web_search", "agent"],
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["agent"]["tool_categories"] == ["web_search"]


def test_from_template_allows_empty_list_overrides(template_manager):
    key = _personal_key()
    resp = client.post(
        "/v1/agents/from-template",
        headers=_bearer(key),
        json={
            "template_id": "qa",
            "name": "Template agent without defaults",
            "knowledge_bases": [],
            "skills": [],
            "tool_categories": [],
            "suggested_prompts": [],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["agent"]["knowledge_bases"] == []
    assert body["agent"]["skills"] == []
    assert body["agent"]["tool_categories"] == []
    assert body["agent"]["suggested_prompts"] == []


def test_unknown_template_returns_stable_error(template_manager):
    key = _personal_key()
    resp = client.post(
        "/v1/agents/from-template",
        headers=_bearer(key),
        json={"template_id": "missing"},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "template_not_found"


def test_from_template_rolls_back_agent_when_key_step_fails(template_manager):
    """The from-template path shares the plain-create atomic boundary: a
    key-step failure must not leave the template-derived agent behind."""
    key = _personal_key()
    name = "atomic-template agent"

    with patch(
        "xagent.web.services.agent_management.AgentApiKeyService.stage_rotated_key",
        side_effect=RuntimeError("staged key write blew up"),
    ):
        resp = client.post(
            "/v1/agents/from-template",
            headers=_bearer(key),
            json={"template_id": "qa", "name": name},
        )
    assert resp.status_code == 500, resp.text

    db = _direct_db_session()
    try:
        leftover = db.query(Agent).filter(Agent.name == name).count()
    finally:
        db.close()
    assert leftover == 0

    retry = client.post(
        "/v1/agents/from-template",
        headers=_bearer(key),
        json={"template_id": "qa", "name": name},
    )
    assert retry.status_code == 200, retry.text
    assert retry.json()["agent"]["name"] == name


def test_from_template_rejects_template_kb_without_knowledge_category(
    template_manager,
):
    """Template-sourced KB fields go through the same validation as user
    input: a template that pre-sets a KB without the knowledge category
    is rejected, not silently persisted."""
    key = _personal_key()
    resp = client.post(
        "/v1/agents/from-template",
        headers=_bearer(key),
        json={"template_id": "kb-no-tool", "name": "from bad template"},
    )
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["code"] == "invalid_input"


def test_from_template_rejects_invisible_kb_override(template_manager):
    """A KB override that is not visible to the user -> 400 on the
    from-template path too."""
    key = _personal_key()
    resp = client.post(
        "/v1/agents/from-template",
        headers=_bearer(key),
        json={
            "template_id": "qa",
            "name": "from template invisible kb",
            "knowledge_bases": ["nonexistent-kb"],
            "tool_categories": ["knowledge"],
        },
    )
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["code"] == "invalid_input"
