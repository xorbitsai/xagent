import os
import tempfile
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from xagent.web.api.agents import router as agents_router
from xagent.web.api.auth import auth_router
from xagent.web.api.tools import tools_router
from xagent.web.models.database import Base, get_db, get_engine


def override_get_db():
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
app_for_tests.include_router(tools_router)
app_for_tests.dependency_overrides[get_db] = override_get_db
client = TestClient(app_for_tests)


def _setup_test_db() -> str:
    from xagent.web.models.database import init_db

    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test.db")
    db_url = f"sqlite:///{temp_db_path}"

    with patch("xagent.web.models.database.try_upgrade_db"):
        init_db(db_url=db_url)

    return temp_dir


def _setup_admin() -> None:
    status_response = client.get("/api/auth/setup-status")
    assert status_response.status_code == 200
    if status_response.json().get("needs_setup", True):
        setup_response = client.post(
            "/api/auth/setup-admin", json={"username": "admin", "password": "admin123"}
        )
        assert setup_response.status_code == 200


def _login(username: str, password: str) -> dict[str, str]:
    response = client.post(
        "/api/auth/login", json={"username": username, "password": password}
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _create_agent(headers: dict[str, str], name: str) -> int:
    create_response = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": name,
            "description": "publish status test agent",
            "instructions": "test",
            "execution_mode": "react",
        },
    )
    assert create_response.status_code == 200
    return int(create_response.json()["id"])


def test_published_agent_is_callable_for_owner_and_hidden_from_other_users() -> None:
    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")

        register_response = client.post(
            "/api/auth/register",
            json={"username": "regular", "password": "password123"},
        )
        assert register_response.status_code == 200
        regular_headers = _login("regular", "password123")

        agent_id = _create_agent(admin_headers, "Admin Published Agent")

        publish_response = client.post(
            f"/api/agents/{agent_id}/publish",
            headers=admin_headers,
        )
        assert publish_response.status_code == 200

        owner_tools = client.get("/api/tools/available", headers=admin_headers)
        assert owner_tools.status_code == 200
        owner_names = {tool["name"] for tool in owner_tools.json()["tools"]}
        assert "call_agent_admin_published_agent" in owner_names

        other_tools = client.get("/api/tools/available", headers=regular_headers)
        assert other_tools.status_code == 200
        other_names = {tool["name"] for tool in other_tools.json()["tools"]}
        assert "call_agent_admin_published_agent" not in other_names
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_unpublish_moves_agent_back_to_non_callable_state() -> None:
    temp_dir = _setup_test_db()
    try:
        _setup_admin()
        admin_headers = _login("admin", "admin123")
        agent_id = _create_agent(admin_headers, "Admin Toggle Agent")

        publish_response = client.post(
            f"/api/agents/{agent_id}/publish",
            headers=admin_headers,
        )
        assert publish_response.status_code == 200

        tools_after_publish = client.get("/api/tools/available", headers=admin_headers)
        assert tools_after_publish.status_code == 200
        names_after_publish = {
            tool["name"] for tool in tools_after_publish.json()["tools"]
        }
        assert "call_agent_admin_toggle_agent" in names_after_publish

        unpublish_response = client.post(
            f"/api/agents/{agent_id}/unpublish",
            headers=admin_headers,
        )
        assert unpublish_response.status_code == 200

        tools_after_unpublish = client.get(
            "/api/tools/available", headers=admin_headers
        )
        assert tools_after_unpublish.status_code == 200
        names_after_unpublish = {
            tool["name"] for tool in tools_after_unpublish.json()["tools"]
        }
        assert "call_agent_admin_toggle_agent" not in names_after_unpublish
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass
