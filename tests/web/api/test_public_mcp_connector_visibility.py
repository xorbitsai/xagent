import os
import tempfile

from fastapi import FastAPI
from fastapi.testclient import TestClient

from xagent.web.api.admin_mcp import admin_mcp_router
from xagent.web.api.auth import auth_router
from xagent.web.api.mcp import mcp_router
from xagent.web.models.database import Base, get_db, get_engine
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.user import User


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
app_for_tests.include_router(mcp_router)
app_for_tests.include_router(admin_mcp_router)
app_for_tests.dependency_overrides[get_db] = override_get_db
client = TestClient(app_for_tests)


def _setup_test_db() -> str:
    from xagent.web.models.database import init_db

    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test.db")
    db_url = f"sqlite:///{temp_db_path}"
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
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _create_public_app(
    headers: dict[str, str], app_id: str, name: str, is_visible_in_connector: bool
) -> None:
    response = client.post(
        "/api/admin/mcp/apps",
        headers=headers,
        json={
            "app_id": app_id,
            "name": name,
            "description": f"{name} description",
            "icon": "",
            "transport": "oauth",
            "provider_name": None,
            "category": "Communication",
            "oauth_scopes": [],
            "is_visible_in_connector": is_visible_in_connector,
            "launch_config": {},
        },
    )
    assert response.status_code == 200


def _connect_app_for_user(username: str, server_name: str) -> None:
    db = next(get_db())
    try:
        user = db.query(User).filter(User.username == username).first()
        assert user is not None

        server = MCPServer(
            name=server_name,
            description="connected hidden app",
            managed="external",
            transport="oauth",
        )
        db.add(server)
        db.flush()

        db.add(
            UserMCPServer(
                user_id=user.id,
                mcpserver_id=server.id,
                is_owner=True,
                can_edit=True,
                can_delete=True,
                is_active=True,
            )
        )
        db.commit()
    finally:
        db.close()


def test_hidden_public_mcp_app_is_excluded_from_remote_connector_list() -> None:
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

        _create_public_app(admin_headers, "visible-app", "Visible App", True)
        _create_public_app(admin_headers, "hidden-app", "Hidden App", False)

        response = client.get("/api/mcp/apps?location=remote", headers=regular_headers)
        assert response.status_code == 200

        app_ids = {app["id"] for app in response.json()}
        assert "visible-app" in app_ids
        assert "hidden-app" not in app_ids
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass


def test_connected_hidden_public_mcp_app_is_excluded_in_strong_hide_mode() -> None:
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

        _create_public_app(admin_headers, "hidden-app", "Hidden App", False)
        _connect_app_for_user("regular", "Hidden App")

        response = client.get("/api/mcp/apps?location=remote", headers=regular_headers)
        assert response.status_code == 200

        app_ids = {app["id"] for app in response.json()}
        assert "hidden-app" not in app_ids
    finally:
        Base.metadata.drop_all(bind=get_engine())
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except OSError:
            pass
