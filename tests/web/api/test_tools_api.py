"""
Tests for Tools API endpoints.

This module tests the /api/tools endpoints, including the /available endpoint
which lists all tools that can be used by agents.
"""

import tempfile
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from xagent.web.api.auth import auth_router
from xagent.web.api.tools import tools_router
from xagent.web.models.database import Base, get_db, get_engine, init_db


def override_get_db():
    db = None
    try:
        db = next(get_db())
        yield db
    finally:
        if db is not None:
            db.close()


# Create test app without startup events
test_app = FastAPI()
test_app.include_router(auth_router)
test_app.include_router(tools_router)
test_app.dependency_overrides[get_db] = override_get_db

# Create test client
client = TestClient(test_app)


def ensure_system_initialized() -> None:
    status_response = client.get("/api/auth/setup-status")
    assert status_response.status_code == 200
    status_data = status_response.json()

    if status_data.get("needs_setup", True):
        setup_response = client.post(
            "/api/auth/setup-admin", json={"username": "admin", "password": "admin123"}
        )
        assert setup_response.status_code == 200
        assert setup_response.json().get("success") is True


@pytest.fixture(scope="function")
def test_db():
    """Create test database"""
    import os
    import shutil

    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test.db")
    SQLALCHEMY_DATABASE_URL = f"sqlite:///{temp_db_path}"

    with patch("xagent.web.models.database.try_upgrade_db"):
        init_db(db_url=SQLALCHEMY_DATABASE_URL)

    engine = get_engine()

    yield temp_dir

    Base.metadata.drop_all(bind=engine)
    shutil.rmtree(temp_dir)


class TestToolsAvailableAPI:
    """Test /api/tools/available endpoint."""

    @pytest.fixture(autouse=True)
    def setup(self, test_db):
        """Setup system initialization before each test."""
        ensure_system_initialized()
        yield

    def test_get_available_tools_without_workspace(self):
        """Test that /api/tools/available works without a real workspace.

        This endpoint is used to list available tools for the UI.
        It should work even when there's no active task/workspace.
        """
        # Login to get token
        login_response = client.post(
            "/api/auth/login", json={"username": "admin", "password": "admin123"}
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        # Make request to /api/tools/available
        response = client.get(
            "/api/tools/available", headers={"Authorization": f"Bearer {token}"}
        )

        # Should succeed without errors
        assert response.status_code == 200

        data = response.json()
        assert "tools" in data
        assert "count" in data

        tools = data["tools"]
        assert isinstance(tools, list)

        # Check that basic tool categories are present
        tool_names = [t["name"] for t in tools]

        # Should always have these knowledge tools
        assert "knowledge_search" in tool_names
        assert "list_knowledge_bases" in tool_names

        # Should have PPTX tools (don't require workspace)
        assert "read_pptx" in tool_names
        assert "unpack_pptx" in tool_names
        assert "pack_pptx" in tool_names
        assert "clean_pptx" in tool_names

        # Should have browser tools (when enabled)
        assert "browser_navigate" in tool_names
        assert "browser_click" in tool_names

        # Basic tools - web search depends on API keys being set
        has_web_search = "web_search" in tool_names or "zhipu_web_search" in tool_names
        if has_web_search:
            # At least one web search tool is present (if API keys configured)
            pass

        # Code execution tools should now be present (workspace is created)
        assert "execute_python_code" in tool_names, "Should have python executor"
        assert "execute_javascript_code" in tool_names, (
            "Should have javascript executor"
        )

        # File tools should also be present (workspace is created)
        assert "read_file" in tool_names, "Should have read_file tool"
        assert "write_file" in tool_names, "Should have write_file tool"

        # Skill file access tools should be present
        assert "read_skill_doc" in tool_names, "Should have read_skill_doc tool"
        assert "list_skill_docs" in tool_names, "Should have list_skill_docs tool"
        assert "fetch_skill_file" in tool_names, "Should have fetch_skill_file tool"

    def test_skill_category_in_available_tools(self):
        """Test that skill tools appear with correct category."""
        # Login to get token
        login_response = client.post(
            "/api/auth/login", json={"username": "admin", "password": "admin123"}
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        response = client.get(
            "/api/tools/available", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        data = response.json()

        # Check for skill tools
        skill_tools = [
            tool for tool in data["tools"] if tool.get("category") == "skill"
        ]

        # Should have read_skill_doc and list_skill_docs
        skill_tool_names = {tool["name"] for tool in skill_tools}
        assert "read_skill_doc" in skill_tool_names
        assert "list_skill_docs" in skill_tool_names
        assert "fetch_skill_file" in skill_tool_names

        # Verify tool type and display category
        for tool in skill_tools:
            assert tool["type"] == "skill"
            assert tool["display_category"] == "Skill"

    def test_get_available_tools_includes_usage_count(self):
        """Test that /api/tools/available includes usage statistics."""
        # Login to get token
        login_response = client.post(
            "/api/auth/login", json={"username": "admin", "password": "admin123"}
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        response = client.get(
            "/api/tools/available", headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200

        data = response.json()
        tools = data["tools"]

        # Each tool should have usage_count field
        for tool in tools:
            assert "usage_count" in tool
            assert isinstance(tool["usage_count"], int)
            assert "requires_configuration" in tool
            assert isinstance(tool["requires_configuration"], bool)

        sql_tools = [tool for tool in tools if tool["category"] == "database"]
        assert sql_tools
        assert all(tool["requires_configuration"] is True for tool in sql_tools)

    def test_get_available_tools_tool_categories(self):
        """Test that tools have correct category information."""
        # Login to get token
        login_response = client.post(
            "/api/auth/login", json={"username": "admin", "password": "admin123"}
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        response = client.get(
            "/api/tools/available", headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200

        data = response.json()
        tools = data["tools"]

        # Build a map of tool names to categories
        tool_categories = {t["name"]: t["category"] for t in tools}
        tool_display_categories = {t["name"]: t["display_category"] for t in tools}

        # Verify categories
        assert tool_categories.get("knowledge_search") == "knowledge"
        assert tool_display_categories.get("knowledge_search") == "Knowledge"

        # PPT display name should be "PPT" not "Ppt"
        assert tool_display_categories.get("read_pptx") == "PPT"
        assert tool_categories.get("read_pptx") == "ppt"

        assert tool_display_categories.get("browser_navigate") == "Browser"
        assert tool_categories.get("browser_navigate") == "browser"

    def test_get_available_tools_requires_auth(self):
        """Test that /api/tools/available requires authentication."""
        response = client.get("/api/tools/available")

        # Should return 401 (older FastAPI) or 403 (newer FastAPI) without auth
        assert response.status_code in [401, 403]

    def test_get_available_tools_falls_back_to_other_when_metadata_missing(self):
        login_response = client.post(
            "/api/auth/login", json={"username": "admin", "password": "admin123"}
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        class _Category:
            value = "basic"

        class _Metadata:
            category = _Category()

        class _ToolWithoutMetadata:
            name = "tool_without_metadata"
            description = ""

        class _ToolWithMetadata:
            name = "tool_with_metadata"
            description = ""
            metadata = _Metadata()

        with patch(
            "xagent.core.tools.adapters.vibe.factory.ToolFactory.create_all_tools",
            return_value=[_ToolWithoutMetadata(), _ToolWithMetadata()],
        ):
            response = client.get(
                "/api/tools/available", headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        payload = response.json()
        categories = {item["name"]: item["category"] for item in payload["tools"]}
        assert categories["tool_without_metadata"] == "other"
        assert categories["tool_with_metadata"] == "basic"


class TestToolsGovernanceAPI:
    @pytest.fixture(autouse=True)
    def setup(self, test_db):
        ensure_system_initialized()
        yield

    def _admin_headers(self) -> dict[str, str]:
        login_response = client.post(
            "/api/auth/login", json={"username": "admin", "password": "admin123"}
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def _user_headers(self, username: str) -> dict[str, str]:
        register_response = client.post(
            "/api/auth/register", json={"username": username, "password": "password123"}
        )
        assert register_response.status_code == 200

        login_response = client.post(
            "/api/auth/login", json={"username": username, "password": "password123"}
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def test_enable_unknown_tool_creates_policy_record(self):
        headers = self._admin_headers()

        response = client.put(
            "/api/tools/custom_runtime_tool/enabled",
            headers=headers,
            json={"enabled": False},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["tool_name"] == "custom_runtime_tool"
        assert data["enabled"] is False

    def test_configurable_credentials_put_and_get_masked(self):
        headers = self._admin_headers()

        put_resp = client.put(
            "/api/tools/zhipu_web_search/credentials",
            headers=headers,
            json={
                "credentials": {
                    "api_key": {"value": "test-secret-zhipu-key-1234"},
                    "base_url": {"value": "https://open.bigmodel.cn"},
                }
            },
        )
        assert put_resp.status_code == 200

        get_resp = client.get(
            "/api/tools/zhipu_web_search/credentials",
            headers=headers,
        )
        assert get_resp.status_code == 200
        payload = get_resp.json()

        assert payload["tool_name"] == "zhipu_web_search"
        assert payload["configured"] is True
        assert payload["fields"]["api_key"]["source"] == "db"
        assert payload["fields"]["api_key"]["is_configured"] is True
        assert "1234" in payload["fields"]["api_key"]["masked"]
        assert (
            "test-secret-zhipu-key-1234" not in payload["fields"]["api_key"]["masked"]
        )

    def test_configurable_credentials_env_source_when_not_stored(self, monkeypatch):
        headers = self._admin_headers()
        monkeypatch.setenv("TAVILY_API_KEY", "env-only-tavily-key-5678")

        resp = client.get("/api/tools/tavily_web_search/credentials", headers=headers)
        assert resp.status_code == 200
        payload = resp.json()

        assert payload["fields"]["api_key"]["source"] == "env"
        assert payload["fields"]["api_key"]["is_configured"] is True
        assert "5678" in payload["fields"]["api_key"]["masked"]

    def test_sql_connections_crud_and_db_priority_over_env(self, monkeypatch):
        headers = self._admin_headers()
        monkeypatch.setenv(
            "XAGENT_EXTERNAL_DB_ANALYTICS",
            "postgresql://env_user:env_pass@localhost:5432/env_db",
        )

        initial = client.get("/api/tools/sql-connections", headers=headers)
        assert initial.status_code == 200
        initial_items = {item["name"]: item for item in initial.json()["connections"]}
        assert initial_items["ANALYTICS"]["source"] == "env"

        upsert = client.put(
            "/api/tools/sql-connections/analytics",
            headers=headers,
            json={
                "connection_url": "postgresql://db_user:db_pass@localhost:5432/db_db"
            },
        )
        assert upsert.status_code == 200

        after_upsert = client.get("/api/tools/sql-connections", headers=headers)
        assert after_upsert.status_code == 200
        upsert_items = {
            item["name"]: item for item in after_upsert.json()["connections"]
        }
        assert upsert_items["ANALYTICS"]["source"] == "db"
        assert "db_pass" not in upsert_items["ANALYTICS"]["masked"]

        delete_resp = client.delete(
            "/api/tools/sql-connections/analytics", headers=headers
        )
        assert delete_resp.status_code == 200

        after_delete = client.get("/api/tools/sql-connections", headers=headers)
        assert after_delete.status_code == 200
        delete_items = {
            item["name"]: item for item in after_delete.json()["connections"]
        }
        assert delete_items["ANALYTICS"]["source"] == "env"

    def test_sql_connection_rejects_unsupported_scheme(self):
        headers = self._admin_headers()

        upsert = client.put(
            "/api/tools/sql-connections/analytics",
            headers=headers,
            json={"connection_url": "redis://localhost:6379/0"},
        )

        assert upsert.status_code == 400
        assert "Unsupported SQLAlchemy URL scheme" in upsert.json()["detail"]

    def test_sql_connections_are_user_scoped(self):
        user1_headers = self._user_headers("user1")
        user2_headers = self._user_headers("user2")

        user1_upsert = client.put(
            "/api/tools/sql-connections/analytics",
            headers=user1_headers,
            json={"connection_url": "postgresql://user1:pass1@localhost:5432/user1_db"},
        )
        assert user1_upsert.status_code == 200

        user2_initial = client.get("/api/tools/sql-connections", headers=user2_headers)
        assert user2_initial.status_code == 200
        assert user2_initial.json()["connections"] == []

        user2_upsert = client.put(
            "/api/tools/sql-connections/analytics",
            headers=user2_headers,
            json={"connection_url": "postgresql://user2:pass2@localhost:5432/user2_db"},
        )
        assert user2_upsert.status_code == 200

        user1_items = {
            item["name"]: item
            for item in client.get(
                "/api/tools/sql-connections", headers=user1_headers
            ).json()["connections"]
        }
        user2_items = {
            item["name"]: item
            for item in client.get(
                "/api/tools/sql-connections", headers=user2_headers
            ).json()["connections"]
        }

        assert user1_items["ANALYTICS"]["source"] == "db"
        assert user2_items["ANALYTICS"]["source"] == "db"
        assert user1_items["ANALYTICS"]["masked"] != user2_items["ANALYTICS"]["masked"]

        user1_delete = client.delete(
            "/api/tools/sql-connections/analytics", headers=user1_headers
        )
        assert user1_delete.status_code == 200

        user1_after_delete = client.get(
            "/api/tools/sql-connections", headers=user1_headers
        )
        user2_after_delete = client.get(
            "/api/tools/sql-connections", headers=user2_headers
        )
        assert user1_after_delete.status_code == 200
        assert user2_after_delete.status_code == 200
        assert user1_after_delete.json()["connections"] == []
        remaining_user2 = {
            item["name"]: item for item in user2_after_delete.json()["connections"]
        }
        assert remaining_user2["ANALYTICS"]["source"] == "db"

    def test_non_admin_cannot_access_global_credentials(self):
        user_headers = self._user_headers("nonadmin")

        configurable_resp = client.get("/api/tools/configurable", headers=user_headers)
        credential_resp = client.get(
            "/api/tools/zhipu_web_search/credentials", headers=user_headers
        )

        assert configurable_resp.status_code == 403
        assert credential_resp.status_code == 403
