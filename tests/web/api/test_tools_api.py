"""
Tests for Tools API endpoints.

This module tests the /api/tools endpoints, including the /available endpoint
which lists all tools that can be used by agents.
"""

import os
import shutil
import tempfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from xagent.web.api.auth import auth_router
from xagent.web.api.tool_credentials import tool_credentials_router
from xagent.web.api.tools import tools_router
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.services.tool_credentials import (
    set_credential_fallback_scopes_hook,
    set_instance_credentials_enabled,
)


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
test_app.include_router(tool_credentials_router)
test_app.dependency_overrides[get_db] = override_get_db

# Create test client
client = TestClient(test_app)


def ensure_system_initialized() -> None:
    status_response = client.get("/api/auth/setup-status")
    assert status_response.status_code == 200
    status_data = status_response.json()

    if status_data.get("needs_setup", True):
        setup_response = client.post(
            "/api/auth/setup-admin",
            json={
                "username": "admin",
                "email": "admin@example.com",
                "password": "admin123",
            },
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
        set_instance_credentials_enabled(True)
        ensure_system_initialized()
        yield
        set_instance_credentials_enabled(True)

    def test_get_available_tools_without_workspace(self, monkeypatch):
        """Test that /api/tools/available works without a real workspace.

        This endpoint is used to list available tools for the UI.
        It should work even when there's no active task/workspace.
        """
        monkeypatch.setenv("XAGENT_WEB_SEARCH_PROVIDER", "google")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CSE_ID", raising=False)

        # Login to get token
        login_response = client.post(
            "/api/auth/login", json={"username": "admin", "password": "admin123"}
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        credential_response = client.put(
            "/api/tool-credentials/web_search?scope=user",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "credentials": {
                    "api_key": {"value": "user-google-key"},
                    "cse_id": {"value": "user-google-cse"},
                }
            },
        )
        assert credential_response.status_code == 200

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

        assert "web_search" in tool_names
        web_search_tool = next(tool for tool in tools if tool["name"] == "web_search")
        assert web_search_tool["requires_configuration"] is True

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

        assert tool_display_categories.get("fetch_web_content") == "Web Search"
        assert tool_categories.get("fetch_web_content") == "web_search"

    def test_get_available_tools_requires_auth(self):
        """Test that /api/tools/available requires authentication."""
        response = client.get("/api/tools/available")

        # Unauthenticated requests return 403
        assert response.status_code == 403

    def test_get_available_tools_falls_back_to_other_when_metadata_missing(
        self, monkeypatch
    ):
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

        # Mock async create_all_tools to return test tools
        async def mock_create_all_tools(config, apply_user_override_filter=True):
            return [_ToolWithoutMetadata(), _ToolWithMetadata()]

        monkeypatch.setattr(
            "xagent.core.tools.adapters.vibe.factory.ToolFactory.create_all_tools",
            mock_create_all_tools,
        )

        response = client.get(
            "/api/tools/available", headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        payload = response.json()
        categories = {item["name"]: item["category"] for item in payload["tools"]}
        assert categories["tool_without_metadata"] == "other"
        assert categories["tool_with_metadata"] == "basic"

    def test_get_available_tools_applies_user_override(self):
        """Test that user tool override hook marks disabled tools in /available."""
        from xagent.web.services.tool_credentials import set_user_tool_overrides_hook

        set_user_tool_overrides_hook(
            lambda db, user: {"browser_navigate": {"enabled": False}}
        )
        try:
            login_response = client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "admin123"},
            )
            assert login_response.status_code == 200
            token = login_response.json()["access_token"]

            response = client.get(
                "/api/tools/available",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert response.status_code == 200
            payload = response.json()

            tool_map = {item["name"]: item for item in payload["tools"]}
            # In the display layer the tool remains visible but is marked disabled.
            assert "browser_navigate" in tool_map
            assert tool_map["browser_navigate"]["enabled"] is False
            assert tool_map["browser_navigate"]["status"] == "disabled"
        finally:
            set_user_tool_overrides_hook(None)

    def test_get_available_tools_override_does_not_mask_missing_model(
        self, monkeypatch
    ):
        """Test that enabled=True override cannot mask resource-missing states."""
        from xagent.web.services.tool_credentials import set_user_tool_overrides_hook

        class _Category:
            value = "vision"

        class _Metadata:
            category = _Category()

        class _VisionTool:
            name = "vision_test_tool"
            description = ""
            metadata = _Metadata()

        async def mock_create_all_tools(config, apply_user_override_filter=True):
            return [_VisionTool()]

        monkeypatch.setattr(
            "xagent.core.tools.adapters.vibe.factory.ToolFactory.create_all_tools",
            mock_create_all_tools,
        )
        monkeypatch.setattr(
            "xagent.web.tools.config.WebToolConfig.get_vision_model",
            lambda self: None,
        )

        set_user_tool_overrides_hook(
            lambda db, user: {"vision_test_tool": {"enabled": True}}
        )
        try:
            login_response = client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "admin123"},
            )
            assert login_response.status_code == 200
            token = login_response.json()["access_token"]

            response = client.get(
                "/api/tools/available",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert response.status_code == 200
            payload = response.json()

            tool_map = {item["name"]: item for item in payload["tools"]}
            assert "vision_test_tool" in tool_map
            assert tool_map["vision_test_tool"]["status"] == "missing_model"
            assert tool_map["vision_test_tool"]["enabled"] is False
        finally:
            set_user_tool_overrides_hook(None)

    def test_get_available_tools_override_enables_globally_disabled_tool(self):
        """Test that enabled=True override can re-enable a globally disabled tool."""
        from xagent.web.services.tool_credentials import set_user_tool_overrides_hook

        # Step 1: globally disable browser_navigate via admin API
        headers = {"Authorization": f"Bearer {self._login_admin()}"}
        put_resp = client.put(
            "/api/tools/browser_navigate/enabled",
            headers=headers,
            json={"enabled": False},
        )
        assert put_resp.status_code == 200

        # Step 2: set hook to re-enable it
        set_user_tool_overrides_hook(
            lambda db, user: {"browser_navigate": {"enabled": True}}
        )
        try:
            response = client.get(
                "/api/tools/available",
                headers=headers,
            )
            assert response.status_code == 200
            payload = response.json()

            tool_map = {item["name"]: item for item in payload["tools"]}
            assert "browser_navigate" in tool_map
            assert tool_map["browser_navigate"]["enabled"] is True
            assert tool_map["browser_navigate"]["status"] == "available"
        finally:
            set_user_tool_overrides_hook(None)

    def _login_admin(self) -> str:
        login_response = client.post(
            "/api/auth/login", json={"username": "admin", "password": "admin123"}
        )
        assert login_response.status_code == 200
        return login_response.json()["access_token"]


class TestToolsGovernanceAPI:
    @pytest.fixture(autouse=True)
    def setup(self, test_db):
        set_instance_credentials_enabled(True)
        set_credential_fallback_scopes_hook(None)
        ensure_system_initialized()
        yield
        set_credential_fallback_scopes_hook(None)
        set_instance_credentials_enabled(True)

    def _admin_headers(self) -> dict[str, str]:
        login_response = client.post(
            "/api/auth/login", json={"username": "admin", "password": "admin123"}
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def _user_headers(self, username: str) -> dict[str, str]:
        register_response = client.post(
            "/api/auth/register",
            json={
                "username": username,
                "email": f"{username}@example.com",
                "password": "password123",
            },
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

    def test_user_scoped_credentials_put_and_get_masked(self):
        headers = self._user_headers("credential-user")

        put_resp = client.put(
            "/api/tool-credentials/zhipu_web_search?scope=user",
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
            "/api/tool-credentials/zhipu_web_search?scope=user",
            headers=headers,
        )
        assert get_resp.status_code == 200
        payload = get_resp.json()

        assert payload["tool_name"] == "zhipu_web_search"
        assert payload["configured"] is True
        assert payload["fields"]["api_key"]["source"] == "user"
        assert payload["fields"]["api_key"]["is_configured"] is True
        assert "1234" in payload["fields"]["api_key"]["masked"]
        assert (
            "test-secret-zhipu-key-1234" not in payload["fields"]["api_key"]["masked"]
        )

    def test_instance_scoped_credentials_require_admin(self):
        headers = self._admin_headers()
        user_headers = self._user_headers("credential-nonadmin")

        put_resp = client.put(
            "/api/tool-credentials/zhipu_web_search?scope=instance",
            headers=headers,
            json={"credentials": {"api_key": {"value": "instance-secret"}}},
        )
        assert put_resp.status_code == 200

        denied = client.put(
            "/api/tool-credentials/zhipu_web_search?scope=instance",
            headers=user_headers,
            json={"credentials": {"api_key": {"value": "user-should-not-set"}}},
        )
        assert denied.status_code == 403

    def test_standalone_resolver_precedence_user_instance_env(self, monkeypatch):
        from xagent.web.models.database import get_session_local
        from xagent.web.services.tool_credentials import (
            clear_scoped_tool_credential,
            resolve_tool_credential,
            set_scoped_tool_credentials,
        )

        headers = self._admin_headers()
        monkeypatch.setenv("TAVILY_API_KEY", "env-tavily-key")

        put_resp = client.put(
            "/api/tool-credentials/tavily_web_search?scope=instance",
            headers=headers,
            json={"credentials": {"api_key": {"value": "instance-tavily-key"}}},
        )
        assert put_resp.status_code == 200

        SessionLocal = get_session_local()
        db = SessionLocal()
        try:
            set_scoped_tool_credentials(
                db,
                scope_type="user",
                scope_id=42,
                tool_name="tavily_web_search",
                values={"api_key": "user-tavily-key"},
            )
            assert (
                resolve_tool_credential(
                    db,
                    "tavily_web_search",
                    "api_key",
                    user_id=42,
                )
                == "user-tavily-key"
            )
            clear_scoped_tool_credential(
                db,
                scope_type="user",
                scope_id=42,
                tool_name="tavily_web_search",
                field_name="api_key",
            )
            assert (
                resolve_tool_credential(
                    db,
                    "tavily_web_search",
                    "api_key",
                    user_id=42,
                )
                == "instance-tavily-key"
            )
            clear_scoped_tool_credential(
                db,
                scope_type="instance",
                scope_id=None,
                tool_name="tavily_web_search",
                field_name="api_key",
            )
            assert (
                resolve_tool_credential(
                    db,
                    "tavily_web_search",
                    "api_key",
                    user_id=42,
                )
                == "env-tavily-key"
            )
        finally:
            db.close()

    def test_configurable_credentials_env_source_when_not_stored(self, monkeypatch):
        headers = self._admin_headers()
        monkeypatch.setenv("TAVILY_API_KEY", "env-only-tavily-key-5678")

        resp = client.get(
            "/api/tool-credentials/tavily_web_search?scope=instance",
            headers=headers,
        )
        assert resp.status_code == 200
        payload = resp.json()

        assert payload["fields"]["api_key"]["source"] == "env"
        assert payload["fields"]["api_key"]["is_configured"] is True
        assert "5678" in payload["fields"]["api_key"]["masked"]

    def test_scoped_sql_connections_crud_and_precedence(self, monkeypatch):
        headers = self._admin_headers()
        monkeypatch.setenv(
            "XAGENT_EXTERNAL_DB_ANALYTICS",
            "postgresql://env_user:env_pass@localhost:5432/env_db",
        )

        initial = client.get(
            "/api/tool-credentials/sql_query?scope=instance",
            headers=headers,
        )
        assert initial.status_code == 200
        assert initial.json()["fields"]["ANALYTICS"]["source"] == "env"

        upsert = client.put(
            "/api/tool-credentials/sql_query?scope=instance",
            headers=headers,
            json={
                "credentials": {
                    "analytics": {
                        "value": "postgresql://db_user:db_pass@localhost:5432/db_db"
                    }
                }
            },
        )
        assert upsert.status_code == 200

        after_upsert = client.get(
            "/api/tool-credentials/sql_query?scope=instance",
            headers=headers,
        )
        assert after_upsert.status_code == 200
        field = after_upsert.json()["fields"]["ANALYTICS"]
        assert field["source"] == "instance"
        assert "db_pass" not in field["masked"]

        delete_resp = client.delete(
            "/api/tool-credentials/sql_query/analytics?scope=instance",
            headers=headers,
        )
        assert delete_resp.status_code == 200
        assert delete_resp.json()["fields"]["ANALYTICS"]["source"] == "env"

    def test_sql_connection_rejects_unsupported_scheme(self):
        headers = self._admin_headers()

        upsert = client.put(
            "/api/tool-credentials/sql_query?scope=user",
            headers=headers,
            json={"credentials": {"analytics": {"value": "redis://localhost:6379/0"}}},
        )

        assert upsert.status_code == 400
        assert "Unsupported SQLAlchemy URL scheme" in upsert.json()["detail"]

    def test_sql_connection_accepts_duckdb_scheme(self):
        headers = self._admin_headers()

        upsert = client.put(
            "/api/tool-credentials/sql_query?scope=user",
            headers=headers,
            json={
                "credentials": {
                    "local_warehouse": {"value": "duckdb:///tmp/warehouse.duckdb"}
                }
            },
        )

        assert upsert.status_code == 200
        fields = upsert.json()["fields"]
        assert fields["LOCAL_WAREHOUSE"]["source"] == "user"
        assert fields["LOCAL_WAREHOUSE"]["is_configured"] is True

    def test_sql_connections_are_user_scoped(self):
        user1_headers = self._user_headers("user1")
        user2_headers = self._user_headers("user2")

        user1_upsert = client.put(
            "/api/tool-credentials/sql_query?scope=user",
            headers=user1_headers,
            json={
                "credentials": {
                    "analytics": {
                        "value": "postgresql://user1:pass1@localhost:5432/user1_db"
                    }
                }
            },
        )
        assert user1_upsert.status_code == 200

        user2_initial = client.get(
            "/api/tool-credentials/sql_query?scope=user",
            headers=user2_headers,
        )
        assert user2_initial.status_code == 200
        assert user2_initial.json()["fields"] == {}

        user2_upsert = client.put(
            "/api/tool-credentials/sql_query?scope=user",
            headers=user2_headers,
            json={
                "credentials": {
                    "analytics": {
                        "value": "postgresql://user2:pass2@localhost:5432/user2_db"
                    }
                }
            },
        )
        assert user2_upsert.status_code == 200

        user1_items = client.get(
            "/api/tool-credentials/sql_query?scope=user",
            headers=user1_headers,
        ).json()["fields"]
        user2_items = client.get(
            "/api/tool-credentials/sql_query?scope=user",
            headers=user2_headers,
        ).json()["fields"]

        assert user1_items["ANALYTICS"]["source"] == "user"
        assert user2_items["ANALYTICS"]["source"] == "user"
        assert user1_items["ANALYTICS"]["masked"] != user2_items["ANALYTICS"]["masked"]

        user1_delete = client.delete(
            "/api/tool-credentials/sql_query/analytics?scope=user",
            headers=user1_headers,
        )
        assert user1_delete.status_code == 200

        user1_after_delete = client.get(
            "/api/tool-credentials/sql_query?scope=user",
            headers=user1_headers,
        )
        user2_after_delete = client.get(
            "/api/tool-credentials/sql_query?scope=user",
            headers=user2_headers,
        )
        assert user1_after_delete.status_code == 200
        assert user2_after_delete.status_code == 200
        assert user1_after_delete.json()["fields"] == {}
        remaining_user2 = user2_after_delete.json()["fields"]
        assert remaining_user2["ANALYTICS"]["source"] == "user"

    def test_old_credential_endpoints_are_removed(self):
        user_headers = self._user_headers("nonadmin")

        configurable_resp = client.get("/api/tools/configurable", headers=user_headers)
        credential_resp = client.get(
            "/api/tools/zhipu_web_search/credentials", headers=user_headers
        )

        assert configurable_resp.status_code == 404
        assert credential_resp.status_code == 404

    def test_instance_scoped_credentials_upsert_single_row(self):
        from xagent.web.models.database import get_session_local
        from xagent.web.models.tool_config import ScopedToolCredential

        headers = self._admin_headers()

        for value in ("first-instance-secret", "second-instance-secret"):
            response = client.put(
                "/api/tool-credentials/tavily_web_search?scope=instance",
                headers=headers,
                json={"credentials": {"api_key": {"value": value}}},
            )
            assert response.status_code == 200

        SessionLocal = get_session_local()
        db = SessionLocal()
        try:
            rows = (
                db.query(ScopedToolCredential)
                .filter(
                    ScopedToolCredential.scope_type == "instance",
                    ScopedToolCredential.scope_id.is_(None),
                    ScopedToolCredential.tool_name == "tavily_web_search",
                    ScopedToolCredential.field_name == "api_key",
                )
                .all()
            )
        finally:
            db.close()

        assert len(rows) == 1

    def test_core_tool_credentials_reject_team_scope(self):
        headers = self._admin_headers()

        response = client.get(
            "/api/tool-credentials/tavily_web_search?scope=team",
            headers=headers,
        )

        assert response.status_code == 422


def test_user_tool_overrides_hook_noop_by_default():
    """Without a hook set, get_user_tool_overrides returns an empty dict."""
    from xagent.web.services.tool_credentials import get_user_tool_overrides

    result = get_user_tool_overrides(db=None, user=None)
    assert result == {}


def test_user_tool_overrides_hook_returns_injected_data():
    """When a hook is set, it returns the hook's result."""
    from xagent.web.services.tool_credentials import (
        get_user_tool_overrides,
        set_user_tool_overrides_hook,
    )

    def my_hook(db, user):
        return {
            "calculator": {"enabled": False},
            "web_search": {"config": {"api_key": "x"}},
        }

    set_user_tool_overrides_hook(my_hook)
    try:
        result = get_user_tool_overrides(db=None, user=None)
        assert result["calculator"]["enabled"] is False
        assert result["web_search"]["config"] == {"api_key": "x"}
        assert "nonexistent" not in result
    finally:
        set_user_tool_overrides_hook(None)


def test_user_tool_overrides_hook_reset_to_none():
    """Setting hook to None restores default empty behavior."""
    from xagent.web.services.tool_credentials import (
        get_user_tool_overrides,
        set_user_tool_overrides_hook,
    )

    set_user_tool_overrides_hook(lambda db, user: {"test": {"enabled": True}})
    set_user_tool_overrides_hook(None)
    result = get_user_tool_overrides(db=None, user=None)
    assert result == {}


class TestWebToolConfigUserOverride:
    """Verify WebToolConfig.get_user_tool_overrides() resolves user correctly."""

    def test_get_tool_credential_uses_current_user_scope(self):
        """Personal tool credentials must be visible when creating runtime tools."""
        from unittest.mock import MagicMock

        from xagent.web.models.database import get_engine, get_session_local, init_db
        from xagent.web.services.tool_credentials import (
            set_scoped_tool_credentials,
        )
        from xagent.web.tools.config import WebToolConfig

        temp_dir = tempfile.mkdtemp()
        temp_db_path = os.path.join(temp_dir, "test.db")
        init_db(db_url=f"sqlite:///{temp_db_path}")
        engine = get_engine()
        SessionLocal = get_session_local()
        db = SessionLocal()
        try:
            set_scoped_tool_credentials(
                db,
                scope_type="user",
                scope_id=42,
                tool_name="web_search",
                values={"api_key": "user-google-key", "cse_id": "user-google-cse"},
            )

            cfg = WebToolConfig(
                db=db,
                request=MagicMock(user=MagicMock(id=42)),
                user_id=42,
                workspace_config={"base_dir": "/tmp", "task_id": "test"},
            )

            assert cfg.get_tool_credential("web_search", "api_key") == "user-google-key"
            assert cfg.get_tool_credential("web_search", "cse_id") == "user-google-cse"
        finally:
            db.close()
            Base.metadata.drop_all(bind=engine)
            shutil.rmtree(temp_dir)

    def test_get_sql_connections_uses_registered_fallback_scope(self):
        """Shared credentials from a registered scope must be visible at runtime."""
        from unittest.mock import MagicMock

        from xagent.web.models.database import get_engine, get_session_local, init_db
        from xagent.web.services.tool_credentials import (
            CredentialScopeRef,
            set_credential_fallback_scopes_hook,
            set_scoped_tool_credentials,
        )
        from xagent.web.tools.config import WebToolConfig

        temp_dir = tempfile.mkdtemp()
        temp_db_path = os.path.join(temp_dir, "test.db")
        init_db(db_url=f"sqlite:///{temp_db_path}")
        engine = get_engine()
        SessionLocal = get_session_local()
        db = SessionLocal()
        set_credential_fallback_scopes_hook(
            lambda _db, user: [
                CredentialScopeRef(
                    "shared", getattr(user, "shared_scope_id", None), "Shared"
                )
            ]
        )
        try:
            set_scoped_tool_credentials(
                db,
                scope_type="shared",
                scope_id=7,
                tool_name="sql_query",
                values={"analytics": "sqlite:///analytics.db"},
            )

            cfg = WebToolConfig(
                db=db,
                request=MagicMock(user=MagicMock(id=42, shared_scope_id=7)),
                user_id=42,
                workspace_config={"base_dir": "/tmp", "task_id": "test"},
            )

            assert cfg.get_sql_connections()["ANALYTICS"] == "sqlite:///analytics.db"
        finally:
            set_credential_fallback_scopes_hook(None)
            db.close()
            Base.metadata.drop_all(bind=engine)
            shutil.rmtree(temp_dir)

    def test_registered_fallback_scope_preserves_instance_credential_fallback(self):
        """Shared scopes should precede, not replace, instance fallback credentials."""
        from unittest.mock import MagicMock

        from xagent.web.models.database import get_engine, get_session_local, init_db
        from xagent.web.services.tool_credentials import (
            CredentialScopeRef,
            resolve_tool_credential,
            set_credential_fallback_scopes_hook,
            set_scoped_tool_credentials,
        )

        temp_dir = tempfile.mkdtemp()
        temp_db_path = os.path.join(temp_dir, "test.db")
        init_db(db_url=f"sqlite:///{temp_db_path}")
        engine = get_engine()
        SessionLocal = get_session_local()
        db = SessionLocal()
        set_credential_fallback_scopes_hook(
            lambda _db, user: [
                CredentialScopeRef(
                    "shared", getattr(user, "shared_scope_id", None), "Shared"
                )
            ]
        )
        try:
            set_scoped_tool_credentials(
                db,
                scope_type="instance",
                scope_id=None,
                tool_name="tavily_web_search",
                values={"api_key": "instance-tavily-key"},
            )

            assert (
                resolve_tool_credential(
                    db,
                    "tavily_web_search",
                    "api_key",
                    user_id=42,
                    user=MagicMock(id=42, shared_scope_id=7),
                )
                == "instance-tavily-key"
            )
        finally:
            set_credential_fallback_scopes_hook(None)
            db.close()
            Base.metadata.drop_all(bind=engine)
            shutil.rmtree(temp_dir)

    def test_sql_connection_map_preserves_instance_entries_with_registered_fallback_scope(
        self,
    ):
        """Instance SQL connections remain visible when shared scopes have no matching row."""
        from unittest.mock import MagicMock

        from xagent.web.models.database import get_engine, get_session_local, init_db
        from xagent.web.services.tool_credentials import (
            CredentialScopeRef,
            get_sql_connection_map,
            set_credential_fallback_scopes_hook,
            set_scoped_tool_credentials,
        )

        temp_dir = tempfile.mkdtemp()
        temp_db_path = os.path.join(temp_dir, "test.db")
        init_db(db_url=f"sqlite:///{temp_db_path}")
        engine = get_engine()
        SessionLocal = get_session_local()
        db = SessionLocal()
        set_credential_fallback_scopes_hook(
            lambda _db, user: [
                CredentialScopeRef(
                    "shared", getattr(user, "shared_scope_id", None), "Shared"
                )
            ]
        )
        try:
            set_scoped_tool_credentials(
                db,
                scope_type="instance",
                scope_id=None,
                tool_name="sql_query",
                values={"analytics": "sqlite:///instance-analytics.db"},
            )

            connections = get_sql_connection_map(
                db,
                42,
                user=MagicMock(id=42, shared_scope_id=7),
            )

            assert connections["ANALYTICS"] == "sqlite:///instance-analytics.db"
        finally:
            set_credential_fallback_scopes_hook(None)
            db.close()
            Base.metadata.drop_all(bind=engine)
            shutil.rmtree(temp_dir)

    def test_explicit_user_param_takes_priority(self):
        """When user keyword arg is passed, it is used even when request has no .user."""
        from unittest.mock import MagicMock

        from xagent.web.services.tool_credentials import set_user_tool_overrides_hook
        from xagent.web.tools.config import WebToolConfig

        def _hook(db, user):
            return {"browser_navigate": {"enabled": False}}

        set_user_tool_overrides_hook(_hook)
        try:
            # Simulate TaskCreateRequest: no .user attribute
            request_without_user = MagicMock()
            del request_without_user.user

            cfg = WebToolConfig(
                db=MagicMock(),
                request=request_without_user,
                user_id=42,
                user=MagicMock(id=42),  # explicit user
                workspace_config={"base_dir": "/tmp", "task_id": "test"},
            )
            assert cfg.get_user_tool_overrides() == {
                "browser_navigate": {"enabled": False}
            }
        finally:
            set_user_tool_overrides_hook(None)

    def test_falls_back_to_request_user_when_explicit_not_given(self):
        """Without explicit user, request.user is used (existing behavior)."""
        from unittest.mock import MagicMock

        from xagent.web.services.tool_credentials import set_user_tool_overrides_hook
        from xagent.web.tools.config import WebToolConfig

        def _hook(db, user):
            return {"browser_navigate": {"enabled": False}}

        set_user_tool_overrides_hook(_hook)
        try:
            mock_user = MagicMock(id=42)
            request = MagicMock(user=mock_user)

            cfg = WebToolConfig(
                db=MagicMock(),
                request=request,
                user_id=42,
                workspace_config={"base_dir": "/tmp", "task_id": "test"},
            )
            assert cfg.get_user_tool_overrides() == {
                "browser_navigate": {"enabled": False}
            }
        finally:
            set_user_tool_overrides_hook(None)

    def test_returns_empty_when_no_user_at_all(self):
        """When neither explicit user nor request.user is available, returns {}."""
        from unittest.mock import MagicMock

        from xagent.web.services.tool_credentials import set_user_tool_overrides_hook
        from xagent.web.tools.config import WebToolConfig

        def _hook(db, user):
            return {"browser_navigate": {"enabled": False}}

        set_user_tool_overrides_hook(_hook)
        try:
            request_without_user = MagicMock()
            del request_without_user.user

            cfg = WebToolConfig(
                db=MagicMock(),
                request=request_without_user,
                user_id=42,
                workspace_config={"base_dir": "/tmp", "task_id": "test"},
            )
            assert cfg.get_user_tool_overrides() == {}
        finally:
            set_user_tool_overrides_hook(None)

    @pytest.mark.asyncio
    async def test_create_all_tools_filters_disabled_when_user_is_explicit(self):
        """End-to-end: ToolFactory filters tools disabled by per-user hook
        even when request has no .user but explicit user is provided."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from xagent.core.tools.adapters.vibe.factory import ToolFactory
        from xagent.web.services.tool_credentials import set_user_tool_overrides_hook
        from xagent.web.tools.config import WebToolConfig

        def _hook(db, user):
            return {"browser_navigate": {"enabled": False}}

        set_user_tool_overrides_hook(_hook)
        try:
            # Simulate TaskCreateRequest: no .user
            request_without_user = MagicMock()
            del request_without_user.user

            cfg = WebToolConfig(
                db=MagicMock(),
                request=request_without_user,
                user=MagicMock(id=42),  # explicit user
                user_id=42,
                workspace_config={"base_dir": "/tmp", "task_id": "test"},
            )

            # Create mock tools with string .name attributes
            tool_browser = MagicMock()
            tool_browser.name = "browser_navigate"
            tool_calc = MagicMock()
            tool_calc.name = "calculator"

            with patch(
                "xagent.core.tools.adapters.vibe.factory.ToolRegistry.create_registered_tools",
                AsyncMock(return_value=[tool_browser, tool_calc]),
            ):
                result = await ToolFactory.create_all_tools(cfg)

            tool_names = [t.name for t in result]
            assert "browser_navigate" not in tool_names, (
                "Disabled tool was NOT filtered from create_all_tools"
            )
            assert "calculator" in tool_names, "Non-disabled tool should remain"
        finally:
            set_user_tool_overrides_hook(None)

    @pytest.mark.asyncio
    async def test_create_all_tools_skips_filter_when_no_user_at_all(self):
        """When neither explicit user nor request.user is set,
        get_user_tool_overrides() returns {} and ToolFactory filtering is skipped.
        This is the safe fallback — no user means no per-user policy can apply."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from xagent.core.tools.adapters.vibe.factory import ToolFactory
        from xagent.web.services.tool_credentials import set_user_tool_overrides_hook
        from xagent.web.tools.config import WebToolConfig

        def _hook(db, user):
            return {"browser_navigate": {"enabled": False}}

        set_user_tool_overrides_hook(_hook)
        try:
            request_without_user = MagicMock()
            del request_without_user.user

            cfg = WebToolConfig(
                db=MagicMock(),
                request=request_without_user,
                user_id=42,
                # No explicit user passed — this is the pre-fix bug path
                workspace_config={"base_dir": "/tmp", "task_id": "test"},
            )

            tool_browser = MagicMock()
            tool_browser.name = "browser_navigate"
            tool_calc = MagicMock()
            tool_calc.name = "calculator"

            with patch(
                "xagent.core.tools.adapters.vibe.factory.ToolRegistry.create_registered_tools",
                AsyncMock(return_value=[tool_browser, tool_calc]),
            ):
                result = await ToolFactory.create_all_tools(cfg)

            tool_names = [t.name for t in result]
            # Without explicit user, overrides are {} and filtering is skipped
            assert "browser_navigate" in tool_names, (
                "No filtering when no user (existing behavior)"
            )
            assert "calculator" in tool_names
        finally:
            set_user_tool_overrides_hook(None)

    @pytest.mark.asyncio
    async def test_create_all_tools_keeps_disabled_when_filter_false(self):
        """When apply_user_override_filter=False, disabled tools remain in the list.

        This is the display-layer path: tools are visible so they can be
        shown as ``enabled=False`` in the UI.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        from xagent.core.tools.adapters.vibe.factory import ToolFactory
        from xagent.web.services.tool_credentials import set_user_tool_overrides_hook
        from xagent.web.tools.config import WebToolConfig

        def _hook(db, user):
            return {"browser_navigate": {"enabled": False}}

        set_user_tool_overrides_hook(_hook)
        try:
            request_without_user = MagicMock()
            del request_without_user.user

            cfg = WebToolConfig(
                db=MagicMock(),
                request=request_without_user,
                user=MagicMock(id=42),
                user_id=42,
                workspace_config={"base_dir": "/tmp", "task_id": "test"},
            )

            tool_browser = MagicMock()
            tool_browser.name = "browser_navigate"
            tool_calc = MagicMock()
            tool_calc.name = "calculator"

            with patch(
                "xagent.core.tools.adapters.vibe.factory.ToolRegistry.create_registered_tools",
                AsyncMock(return_value=[tool_browser, tool_calc]),
            ):
                result = await ToolFactory.create_all_tools(
                    cfg, apply_user_override_filter=False
                )

            tool_names = [t.name for t in result]
            assert "browser_navigate" in tool_names, (
                "Disabled tool should remain when apply_user_override_filter=False"
            )
            assert "calculator" in tool_names
        finally:
            set_user_tool_overrides_hook(None)


class TestWebToolConfigCustomApi:
    """Verify WebToolConfig.get_custom_api_configs() exposes the body field
    so that POST custom-api tools actually send their configured payload."""

    def test_get_custom_api_configs_includes_body(self):
        """Regression test for production bug (Task 898 / Agent 170):

        A user had a Custom API entry with method=POST and a JSON body
        template, but their POST requests went out with empty bodies.
        Root cause: WebToolConfig.get_custom_api_configs() built its
        config dict without the `body` field, so by the time the tool
        was constructed the body template was already gone.

        This test mirrors the actual production record (Post_HelloAPI)
        and asserts the body field survives the DB -> dict translation.
        """
        from unittest.mock import MagicMock

        from xagent.web.tools.config import WebToolConfig

        # Mirror of production custom_apis row (id=5, name=Post_HelloAPI)
        api = MagicMock()
        api.name = "Post_HelloAPI"
        api.description = None
        api.url = "https://helloapi-u6nc.onrender.com/"
        api.method = "POST"
        api.headers = {}
        api.body = '{\n  "message": "example message"\n}'
        api.env = None

        user_api = MagicMock()
        user_api.custom_api = api

        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = [user_api]

        cfg = WebToolConfig(
            db=db,
            request=MagicMock(),
            user_id=33,
            workspace_config={"base_dir": "/tmp", "task_id": "test"},
        )
        configs = cfg.get_custom_api_configs()

        assert len(configs) == 1
        config = configs[0]
        assert config["name"] == "Post_HelloAPI"
        assert config["method"] == "POST"
        # The crucial assertion: body must propagate
        assert config["body"] == '{\n  "message": "example message"\n}'

    def test_get_custom_api_configs_body_optional(self):
        """When the user has not configured a body (e.g. GET-only tools),
        the body field should still be present in the config but None,
        so downstream code can rely on the key existing."""
        from unittest.mock import MagicMock

        from xagent.web.tools.config import WebToolConfig

        api = MagicMock()
        api.name = "Get_HiAPI"
        api.description = None
        api.url = "https://helloapi-u6nc.onrender.com/"
        api.method = "GET"
        api.headers = {"message": ""}
        api.body = None
        api.env = None

        user_api = MagicMock()
        user_api.custom_api = api

        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = [user_api]

        cfg = WebToolConfig(
            db=db,
            request=MagicMock(),
            user_id=33,
            workspace_config={"base_dir": "/tmp", "task_id": "test"},
        )
        configs = cfg.get_custom_api_configs()

        assert len(configs) == 1
        assert "body" in configs[0]
        assert configs[0]["body"] is None


class TestWebToolConfigMCPAuth:
    @pytest.mark.asyncio
    async def test_get_mcp_server_configs_includes_concurrency_config(self):
        from unittest.mock import MagicMock

        from xagent.web.tools.config import WebToolConfig

        server = MagicMock()
        server.name = "local"
        server.transport = "stdio"
        server.description = "Local MCP"
        server.command = "npx"
        server.args = ["-y", "@modelcontextprotocol/server-everything"]
        server.env = None
        server.cwd = None
        server.managed = "external"
        server.concurrency_safe = True
        server.concurrent_tools = ["echo", "get_sum"]

        db = MagicMock()
        db.query.return_value.join.return_value.filter.return_value.all.return_value = [
            server
        ]

        cfg = WebToolConfig(
            db=db,
            request=MagicMock(),
            user_id=1,
            workspace_config={"base_dir": "/tmp", "task_id": "test"},
        )
        configs = await cfg.get_mcp_server_configs()

        assert len(configs) == 1
        assert configs[0]["name"] == "local"
        assert configs[0]["config"]["concurrency_safe"] is True
        assert configs[0]["config"]["concurrent_tools"] == ["echo", "get_sum"]

    @pytest.mark.asyncio
    async def test_get_mcp_server_configs_maps_bearer_auth_to_headers(self):
        from unittest.mock import MagicMock

        from xagent.web.tools.config import WebToolConfig

        server = MagicMock()
        server.name = "local"
        server.transport = "streamable_http"
        server.description = "Local MCP"
        server.url = "http://127.0.0.1:18000/mcp"
        server.headers = {"X-Test": "true"}
        server.auth = {"type": "bearer", "bearer_token": "test-token-123"}
        server.managed = "external"
        server._decrypt_auth_config.side_effect = lambda auth: auth
        server._merge_auth_headers.side_effect = lambda headers, auth: {
            **(headers or {}),
            "Authorization": f"Bearer {auth['bearer_token']}",
        }

        db = MagicMock()
        db.query.return_value.join.return_value.filter.return_value.all.return_value = [
            server
        ]

        cfg = WebToolConfig(
            db=db,
            request=MagicMock(),
            user_id=1,
            workspace_config={"base_dir": "/tmp", "task_id": "test"},
        )
        configs = await cfg.get_mcp_server_configs()

        assert len(configs) == 1
        assert configs[0]["name"] == "local"
        assert configs[0]["config"]["url"] == "http://127.0.0.1:18000/mcp"
        assert configs[0]["config"]["headers"]["X-Test"] == "true"
        assert (
            configs[0]["config"]["headers"]["Authorization"] == "Bearer test-token-123"
        )
