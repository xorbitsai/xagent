"""
Tests for Tools API endpoints.

This module tests the /api/tools endpoints, including the /available endpoint
which lists all tools that can be used by agents.
"""

import tempfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from xagent.web.api.auth import auth_router
from xagent.web.api.tools import _create_tool_info, tools_router
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


def test_sound_effect_tool_requires_sound_effect_model() -> None:
    class Tool:
        name = "generate_sound_effect"
        description = ""

    unavailable = _create_tool_info(
        Tool(),
        "audio",
    )
    available = _create_tool_info(
        Tool(),
        "audio",
        sound_effect_models={"sfx": object()},
    )

    assert unavailable["enabled"] is False
    assert unavailable["status"] == "missing_model"
    assert available["enabled"] is True
    assert available["status"] == "available"


def test_music_tool_requires_music_model() -> None:
    class Tool:
        name = "generate_music"
        description = ""

    unavailable = _create_tool_info(Tool(), "audio")
    available = _create_tool_info(
        Tool(),
        "audio",
        music_models={"music": object()},
    )

    assert unavailable["enabled"] is False
    assert unavailable["status"] == "missing_model"
    assert available["enabled"] is True
    assert available["status"] == "available"


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
        server.to_connection_dict.side_effect = lambda: {
            "name": server.name,
            "transport": server.transport,
            "url": server.url,
            "headers": server._merge_auth_headers(
                server.headers,
                server._decrypt_auth_config(server.auth),
            ),
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


class TestAvailableToolsSandboxLifecycle:
    """Sandbox lifecycle handling in /api/tools/available."""

    @pytest.fixture(autouse=True)
    def setup(self, test_db):
        ensure_system_initialized()
        yield

    def _login(self) -> str:
        login_response = client.post(
            "/api/auth/login", json={"username": "admin", "password": "admin123"}
        )
        assert login_response.status_code == 200
        return login_response.json()["access_token"]

    def test_reclaimed_provider_before_attach_lists_without_sandbox(
        self, monkeypatch
    ) -> None:
        """attach() returning False (provider reclaimed between creation and
        attach) must drop the stale reference and still serve the listing,
        without releasing what was never attached."""
        from unittest.mock import AsyncMock, MagicMock

        sandbox_manager = MagicMock()
        sandbox_manager.get_or_create_lease_provider = AsyncMock(
            return_value=MagicMock()
        )
        sandbox_manager.attach = AsyncMock(return_value=False)
        sandbox_manager.release = AsyncMock()
        monkeypatch.setattr(
            "xagent.web.sandbox_manager.get_sandbox_manager",
            lambda: sandbox_manager,
        )

        token = self._login()
        response = client.get(
            "/api/tools/available", headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        assert response.json()["count"] > 0
        sandbox_manager.attach.assert_awaited_once()
        sandbox_manager.release.assert_not_awaited()

    def test_attached_sandbox_is_released_after_listing(self, monkeypatch) -> None:
        """A successfully attached sandbox is released exactly once."""
        from unittest.mock import AsyncMock, MagicMock

        provider = MagicMock()
        provider.exec = AsyncMock()
        provider.primary_sandbox.exec = AsyncMock()
        sandbox_manager = MagicMock()
        sandbox_manager.get_or_create_lease_provider = AsyncMock(return_value=provider)
        sandbox_manager.attach = AsyncMock(return_value=True)
        sandbox_manager.release = AsyncMock()
        monkeypatch.setattr(
            "xagent.web.sandbox_manager.get_sandbox_manager",
            lambda: sandbox_manager,
        )

        token = self._login()
        response = client.get(
            "/api/tools/available", headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        sandbox_manager.release.assert_awaited_once()
        release_args = sandbox_manager.release.await_args.args
        assert release_args[0] == "tools"
