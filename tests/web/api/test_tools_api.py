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
