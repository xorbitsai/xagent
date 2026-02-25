"""Test templates API endpoints"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from xagent.web.api.auth import auth_router
from xagent.web.api.templates import router as templates_router
from xagent.web.models.database import Base, get_db, get_engine


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
test_app.include_router(templates_router)
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
    from xagent.web.models.database import init_db

    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test.db")
    SQLALCHEMY_DATABASE_URL = f"sqlite:///{temp_db_path}"

    with patch("xagent.web.models.database.try_upgrade_db"):
        init_db(db_url=SQLALCHEMY_DATABASE_URL)

    engine = get_engine()

    yield temp_dir

    Base.metadata.drop_all(bind=engine)
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def templates_dir(tmp_path):
    """Create temporary templates directory with sample templates"""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()

    # Create sample templates
    template1 = templates_dir / "customer_support.yaml"
    template1.write_text(
        """
id: customer_support
name: Customer Support Agent
category: Support
tags:
  - support
  - customer
descriptions:
  en: Professional customer support assistant
  zh: 专业的客服助手
author: xAgent
version: "1.0"

agent_config:
  instructions: |
    You are a customer support assistant.
  skills:
    - product_knowledge
  tool_categories:
    - web_search
"""
    )

    template2 = templates_dir / "sales_assistant.yaml"
    template2.write_text(
        """
id: sales_assistant
name: Sales Assistant
category: Sales
tags:
  - sales
  - marketing
descriptions:
  en: Professional sales assistant
  zh: 专业的销售助手
author: xAgent
version: "1.0"

agent_config:
  instructions: |
    You are a sales assistant.
  skills:
    - sales_techniques
  tool_categories:
    - file_operations
"""
    )

    return templates_dir


@pytest.fixture(scope="function")
def template_manager(templates_dir):
    """Create TemplateManager fixture"""
    from xagent.templates.manager import TemplateManager

    manager = TemplateManager(templates_root=templates_dir)
    return manager


@pytest.fixture(scope="function")
def mock_app_state(template_manager):
    """Mock app.state.template_manager"""
    # Initialize the manager
    import asyncio

    asyncio.run(template_manager.initialize())

    # Create mock app state
    mock_state = MagicMock()
    mock_state.template_manager = template_manager
    return mock_state


@pytest.fixture(scope="function")
def admin_user(test_db):
    """Create admin user for testing"""
    ensure_system_initialized()

    db = next(get_db())
    from xagent.web.models.user import User

    admin = db.query(User).filter(User.username == "admin").first()
    assert admin is not None
    db.close()
    return {"id": admin.id, "username": admin.username}


@pytest.fixture(scope="function")
def admin_headers(admin_user):
    """Authentication headers for admin user"""
    response = client.post(
        "/api/auth/login",
        json={"username": admin_user["username"], "password": "admin123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("success") is True

    return {"Authorization": f"Bearer {data['access_token']}"}


class TestTemplatesAPI:
    """测试 Templates API"""

    def test_list_templates_success(self, mock_app_state, admin_headers):
        """测试成功获取模板列表"""
        with patch.object(client.app, "state", mock_app_state):
            response = client.get("/api/templates/", headers=admin_headers)

            assert response.status_code == 200
            templates = response.json()
            assert isinstance(templates, list)
            assert len(templates) == 2

            template_ids = [t["id"] for t in templates]
            assert "customer_support" in template_ids
            assert "sales_assistant" in template_ids

    def test_list_templates_with_stats(self, mock_app_state, admin_headers):
        """测试模板列表包含统计数据"""
        with patch.object(client.app, "state", mock_app_state):
            response = client.get("/api/templates/", headers=admin_headers)

            assert response.status_code == 200
            templates = response.json()

            # 检查统计数据字段
            template = templates[0]
            assert "views" in template
            assert "likes" in template
            assert "used_count" in template
            assert template["views"] == 0
            assert template["likes"] == 0
            assert template["used_count"] == 0

    def test_get_template_detail(self, mock_app_state, admin_headers):
        """测试获取模板详情"""
        with patch.object(client.app, "state", mock_app_state):
            response = client.get(
                "/api/templates/customer_support", headers=admin_headers
            )

            assert response.status_code == 200
            template = response.json()

            assert template["id"] == "customer_support"
            assert template["name"] == "Customer Support Agent"
            assert template["category"] == "Support"
            assert "agent_config" in template

            # 检查 agent_config
            agent_config = template["agent_config"]
            assert "instructions" in agent_config
            assert "customer support assistant" in agent_config["instructions"].lower()
            assert agent_config["skills"] == ["product_knowledge"]
            assert agent_config["tool_categories"] == ["web_search"]

    def test_get_template_not_found(self, mock_app_state, admin_headers):
        """测试获取不存在的模板"""
        with patch.object(client.app, "state", mock_app_state):
            response = client.get("/api/templates/nonexistent", headers=admin_headers)

            assert response.status_code == 404

    def test_like_template(self, mock_app_state, admin_headers):
        """测试点赞模板"""
        with patch.object(client.app, "state", mock_app_state):
            # 第一次点赞
            response = client.post(
                "/api/templates/customer_support/like", headers=admin_headers
            )

            assert response.status_code == 200
            result = response.json()
            assert result["liked"] is True
            assert result["likes"] == 1

            # 获取模板详情验证点赞数
            response = client.get(
                "/api/templates/customer_support", headers=admin_headers
            )
            assert response.json()["likes"] == 1

    def test_use_template(self, mock_app_state, admin_headers):
        """测试使用模板"""
        with patch.object(client.app, "state", mock_app_state):
            response = client.post(
                "/api/templates/customer_support/use", headers=admin_headers
            )

            assert response.status_code == 200
            result = response.json()
            assert result["template_id"] == "customer_support"
            assert result["used_count"] == 1
            assert "message" in result

            # 获取模板详情验证使用次数
            response = client.get(
                "/api/templates/customer_support", headers=admin_headers
            )
            assert response.json()["used_count"] == 1

    def test_get_template_increments_views(self, mock_app_state, admin_headers):
        """测试获取模板详情增加访问次数"""
        with patch.object(client.app, "state", mock_app_state):
            # 第一次访问
            response = client.get(
                "/api/templates/customer_support", headers=admin_headers
            )
            assert response.status_code == 200
            assert response.json()["views"] == 1

            # 第二次访问
            response = client.get(
                "/api/templates/customer_support", headers=admin_headers
            )
            assert response.status_code == 200
            assert response.json()["views"] == 2

    def test_unauthorized_access(self, mock_app_state):
        """测试未授权访问"""
        with patch.object(client.app, "state", mock_app_state):
            response = client.get("/api/templates/")

            # 未授权访问返回 403 Forbidden 或 401 Unauthorized
            assert response.status_code in [401, 403]

    def test_template_data_structure(self, mock_app_state, admin_headers):
        """测试模板数据结构完整性"""
        with patch.object(client.app, "state", mock_app_state):
            response = client.get("/api/templates/", headers=admin_headers)

            assert response.status_code == 200
            templates = response.json()

            template = templates[0]

            # 检查必需字段
            required_fields = [
                "id",
                "name",
                "category",
                "featured",
                "description",
                "tags",
                "author",
                "version",
                "views",
                "likes",
                "used_count",
            ]
            for field in required_fields:
                assert field in template, f"Missing field: {field}"
