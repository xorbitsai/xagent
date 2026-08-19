"""Test templates API endpoints"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError

from xagent.web.api.auth import auth_router, hash_password
from xagent.web.api.templates import (
    get_or_create_template_stats,
    increment_template_likes,
    increment_template_used_count,
)
from xagent.web.api.templates import router as templates_router
from xagent.web.models.agent import Agent, AgentOrigin, AgentStatus
from xagent.web.models.database import Base, get_db, get_engine
from xagent.web.models.template_stats import TemplateStats, UserTemplateRelation
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce, WorkforceAgent


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
            "/api/auth/setup-admin",
            json={
                "username": "admin",
                "email": "admin@example.com",
                "password": "admin123",
            },
        )
        assert setup_response.status_code == 200
        assert setup_response.json().get("success") is True


def create_user_headers(username: str, password: str = "user123") -> dict[str, str]:
    db = next(get_db())
    user = User(username=username, password_hash=hash_password(password))
    db.add(user)
    db.commit()
    db.close()

    response = client.post(
        "/api/auth/login",
        json={"username": username, "password": password},
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("success") is True
    return {"Authorization": f"Bearer {data['access_token']}"}


@pytest.fixture(scope="function")
def test_db():
    """Create test database"""
    from xagent.web.models.database import init_db

    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test.db")
    SQLALCHEMY_DATABASE_URL = f"sqlite:///{temp_db_path}"

    # Note: Previously mocked try_upgrade_db to skip db migrations.
    # For new databases, try_upgrade_db only stamps the latest revision,
    # which is safe for tests and provides better coverage.
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
author: Xagent
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
sample_prompts:
  en:
  - title: Draft an outreach email
    prompt: 'Draft an outreach email for contact.'
    highlights:
    - contact
  - title: Summarise a call
    prompt: Summarise our last call and suggest next steps.
    highlights: []
  zh:
  - title: 起草外联邮件
    prompt: 为 联系人 起草一封外联邮件。
    highlights:
    - 联系人
  - title: 总结通话
    prompt: 总结我们上次的通话并给出下一步建议。
    highlights: []
author: Xagent
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
def workforce_templates_dir(tmp_path):
    """A separate templates directory (not shared with `templates_dir`,
    whose exact template count other tests assert on) holding a
    workforce-type template plus the two single-agent templates its
    `workforce_config.agents[].template_id` entries reference.
    """
    templates_dir = tmp_path / "workforce_templates"
    templates_dir.mkdir()

    (templates_dir / "ga_analyzer.yaml").write_text(
        """
id: ga_analyzer
name: GA Analyzer
category: Marketing
descriptions:
  en: Explains GA4 trends.
author: Xagent
version: "1.0"

agent_config:
  instructions: |
    You are the GA Analyzer.
  skills: []
  tool_categories: []
"""
    )
    (templates_dir / "ads_recommendation.yaml").write_text(
        """
id: ads_recommendation
name: Ads Recommendation
category: Marketing
descriptions:
  en: Recommends ad optimizations.
author: Xagent
version: "1.0"

agent_config:
  instructions: |
    You are the Ads Recommendation agent.
  skills: []
  tool_categories: []
"""
    )
    (templates_dir / "growth_workforce.yaml").write_text(
        """
id: growth_workforce
name: Growth Marketing Workforce
category: Marketing
type: workforce
descriptions:
  en: Orchestrates GA Analyzer and Ads Recommendation.
  zh: 编排 GA Analyzer 和 Ads Recommendation。
author: Xagent
version: "1.0"

workforce_config:
  manager:
    name: Growth Marketing Manager
    description: Orchestrates the workforce.
    instructions: |
      You are the Growth Marketing Manager.
  agents:
  - template_id: ga_analyzer
    name: GA Analyzer
    alias: GA Analyzer
    assignment_instructions: Produce a Signals for Ads table.
  - template_id: ads_recommendation
    name: Ads Recommendation
    alias: Ads Recommendation
    assignment_instructions: Recommend ad changes using the Signals for Ads table.
"""
    )
    return templates_dir


@pytest.fixture(scope="function")
def workforce_template_manager(workforce_templates_dir):
    """Create a TemplateManager fixture over `workforce_templates_dir`"""
    from xagent.templates.manager import TemplateManager

    return TemplateManager(templates_root=workforce_templates_dir)


@pytest.fixture(scope="function")
def workforce_mock_app_state(workforce_template_manager):
    """Mock app.state.template_manager backed by the workforce templates dir"""
    import asyncio

    asyncio.run(workforce_template_manager.initialize())

    mock_state = MagicMock()
    mock_state.template_manager = workforce_template_manager
    return mock_state


@pytest.fixture(scope="function")
def persona_templates_dir(tmp_path):
    """A separate templates directory holding three templates - one with a
    full `persona` block, one with a name+role-only persona, and one with
    no persona at all - isolated from `templates_dir` so its exact-count
    assertions elsewhere aren't affected.
    """
    templates_dir = tmp_path / "persona_templates"
    templates_dir.mkdir()

    (templates_dir / "with_persona.yaml").write_text(
        """
id: with_persona
name: Social Media Content Manager
category: Marketing
descriptions:
  en: Turns a brief into platform-native posts.
  zh: 将简报转化为平台原生帖子。
persona:
  name: Maya
  role:
    en: Social Media Content Manager
    zh: 社媒内容经理
  avatar: /marketplace/avatars/maya.png
  intro:
    en: "Hi there — I'm Maya, your Social Media Content Manager."
    zh: 你好，我是 Maya，你的社媒内容经理。
  kickoff_questions:
    en:
    - Which platforms are in scope?
    - Do you have brand guidelines?
    zh:
    - 涉及哪些平台？
    - 有品牌规范吗？
author: Xagent
version: "1.0"

agent_config:
  instructions: |
    You are the Social Media Content Manager.
  skills: []
  tool_categories: []
"""
    )
    (templates_dir / "no_persona.yaml").write_text(
        """
id: no_persona
name: Plain Template
category: Support
descriptions:
  en: A template authored with no marketplace persona.
author: Xagent
version: "1.0"

agent_config:
  instructions: |
    You are a plain agent.
  skills: []
  tool_categories: []
"""
    )
    (templates_dir / "role_only_persona.yaml").write_text(
        """
id: role_only_persona
name: Role Only Template
category: Support
descriptions:
  en: A template whose persona authors only a name and role.
persona:
  name: Nia
  role:
    en: Some Role
author: Xagent
version: "1.0"

agent_config:
  instructions: |
    You are Nia.
  skills: []
  tool_categories: []
"""
    )
    return templates_dir


@pytest.fixture(scope="function")
def persona_template_manager(persona_templates_dir):
    """Create a TemplateManager fixture over `persona_templates_dir`"""
    from xagent.templates.manager import TemplateManager

    return TemplateManager(templates_root=persona_templates_dir)


@pytest.fixture(scope="function")
def persona_mock_app_state(persona_template_manager):
    """Mock app.state.template_manager backed by the persona templates dir"""
    import asyncio

    asyncio.run(persona_template_manager.initialize())

    mock_state = MagicMock()
    mock_state.template_manager = persona_template_manager
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
            assert "is_liked" in template
            assert template["views"] == 0
            assert template["likes"] == 0
            assert template["used_count"] == 0
            assert template["is_liked"] is False

            db = next(get_db())
            try:
                assert db.query(TemplateStats).count() == 2
            finally:
                db.close()

            response = client.get("/api/templates/", headers=admin_headers)
            assert response.status_code == 200

            db = next(get_db())
            try:
                assert db.query(TemplateStats).count() == 2
            finally:
                db.close()

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

            # The top-level tool_categories/skills mirror agent_config's,
            # so a marketplace card can render capability tags without a
            # second detail fetch (see test_list_templates_exposes_capabilities).
            assert template["skills"] == ["product_knowledge"]
            assert template["tool_categories"] == ["web_search"]

    def test_list_templates_exposes_capabilities(self, mock_app_state, admin_headers):
        """The list endpoint's TemplateInfo carries the same tool_categories/
        skills as the detail endpoint, since PersonaCard rendering needs
        them at list-render time, not just on a per-template detail fetch."""
        with patch.object(client.app, "state", mock_app_state):
            response = client.get("/api/templates/", headers=admin_headers)
            assert response.status_code == 200
            listed = {t["id"]: t for t in response.json()}

            assert listed["customer_support"]["tool_categories"] == ["web_search"]
            assert listed["customer_support"]["skills"] == ["product_knowledge"]
            assert listed["sales_assistant"]["tool_categories"] == ["file_operations"]
            assert listed["sales_assistant"]["skills"] == ["sales_techniques"]

    def test_workforce_template_exposes_no_capabilities(
        self, workforce_mock_app_state, admin_headers
    ):
        """A workforce-type template's real configuration lives in
        workforce_config, not agent_config - tool_categories/skills must
        report empty rather than leaking a stray/unused agent_config."""
        with patch.object(client.app, "state", workforce_mock_app_state):
            response = client.get(
                "/api/templates/growth_workforce", headers=admin_headers
            )
            assert response.status_code == 200
            body = response.json()
            assert body["tool_categories"] == []
            assert body["skills"] == []

    def test_sample_prompts_localization_and_default(
        self, mock_app_state, admin_headers
    ):
        """测试 sample_prompts 按语言解析，且未定义时默认为空列表"""
        with patch.object(client.app, "state", mock_app_state):
            # Template without sample_prompts defaults to an empty list.
            response = client.get(
                "/api/templates/customer_support?lang=en", headers=admin_headers
            )
            assert response.status_code == 200
            assert response.json()["sample_prompts"] == []

            # Template with sample_prompts resolves the requested locale.
            response = client.get(
                "/api/templates/sales_assistant?lang=en", headers=admin_headers
            )
            assert response.status_code == 200
            en_prompts = response.json()["sample_prompts"]
            assert len(en_prompts) == 2
            assert en_prompts[0]["title"] == "Draft an outreach email"
            assert en_prompts[0]["highlights"] == ["contact"]

            response = client.get(
                "/api/templates/sales_assistant?lang=zh", headers=admin_headers
            )
            assert response.status_code == 200
            zh_prompts = response.json()["sample_prompts"]
            assert zh_prompts[0]["title"] == "起草外联邮件"

            # The list endpoint carries the same field.
            response = client.get("/api/templates/?lang=en", headers=admin_headers)
            assert response.status_code == 200
            listed = {t["id"]: t for t in response.json()}
            assert len(listed["sales_assistant"]["sample_prompts"]) == 2
            assert listed["customer_support"]["sample_prompts"] == []

            # An unrecognised locale falls back to English rather than
            # erroring or returning an empty list.
            response = client.get(
                "/api/templates/sales_assistant?lang=fr", headers=admin_headers
            )
            assert response.status_code == 200
            fr_prompts = response.json()["sample_prompts"]
            assert fr_prompts == en_prompts

    def test_get_template_not_found(self, mock_app_state, admin_headers):
        """测试获取不存在的模板"""
        with patch.object(client.app, "state", mock_app_state):
            response = client.get("/api/templates/nonexistent", headers=admin_headers)

            assert response.status_code == 404

    def test_like_template(self, mock_app_state, admin_headers):
        """测试同一用户重复点赞只计数一次"""
        with patch.object(client.app, "state", mock_app_state):
            # 第一次点赞
            response = client.post(
                "/api/templates/customer_support/like", headers=admin_headers
            )

            assert response.status_code == 200
            result = response.json()
            assert result["liked"] is True
            assert result["likes"] == 1

            # 同一用户重复点赞应幂等，不重复增加
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
            detail = response.json()
            assert detail["likes"] == 1
            assert detail["is_liked"] is True

            # 获取模板列表验证当前用户点赞状态
            response = client.get("/api/templates/", headers=admin_headers)
            assert response.status_code == 200
            templates = response.json()
            liked_template = next(
                template
                for template in templates
                if template["id"] == "customer_support"
            )
            assert liked_template["likes"] == 1
            assert liked_template["is_liked"] is True

    def test_like_template_from_different_users_counts_separately(
        self, mock_app_state, admin_headers
    ):
        """测试不同用户点赞同一模板分别计数"""
        user_headers = create_user_headers("template_like_user")

        with patch.object(client.app, "state", mock_app_state):
            response = client.post(
                "/api/templates/customer_support/like", headers=admin_headers
            )
            assert response.status_code == 200
            assert response.json()["likes"] == 1

            response = client.post(
                "/api/templates/customer_support/like", headers=user_headers
            )
            assert response.status_code == 200
            assert response.json()["likes"] == 2

    def test_like_template_not_found(self, mock_app_state, admin_headers):
        """测试点赞不存在模板仍返回 404"""
        with patch.object(client.app, "state", mock_app_state):
            response = client.post(
                "/api/templates/nonexistent/like", headers=admin_headers
            )
            assert response.status_code == 404

    def test_user_template_relation_unique_constraint(self, admin_user):
        """测试同一用户同一模板同一关系类型只能有一条数据"""
        db = next(get_db())
        try:
            relation = UserTemplateRelation(
                user_id=admin_user["id"],
                template_id="customer_support",
                relation_type="like",
                is_active=True,
            )
            duplicate = UserTemplateRelation(
                user_id=admin_user["id"],
                template_id="customer_support",
                relation_type="like",
                is_active=True,
            )
            db.add(relation)
            db.commit()

            db.add(duplicate)
            with pytest.raises(IntegrityError):
                db.commit()
            db.rollback()
        finally:
            db.close()

    def test_inactive_like_relation_reactivates(
        self, mock_app_state, admin_headers, admin_user
    ):
        """测试 inactive 的点赞关系再次 like 会重新激活并增加计数"""
        db = next(get_db())
        try:
            db.add(TemplateStats(template_id="customer_support", likes=0))
            db.add(
                UserTemplateRelation(
                    user_id=admin_user["id"],
                    template_id="customer_support",
                    relation_type="like",
                    is_active=False,
                )
            )
            db.commit()
        finally:
            db.close()

        with patch.object(client.app, "state", mock_app_state):
            response = client.post(
                "/api/templates/customer_support/like", headers=admin_headers
            )
            assert response.status_code == 200
            assert response.json()["likes"] == 1

        db = next(get_db())
        try:
            relation = (
                db.query(UserTemplateRelation)
                .filter(
                    UserTemplateRelation.user_id == admin_user["id"],
                    UserTemplateRelation.template_id == "customer_support",
                    UserTemplateRelation.relation_type == "like",
                )
                .one()
            )
            assert relation.is_active is True
        finally:
            db.close()

    def test_increment_template_likes_uses_database_atomic_update(self, test_db):
        """测试点赞计数不会被 stale session 覆盖"""
        db = next(get_db())
        try:
            db.add(TemplateStats(template_id="customer_support", likes=0))
            db.commit()
        finally:
            db.close()

        db1 = next(get_db())
        db2 = next(get_db())
        try:
            stats1 = (
                db1.query(TemplateStats)
                .filter(TemplateStats.template_id == "customer_support")
                .one()
            )
            stats2 = (
                db2.query(TemplateStats)
                .filter(TemplateStats.template_id == "customer_support")
                .one()
            )
            assert stats1.likes == 0
            assert stats2.likes == 0

            increment_template_likes(db1, "customer_support")
            db1.commit()
            db1.refresh(stats1)
            increment_template_likes(db2, "customer_support")
            db2.commit()
            db2.refresh(stats2)
        finally:
            db1.close()
            db2.close()

        db = next(get_db())
        try:
            stats = (
                db.query(TemplateStats)
                .filter(TemplateStats.template_id == "customer_support")
                .one()
            )
            assert stats.likes == 2
        finally:
            db.close()

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

    def test_use_rejects_workforce_template(
        self, workforce_mock_app_state, admin_headers
    ):
        """The legacy POST /use only knows how to record usage for an
        agent-creation flow (the frontend never calls it for a workforce
        template today, but nothing stopped an external caller from
        hitting it directly and getting a misleading 200 "recorded" while
        creating nothing - every other agent-creation surface already
        refuses workforce templates) (PR #1127 re-review, m2)."""
        with patch.object(client.app, "state", workforce_mock_app_state):
            response = client.post(
                "/api/templates/growth_workforce/use", headers=admin_headers
            )
            assert response.status_code == 400

            detail = client.get(
                "/api/templates/growth_workforce", headers=admin_headers
            ).json()
            assert detail["used_count"] == 0

    def test_use_count_increments_atomically_under_concurrent_reads(
        self, mock_app_state, admin_headers
    ):
        """`stats.used_count += 1` is an ORM read-modify-write: two sessions
        that both read used_count=0 before either commits would both write
        1, losing an increment. `increment_template_used_count` (mirroring
        the existing `increment_template_likes` pattern) uses an atomic
        `UPDATE ... SET used_count = used_count + 1` instead - simulate two
        already-open sessions racing to prove the count doesn't stall at 1
        (PR #1127 re-review, m3)."""
        with patch.object(client.app, "state", mock_app_state):
            db1 = next(get_db())
            db2 = next(get_db())
            try:
                stats1 = get_or_create_template_stats(db1, "customer_support")
                stats2 = get_or_create_template_stats(db2, "customer_support")
                assert stats1.used_count == 0
                assert stats2.used_count == 0

                increment_template_used_count(db1, "customer_support")
                db1.commit()
                increment_template_used_count(db2, "customer_support")
                db2.commit()
            finally:
                db1.close()
                db2.close()

            db = next(get_db())
            try:
                stats = (
                    db.query(TemplateStats)
                    .filter(TemplateStats.template_id == "customer_support")
                    .one()
                )
                assert stats.used_count == 2
            finally:
                db.close()

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
        """Test unauthorized access is rejected"""
        with patch.object(client.app, "state", mock_app_state):
            response = client.get("/api/templates/")

            # A request with no Authorization header is rejected by
            # HTTPBearer's own dependency resolution before get_current_user
            # ever runs. Which status it gets depends on the installed
            # FastAPI: older versions raised 403 for a missing header, newer
            # ones raise 401 "Not authenticated" (corrected for RFC 6750
            # compliance; 0.135.1 returns 401 - verified via a standalone
            # HTTPBearer repro, not test ordering/pollution). pyproject only
            # pins `fastapi >= 0.35.0`, so both behaviors are legitimately
            # installable; hardcoding either status just fails on the other
            # side of the version line. What this test actually protects is
            # "no auth header never reaches the handler".
            assert response.status_code in (401, 403)

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
                "is_liked",
                "persona",
                "hired",
                "hired_agent_id",
            ]
            for field in required_fields:
                assert field in template, f"Missing field: {field}"


class TestTemplatePersona:
    """测试 marketplace persona 字段的解析与本地化"""

    def test_persona_included_and_localized(
        self, persona_mock_app_state, admin_headers
    ):
        with patch.object(client.app, "state", persona_mock_app_state):
            response = client.get(
                "/api/templates/with_persona?lang=en", headers=admin_headers
            )
            assert response.status_code == 200
            persona = response.json()["persona"]
            assert persona["name"] == "Maya"
            assert persona["role"] == "Social Media Content Manager"
            assert persona["avatar"] == "/marketplace/avatars/maya.png"
            assert "Maya" in persona["intro"]
            assert len(persona["kickoff_questions"]) == 2

            response_zh = client.get(
                "/api/templates/with_persona?lang=zh", headers=admin_headers
            )
            assert response_zh.json()["persona"]["role"] == "社媒内容经理"

            # The list endpoint carries the same field.
            response = client.get("/api/templates/?lang=en", headers=admin_headers)
            listed = {t["id"]: t for t in response.json()}
            assert listed["with_persona"]["persona"]["name"] == "Maya"
            assert listed["no_persona"]["persona"] is None

    def test_persona_absent_when_not_authored(
        self, persona_mock_app_state, admin_headers
    ):
        """A template with no `persona` block (e.g. a workforce-type
        template, or one just not yet given marketplace treatment) reports
        `persona: null` rather than a missing field or a 500."""
        with patch.object(client.app, "state", persona_mock_app_state):
            response = client.get("/api/templates/no_persona", headers=admin_headers)
            assert response.status_code == 200
            assert response.json()["persona"] is None

    def test_persona_falls_back_to_english_when_lang_is_omitted(
        self, persona_mock_app_state, admin_headers
    ):
        with patch.object(client.app, "state", persona_mock_app_state):
            response = client.get("/api/templates/with_persona", headers=admin_headers)
            assert response.status_code == 200
            assert response.json()["persona"]["role"] == "Social Media Content Manager"

    def test_persona_falls_back_to_english_for_an_unrecognized_locale(
        self, persona_mock_app_state, admin_headers
    ):
        with patch.object(client.app, "state", persona_mock_app_state):
            response = client.get(
                "/api/templates/with_persona?lang=fr", headers=admin_headers
            )
            assert response.status_code == 200
            assert response.json()["persona"]["role"] == "Social Media Content Manager"

    def test_persona_with_only_role_authored_yields_pydantic_defaults_not_a_500(
        self, persona_mock_app_state, admin_headers
    ):
        """A persona authoring only name/role (no intro/kickoff_questions -
        both fully optional, see TestValidatePersona) must resolve through
        PersonaInfo's own field defaults (`intro=""`,
        `kickoff_questions=[]`), not raise a pydantic ValidationError."""
        with patch.object(client.app, "state", persona_mock_app_state):
            response = client.get(
                "/api/templates/role_only_persona?lang=en", headers=admin_headers
            )
            assert response.status_code == 200
            persona = response.json()["persona"]
            assert persona["name"] == "Nia"
            assert persona["role"] == "Some Role"
            assert persona["intro"] == ""
            assert persona["kickoff_questions"] == []


class TestTemplateHiredFlag:
    """测试 hired / hired_agent_id 是否正确反映当前用户的 quick-access agent"""

    def test_false_when_no_quick_access_agent_exists(
        self, mock_app_state, admin_headers
    ):
        with patch.object(client.app, "state", mock_app_state):
            response = client.get("/api/templates/", headers=admin_headers)
            for template in response.json():
                assert template["hired"] is False
                assert template["hired_agent_id"] is None

            detail = client.get(
                "/api/templates/customer_support", headers=admin_headers
            ).json()
            assert detail["hired"] is False
            assert detail["hired_agent_id"] is None

    def test_true_for_the_users_own_quick_access_agent(
        self, mock_app_state, admin_headers, admin_user
    ):
        db = next(get_db())
        agent = Agent(
            user_id=admin_user["id"],
            name="Customer Support Agent (mine)",
            template_id="customer_support",
            origin=AgentOrigin.TEMPLATE_QUICK_ACCESS.value,
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        agent_id = agent.id
        db.close()

        with patch.object(client.app, "state", mock_app_state):
            response = client.get("/api/templates/", headers=admin_headers)
            listed = {t["id"]: t for t in response.json()}
            assert listed["customer_support"]["hired"] is True
            assert listed["customer_support"]["hired_agent_id"] == agent_id
            # A different template the user has no quick-access agent for
            # must not be marked hired just because *some* template is.
            assert listed["sales_assistant"]["hired"] is False
            assert listed["sales_assistant"]["hired_agent_id"] is None

            detail = client.get(
                "/api/templates/customer_support", headers=admin_headers
            ).json()
            assert detail["hired"] is True
            assert detail["hired_agent_id"] == agent_id

    def test_archived_quick_access_agent_still_counts_as_hired(
        self, mock_app_state, admin_headers, admin_user
    ):
        """hired deliberately has no status filter (see
        get_hired_agent_map's docstring): the resolve flow returns a found
        quick-access agent as-is whatever its status - DRAFT and ARCHIVED
        alike - so an archived one is still "what Hire returns" and must
        report hired (PR #1498 round-3 review)."""
        db = next(get_db())
        agent = Agent(
            user_id=admin_user["id"],
            name="Customer Support Agent (archived)",
            template_id="customer_support",
            origin=AgentOrigin.TEMPLATE_QUICK_ACCESS.value,
            status=AgentStatus.ARCHIVED,
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        agent_id = agent.id
        db.close()

        with patch.object(client.app, "state", mock_app_state):
            listed = {
                t["id"]: t
                for t in client.get("/api/templates/", headers=admin_headers).json()
            }
            assert listed["customer_support"]["hired"] is True
            assert listed["customer_support"]["hired_agent_id"] == agent_id

    def test_hired_resolves_to_the_quick_access_agent_not_a_user_origin_one(
        self, mock_app_state, admin_headers, admin_user
    ):
        """A plain user-origin agent that happens to carry the same
        template_id (e.g. minted by the workforce-builder UI under a
        user-chosen name via the plain POST /from-template path) must not
        count as hired - hired specifically means the quick-access
        instance, matching resolve_agent_from_template's own
        origin-scoped reuse query.

        Creates a real TEMPLATE_QUICK_ACCESS-origin agent for the same
        template too, deliberately with a *lower* id than the USER-origin
        row: `get_hired_agent_map` has no ORDER BY (last row wins in a
        filterless dict build), so with the quick-access row inserted
        first, an implementation missing the origin filter would resolve
        to the later USER-origin row and fail here - mutation testing on
        the round-1 version of this test (quick-access inserted second)
        showed removing the filter still passed (PR #1498 round-2 review,
        N4)."""
        db = next(get_db())
        quick_access_agent = Agent(
            user_id=admin_user["id"],
            name="Customer Support Agent",
            template_id="customer_support",
            origin=AgentOrigin.TEMPLATE_QUICK_ACCESS.value,
        )
        db.add(quick_access_agent)
        db.commit()
        db.refresh(quick_access_agent)
        quick_access_agent_id = quick_access_agent.id

        user_origin_agent = Agent(
            user_id=admin_user["id"],
            name="My Own Copy",
            template_id="customer_support",
            origin=AgentOrigin.USER.value,
        )
        db.add(user_origin_agent)
        db.commit()
        db.refresh(user_origin_agent)
        assert user_origin_agent.id > quick_access_agent_id
        db.close()

        with patch.object(client.app, "state", mock_app_state):
            response = client.get("/api/templates/", headers=admin_headers)
            listed = {t["id"]: t for t in response.json()}
            assert listed["customer_support"]["hired"] is True
            assert listed["customer_support"]["hired_agent_id"] == quick_access_agent_id

    def test_false_when_the_only_agent_is_user_origin(
        self, mock_app_state, admin_headers, admin_user
    ):
        """The complementary case to the test above: a template whose only
        agent for this user is a plain USER-origin one must not report
        hired at all."""
        db = next(get_db())
        user_origin_agent = Agent(
            user_id=admin_user["id"],
            name="My Own Copy Only",
            template_id="customer_support",
            origin=AgentOrigin.USER.value,
        )
        db.add(user_origin_agent)
        db.commit()
        db.close()

        with patch.object(client.app, "state", mock_app_state):
            response = client.get("/api/templates/", headers=admin_headers)
            listed = {t["id"]: t for t in response.json()}
            assert listed["customer_support"]["hired"] is False
            assert listed["customer_support"]["hired_agent_id"] is None

    def test_scoped_to_the_current_user(
        self, mock_app_state, admin_headers, admin_user
    ):
        """Another user's quick-access agent for the same template must
        not mark it hired for the current user."""
        db = next(get_db())
        other_user = User(
            username="other_user_hired_flag_test",
            password_hash=hash_password("other123"),
        )
        db.add(other_user)
        db.commit()
        db.refresh(other_user)
        agent = Agent(
            user_id=other_user.id,
            name="Someone Else's Copy",
            template_id="customer_support",
            origin=AgentOrigin.TEMPLATE_QUICK_ACCESS.value,
        )
        db.add(agent)
        db.commit()
        db.close()

        with patch.object(client.app, "state", mock_app_state):
            response = client.get("/api/templates/", headers=admin_headers)
            listed = {t["id"]: t for t in response.json()}
            assert listed["customer_support"]["hired"] is False
            assert listed["customer_support"]["hired_agent_id"] is None

            # Same scoping on the detail endpoint (a separate call site
            # computing hired from the same map).
            detail = client.get(
                "/api/templates/customer_support", headers=admin_headers
            ).json()
            assert detail["hired"] is False
            assert detail["hired_agent_id"] is None


class TestUseTemplateAsWorkforce:
    """测试 POST /api/templates/{id}/use-as-workforce"""

    def test_rejects_non_workforce_template(self, mock_app_state, admin_headers):
        with patch.object(client.app, "state", mock_app_state):
            response = client.post(
                "/api/templates/customer_support/use-as-workforce",
                headers=admin_headers,
            )
            assert response.status_code == 400

    def test_rejects_unknown_template(self, mock_app_state, admin_headers):
        with patch.object(client.app, "state", mock_app_state):
            response = client.post(
                "/api/templates/nonexistent/use-as-workforce",
                headers=admin_headers,
            )
            assert response.status_code == 404

    def test_creates_manager_and_worker_agents(
        self, workforce_mock_app_state, admin_headers
    ):
        with patch.object(client.app, "state", workforce_mock_app_state):
            response = client.post(
                "/api/templates/growth_workforce/use-as-workforce",
                headers=admin_headers,
            )
            assert response.status_code == 200, response.text
            payload = response.json()
            assert payload["template_id"] == "growth_workforce"
            workforce_id = payload["workforce_id"]

            db = next(get_db())
            try:
                workforce = db.get(Workforce, workforce_id)
                assert workforce is not None

                manager = db.get(Agent, workforce.manager_agent_id)
                assert manager is not None
                assert manager.origin == AgentOrigin.WORKFORCE_GENERATED_MANAGER.value
                assert manager.status == AgentStatus.PUBLISHED
                assert "Growth Marketing Manager" in manager.instructions

                workers = (
                    db.query(WorkforceAgent)
                    .filter(WorkforceAgent.workforce_id == workforce_id)
                    .all()
                )
                assert len(workers) == 2
                # WorkforceAgent.template_id records provenance - which
                # template's workforce_config.agents[] entry created this
                # worker link (PR #1127 review: previously always NULL).
                assert {w.template_id for w in workers} == {
                    "ga_analyzer",
                    "ads_recommendation",
                }
                worker_agent_ids = {w.agent_id for w in workers}
                worker_agents = (
                    db.query(Agent).filter(Agent.id.in_(worker_agent_ids)).all()
                )
                assert len(worker_agents) == 2
                assert all(
                    agent.origin == AgentOrigin.TEMPLATE_QUICK_ACCESS.value
                    for agent in worker_agents
                )
                assert all(
                    agent.status == AgentStatus.PUBLISHED for agent in worker_agents
                )
                assert {agent.template_id for agent in worker_agents} == {
                    "ga_analyzer",
                    "ads_recommendation",
                }
            finally:
                db.close()

            # Usage is recorded, same as the single-agent /use endpoint.
            detail = client.get(
                "/api/templates/growth_workforce", headers=admin_headers
            ).json()
            assert detail["used_count"] == 1

    def test_lang_query_param_localizes_the_new_workforces_description(
        self, workforce_mock_app_state, admin_headers
    ):
        """Every sibling GET endpoint accepts ?lang= and localizes through
        get_localized_value; this endpoint previously had no way to know
        the caller's locale at all, so a zh-locale user's new Workforce
        always got the English description regardless (PR #1127
        re-review, F4)."""
        with patch.object(client.app, "state", workforce_mock_app_state):
            response = client.post(
                "/api/templates/growth_workforce/use-as-workforce?lang=zh",
                headers=admin_headers,
            )
            assert response.status_code == 200, response.text
            workforce_id = response.json()["workforce_id"]

            db = next(get_db())
            try:
                workforce = db.get(Workforce, workforce_id)
                assert (
                    workforce.description == "编排 GA Analyzer 和 Ads Recommendation。"
                )
            finally:
                db.close()

    def test_reuses_worker_agents_across_two_instantiations(
        self, workforce_mock_app_state, admin_headers
    ):
        """Instantiating the same workforce template twice must mint a new
        manager + Workforce each time (each run gets its own orchestrator)
        but must NOT mint duplicate worker agents - the second call should
        reuse the first call's quick-access GA Analyzer / Ads Recommendation
        agents (see AGENT_TEMPLATE_QUICK_ACCESS_UNIQUE_INDEX).

        This is deliberate, not an oversight: there is no server-side
        idempotency on the Workforce + manager themselves, only on workers.
        A double-click or a retried request will leave a duplicate draft
        Workforce plus a stray published manager agent - the client-side
        `creatingWorkforceId` lock is the only guard, and it isn't atomic
        against network retries or two tabs. Flagged as a product question
        in the PR #1127 re-review (F5), not changed here."""
        with patch.object(client.app, "state", workforce_mock_app_state):
            first = client.post(
                "/api/templates/growth_workforce/use-as-workforce",
                headers=admin_headers,
            )
            assert first.status_code == 200, first.text
            second = client.post(
                "/api/templates/growth_workforce/use-as-workforce",
                headers=admin_headers,
            )
            assert second.status_code == 200, second.text

            first_workforce_id = first.json()["workforce_id"]
            second_workforce_id = second.json()["workforce_id"]
            assert first_workforce_id != second_workforce_id

            db = next(get_db())
            try:
                first_workforce = db.get(Workforce, first_workforce_id)
                second_workforce = db.get(Workforce, second_workforce_id)
                # Two distinct, freshly-minted managers.
                assert (
                    first_workforce.manager_agent_id
                    != second_workforce.manager_agent_id
                )

                first_worker_ids = {
                    w.agent_id
                    for w in db.query(WorkforceAgent).filter(
                        WorkforceAgent.workforce_id == first_workforce_id
                    )
                }
                second_worker_ids = {
                    w.agent_id
                    for w in db.query(WorkforceAgent).filter(
                        WorkforceAgent.workforce_id == second_workforce_id
                    )
                }
                assert first_worker_ids == second_worker_ids

                quick_access_count = (
                    db.query(Agent)
                    .filter(Agent.origin == AgentOrigin.TEMPLATE_QUICK_ACCESS.value)
                    .count()
                )
                assert quick_access_count == 2
            finally:
                db.close()

    def test_rejects_when_workforce_creation_is_not_allowed(
        self, workforce_mock_app_state, admin_headers
    ):
        """403 when the workforce access policy (can_create_workforce)
        denies the request - not covered before this test (PR #1127
        review)."""
        with (
            patch.object(client.app, "state", workforce_mock_app_state),
            patch(
                "xagent.web.services.workforce_creator.can_create_workforce",
                return_value=False,
            ),
        ):
            response = client.post(
                "/api/templates/growth_workforce/use-as-workforce",
                headers=admin_headers,
            )
            assert response.status_code == 403

    def test_rejects_unknown_worker_template_at_use_time(self, tmp_path, admin_headers):
        """workforce_config.agents[].template_id is only checked for being a
        non-empty string at load time (cross-file references can't be
        validated per-file) - a dangling reference must surface as a clean
        400 when the template is actually used, not an unhandled crash
        (PR #1127 review)."""
        import asyncio

        from xagent.templates.manager import TemplateManager

        templates_dir = tmp_path / "dangling_workforce_templates"
        templates_dir.mkdir()
        (templates_dir / "dangling_workforce.yaml").write_text(
            """
id: dangling_workforce
name: Dangling Workforce
category: Marketing
type: workforce
descriptions:
  en: References a worker template that does not exist.
author: Xagent
version: "1.0"

workforce_config:
  manager:
    name: Dangling Manager
    instructions: |
      You are the manager.
  agents:
  - template_id: ghost_agent
    name: Ghost Worker
    assignment_instructions: Do the thing.
"""
        )
        template_manager = TemplateManager(templates_root=templates_dir)
        asyncio.run(template_manager.initialize())
        mock_state = MagicMock()
        mock_state.template_manager = template_manager

        with patch.object(client.app, "state", mock_state):
            response = client.post(
                "/api/templates/dangling_workforce/use-as-workforce",
                headers=admin_headers,
            )
            assert response.status_code == 400

            db = next(get_db())
            try:
                # The failed attempt must not have left a partial
                # manager/Workforce behind.
                assert db.query(Workforce).count() == 0
                assert db.query(Agent).count() == 0
            finally:
                db.close()

    def test_rejects_a_worker_that_is_itself_a_workforce_template(
        self, tmp_path, admin_headers
    ):
        """`workforce_config.agents[].template_id` must resolve to an
        'agent'-type template. TemplateManager._enrich_template nulls
        agent_config for any non-agent template, so without this runtime
        guard a workforce referencing another workforce as a worker would
        crash on `None.get(...)` instead of failing with a clear 400 - only
        the load-time warning for this case had a test before this."""
        import asyncio

        from xagent.templates.manager import TemplateManager

        templates_dir = tmp_path / "nested_workforce_templates"
        templates_dir.mkdir()
        (templates_dir / "inner_workforce.yaml").write_text(
            """
id: inner_workforce
name: Inner Workforce
category: Marketing
type: workforce
descriptions:
  en: A workforce, wrongly referenced as a worker below.
author: Xagent
version: "1.0"

workforce_config:
  manager:
    name: Inner Manager
    instructions: |
      You are the inner manager.
  agents:
  - template_id: leaf_agent
    name: Leaf
    assignment_instructions: Do the thing.
"""
        )
        (templates_dir / "leaf_agent.yaml").write_text(
            """
id: leaf_agent
name: Leaf Agent
category: Marketing
descriptions:
  en: A normal single-agent template.
author: Xagent
version: "1.0"

agent_config:
  instructions: |
    You help with things.
"""
        )
        (templates_dir / "outer_workforce.yaml").write_text(
            """
id: outer_workforce
name: Outer Workforce
category: Marketing
type: workforce
descriptions:
  en: References inner_workforce (a workforce) as if it were a single agent.
author: Xagent
version: "1.0"

workforce_config:
  manager:
    name: Outer Manager
    instructions: |
      You are the outer manager.
  agents:
  - template_id: inner_workforce
    name: Nested Workforce
    assignment_instructions: Do the thing.
"""
        )
        template_manager = TemplateManager(templates_root=templates_dir)
        asyncio.run(template_manager.initialize())
        mock_state = MagicMock()
        mock_state.template_manager = template_manager

        with patch.object(client.app, "state", mock_state):
            response = client.post(
                "/api/templates/outer_workforce/use-as-workforce",
                headers=admin_headers,
            )
            assert response.status_code == 400
            assert "inner_workforce" in response.json()["detail"]

            db = next(get_db())
            try:
                assert db.query(Workforce).count() == 0
                assert db.query(Agent).count() == 0
            finally:
                db.close()

    def test_rejects_with_actionable_message_when_worker_agent_is_unpublished(
        self, workforce_mock_app_state, admin_headers, admin_user
    ):
        """A user who separately unpublished their quick-access GA Analyzer
        agent must get a specific, actionable 400 - not the generic
        "please try again" a retry can never resolve - and the failed
        attempt must not leave a stray manager/Workforce behind
        (PR #1127 re-review, F1)."""
        from xagent.web.services.workforce_creator import (
            WORKFORCE_WORKER_UNPUBLISHED_CODE,
        )

        db = next(get_db())
        try:
            db.add(
                Agent(
                    user_id=admin_user["id"],
                    name="GA Analyzer",
                    instructions="You are the GA Analyzer.",
                    execution_mode="balanced",
                    template_id="ga_analyzer",
                    origin=AgentOrigin.TEMPLATE_QUICK_ACCESS.value,
                    status=AgentStatus.DRAFT,
                )
            )
            db.commit()
        finally:
            db.close()

        with patch.object(client.app, "state", workforce_mock_app_state):
            response = client.post(
                "/api/templates/growth_workforce/use-as-workforce",
                headers=admin_headers,
            )
            assert response.status_code == 400
            # Structured {code, message, params} detail - the frontend maps
            # the machine code and interpolates params.agent_name rather
            # than parsing the English message (PR #1127 re-review, m4).
            detail = response.json()["detail"]
            assert detail["code"] == WORKFORCE_WORKER_UNPUBLISHED_CODE
            assert detail["params"]["agent_name"] == "GA Analyzer"
            assert "GA Analyzer" in detail["message"]

            db = next(get_db())
            try:
                assert db.query(Workforce).count() == 0
                # Only the pre-existing draft agent - no manager, no
                # duplicate worker created for the failed attempt.
                assert db.query(Agent).count() == 1
            finally:
                db.close()

    def test_recovers_from_a_genuine_concurrent_workforce_name_collision(
        self, workforce_mock_app_state, admin_headers, admin_user
    ):
        """Two concurrent instantiations of THIS SAME template resolve the
        SAME Workforce name - unlike `create_workforce_from_prompt`'s
        LLM-generated name, a workforce template's name is fixed, so this
        collision is the realistic case, not a theoretical one. Uses a
        genuinely separate session (not a row staged inside the same
        SAVEPOINT scope, which a first attempt at this test class of
        problem already proved gets rolled back right along with the
        failed insert - see the worker-resolution race tests) to commit a
        colliding `Workforce` row between our own `resolve_unique_workforce_name`
        check and our own insert, reproducing the exact TOCTOU window a
        double-click, a double-tab, or a retried request can hit
        (PR #1127 re-review, M1)."""
        from sqlalchemy.orm import sessionmaker

        from xagent.web.models.database import get_engine
        from xagent.web.services import workforce_creator

        real_resolve_unique_workforce_name = (
            workforce_creator.resolve_unique_workforce_name
        )
        session_factory = sessionmaker(bind=get_engine())
        call_count = {"n": 0}
        competitor_manager_id: dict[str, int] = {}

        def _collide_once_then_resolve(db, *, scope_type, scope_id, name):
            call_count["n"] += 1
            if call_count["n"] == 1:
                competitor_session = session_factory()
                try:
                    # This engine enforces FK constraints (PRAGMA
                    # foreign_keys=ON, see src/xagent/db/sqlite.py) -
                    # manager_agent_id needs a real Agent row.
                    competitor_manager = Agent(
                        user_id=admin_user["id"],
                        name="Competitor Manager",
                        instructions="from the concurrent winner",
                        execution_mode="think",
                        origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
                        status=AgentStatus.PUBLISHED,
                    )
                    competitor_session.add(competitor_manager)
                    competitor_session.flush()
                    competitor_manager_id["id"] = competitor_manager.id
                    competitor_session.add(
                        Workforce(
                            owner_user_id=admin_user["id"],
                            scope_type=scope_type,
                            scope_id=scope_id,
                            name=name,
                            manager_agent_id=competitor_manager.id,
                            status="draft",
                        )
                    )
                    competitor_session.commit()
                finally:
                    competitor_session.close()
                return name
            return real_resolve_unique_workforce_name(
                db, scope_type=scope_type, scope_id=scope_id, name=name
            )

        with (
            patch.object(client.app, "state", workforce_mock_app_state),
            patch.object(
                workforce_creator,
                "resolve_unique_workforce_name",
                side_effect=_collide_once_then_resolve,
            ),
        ):
            response = client.post(
                "/api/templates/growth_workforce/use-as-workforce",
                headers=admin_headers,
            )

        assert response.status_code == 200, response.text
        assert call_count["n"] == 2

        db = next(get_db())
        try:
            workforces = db.query(Workforce).all()
            assert len(workforces) == 2
            # The competitor's manager agent is a fixture of this test, not
            # the real request's - so whichever row does NOT point at it is
            # the real request's Workforce, and it must have gotten a
            # disambiguated name rather than colliding again.
            real_workforce = next(
                w
                for w in workforces
                if w.manager_agent_id != competitor_manager_id["id"]
            )
            competitor_workforce = next(
                w
                for w in workforces
                if w.manager_agent_id == competitor_manager_id["id"]
            )
            assert competitor_workforce.name == "Growth Marketing Workforce"
            assert real_workforce.name != "Growth Marketing Workforce"
            assert real_workforce.name.startswith("Growth Marketing Workforce")
        finally:
            db.close()

    def test_second_collision_advances_the_suffix_instead_of_compounding_it(
        self, workforce_mock_app_state, admin_headers, admin_user
    ):
        """The retry loop must re-resolve from the ORIGINAL base name on
        every collision, not from the previous (already-suffixed) attempt.
        Feeding the suffixed name back in would compound suffixes on a
        second collision within the same request ("X 2" -> "X 2 2")
        instead of advancing to "X 3" - reachable since
        TEMPLATE_RESOLVE_RACE_RETRIES allows more than one retry.

        Pre-seeds two already-taken names ("Growth Marketing Workforce" and
        "...  2") and makes the first two `resolve_unique_workforce_name`
        calls lie that each is free (forcing two real INSERT collisions),
        then lets the third call resolve for real - so the *result* proves
        which base name the retry loop actually passed in.
        """
        from sqlalchemy.orm import sessionmaker

        from xagent.web.models.database import get_engine
        from xagent.web.services import workforce_creator

        real_resolve_unique_workforce_name = (
            workforce_creator.resolve_unique_workforce_name
        )
        session_factory = sessionmaker(bind=get_engine())

        seed_session = session_factory()
        try:
            for taken_name in (
                "Growth Marketing Workforce",
                "Growth Marketing Workforce 2",
            ):
                manager = Agent(
                    user_id=admin_user["id"],
                    name=f"Manager for {taken_name}",
                    instructions="pre-seeded",
                    execution_mode="think",
                    origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
                    status=AgentStatus.PUBLISHED,
                )
                seed_session.add(manager)
                seed_session.flush()
                seed_session.add(
                    Workforce(
                        owner_user_id=admin_user["id"],
                        scope_type="user",
                        scope_id=str(admin_user["id"]),
                        name=taken_name,
                        manager_agent_id=manager.id,
                        status="draft",
                    )
                )
            seed_session.commit()
        finally:
            seed_session.close()

        call_args: list[str] = []

        def _lie_twice_then_resolve_for_real(db, *, scope_type, scope_id, name):
            call_args.append(name)
            if len(call_args) == 1:
                return "Growth Marketing Workforce"  # actually taken
            if len(call_args) == 2:
                return "Growth Marketing Workforce 2"  # actually taken
            return real_resolve_unique_workforce_name(
                db, scope_type=scope_type, scope_id=scope_id, name=name
            )

        with (
            patch.object(client.app, "state", workforce_mock_app_state),
            patch.object(
                workforce_creator,
                "resolve_unique_workforce_name",
                side_effect=_lie_twice_then_resolve_for_real,
            ),
        ):
            response = client.post(
                "/api/templates/growth_workforce/use-as-workforce",
                headers=admin_headers,
            )

        assert response.status_code == 200, response.text
        # All three calls resolved from the same original base name - never
        # from a previously-suffixed attempt.
        assert call_args == ["Growth Marketing Workforce"] * 3

        db = next(get_db())
        try:
            workforce = db.get(Workforce, response.json()["workforce_id"])
            assert workforce.name == "Growth Marketing Workforce 3"
        finally:
            db.close()

    def test_collision_after_an_already_suffixed_initial_resolution(
        self, workforce_mock_app_state, admin_headers, admin_user
    ):
        """The retry's base name must be the TEMPLATE's raw name, not the
        initial resolution's result - the initial resolution itself already
        carries a suffix whenever the user instantiated this template
        before ("Growth Marketing Workforce 2" on the second run). A retry
        that re-resolved from that resolved name would compound
        ("X 2" -> "X 2 2") even with the loop's own collided names handled
        correctly.

        Pre-seeds "X" and "X 2" as taken, has the initial resolution lie
        that "X 2" is free (as if it resolved before a concurrent winner
        committed it), and lets the retry resolve for real: from the raw
        base it must skip both taken names and land on "X 3".
        """
        from sqlalchemy.orm import sessionmaker

        from xagent.web.models.database import get_engine
        from xagent.web.services import workforce_creator

        real_resolve_unique_workforce_name = (
            workforce_creator.resolve_unique_workforce_name
        )
        session_factory = sessionmaker(bind=get_engine())

        seed_session = session_factory()
        try:
            for taken_name in (
                "Growth Marketing Workforce",
                "Growth Marketing Workforce 2",
            ):
                manager = Agent(
                    user_id=admin_user["id"],
                    name=f"Manager for {taken_name}",
                    instructions="pre-seeded",
                    execution_mode="think",
                    origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
                    status=AgentStatus.PUBLISHED,
                )
                seed_session.add(manager)
                seed_session.flush()
                seed_session.add(
                    Workforce(
                        owner_user_id=admin_user["id"],
                        scope_type="user",
                        scope_id=str(admin_user["id"]),
                        name=taken_name,
                        manager_agent_id=manager.id,
                        status="draft",
                    )
                )
            seed_session.commit()
        finally:
            seed_session.close()

        call_args: list[str] = []

        def _lie_once_then_resolve_for_real(db, *, scope_type, scope_id, name):
            call_args.append(name)
            if len(call_args) == 1:
                return "Growth Marketing Workforce 2"  # actually taken
            return real_resolve_unique_workforce_name(
                db, scope_type=scope_type, scope_id=scope_id, name=name
            )

        with (
            patch.object(client.app, "state", workforce_mock_app_state),
            patch.object(
                workforce_creator,
                "resolve_unique_workforce_name",
                side_effect=_lie_once_then_resolve_for_real,
            ),
        ):
            response = client.post(
                "/api/templates/growth_workforce/use-as-workforce",
                headers=admin_headers,
            )

        assert response.status_code == 200, response.text
        # The retry received the raw template name, not "...2".
        assert call_args == ["Growth Marketing Workforce"] * 2

        db = next(get_db())
        try:
            workforce = db.get(Workforce, response.json()["workforce_id"])
            assert workforce.name == "Growth Marketing Workforce 3"
        finally:
            db.close()

    def test_recovers_from_a_genuine_concurrent_worker_agent_collision(
        self, workforce_mock_app_state, admin_headers, admin_user
    ):
        """Two-session counterpart of the worker-resolution unit tests
        (which run on a shared-connection in-memory DB): the competing
        quick-access GA Analyzer is committed by a genuinely separate
        session/transaction against the file-backed test DB, and only the
        resolver's initial SELECT miss is simulated - so the INSERT
        collision, the SAVEPOINT rollback, the constraint classification,
        and the recovery re-read all run against a real concurrent
        transaction's committed row (PR #1127 re-review, m5).

        The competitor must commit BEFORE the request starts: by the time
        worker resolution runs, the request's transaction already holds
        the manager-agent/Workforce writes, and SQLite's single-writer
        model blocks any other session's commit until that transaction
        ends ("database is locked" - reproduced). Interleaving a
        mid-transaction competitor commit is only physically possible on
        PostgreSQL; this is the maximal genuine reproduction SQLite
        permits.
        """
        from sqlalchemy.orm import sessionmaker

        from xagent.web.models.database import get_engine
        from xagent.web.services import workforce_creator

        real_find = workforce_creator._find_quick_access_worker_agent
        session_factory = sessionmaker(bind=get_engine())
        competitor_agent_id: dict[str, int] = {}
        ga_lookups = {"n": 0}

        competitor_session = session_factory()
        try:
            competitor = Agent(
                user_id=admin_user["id"],
                name="GA Analyzer",
                instructions="from the concurrent winner",
                execution_mode="balanced",
                template_id="ga_analyzer",
                origin=AgentOrigin.TEMPLATE_QUICK_ACCESS.value,
                status=AgentStatus.PUBLISHED,
            )
            competitor_session.add(competitor)
            competitor_session.commit()
            competitor_agent_id["id"] = competitor.id
        finally:
            competitor_session.close()

        def _miss_once_then_delegate(db, *, user_id, template_id):
            if template_id != "ga_analyzer":
                return real_find(db, user_id=user_id, template_id=template_id)
            ga_lookups["n"] += 1
            if ga_lookups["n"] == 1:
                # Simulates the TOCTOU window: this SELECT ran before the
                # concurrent winner's commit became visible.
                return None
            return real_find(db, user_id=user_id, template_id=template_id)

        with (
            patch.object(client.app, "state", workforce_mock_app_state),
            patch.object(
                workforce_creator,
                "_find_quick_access_worker_agent",
                side_effect=_miss_once_then_delegate,
            ),
        ):
            response = client.post(
                "/api/templates/growth_workforce/use-as-workforce",
                headers=admin_headers,
            )

        assert response.status_code == 200, response.text
        # Initial miss + the post-IntegrityError recovery re-read.
        assert ga_lookups["n"] == 2

        db = next(get_db())
        try:
            workforce_id = response.json()["workforce_id"]
            worker_agent_ids = {
                w.agent_id
                for w in db.query(WorkforceAgent).filter(
                    WorkforceAgent.workforce_id == workforce_id
                )
            }
            # The concurrent winner's row was reused, not duplicated.
            assert competitor_agent_id["id"] in worker_agent_ids
            assert (
                db.query(Agent)
                .filter(
                    Agent.origin == AgentOrigin.TEMPLATE_QUICK_ACCESS.value,
                    Agent.template_id == "ga_analyzer",
                )
                .count()
                == 1
            )
        finally:
            db.close()
