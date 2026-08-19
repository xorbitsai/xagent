"""
Tests for TemplateManager
"""

import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from src.xagent.templates.manager import TemplateManager


@pytest.fixture
def temp_templates_dir():
    """创建临时 templates 目录"""
    temp_dir = tempfile.mkdtemp()
    templates_dir = Path(temp_dir)

    # 创建有效的模板文件
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

    # 创建无效的模板文件（缺少必需字段）
    invalid_template = templates_dir / "invalid.yaml"
    invalid_template.write_text(
        """
name: Invalid Template
descriptions:
  en: This template is missing required fields
"""
    )

    # 创建非 YAML 文件（应被忽略）
    (templates_dir / "readme.txt").write_text("This is a readme file")

    yield templates_dir

    # 清理
    shutil.rmtree(temp_dir)


class TestTemplateManager:
    """测试 TemplateManager"""

    @pytest.mark.asyncio
    async def test_initialize_and_list_templates(self, temp_templates_dir):
        """测试初始化和列出模板"""
        manager = TemplateManager(templates_root=temp_templates_dir)
        await manager.initialize()

        templates = await manager.list_templates()

        assert len(templates) == 2
        template_ids = [t["id"] for t in templates]
        assert "customer_support" in template_ids
        assert "sales_assistant" in template_ids

    @pytest.mark.asyncio
    async def test_get_template(self, temp_templates_dir):
        """测试获取单个模板"""
        manager = TemplateManager(templates_root=temp_templates_dir)
        await manager.initialize()

        template = await manager.get_template("customer_support")

        assert template is not None
        assert template["id"] == "customer_support"
        assert template["name"] == "Customer Support Agent"
        assert template["category"] == "Support"
        assert (
            template["descriptions"]["en"] == "Professional customer support assistant"
        )
        assert template["descriptions"]["zh"] == "专业的客服助手"
        assert template["author"] == "Xagent"
        assert template["version"] == "1.0"
        assert "support" in template["tags"]
        assert "customer" in template["tags"]

    @pytest.mark.asyncio
    async def test_get_template_with_agent_config(self, temp_templates_dir):
        """测试获取模板的 agent_config"""
        manager = TemplateManager(templates_root=temp_templates_dir)
        await manager.initialize()

        template = await manager.get_template("customer_support")

        assert "agent_config" in template
        assert "instructions" in template["agent_config"]
        assert (
            "customer support assistant"
            in template["agent_config"]["instructions"].lower()
        )
        assert template["agent_config"]["skills"] == ["product_knowledge"]
        assert template["agent_config"]["tool_categories"] == ["web_search"]

    @pytest.mark.asyncio
    async def test_get_nonexistent_template(self, temp_templates_dir):
        """测试获取不存在的模板"""
        manager = TemplateManager(templates_root=temp_templates_dir)
        await manager.initialize()

        template = await manager.get_template("nonexistent")

        assert template is None

    @pytest.mark.asyncio
    async def test_reload_templates(self, temp_templates_dir):
        """测试重新加载模板"""
        manager = TemplateManager(templates_root=temp_templates_dir)
        await manager.initialize()

        # 初始加载
        templates = await manager.list_templates()
        assert len(templates) == 2

        # 添加新模板
        new_template = temp_templates_dir / "data_analyst.yaml"
        new_template.write_text(
            """
id: data_analyst
name: Data Analyst
category: Data & Dev
tags:
  - data
descriptions:
  en: Data analysis expert
  zh: 数据分析专家
author: Xagent
version: "1.0"

agent_config:
  instructions: |
    You are a data analyst.
  skills: []
  tool_categories: []
"""
        )

        # 重新加载
        await manager.reload()
        templates = await manager.list_templates()

        assert len(templates) == 3
        template_ids = [t["id"] for t in templates]
        assert "data_analyst" in template_ids

    @pytest.mark.asyncio
    async def test_ensure_initialized(self, temp_templates_dir):
        """测试懒加载初始化"""
        manager = TemplateManager(templates_root=temp_templates_dir)

        # 未初始化时，has_templates 应该返回 False
        assert not manager.has_templates()

        # 调用 ensure_initialized
        await manager.ensure_initialized()

        # 初始化后，has_templates 应该返回 True
        assert manager.has_templates()

    @pytest.mark.asyncio
    async def test_parse_yaml_with_defaults(self, temp_templates_dir):
        """测试 YAML 解析时的默认值设置"""
        # 创建缺少可选字段的模板
        minimal_template = temp_templates_dir / "minimal.yaml"
        minimal_template.write_text(
            """
id: minimal_template
name: Minimal Template
category: Other
descriptions:
  en: A minimal template
  zh: 最小模板
"""
        )

        manager = TemplateManager(templates_root=temp_templates_dir)
        await manager.initialize()

        template = await manager.get_template("minimal_template")

        assert template is not None
        # tags/features are per-locale dicts (like descriptions/sample_prompts),
        # so their "not authored" default is {} - get_localized_value resolves
        # that to [] per-locale at the API layer (see test_template_data_structure).
        assert template["tags"] == {}
        assert template["features"] == {}
        assert template["author"] == "Xagent"
        assert template["version"] == "1.0"
        assert template["featured"] is False
        assert template["sample_prompts"] == {}
        assert template["agent_config"]["instructions"] == ""
        assert template["agent_config"]["skills"] == []
        assert template["agent_config"]["tool_categories"] == []

    @pytest.mark.asyncio
    async def test_get_template_with_sample_prompts(self, temp_templates_dir):
        """测试模板的 sample_prompts 能正确加载并保留本地化结构"""
        template_with_prompts = temp_templates_dir / "with_prompts.yaml"
        template_with_prompts.write_text(
            """
id: with_prompts
name: Template With Prompts
category: Support
descriptions:
  en: A template with sample prompts
  zh: 一个带有示例提示词的模板
sample_prompts:
  en:
  - title: Do the thing
    prompt: 'Do the thing: paste input.'
    highlights:
    - paste input
  - title: Do another thing
    prompt: Do another thing entirely.
    highlights: []
  zh:
  - title: 做这件事
    prompt: 做这件事：粘贴输入。
    highlights:
    - 粘贴输入
  - title: 做另一件事
    prompt: 做另一件完全不同的事。
    highlights: []
"""
        )

        manager = TemplateManager(templates_root=temp_templates_dir)
        await manager.initialize()

        template = await manager.get_template("with_prompts")

        assert template is not None
        assert len(template["sample_prompts"]["en"]) == 2
        assert template["sample_prompts"]["en"][0]["title"] == "Do the thing"
        assert template["sample_prompts"]["en"][0]["highlights"] == ["paste input"]
        assert template["sample_prompts"]["zh"][0]["title"] == "做这件事"

    @pytest.mark.asyncio
    async def test_malformed_sample_prompts_entry_is_skipped_not_fatal(
        self, temp_templates_dir
    ):
        """一个模板的 sample_prompts 条目缺少必需字段时，该模板应被跳过而不是让
        整个模板目录加载失败（避免单个作者错误导致 GET /api/templates/ 500）。"""
        bad_template = temp_templates_dir / "bad_prompts.yaml"
        bad_template.write_text(
            """
id: bad_prompts
name: Bad Prompts Template
category: Support
descriptions:
  en: A template with a malformed sample prompt
  zh: 一个带有格式错误提示词的模板
sample_prompts:
  en:
  - title: Missing the prompt field
    highlights: []
"""
        )

        manager = TemplateManager(templates_root=temp_templates_dir)
        await manager.initialize()

        template = await manager.get_template("bad_prompts")
        assert template is None

        templates = await manager.list_templates()
        template_ids = [t["id"] for t in templates]
        assert "bad_prompts" not in template_ids
        # The other, valid templates in the fixture directory still load.
        assert "customer_support" in template_ids
        assert "sales_assistant" in template_ids

    @pytest.mark.asyncio
    async def test_flat_list_sample_prompts_is_rejected(self, temp_templates_dir):
        """sample_prompts 必须按语言分组（{"en": [...]}），写成扁平列表会静默
        跳过本地化解析，因此在解析阶段就应当拒绝该写法。"""
        bad_template = temp_templates_dir / "flat_prompts.yaml"
        bad_template.write_text(
            """
id: flat_prompts
name: Flat Prompts Template
category: Support
descriptions:
  en: A template with a flat-list sample_prompts shape
  zh: 一个 sample_prompts 为扁平列表的模板
sample_prompts:
- title: Not locale-keyed
  prompt: This should have been nested under 'en'.
  highlights: []
"""
        )

        manager = TemplateManager(templates_root=temp_templates_dir)
        await manager.initialize()

        template = await manager.get_template("flat_prompts")
        assert template is None

    @pytest.mark.asyncio
    async def test_get_template_with_persona(self, temp_templates_dir):
        """persona 应正确加载并保留本地化结构，供 marketplace 卡片使用。"""
        template_with_persona = temp_templates_dir / "with_persona.yaml"
        template_with_persona.write_text(
            """
id: with_persona
name: Social Media Content Manager
category: Marketing
descriptions:
  en: A template with a persona
  zh: 一个带有 persona 的模板
persona:
  name: Maya
  role:
    en: Social Media Content Manager
    zh: 社媒内容经理
  avatar: https://example.com/maya.png
  intro:
    en: "Hi — I'm Maya, your Social Media Content Manager."
    zh: 你好，我是 Maya，你的社媒内容经理。
  kickoff_questions:
    en:
    - Which platforms are in scope?
    - Do you have brand guidelines?
    zh:
    - 涉及哪些平台？
    - 有品牌规范吗？
"""
        )

        manager = TemplateManager(templates_root=temp_templates_dir)
        await manager.initialize()

        template = await manager.get_template("with_persona")

        assert template is not None
        persona = template["persona"]
        assert persona["name"] == "Maya"
        assert persona["role"]["en"] == "Social Media Content Manager"
        assert persona["role"]["zh"] == "社媒内容经理"
        assert persona["avatar"] == "https://example.com/maya.png"
        assert len(persona["kickoff_questions"]["en"]) == 2

    @pytest.mark.asyncio
    async def test_persona_defaults_to_none_when_absent(self, temp_templates_dir):
        """没有 persona 字段的模板（如 workforce 类型）应保持 persona=None，
        而不是要求每个模板都作者一份 marketplace 卡片内容。"""
        manager = TemplateManager(templates_root=temp_templates_dir)
        await manager.initialize()

        template = await manager.get_template("customer_support")

        assert template is not None
        assert template["persona"] is None

    @pytest.mark.asyncio
    async def test_persona_role_missing_en_falls_back_to_template_name_end_to_end(
        self, temp_templates_dir
    ):
        """A persona.role authored with no 'en' key loads successfully - the
        full YAML pipeline backfills persona.role.en from the template's
        own top-level `name` (see the unit-level assertion in
        TestValidatePersona for the exact mechanics)."""
        template_file = temp_templates_dir / "no_en_persona_role.yaml"
        template_file.write_text(
            """
id: no_en_persona_role
name: No En Persona Role Template
category: Support
descriptions:
  en: A template with a persona.role missing 'en'
  zh: 一个 persona.role 缺少 'en' 的模板
persona:
  name: Nia
  role:
    zh: 某个角色
"""
        )

        manager = TemplateManager(templates_root=temp_templates_dir)
        await manager.initialize()

        template = await manager.get_template("no_en_persona_role")
        assert template is not None
        assert template["persona"]["role"]["en"] == "No En Persona Role Template"
        assert template["persona"]["role"]["zh"] == "某个角色"

    @pytest.mark.asyncio
    async def test_persona_intro_and_kickoff_without_en_key_still_optional(
        self, temp_templates_dir
    ):
        """persona.intro / persona.kickoff_questions themselves stay
        optional - omitting them entirely (not just their 'en' key) must
        not fail load. Unlike persona.role (backfilled from the template's
        `name` - see the test above), intro/kickoff have no fallback
        content, so an empty dict is the honest result of "not authored"."""
        template_file = temp_templates_dir / "persona_no_intro.yaml"
        template_file.write_text(
            """
id: persona_no_intro
name: Persona No Intro Template
category: Support
descriptions:
  en: A template whose persona has no intro or kickoff_questions at all
persona:
  name: Nia
  role:
    en: Some Role
"""
        )

        manager = TemplateManager(templates_root=temp_templates_dir)
        await manager.initialize()

        template = await manager.get_template("persona_no_intro")
        assert template is not None
        assert template["persona"]["intro"] == {}
        assert template["persona"]["kickoff_questions"] == {}

    @pytest.mark.asyncio
    async def test_invalid_persona_skips_only_that_template(self, temp_templates_dir):
        """The end-to-end half of what TestValidatePersona covers at the
        unit level: a persona that fails validation must make the loader
        skip that one template (like any other parse failure), not crash
        the whole directory load or slip through unvalidated."""
        bad_template = temp_templates_dir / "flat_persona_role.yaml"
        bad_template.write_text(
            """
id: flat_persona_role
name: Flat Persona Role Template
category: Support
descriptions:
  en: A template with a flat persona.role
persona:
  name: Nia
  role: Some Role
"""
        )

        manager = TemplateManager(templates_root=temp_templates_dir)
        await manager.initialize()

        assert await manager.get_template("flat_persona_role") is None
        # The sibling fixture templates in temp_templates_dir still load.
        assert await manager.list_templates()

    @pytest.mark.asyncio
    async def test_non_string_template_name_skips_the_template(
        self, temp_templates_dir
    ):
        """A non-string top-level `name` (e.g. YAML parsing `name: 2026` as
        an int) must fail at load time. Before this check, it was only
        tested for *presence*, so it flowed into the persona.role.en
        backfill and blew up as a pydantic ValidationError - a 500 on the
        whole list endpoint - the first time a response was built from it
        (PR #1498 round-2 review, M2)."""
        template_file = temp_templates_dir / "int_name.yaml"
        template_file.write_text(
            """
id: int_name
name: 2026
category: Support
descriptions:
  en: A template whose name YAML-parses as an int
persona:
  name: Nia
"""
        )

        manager = TemplateManager(templates_root=temp_templates_dir)
        await manager.initialize()

        assert await manager.get_template("int_name") is None

    @pytest.mark.asyncio
    async def test_template_name_is_stripped_before_the_role_en_backfill(
        self, temp_templates_dir
    ):
        """The top-level `name` is stripped at load time, so the
        persona.role.en backfilled from it never carries surrounding
        whitespace that an authored (always-stripped) role.zh would not."""
        template_file = temp_templates_dir / "padded_name.yaml"
        template_file.write_text(
            """
id: padded_name
name: "  Padded Name Template  "
category: Support
descriptions:
  en: A template whose name has surrounding whitespace
persona:
  name: Nia
"""
        )

        manager = TemplateManager(templates_root=temp_templates_dir)
        await manager.initialize()

        template = await manager.get_template("padded_name")
        assert template is not None
        assert template["name"] == "Padded Name Template"
        assert template["persona"]["role"]["en"] == "Padded Name Template"

    @pytest.mark.asyncio
    async def test_workforce_template_with_a_persona_is_rejected(self, tmp_path):
        """persona is an agent-template concept - a workforce card renders
        from workforce_config instead (see _validate_persona's docstring).
        A workforce template authoring one anyway must fail to load rather
        than have the persona validated, backfilled, and served in silent
        contradiction of that contract (PR #1498 round-2 review, N3)."""
        (tmp_path / "persona_workforce.yaml").write_text(
            """
id: persona_workforce
name: Persona Workforce
category: Marketing
type: workforce
descriptions:
  en: A workforce template that wrongly authors a persona.
persona:
  name: Nia
workforce_config:
  manager:
    name: Growth Manager
    instructions: Orchestrate.
  agents:
  - template_id: ga-analyzer
    name: GA Analyzer
    assignment_instructions: Measure performance.
"""
        )
        manager = TemplateManager(templates_root=tmp_path)
        await manager.initialize()

        assert await manager.get_template("persona_workforce") is None

    @pytest.mark.asyncio
    async def test_skip_invalid_templates(self, temp_templates_dir):
        """测试跳过无效的模板文件"""
        manager = TemplateManager(templates_root=temp_templates_dir)
        await manager.initialize()

        # invalid.yaml 缺少必需字段，应该被跳过
        # readme.txt 不是 YAML 文件，应该被忽略
        templates = await manager.list_templates()

        assert len(templates) == 2
        template_ids = [t["id"] for t in templates]
        assert "invalid" not in template_ids

    @pytest.mark.asyncio
    async def test_empty_templates_directory(self, tmp_path):
        """测试空的模板目录"""
        manager = TemplateManager(templates_root=tmp_path)
        await manager.initialize()

        templates = await manager.list_templates()

        assert len(templates) == 0
        assert not manager.has_templates()

    def test_builtin_templates_that_use_web_search_select_web_search_category(self):
        built_in_dir = (
            Path(__file__).resolve().parents[2] / "src/xagent/templates/built_in"
        )
        markers = (
            "web_search",
            "zhipu_web_search",
            "exa_web_search",
            "tavily_web_search",
            "web research",
            "web search",
        )
        offenders: list[str] = []

        for template_file in built_in_dir.glob("*.yaml"):
            data = yaml.safe_load(template_file.read_text(encoding="utf-8")) or {}
            agent_config = data.get("agent_config") or {}
            instructions = str(agent_config.get("instructions") or "").lower()
            tool_categories = agent_config.get("tool_categories") or []
            if any(marker in instructions for marker in markers):
                if "web_search" not in tool_categories:
                    offenders.append(template_file.name)

        assert not offenders

    def test_builtin_templates_directory_loads_every_file_without_silent_skips(self):
        """TemplateManager.reload() silently skips a file that fails to parse or is
        missing a required field (see manager.py's except/continue in reload()). Assert
        every real built_in/*.yaml is actually indexed, so a broken new template fails
        this test instead of just vanishing from the UI at runtime."""
        built_in_dir = (
            Path(__file__).resolve().parents[2] / "src/xagent/templates/built_in"
        )
        yaml_files = list(built_in_dir.glob("*.yaml"))
        assert yaml_files, "expected at least one built-in template file"

        manager = TemplateManager(templates_root=built_in_dir)
        ids = {manager._parse_yaml_file(f)["id"] for f in yaml_files}

        assert len(ids) == len(yaml_files)
        for expected_id in (
            "sales-meeting-agent",
            "marketing-google-analytics-analyzer",
            "marketing-google-ads-recommendation",
            "operations-devops-ai-agent",
        ):
            assert expected_id in ids

    def test_builtin_agent_templates_all_have_a_persona(self):
        """Every 'agent'-type built-in template must carry a marketplace
        persona - a newly added agent template with no persona would
        otherwise silently ship a nameless, avatar-less marketplace card
        with no test catching the gap. The single 'workforce'-type
        template is deliberately excluded (its card renders from
        workforce_config instead - see TemplateManager._enrich_template)."""
        built_in_dir = (
            Path(__file__).resolve().parents[2] / "src/xagent/templates/built_in"
        )
        manager = TemplateManager(templates_root=built_in_dir)
        offenders = [
            template_file.name
            for template_file in built_in_dir.glob("*.yaml")
            if (data := manager._parse_yaml_file(template_file)).get("type", "agent")
            == "agent"
            and data.get("persona") is None
        ]

        assert not offenders

    def test_builtin_persona_avatars_resolve_to_committed_files(self):
        """persona.avatar is an app-relative path resolved against the
        Next.js frontend's own static assets (frontend/public/...), a
        distinct convention from Agent.logo_url's /uploads/agent_logos
        path or connections[].logo's external hotlinks (PR #1498 review,
        M2/M3) - nothing ties it to a real committed file, so a future
        rename or a typo'd path would ship a 404 image with no test
        failure. Mirrors the existing connections/MCP-app invariant test
        below for the same class of drift."""
        repo_root = Path(__file__).resolve().parents[2]
        built_in_dir = repo_root / "src/xagent/templates/built_in"
        frontend_public_dir = repo_root / "frontend/public"
        manager = TemplateManager(templates_root=built_in_dir)

        offenders: list[str] = []
        for template_file in built_in_dir.glob("*.yaml"):
            persona = manager._parse_yaml_file(template_file).get("persona")
            if not persona or not persona.get("avatar"):
                continue
            avatar_path = persona["avatar"]
            if not avatar_path.startswith("/"):
                offenders.append(
                    f"{template_file.name}: not app-relative: {avatar_path}"
                )
                continue
            if not (frontend_public_dir / avatar_path.lstrip("/")).is_file():
                offenders.append(f"{template_file.name}: missing file: {avatar_path}")

        assert not offenders

    def test_builtin_persona_names_and_avatars_are_unique(self):
        """Two templates sharing a persona name or avatar would make two
        distinct marketplace cards look like the same AI teammate."""
        built_in_dir = (
            Path(__file__).resolve().parents[2] / "src/xagent/templates/built_in"
        )
        manager = TemplateManager(templates_root=built_in_dir)

        names: list[str] = []
        avatars: list[str] = []
        for template_file in built_in_dir.glob("*.yaml"):
            persona = manager._parse_yaml_file(template_file).get("persona")
            if not persona:
                continue
            names.append(persona["name"])
            if persona.get("avatar"):
                avatars.append(persona["avatar"])

        assert len(names) == len(set(names)), f"duplicate persona names: {names}"
        assert len(avatars) == len(set(avatars)), (
            f"duplicate persona avatars: {avatars}"
        )

    def test_builtin_template_connections_resolve_to_a_registered_mcp_app(self):
        """A connections[].name that the build wizard can't resolve to a registered
        built-in MCP app (src/xagent/web/builtin_mcp_registry.py) makes its Configure
        step un-completable (see PR #1023 review). Mirror the frontend's lenient
        name/app_id matching (lowercase + trim, plus a hyphen-for-space variant) so
        this doesn't false-flag entries the wizard would actually resolve."""
        from src.xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

        def lookup_keys(*values):
            keys = set()
            for value in values:
                normalized = str(value or "").strip().lower()
                if not normalized:
                    continue
                keys.add(normalized)
                keys.add(normalized.replace(" ", "-"))
            return keys

        registered_keys = set()
        for row in get_builtin_public_mcp_app_rows():
            registered_keys |= lookup_keys(row.get("name"), row.get("app_id"))

        built_in_dir = (
            Path(__file__).resolve().parents[2] / "src/xagent/templates/built_in"
        )
        offenders: list[str] = []

        for template_file in built_in_dir.glob("*.yaml"):
            data = yaml.safe_load(template_file.read_text(encoding="utf-8")) or {}
            for conn in data.get("connections") or []:
                name = conn.get("name") if isinstance(conn, dict) else conn
                if name and not (lookup_keys(name) & registered_keys):
                    offenders.append(f"{template_file.name}: {name}")

        assert not offenders

    def test_builtin_workforce_connections_are_union_of_sub_template_connections(self):
        """A `type: workforce` template's `connections:` is hand-maintained
        display-only data (see the comment above `connections:` in
        marketing-growth-marketing-workforce.yaml) - nothing in
        `TemplateManager` derives or validates it against the sub-templates
        it actually references. That invariant has drifted and been
        hand-fixed at least twice already (PR #1127 re-review, F7; and
        again when the Google Analytics connector was added to
        marketing-google-analytics-analyzer). Assert it generically for
        every built-in workforce template so this class of drift is caught
        automatically instead of relying on a reviewer to notice."""
        built_in_dir = (
            Path(__file__).resolve().parents[2] / "src/xagent/templates/built_in"
        )

        def connection_names(data):
            return {
                conn.get("name") if isinstance(conn, dict) else conn
                for conn in data.get("connections") or []
            }

        templates_by_id = {}
        for template_file in built_in_dir.glob("*.yaml"):
            data = yaml.safe_load(template_file.read_text(encoding="utf-8")) or {}
            template_id = data.get("id")
            if template_id:
                templates_by_id[template_id] = data

        offenders: list[str] = []
        for template_id, data in templates_by_id.items():
            if data.get("type") != "workforce":
                continue

            workforce_config = data.get("workforce_config") or {}
            sub_template_ids = [
                agent.get("template_id")
                for agent in workforce_config.get("agents") or []
            ]
            expected = set()
            for sub_template_id in sub_template_ids:
                sub_template = templates_by_id.get(sub_template_id)
                if sub_template is not None:
                    expected |= connection_names(sub_template)

            actual = connection_names(data)
            if actual != expected:
                offenders.append(
                    f"{template_id}: connections {sorted(actual)} != union of "
                    f"sub-template connections {sorted(expected)}"
                )

        assert not offenders, "\n".join(offenders)

    @pytest.mark.asyncio
    async def test_builtin_ga_analyzer_preconfigures_google_analytics_connector(self):
        """The GA Analyzer template must ship with the Google Analytics
        connector preconfigured: declared under `connections:` (so the card
        advertises it and the build wizard preconnects it) and merged into
        the enriched agent_config.tool_categories as an `mcp:` entry (so an
        agent created from the template actually gets the connector). Its
        instructions promise a live GA4 pull - without the connector that
        promise is dead on arrival."""
        built_in_dir = (
            Path(__file__).resolve().parents[2] / "src/xagent/templates/built_in"
        )
        manager = TemplateManager(templates_root=built_in_dir)
        template = await manager.get_template("marketing-google-analytics-analyzer")

        assert template is not None
        connection_names = [
            conn.get("name")
            for conn in template["connections"]
            if isinstance(conn, dict)
        ]
        assert "Google Analytics" in connection_names
        assert "mcp:Google Analytics" in template["agent_config"]["tool_categories"]

    def test_builtin_sample_prompt_highlights_are_literal_substrings(self):
        """Every highlight must be a literal substring of its own prompt, in
        both locales - otherwise the frontend's highlight matching
        (replaceFirstOccurrence) silently underlines nothing, or the wrong
        word if an earlier literal duplicate shadows the intended placeholder
        (this caught a real bug: see marketing-content-agent and
        marketing-seo-brief-writer's original prompt wording)."""
        built_in_dir = (
            Path(__file__).resolve().parents[2] / "src/xagent/templates/built_in"
        )
        offenders: list[str] = []

        for template_file in sorted(built_in_dir.glob("*.yaml")):
            data = yaml.safe_load(template_file.read_text(encoding="utf-8")) or {}
            sample_prompts = data.get("sample_prompts") or {}
            if not isinstance(sample_prompts, dict):
                offenders.append(
                    f"{template_file.name}: sample_prompts is not a per-locale dict"
                )
                continue

            for locale, prompts in sample_prompts.items():
                for entry in prompts or []:
                    prompt_text = entry.get("prompt") or ""
                    for highlight in entry.get("highlights") or []:
                        if highlight not in prompt_text:
                            offenders.append(
                                f"{template_file.name} [{locale}]: highlight "
                                f"{highlight!r} not found in prompt {prompt_text!r}"
                            )

        assert not offenders, "\n".join(offenders)

    def test_builtin_templates_cap_sample_prompts_at_two(self):
        """The Task-page quick-access grid only ever renders the first 2
        sample prompts per template (TemplateQuickAccess.tsx), so authoring
        more than that on a built-in template is dead content that silently
        never shows - catch it at authoring time instead."""
        built_in_dir = (
            Path(__file__).resolve().parents[2] / "src/xagent/templates/built_in"
        )
        offenders: list[str] = []

        for template_file in sorted(built_in_dir.glob("*.yaml")):
            data = yaml.safe_load(template_file.read_text(encoding="utf-8")) or {}
            sample_prompts = data.get("sample_prompts") or {}
            if not isinstance(sample_prompts, dict):
                continue
            for locale, prompts in sample_prompts.items():
                if isinstance(prompts, list) and len(prompts) > 2:
                    offenders.append(
                        f"{template_file.name} [{locale}]: {len(prompts)} sample_prompts (max 2 are shown)"
                    )

        assert not offenders, "\n".join(offenders)

    @pytest.mark.asyncio
    async def test_nonexistent_templates_directory(self, tmp_path):
        """测试不存在的模板目录"""
        nonexistent_dir = tmp_path / "nonexistent"
        manager = TemplateManager(templates_root=nonexistent_dir)

        # 应该不抛出异常，只是记录警告
        await manager.initialize()

        templates = await manager.list_templates()

        assert len(templates) == 0

    # ----- workforce_config validation (PR #1127 review: zero coverage) -----

    @staticmethod
    def _valid_workforce_config():
        return {
            "manager": {
                "name": "Growth Manager",
                "instructions": "Orchestrate.",
                "execution_mode": "think",
                "tool_categories": ["basic"],
                "skills": [],
            },
            "agents": [
                {
                    "template_id": "ga-analyzer",
                    "name": "GA Analyzer",
                    "assignment_instructions": "Measure performance.",
                },
                {
                    "template_id": "ads-recommendation",
                    "name": "Ads Recommendation",
                    "assignment_instructions": "Recommend spend changes.",
                },
            ],
        }

    def test_valid_workforce_config_does_not_raise(self, tmp_path):
        manager = TemplateManager(templates_root=tmp_path)
        manager._validate_workforce_config(self._valid_workforce_config())

    @pytest.mark.parametrize(
        "mutate,match",
        [
            (lambda c: "not a dict", "must be a mapping"),
            (lambda c: {**c, "manager": None}, "manager.*must be a mapping"),
            (
                lambda c: {**c, "manager": {"name": "M", "instructions": "  "}},
                "manager.instructions",
            ),
            (lambda c: {**c, "manager": {"instructions": "Go"}}, "manager.name"),
            # Strict isinstance(str) - str(...) coercion used to let YAML
            # lists/ints through load-time validation, only to surface as
            # garbage prompts/ids (or an AttributeError -> 500 for the
            # normalize_text'd optional fields) at instantiation time
            # (PR #1127 re-review, m1).
            (
                lambda c: {
                    **c,
                    "manager": {**c["manager"], "instructions": ["not", "a", "str"]},
                },
                "manager.instructions",
            ),
            (
                lambda c: {**c, "manager": {**c["manager"], "description": 123}},
                "manager.description",
            ),
            (
                lambda c: {**c, "agents": [{**c["agents"][0], "template_id": 42}]},
                r"agents\[0\]\.template_id",
            ),
            (
                lambda c: {**c, "agents": [{**c["agents"][0], "alias": ["GA"]}]},
                r"agents\[0\]\.alias",
            ),
            (lambda c: {**c, "agents": None}, "agents.*non-empty list"),
            (lambda c: {**c, "agents": []}, "agents.*non-empty list"),
            (lambda c: {**c, "agents": ["not a dict"]}, r"agents\[0\].*mapping"),
            (
                lambda c: {**c, "agents": [{**c["agents"][0], "template_id": ""}]},
                r"agents\[0\]\.template_id",
            ),
            (
                lambda c: {**c, "agents": [{**c["agents"][0], "name": ""}]},
                r"agents\[0\]\.name",
            ),
            (
                lambda c: {
                    **c,
                    "agents": [{**c["agents"][0], "assignment_instructions": ""}],
                },
                r"agents\[0\]\.assignment_instructions",
            ),
            (
                lambda c: {
                    **c,
                    "agents": [
                        c["agents"][0],
                        {
                            **c["agents"][1],
                            "template_id": c["agents"][0]["template_id"],
                        },
                    ],
                },
                "duplicate template_id",
            ),
            (
                lambda c: {
                    **c,
                    "manager": {**c["manager"], "execution_mode": "sonic"},
                },
                "manager.execution_mode",
            ),
            (
                lambda c: {
                    **c,
                    "manager": {**c["manager"], "tool_categories": "basic"},
                },
                "manager.tool_categories",
            ),
            (
                lambda c: {**c, "manager": {**c["manager"], "skills": [123]}},
                "manager.skills",
            ),
        ],
    )
    def test_invalid_workforce_config_raises(self, tmp_path, mutate, match):
        manager = TemplateManager(templates_root=tmp_path)
        config = mutate(self._valid_workforce_config())
        with pytest.raises(ValueError, match=match):
            manager._validate_workforce_config(config)

    def test_workforce_config_strings_are_normalized_in_place(self, tmp_path):
        """Validation strips string fields in place. A template_id with
        surrounding whitespace used to be stripped for the duplicate check
        but looked up RAW at instantiation time, guaranteeing a
        "references an unknown template" 400 the moment anyone used the
        template (PR #1127 re-review, m1)."""
        manager = TemplateManager(templates_root=tmp_path)
        config = self._valid_workforce_config()
        config["agents"][0]["template_id"] = "  ga-analyzer  "
        config["manager"]["name"] = " Growth Manager "

        manager._validate_workforce_config(config)

        assert config["agents"][0]["template_id"] == "ga-analyzer"
        assert config["manager"]["name"] == "Growth Manager"

    @pytest.mark.asyncio
    async def test_workforce_type_template_requires_workforce_config(self, tmp_path):
        """A `type: workforce` template with no workforce_config at all must
        fail to load (not silently become a broken, config-less template)."""
        (tmp_path / "broken_workforce.yaml").write_text(
            """
id: broken_workforce
name: Broken Workforce
category: Marketing
type: workforce
descriptions:
  en: Missing workforce_config entirely.
"""
        )
        manager = TemplateManager(templates_root=tmp_path)
        await manager.initialize()

        assert await manager.get_template("broken_workforce") is None

    @pytest.mark.asyncio
    async def test_agent_type_template_never_sees_workforce_validation(self, tmp_path):
        """type defaults to 'agent' and must not trip workforce_config
        validation just because a template happens to omit it (the common
        case for every pre-existing template)."""
        (tmp_path / "plain_agent.yaml").write_text(
            """
id: plain_agent
name: Plain Agent
category: General
descriptions:
  en: A normal single-agent template.
agent_config:
  instructions: You help with things.
"""
        )
        manager = TemplateManager(templates_root=tmp_path)
        await manager.initialize()

        template = await manager.get_template("plain_agent")
        assert template is not None
        assert template["type"] == "agent"
        assert template["workforce_config"] is None
        assert template["agent_config"]["instructions"]

    # ----- _warn_on_dangling_workforce_references -----

    @pytest.mark.asyncio
    async def test_dangling_workforce_reference_warns_but_still_loads(
        self, tmp_path, caplog
    ):
        (tmp_path / "wf.yaml").write_text(
            """
id: wf
name: WF
category: Marketing
type: workforce
descriptions:
  en: A workforce with a typo'd worker reference.
workforce_config:
  manager:
    name: Manager
    instructions: Orchestrate.
  agents:
  - template_id: does-not-exist
    name: Ghost Worker
    assignment_instructions: Do the thing.
"""
        )
        manager = TemplateManager(templates_root=tmp_path)
        with caplog.at_level("WARNING"):
            await manager.initialize()

        assert await manager.get_template("wf") is not None
        assert any(
            "does-not-exist" in record.message and record.levelname == "WARNING"
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_workforce_referencing_another_workforce_warns(
        self, tmp_path, caplog
    ):
        """agents[].template_id must resolve to an 'agent'-type template - a
        workforce referencing another workforce as a worker would crash at
        instantiation time (that template's agent_config is null), so this
        must be flagged at load time too, not just guarded at the point of
        use."""
        (tmp_path / "inner.yaml").write_text(
            """
id: inner
name: Inner
category: Marketing
type: workforce
descriptions:
  en: Another workforce, wrongly used as a worker below.
workforce_config:
  manager:
    name: Inner Manager
    instructions: Orchestrate.
  agents:
  - template_id: leaf
    name: Leaf
    assignment_instructions: Do the thing.
"""
        )
        (tmp_path / "leaf.yaml").write_text(
            """
id: leaf
name: Leaf
category: Marketing
descriptions:
  en: A normal single-agent template.
agent_config:
  instructions: You help with things.
"""
        )
        (tmp_path / "outer.yaml").write_text(
            """
id: outer
name: Outer
category: Marketing
type: workforce
descriptions:
  en: References 'inner' (a workforce) as if it were a single agent.
workforce_config:
  manager:
    name: Outer Manager
    instructions: Orchestrate.
  agents:
  - template_id: inner
    name: Nested Workforce
    assignment_instructions: Do the thing.
"""
        )
        manager = TemplateManager(templates_root=tmp_path)
        with caplog.at_level("WARNING"):
            await manager.initialize()

        assert any(
            "outer" in record.message
            and "inner" in record.message
            and "not 'agent'" in record.message
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_builtin_workforce_template_references_resolve(self):
        """The real shipped YAML never loaded in any test before this - a
        typo'd `workforce_config.agents[].template_id` on the built-in
        Growth Marketing Workforce template would not have failed CI. Load
        the actual built_in/ directory and assert every workforce
        template's references resolve to a loaded 'agent'-type template."""
        built_in_dir = (
            Path(__file__).resolve().parents[2] / "src/xagent/templates/built_in"
        )
        manager = TemplateManager(templates_root=built_in_dir)
        await manager.initialize()
        templates = {t["id"]: t for t in await manager.list_templates()}

        workforce_templates = [
            t for t in templates.values() if t["type"] == "workforce"
        ]
        assert workforce_templates, "expected at least one built-in workforce template"

        offenders: list[str] = []
        for template in workforce_templates:
            workforce_config = template.get("workforce_config") or {}
            for agent in workforce_config.get("agents") or []:
                referenced_id = agent.get("template_id")
                referenced = templates.get(referenced_id)
                if referenced is None:
                    offenders.append(
                        f"{template['id']} -> unknown template_id {referenced_id!r}"
                    )
                elif referenced.get("type", "agent") != "agent":
                    offenders.append(
                        f"{template['id']} -> {referenced_id!r} has type="
                        f"{referenced.get('type')!r}, not 'agent'"
                    )

        assert not offenders, "\n".join(offenders)


class TestValidatePersona:
    """Unit tests calling TemplateManager._validate_persona directly - the
    same stronger pattern TestTemplateManager.test_invalid_workforce_config_raises
    already uses for _validate_workforce_config, so each rejection asserts
    the actual error message (not just "the template failed to load",
    which passes for any unrelated parse failure too)."""

    @pytest.fixture
    def manager(self, tmp_path):
        return TemplateManager(templates_root=tmp_path)

    def test_none_is_a_no_op(self, manager):
        """persona is optional; None must not raise."""
        manager._validate_persona(None)

    def test_not_a_mapping_is_rejected(self, manager):
        for bad_value in ("a string", ["a", "list"], 42):
            with pytest.raises(ValueError, match="persona.*must be a mapping"):
                manager._validate_persona(bad_value)

    def test_unknown_key_is_rejected(self, manager):
        """A typo'd or misspelled key (`avator`, `kickoff_question` singular)
        must fail loudly rather than silently parsing as an absent field -
        the whole point of `persona` being a real, validated schema."""
        persona = {"name": "Nia", "avator": "/marketplace/avatars/nia.png"}
        with pytest.raises(ValueError, match=r"unknown key\(s\).*avator"):
            manager._validate_persona(persona)

    def test_missing_name_is_rejected(self, manager):
        with pytest.raises(ValueError, match="persona.name"):
            manager._validate_persona({"role": {"en": "Some Role"}})

    def test_name_whitespace_is_stripped_in_place(self, manager):
        persona = {"name": "  Nia  "}
        manager._validate_persona(persona)
        assert persona["name"] == "Nia"

    def test_role_flat_string_is_rejected(self, manager):
        """persona.role must be locale-keyed ({"en": ...}); a flat string
        would silently bypass localization if it were allowed through."""
        with pytest.raises(ValueError, match="persona.role"):
            manager._validate_persona({"name": "Nia", "role": "Some Role"})

    def test_role_empty_locale_value_is_rejected(self, manager):
        with pytest.raises(ValueError, match="persona.role.en"):
            manager._validate_persona({"name": "Nia", "role": {"en": "   "}})

    def test_role_explicit_null_is_treated_as_absent(self, manager):
        """`role:` (YAML null) means "not provided", same as omitting the
        key - it must not reach the isinstance(role, dict) check as None."""
        persona = {"name": "Nia", "role": None}
        manager._validate_persona(persona)
        assert persona["role"] == {}

    def test_role_falsy_junk_is_still_rejected(self, manager):
        """Only None gets the treated-as-absent leniency: falsy junk like
        `role: ""` or `role: []` is a type error, not "not provided" - an
        `or {}` null-check would silently swallow these."""
        for junk in ("", []):
            with pytest.raises(ValueError, match="persona.role"):
                manager._validate_persona({"name": "Nia", "role": junk})

    def test_role_values_are_stripped_in_place(self, manager):
        persona = {"name": "Nia", "role": {"en": "  Some Role  "}}
        manager._validate_persona(persona)
        assert persona["role"]["en"] == "Some Role"

    def test_avatar_defaults_to_none_when_omitted(self, manager):
        persona = {"name": "Nia"}
        manager._validate_persona(persona)
        assert persona["avatar"] is None

    def test_avatar_wrong_type_is_rejected(self, manager):
        with pytest.raises(ValueError, match="persona.avatar"):
            manager._validate_persona({"name": "Nia", "avatar": 42})

    def test_avatar_empty_string_is_rejected(self, manager):
        with pytest.raises(ValueError, match="persona.avatar"):
            manager._validate_persona({"name": "Nia", "avatar": "   "})

    def test_avatar_is_stripped_in_place(self, manager):
        persona = {"name": "Nia", "avatar": "  /marketplace/avatars/nia.png  "}
        manager._validate_persona(persona)
        assert persona["avatar"] == "/marketplace/avatars/nia.png"

    def test_intro_flat_string_is_rejected(self, manager):
        with pytest.raises(ValueError, match="persona.intro"):
            manager._validate_persona({"name": "Nia", "intro": "Hi there"})

    def test_intro_explicit_null_is_treated_as_absent(self, manager):
        """Same null-vs-absent hazard as persona.role above, for the field
        the PR review specifically flagged (m2): `intro:` (YAML null) used
        to reach the isinstance check as None and raise the wrong message
        ("must be a dict... not a flat string") instead of being treated
        as simply not authored."""
        persona = {"name": "Nia", "intro": None}
        manager._validate_persona(persona)
        assert persona["intro"] == {}

    def test_intro_missing_en_is_rejected(self, manager):
        """persona.intro, once authored at all, must include 'en' -
        get_localized_value's fallback-to-'en' would otherwise resolve to
        "" for an English requester, silently seeding a blank opening
        message instead of failing loudly here at load time."""
        persona = {"name": "Nia", "intro": {"zh": "你好，我是 Nia。"}}
        with pytest.raises(ValueError, match="persona.intro.*'en'"):
            manager._validate_persona(persona)

    def test_intro_empty_locale_value_is_rejected(self, manager):
        with pytest.raises(ValueError, match="persona.intro.en"):
            manager._validate_persona({"name": "Nia", "intro": {"en": "  "}})

    def test_intro_values_are_stripped_in_place(self, manager):
        persona = {"name": "Nia", "intro": {"en": "  Hi there  "}}
        manager._validate_persona(persona)
        assert persona["intro"]["en"] == "Hi there"

    def test_kickoff_questions_flat_list_is_rejected(self, manager):
        """The likeliest real authoring mistake: writing kickoff_questions
        as a plain list instead of locale-keyed, the same class of bug
        `test_flat_list_sample_prompts_is_rejected` guards against for
        sample_prompts."""
        persona = {
            "name": "Nia",
            "kickoff_questions": ["What platforms are in scope?"],
        }
        with pytest.raises(ValueError, match="persona.kickoff_questions"):
            manager._validate_persona(persona)

    def test_kickoff_questions_explicit_null_is_treated_as_absent(self, manager):
        persona = {"name": "Nia", "kickoff_questions": None}
        manager._validate_persona(persona)
        assert persona["kickoff_questions"] == {}

    def test_kickoff_questions_empty_flat_list_is_still_rejected(self, manager):
        """An empty flat list is the same shape mistake as a populated one
        (test_kickoff_questions_flat_list_is_rejected) - being falsy must
        not earn it the None-only treated-as-absent leniency."""
        with pytest.raises(ValueError, match="persona.kickoff_questions"):
            manager._validate_persona({"name": "Nia", "kickoff_questions": []})

    def test_kickoff_questions_missing_en_is_rejected(self, manager):
        persona = {
            "name": "Nia",
            "kickoff_questions": {"zh": ["第一个问题？"]},
        }
        with pytest.raises(ValueError, match="persona.kickoff_questions.*'en'"):
            manager._validate_persona(persona)

    def test_kickoff_questions_empty_item_is_rejected(self, manager):
        persona = {"name": "Nia", "kickoff_questions": {"en": ["  "]}}
        with pytest.raises(ValueError, match="persona.kickoff_questions.en"):
            manager._validate_persona(persona)

    def test_kickoff_questions_non_string_item_is_rejected(self, manager):
        persona = {"name": "Nia", "kickoff_questions": {"en": [42]}}
        with pytest.raises(ValueError, match="persona.kickoff_questions.en"):
            manager._validate_persona(persona)

    def test_kickoff_questions_items_are_stripped_in_place(self, manager):
        persona = {
            "name": "Nia",
            "kickoff_questions": {"en": ["  What platforms?  "]},
        }
        manager._validate_persona(persona)
        assert persona["kickoff_questions"]["en"] == ["What platforms?"]
