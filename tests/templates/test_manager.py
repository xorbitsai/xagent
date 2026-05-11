"""
Tests for TemplateManager
"""

import shutil
import tempfile
from pathlib import Path

import pytest

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
        assert template["tags"] == []
        assert template["author"] == "Xagent"
        assert template["version"] == "1.0"
        assert template["featured"] is False
        assert template["agent_config"]["instructions"] == ""
        assert template["agent_config"]["skills"] == []
        assert template["agent_config"]["tool_categories"] == []

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

    @pytest.mark.asyncio
    async def test_nonexistent_templates_directory(self, tmp_path):
        """测试不存在的模板目录"""
        nonexistent_dir = tmp_path / "nonexistent"
        manager = TemplateManager(templates_root=nonexistent_dir)

        # 应该不抛出异常，只是记录警告
        await manager.initialize()

        templates = await manager.list_templates()

        assert len(templates) == 0
