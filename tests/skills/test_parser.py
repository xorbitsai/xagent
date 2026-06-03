"""
Tests for Skills Module
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from src.xagent.skills.parser import SkillParser


@pytest.fixture
def temp_skill_dir():
    """创建临时 skill 目录"""
    temp_dir = tempfile.mkdtemp()
    skill_dir = Path(temp_dir) / "test_skill"
    skill_dir.mkdir()

    # 创建 SKILL.md
    (skill_dir / "SKILL.md").write_text(
        """# Test Skill

## Description
This is a test skill for unit testing.

## When to Use
When testing is needed.

## Execution Flow
1. Do this
2. Do that
3. Done
"""
    )

    # 创建 template.md
    (skill_dir / "template.md").write_text("Template: {task}")

    yield skill_dir

    # 清理
    shutil.rmtree(temp_dir)


class TestSkillParser:
    """测试 SkillParser"""

    def test_parse_skill(self, temp_skill_dir):
        """测试解析 skill"""
        skill = SkillParser.parse(temp_skill_dir)

        assert skill["name"] == "test_skill"
        assert skill["description"] == "This is a test skill for unit testing."
        assert skill["when_to_use"] == "When testing is needed."
        assert skill["execution_flow"] == "1. Do this\n2. Do that\n3. Done"
        assert skill["template"] == "Template: {task}"
        assert "SKILL.md" in skill["files"]
        assert "template.md" in skill["files"]
        assert skill["path"] == str(temp_skill_dir)

    def test_extract_tags(self, temp_skill_dir):
        """测试标签提取"""
        skill = SkillParser.parse(temp_skill_dir)
        # 这个 skill 应该包含 testing 标签
        assert "testing" in skill["tags"] or "debugging" in skill["tags"]

    def test_parse_skill_without_skilled_md(self, tmp_path):
        """测试缺少 SKILL.md 的目录"""
        skill_dir = tmp_path / "invalid_skill"
        skill_dir.mkdir()

        with pytest.raises(ValueError, match="SKILL.md not found"):
            SkillParser.parse(skill_dir)

    def test_parse_skill_with_optional_files(self, tmp_path):
        """测试带有可选文件的 skill"""
        skill_dir = tmp_path / "complete_skill"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text(
            """
# Complete Skill

## Description
A complete skill example.

## When to Use
For testing purposes.

## Execution Flow
Follow the steps.
"""
        )

        # 创建 examples 目录
        examples_dir = skill_dir / "examples"
        examples_dir.mkdir()
        (examples_dir / "example1.md").write_text("Example 1")

        skill = SkillParser.parse(skill_dir)

        assert "examples/example1.md" in skill["files"]
        assert len(skill["files"]) == 2  # SKILL.md and examples/example1.md

    def test_parse_frontmatter_metadata_when_sections_are_missing(self, tmp_path):
        """Frontmatter metadata should feed skill selection summaries."""
        skill_dir = tmp_path / "frontmatter_skill"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text(
            """---
name: frontmatter-name
description: "Create standalone poster images."
when_to_use: "Use when the requested output is a poster or image."
tags:
  - image
  - design
---

# Frontmatter Skill

Body-only skill instructions.
"""
        )

        skill = SkillParser.parse(skill_dir)

        assert skill["name"] == "frontmatter_skill"
        assert skill["description"] == "Create standalone poster images."
        assert (
            skill["when_to_use"]
            == "Use when the requested output is a poster or image."
        )
        assert skill["tags"] == ["image", "design"]

    def test_parse_frontmatter_empty_scalar_fields(self, tmp_path):
        """Empty frontmatter scalar fields should not become list strings."""
        skill_dir = tmp_path / "empty_frontmatter_skill"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text(
            """---
description:
when_to_use:
tags:
  - rag
---

# Empty Frontmatter Skill
"""
        )

        skill = SkillParser.parse(skill_dir)

        assert skill["description"] == ""
        assert skill["when_to_use"] == ""
        assert skill["tags"] == ["rag"]

    def test_parse_frontmatter_ignores_yaml_comments(self, tmp_path):
        """Frontmatter comments should not leak into skill metadata."""
        skill_dir = tmp_path / "commented_frontmatter_skill"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text(
            """---
# Skill metadata
description: "Create # tagged assets." # inline note
when_to_use: Use for visual output. # routing note
tags:
  - image # visual
---

# Commented Frontmatter Skill
"""
        )

        skill = SkillParser.parse(skill_dir)

        assert skill["description"] == "Create # tagged assets."
        assert skill["when_to_use"] == "Use for visual output."
        assert skill["tags"] == ["image"]

    def test_parse_frontmatter_uses_yaml_parser(self, tmp_path):
        """Valid YAML frontmatter forms should populate routing metadata."""
        skill_dir = tmp_path / "yaml_frontmatter_skill"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text(
            """---
description: |
  Create assets with multiline instructions.
  Use C# examples when requested.
when_to_use: >
  Use when the task asks for visual
  output variants.
tags: [image, design]
---

# YAML Frontmatter Skill
"""
        )

        skill = SkillParser.parse(skill_dir)

        assert (
            skill["description"]
            == "Create assets with multiline instructions.\nUse C# examples when requested.\n"
        )
        assert skill["when_to_use"] == (
            "Use when the task asks for visual output variants.\n"
        )
        assert skill["tags"] == ["image", "design"]

    def test_parse_bundle_without_filesystem_directory(self):
        """DB-backed skills should parse from an in-memory file bundle."""
        skill = SkillParser.parse_bundle(
            name="db_skill",
            files={
                "SKILL.md": b"""---
description: "Database skill."
when_to_use: "Use for DB-backed skill tests."
tags: [db, test]
---

# DB Skill

## Execution Flow
Read stored blobs.
""",
                "references/guide.md": b"# Guide",
                "template.md": b"Template from DB",
            },
            path="provider://personal/db_skill",
        )

        assert skill["name"] == "db_skill"
        assert skill["path"] == "provider://personal/db_skill"
        assert skill["description"] == "Database skill."
        assert skill["when_to_use"] == "Use for DB-backed skill tests."
        assert skill["execution_flow"] == "Read stored blobs."
        assert skill["tags"] == ["db", "test"]
        assert skill["template"] == "Template from DB"
        assert skill["files"] == ["SKILL.md", "references/guide.md", "template.md"]
