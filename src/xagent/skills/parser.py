"""
Skill Parser - Parse SKILL.md and related files
"""

import re
from pathlib import Path
from typing import Dict, List

import yaml


class SkillParser:
    """Parse SKILL.md files"""

    @staticmethod
    def parse(skill_dir: Path) -> Dict:
        """
        Parse skill directory

        Args:
            skill_dir: Skill directory path

        Returns:
            {
                "name": "code_reviewer",
                "path": "/path/to/skill",
                "description": "Skill description",
                "when_to_use": "Usage scenario",
                "template": "Template content or empty",
                "execution_flow": "Execution flow",
                "tags": ["code", "review"],
                "files": ["SKILL.md", "template.md"]
            }

        Raises:
            ValueError: If SKILL.md does not exist
        """
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            raise ValueError(f"SKILL.md not found in {skill_dir}")

        files = {
            str(file_path.relative_to(skill_dir)).replace(
                "\\", "/"
            ): file_path.read_bytes()
            for file_path in skill_dir.rglob("*")
            if file_path.is_file()
        }
        return SkillParser.parse_bundle(
            name=skill_dir.name,
            files=files,
            path=str(skill_dir),
        )

    @staticmethod
    def parse_bundle(
        name: str,
        files: Dict[str, bytes],
        path: str = "",
        encoding: str = "utf-8",
    ) -> Dict:
        """Parse a skill from an in-memory file bundle."""
        if "SKILL.md" not in files:
            raise ValueError(f"SKILL.md not found in {path or name}")

        content = files["SKILL.md"].decode(encoding)
        template_bytes = files.get("template.md")
        template_content = template_bytes.decode(encoding) if template_bytes else ""

        frontmatter = SkillParser._extract_frontmatter(content)
        section_description = SkillParser._extract_section(content, "Description")
        section_when_to_use = SkillParser._extract_section(content, "When to Use")
        section_tags = SkillParser._extract_tags(content)
        frontmatter_tags = SkillParser._frontmatter_string_list(frontmatter, "tags")
        tags = frontmatter_tags or section_tags

        return {
            "name": name,
            "path": path,
            "content": content,  # Complete SKILL.md content
            "template": template_content,  # template.md content (if exists)
            "description": section_description
            or SkillParser._frontmatter_string(frontmatter, "description"),
            "when_to_use": section_when_to_use
            or SkillParser._frontmatter_string(frontmatter, "when_to_use"),
            "execution_flow": SkillParser._extract_section(content, "Execution Flow"),
            "tags": tags,
            "files": sorted(files),
        }

    @staticmethod
    def _extract_frontmatter(content: str) -> Dict:
        """Extract YAML frontmatter from a skill file."""
        stripped = content.lstrip()
        if not stripped.startswith("---"):
            return {}

        lines = stripped.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}

        end_index = None
        for index, line in enumerate(lines[1:], start=1):
            if line.strip().startswith("---"):
                end_index = index
                break
        if end_index is None:
            return {}

        frontmatter_text = "\n".join(lines[1:end_index])
        try:
            metadata = yaml.safe_load(frontmatter_text)
        except yaml.YAMLError:
            return {}

        return metadata if isinstance(metadata, dict) else {}

    @staticmethod
    def _frontmatter_string(frontmatter: Dict, key: str) -> str:
        """Return a scalar frontmatter field or an empty string."""
        value = frontmatter.get(key)
        return value if isinstance(value, str) else ""

    @staticmethod
    def _frontmatter_string_list(frontmatter: Dict, key: str) -> List[str]:
        """Return a list of string frontmatter values or an empty list."""
        value = frontmatter.get(key)
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str)]

    @staticmethod
    def _extract_section(content: str, section_name: str) -> str:
        """Extract section content"""
        pattern = rf"## {section_name}\s*\n(.*?)(?=\n##|\Z)"
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _list_files(skill_dir: Path) -> List[str]:
        """List all files in skill directory"""
        files = []
        for file_path in skill_dir.rglob("*"):
            if file_path.is_file():
                files.append(str(file_path.relative_to(skill_dir)))
        return sorted(files)

    @staticmethod
    def _extract_tags(content: str) -> List[str]:
        """Extract tags from content"""
        tags = []
        content_lower = content.lower()

        tag_keywords = {
            "code": ["code", "programming", "development"],
            "testing": ["test", "testing", "verify"],
            "security": ["security", "audit"],
            "documentation": ["document", "docs", "readme"],
            "deployment": ["deploy", "release"],
            "debugging": ["debug", "fix", "error"],
            "analysis": ["analyze", "analysis"],
            "optimization": ["optimize", "performance"],
            "rag": ["rag", "retrieval", "knowledge base", "evidence"],
            "verification": ["verification", "fact-check", "due diligence"],
        }

        for tag, keywords in tag_keywords.items():
            if any(kw in content_lower for kw in keywords):
                tags.append(tag)

        return tags
