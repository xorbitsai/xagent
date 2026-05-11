"""
Skill Parser - Parse SKILL.md and related files
"""

import re
from pathlib import Path
from typing import Dict, List


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

        content = skill_md.read_text()

        # Try to read template.md
        template_md = skill_dir / "template.md"
        template_content = template_md.read_text() if template_md.exists() else ""

        frontmatter = SkillParser._extract_frontmatter(content)
        section_description = SkillParser._extract_section(content, "Description")
        section_when_to_use = SkillParser._extract_section(content, "When to Use")
        section_tags = SkillParser._extract_tags(content)
        frontmatter_tags = frontmatter.get("tags")
        tags = (
            frontmatter_tags
            if isinstance(frontmatter_tags, list) and frontmatter_tags
            else section_tags
        )

        return {
            "name": skill_dir.name,
            "path": str(skill_dir),
            "content": content,  # Complete SKILL.md content
            "template": template_content,  # template.md content (if exists)
            "description": section_description
            or SkillParser._frontmatter_string(frontmatter, "description"),
            "when_to_use": section_when_to_use
            or SkillParser._frontmatter_string(frontmatter, "when_to_use"),
            "execution_flow": SkillParser._extract_section(content, "Execution Flow"),
            "tags": tags,
            "files": SkillParser._list_files(skill_dir),
        }

    @staticmethod
    def _extract_frontmatter(content: str) -> Dict:
        """Extract simple YAML-style frontmatter from a skill file."""
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

        metadata: Dict[str, object] = {}
        current_list_key: str | None = None
        for raw_line in lines[1:end_index]:
            line = raw_line.rstrip()
            if not line.strip():
                continue

            stripped_line = SkillParser._strip_yaml_comment(line).strip()
            if not stripped_line:
                continue
            if stripped_line.startswith("- ") and current_list_key:
                values = metadata.get(current_list_key)
                if not isinstance(values, list):
                    values = []
                    metadata[current_list_key] = values
                value = stripped_line[2:].strip().strip("\"'")
                values.append(value)
                continue

            current_list_key = None
            if ":" not in stripped_line:
                continue

            key, value = stripped_line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not value:
                metadata[key] = ""
                current_list_key = key
                continue

            metadata[key] = value.strip("\"'")

        return metadata

    @staticmethod
    def _frontmatter_string(frontmatter: Dict, key: str) -> str:
        """Return a scalar frontmatter field or an empty string."""
        value = frontmatter.get(key)
        return value if isinstance(value, str) else ""

    @staticmethod
    def _strip_yaml_comment(line: str) -> str:
        """Strip simple YAML comments while preserving quoted # characters."""
        quote: str | None = None
        escaped = False

        for index, char in enumerate(line):
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char in {"'", '"'}:
                if quote == char:
                    quote = None
                elif quote is None:
                    quote = char
                continue
            if char == "#" and quote is None:
                return line[:index].rstrip()

        return line

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
