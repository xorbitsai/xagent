"""Model-invocable skill loading tool.

ReAct runs with an active skill manager get a lightweight skill index in the
system context plus this ``load_skill`` tool, so the model pulls in full skill
guidance on demand instead of a framework-driven LLM selection call.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from .enrichment import (
    SELECTED_SKILL_METADATA_KEY,
    SKILL_CONTEXT_METADATA_KEY,
    build_skill_context,
)

logger = logging.getLogger(__name__)

LOAD_SKILL_TOOL_NAME = "load_skill"
SKILL_INDEX_METADATA_KEY = "available_skills_index"
LOADED_SKILLS_METADATA_KEY = "loaded_skills"
INDEX_ENTRY_MAX_CHARS = 200


def _index_text(value: Any, limit: int = INDEX_ENTRY_MAX_CHARS) -> str:
    """Collapse a skill meta field to a bounded single line for the index.

    Skill descriptions come from whole SKILL.md sections and may span
    paragraphs; the index goes into every LLM call's system context, so
    keep each entry to one short line.
    """
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


_LOAD_SKILL_DESCRIPTION = """Load a skill's full instructions into the system context.

The available skills are listed in the system context with one-line summaries. Call this when one of them clearly matches the current task; its detailed guidance becomes available from the next step on. Do not load skills that are unrelated to the task."""


class LoadSkillArgs(BaseModel):
    skill_name: str = Field(
        description="Exact name of the skill to load, as listed in the skill index."
    )


class LoadSkillTool:
    """Execution-scoped ``load_skill`` tool bound to a skill manager and context."""

    name = LOAD_SKILL_TOOL_NAME
    description = _LOAD_SKILL_DESCRIPTION
    args_schema = LoadSkillArgs

    def __init__(
        self,
        *,
        skill_manager: Any,
        context: Any,
        allowed_skills: list[str] | None = None,
    ) -> None:
        self.skill_manager = skill_manager
        self.context = context
        self.allowed_skills = allowed_skills

    async def execute(self, skill_name: str) -> dict[str, Any]:
        skill_name = str(skill_name or "").strip()
        if not skill_name:
            return {"success": False, "error": "skill_name must be a non-empty string."}
        if self.allowed_skills is not None and skill_name not in set(
            self.allowed_skills
        ):
            return {
                "success": False,
                "error": f"Skill '{skill_name}' is not available for this agent.",
            }

        loaded = self.context.metadata.setdefault(LOADED_SKILLS_METADATA_KEY, [])
        if isinstance(loaded, list) and skill_name in loaded:
            return {
                "success": True,
                "skill_name": skill_name,
                "message": (
                    "Skill already loaded; its guidance is in the system context."
                ),
            }

        skill = await self.skill_manager.get_skill(skill_name)
        if not skill or not isinstance(skill, dict):
            available = ", ".join(
                str(entry.get("name"))
                for entry in self.context.metadata.get(SKILL_INDEX_METADATA_KEY) or []
                if isinstance(entry, dict) and entry.get("name")
            )
            return {
                "success": False,
                "error": (
                    f"Skill '{skill_name}' not found. Available skills: "
                    f"{available or '(none)'}."
                ),
            }

        skill_context = build_skill_context(skill)
        existing = str(self.context.metadata.get(SKILL_CONTEXT_METADATA_KEY) or "")
        self.context.metadata[SKILL_CONTEXT_METADATA_KEY] = (
            f"{existing}\n\n{skill_context}" if existing.strip() else skill_context
        )
        self.context.metadata[SELECTED_SKILL_METADATA_KEY] = {
            "name": skill.get("name"),
            "description": skill.get("description"),
            "when_to_use": skill.get("when_to_use"),
        }
        if isinstance(loaded, list):
            loaded.append(skill_name)
        logger.info(
            "Loaded skill %s into execution %s",
            skill_name,
            getattr(self.context, "execution_id", None),
        )
        return {
            "success": True,
            "skill_name": skill_name,
            "message": (
                "Skill guidance loaded into the system context; follow it from "
                "the next step on."
            ),
        }


async def build_load_skill_tool(
    *,
    skill_manager: Any | None,
    context: Any,
    allowed_skills: list[str] | None = None,
) -> LoadSkillTool | None:
    """Create a ``load_skill`` tool and stamp the skill index onto the context.

    Returns None (and clears nothing) when there is no skill manager or no
    skills remain after ``allowed_skills`` filtering.
    """

    if skill_manager is None:
        return None
    try:
        skills = await skill_manager.list_skills()
    except Exception:
        logger.exception("Failed to list skills for the skill index")
        return None
    if not skills:
        return None
    skills = [skill for skill in skills if isinstance(skill, dict)]
    if allowed_skills is not None:
        allowed = set(allowed_skills)
        skills = [skill for skill in skills if skill.get("name") in allowed]
    if not skills:
        return None

    context.metadata[SKILL_INDEX_METADATA_KEY] = [
        {
            "name": skill.get("name"),
            "description": _index_text(skill.get("description")),
            "when_to_use": _index_text(skill.get("when_to_use")),
        }
        for skill in skills
    ]
    return LoadSkillTool(
        skill_manager=skill_manager,
        context=context,
        allowed_skills=allowed_skills,
    )
