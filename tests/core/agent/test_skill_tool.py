"""Tests for the model-invocable ``load_skill`` tool and skill index."""

from typing import Any

import pytest

from xagent.core.agent.context import ExecutionContext
from xagent.core.agent.context.enrichment import (
    SELECTED_SKILL_METADATA_KEY,
    SKILL_CONTEXT_METADATA_KEY,
)
from xagent.core.agent.context.skill_tool import (
    LOADED_SKILLS_METADATA_KEY,
    SKILL_INDEX_METADATA_KEY,
    LoadSkillTool,
    build_load_skill_tool,
)

WRITER_SKILL = {
    "name": "writer",
    "description": "Writes concise copy",
    "when_to_use": "Writing tasks",
    "content": "Use short sentences.",
}
CODER_SKILL = {
    "name": "coder",
    "description": "Writes code",
    "when_to_use": "Coding tasks",
    "content": "Prefer small functions.",
}


class FakeSkillManager:
    def __init__(self, skills: list[dict[str, Any]]) -> None:
        self.skills = {skill["name"]: skill for skill in skills}

    async def list_skills(self) -> list[dict[str, Any]]:
        return [
            {
                "name": skill["name"],
                "description": skill.get("description", ""),
                "when_to_use": skill.get("when_to_use", ""),
            }
            for skill in self.skills.values()
        ]

    async def get_skill(self, name: str) -> dict[str, Any] | None:
        return self.skills.get(name)


@pytest.mark.asyncio
async def test_build_load_skill_tool_stamps_index_on_context() -> None:
    context = ExecutionContext()
    manager = FakeSkillManager([WRITER_SKILL, CODER_SKILL])

    tool = await build_load_skill_tool(skill_manager=manager, context=context)

    assert isinstance(tool, LoadSkillTool)
    index = context.metadata[SKILL_INDEX_METADATA_KEY]
    assert [entry["name"] for entry in index] == ["writer", "coder"]
    assert index[0]["description"] == "Writes concise copy"


@pytest.mark.asyncio
async def test_build_load_skill_tool_filters_allowed_skills() -> None:
    context = ExecutionContext()
    manager = FakeSkillManager([WRITER_SKILL, CODER_SKILL])

    tool = await build_load_skill_tool(
        skill_manager=manager, context=context, allowed_skills=["coder"]
    )

    assert tool is not None
    index = context.metadata[SKILL_INDEX_METADATA_KEY]
    assert [entry["name"] for entry in index] == ["coder"]


@pytest.mark.asyncio
async def test_build_load_skill_tool_returns_none_without_skills() -> None:
    context = ExecutionContext()
    assert await build_load_skill_tool(skill_manager=None, context=context) is None
    assert (
        await build_load_skill_tool(skill_manager=FakeSkillManager([]), context=context)
        is None
    )
    assert (
        await build_load_skill_tool(
            skill_manager=FakeSkillManager([WRITER_SKILL]),
            context=context,
            allowed_skills=[],
        )
        is None
    )
    assert SKILL_INDEX_METADATA_KEY not in context.metadata


@pytest.mark.asyncio
async def test_load_skill_puts_guidance_into_context_metadata() -> None:
    context = ExecutionContext()
    manager = FakeSkillManager([WRITER_SKILL])
    tool = await build_load_skill_tool(skill_manager=manager, context=context)
    assert tool is not None

    result = await tool.execute(skill_name="writer")

    assert result["success"] is True
    assert "Use short sentences." in context.metadata[SKILL_CONTEXT_METADATA_KEY]
    assert context.metadata[SELECTED_SKILL_METADATA_KEY]["name"] == "writer"
    assert context.metadata[LOADED_SKILLS_METADATA_KEY] == ["writer"]


@pytest.mark.asyncio
async def test_load_skill_appends_second_skill_and_skips_reload() -> None:
    context = ExecutionContext()
    manager = FakeSkillManager([WRITER_SKILL, CODER_SKILL])
    tool = await build_load_skill_tool(skill_manager=manager, context=context)
    assert tool is not None

    await tool.execute(skill_name="writer")
    again = await tool.execute(skill_name="writer")
    second = await tool.execute(skill_name="coder")

    assert again["success"] is True
    assert "already loaded" in again["message"]
    assert second["success"] is True
    skill_context = context.metadata[SKILL_CONTEXT_METADATA_KEY]
    assert "Use short sentences." in skill_context
    assert "Prefer small functions." in skill_context
    assert context.metadata[LOADED_SKILLS_METADATA_KEY] == ["writer", "coder"]


@pytest.mark.asyncio
async def test_load_skill_rejects_unknown_and_disallowed_skills() -> None:
    context = ExecutionContext()
    manager = FakeSkillManager([WRITER_SKILL, CODER_SKILL])
    tool = await build_load_skill_tool(skill_manager=manager, context=context)
    assert tool is not None

    missing = await tool.execute(skill_name="unknown")
    empty = await tool.execute(skill_name="  ")

    assert missing["success"] is False
    assert "writer" in missing["error"]
    assert empty["success"] is False

    restricted_context = ExecutionContext()
    restricted = await build_load_skill_tool(
        skill_manager=manager,
        context=restricted_context,
        allowed_skills=["writer"],
    )
    assert restricted is not None
    disallowed = await restricted.execute(skill_name="coder")

    assert disallowed["success"] is False
    assert "not available" in disallowed["error"]
    assert SKILL_CONTEXT_METADATA_KEY not in context.metadata
    assert SKILL_CONTEXT_METADATA_KEY not in restricted_context.metadata


def test_skill_index_renders_into_system_context() -> None:
    context = ExecutionContext(system_prompt="Base prompt.")
    context.metadata[SKILL_INDEX_METADATA_KEY] = [
        {
            "name": "writer",
            "description": "Writes concise copy",
            "when_to_use": "Writing tasks",
        },
        {"name": "coder", "description": "Writes code", "when_to_use": ""},
    ]
    context.add_user_message("hello")

    system_content = context.get_messages_for_llm()[0]["content"]

    assert "Available skills:" in system_content
    assert "load_skill" in system_content
    assert "- writer: Writes concise copy When to use: Writing tasks" in system_content
    assert "- coder: Writes code" in system_content


def test_skill_index_hides_already_loaded_skills() -> None:
    context = ExecutionContext(system_prompt="Base prompt.")
    context.metadata[SKILL_INDEX_METADATA_KEY] = [
        {"name": "writer", "description": "Writes concise copy", "when_to_use": ""},
        {"name": "coder", "description": "Writes code", "when_to_use": ""},
    ]
    context.metadata[LOADED_SKILLS_METADATA_KEY] = ["writer", "coder"]
    context.add_user_message("hello")

    system_content = context.get_messages_for_llm()[0]["content"]

    assert "Available skills:" not in system_content


@pytest.mark.asyncio
async def test_build_load_skill_tool_bounds_index_entries() -> None:
    context = ExecutionContext()
    long_description = "A very detailed description. " * 30
    manager = FakeSkillManager(
        [
            {
                "name": "verbose",
                "description": long_description,
                "when_to_use": "Line one.\n\nLine two with   spaces.",
                "content": "...",
            }
        ]
    )

    tool = await build_load_skill_tool(skill_manager=manager, context=context)

    assert tool is not None
    entry = context.metadata[SKILL_INDEX_METADATA_KEY][0]
    assert len(entry["description"]) <= 200
    assert entry["description"].endswith("…")
    assert "\n" not in entry["when_to_use"]
    assert entry["when_to_use"] == "Line one. Line two with spaces."


def test_skill_index_renders_name_only_for_empty_meta() -> None:
    context = ExecutionContext(system_prompt="Base prompt.")
    context.metadata[SKILL_INDEX_METADATA_KEY] = [
        {"name": "bare-skill", "description": "", "when_to_use": ""},
    ]
    context.add_user_message("hello")

    system_content = context.get_messages_for_llm()[0]["content"]

    assert "- bare-skill\n" in system_content or system_content.rstrip().endswith(
        "- bare-skill"
    )
    assert "- bare-skill:" not in system_content


@pytest.mark.asyncio
async def test_build_load_skill_tool_handles_none_skill_list() -> None:
    class NoneListManager:
        async def list_skills(self) -> None:
            return None

    context = ExecutionContext()
    tool = await build_load_skill_tool(
        skill_manager=NoneListManager(),
        context=context,
        allowed_skills=["writer"],
    )

    assert tool is None
    assert SKILL_INDEX_METADATA_KEY not in context.metadata


@pytest.mark.asyncio
async def test_load_skill_rejects_non_dict_skill_payload() -> None:
    class MalformedManager(FakeSkillManager):
        async def get_skill(self, name: str):  # type: ignore[override]
            return "not-a-dict"

    context = ExecutionContext()
    tool = await build_load_skill_tool(
        skill_manager=MalformedManager([WRITER_SKILL]), context=context
    )
    assert tool is not None

    result = await tool.execute(skill_name="writer")

    assert result["success"] is False
    assert SKILL_CONTEXT_METADATA_KEY not in context.metadata
