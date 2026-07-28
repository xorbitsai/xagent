from types import SimpleNamespace
from typing import Any

import pytest

from xagent.core.task_runtime import TaskRuntimeContribution
from xagent.core.tools.adapters.vibe.base import ToolCategory
from xagent.core.tools.adapters.vibe.config import ToolConfig
from xagent.core.tools.adapters.vibe.factory import ToolFactory, ToolRegistry


def _tool(name: str) -> Any:
    return SimpleNamespace(
        name=name,
        metadata=SimpleNamespace(category=ToolCategory.OTHER),
    )


@pytest.mark.asyncio
async def test_runtime_tools_enter_normal_selection_pipeline(monkeypatch) -> None:
    base_tool = _tool("base_tool")
    runtime_tool = _tool("runtime_tool")

    async def create_registered_tools(config: Any) -> list[Any]:
        return [base_tool]

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        create_registered_tools,
    )
    config = ToolConfig({"allowed_tools": ["runtime_tool"]})

    tools = await ToolFactory.create_all_tools(
        config,
        additional_tools=(runtime_tool,),
    )

    assert tools == [runtime_tool]


@pytest.mark.asyncio
async def test_runtime_tools_are_restored_from_config_on_rebuild(monkeypatch) -> None:
    base_tool = _tool("base_tool")
    runtime_tool = _tool("runtime_tool")

    async def create_registered_tools(config: Any) -> list[Any]:
        return [base_tool]

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        create_registered_tools,
    )
    config = ToolConfig({})
    config.get_task_runtime_contribution = lambda: TaskRuntimeContribution(
        tools=(runtime_tool,)
    )

    first = await ToolFactory.create_all_tools(config)
    rebuilt = await ToolFactory.create_all_tools(config)

    assert [tool.name for tool in first] == ["base_tool", "runtime_tool"]
    assert [tool.name for tool in rebuilt] == ["base_tool", "runtime_tool"]


@pytest.mark.asyncio
async def test_runtime_tools_cannot_shadow_existing_tool(monkeypatch) -> None:
    async def create_registered_tools(config: Any) -> list[Any]:
        return [_tool("computer")]

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        create_registered_tools,
    )

    with pytest.raises(ValueError, match="duplicate tool 'computer'"):
        await ToolFactory.create_all_tools(
            ToolConfig({}),
            additional_tools=(_tool("computer"),),
        )


@pytest.mark.asyncio
async def test_runtime_tools_require_a_non_empty_string_name(monkeypatch) -> None:
    async def create_registered_tools(config: Any) -> list[Any]:
        return []

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        create_registered_tools,
    )

    with pytest.raises(TypeError, match="non-empty string 'name'"):
        await ToolFactory.create_all_tools(
            ToolConfig({}),
            additional_tools=(SimpleNamespace(),),
        )
