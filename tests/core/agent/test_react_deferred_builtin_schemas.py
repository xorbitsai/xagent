"""Built-in tool categories can be deferred like connector tools.

``defer_builtin_tool_categories`` withholds the schemas of built-in tools in
the named categories until ``load_tools`` names them (by tool name or by
category name) or the user's message mentions them. Other tools, and runs that
do not set it, are untouched.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from xagent.core.agent.context.execution import ExecutionContext
from xagent.core.agent.deferred_tools import LOAD_TOOLS_NAME
from xagent.core.agent.pattern.react import ReActPattern
from xagent.core.tools.adapters.vibe.base import ToolCategory


class BuiltinTool:
    def __init__(self, name: str, category: ToolCategory, *, fat: int = 0) -> None:
        self.metadata = SimpleNamespace(
            name=name, description=f"{name} tool", category=category
        )
        self.name = name
        self.fat = fat
        self.calls: list[dict[str, Any]] = []

    @property
    def args(self) -> dict[str, Any]:
        props: dict[str, Any] = {"prompt": {"type": "string"}}
        if self.fat:
            props["blob"] = {"type": "string", "description": "x" * self.fat}
        return props

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"success": True, "output": f"{self.name} ok"}


class ScriptedLLM:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _call(name: str, args: dict[str, Any], call_id: str = "c1") -> dict[str, Any]:
    return {
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }


def _names(call: dict[str, Any]) -> list[str]:
    return [t["function"]["name"] for t in call.get("tools") or []]


def _system(call: dict[str, Any]) -> str:
    return "\n".join(
        str(m.get("content")) for m in call["messages"] if m.get("role") == "system"
    )


def _tools() -> list[Any]:
    return [
        BuiltinTool("read_file", ToolCategory.FILE),
        BuiltinTool("web_search", ToolCategory.WEB_TOOLS),
        BuiltinTool("generate_image", ToolCategory.IMAGE, fat=2000),
        BuiltinTool("edit_image", ToolCategory.IMAGE),
        BuiltinTool("generate_video", ToolCategory.VIDEO),
        BuiltinTool("synthesize_speech", ToolCategory.AUDIO),
        BuiltinTool("browser_navigate", ToolCategory.BROWSER),
        BuiltinTool("create_ppt", ToolCategory.PPT),
    ]


DEFERRED = {
    "generate_image",
    "edit_image",
    "generate_video",
    "synthesize_speech",
    "browser_navigate",
    "create_ppt",
}
CATEGORIES = ("image", "video", "audio", "browser", "ppt")


def _ctx(text: str = "What is two plus two?") -> ExecutionContext:
    context = ExecutionContext()
    context.add_user_message(text)
    return context


def _pattern(**kwargs: Any) -> ReActPattern:
    kwargs.setdefault("defer_builtin_tool_categories", CATEGORIES)
    return ReActPattern(max_iterations=6, **kwargs)


@pytest.mark.asyncio
async def test_default_sends_every_builtin_schema() -> None:
    llm = ScriptedLLM([{"content": "hi"}])
    await ReActPattern(max_iterations=3).run(context=_ctx(), tools=_tools(), llm=llm)

    names = _names(llm.calls[0])
    assert DEFERRED <= set(names)
    assert LOAD_TOOLS_NAME not in names


@pytest.mark.asyncio
async def test_deferred_categories_leave_the_schemas_but_stay_in_the_catalog() -> None:
    llm = ScriptedLLM([{"content": "hi"}])
    await _pattern().run(context=_ctx(), tools=_tools(), llm=llm)

    names = _names(llm.calls[0])
    assert not DEFERRED & set(names)
    assert {"read_file", "web_search", LOAD_TOOLS_NAME} <= set(names)
    system = _system(llm.calls[0])
    for name in DEFERRED:
        assert name in system
    assert "built-in image" in system


@pytest.mark.asyncio
async def test_only_the_listed_categories_are_deferred() -> None:
    llm = ScriptedLLM([{"content": "hi"}])
    await _pattern(defer_builtin_tool_categories=["Image"]).run(
        context=_ctx(), tools=_tools(), llm=llm
    )

    names = set(_names(llm.calls[0]))
    assert not {"generate_image", "edit_image"} & names
    assert {"generate_video", "create_ppt", "browser_navigate"} <= names


@pytest.mark.asyncio
async def test_load_by_tool_name_then_call() -> None:
    tools = _tools()
    llm = ScriptedLLM(
        [
            _call(LOAD_TOOLS_NAME, {"names": ["Generate_Image"]}),
            _call("generate_image", {"prompt": "a cat"}, "c2"),
            {"content": "done"},
        ]
    )
    await _pattern().run(context=_ctx("Please do it"), tools=tools, llm=llm)

    first, second, third = (_names(c) for c in llm.calls)
    assert "generate_image" not in first
    assert second[-1] == "generate_image" and "edit_image" not in second
    assert third == second
    image = next(t for t in tools if t.name == "generate_image")
    assert image.calls == [{"prompt": "a cat"}]
    assert _system(llm.calls[0]) == _system(llm.calls[1])


@pytest.mark.asyncio
async def test_load_by_category_name_is_case_insensitive() -> None:
    llm = ScriptedLLM([_call(LOAD_TOOLS_NAME, {"names": ["IMAGE"]}), {"content": "ok"}])
    context = _ctx("Please do it")
    await _pattern().run(context=context, tools=_tools(), llm=llm)

    second = _names(llm.calls[1])
    assert {"generate_image", "edit_image"} <= set(second)
    assert not {"generate_video", "create_ppt"} & set(second)
    tool_text = context.get_messages_by_role("tool")[0].content
    assert "Loaded: generate_image, edit_image" in str(tool_text)


@pytest.mark.asyncio
async def test_a_tool_name_wins_over_a_category_of_the_same_name() -> None:
    tools = [
        BuiltinTool("image", ToolCategory.VIDEO),
        BuiltinTool("generate_image", ToolCategory.IMAGE),
    ]
    llm = ScriptedLLM([_call(LOAD_TOOLS_NAME, {"names": ["image"]}), {"content": "k"}])
    await _pattern().run(context=_ctx("go"), tools=tools, llm=llm)

    second = _names(llm.calls[1])
    assert "image" in second and "generate_image" not in second


@pytest.mark.asyncio
async def test_loaded_set_is_sticky_across_iterations() -> None:
    llm = ScriptedLLM(
        [
            _call(LOAD_TOOLS_NAME, {"names": ["image"]}, "c1"),
            _call("generate_image", {"prompt": "a"}, "c2"),
            _call("edit_image", {"prompt": "b"}, "c3"),
            {"content": "done"},
        ]
    )
    await _pattern().run(context=_ctx("go"), tools=_tools(), llm=llm)

    for call in llm.calls[1:]:
        assert {"generate_image", "edit_image"} <= set(_names(call))
    assert _names(llm.calls[0]).count(LOAD_TOOLS_NAME) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "loaded"),
    [
        ("Please make an image of a fox", {"generate_image", "edit_image"}),
        ("Make me a PowerPoint about Q3", {"create_ppt"}),
        ("Turn this into a Video", {"generate_video"}),
        ("call synthesize_speech on this", {"synthesize_speech"}),
        ("What is two plus two?", set()),
    ],
)
async def test_mentions_preload_only_what_they_name(
    text: str, loaded: set[str]
) -> None:
    llm = ScriptedLLM([{"content": "hi"}])
    await _pattern().run(context=_ctx(text), tools=_tools(), llm=llm)

    assert DEFERRED & set(_names(llm.calls[0])) == loaded


@pytest.mark.asyncio
async def test_mention_preload_respects_the_token_budget() -> None:
    llm = ScriptedLLM([{"content": "hi"}])
    await _pattern(deferred_preload_token_budget=10).run(
        context=_ctx("make an image"), tools=_tools(), llm=llm
    )

    assert not DEFERRED & set(_names(llm.calls[0]))


@pytest.mark.asyncio
async def test_excluded_categories_are_not_in_the_catalog() -> None:
    tools = [t for t in _tools() if t.metadata.category != ToolCategory.VIDEO]
    llm = ScriptedLLM([{"content": "hi"}])
    await _pattern().run(context=_ctx(), tools=tools, llm=llm)

    system = _system(llm.calls[0])
    assert "generate_video" not in system and "built-in video" not in system
    assert "generate_image" in system


@pytest.mark.asyncio
async def test_connector_tools_are_not_treated_as_builtin() -> None:
    connector = BuiltinTool("mcp_image_search", ToolCategory.IMAGE)
    connector.metadata.source_server = "images"
    llm = ScriptedLLM([{"content": "hi"}])
    await _pattern().run(context=_ctx(), tools=[connector], llm=llm)

    assert "mcp_image_search" in _names(llm.calls[0])
    assert LOAD_TOOLS_NAME not in _names(llm.calls[0])


@pytest.mark.asyncio
async def test_connectors_and_builtins_defer_together() -> None:
    connector = BuiltinTool("hubspot_search", ToolCategory.MCP)
    connector.metadata.source_server = "hubspot"
    llm = ScriptedLLM([{"content": "hi"}])
    await _pattern(defer_connector_schemas=True).run(
        context=_ctx(), tools=[*_tools(), connector], llm=llm
    )

    names = set(_names(llm.calls[0]))
    assert not ({"hubspot_search"} | DEFERRED) & names
    assert {"read_file", LOAD_TOOLS_NAME} <= names
    system = _system(llm.calls[0])
    assert "hubspot_search" in system and "built-in image" in system


def test_setting_reaches_the_react_and_auto_patterns() -> None:
    from xagent.core.agent.execution_adapter import (
        AgentExecutionAdapter,
        AgentExecutionConfig,
    )

    for mode in ("react", "auto"):
        adapter = AgentExecutionAdapter(
            AgentExecutionConfig(
                name="t",
                llm=object(),
                pattern=mode,
                defer_builtin_tool_categories=("image", "ppt"),
            )
        )
        pattern, _ = adapter._build_pattern()
        react = pattern.react_pattern if mode == "auto" else pattern
        assert react.defer_builtin_tool_categories == {"image", "ppt"}
    off = AgentExecutionAdapter(
        AgentExecutionConfig(name="t", llm=object(), pattern="react")
    )
    assert off._build_pattern()[0].defer_builtin_tool_categories == frozenset()
