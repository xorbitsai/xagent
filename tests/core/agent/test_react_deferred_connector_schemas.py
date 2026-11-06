"""Connector tool schemas are withheld until the model loads them.

The opt-in (``defer_connector_schemas``) keeps every connector tool in the
run's tool list but sends its JSON schema only after ``load_tools`` names it,
or when the user named its connector. These tests pin the contract the hosts
rely on: off by default, an unloaded call never reaches the tool, the loaded
set survives a checkpoint, and nothing is sent under ``tool_choice="none"``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from xagent.core.agent.context.components import LoadedToolsComponent
from xagent.core.agent.context.execution import ExecutionContext
from xagent.core.agent.deferred_tools import (
    LOAD_TOOLS_NAME,
    LOADED_TOOLS_COMPONENT,
    MAX_TOOLS_PER_LOAD,
)
from xagent.core.agent.pattern.react import ReActPattern


class ConnectorTool:
    def __init__(self, name: str, source: str, *, fat: bool = False) -> None:
        self.metadata = SimpleNamespace(
            name=name, description=f"{name} tool", source_server=source
        )
        self.name = name
        self.fat = fat
        self.calls: list[dict[str, Any]] = []

    @property
    def args(self) -> dict[str, Any]:
        props: dict[str, Any] = {"query": {"type": "string"}}
        if self.fat:
            props["blob"] = {"type": "string", "description": "x" * 4000}
        return props

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"success": True, "output": f"{self.name} ok"}


class PlainTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"{name} tool"
        self.args = {"q": {"type": "string"}}

    async def execute(self, **_kwargs: Any) -> dict[str, Any]:
        return {"success": True, "output": "plain ok"}


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


def _system_text(call: dict[str, Any]) -> str:
    return "\n".join(
        str(m.get("content")) for m in call["messages"] if m.get("role") == "system"
    )


def _tools() -> list[Any]:
    return [
        PlainTool("get_current_time"),
        ConnectorTool("hubspot_search_contacts", "hubspot"),
        ConnectorTool("hubspot_create_note", "hubspot"),
        ConnectorTool("notion_search", "notion"),
    ]


def _ctx(text: str = "What is on my calendar?") -> ExecutionContext:
    context = ExecutionContext()
    context.add_user_message(text)
    return context


def _pattern(**kwargs: Any) -> ReActPattern:
    return ReActPattern(max_iterations=6, defer_connector_schemas=True, **kwargs)


@pytest.mark.asyncio
async def test_off_by_default_sends_every_schema() -> None:
    llm = ScriptedLLM([{"content": "hi"}])
    await ReActPattern(max_iterations=3).run(context=_ctx(), tools=_tools(), llm=llm)

    names = _names(llm.calls[0])
    assert {"hubspot_search_contacts", "hubspot_create_note", "notion_search"} <= set(
        names
    )
    assert LOAD_TOOLS_NAME not in names
    assert "are deferred" not in _system_text(llm.calls[0])


@pytest.mark.asyncio
async def test_on_withholds_connector_schemas_and_lists_an_index() -> None:
    llm = ScriptedLLM([{"content": "hi"}])
    await _pattern().run(context=_ctx(), tools=_tools(), llm=llm)

    names = _names(llm.calls[0])
    assert "get_current_time" in names
    assert LOAD_TOOLS_NAME in names
    assert not {
        "hubspot_search_contacts",
        "hubspot_create_note",
        "notion_search",
    } & set(names)
    system = _system_text(llm.calls[0])
    assert "are deferred" in system and system.count("notion_search") == 1
    for name in ("hubspot_search_contacts", "hubspot_create_note", "notion_search"):
        assert name in system


@pytest.mark.asyncio
async def test_load_then_call_runs_the_tool_with_a_stable_prefix() -> None:
    tools = _tools()
    llm = ScriptedLLM(
        [
            _call(LOAD_TOOLS_NAME, {"names": ["notion_search"]}, "c1"),
            _call("notion_search", {"query": "plan"}, "c2"),
            {"content": "done"},
        ]
    )
    context = _ctx()
    await _pattern().run(context=context, tools=tools, llm=llm)

    first, second, third = (_names(c) for c in llm.calls)
    assert "notion_search" not in first
    assert second[: len(first)] == first
    assert second[-1] == "notion_search"
    assert third == second
    notion = next(t for t in tools if t.name == "notion_search")
    assert notion.calls == [{"query": "plan"}]
    # The full schema, not just the name, is what the model receives.
    schema = next(
        t for t in llm.calls[1]["tools"] if t["function"]["name"] == "notion_search"
    )
    assert "query" in schema["function"]["parameters"]["properties"]
    # The system message is byte-identical across the load so a provider's
    # cached prefix is not invalidated by it.
    assert _system_text(llm.calls[0]) == _system_text(llm.calls[1])


@pytest.mark.asyncio
async def test_unloaded_call_never_reaches_the_tool() -> None:
    tools = _tools()
    llm = ScriptedLLM(
        [_call("notion_search", {"query": "plan"}), {"content": "recovered"}]
    )
    context = _ctx()
    result = await _pattern().run(context=context, tools=tools, llm=llm)

    notion = next(t for t in tools if t.name == "notion_search")
    assert notion.calls == []
    error = context.get_messages_by_role("tool")[0].metadata["raw_result"]["error"]
    assert LOAD_TOOLS_NAME in error and "notion_search" in error
    assert result["success"] is True


@pytest.mark.asyncio
async def test_load_tools_reports_unknown_names_and_caps_the_batch() -> None:
    many = [ConnectorTool(f"hubspot_tool_{i}", "hubspot") for i in range(15)]
    llm = ScriptedLLM(
        [
            _call(
                LOAD_TOOLS_NAME,
                {"names": [t.name for t in many] + ["hubspot_tool_99x"]},
            ),
            {"content": "ok"},
        ]
    )
    context = _ctx()
    await _pattern().run(context=context, tools=many, llm=llm)

    loaded = context.get_component(LOADED_TOOLS_COMPONENT).names
    assert loaded == [f"hubspot_tool_{i}" for i in range(MAX_TOOLS_PER_LOAD)]
    text = context.get_messages_by_role("tool")[0].content
    assert "Not loaded" in text and "Unknown tool 'hubspot_tool_99x'" in text


@pytest.mark.asyncio
async def test_naming_a_connector_preloads_only_that_connector() -> None:
    llm = ScriptedLLM([{"content": "hi"}])
    await _pattern().run(
        context=_ctx("Find the Acme contact in HubSpot please"),
        tools=_tools(),
        llm=llm,
    )

    names = _names(llm.calls[0])
    assert {"hubspot_search_contacts", "hubspot_create_note"} <= set(names)
    assert "notion_search" not in names


@pytest.mark.asyncio
async def test_preload_respects_the_token_budget() -> None:
    tools = [
        ConnectorTool("hubspot_search_contacts", "hubspot", fat=True),
        ConnectorTool("notion_search", "notion"),
    ]
    llm = ScriptedLLM([{"content": "hi"}])
    await _pattern(deferred_preload_token_budget=200).run(
        context=_ctx("Use HubSpot and Notion"), tools=tools, llm=llm
    )

    names = _names(llm.calls[0])
    assert "hubspot_search_contacts" not in names  # over budget
    assert "notion_search" in names


@pytest.mark.asyncio
async def test_loaded_set_survives_a_checkpoint_round_trip() -> None:
    tools = _tools()
    first_llm = ScriptedLLM(
        [_call(LOAD_TOOLS_NAME, {"names": ["notion_search"]}), {"content": "ok"}]
    )
    context = _ctx()
    await _pattern().run(context=context, tools=tools, llm=first_llm)

    restored = ExecutionContext.from_dict(context.to_dict())
    restored.add_user_message("Now search it")
    second_llm = ScriptedLLM(
        [_call("notion_search", {"query": "plan"}), {"content": "done"}]
    )
    await _pattern().run(context=restored, tools=tools, llm=second_llm)

    assert "notion_search" in _names(second_llm.calls[0])
    notion = next(t for t in tools if t.name == "notion_search")
    assert notion.calls == [{"query": "plan"}]


@pytest.mark.asyncio
async def test_tool_choice_none_sends_no_tools_even_when_loaded() -> None:
    context = _ctx()
    context.set_component(
        LOADED_TOOLS_COMPONENT,
        LoadedToolsComponent(names=["notion_search"]),
    )
    llm = ScriptedLLM([{"content": "text only"}])
    await ReActPattern(
        max_iterations=2, tool_choice="none", defer_connector_schemas=True
    ).run(context=context, tools=_tools(), llm=llm)

    assert _names(llm.calls[0]) == []


def test_forced_answer_surface_never_carries_loaded_connector_schemas() -> None:
    pattern = _pattern()
    loaded = {
        "type": "function",
        "function": {"name": "notion_search", "parameters": {}},
    }
    schemas = pattern._forced_answer_tool_schemas([loaded])

    assert [s["function"]["name"] for s in schemas] == ["final_answer"]


@pytest.mark.asyncio
async def test_index_lists_every_connector_tool_including_failing_ones() -> None:
    class Failing(ConnectorTool):
        async def execute(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("server unavailable")

    tools = [Failing("xero_list_invoices", "xero"), *_tools()]
    llm = ScriptedLLM([{"content": "hi"}])
    await _pattern().run(context=_ctx(), tools=tools, llm=llm)

    assert "xero_list_invoices" in _system_text(llm.calls[0])
    assert "xero_list_invoices" not in _names(llm.calls[0])


class ApprovalConnectorTool:
    """A connector write that pauses for approval, then runs once approved."""

    def __init__(self) -> None:
        self.metadata = SimpleNamespace(
            name="hubspot_create_note",
            description="Create a note.",
            source_server="hubspot",
        )
        self.approved: str | None = None
        self.writes: list[dict[str, Any]] = []

    def args_type(self) -> Any:
        from pydantic import BaseModel

        class Args(BaseModel):
            body: str

        return Args

    def resume_user_interaction(self, *, interaction_id: str, response: str) -> None:
        self.approved = response

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        if self.approved is None:
            return {
                "success": False,
                "status": "waiting_for_user",
                "interaction_id": "approval-1",
                "message": "Approve this note?",
                "message_type": "confirmation",
                "interactions": [
                    {
                        "type": "select_one",
                        "field": "decision",
                        "label": "Decision",
                        "options": [{"label": "Approve", "value": "approve"}],
                    }
                ],
            }
        self.writes.append(args)
        return {"success": True, "output": "note created"}


@pytest.mark.asyncio
async def test_approval_pause_and_resume_keep_the_loaded_tool_callable() -> None:
    tool = ApprovalConnectorTool()
    first_llm = ScriptedLLM(
        [
            _call(LOAD_TOOLS_NAME, {"names": ["hubspot_create_note"]}, "c1"),
            _call("hubspot_create_note", {"body": "hi"}, "c2"),
        ]
    )
    context = _ctx("Please add a note")
    pattern = _pattern()
    waiting = await pattern.run(context=context, tools=[tool], llm=first_llm)
    assert waiting["status"] == "waiting_for_user"
    assert tool.writes == []

    # The approval click arrives in a later run, from a checkpointed context.
    restored = ExecutionContext.from_dict(context.to_dict())
    restored.add_user_message("approve")
    resumed_pattern = _pattern()
    resumed_pattern.load_state(pattern.get_state())
    resumed_llm = ScriptedLLM(
        [
            _call("hubspot_create_note", {"body": "hi"}, "c3"),
            {"content": "done"},
        ]
    )
    resumed = await resumed_pattern.run(context=restored, tools=[tool], llm=resumed_llm)

    assert resumed["success"] is True
    assert tool.writes == [{"body": "hi"}]
    assert "hubspot_create_note" in _names(resumed_llm.calls[0])
