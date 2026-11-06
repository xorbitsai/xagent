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


def _batch(*calls: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "content": "",
        "tool_calls": [
            {
                "id": f"b{i}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            for i, (name, args) in enumerate(calls)
        ],
    }


@pytest.mark.asyncio
async def test_a_call_in_the_same_response_as_its_load_is_refused() -> None:
    tools = _tools()
    llm = ScriptedLLM(
        [
            _batch(
                (LOAD_TOOLS_NAME, {"names": ["notion_search"]}),
                ("notion_search", {"query": "guessed blind"}),
            ),
            _call("notion_search", {"query": "with the schema in view"}, "c2"),
            {"content": "done"},
        ]
    )
    await _pattern().run(context=_ctx(), tools=tools, llm=llm)

    notion = next(t for t in tools if t.name == "notion_search")
    assert notion.calls == [{"query": "with the schema in view"}]
    # The load took effect for the next model call, which therefore saw it.
    assert "notion_search" not in _names(llm.calls[0])
    assert "notion_search" in _names(llm.calls[1])


class ValidatingLLM:
    """Like DeepSeek/OpenRouter: a call outside the sent schemas is refused."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        from xagent.core.model.chat.exceptions import LLMToolProtocolError

        self.calls.append(kwargs)
        response = self.responses.pop(0)
        sent = set(_names(kwargs))
        for call in response.get("tool_calls") or []:
            name = call["function"]["name"]
            if name not in sent:
                raise LLMToolProtocolError(
                    provider="validating",
                    code="unavailable_tool_call",
                    message=f"unavailable tool call {name!r}",
                    details={"tool_name": name},
                )
        return response


@pytest.mark.asyncio
async def test_a_provider_refusing_an_unloaded_call_gets_the_tool_loaded() -> None:
    tools = _tools()
    llm = ValidatingLLM(
        [
            _call("notion_search", {"query": "direct"}, "c1"),
            _call("notion_search", {"query": "retried"}, "c2"),
            {"content": "done"},
        ]
    )
    result = await _pattern().run(context=_ctx(), tools=tools, llm=llm)

    assert result["success"] is True
    notion = next(t for t in tools if t.name == "notion_search")
    assert notion.calls == [{"query": "retried"}]
    assert "notion_search" not in _names(llm.calls[0])
    assert "notion_search" in _names(llm.calls[1])


@pytest.mark.asyncio
async def test_a_follow_up_message_preloads_the_connector_it_names() -> None:
    tools = _tools()
    context = _ctx("Hello")
    first = ScriptedLLM([{"content": "hi"}])
    await _pattern().run(context=context, tools=tools, llm=first)
    assert "hubspot_search_contacts" not in _names(first.calls[0])

    context.add_user_message("Now look it up in HubSpot")
    second = ScriptedLLM([{"content": "ok"}])
    await _pattern().run(context=context, tools=tools, llm=second)
    assert "hubspot_search_contacts" in _names(second.calls[0])

    # Resuming without a new user message does not scan again.
    third = ScriptedLLM([{"content": "ok"}])
    resumed = ExecutionContext.from_dict(context.to_dict())
    await _pattern().run(context=resumed, tools=tools, llm=third)
    assert _names(third.calls[0]).count("hubspot_search_contacts") == 1


@pytest.mark.asyncio
async def test_a_quoted_exact_tool_name_is_preloaded() -> None:
    llm = ScriptedLLM([{"content": "hi"}])
    await _pattern().run(
        context=_ctx("please call notion_search for the roadmap"),
        tools=_tools(),
        llm=llm,
    )

    names = _names(llm.calls[0])
    assert "notion_search" in names
    assert "hubspot_search_contacts" not in names


@pytest.mark.asyncio
async def test_preload_matches_what_the_user_typed_not_injected_context() -> None:
    context = ExecutionContext()
    context.add_user_message(
        "[system context mentions HubSpot and Notion] hello",
        metadata={"display_message": "hello"},
    )
    llm = ScriptedLLM([{"content": "hi"}])
    await _pattern().run(context=context, tools=_tools(), llm=llm)

    assert "hubspot_search_contacts" not in _names(llm.calls[0])


@pytest.mark.asyncio
async def test_preload_keeps_tool_order_when_the_budget_fits_only_one() -> None:
    servers = ["zeta", "yankee", "xray", "whiskey", "victor", "uniform", "tango"]
    tools = [ConnectorTool(f"{name}_search", name, fat=True) for name in servers]
    cost = ReActPattern()._tool_schema_cost(tools[0])
    llm = ScriptedLLM([{"content": "hi"}])
    await _pattern(deferred_preload_token_budget=cost + 10).run(
        context=_ctx(" and ".join(servers)), tools=tools, llm=llm
    )

    # The first connector in tool order wins whatever the hash seed is.
    assert [n for n in _names(llm.calls[0]) if n.endswith("_search")] == ["zeta_search"]


@pytest.mark.asyncio
async def test_a_forced_answer_turn_sends_no_loaded_connector_schema() -> None:
    tools = _tools()
    llm = ScriptedLLM(
        [
            _call(LOAD_TOOLS_NAME, {"names": ["notion_search"]}, "c1"),
            _call("notion_search", {"query": "plan"}, "c2"),
            _call(
                "final_answer",
                {
                    "response_language": "English",
                    "answer": "done",
                    "outcome": "completed",
                },
                "c3",
            ),
        ]
    )
    await ReActPattern(max_iterations=2, defer_connector_schemas=True).run(
        context=_ctx(), tools=tools, llm=llm
    )

    assert "notion_search" in _names(llm.calls[1])
    forced = _names(llm.calls[-1])
    assert "notion_search" not in forced and "final_answer" in forced


@pytest.mark.asyncio
async def test_already_loaded_names_do_not_use_the_per_call_cap() -> None:
    many = [ConnectorTool(f"hubspot_tool_{i}", "hubspot") for i in range(14)]
    names = [t.name for t in many]
    llm = ScriptedLLM(
        [
            _call(LOAD_TOOLS_NAME, {"names": names[:12]}, "c1"),
            _call(LOAD_TOOLS_NAME, {"names": names}, "c2"),
            {"content": "ok"},
        ]
    )
    context = _ctx()
    await _pattern().run(context=context, tools=many, llm=llm)

    second = context.get_messages_by_role("tool")[1].content
    assert "Loaded: hubspot_tool_12, hubspot_tool_13." in second
    assert "Not loaded" not in second
    assert second.count("hubspot_tool_0,") == 1  # listed once, as already loaded


@pytest.mark.asyncio
async def test_a_host_supplied_load_tools_disables_deferral() -> None:
    class HostLoadTools(PlainTool):
        pass

    tools = [HostLoadTools(LOAD_TOOLS_NAME), *_tools()]
    llm = ScriptedLLM([{"content": "hi"}])
    await _pattern().run(context=_ctx(), tools=tools, llm=llm)

    names = _names(llm.calls[0])
    assert names.count(LOAD_TOOLS_NAME) == 1
    assert "notion_search" in names


@pytest.mark.asyncio
async def test_a_misspelled_connector_tool_gets_a_load_hint() -> None:
    llm = ScriptedLLM([_call("notion_serch", {"query": "x"}), {"content": "ok"}])
    context = _ctx()
    await _pattern().run(context=context, tools=_tools(), llm=llm)

    error = context.get_messages_by_role("tool")[0].metadata["raw_result"]["error"]
    assert "notion_search" in error and LOAD_TOOLS_NAME in error


@pytest.mark.asyncio
async def test_a_replayed_committed_load_is_put_back() -> None:
    from xagent.core.agent.runtime import PatternRuntime

    tools = _tools()
    pattern = _pattern()
    context = _ctx()
    prepared = pattern._tools_with_deferral(tools, context)
    runtime = PatternRuntime(execution_id="replay")

    async def committed(_call: dict[str, Any]) -> dict[str, Any]:
        return {
            "result": {"success": True, "output": "Loaded: notion_search."},
            "error": None,
        }

    runtime.load_committed_tool_outcome = committed  # type: ignore[method-assign]
    call = {
        "id": "r1",
        "name": LOAD_TOOLS_NAME,
        "args": {"names": ["notion_search"]},
    }
    await pattern._execute_tool_safely(call, prepared, runtime, context=context)

    schemas = pattern._normal_tool_schemas([], None, prepared, context)
    assert [s["function"]["name"] for s in schemas] == ["notion_search"]


def test_service_flags_reach_the_react_and_auto_patterns() -> None:
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
                defer_connector_schemas=True,
                deferred_preload_token_budget=123,
            )
        )
        pattern, _ = adapter._build_pattern()
        react = pattern.react_pattern if mode == "auto" else pattern
        assert react.defer_connector_schemas is True
        assert react.deferred_preload_token_budget == 123
    off = AgentExecutionAdapter(
        AgentExecutionConfig(name="t", llm=object(), pattern="react")
    )
    assert off._build_pattern()[0].defer_connector_schemas is False


@pytest.mark.asyncio
async def test_loaded_schemas_go_after_the_stored_result_reader() -> None:
    from xagent.core.agent.context.components import SpillRegistryComponent
    from xagent.core.tools.tool_result_spill import SPILL_READ_TOOL_NAME

    tools = [*_tools(), PlainTool(SPILL_READ_TOOL_NAME)]
    context = _ctx()
    context.set_component(
        "spilled_results",
        SpillRegistryComponent(records=[{"relative_path": "stored/result.json"}]),
    )
    llm = ScriptedLLM(
        [
            _call(LOAD_TOOLS_NAME, {"names": ["notion_search"]}, "c1"),
            {"content": "done"},
        ]
    )
    await _pattern().run(context=context, tools=tools, llm=llm)

    first, second = _names(llm.calls[0]), _names(llm.calls[1])
    assert SPILL_READ_TOOL_NAME in first
    assert second[: len(first)] == first
    assert second[-1] == "notion_search"
