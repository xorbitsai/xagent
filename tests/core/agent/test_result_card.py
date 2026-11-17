"""``show_result_card``: off by default, provenance-checked, bounded, non-pausing."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterator

import pytest
from pydantic import BaseModel

from xagent.core.agent import ExecutionContext, PatternRuntime, ReActPattern
from xagent.core.agent.result_card import (
    MAX_FOOTNOTE_CHARS,
    MAX_ITEM_CHARS,
    MAX_ITEMS,
    MAX_TITLE_CHARS,
    RESULT_CARD_TOOL_NAME,
    ResultCardSource,
    RunDisplayContext,
    bind_run_display_context,
    set_result_card_source_resolver,
)
from xagent.core.tools.adapters.vibe.base import AbstractBaseTool
from xagent.core.tools.adapters.vibe.result_card_tool import (
    ShowResultCardTool,
    create_result_card_tool,
)

COSMOS_LOGO = {"mark": "◎", "bg": "#0078D4", "fg": "#fff", "icon_url": None}


@pytest.fixture(autouse=True)
def _reset_hook() -> Iterator[None]:
    yield
    set_result_card_source_resolver(None)


def _enable() -> None:
    set_result_card_source_resolver(
        lambda key: (
            ResultCardSource("Azure Cosmos DB", "cosmos", COSMOS_LOGO)
            if key == "azure_cosmos_db"
            else None
        )
    )


def _args(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "title": "Query · top customers",
        "source_connector": "Azure Cosmos DB",
        "items": ["Contoso Ltd · $182,400", "Northwind Traders · $149,950"],
        "footnote": "2 of 1,284 customers · read-only query",
    }
    base.update(overrides)
    return base


class _Sender:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.sent.append(kwargs)
        return {}


async def _call(args: dict[str, Any], invoked: set[str]) -> tuple[Any, _Sender]:
    sender = _Sender()
    ctx = RunDisplayContext(frozenset(invoked), sender)
    with bind_run_display_context(ctx):
        return await ShowResultCardTool().run_json_async(args), sender


@pytest.mark.asyncio
async def test_tool_is_not_offered_unless_the_host_enables_it() -> None:
    assert await create_result_card_tool(SimpleNamespace()) == []  # type: ignore[arg-type]
    _enable()
    tools = await create_result_card_tool(SimpleNamespace())  # type: ignore[arg-type]
    assert [t.name for t in tools] == [RESULT_CARD_TOOL_NAME]


@pytest.mark.asyncio
async def test_card_for_an_invoked_connector_is_emitted_in_the_contract_shape() -> None:
    _enable()
    result, sender = await _call(_args(), {"azure_cosmos_db"})
    assert result == {"success": True, "shown": True}
    (sent,) = sender.sent
    assert sent["message_type"] == "result_card"
    assert sent["expect_response"] is False
    metadata = sent["metadata"]
    assert metadata["display"] == "result_card"
    card = metadata["result_card"]
    assert card["title"] == "Query · top customers"
    assert card["source"] == {
        "name": "Azure Cosmos DB",
        "app_id": "cosmos",
        "logo": COSMOS_LOGO,
    }
    assert card["items"] == ["Contoso Ltd · $182,400", "Northwind Traders · $149,950"]
    assert card["footnote"] == "2 of 1,284 customers · read-only query"
    assert card["created_at"].endswith("Z")


@pytest.mark.asyncio
async def test_connector_without_display_data_falls_back_to_its_identity() -> None:
    _enable()
    result, sender = await _call(
        _args(source_connector="Other App"), {"other_app", "azure_cosmos_db"}
    )
    assert result["success"] is True
    assert sender.sent[0]["metadata"]["result_card"]["source"] == {
        "name": "other_app",
        "app_id": None,
        "logo": None,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("invoked", [set(), {"azure_cosmos_db_other"}, {"github"}])
async def test_forged_provenance_is_refused_without_emitting(invoked: set[str]) -> None:
    _enable()
    result, sender = await _call(_args(), invoked)
    assert result["success"] is False
    assert "was not invoked" in result["error"]
    assert sender.sent == []


@pytest.mark.asyncio
async def test_outside_a_run_nothing_is_emitted() -> None:
    _enable()
    result = await ShowResultCardTool().run_json_async(_args())
    assert result["success"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "override",
    [
        {"title": ""},
        {"title": "t" * (MAX_TITLE_CHARS + 1)},
        {"items": []},
        {"items": ["x"] * (MAX_ITEMS + 1)},
        {"items": ["ok", " "]},
        {"items": ["x" * (MAX_ITEM_CHARS + 1)]},
        {"footnote": "f" * (MAX_FOOTNOTE_CHARS + 1)},
    ],
)
async def test_oversized_or_empty_cards_are_refused(override: dict[str, Any]) -> None:
    _enable()
    result, sender = await _call(_args(**override), {"azure_cosmos_db"})
    assert result["success"] is False
    assert "Invalid result card" in result["error"]
    assert sender.sent == []


@pytest.mark.asyncio
async def test_limits_are_inclusive() -> None:
    _enable()
    result, sender = await _call(
        _args(
            title="t" * MAX_TITLE_CHARS,
            items=["x" * MAX_ITEM_CHARS] * MAX_ITEMS,
            footnote="f" * MAX_FOOTNOTE_CHARS,
        ),
        {"azure_cosmos_db"},
    )
    assert result["success"] is True and len(sender.sent) == 1


class _Args(BaseModel):
    pass


class _CosmosQuery(AbstractBaseTool):
    source_server = "azure_cosmos_db"

    def __init__(self) -> None:
        self.fail = False

    name = "cosmos_query"  # type: ignore[assignment]
    description = "Query"  # type: ignore[assignment]

    def args_type(self):
        return _Args

    def return_type(self):
        return _Args

    def run_json_sync(self, args):
        return {"success": not self.fail, "rows": 2}

    async def run_json_async(self, args):
        return self.run_json_sync(args)


class _ScriptedLLM:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses

    async def chat(self, **kwargs: Any) -> Any:
        return self.responses.pop(0)


def _call_of(name: str, call_id: str, **args: Any) -> dict[str, Any]:
    return {"id": call_id, "name": name, "arguments": args}


@pytest.mark.asyncio
async def test_react_run_validates_against_tools_it_really_called() -> None:
    _enable()
    cosmos = _CosmosQuery()
    tools = [cosmos, ShowResultCardTool()]
    llm = _ScriptedLLM(
        [
            # Forged: nothing was called yet.
            {"tool_calls": [_call_of(RESULT_CARD_TOOL_NAME, "c0", **_args())]},
            {"tool_calls": [_call_of("cosmos_query", "q1")]},
            {"tool_calls": [_call_of(RESULT_CARD_TOOL_NAME, "c1", **_args())]},
            {"content": "Done.", "done": True},
        ]
    )
    pattern = ReActPattern(max_iterations=8)
    runtime = PatternRuntime()
    context = ExecutionContext()
    context.add_user_message("Top customers?")

    result = await pattern.run(context=context, tools=tools, llm=llm, runtime=runtime)

    assert result["success"] is True  # the card never paused or ended the run
    assert pattern._record_for_tool_call_id("c0").result["success"] is False
    cards = [
        m for m in runtime.outbound_messages if m.get("message_type") == "result_card"
    ]
    assert len(cards) == 1
    assert cards[0]["metadata"]["result_card"]["source"]["app_id"] == "cosmos"
    assert cards[0]["expect_response"] is False


@pytest.mark.asyncio
async def test_failed_connector_call_does_not_count_as_invoked() -> None:
    _enable()
    cosmos = _CosmosQuery()
    cosmos.fail = True
    llm = _ScriptedLLM(
        [
            {"tool_calls": [_call_of("cosmos_query", "q1")]},
            {"tool_calls": [_call_of(RESULT_CARD_TOOL_NAME, "c1", **_args())]},
            {"content": "Done.", "done": True},
        ]
    )
    runtime = PatternRuntime()
    context = ExecutionContext()
    context.add_user_message("Top customers?")
    pattern = ReActPattern(max_iterations=8)
    await pattern.run(
        context=context,
        tools=[cosmos, ShowResultCardTool()],
        llm=llm,
        runtime=runtime,
    )
    assert "was not invoked" in pattern._record_for_tool_call_id("c1").result["error"]
    assert not [
        m for m in runtime.outbound_messages if m.get("message_type") == "result_card"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_factory_offers_the_tool_only_when_enabled(
    monkeypatch, tmp_path, enabled: bool
) -> None:
    from xagent.core.tools.adapters.vibe.config import ToolConfig
    from xagent.core.tools.adapters.vibe.factory import ToolFactory
    from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec

    monkeypatch.setenv("SANDBOX_ENABLED", "false")
    if enabled:
        _enable()
    config = ToolConfig(
        {"workspace": {"base_dir": str(tmp_path), "task_id": "result-card"}}
    )
    config._tool_selection_spec = ToolSelectionSpec.from_raw(
        tool_categories=["basic_tools"]
    )
    names = {tool.name for tool in await ToolFactory.create_all_tools(config)}
    assert (RESULT_CARD_TOOL_NAME in names) is enabled


def _raising_resolver(key: str) -> ResultCardSource:
    raise RuntimeError("host hook broke")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resolver", [_raising_resolver, lambda key: {"name": "not a source"}]
)
async def test_faulty_resolver_falls_back_to_the_connector_identity(
    resolver: Any,
) -> None:
    set_result_card_source_resolver(resolver)
    result, sender = await _call(_args(), {"azure_cosmos_db"})
    assert result == {"success": True, "shown": True}
    card = sender.sent[0]["metadata"]["result_card"]
    assert card["source"]["name"] == "azure_cosmos_db"
    assert card["source"]["logo"] is None
