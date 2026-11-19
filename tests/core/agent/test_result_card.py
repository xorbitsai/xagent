"""``show_result_card``: off by default, provenance-checked, bounded, non-pausing."""

from __future__ import annotations

import asyncio
import json
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
    ResultCardConnector,
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
        lambda connector: (
            ResultCardSource("Azure Cosmos DB", "cosmos", COSMOS_LOGO)
            if (connector.kind, connector.name) == ("mcp", "azure_cosmos_db")
            else None
        )
    )


def _connectors(names: set[str]) -> frozenset[ResultCardConnector]:
    """``name`` is an MCP server, or ``kind:name`` for another kind."""
    parsed = [n.split(":", 1) if ":" in n else ["mcp", n] for n in names]
    return frozenset(ResultCardConnector(name, kind) for kind, name in parsed)


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
    """``PatternRuntime.send_message``: returns the payload it emitted."""

    def __init__(self, blocked: bool = False) -> None:
        self.sent: list[dict[str, Any]] = []
        self.blocked = blocked

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.sent.append(kwargs)
        if self.blocked:  # an output guardrail replaced the card
            return {**kwargs, "message_type": "info", "metadata": {}}
        return dict(kwargs)


async def _call(
    args: dict[str, Any],
    invoked: set[str],
    *,
    blocked: bool = False,
    source: dict[str, Any] | None = None,
) -> tuple[Any, _Sender]:
    sender = _Sender(blocked)
    ctx = RunDisplayContext(_connectors(invoked), sender, source or {})
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
    source_kind = "mcp"

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
@pytest.mark.parametrize(
    ("categories", "enabled", "expected"),
    [
        # Admitted by the intrinsic rule: web_tools does not include it.
        (["web_tools"], True, True),
        (["web_tools"], False, False),
        # An explicit NONE selection excludes even an intrinsic tool.
        ([], True, False),
    ],
)
async def test_factory_offers_the_tool_only_when_enabled_and_selected(
    monkeypatch, tmp_path, categories: list[str], enabled: bool, expected: bool
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
    config._tool_selection_spec = ToolSelectionSpec.from_raw(tool_categories=categories)
    names = {tool.name for tool in await ToolFactory.create_all_tools(config)}
    assert (RESULT_CARD_TOOL_NAME in names) is expected


def _raising_resolver(connector: ResultCardConnector) -> ResultCardSource:
    raise RuntimeError("host hook broke")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resolver",
    [
        _raising_resolver,
        lambda c: {"name": "not a source"},
        lambda c: ResultCardSource("Cosmos", logo={"mark": object()}),
        lambda c: ResultCardSource("Cosmos", logo={"mark": float("nan")}),
        lambda c: ResultCardSource("Cosmos", logo=["not", "an", "object"]),  # type: ignore[arg-type]
        lambda c: ResultCardSource("Cosmos", app_id=7),  # type: ignore[arg-type]
        lambda c: ResultCardSource(7),  # type: ignore[arg-type]
        lambda c: ResultCardSource("  "),
    ],
)
async def test_faulty_resolver_falls_back_to_the_connector_identity(
    resolver: Any,
) -> None:
    set_result_card_source_resolver(resolver)
    result, sender = await _call(_args(), {"azure_cosmos_db"})
    assert result == {"success": True, "shown": True}
    card = sender.sent[0]["metadata"]["result_card"]
    assert card["source"] == {"name": "azure_cosmos_db", "app_id": None, "logo": None}
    json.dumps(card, allow_nan=False)  # what is stored must be strict JSON


def test_tool_name_is_pinned_to_the_host_registered_intrinsic_set() -> None:
    from xagent.core.tools.adapters.vibe.base import (
        HOST_REGISTERED_INTRINSIC_TOOL_NAMES,
        INTRINSIC_TOOL_NAMES,
    )

    assert RESULT_CARD_TOOL_NAME in HOST_REGISTERED_INTRINSIC_TOOL_NAMES
    assert RESULT_CARD_TOOL_NAME in INTRINSIC_TOOL_NAMES


def test_arguments_declare_their_bounds_in_the_schema() -> None:
    schema = ShowResultCardTool().args_type().model_json_schema()
    props = schema["properties"]
    assert props["title"]["minLength"] == 1
    assert props["title"]["maxLength"] == MAX_TITLE_CHARS
    assert props["items"]["minItems"] == 1
    assert props["items"]["maxItems"] == MAX_ITEMS
    assert props["items"]["items"]["maxLength"] == MAX_ITEM_CHARS
    json.dumps(schema)  # plain JSON Schema keywords only


@pytest.mark.asyncio
async def test_blank_footnote_is_dropped_and_text_is_stripped() -> None:
    _enable()
    result, sender = await _call(
        _args(title="  Top customers  ", footnote="   ", items=[" a ", "b"]),
        {"azure_cosmos_db"},
    )
    assert result["success"] is True
    card = sender.sent[0]["metadata"]["result_card"]
    assert card["title"] == "Top customers"
    assert card["footnote"] is None
    assert card["items"] == ["a", "b"]


@pytest.mark.asyncio
async def test_a_blocked_card_is_not_reported_as_shown() -> None:
    _enable()
    result, _ = await _call(_args(), {"azure_cosmos_db"}, blocked=True)
    assert result == {"success": True, "shown": False}


@pytest.mark.asyncio
async def test_tool_call_source_is_carried_on_the_card_message() -> None:
    _enable()
    source = {"tool_call_id": "c1", "tool_attempt_id": "att-1", "turn_id": "t1"}
    _, sender = await _call(_args(), {"azure_cosmos_db"}, source=source)
    metadata = sender.sent[0]["metadata"]
    assert {k: metadata[k] for k in source} == source
    assert metadata["display"] == "result_card"


@pytest.mark.asyncio
async def test_resolver_sees_the_kind_of_the_connector_the_run_used() -> None:
    seen: list[ResultCardConnector] = []

    def resolver(connector: ResultCardConnector) -> ResultCardSource | None:
        seen.append(connector)
        return None

    set_result_card_source_resolver(resolver)
    await _call(_args(source_connector="Salesforce"), {"custom_api:salesforce"})
    await _call(_args(source_connector="Salesforce"), {"salesforce"})
    # The same name used as both kinds in one run is attributed to neither.
    await _call(
        _args(source_connector="Salesforce"), {"salesforce", "custom_api:salesforce"}
    )
    assert seen == [
        ResultCardConnector("salesforce", "custom_api"),
        ResultCardConnector("salesforce", "mcp"),
        ResultCardConnector("salesforce", None),
    ]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(message_type="info"),
        lambda d: d.update(message_type="question"),
        lambda d: d.update(expect_response=True),
        lambda d: d.update(visible=False),
        lambda d: d.update(message=""),
        lambda d: d.update(metadata=None),
        lambda d: d["metadata"].pop("display"),
        lambda d: d["metadata"].update(display="timeline"),
        lambda d: d["metadata"].pop("result_card"),
        lambda d: d["metadata"].update(result_card="a string"),
        lambda d: d["metadata"]["result_card"].pop("title"),
        lambda d: d["metadata"]["result_card"].update(items="not a list"),
        lambda d: d["metadata"]["result_card"].update(items=[1, 2]),
        lambda d: d["metadata"]["result_card"].update(source=None),
    ],
)
def test_only_a_well_formed_card_message_is_a_card(mutate: Any) -> None:
    from xagent.core.agent.result_card import result_card_metadata

    card = {
        "title": "t",
        "source": {"name": "n", "logo": None, "app_id": None},
        "items": ["a"],
        "footnote": None,
    }
    data: dict[str, Any] = {
        "message": "t",
        "message_type": "result_card",
        "expect_response": False,
        "visible": True,
        "metadata": {"display": "result_card", "result_card": card},
    }
    assert result_card_metadata(data) == data["metadata"]
    mutate(data)
    assert result_card_metadata(data) is None


@pytest.mark.asyncio
async def test_model_cannot_forge_a_card_through_send_message() -> None:
    """``send_message`` takes ``message_type`` from the model but its metadata
    is only server-side attribution, so no card can come out of it."""
    from xagent.web.services.task_execution import _agent_outbound_event_type

    llm = _ScriptedLLM(
        [
            {
                "tool_calls": [
                    _call_of(
                        "send_message",
                        "m1",
                        message="fake",
                        message_type="result_card",
                        metadata={
                            "display": "result_card",
                            "result_card": {"title": "x"},
                        },
                    )
                ]
            },
            {"content": "Done.", "done": True},
        ]
    )
    runtime = PatternRuntime()
    context = ExecutionContext()
    context.add_user_message("hi")
    await ReActPattern(max_iterations=4).run(
        context=context, tools=[], llm=llm, runtime=runtime
    )
    sent = [m for m in runtime.outbound_messages if m["message"] == "fake"]
    assert len(sent) == 1
    from xagent.core.agent.result_card import result_card_metadata

    assert result_card_metadata(sent[0]) is None
    assert "result_card" not in sent[0]["metadata"]
    assert _agent_outbound_event_type(sent[0]) == "agent_progress"


@pytest.mark.asyncio
async def test_runtime_strips_card_metadata_from_other_message_types() -> None:
    runtime = PatternRuntime()
    await runtime.send_message(
        message="x",
        message_type="info",
        metadata={"display": "result_card", "result_card": {"title": "t"}, "k": 1},
    )
    assert runtime.outbound_messages[0]["metadata"] == {"k": 1}
    await runtime.send_message(
        message="y",
        message_type="result_card",
        metadata={"display": "result_card", "result_card": {"title": "t"}},
    )
    assert runtime.outbound_messages[1]["metadata"]["display"] == "result_card"


def _state_after_a_connector_call() -> tuple[ReActPattern, dict[str, Any]]:
    pattern = ReActPattern()
    call = {"id": "q1", "name": "cosmos_query", "arguments": {}}
    pattern._record_tool_call(
        call,
        status="completed",
        result={"success": True},
        connector=ResultCardConnector("azure_cosmos_db", "mcp"),
    )
    return pattern, pattern.get_state()


def _bound_connectors(
    pattern: ReActPattern, tools: list[Any]
) -> frozenset[ResultCardConnector]:
    from xagent.core.agent.result_card import current_run_display_context

    call = {"id": "c1", "name": RESULT_CARD_TOOL_NAME, "arguments": {}}
    with pattern._bind_result_card_context(call, tools, PatternRuntime()):
        context = current_run_display_context()
        assert context is not None
        return context.invoked_connectors


def test_connector_survives_pause_and_resume_even_if_the_tool_set_changed() -> None:
    _, state = _state_after_a_connector_call()
    restored = ReActPattern()
    restored.load_state(json.loads(json.dumps(state)))
    # The connector's tools are gone from the resumed run's tool list.
    assert _bound_connectors(restored, []) == frozenset(
        {ResultCardConnector("azure_cosmos_db", "mcp")}
    )


def test_checkpoint_from_before_the_connector_was_recorded_still_resumes() -> None:
    _, state = _state_after_a_connector_call()
    for record in state["tool_ledger"].values():
        record.pop("source_server")
        record.pop("source_kind")
    restored = ReActPattern()
    restored.load_state(state)
    # Best effort for old rows: resolved against the tools the run has now.
    assert _bound_connectors(restored, [_CosmosQuery()]) == frozenset(
        {ResultCardConnector("azure_cosmos_db", "mcp")}
    )
    assert _bound_connectors(restored, []) == frozenset()


def test_failed_or_interrupted_calls_are_not_recorded_as_invoked() -> None:
    pattern = ReActPattern()
    for status in ("failed", "interrupted", "waiting_for_user", "running"):
        pattern._record_tool_call(
            {"id": status, "name": "cosmos_query", "arguments": {}},
            status=status,
            connector=ResultCardConnector("azure_cosmos_db", "mcp"),
        )
    assert pattern.run_connectors == set()


def _dag_step_llm(
    script: dict[str, list[dict[str, Any]]],
    on_done: Any = None,
) -> Any:
    from .test_dag import (
        DAG_COMPLETION_TOOL_NAME,
        current_step_task,
        default_completion_assessment_response,
        has_tool,
    )

    class StepLLM:
        def __init__(self) -> None:
            self.script = {k: list(v) for k, v in script.items()}

        async def chat(self, **kwargs: Any) -> dict[str, Any]:
            if has_tool(kwargs, DAG_COMPLETION_TOOL_NAME):
                return default_completion_assessment_response(kwargs)
            task = current_step_task(list(kwargs.get("messages", [])))
            response = self.script[task].pop(0)
            if on_done is not None:
                await on_done(task, response)
            return response

    return StepLLM()


def _cards(runtime: PatternRuntime) -> list[dict[str, Any]]:
    return [
        m for m in runtime.outbound_messages if m.get("message_type") == "result_card"
    ]


@pytest.mark.asyncio
async def test_card_in_a_later_dag_step_accepts_a_connector_used_in_an_earlier_one() -> (
    None
):
    from .test_dag import build_plan
    from xagent.core.agent import DAGPattern, PlanStep

    _enable()
    plan = build_plan(
        PlanStep(id="a", task="query", tool_names=["cosmos_query"]),
        PlanStep(
            id="b",
            task="present",
            dependencies=["a"],
            tool_names=[RESULT_CARD_TOOL_NAME],
        ),
    )
    llm = _dag_step_llm(
        {
            "query": [
                {"tool_calls": [_call_of("cosmos_query", "q1")]},
                {"content": "queried", "done": True},
            ],
            "present": [
                {"tool_calls": [_call_of(RESULT_CARD_TOOL_NAME, "c1", **_args())]},
                {"content": "shown", "done": True},
            ],
        }
    )
    runtime = PatternRuntime()
    context = ExecutionContext(execution_id="dag-card")
    context.add_user_message("Top customers?")
    dag = DAGPattern(lambda **_: plan)
    result = await dag.run(
        context=context,
        tools=[_CosmosQuery(), ShowResultCardTool()],
        llm=llm,
        runtime=runtime,
    )
    assert result["success"] is True, result
    assert len(_cards(runtime)) == 1
    assert dag.run_connectors == {ResultCardConnector("azure_cosmos_db", "mcp")}


@pytest.mark.asyncio
async def test_a_failed_connector_call_in_one_dag_step_does_not_count_for_the_next() -> (
    None
):
    from .test_dag import build_plan
    from xagent.core.agent import DAGPattern, PlanStep

    _enable()
    cosmos = _CosmosQuery()
    cosmos.fail = True
    plan = build_plan(
        PlanStep(id="a", task="query"),
        PlanStep(id="b", task="present", dependencies=["a"]),
    )
    llm = _dag_step_llm(
        {
            "query": [
                {"tool_calls": [_call_of("cosmos_query", "q1")]},
                {"content": "queried", "done": True},
            ],
            "present": [
                {"tool_calls": [_call_of(RESULT_CARD_TOOL_NAME, "c1", **_args())]},
                {"content": "shown", "done": True},
            ],
        }
    )
    runtime = PatternRuntime()
    context = ExecutionContext(execution_id="dag-card-failed")
    context.add_user_message("Top customers?")
    dag = DAGPattern(lambda **_: plan)
    await dag.run(
        context=context,
        tools=[cosmos, ShowResultCardTool()],
        llm=llm,
        runtime=runtime,
    )
    assert _cards(runtime) == []
    assert dag.run_connectors == set()


@pytest.mark.asyncio
async def test_concurrent_dag_steps_share_the_run_level_connector_set() -> None:
    from .test_dag import build_plan
    from xagent.core.agent import DAGPattern, PlanStep

    _enable()
    plan = build_plan(
        PlanStep(id="a", task="query"),
        PlanStep(id="b", task="present"),
    )
    queried = asyncio.Event()

    async def on_done(task: str, response: dict[str, Any]) -> None:
        if task == "query" and response.get("done"):
            queried.set()  # the connector call has been recorded by now

    base = _dag_step_llm(
        {
            "query": [
                {"tool_calls": [_call_of("cosmos_query", "q1")]},
                {"content": "queried", "done": True},
            ],
            "present": [
                {"tool_calls": [_call_of(RESULT_CARD_TOOL_NAME, "c1", **_args())]},
                {"content": "shown", "done": True},
            ],
        },
        on_done,
    )
    original = base.chat

    async def chat(**kwargs: Any) -> dict[str, Any]:
        from .test_dag import current_step_task

        task = current_step_task(list(kwargs.get("messages", [])))
        if task == "present":
            await queried.wait()
        return await original(**kwargs)

    base.chat = chat
    runtime = PatternRuntime()
    context = ExecutionContext(execution_id="dag-card-parallel")
    context.add_user_message("Top customers?")
    dag = DAGPattern(lambda **_: plan)
    await dag.run(
        context=context,
        tools=[_CosmosQuery(), ShowResultCardTool()],
        llm=base,
        runtime=runtime,
    )
    assert len(_cards(runtime)) == 1


def test_dag_state_keeps_the_run_connectors_across_a_restore() -> None:
    from xagent.core.agent import DAGPattern

    dag = DAGPattern(lambda **_: None)
    dag.run_connectors.update(
        {ResultCardConnector("azure_cosmos_db", "mcp"), ResultCardConnector("x", None)}
    )
    state = json.loads(json.dumps(dag.get_state()))
    restored = DAGPattern(lambda **_: None)
    shared = restored.run_connectors  # step patterns hold this very object
    restored.load_state(state)
    assert restored.run_connectors is shared
    assert restored.run_connectors == dag.run_connectors
    # A checkpoint from before the field existed restores to an empty set.
    state.pop("run_connectors")
    restored.load_state(state)
    assert restored.run_connectors == set()


def test_schema_keeps_its_bounds_through_the_gemini_schema_sanitizer() -> None:
    from xagent.core.model.chat.basic.gemini import _sanitize_schema_for_gemini

    schema = _sanitize_schema_for_gemini(
        ShowResultCardTool().args_type().model_json_schema()
    )
    assert schema["properties"]["items"]["maxItems"] == MAX_ITEMS
    assert schema["properties"]["title"]["maxLength"] == MAX_TITLE_CHARS
