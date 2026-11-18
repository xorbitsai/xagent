"""``find_app_connection``: resolver-driven connect card and server-side resume."""

from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from xagent.core.agent import ExecutionContext, PatternRuntime, ReActPattern
from xagent.core.tools import connector_resolver
from xagent.core.tools.adapters.vibe.find_app_connection_tool import (
    FindAppConnectionTool,
    create_find_app_connection_tool,
)
from xagent.core.tools.adapters.vibe.interaction_types import (
    INTERACTION_TYPE_ALIASES,
    INTERACTION_TYPES,
)
from xagent.core.tools.adapters.vibe.mcp_approval_gate import (
    ToolCallExecutionContext,
    bind_tool_call_execution_context,
)
from xagent.core.tools.connector_resolver import (
    ConnectorConnectionState,
    ConnectorResolution,
    ConnectorResolveRequest,
)
from xagent.core.tools.user_interaction import (
    ToolInteractionSettlement,
    tool_result_waits_for_user,
    user_interaction_resume_callable,
)


class FakeResolver:
    def __init__(
        self,
        resolution: ConnectorResolution | None = None,
        state: ConnectorConnectionState | None = None,
    ) -> None:
        self.resolution = resolution
        self.state = state
        self.resolve_error: Exception | None = None
        self.state_error: Exception | None = None
        self.requests: list[ConnectorResolveRequest] = []
        self.state_calls: list[dict[str, Any]] = []

    async def resolve(self, request: ConnectorResolveRequest) -> ConnectorResolution:
        self.requests.append(request)
        if self.resolve_error is not None:
            raise self.resolve_error
        assert self.resolution is not None
        return self.resolution

    async def connection_state(
        self,
        *,
        interaction_id: str,
        user_id: int | None,
        task_id: str | None,
        task_source: str | None,
    ) -> ConnectorConnectionState | None:
        self.state_calls.append(
            {
                "interaction_id": interaction_id,
                "user_id": user_id,
                "task_id": task_id,
                "task_source": task_source,
            }
        )
        if self.state_error is not None:
            raise self.state_error
        return self.state


def _tool(resolver: FakeResolver) -> FindAppConnectionTool:
    return FindAppConnectionTool(resolver=resolver, user_id=7, task_id="t-1")


ARGS = {"app_name": "Cosmos", "purpose": "Count orders"}
CONNECTOR = {"name": "Azure Cosmos DB", "fields": [{"label": "Endpoint"}]}


@pytest.mark.asyncio
async def test_needs_connection_returns_a_tool_authored_waiting_result() -> None:
    resolver = FakeResolver(
        ConnectorResolution("needs_connection", connector=CONNECTOR)
    )
    result = await _tool(resolver).run_json_async(ARGS)

    assert tool_result_waits_for_user(result)
    [interaction] = result["interactions"]
    assert interaction["type"] == "connect_connector"
    assert interaction["connector"] == CONNECTOR
    assert interaction["live_widget"] is True
    # The id the resolver was given is the id the user is waiting on.
    assert result["interaction_id"] == resolver.requests[0].interaction_id
    assert resolver.requests[0].user_id == 7
    assert resolver.requests[0].purpose == "Count orders"


def test_connect_connector_is_not_model_facing() -> None:
    assert "connect_connector" not in INTERACTION_TYPES
    assert "connect_connector" not in INTERACTION_TYPE_ALIASES


@pytest.mark.asyncio
async def test_already_connected_succeeds_without_waiting() -> None:
    resolver = FakeResolver(ConnectorResolution("connected", result={"server": "db"}))
    result = await _tool(resolver).run_json_async(ARGS)

    assert not tool_result_waits_for_user(result)
    assert result["success"] is True
    assert result["status"] == "connected"
    assert result["server"] == "db"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resolution",
    [
        ConnectorResolution("not_found"),
        # A needs_connection with nothing to render must not strand the run.
        ConnectorResolution("needs_connection", connector=None),
    ],
)
async def test_not_found_is_a_plain_result(resolution: ConnectorResolution) -> None:
    result = await _tool(FakeResolver(resolution)).run_json_async(ARGS)

    assert not tool_result_waits_for_user(result)
    assert result["status"] == "not_found"
    assert "not available" in result["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            ConnectorConnectionState("connected", result={"server": "db"}),
            {"success": True, "status": "connected", "server": "db"},
        ),
        (ConnectorConnectionState("skipped"), {"status": "skipped"}),
    ],
)
async def test_resume_settles_from_server_state(
    state: ConnectorConnectionState, expected: dict[str, Any]
) -> None:
    resolver = FakeResolver(state=state)
    resume = user_interaction_resume_callable(_tool(resolver))
    assert resume is not None

    settlement = await resume(interaction_id="i-1", response="ignored")

    assert isinstance(settlement, ToolInteractionSettlement)
    assert settlement.status == "succeeded"
    assert expected.items() <= settlement.result.items()
    assert resolver.state_calls == [
        {
            "interaction_id": "i-1",
            "user_id": 7,
            "task_id": "t-1",
            "task_source": None,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state", [None, ConnectorConnectionState("pending")], ids=["unknown", "pending"]
)
async def test_resume_returns_none_so_the_model_replans(state) -> None:
    assert (
        await _tool(FakeResolver(state=state)).resume_user_interaction(
            interaction_id="i-1", response="done"
        )
        is None
    )


@pytest.mark.asyncio
async def test_resume_ignores_the_reply_text() -> None:
    """A reply claiming success must not settle anything; a secret in it is
    never read."""

    pending = FakeResolver(state=ConnectorConnectionState("pending"))
    assert (
        await _tool(pending).resume_user_interaction(
            interaction_id="i-1", response="connected! key=SECRET-123"
        )
        is None
    )
    connected = FakeResolver(state=ConnectorConnectionState("connected"))
    settlement = await _tool(connected).resume_user_interaction(
        interaction_id="i-1", response="SECRET-123"
    )
    assert settlement is not None
    assert "SECRET-123" not in repr(settlement)


@pytest.fixture
def _clean_hook():
    previous = connector_resolver.get_connector_resolver_hook()
    connector_resolver.set_connector_resolver_hook(None)
    yield
    connector_resolver.set_connector_resolver_hook(previous)


class _Config:
    def __init__(self, parent_task_id: str | None = None) -> None:
        self._parent_task_id = parent_task_id

    def get_parent_task_id(self) -> str | None:
        return self._parent_task_id

    def get_user_id(self) -> int:
        return 7

    def get_task_id(self) -> str:
        return "t-1"


@pytest.mark.asyncio
async def test_tool_is_offered_only_when_a_resolver_is_registered(_clean_hook) -> None:
    assert await create_find_app_connection_tool(_Config()) == []  # type: ignore[arg-type]

    resolver = FakeResolver()
    connector_resolver.set_connector_resolver_hook(resolver)
    assert connector_resolver.get_connector_resolver_hook() is resolver
    [tool] = await create_find_app_connection_tool(_Config())  # type: ignore[arg-type]
    assert tool.name == "find_app_connection"


def _source(task_source: str | None):
    if task_source is None:
        return nullcontext()
    return bind_tool_call_execution_context(
        ToolCallExecutionContext(
            task_source=task_source,
            task_id="t-1",
            run_id="r-1",
            turn_id="u-1",
            tool_call_id="c-1",
            pattern="react",
            react_step_id="s-1",
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["widget", "shared_link"])
async def test_guest_sources_are_refused_without_calling_the_resolver(source) -> None:
    resolver = FakeResolver(
        ConnectorResolution("needs_connection", connector=CONNECTOR),
        ConnectorConnectionState("connected"),
    )
    with _source(source):
        result = await _tool(resolver).run_json_async(ARGS)
        resumed = await _tool(resolver).resume_user_interaction(
            interaction_id="i-1", response=""
        )

    assert result["status"] == "unavailable"
    assert not tool_result_waits_for_user(result)
    assert resumed is None
    assert resolver.requests == [] and resolver.state_calls == []


@pytest.mark.asyncio
async def test_task_source_reaches_both_resolver_methods() -> None:
    resolver = FakeResolver(
        ConnectorResolution("needs_connection", connector=CONNECTOR),
        ConnectorConnectionState("connected"),
    )
    with _source("internal"):
        await _tool(resolver).run_json_async(ARGS)
        await _tool(resolver).resume_user_interaction(interaction_id="i", response="")

    assert resolver.requests[0].task_source == "internal"
    assert resolver.state_calls[0]["task_source"] == "internal"


@pytest.mark.asyncio
async def test_a_raising_state_lookup_replans_instead_of_failing() -> None:
    resolver = FakeResolver(state=ConnectorConnectionState("connected"))
    resolver.state_error = RuntimeError("db down")
    assert (
        await _tool(resolver).resume_user_interaction(interaction_id="i", response="")
        is None
    )


@pytest.mark.asyncio
async def test_a_raising_resolver_gives_a_plain_unavailable_result() -> None:
    resolver = FakeResolver()
    resolver.resolve_error = RuntimeError("db down")
    result = await _tool(resolver).run_json_async(ARGS)
    assert result["status"] == "unavailable"
    assert "db down" not in repr(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hostile",
    [
        {"status": "error"},
        {"success": False},
        {"is_error": True},
        {"status": "waiting_for_user", "message": "x"},
    ],
)
async def test_host_result_cannot_override_control_keys(hostile) -> None:
    resolver = FakeResolver(
        ConnectorResolution("connected", result={**hostile, "server": "db"}),
        ConnectorConnectionState("connected", result={**hostile, "server": "db"}),
    )
    result = await _tool(resolver).run_json_async(ARGS)
    assert result["success"] is True and result["status"] == "connected"
    assert result["server"] == "db" and "is_error" not in result

    settlement = await _tool(resolver).resume_user_interaction(
        interaction_id="i", response=""
    )
    assert settlement is not None and settlement.status == "succeeded"
    assert settlement.result["status"] == "connected"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resolution",
    [
        ConnectorResolution("needs_connection", connector=None),
        ConnectorResolution("bogus"),  # type: ignore[arg-type]
    ],
    ids=["no-connector", "unknown-status"],
)
async def test_host_contract_violations_are_logged(resolution, caplog) -> None:
    caplog.set_level(logging.WARNING)
    result = await _tool(FakeResolver(resolution)).run_json_async(ARGS)
    assert result["status"] == "not_found"
    assert any("find_app_connection" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_skipped_tells_the_model_not_to_retry() -> None:
    settlement = await _tool(
        FakeResolver(state=ConnectorConnectionState("skipped"))
    ).resume_user_interaction(interaction_id="i", response="")
    assert settlement is not None
    assert settlement.result["status"] == "skipped"
    assert "Do not call" in settlement.result["message"]


@pytest.mark.asyncio
async def test_delegated_child_agents_are_not_offered_the_tool(_clean_hook) -> None:
    connector_resolver.set_connector_resolver_hook(FakeResolver())
    assert await create_find_app_connection_tool(_Config(parent_task_id="p-1")) == []  # type: ignore[arg-type]


def test_by_category_selection_without_basic_still_admits_the_tool() -> None:
    from xagent.core.tools.adapters.vibe.base import INTRINSIC_TOOL_NAMES
    from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec

    assert "find_app_connection" in INTRINSIC_TOOL_NAMES
    tools = [
        SimpleNamespace(
            name=name,
            metadata=SimpleNamespace(
                name=name, category=SimpleNamespace(value="other"), source_server=None
            ),
        )
        for name in ("find_app_connection", "calculator")
    ]
    allowed = ToolSelectionSpec.from_raw(
        tool_categories=["knowledge"]
    ).compute_allowed_names(tools)
    assert allowed is not None
    assert "find_app_connection" in allowed and "calculator" not in allowed


class _FakeLLM:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses

    async def chat(self, **kwargs: Any) -> Any:
        return self.responses.pop(0)


def _call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "tool_calls": [
            {"id": call_id, "function": {"name": name, "arguments": json.dumps(args)}}
        ]
    }


@pytest.mark.asyncio
async def test_end_to_end_pause_checkpoint_resume_settlement() -> None:
    resolver = FakeResolver(
        ConnectorResolution("needs_connection", connector=CONNECTOR)
    )
    context = ExecutionContext(execution_id="e2e-connect")
    context.add_user_message("Count my Cosmos orders")
    runtime = PatternRuntime(execution_id="e2e-connect")
    pattern = ReActPattern(max_iterations=4)

    waiting = await pattern.run(
        context=context,
        tools=[_tool(resolver)],
        llm=_FakeLLM([_call("find_app_connection", ARGS, "c1")]),
        runtime=runtime,
    )

    assert waiting["status"] == "waiting_for_user"
    # The card is the only control: no default text box beside it.
    assert [i["type"] for i in waiting["interactions"]] == ["connect_connector"]
    _assert_interaction_id_on_every_layer(waiting, pattern, runtime, resolver)
    [published] = [m for m in runtime.outbound_messages if m.get("expect_response")]
    assert published["metadata"]["interactions"] == waiting["interactions"]

    # The credential goes through the host endpoint; the chat reply is only
    # a nudge and carries a decoy that must never be read.
    resolver.state = ConnectorConnectionState("connected", result={"server": "db"})
    context.add_user_message("done, key=SECRET-123")
    resumed = ReActPattern(max_iterations=4)
    resumed.load_state(pattern.get_state())
    final = await resumed.run(
        context=context,
        tools=[_tool(resolver)],
        llm=_FakeLLM(
            [
                _call(
                    "final_answer",
                    {
                        "response_language": "English",
                        "answer": "Connected.",
                        "outcome": "completed",
                    },
                    "c2",
                )
            ]
        ),
    )

    assert final["success"] is True
    assert resolver.state_calls[0]["interaction_id"] == (
        resolver.requests[0].interaction_id
    )
    record = resumed._record_for_tool_call_id("c1")
    assert record.status == "completed"
    assert record.result["status"] == "connected" and record.result["server"] == "db"
    assert "SECRET-123" not in repr(record.result)


def _assert_interaction_id_on_every_layer(waiting, pattern, runtime, resolver) -> None:
    """The id the card needs must survive every hop from tool result to client."""
    from xagent.web.services.task_clarification_draft import (
        build_clarification_payload,
        parse_clarification_payload,
    )

    issued = resolver.requests[0].interaction_id
    [item] = waiting["interactions"]
    assert item["interaction_id"] == issued
    assert item["connector"] == CONNECTOR  # forwarded untouched
    # What the client receives on the outbound waiting message.
    [published] = [m for m in runtime.outbound_messages if m.get("expect_response")]
    assert published["metadata"]["interactions"][0]["interaction_id"] == issued
    # The engine's own record of the wait, and the per-tool attribution.
    request = pattern.waiting_for_user_request
    assert request["interactions"][0]["interaction_id"] == issued
    assert request["requests"][0]["interaction_id"] == issued
    assert request["requests"][0]["interactions"][0]["interaction_id"] == issued
    # The draft that is persisted and replayed.
    draft = waiting["clarification_draft"]
    assert draft.interactions[0]["interaction_id"] == issued
    payload = build_clarification_payload(draft)
    assert payload["interactions"][0]["interaction_id"] == issued
    _, legacy_interactions = parse_clarification_payload(payload)
    assert legacy_interactions is not None
    assert legacy_interactions[0]["interaction_id"] == issued
    assert legacy_interactions[0]["connector"] == CONNECTOR


class _AskTool:
    """A second, callback-less tool that pauses with an ordinary text field."""

    def __init__(self) -> None:
        self.metadata = SimpleNamespace(name="ask_region", description="Ask.")

    def args_type(self) -> type[BaseModel]:
        class Args(BaseModel):
            pass

        return Args

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        return {
            "success": False,
            "status": "waiting_for_user",
            "interaction_id": "ask-1",
            "message": "Which region?",
            "message_type": "question",
            "interactions": [{"type": "text_input", "field": "region", "label": "R"}],
        }


def _two_calls() -> dict[str, Any]:
    return {
        "tool_calls": [
            {
                "id": "c-connect",
                "function": {
                    "name": "find_app_connection",
                    "arguments": json.dumps(ARGS),
                },
            },
            {"id": "c-ask", "function": {"name": "ask_region", "arguments": "{}"}},
        ]
    }


def _final() -> dict[str, Any]:
    return _call(
        "final_answer",
        {
            "response_language": "English",
            "answer": "Done.",
            "outcome": "completed",
        },
        "c-final",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("connected", [False, True], ids=["pending", "connected"])
async def test_batch_resumed_by_the_other_tools_answer(connected: bool) -> None:
    """Two tools pause together; the user answers only the other tool.

    The reply is delivered to every resumable interaction. While the connection
    is still pending ``find_app_connection`` returns ``None`` (its call stays
    unanswered, the run is not failed); once the host recorded the connection
    it settles from that state regardless of the reply text.
    """
    resolver = FakeResolver(
        ConnectorResolution("needs_connection", connector=CONNECTOR)
    )

    class _ParallelSafeConnect(FindAppConnectionTool):
        """The tool is not concurrency-safe, so ReAct never co-pauses it with
        another tool; it is marked safe here only to force a shared batch."""

        @property
        def metadata(self):
            return super().metadata.model_copy(update={"concurrency_safe": True})

    ask = _AskTool()
    ask.metadata = SimpleNamespace(
        name="ask_region", description="Ask.", concurrency_safe=True
    )
    tools = [_ParallelSafeConnect(resolver=resolver, user_id=7, task_id="t-1"), ask]
    context = ExecutionContext(execution_id="batch-connect")
    context.add_user_message("Connect Cosmos and pick a region")
    runtime = PatternRuntime(execution_id="batch-connect")
    pattern = ReActPattern(max_iterations=4, tool_parallel_enabled=True)
    waiting = await pattern.run(
        context=context,
        tools=tools,
        llm=_FakeLLM([_two_calls()]),
        runtime=runtime,
    )
    assert waiting["status"] == "waiting_for_user"
    # The sibling's text field is kept: a live widget does not hide it.
    assert {i["type"] for i in waiting["interactions"]} == {
        "connect_connector",
        "text_input",
    }

    resolver.state = ConnectorConnectionState("connected" if connected else "pending")
    context.add_user_message("eu-west")
    resumed = ReActPattern(max_iterations=4, tool_parallel_enabled=True)
    resumed.load_state(pattern.get_state())
    final = await resumed.run(context=context, tools=tools, llm=_FakeLLM([_final()]))

    assert final["success"] is True  # the run was not failed
    assert [c["interaction_id"] for c in resolver.state_calls] == [
        resolver.requests[0].interaction_id
    ]
    record = resumed._record_for_tool_call_id("c-connect")
    assert record.status == ("completed" if connected else "waiting_for_user")
