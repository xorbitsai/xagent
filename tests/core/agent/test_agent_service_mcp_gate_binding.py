"""End to end: a host's ``context`` dict decides whether the gate fires.

This is the seam the host wiring exists for. A channel bot (or the WebSocket
turn path) puts the task row's ``source`` into the ``context`` dict it hands
``AgentService.execute_task``; the runner mirrors that into
``ExecutionContext.metadata``; ReAct reads ``metadata["task_source"]`` when it
builds the ``ToolCallExecutionContext``; and the gate selects a registration
by that exact string.

The two cases below are the enablement contract in one file:

* A source with a registration (``"slack"``) stops before dispatch.
* A source without one (``"telegram"``) dispatches exactly as it did before
  the wrapper existed -- no fail-closed blast radius for other hosts.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import BaseModel

from xagent.core.agent.service import AgentService
from xagent.core.tools.adapters.vibe.base import AbstractBaseTool, ToolMetadata
from xagent.core.tools.adapters.vibe.mcp_approval_gate import (
    GatedCall,
    GateDecision,
    gate_mcp_tools,
    register_mcp_approval_gate,
    unregister_mcp_approval_gate,
)


class _WriteArgs(BaseModel):
    text: str = ""


class _ConnectorWrite(AbstractBaseTool):
    """Stands in for a wrapped MCP write tool."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._metadata = ToolMetadata(
            name="mcp_LinkedIn_create_post",
            description="Create a post.",
            concurrency_safe=False,
            read_only=False,
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    @property
    def name(self) -> str:
        return "mcp_LinkedIn_create_post"

    @property
    def description(self) -> str:
        return "Create a post."

    def args_type(self) -> type[BaseModel]:
        return _WriteArgs

    def return_type(self) -> type[BaseModel]:
        return BaseModel

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise AssertionError("the async path is the one under test")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        self.calls.append(dict(args))
        return {"success": True, "posted": dict(args)}


class _ScriptedLLM:
    model_name = "stub-model"

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)

    async def chat(self, **_kwargs: Any) -> Any:
        return self.responses.pop(0)


def _llm() -> _ScriptedLLM:
    return _ScriptedLLM(
        [
            {
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "mcp_LinkedIn_create_post",
                            "arguments": '{"text":"publish"}',
                        },
                    }
                ]
            },
            {"content": "Done.", "done": True},
        ]
    )


async def _run_turn(*, task_source: str, target: _ConnectorWrite) -> dict[str, Any]:
    (gated,) = gate_mcp_tools([target], connection={"id": 41})
    service = AgentService(
        name="channel-agent",
        id="svc-gate",
        tools=[gated],
        llm=_llm(),
        pattern="react",
    )
    return await service.execute_task(
        "Publish the post.",
        # Exactly the dict shape the channel bots and the WebSocket turn
        # path build. ``run_id`` rides along the same way.
        context={"task_source": task_source, "run_id": "run-1", "turn_id": "turn-1"},
        task_id="task-248032",
    )


@pytest.mark.asyncio
async def test_a_registered_source_stops_the_write_at_the_gate() -> None:
    seen: list[GatedCall] = []

    async def gate(call: GatedCall) -> GateDecision:
        seen.append(call)
        return GateDecision.deny(message="Not approved.")

    async def resume(**_: Any) -> None:
        raise AssertionError("resume must not run for a denied call")

    handle = register_mcp_approval_gate(task_source="slack", gate=gate, resume=resume)
    target = _ConnectorWrite()
    try:
        await _run_turn(task_source="slack", target=target)
    finally:
        unregister_mcp_approval_gate(handle)

    assert target.calls == [], "the connector write must not have been dispatched"
    assert len(seen) == 1
    execution = seen[0].execution_context
    assert execution.task_source == "slack"
    assert execution.task_id == "task-248032"
    assert execution.run_id == "run-1"
    assert execution.turn_id == "turn-1"
    assert execution.tool_call_id == "call-1"
    assert seen[0].connector_ref.to_wire() == {
        "connector_type": "mcp",
        "connector_id": 41,
    }


@pytest.mark.asyncio
async def test_an_unregistered_source_dispatches_exactly_as_before() -> None:
    """The other half of the contract, and the mutation guard for the first
    test: the same registration, the same wrapped tool, a different bound
    source. If the wrapper ever gated on ``_has_registrations()`` instead of
    the call's own source, this would fail with an unavailable-gate error
    for every non-Toby host in the process."""

    async def gate(_call: GatedCall) -> GateDecision:
        raise AssertionError("a telegram call must not reach the slack hook")

    async def resume(**_: Any) -> None:
        raise AssertionError("resume must not run")

    handle = register_mcp_approval_gate(task_source="slack", gate=gate, resume=resume)
    target = _ConnectorWrite()
    try:
        await _run_turn(task_source="telegram", target=target)
    finally:
        unregister_mcp_approval_gate(handle)

    assert target.calls == [{"text": "publish"}]


@pytest.mark.asyncio
async def test_a_host_that_never_binds_a_source_is_not_gated() -> None:
    """Workforce, the builtin executor, agent previews, the agent-builder
    chat, triggers, and sub-agents of an unregistered or unbound parent
    deliberately bind no ``task_source``. They must keep dispatching, not
    fail closed, while a gate is live.

    Sub-agents of a *registered* parent source are a different case, not
    this one: ``AgentTool`` refuses to materialize governed MCP connectors
    for them instead (see ``_nested_mcp_refusal_reason`` in
    ``agent_tool.py`` and ``tests/core/tools/adapters/vibe/
    test_nested_agent_mcp_refusal.py``). This test exercises a bare
    ``AgentService.execute_task`` call with no bound execution context at
    all -- the shape workforce runs, the builtin executor, previews and
    triggers actually have -- not the delegated-child path."""

    async def gate(_call: GatedCall) -> GateDecision:
        raise AssertionError("an unbound source must not reach any hook")

    async def resume(**_: Any) -> None:
        raise AssertionError("resume must not run")

    handle = register_mcp_approval_gate(task_source="slack", gate=gate, resume=resume)
    target = _ConnectorWrite()
    (gated,) = gate_mcp_tools([target], connection={"id": 41})
    service = AgentService(
        name="workforce-agent",
        id="svc-unbound",
        tools=[gated],
        llm=_llm(),
        pattern="react",
    )
    try:
        await service.execute_task(
            "Publish the post.",
            context={"turn_id": "turn-1"},
            task_id="task-248032",
        )
    finally:
        unregister_mcp_approval_gate(handle)

    assert target.calls == [{"text": "publish"}]
