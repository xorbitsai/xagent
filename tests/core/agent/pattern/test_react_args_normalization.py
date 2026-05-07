import json
from typing import Any, Optional

import pytest
from pydantic import BaseModel, Field, field_validator

import xagent.core.agent.pattern.react as react_module
from xagent.core.agent.pattern.base import Action, Tool, ToolRegistry
from xagent.core.agent.pattern.react import ReActPattern
from xagent.core.tools.adapters.vibe.base import ToolMetadata


class MockUpdateArgs(BaseModel):
    agent_id: int
    tool_categories: Optional[list[str]] = Field(default=None)

    @field_validator("tool_categories", mode="before")
    @classmethod
    def parse_stringified_lists(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                pass
        return v


class MockUpdateAgentTool(Tool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(name="update_agent", description="desc")

    def args_type(self):
        return MockUpdateArgs

    def return_type(self):
        return dict

    def state_type(self):
        return None

    def is_async(self):
        return True

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        # Simulate failure for PUBLISHED agent
        if args.get("agent_id") == 100:
            return {
                "agent_id": 100,
                "status": "error",
                "message": "Error: Only DRAFT agents can be updated",
            }
        return {"agent_id": args.get("agent_id"), "status": "success", "message": "OK"}

    def run_json_sync(self, args: dict[str, Any]) -> Any:
        pass

    async def save_state_json(self):
        pass

    async def load_state_json(self, state: dict[str, Any]):
        pass

    def return_value_as_string(self, value: Any) -> str:
        return str(value)


class MockTracer:
    def __init__(self):
        self.traces = []

    async def trace_event(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        async def mock_method(*args, **kwargs):
            if name == "trace_tool_execution_start":
                self.traces.append(
                    {"event": "tool_execution_start", "args": args, "kwargs": kwargs}
                )
            elif name == "trace_action_end":
                self.traces.append(
                    {"event": "tool_execution_end", "args": args, "kwargs": kwargs}
                )

        return mock_method


@pytest.fixture(autouse=True)
def patch_traces(monkeypatch):
    tracer_instance = MockTracer()

    async def mock_trace_tool_execution_start(tracer, *args, **kwargs):
        tracer.traces.append(
            {"event": "tool_execution_start", "args": args, "kwargs": kwargs}
        )

    async def mock_trace_action_end(tracer, *args, **kwargs):
        tracer.traces.append(
            {"event": "tool_execution_end", "args": args, "kwargs": kwargs}
        )

    monkeypatch.setattr(
        react_module, "trace_tool_execution_start", mock_trace_tool_execution_start
    )
    monkeypatch.setattr(react_module, "trace_action_end", mock_trace_action_end)
    return tracer_instance


@pytest.mark.asyncio
async def test_react_tool_args_normalization(patch_traces):
    """Test that tool_args are normalized before tool execution and tracing."""
    registry = ToolRegistry()
    tool = MockUpdateAgentTool()
    registry.register(tool)

    tracer = patch_traces

    class MockReAct(ReActPattern):
        def __init__(self):
            self.tool_registry = registry
            self.tracer = tracer
            self._current_step_name = "main"

    pattern = MockReAct()

    # 1. Test success case with normalization
    action = Action(
        type="tool_call",
        reasoning="test",
        tool_name="update_agent",
        tool_args={
            "agent_id": 26,
            "tool_categories": '["vision"]',
        },
    )

    res = await pattern._execute_action(action, [], task_id="t1", step_id="s1")

    # Check that returned tool_args is normalized
    assert res["tool_args"]["tool_categories"] == ["vision"]

    # Check that trace got normalized args
    start_trace = next(t for t in tracer.traces if t["event"] == "tool_execution_start")
    assert start_trace["kwargs"]["data"]["tool_args"]["tool_categories"] == ["vision"]

    end_trace = next(
        t
        for t in tracer.traces
        if t["event"] == "tool_execution_end"
        and t["kwargs"]["data"].get("tool_name") == "update_agent"
    )
    assert end_trace["kwargs"]["data"]["tool_args"]["tool_categories"] == ["vision"]
    assert end_trace["kwargs"]["data"]["result"]["status"] == "success"


@pytest.mark.asyncio
async def test_react_tool_execution_failure_not_mutating_config(patch_traces):
    """Test that a failed tool execution (like updating PUBLISHED agent) is correctly traced as error."""
    registry = ToolRegistry()
    tool = MockUpdateAgentTool()
    registry.register(tool)

    tracer = patch_traces

    class MockReAct(ReActPattern):
        def __init__(self):
            self.tool_registry = registry
            self.tracer = tracer
            self._current_step_name = "main"

    pattern = MockReAct()

    # 2. Test failure case (e.g. published agent)
    action = Action(
        type="tool_call",
        reasoning="test published",
        tool_name="update_agent",
        tool_args={
            "agent_id": 100,
            "tool_categories": '["vision"]',
        },
    )

    res = await pattern._execute_action(action, [], task_id="t2", step_id="s2")

    # Tool args should still be normalized
    assert res["tool_args"]["tool_categories"] == ["vision"]

    # The result should contain the error status
    assert res["result"]["status"] == "error"

    # The frontend uses data.result.status === "success" to decide whether to update the config.
    # We verify that the trace contains the error status in the result.
    end_trace = next(
        t
        for t in tracer.traces
        if t["event"] == "tool_execution_end"
        and t["kwargs"]["data"].get("result", {}).get("agent_id") == 100
    )
    assert end_trace["kwargs"]["data"]["result"]["status"] == "error"
