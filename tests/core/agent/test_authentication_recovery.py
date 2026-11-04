"""Configured API authentication failures pause once and resume by replanning."""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from xagent.core.agent import (
    DAGPattern,
    ExecutionContext,
    ExecutionPlan,
    PatternRuntime,
    PlanStep,
    ReActPattern,
)
from xagent.core.tools.adapters.vibe.api_tool import APITool
from xagent.core.tools.adapters.vibe.api_tool_adapter import CustomApiTool


class ScriptedLLM:
    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)


def tool_call(name: str, call_id: str, **args: Any) -> dict[str, Any]:
    return {"id": call_id, "name": name, "arguments": args}


def connector(authorization: str) -> CustomApiTool:
    return CustomApiTool(
        name="records",
        description="Read records",
        env={},
        url="https://records.invalid",
        headers={"Authorization": authorization},
    )


def http_result(status: int) -> dict[str, Any]:
    return {
        "success": status == 200,
        "status_code": status,
        "headers": {},
        "body": {"records": []} if status == 200 else {"detail": "denied"},
        "error": None if status == 200 else f"HTTP {status}",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("resume_action", ["reauthorize", "change_scope", "use_export"])
async def test_401_pauses_cancels_fallback_and_replans_after_checkpoint(
    monkeypatch, resume_action
):
    request = AsyncMock(return_value=http_result(401))
    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.api_tool_adapter.call_api", request
    )
    api = connector("expired")
    fallback = APITool()
    fallback_request = AsyncMock(return_value=http_result(401))
    monkeypatch.setattr(fallback._client, "call_api", fallback_request)
    llm = ScriptedLLM(
        [
            {
                "tool_calls": [
                    tool_call(api.name, "original", params={"store": "A"}),
                    tool_call(api.name, "retry", params={"store": "A"}),
                    tool_call(fallback.name, "fallback", url="https://records.invalid"),
                ]
            }
        ]
    )
    pattern = ReActPattern(max_iterations=6)
    runtime = PatternRuntime()
    context = ExecutionContext()
    context.add_user_message("Read store A's records.")

    result = await pattern.run(
        context=context, tools=[api, fallback], llm=llm, runtime=runtime
    )

    assert result["status"] == "waiting_for_user"
    assert len(llm.calls) == 1
    request.assert_awaited_once()
    fallback_request.assert_not_awaited()
    assert pattern._record_for_tool_call_id("original").status == "waiting_for_user"
    assert pattern._record_for_tool_call_id("retry").status == "cancelled"
    assert pattern._record_for_tool_call_id("fallback").status == "cancelled"
    assert runtime.outbound_messages[-1]["expect_response"] is True
    assert "connection settings" in result["message"]
    checkpoint = next(
        c for c in reversed(runtime.checkpoints) if c["label"] == "waiting_for_user"
    )
    assert checkpoint["pattern_state"]["pending_tool_calls"] == []

    restored = ReActPattern(max_iterations=6)
    restored.load_state(checkpoint["pattern_state"])
    restored_context = ExecutionContext.from_dict(checkpoint["context"])
    # Restarting without a user reply must not replay the unauthorized call.
    parked_llm = ScriptedLLM([])
    parked = await restored.run(context=restored_context, tools=[api], llm=parked_llm)
    assert parked["status"] == "waiting_for_user"
    assert not parked_llm.calls
    request.assert_awaited_once()

    responses = [{"content": "No records.", "done": True}]
    request.reset_mock()
    request.return_value = http_result(200)
    if resume_action == "use_export":
        restored_context.add_user_message(
            "Use this complete empty export instead; do not call the API."
        )
    else:
        target = "B" if resume_action == "change_scope" else "A"
        restored_context.add_user_message(f"Reauthorized. Read store {target} now.")
        responses.insert(
            0, {"tool_calls": [tool_call(api.name, "fresh", params={"store": target})]}
        )
    resumed_llm = ScriptedLLM(responses)
    resumed = await restored.run(
        context=restored_context, tools=[connector("updated")], llm=resumed_llm
    )

    assert resumed["success"] is True
    if resume_action == "use_export":
        request.assert_not_awaited()
    else:
        request.assert_awaited_once()
        assert request.call_args.kwargs["params"] == {"store": target}
        assert request.call_args.kwargs["headers"] == {"Authorization": "updated"}
    fallback_request.assert_not_awaited()
    assert "Read store A's records." in str(resumed_llm.calls[0]["messages"])


@pytest.mark.asyncio
async def test_concurrent_authentication_pause_keeps_sibling_and_cancels_fallback(
    monkeypatch,
):
    sibling_started = asyncio.Event()

    class EmptyArgs(BaseModel):
        pass

    class SafeTool:
        def __init__(self, name):
            self.metadata = SimpleNamespace(
                name=name, description=name, concurrency_safe=True
            )

        def args_type(self):
            return EmptyArgs

        async def run_json_async(self, args):
            if self.metadata.name == "auth_read":
                # Neither call can finish this batch if dispatch is serial.
                await sibling_started.wait()
                return {
                    "success": False,
                    "is_error": True,
                    "failure_code": "authentication_required",
                    "error": "HTTP 401",
                }
            sibling_started.set()
            return {"success": True, "records": [1]}

    fallback = APITool()
    request = AsyncMock()
    monkeypatch.setattr(fallback._client, "call_api", request)
    llm = ScriptedLLM(
        [
            {
                "tool_calls": [
                    tool_call("auth_read", "auth"),
                    tool_call("other_read", "sibling"),
                    tool_call(fallback.name, "fallback", url="https://records.invalid"),
                ]
            }
        ]
    )
    pattern = ReActPattern(max_iterations=4, tool_parallel_enabled=True)
    runtime = PatternRuntime()
    context = ExecutionContext()
    context.add_user_message("Read both sources.")

    result = await asyncio.wait_for(
        pattern.run(
            context=context,
            tools=[SafeTool("auth_read"), SafeTool("other_read"), fallback],
            llm=llm,
            runtime=runtime,
        ),
        timeout=5,
    )

    assert result["status"] == "waiting_for_user"
    assert len(llm.calls) == 1
    assert any(c["label"] == "before_tool_batch" for c in runtime.checkpoints)
    assert pattern._record_for_tool_call_id("auth").status == "waiting_for_user"
    assert pattern._record_for_tool_call_id("sibling").status == "completed"
    assert pattern._record_for_tool_call_id("sibling").result["records"] == [1]
    assert pattern._record_for_tool_call_id("fallback").status == "cancelled"
    assert runtime.checkpoints[-1]["pattern_state"]["pending_tool_calls"] == []
    assert len(runtime.outbound_messages) == 1
    request.assert_not_awaited()


@pytest.mark.asyncio
async def test_401_headless_run_keeps_failure_available_for_partial_delivery(
    monkeypatch,
):
    request = AsyncMock(return_value=http_result(401))
    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.api_tool_adapter.call_api", request
    )
    api = connector("expired")
    llm = ScriptedLLM(
        [
            {"tool_calls": [tool_call(api.name, "original")]},
            {
                "content": "Available analysis is attached; records unavailable because authorization failed.",
                "done": True,
            },
        ]
    )
    context = ExecutionContext()
    context.add_user_message("Read records or explain what is missing.")
    pattern = ReActPattern(max_iterations=4, user_interaction_enabled=False)
    result = await pattern.run(context=context, tools=[api], llm=llm)

    assert result["success"] is True
    assert "authorization failed" in result["response"]
    assert pattern._record_for_tool_call_id("original").status == "failed"
    assert "authentication_required" in str(llm.calls[1]["messages"])


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,status", [("configured", 403), ("generic", 401)])
async def test_unclassified_http_failures_do_not_force_authentication_pause(
    monkeypatch, kind, status
):
    request = AsyncMock(return_value=http_result(status))
    if kind == "configured":
        api = connector("valid")
        monkeypatch.setattr(
            "xagent.core.tools.adapters.vibe.api_tool_adapter.call_api", request
        )
    else:
        api = APITool()
        monkeypatch.setattr(api._client, "call_api", request)
    llm = ScriptedLLM(
        [
            {
                "tool_calls": [
                    tool_call(api.name, "original", url="https://records.invalid")
                ]
            },
            {"content": "The requested resource is unavailable.", "done": True},
        ]
    )
    context = ExecutionContext()
    context.add_user_message("Read records.")
    result = await ReActPattern(max_iterations=4).run(
        context=context, tools=[api], llm=llm
    )

    assert result["success"] is True
    assert len(llm.calls) == 2
    request.assert_awaited_once()


@pytest.mark.asyncio
async def test_dag_child_authentication_pause_reaches_parent(monkeypatch):
    request = AsyncMock(return_value=http_result(401))
    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.api_tool_adapter.call_api", request
    )
    api = connector("expired")
    plan = ExecutionPlan(
        steps=[PlanStep(id="read", task="Read records", tool_names=[api.name])]
    )
    pattern = DAGPattern(lambda **_: plan)
    context = ExecutionContext()
    context.add_user_message("Read records and summarize.")
    llm = ScriptedLLM([{"tool_calls": [tool_call(api.name, "read-records")]}])

    result = await pattern.run(context=context, tools=[api], llm=llm)

    assert result["status"] == "waiting_for_user"
    assert pattern.active_step_pattern_states["read"]["status"] == "waiting_for_user"
    assert pattern.step_results == {}
    request.assert_awaited_once()
    assert len(llm.calls) == 1
