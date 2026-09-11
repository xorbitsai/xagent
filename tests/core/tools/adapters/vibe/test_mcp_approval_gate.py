from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest
from pydantic import BaseModel

from xagent.core.tools.adapters.vibe.base import AbstractBaseTool, ToolMetadata
from xagent.core.tools.adapters.vibe.mcp_approval_gate import (
    GatedCall,
    GateDecision,
    ToolCallExecutionContext,
    bind_tool_call_execution_context,
    current_mcp_approval_replay_context,
    gate_mcp_tools,
    register_mcp_approval_gate,
    unregister_mcp_approval_gate,
)
from xagent.core.tools.user_interaction import ToolInteractionSettlement


class _Args(BaseModel):
    text: str = ""


class _Target(AbstractBaseTool):
    def __init__(
        self, *, sandboxed: bool = False, concurrency_safe: bool = False
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.is_sandboxed = sandboxed
        self.replay_contexts: list[Any] = []
        self._metadata = ToolMetadata(
            name=self.name,
            concurrency_safe=concurrency_safe,
            read_only=concurrency_safe,
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
        return _Args

    def return_type(self) -> type[BaseModel]:
        return BaseModel

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        self.calls.append(dict(args))
        return {"success": True}

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        self.calls.append(dict(args))
        self.replay_contexts.append(current_mcp_approval_replay_context())
        return {"success": True, "arguments": dict(args)}


def _context(
    *, task_source: str | None = "slack", pattern: str = "react"
) -> ToolCallExecutionContext:
    return ToolCallExecutionContext(
        task_source=task_source,
        task_id="248032",
        run_id="run-1",
        turn_id="turn-1",
        tool_call_id="call-1",
        pattern=pattern,  # type: ignore[arg-type]
        react_step_id="react-1",
        dag_step_id="dag-1" if pattern == "dag" else None,
    )


@pytest.fixture
def registrations() -> list[Any]:
    handles: list[Any] = []
    yield handles
    for handle in reversed(handles):
        unregister_mcp_approval_gate(handle)


def _register(registrations: list[Any], gate: Any, resume: Any, **kwargs: Any) -> None:
    handle = register_mcp_approval_gate(
        task_source="slack", gate=gate, resume=resume, **kwargs
    )
    registrations.append(handle)


async def _unused_resume(**_: Any) -> None:
    raise AssertionError("resume hook should not run")


@pytest.mark.asyncio
async def test_no_matching_hook_preserves_legacy_execution() -> None:
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    result = await tool.run_json_async({"text": "publish"})

    assert result["success"] is True
    assert target.calls == [{"text": "publish"}]


@pytest.mark.asyncio
async def test_registration_only_applies_to_its_task_source(
    registrations: list[Any],
) -> None:
    async def gate(_: GatedCall) -> GateDecision:
        raise AssertionError("another task source must not reach this hook")

    _register(registrations, gate, _unused_resume)
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    with bind_tool_call_execution_context(_context(task_source="web")):
        result = await tool.run_json_async({"text": "web call"})

    assert result["success"] is True
    assert target.calls == [{"text": "web call"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("task_source", [None, ""])
async def test_missing_source_fails_closed_when_any_gate_is_registered(
    registrations: list[Any], task_source: str | None
) -> None:
    _register(registrations, lambda _: None, _unused_resume)
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})
    missing_source = replace(_context(), task_source=task_source)

    with bind_tool_call_execution_context(missing_source):
        result = await tool.run_json_async({"text": "must not publish"})
        with pytest.raises(RuntimeError, match="requires async approval"):
            tool.run_json_sync({"text": "must not publish"})

    assert result["status"] == "error"
    assert target.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("task_source", [None, ""])
async def test_missing_source_resume_fails_closed_when_any_gate_is_registered(
    registrations: list[Any], task_source: str | None
) -> None:
    _register(registrations, lambda _: None, _unused_resume)
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    with bind_tool_call_execution_context(replace(_context(), task_source=task_source)):
        settlement = await tool.resume_user_interaction(
            interaction_id="interaction-1", response="approve"
        )

    assert settlement is not None
    assert settlement.status == "failed"
    assert settlement.projected_result()["status"] == "error"
    assert target.calls == []


def test_registered_gate_revokes_concurrency_metadata(
    registrations: list[Any],
) -> None:
    target = _Target(concurrency_safe=True)
    (tool,) = gate_mcp_tools([target], connection={"id": 41})
    assert tool.metadata.concurrency_safe is True
    assert tool.metadata.read_only is True

    _register(registrations, lambda _: None, _unused_resume)

    assert tool.metadata.concurrency_safe is False
    assert tool.metadata.read_only is False
    assert target.metadata.concurrency_safe is True
    assert target.metadata.read_only is True


@pytest.mark.asyncio
async def test_matching_gate_requires_complete_execution_and_connector_identity(
    registrations: list[Any],
) -> None:
    async def gate(_: GatedCall) -> GateDecision:
        raise AssertionError("incomplete identity must fail before the hook")

    _register(registrations, gate, _unused_resume)
    incomplete = replace(_context(), run_id=None)

    for connection in ({"id": 41}, {"transport": "oauth"}):
        target = _Target()
        (tool,) = gate_mcp_tools([target], connection=connection)
        context = incomplete if connection.get("id") else _context()
        with bind_tool_call_execution_context(context):
            result = await tool.run_json_async({"text": "must not publish"})
        assert result["status"] == "error"
        assert target.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("sandboxed", [False, True], ids=["direct", "sandbox"])
async def test_approval_stops_both_transports_before_dispatch(
    registrations: list[Any], sandboxed: bool
) -> None:
    seen: list[GatedCall] = []

    async def gate(call: GatedCall) -> GateDecision:
        seen.append(call)
        return GateDecision.require_approval("interaction-1", message="Publish?")

    _register(registrations, gate, _unused_resume)
    target = _Target(sandboxed=sandboxed)
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    with bind_tool_call_execution_context(_context()):
        result = await tool.run_json_async({"text": "approved text"})

    assert result["status"] == "waiting_for_user"
    assert result["interaction_id"] == "interaction-1"
    assert target.calls == []
    assert seen[0].connector_ref.to_wire() == {
        "connector_type": "mcp",
        "connector_id": 41,
    }


@pytest.mark.asyncio
async def test_canonical_snapshot_is_deep_and_drives_allowed_dispatch(
    registrations: list[Any],
) -> None:
    original = {"text": "内容 A", "image": {"path": "/tmp/a.png"}}
    seen: list[GatedCall] = []

    async def gate(call: GatedCall) -> GateDecision:
        seen.append(call)
        original["image"]["path"] = "/tmp/changed.png"
        return GateDecision.allow()

    _register(registrations, gate, _unused_resume)
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    with bind_tool_call_execution_context(_context()):
        await tool.run_json_async(original)

    canonical = '{"image":{"path":"/tmp/a.png"},"text":"内容 A"}'
    assert seen[0].canonical_arguments_json == canonical
    assert (
        seen[0].arguments_sha256
        == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    )
    assert seen[0].arguments == json.loads(canonical)
    assert target.calls == [json.loads(canonical)]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exception", "timeout", "sync", "invalid"])
async def test_gate_failures_never_dispatch(
    registrations: list[Any], failure: str
) -> None:
    if failure == "exception":

        async def gate(_: GatedCall) -> GateDecision:
            raise RuntimeError("database unavailable")

    elif failure == "timeout":

        async def gate(_: GatedCall) -> GateDecision:
            await asyncio.sleep(1)
            return GateDecision.allow()

    elif failure == "sync":

        def gate(_: GatedCall) -> GateDecision:
            return GateDecision.allow()

    else:

        async def gate(_: GatedCall) -> Any:
            return "allow"

    _register(
        registrations,
        gate,
        _unused_resume,
        timeout_seconds=0.001 if failure == "timeout" else 1,
    )
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    with bind_tool_call_execution_context(_context()):
        result = await tool.run_json_async({"text": "must not publish"})

    assert result["status"] == "error"
    assert target.calls == []


@pytest.mark.asyncio
async def test_pause_is_refused_for_dag_execution(registrations: list[Any]) -> None:
    async def gate(_: GatedCall) -> GateDecision:
        return GateDecision.require_approval("orphan")

    _register(registrations, gate, _unused_resume)
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    with bind_tool_call_execution_context(_context(pattern="dag")):
        result = await tool.run_json_async({"text": "must not publish"})

    assert result["status"] == "denied"
    assert target.calls == []


@pytest.mark.asyncio
async def test_resume_uses_one_ephemeral_executor_for_host_payload(
    registrations: list[Any],
) -> None:
    frozen = {"text": "approved", "image": {"file_id": "file-1"}}
    second_error: list[str] = []

    async def gate(_: GatedCall) -> GateDecision:
        return GateDecision.require_approval("interaction-1")

    async def resume(*, executor: Any, **_: Any) -> ToolInteractionSettlement:
        result = await executor(frozen)
        with pytest.raises(RuntimeError, match="no longer available") as exc:
            await executor(frozen)
        second_error.append(str(exc.value))
        return ToolInteractionSettlement.succeeded(result)

    _register(registrations, gate, resume)
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    with bind_tool_call_execution_context(_context()):
        paused = await tool.run_json_async({"text": "approved"})
        result = await tool.resume_user_interaction(
            interaction_id=paused["interaction_id"], response="approve"
        )

    assert result.status == "succeeded"
    assert result.projected_result()["success"] is True
    assert target.calls == [frozen]
    replay = target.replay_contexts[0]
    assert replay.interaction_id == "interaction-1"
    assert replay.execution_context.tool_call_id == "call-1"
    assert current_mcp_approval_replay_context() is None
    assert second_error


@pytest.mark.asyncio
async def test_resume_hook_failures_never_dispatch(registrations: list[Any]) -> None:
    async def gate(_: GatedCall) -> GateDecision:
        return GateDecision.require_approval("interaction-1")

    async def resume(**_: Any) -> Any:
        raise RuntimeError("ledger unavailable")

    _register(registrations, gate, resume)
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    with bind_tool_call_execution_context(_context()):
        result = await tool.resume_user_interaction(
            interaction_id="interaction-1", response="approve"
        )

    assert result.status == "failed"
    assert result.projected_result()["status"] == "error"
    assert target.calls == []


@pytest.mark.asyncio
async def test_resume_timeout_does_not_cancel_a_started_dispatch(
    registrations: list[Any],
) -> None:
    completed = asyncio.Event()

    class SlowTarget(_Target):
        async def run_json_async(self, args: Mapping[str, Any]) -> Any:
            await asyncio.sleep(0.02)
            completed.set()
            return await super().run_json_async(args)

    async def resume(*, executor: Any, **_: Any) -> ToolInteractionSettlement:
        result = await executor({"text": "approved"})
        return ToolInteractionSettlement.succeeded(result)

    _register(registrations, lambda _: None, resume, timeout_seconds=0.001)
    target = SlowTarget()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    with bind_tool_call_execution_context(_context()):
        result = await tool.resume_user_interaction(
            interaction_id="interaction-1", response="approve"
        )

    assert completed.is_set()
    assert result.status == "succeeded"
    assert result.projected_result()["success"] is True
    assert target.calls == [{"text": "approved"}]


@pytest.mark.asyncio
async def test_unhandled_failure_after_dispatch_is_reported_as_unknown(
    registrations: list[Any],
) -> None:
    class FailingTarget(_Target):
        async def run_json_async(self, args: Mapping[str, Any]) -> Any:
            self.calls.append(dict(args))
            raise RuntimeError("connection lost after request write")

    async def resume(*, executor: Any, **_: Any) -> ToolInteractionSettlement:
        result = await executor({"text": "approved"})
        return ToolInteractionSettlement.succeeded(result)

    _register(registrations, lambda _: None, resume)
    target = FailingTarget()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    with bind_tool_call_execution_context(_context()):
        result = await tool.resume_user_interaction(
            interaction_id="interaction-1", response="approve"
        )

    assert result.status == "dispatch_unknown"
    assert target.calls == [{"text": "approved"}]


def test_registration_is_scoped_and_not_last_writer_wins(
    registrations: list[Any],
) -> None:
    registrations.append(
        register_mcp_approval_gate(
            task_source="slack", gate=lambda _: None, resume=lambda **_: None
        )
    )

    with pytest.raises(RuntimeError, match="already registered"):
        register_mcp_approval_gate(
            task_source="slack", gate=lambda _: None, resume=lambda **_: None
        )
