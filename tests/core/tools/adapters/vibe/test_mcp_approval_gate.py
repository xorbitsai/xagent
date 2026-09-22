from __future__ import annotations

import asyncio
import hashlib
import json
import logging
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
    _gated_interaction_id,
    _gated_interaction_source,
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
@pytest.mark.parametrize("task_source", [None, "", "web", "telegram"])
async def test_unregistered_or_absent_source_is_not_gated(
    registrations: list[Any], task_source: str | None
) -> None:
    """An active registration for one source must not touch any other source.

    This is the contract the fleet depends on: the wrapper sits on the single
    loader boundary shared by every entry point in the process, so a call whose
    own source has no registration dispatches exactly as the unwrapped tool
    would - including when the source is missing entirely.
    """

    _register(registrations, lambda _: None, _unused_resume)
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})
    other_source = replace(_context(), task_source=task_source)

    with bind_tool_call_execution_context(other_source):
        result = await tool.run_json_async({"text": "ungated"})
        sync_result = tool.run_json_sync({"text": "ungated sync"})

    assert result["success"] is True
    assert sync_result["success"] is True
    assert target.calls == [{"text": "ungated"}, {"text": "ungated sync"}]


@pytest.mark.asyncio
async def test_unregistered_source_passes_through_with_no_bound_context(
    registrations: list[Any],
) -> None:
    """Entry points that bind no context at all keep working unchanged."""

    _register(registrations, lambda _: None, _unused_resume)
    target = _Target(concurrency_safe=True)
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    result = await tool.run_json_async({"text": "read only call"})

    assert result["success"] is True
    assert target.calls == [{"text": "read only call"}]


@pytest.mark.asyncio
async def test_registered_source_still_fails_closed_on_malformed_identity(
    registrations: list[Any],
) -> None:
    """Fail-closed survives, scoped to a call whose OWN source is registered."""

    # An *allowing* hook that records being reached. If the identity guard
    # were removed, this would allow the call and the target would dispatch,
    # so the failure is observable rather than being laundered through the
    # wrapper's own except-Exception fail-closed path.
    reached: list[GatedCall] = []

    async def gate(call: GatedCall) -> GateDecision:
        reached.append(call)
        return GateDecision.allow()

    _register(registrations, gate, _unused_resume)
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})
    incomplete = replace(_context(), run_id=None)

    with bind_tool_call_execution_context(incomplete):
        result = await tool.run_json_async({"text": "must not publish"})
        with pytest.raises(RuntimeError, match="requires async approval"):
            tool.run_json_sync({"text": "must not publish"})

    assert result["status"] == "error"
    assert reached == []
    assert target.calls == []


def test_registered_gate_revokes_only_concurrency_metadata(
    registrations: list[Any],
) -> None:
    target = _Target(concurrency_safe=True)
    (tool,) = gate_mcp_tools([target], connection={"id": 41})
    assert tool.metadata.concurrency_safe is True
    assert tool.metadata.read_only is True

    _register(registrations, lambda _: None, _unused_resume)

    # concurrency_safe is the only field the ReAct scheduler reads, and a
    # gated tool can pause mid-batch, so it is revoked.
    assert tool.metadata.concurrency_safe is False
    # read_only has no scheduler consumer; forcing it would only mis-describe
    # the wrapped tool.
    assert tool.metadata.read_only is True
    assert target.metadata.concurrency_safe is True
    assert target.metadata.read_only is True


def test_metadata_passes_through_untouched_with_no_registrations() -> None:
    target = _Target(concurrency_safe=True)
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    assert tool.metadata is target.metadata


@pytest.mark.asyncio
async def test_matching_gate_requires_a_connector_identity(
    registrations: list[Any],
) -> None:
    reached: list[GatedCall] = []

    async def gate(call: GatedCall) -> GateDecision:
        reached.append(call)
        return GateDecision.allow()

    _register(registrations, gate, _unused_resume)
    target = _Target()
    # No positive integer id: the loader could not establish which connector
    # this tool belongs to, so no policy can be applied to it.
    (tool,) = gate_mcp_tools([target], connection={"transport": "oauth"})

    with bind_tool_call_execution_context(_context()):
        result = await tool.run_json_async({"text": "must not publish"})

    assert result["status"] == "error"
    assert reached == []
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
    # The published id carries the gating source durably, and the host's own
    # id is recoverable from it verbatim.
    assert result["interaction_id"] != "interaction-1"
    assert _gated_interaction_source(result["interaction_id"]) == (
        "slack",
        "interaction-1",
    )
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
async def test_pause_is_refused_before_the_hook_for_dag_execution(
    registrations: list[Any],
) -> None:
    """An unsupported pattern must be rejected BEFORE a decision is issued.

    If the hook ran first it would have already persisted and posted a real
    approval prompt (a Slack message the user can click) that the wrapper then
    discards, leaving an interaction nothing can ever resume.
    """

    reached: list[GatedCall] = []

    async def gate(call: GatedCall) -> GateDecision:
        reached.append(call)
        return GateDecision.require_approval("orphan")

    _register(registrations, gate, _unused_resume)
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    with bind_tool_call_execution_context(_context(pattern="dag")):
        result = await tool.run_json_async({"text": "must not publish"})

    assert result["status"] == "denied"
    assert reached == []
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
async def test_resume_hook_failures_never_dispatch(
    registrations: list[Any],
) -> None:
    async def gate(_: GatedCall) -> GateDecision:
        return GateDecision.require_approval("interaction-1")

    async def resume(**_: Any) -> Any:
        raise RuntimeError("ledger unavailable")

    _register(registrations, gate, resume)
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    with bind_tool_call_execution_context(_context()):
        result = await tool.resume_user_interaction(
            interaction_id=_gated_interaction_id("slack", "interaction-1"),
            response="approve",
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
            interaction_id=_gated_interaction_id("slack", "interaction-1"),
            response="approve",
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
            interaction_id=_gated_interaction_id("slack", "interaction-1"),
            response="approve",
        )

    assert result.status == "dispatch_unknown"
    assert target.calls == [{"text": "approved"}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resume_source", [None, "", "web"], ids=["missing", "empty", "other"]
)
async def test_gated_interaction_without_its_source_fails_closed(
    registrations: list[Any], resume_source: str | None
) -> None:
    """An approval that loses its binding must never replay the write.

    This is round 1's protection, kept exactly where it matters: the id proves
    this interaction WAS gated for 'slack', so resuming it from an execution
    that no longer presents 'slack' is refused rather than dispatched.
    """

    _register(registrations, lambda _: None, _unused_resume)
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})
    gated_id = _gated_interaction_id("slack", "interaction-1")

    with bind_tool_call_execution_context(
        replace(_context(), task_source=resume_source)
    ):
        settlement = await tool.resume_user_interaction(
            interaction_id=gated_id, response="approve"
        )

    assert settlement is not None
    assert settlement.status == "failed"
    assert settlement.projected_result()["status"] == "error"
    assert target.calls == []


@pytest.mark.asyncio
async def test_gated_interaction_fails_closed_after_its_gate_is_removed() -> None:
    """A gated approval outliving its registration must not be replayed."""

    handle = register_mcp_approval_gate(
        task_source="slack", gate=lambda _: None, resume=_unused_resume
    )
    gated_id = _gated_interaction_id("slack", "interaction-1")
    unregister_mcp_approval_gate(handle)

    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    with bind_tool_call_execution_context(_context()):
        settlement = await tool.resume_user_interaction(
            interaction_id=gated_id, response="approve"
        )

    assert settlement is not None
    assert settlement.status == "failed"
    assert target.calls == []


@pytest.mark.asyncio
async def test_never_gated_interaction_resumes_on_the_legacy_path(
    registrations: list[Any],
) -> None:
    """The gate returns None for an interaction it did not issue.

    ReAct reads None as "legacy replan", so unconditional exposure of
    resume_user_interaction is inert for an ungated tool.
    """

    class ResumableTarget(_Target):
        def __init__(self) -> None:
            super().__init__()
            self.resume_calls: list[dict[str, str]] = []

        async def resume_user_interaction(
            self, *, interaction_id: str, response: str
        ) -> None:
            self.resume_calls.append(
                {"interaction_id": interaction_id, "response": response}
            )
            return None

    _register(registrations, lambda _: None, _unused_resume)
    target = ResumableTarget()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    # Even bound to the REGISTERED source: what decides the path is whether
    # this wrapper gated this interaction, not whether a gate exists now.
    with bind_tool_call_execution_context(_context()):
        settlement = await tool.resume_user_interaction(
            interaction_id="host-owned-interaction", response="Continue"
        )

    assert settlement is None
    assert target.resume_calls == [
        {"interaction_id": "host-owned-interaction", "response": "Continue"}
    ]


@pytest.mark.asyncio
async def test_never_gated_interaction_without_a_target_callback_returns_none(
    registrations: list[Any],
) -> None:
    _register(registrations, lambda _: None, _unused_resume)
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    with bind_tool_call_execution_context(_context()):
        settlement = await tool.resume_user_interaction(
            interaction_id="host-owned-interaction", response="Continue"
        )

    assert settlement is None
    assert target.calls == []


@pytest.mark.parametrize(
    "task_source",
    ["slack", "a:b", "xgate", "3", "with:many:colons:and:digits:12"],
)
def test_gated_interaction_id_round_trips_any_registered_source(
    task_source: str,
) -> None:
    """The source is length-prefixed, so a separator inside it is safe."""

    stamped = _gated_interaction_id(task_source, "host:interaction:1")

    assert _gated_interaction_source(stamped) == (task_source, "host:interaction:1")


@pytest.mark.parametrize(
    "interaction_id",
    [
        "",
        "interaction-1",
        "xgate",
        "xgate:slack",
        "xgate:0:slack",
        "xgate:nope:slack",
        "xgate:99:slack",
        "xgate::slack",
        # Numeric-but-not-decimal: str.isdigit() accepts these, int() does not.
        # Parsing must reject them, never raise - this helper runs outside any
        # try in resume_user_interaction, and ReAct invokes that callback
        # outside its own rollback block, so a raise here would strand the
        # pending entry and wedge the task forever.
        "xgate:²:slackfoo",
        "xgate:⁵:slackfoo",
    ],
)
def test_ids_this_wrapper_did_not_issue_are_not_treated_as_gated(
    interaction_id: str,
) -> None:
    assert _gated_interaction_source(interaction_id) is None


@pytest.mark.asyncio
async def test_resume_hands_the_host_back_its_own_interaction_id(
    registrations: list[Any],
) -> None:
    seen: list[str] = []

    async def gate(_: GatedCall) -> GateDecision:
        return GateDecision.require_approval("host-1")

    async def resume(
        *, interaction_id: str, executor: Any, **_: Any
    ) -> ToolInteractionSettlement:
        seen.append(interaction_id)
        return ToolInteractionSettlement.succeeded(await executor({"text": "ok"}))

    _register(registrations, gate, resume)
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    with bind_tool_call_execution_context(_context()):
        paused = await tool.run_json_async({"text": "ok"})
        settlement = await tool.resume_user_interaction(
            interaction_id=paused["interaction_id"], response="approve"
        )

    assert settlement is not None and settlement.status == "succeeded"
    # The wrapper's stamp is an internal detail; the host only ever sees the
    # id it issued, on pause and on resume alike.
    assert seen == ["host-1"]
    assert target.replay_contexts[0].interaction_id == "host-1"


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_delay", [0, 0.001, 0.01])
async def test_external_cancellation_does_not_kill_an_in_flight_write(
    registrations: list[Any], cancel_delay: float
) -> None:
    """A cancelled resume must not abandon OR interrupt a dispatched write.

    The delay is the whole point. At ``cancel_delay=0`` the gate has not yet
    parked on its post-dispatch await, so the cancellation lands somewhere
    harmless and the ``finally`` drain runs. Give the loop even one extra turn
    and the gate is sitting in ``return await asyncio.shield(hook_task)`` -
    which is the realistic shape of a task cancel or lease loss arriving while
    the connector RPC is in flight. Unshielded, the cancel tore straight into
    the host hook inside ``await executor(...)``, leaving ``hook_task`` already
    done-and-cancelled by the time the ``finally`` ran, so the bounded drain
    and its warning were skipped entirely and the write died mid-RPC with no
    settlement and no log line.
    """

    started = asyncio.Event()
    landed = asyncio.Event()

    class SlowTarget(_Target):
        async def run_json_async(self, args: Mapping[str, Any]) -> Any:
            started.set()
            await asyncio.sleep(0.05)
            landed.set()
            return await super().run_json_async(args)

    async def resume(*, executor: Any, **_: Any) -> ToolInteractionSettlement:
        return ToolInteractionSettlement.succeeded(await executor({"text": "ok"}))

    _register(registrations, lambda _: None, resume, timeout_seconds=30)
    target = SlowTarget()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    async def run() -> Any:
        with bind_tool_call_execution_context(_context()):
            return await tool.resume_user_interaction(
                interaction_id=_gated_interaction_id("slack", "interaction-1"),
                response="approve",
            )

    task = asyncio.ensure_future(run())
    await started.wait()
    if cancel_delay:
        await asyncio.sleep(cancel_delay)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The drain observed the dispatched write to completion inside the
    # cancelled frame rather than killing it or leaving it unobserved.
    assert landed.is_set()
    assert target.calls == [{"text": "ok"}]


@pytest.mark.asyncio
async def test_cancellation_after_dispatch_warns_when_the_write_outlasts_the_drain(
    registrations: list[Any], monkeypatch: Any, caplog: Any
) -> None:
    """The bounded drain gives up loudly instead of hanging the cancellation."""

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_approval_gate."
        "_POST_DISPATCH_DRAIN_SECONDS",
        0.05,
    )
    started = asyncio.Event()

    class HangingTarget(_Target):
        async def run_json_async(self, args: Mapping[str, Any]) -> Any:
            started.set()
            await asyncio.sleep(30)
            return {"success": True}

    async def resume(*, executor: Any, **_: Any) -> ToolInteractionSettlement:
        return ToolInteractionSettlement.succeeded(await executor({"text": "ok"}))

    _register(registrations, lambda _: None, resume, timeout_seconds=30)
    target = HangingTarget()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    async def run() -> Any:
        with bind_tool_call_execution_context(_context()):
            return await tool.resume_user_interaction(
                interaction_id=_gated_interaction_id("slack", "interaction-1"),
                response="approve",
            )

    task = asyncio.ensure_future(run())
    await started.wait()
    await asyncio.sleep(0.01)
    with caplog.at_level(logging.WARNING):
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    # Cancellation completed (it was not blocked forever by the hung RPC)...
    assert task.cancelled()
    # ...and the unobservable write was reported rather than silently dropped.
    assert any("may still land" in record.message for record in caplog.records), [
        record.message for record in caplog.records
    ]


@pytest.mark.asyncio
async def test_cancellation_before_dispatch_still_cancels_the_hook_promptly(
    registrations: list[Any],
) -> None:
    """Shielding the post-dispatch await must not keep a policy hook alive.

    Guards the other half of the branch: pre-dispatch there is no external
    effect to preserve, so the hook is cancelled and drained rather than
    shielded, and the executor is never entered.
    """

    started = asyncio.Event()
    outcome: list[str] = []

    async def resume(*, executor: Any, **_: Any) -> ToolInteractionSettlement:
        started.set()
        try:
            # Policy work only: the executor is never awaited, so
            # dispatch_started stays clear.
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            outcome.append("hook-cancelled")
            raise
        outcome.append("hook-finished")
        return ToolInteractionSettlement.succeeded({"success": True})

    _register(registrations, lambda _: None, resume, timeout_seconds=30)
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    async def run() -> Any:
        with bind_tool_call_execution_context(_context()):
            return await tool.resume_user_interaction(
                interaction_id=_gated_interaction_id("slack", "interaction-1"),
                response="approve",
            )

    task = asyncio.ensure_future(run())
    await started.wait()
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert outcome == ["hook-cancelled"]
    assert target.calls == []


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
