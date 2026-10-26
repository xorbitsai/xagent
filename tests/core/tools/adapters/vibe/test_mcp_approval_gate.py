from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import replace
from threading import Event as ThreadEvent
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from xagent.core.tools.adapters.vibe import mcp_approval_gate as approval_gate_module
from xagent.core.tools.adapters.vibe.base import AbstractBaseTool, ToolMetadata
from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRef
from xagent.core.tools.adapters.vibe.interaction_types import TYPES_REQUIRING_OPTIONS
from xagent.core.tools.adapters.vibe.mcp_approval_gate import (
    GatedCall,
    GateDecision,
    MCPApprovalGateResolution,
    MCPApprovalGateScope,
    MCPApprovalGateTemporarilyUnavailable,
    MCPApprovalReplayExecutor,
    ToolCallExecutionContext,
    _gated_interaction_id,
    _gated_interaction_source,
    bind_tool_call_execution_context,
    current_mcp_approval_replay_context,
    gate_mcp_tools,
    mcp_approval_gate_registration_generation,
    mcp_approval_gate_scope,
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


def test_replay_executor_is_exported_from_vibe_facade() -> None:
    from xagent.core.tools.adapters.vibe import (
        MCPApprovalReplayExecutor as FacadeReplayExecutor,
    )

    assert FacadeReplayExecutor is MCPApprovalReplayExecutor


async def _unused_resume(**_: Any) -> None:
    raise AssertionError("resume hook should not run")


def _register_scoped(
    registrations: list[Any],
    *,
    gate: Any,
    resume: Any = _unused_resume,
    matcher: Any = lambda ref, name: (
        ref == ConnectorRef("mcp", 41) and name == "create_post"
    ),
    classifier: Any = lambda _: True,
) -> MCPApprovalGateScope:
    handle = register_mcp_approval_gate(
        task_source="external",
        gate=gate,
        resume=resume,
        tool_matcher=matcher,
        classifier=classifier,
        timeout_seconds=1,
    )
    registrations.append(handle)
    return mcp_approval_gate_scope(handle)


def _mcp_target(*, raw_name: str, concurrency_safe: bool = True) -> _Target:
    target = _Target(concurrency_safe=concurrency_safe)
    target.mcp_tool = SimpleNamespace(name=raw_name)
    return target


def test_task_resolution_has_strict_states_and_stable_signature() -> None:
    scope = MCPApprovalGateScope("external", "registration-1")

    assert MCPApprovalGateResolution.on(scope).signature_identity == (
        "on",
        "external",
        "registration-1",
        None,
    )
    assert MCPApprovalGateResolution.off(task_source="external").signature_identity == (
        "off",
        "external",
        None,
        None,
    )
    unknown = MCPApprovalGateResolution.unknown(
        task_source="external", reason="provider_unavailable"
    )
    assert unknown.signature_identity == (
        "unknown",
        "external",
        None,
        "provider_unavailable",
    )
    assert (
        unknown.approval_scope
        == approval_gate_module.mcp_approval_gate_rollback_fence(task_source="external")
    )

    with pytest.raises(ValueError, match="same-source scope"):
        MCPApprovalGateResolution(
            state="on",
            task_source="external",
            scope=MCPApprovalGateScope("other", "registration-1"),
        )
    with pytest.raises(ValueError, match="requires a non-empty trimmed reason"):
        MCPApprovalGateResolution.unknown(task_source="external", reason="")


def test_raw_name_lookup_never_invokes_dynamic_proxy_attributes() -> None:
    class DynamicProxy:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"dynamic attribute probe: {name}")

    assert approval_gate_module._raw_mcp_tool_name(DynamicProxy()) is None


def test_explicit_off_preserves_raw_tool_and_metadata(
    registrations: list[Any],
) -> None:
    _register_scoped(
        registrations,
        gate=lambda _: GateDecision.allow(),
    )
    target = _mcp_target(raw_name="create_post")

    (tool,) = gate_mcp_tools(
        [target],
        connector_ref=ConnectorRef("mcp", 41),
        approval_resolution=MCPApprovalGateResolution.off(task_source="external"),
    )

    assert tool is target
    assert tool.metadata is target.metadata


@pytest.mark.asyncio
async def test_scoped_matcher_uses_raw_name_and_non_target_passes_through(
    registrations: list[Any],
) -> None:
    matcher_calls: list[tuple[ConnectorRef, str]] = []
    classifier_calls: list[GatedCall] = []

    def matcher(ref: ConnectorRef, raw_name: str) -> bool:
        matcher_calls.append((ref, raw_name))
        return raw_name == "create_post"

    def classifier(call: GatedCall) -> bool:
        classifier_calls.append(call)
        return call.arguments["text"] == "publish"

    scope = _register_scoped(
        registrations,
        gate=lambda _: GateDecision.require_approval("approval-1"),
        matcher=matcher,
        classifier=classifier,
    )
    write = _mcp_target(raw_name="create_post")
    read = _mcp_target(raw_name="get_profile")
    write_tool, read_tool = gate_mcp_tools(
        [write, read],
        connector_ref=ConnectorRef("mcp", 41),
        approval_resolution=MCPApprovalGateResolution.on(scope),
    )

    assert read_tool is read
    with bind_tool_call_execution_context(
        _context(task_source="external", pattern="dag")
    ):
        passthrough = await write_tool.run_json_async({"text": "read"})
        refused = await write_tool.run_json_async({"text": "publish"})

    assert passthrough["success"] is True
    assert refused["status"] == "denied"
    assert matcher_calls == [
        (ConnectorRef("mcp", 41), "create_post"),
        (ConnectorRef("mcp", 41), "get_profile"),
    ]
    assert [call.raw_tool_name for call in classifier_calls] == [
        "create_post",
        "create_post",
    ]


def test_scoped_non_target_ignores_unrelated_legacy_registration(
    registrations: list[Any],
) -> None:
    scope = _register_scoped(
        registrations,
        gate=lambda _: GateDecision.allow(),
        matcher=lambda _ref, raw_name: raw_name == "create_post",
    )
    _register(registrations, lambda _: None, _unused_resume)
    target = _mcp_target(raw_name="get_profile", concurrency_safe=True)

    (tool,) = gate_mcp_tools(
        [target],
        connector_ref=ConnectorRef("mcp", 41),
        approval_resolution=MCPApprovalGateResolution.on(scope),
    )

    assert tool is target
    assert tool.metadata is target.metadata
    assert tool.metadata.concurrency_safe is True


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_result", [0, None, "yes"])
async def test_malformed_classifier_result_fails_closed_without_gate(
    registrations: list[Any], bad_result: Any
) -> None:
    gate_calls: list[GatedCall] = []

    async def gate(call: GatedCall) -> GateDecision:
        gate_calls.append(call)
        return GateDecision.allow()

    scope = _register_scoped(
        registrations,
        gate=gate,
        classifier=lambda _: bad_result,
    )
    target = _mcp_target(raw_name="create_post")
    (tool,) = gate_mcp_tools(
        [target],
        connector_ref=ConnectorRef("mcp", 41),
        approval_scope=scope,
    )

    with bind_tool_call_execution_context(_context(task_source="external")):
        result = await tool.run_json_async({"text": "publish"})

    assert result == approval_gate_module._GATE_FAILURE
    assert gate_calls == []
    assert target.calls == []


@pytest.mark.asyncio
async def test_unknown_resolution_fences_only_live_matcher_targets(
    registrations: list[Any],
) -> None:
    _register_scoped(
        registrations,
        gate=lambda _: GateDecision.allow(),
        matcher=lambda _ref, raw_name: raw_name == "create_post",
    )
    write = _mcp_target(raw_name="create_post")
    read = _mcp_target(raw_name="get_profile")
    write_tool, read_tool = gate_mcp_tools(
        [write, read],
        connector_ref=ConnectorRef("mcp", 41),
        approval_resolution=MCPApprovalGateResolution.unknown(
            task_source="external", reason="provider_timeout"
        ),
    )

    assert read_tool is read
    with bind_tool_call_execution_context(_context(task_source="external")):
        result = await write_tool.run_json_async({"text": "publish"})

    assert result == approval_gate_module._GATE_FAILURE
    assert write.calls == []


def test_unknown_without_live_matcher_never_becomes_all_tools_refusal() -> None:
    target = _mcp_target(raw_name="create_post")

    (tool,) = gate_mcp_tools(
        [target],
        connector_ref=ConnectorRef("mcp", 41),
        approval_resolution=MCPApprovalGateResolution.unknown(
            task_source="external", reason="registration_unavailable"
        ),
    )

    assert tool is target


@pytest.mark.asyncio
@pytest.mark.parametrize("authority_kind", ["stale", "forged"])
async def test_invalid_scope_uses_live_matcher_and_closes_only_target(
    registrations: list[Any], authority_kind: str
) -> None:
    live_scope = _register_scoped(
        registrations,
        gate=lambda _: GateDecision.allow(),
        matcher=lambda _ref, raw_name: raw_name == "create_post",
    )
    if authority_kind == "stale":
        assert unregister_mcp_approval_gate(registrations.pop()) is True
        stale_scope = live_scope
        _register_scoped(
            registrations,
            gate=lambda _: GateDecision.allow(),
            matcher=lambda _ref, raw_name: raw_name == "create_post",
        )
        invalid_scope = stale_scope
    else:
        invalid_scope = MCPApprovalGateScope("external", "forged-registration")
    write = _mcp_target(raw_name="create_post")
    read = _mcp_target(raw_name="get_profile")
    write_tool, read_tool = gate_mcp_tools(
        [write, read],
        connector_ref=ConnectorRef("mcp", 41),
        approval_scope=invalid_scope,
    )

    assert read_tool is read
    with bind_tool_call_execution_context(_context(task_source="external")):
        result = await write_tool.run_json_async({"text": "publish"})

    assert result == approval_gate_module._GATE_FAILURE
    assert write.calls == []


@pytest.mark.asyncio
async def test_stale_explicit_scope_without_live_matcher_fails_closed() -> None:
    write = _mcp_target(raw_name="create_post")
    read = _mcp_target(raw_name="get_profile")
    tools = gate_mcp_tools(
        [write, read],
        connector_ref=ConnectorRef("mcp", 41),
        approval_scope=MCPApprovalGateScope("external", "stale-registration"),
    )

    with bind_tool_call_execution_context(_context(task_source="external")):
        results = [await tool.run_json_async({"text": "publish"}) for tool in tools]

    assert results == [approval_gate_module._GATE_FAILURE] * 2
    assert write.calls == []
    assert read.calls == []


@pytest.mark.asyncio
async def test_approved_resume_waits_through_fence_then_recovers(
    registrations: list[Any],
) -> None:
    approved = {"text": "approved"}

    async def gate(_: GatedCall) -> GateDecision:
        return GateDecision.require_approval("interaction-1")

    async def resume(
        *, executor: MCPApprovalReplayExecutor, **_: Any
    ) -> ToolInteractionSettlement:
        executor.acknowledge_dispatch()
        return ToolInteractionSettlement.succeeded(await executor(approved))

    scope = _register_scoped(registrations, gate=gate, resume=resume)
    target = _mcp_target(raw_name="create_post")
    (on_tool,) = gate_mcp_tools(
        [target],
        connector_ref=ConnectorRef("mcp", 41),
        approval_resolution=MCPApprovalGateResolution.on(scope),
    )
    with bind_tool_call_execution_context(_context(task_source="external")):
        paused = await on_tool.run_json_async(approved)

    (fenced_tool,) = gate_mcp_tools(
        [target],
        connector_ref=ConnectorRef("mcp", 41),
        approval_resolution=MCPApprovalGateResolution.unknown(
            task_source="external", reason="provider_timeout"
        ),
    )
    with bind_tool_call_execution_context(_context(task_source="external")):
        with pytest.raises(MCPApprovalGateTemporarilyUnavailable):
            await fenced_tool.resume_user_interaction(
                interaction_id=paused["interaction_id"], response="approve"
            )
    assert target.calls == []

    (recovered_tool,) = gate_mcp_tools(
        [target],
        connector_ref=ConnectorRef("mcp", 41),
        approval_resolution=MCPApprovalGateResolution.on(scope),
    )
    with bind_tool_call_execution_context(_context(task_source="external")):
        settlement = await recovered_tool.resume_user_interaction(
            interaction_id=paused["interaction_id"], response="approve"
        )

    assert settlement is not None and settlement.status == "succeeded"
    assert target.calls == [approved]


def test_public_facade_exports_resolution_contract() -> None:
    from xagent.mcp_approval_gate import MCPApprovalGateResolution as PublicResolution
    from xagent.mcp_approval_gate import MCPApprovalGateScope as PublicScope

    assert PublicResolution is MCPApprovalGateResolution
    assert PublicScope is MCPApprovalGateScope


@pytest.mark.asyncio
async def test_no_matching_hook_preserves_legacy_execution() -> None:
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    result = await tool.run_json_async({"text": "publish"})

    assert result["success"] is True
    assert target.calls == [{"text": "publish"}]


@pytest.mark.asyncio
async def test_unscoped_tool_observes_late_legacy_registration(
    registrations: list[Any],
) -> None:
    _register_scoped(
        registrations,
        gate=lambda _: GateDecision.allow(),
    )
    target = _mcp_target(raw_name="create_post")
    (tool,) = gate_mcp_tools([target], connector_ref=ConnectorRef("mcp", 41))

    async def deny(_: GatedCall) -> GateDecision:
        return GateDecision.deny(message="late legacy policy")

    _register(registrations, deny, _unused_resume)
    with bind_tool_call_execution_context(_context()):
        result = await tool.run_json_async({"text": "must not publish"})

    assert result["status"] == "denied"
    assert target.calls == []


@pytest.mark.asyncio
async def test_legacy_path_preserves_approval_unsupported_capability(
    registrations: list[Any],
) -> None:
    reached: list[GatedCall] = []

    async def allow(call: GatedCall) -> GateDecision:
        reached.append(call)
        return GateDecision.allow()

    _register(registrations, allow, _unused_resume)
    target = _mcp_target(raw_name="create_post")
    (tool,) = gate_mcp_tools(
        [target],
        connector_ref=ConnectorRef("mcp", 41),
        approval_supported=False,
    )

    with bind_tool_call_execution_context(_context()):
        result = await tool.run_json_async({"text": "must not publish"})

    assert result == approval_gate_module._NESTED_APPROVAL_UNAVAILABLE
    assert reached == []
    assert target.calls == []


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

    # Value equality, not identity: a real tool builds a fresh ToolMetadata on
    # every access (AbstractBaseTool.metadata), so an identity assertion passes
    # here only because this fake happens to cache one.
    assert tool.metadata == target.metadata


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
    # A `confirm` must carry NO options: it is not in TYPES_REQUIRING_OPTIONS,
    # the write-side validator refuses options on it (`options_forbidden`), and
    # the renderer draws it as a boolean switch that ignores them.
    assert result["interactions"] == [
        {"type": "confirm", "field": "approve", "label": "Approve"}
    ]
    assert all(
        interaction["type"] in TYPES_REQUIRING_OPTIONS or "options" not in interaction
        for interaction in result["interactions"]
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
async def test_replay_executor_ack_is_thread_safe_idempotent_and_expires(
    registrations: list[Any],
) -> None:
    captured: list[MCPApprovalReplayExecutor] = []

    async def resume(
        *, executor: MCPApprovalReplayExecutor, **_: Any
    ) -> ToolInteractionSettlement:
        captured.append(executor)

        def acknowledge_twice() -> None:
            executor.acknowledge_dispatch()
            executor.acknowledge_dispatch()

        await asyncio.to_thread(acknowledge_twice)
        return ToolInteractionSettlement.succeeded(await executor({"text": "approved"}))

    _register(registrations, lambda _: None, resume)
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    with bind_tool_call_execution_context(_context()):
        settlement = await tool.resume_user_interaction(
            interaction_id=_gated_interaction_id("slack", "interaction-1"),
            response="approve",
        )

    assert settlement is not None and settlement.status == "succeeded"
    assert target.calls == [{"text": "approved"}]
    with pytest.raises(RuntimeError, match="no longer available"):
        captured[0].acknowledge_dispatch()
    with pytest.raises(RuntimeError, match="no longer available"):
        await captured[0]({"text": "second"})


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
async def test_resume_timeout_before_durable_ack_is_failed(
    registrations: list[Any],
) -> None:
    async def resume(**_: Any) -> ToolInteractionSettlement:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    _register(registrations, lambda _: None, resume, timeout_seconds=0.01)
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    with bind_tool_call_execution_context(_context()):
        settlement = await tool.resume_user_interaction(
            interaction_id=_gated_interaction_id("slack", "interaction-1"),
            response="approve",
        )

    assert settlement is not None and settlement.status == "failed"
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
async def test_durable_ack_timeout_before_executor_is_dispatch_unknown(
    registrations: list[Any],
) -> None:
    """Cancellation cleanup can acknowledge a marker committed off-loop.

    The worker is deliberately non-abandonable: the policy timeout cancels the
    hook while its commit is still blocked, then the commit wins before the
    cancelled coroutine can return normally and perform its normal-path
    acknowledgement. The cleanup reads that durable fact and acknowledges it;
    classification must happen after the cleanup drain, not before it.
    """

    commit_started = ThreadEvent()
    release_commit = ThreadEvent()
    marker_committed = ThreadEvent()
    cancellation_cleanup_started = asyncio.Event()
    ledger = {"status": "approved"}

    def commit_marker() -> None:
        commit_started.set()
        release_commit.wait()
        ledger["status"] = "executing"
        marker_committed.set()

    async def resume(
        *, executor: MCPApprovalReplayExecutor, **_: Any
    ) -> ToolInteractionSettlement:
        commit_task = asyncio.create_task(asyncio.to_thread(commit_marker))
        try:
            await asyncio.shield(commit_task)
            executor.acknowledge_dispatch()
        except asyncio.CancelledError:
            cancellation_cleanup_started.set()
            await asyncio.shield(commit_task)
            assert marker_committed.is_set()
            executor.acknowledge_dispatch()
            ledger["status"] = "dispatch_unknown"
            raise
        return ToolInteractionSettlement.failed()

    _register(
        registrations,
        lambda _: None,
        resume,
        timeout_seconds=0.01,
        dispatch_observe_seconds=0.02,
    )
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    async def release_after_timeout() -> None:
        await cancellation_cleanup_started.wait()
        release_commit.set()

    releaser = asyncio.create_task(release_after_timeout())
    with bind_tool_call_execution_context(_context()):
        settlement = await asyncio.wait_for(
            tool.resume_user_interaction(
                interaction_id=_gated_interaction_id("slack", "interaction-1"),
                response="approve",
            ),
            timeout=5,
        )
    await releaser

    assert commit_started.is_set() and marker_committed.is_set()
    assert settlement is not None and settlement.status == "dispatch_unknown"
    assert ledger["status"] == "dispatch_unknown"
    assert target.calls == []


@pytest.mark.asyncio
async def test_durable_ack_cancellation_before_executor_cannot_reopen_retry(
    registrations: list[Any],
) -> None:
    """A late marker acknowledged in cancel cleanup replays, not dispatches."""

    commit_started = ThreadEvent()
    release_commit = ThreadEvent()
    marker_committed = ThreadEvent()
    cancellation_cleanup_started = asyncio.Event()
    ledger = {"status": "approved"}

    def commit_marker() -> None:
        commit_started.set()
        release_commit.wait()
        ledger["status"] = "executing"
        marker_committed.set()

    async def resume(
        *, executor: MCPApprovalReplayExecutor, **_: Any
    ) -> ToolInteractionSettlement:
        if ledger["status"] == "dispatch_unknown":
            return ToolInteractionSettlement.dispatch_unknown()
        commit_task = asyncio.create_task(asyncio.to_thread(commit_marker))
        try:
            await asyncio.shield(commit_task)
            executor.acknowledge_dispatch()
        except asyncio.CancelledError:
            cancellation_cleanup_started.set()
            await asyncio.shield(commit_task)
            assert marker_committed.is_set()
            executor.acknowledge_dispatch()
            ledger["status"] = "dispatch_unknown"
            raise
        return ToolInteractionSettlement.failed()

    _register(
        registrations,
        lambda _: None,
        resume,
        timeout_seconds=30,
        dispatch_observe_seconds=0.02,
    )
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    async def run() -> ToolInteractionSettlement | None:
        with bind_tool_call_execution_context(_context()):
            return await tool.resume_user_interaction(
                interaction_id=_gated_interaction_id("slack", "interaction-1"),
                response="approve",
            )

    task = asyncio.create_task(run())
    await asyncio.to_thread(commit_started.wait)
    task.cancel()
    await cancellation_cleanup_started.wait()
    release_commit.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)

    replayed = await run()
    assert replayed is not None and replayed.status == "dispatch_unknown"
    assert marker_committed.is_set()
    assert ledger["status"] == "dispatch_unknown"
    assert target.calls == []


@pytest.mark.asyncio
async def test_durable_ack_is_scoped_to_one_concurrent_resume(
    registrations: list[Any],
) -> None:
    """One interaction's marker must not change another's retry class."""

    first_acknowledged = asyncio.Event()

    async def resume(
        *, interaction_id: str, executor: MCPApprovalReplayExecutor, **_: Any
    ) -> ToolInteractionSettlement:
        if interaction_id == "first":
            executor.acknowledge_dispatch()
            first_acknowledged.set()
            await asyncio.Event().wait()
        await first_acknowledged.wait()
        raise RuntimeError("failed before this interaction's marker")

    _register(
        registrations,
        lambda _: None,
        resume,
        timeout_seconds=0.01,
        dispatch_observe_seconds=0.02,
    )
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    async def run(interaction_id: str) -> ToolInteractionSettlement | None:
        with bind_tool_call_execution_context(_context()):
            return await tool.resume_user_interaction(
                interaction_id=_gated_interaction_id("slack", interaction_id),
                response="approve",
            )

    first, second = await asyncio.gather(run("first"), run("second"))

    assert first is not None and first.status == "dispatch_unknown"
    assert second is not None and second.status == "failed"
    assert target.calls == []


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
async def test_cancellation_after_dispatch_reclaims_a_write_that_outlasts_the_drain(
    registrations: list[Any], monkeypatch: Any, caplog: Any
) -> None:
    """The bounded drain gives up loudly, then reclaims the connector call.

    Giving up on *observing* the write must not mean giving up on its socket
    and (for stdio) its child process, so the hook task is cancelled rather
    than left detached.
    """

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_approval_gate._DISPATCH_OBSERVE_SECONDS",
        0.05,
    )
    started = asyncio.Event()

    torn_down = asyncio.Event()

    class HangingTarget(_Target):
        async def run_json_async(self, args: Mapping[str, Any]) -> Any:
            started.set()
            try:
                await asyncio.sleep(30)
            finally:
                torn_down.set()
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
    # ...the unobservable write was reported rather than silently dropped...
    assert any(
        "may still have landed" in record.message for record in caplog.records
    ), [record.message for record in caplog.records]
    # ...and its transport was actually unwound rather than left detached.
    await asyncio.wait_for(torn_down.wait(), timeout=5)


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


@pytest.mark.asyncio
async def test_hung_connector_settles_as_dispatch_unknown(
    registrations: list[Any], monkeypatch: Any
) -> None:
    """A connector that never answers must not park the resume forever.

    The registration deadline is already spent by the time dispatch starts, so
    without a second bound the success path waits on the RPC indefinitely.
    """

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_approval_gate._DISPATCH_OBSERVE_SECONDS",
        0.05,
    )
    started = asyncio.Event()

    class HungTarget(_Target):
        async def run_json_async(self, args: Mapping[str, Any]) -> Any:
            started.set()
            await asyncio.sleep(30)
            return {"success": True}

    async def resume(*, executor: Any, **_: Any) -> ToolInteractionSettlement:
        return ToolInteractionSettlement.succeeded(await executor({"text": "ok"}))

    _register(registrations, lambda _: None, resume, timeout_seconds=30)
    target = HungTarget()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    with bind_tool_call_execution_context(_context()):
        settlement = await asyncio.wait_for(
            tool.resume_user_interaction(
                interaction_id=_gated_interaction_id("slack", "interaction-1"),
                response="approve",
            ),
            timeout=5,
        )

    assert settlement is not None
    # Never "failed": the write may have reached the external system.
    assert settlement.status == "dispatch_unknown"
    assert started.is_set()


@pytest.mark.asyncio
async def test_unobserved_dispatch_is_cancelled_and_its_transport_torn_down(
    registrations: list[Any], monkeypatch: Any
) -> None:
    """Bounded observation must also bound the connector's resource lifetime.

    Only the ``shield`` wrapper is cancelled when the observation budget
    expires, so the real hook task keeps running unless the gate cancels it:
    its MCP session, socket and (for stdio, which has no read deadline of its
    own) child process would stay alive for as long as the remote stays
    silent, once per approval. The ``finally`` in the target below stands in
    for ``async with create_session(...)`` unwinding.
    """

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_approval_gate._DISPATCH_OBSERVE_SECONDS",
        0.05,
    )
    torn_down = asyncio.Event()

    class HungTarget(_Target):
        async def run_json_async(self, args: Mapping[str, Any]) -> Any:
            try:
                await asyncio.Event().wait()
            finally:
                torn_down.set()
            return {"success": True}

    async def resume(*, executor: Any, **_: Any) -> ToolInteractionSettlement:
        return ToolInteractionSettlement.succeeded(await executor({"text": "ok"}))

    _register(registrations, lambda _: None, resume, timeout_seconds=30)
    (tool,) = gate_mcp_tools([HungTarget()], connection={"id": 41})

    before = {task for task in asyncio.all_tasks()}
    with bind_tool_call_execution_context(_context()):
        settlement = await asyncio.wait_for(
            tool.resume_user_interaction(
                interaction_id=_gated_interaction_id("slack", "interaction-1"),
                response="approve",
            ),
            timeout=5,
        )

    # The caller's contract is unchanged: the write may have landed.
    assert settlement is not None
    assert settlement.status == "dispatch_unknown"

    # ...and the connector call is actually unwound, not merely logged.
    await asyncio.wait_for(torn_down.wait(), timeout=5)
    for _ in range(50):
        leaked = {
            task
            for task in asyncio.all_tasks()
            if task not in before and task is not asyncio.current_task()
        }
        if not leaked:
            break
        await asyncio.sleep(0.01)
    assert not leaked, f"dispatch left {len(leaked)} task(s) running"


@pytest.mark.asyncio
async def test_dispatch_observe_seconds_overrides_the_module_default(
    registrations: list[Any], monkeypatch: Any
) -> None:
    """A registration's own budget is honored even when it differs from the
    module default - proves the value actually threads through to
    ``_call_resume_hook``, not just validated at registration and discarded.
    """

    # Pin the module default far above the test's patience. The assertions
    # below can only pass if the per-registration override - not this
    # constant - is what actually bounds the observation wait.
    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_approval_gate._DISPATCH_OBSERVE_SECONDS",
        999.0,
    )
    torn_down = asyncio.Event()

    class HungTarget(_Target):
        async def run_json_async(self, args: Mapping[str, Any]) -> Any:
            try:
                await asyncio.Event().wait()
            finally:
                torn_down.set()
            return {"success": True}

    async def resume(*, executor: Any, **_: Any) -> ToolInteractionSettlement:
        return ToolInteractionSettlement.succeeded(await executor({"text": "ok"}))

    _register(
        registrations,
        lambda _: None,
        resume,
        timeout_seconds=30,
        dispatch_observe_seconds=0.05,
    )
    (tool,) = gate_mcp_tools([HungTarget()], connection={"id": 41})

    with bind_tool_call_execution_context(_context()):
        settlement = await asyncio.wait_for(
            tool.resume_user_interaction(
                interaction_id=_gated_interaction_id("slack", "interaction-1"),
                response="approve",
            ),
            timeout=5,
        )

    assert settlement is not None
    assert settlement.status == "dispatch_unknown"
    await asyncio.wait_for(torn_down.wait(), timeout=5)


@pytest.mark.asyncio
async def test_cancellation_mid_observation_shares_the_deadline_not_restarts_it(
    registrations: list[Any], monkeypatch: Any
) -> None:
    """Cancellation mid-observation must not re-arm a fresh full budget.

    Regression test for the #2582 review finding on the cancellation branch:
    previously, only the ``TimeoutError`` raised by the first post-dispatch
    ``wait_for`` set ``observed = True``. A cancellation that instead
    interrupted that same wait bypassed it, so the ``finally`` drain re-armed
    a brand-new ``dispatch_observe_seconds`` window regardless of how much of
    the first one had already elapsed - roughly doubling worst-case
    dispatch-to-cancel-completion latency. The fix records a monotonic
    deadline once and shares it between both waits.

    The test asserts the budget the ``finally`` drain is given, not how long
    the cancel takes to complete: the latter would also measure the load of
    the machine running the test.
    """

    dispatch_observe_seconds = 0.4
    started = asyncio.Event()

    # Record the budget each drain is given instead of timing how long the
    # drains take: elapsed time measures the runner's load as much as the
    # budget, while the budget itself is computed from the shared deadline and
    # can only shrink when the runner is slow.
    drain_timeouts: list[float | None] = []
    real_drain = approval_gate_module._drain

    async def recording_drain(awaitable: Any, *, timeout: float | None = None) -> None:
        drain_timeouts.append(timeout)
        await real_drain(awaitable, timeout=timeout)

    monkeypatch.setattr(approval_gate_module, "_drain", recording_drain)

    class HangingTarget(_Target):
        async def run_json_async(self, args: Mapping[str, Any]) -> Any:
            started.set()
            await asyncio.Event().wait()  # never completes on its own
            return {"success": True}

    async def resume(*, executor: Any, **_: Any) -> ToolInteractionSettlement:
        return ToolInteractionSettlement.succeeded(await executor({"text": "ok"}))

    _register(
        registrations,
        lambda _: None,
        resume,
        timeout_seconds=30,
        dispatch_observe_seconds=dispatch_observe_seconds,
    )
    (tool,) = gate_mcp_tools([HangingTarget()], connection={"id": 41})

    async def run() -> Any:
        with bind_tool_call_execution_context(_context()):
            return await tool.resume_user_interaction(
                interaction_id=_gated_interaction_id("slack", "interaction-1"),
                response="approve",
            )

    task = asyncio.ensure_future(run())
    await started.wait()
    # Let the gate arm its observation deadline before the half-budget sleep
    # starts (it needs 2 loop iterations; 3 keep one spare), so a stall can
    # only shrink the drain budget.
    for _ in range(3):
        await asyncio.sleep(0)
    # Let the observation wait consume roughly half its budget before the
    # external cancellation arrives - the shape that used to restart it.
    await asyncio.sleep(dispatch_observe_seconds / 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)

    # The first drain is the one `finally` runs on the dispatched call under
    # the observation budget; any later one is the unobserved-call cleanup,
    # which has its own budget.
    assert drain_timeouts, "the dispatched call was never drained"
    observation_drain_timeout = drain_timeouts[0]
    assert observation_drain_timeout is not None
    # The fix gives it only the REMAINING half of the shared budget (at most
    # ~0.2s here; less the slower the runner is). The bug re-armed a fresh
    # full budget in `finally` (0.4s), so a threshold at 75% of the full
    # budget separates the two, and a slow runner can only push the fixed
    # behaviour further below it.
    assert observation_drain_timeout < dispatch_observe_seconds * 0.75, (
        observation_drain_timeout
    )


def test_wrapper_delegates_unknown_attributes_to_the_wrapped_tool() -> None:
    """Call sites duck-type tools with getattr, so losses would be silent."""

    target = _Target()
    target.source_server = "notion"  # type: ignore[attr-defined]
    target.category = "mcp"  # type: ignore[attr-defined]
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    assert tool.source_server == "notion"
    assert tool.category == "mcp"
    assert tool.target is target
    # Wrapper-defined members keep priority over the delegation.
    assert tool.name == target.name
    assert tool.is_async() is True
    # A private name must not be delegated, or a missing internal would be
    # masked instead of raising.
    with pytest.raises(AttributeError):
        tool._nonexistent_internal  # noqa: B018


def test_wrapper_category_is_none_when_the_target_has_none() -> None:
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    assert tool.category is None


@pytest.mark.asyncio
async def test_denied_decision_returns_the_documented_shape(
    registrations: list[Any],
) -> None:
    async def gate(_: GatedCall) -> GateDecision:
        return GateDecision.deny(message="Connector is out of policy.")

    _register(registrations, gate, _unused_resume)
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    with bind_tool_call_execution_context(_context()):
        result = await tool.run_json_async({"text": "must not publish"})

    assert result == {
        "success": False,
        "status": "denied",
        "error": "Connector is out of policy.",
    }
    assert target.calls == []


@pytest.mark.asyncio
async def test_denied_decision_falls_back_to_a_default_message(
    registrations: list[Any],
) -> None:
    async def gate(_: GatedCall) -> GateDecision:
        return GateDecision.deny()

    _register(registrations, gate, _unused_resume)
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    with bind_tool_call_execution_context(_context()):
        result = await tool.run_json_async({"text": "must not publish"})

    assert result["status"] == "denied"
    assert result["success"] is False
    assert result["error"] == "The connector call was denied."


@pytest.mark.asyncio
async def test_allowed_call_dispatches_the_snapshot_the_hook_approved(
    registrations: list[Any],
) -> None:
    """A hook that mutates the caller's dict cannot change what is dispatched.

    The canonical snapshot is what ``arguments_sha256`` covers, so dispatching
    it - rather than the caller's live object - is what makes "the approved
    call is the executed call" hold on the allow path.
    """

    original: dict[str, Any] = {"text": "approved", "target": {"id": "safe"}}

    async def gate(_: GatedCall) -> GateDecision:
        # A hook that decides on one payload and then rewrites the caller's
        # object must not be able to redirect the dispatch.
        original["target"]["id"] = "attacker-controlled"
        return GateDecision.allow()

    _register(registrations, gate, _unused_resume)
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    with bind_tool_call_execution_context(_context()):
        await tool.run_json_async(original)

    assert target.calls == [{"text": "approved", "target": {"id": "safe"}}]


@pytest.mark.parametrize(
    "task_source",
    [None, 1, "", "  ", " slack", "slack "],
    ids=["none", "int", "empty", "blank", "leading-ws", "trailing-ws"],
)
def test_register_refuses_an_unusable_task_source(task_source: Any) -> None:
    with pytest.raises(ValueError):
        register_mcp_approval_gate(
            task_source=task_source, gate=lambda _: None, resume=lambda **_: None
        )


@pytest.mark.parametrize("bad", ["gate", "resume"])
def test_register_refuses_a_non_callable_hook(bad: str) -> None:
    hooks: dict[str, Any] = {"gate": lambda _: None, "resume": lambda **_: None}
    hooks[bad] = "not callable"

    with pytest.raises(TypeError):
        register_mcp_approval_gate(task_source="slack", **hooks)


@pytest.mark.parametrize(
    "timeout_seconds",
    [0, -1, float("nan"), float("inf")],
    ids=["zero", "neg", "nan", "inf"],
)
def test_register_refuses_an_unusable_timeout(timeout_seconds: float) -> None:
    with pytest.raises(ValueError):
        register_mcp_approval_gate(
            task_source="slack",
            gate=lambda _: None,
            resume=lambda **_: None,
            timeout_seconds=timeout_seconds,
        )


@pytest.mark.parametrize(
    "dispatch_observe_seconds",
    [0, -1, float("nan"), float("inf")],
    ids=["zero", "neg", "nan", "inf"],
)
def test_register_refuses_an_unusable_dispatch_observe_seconds(
    dispatch_observe_seconds: float,
) -> None:
    with pytest.raises(ValueError):
        register_mcp_approval_gate(
            task_source="slack",
            gate=lambda _: None,
            resume=lambda **_: None,
            dispatch_observe_seconds=dispatch_observe_seconds,
        )


def test_unregister_refuses_a_stale_handle() -> None:
    handle = register_mcp_approval_gate(
        task_source="slack", gate=lambda _: None, resume=lambda **_: None
    )
    assert unregister_mcp_approval_gate(handle) is True
    # Already gone: a second removal must not report success, and must not
    # remove whatever a later registration put in that slot.
    assert unregister_mcp_approval_gate(handle) is False

    replacement = register_mcp_approval_gate(
        task_source="slack", gate=lambda _: None, resume=lambda **_: None
    )
    try:
        assert unregister_mcp_approval_gate(handle) is False
    finally:
        assert unregister_mcp_approval_gate(replacement) is True


def test_registration_generation_tracks_only_successful_source_changes() -> None:
    source = "generation-test"
    before = mcp_approval_gate_registration_generation(source)
    handle = register_mcp_approval_gate(
        task_source=source, gate=lambda _: None, resume=lambda **_: None
    )
    after_register = mcp_approval_gate_registration_generation(source)
    assert after_register == before + 1

    assert unregister_mcp_approval_gate(handle) is True
    after_unregister = mcp_approval_gate_registration_generation(source)
    assert after_unregister == after_register + 1

    assert unregister_mcp_approval_gate(handle) is False
    assert mcp_approval_gate_registration_generation(source) == after_unregister


def test_execution_context_requires_a_tool_call_id() -> None:
    with pytest.raises(ValueError, match="tool_call_id"):
        replace(_context(), tool_call_id="")


def test_require_approval_needs_an_interaction_id() -> None:
    with pytest.raises(ValueError, match="interaction_id"):
        GateDecision.require_approval("")


def test_gate_decision_refuses_an_unknown_verdict() -> None:
    with pytest.raises(ValueError, match="invalid gate decision"):
        GateDecision(decision="maybe")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_gated_resume_with_incomplete_identity_fails_closed(
    registrations: list[Any],
) -> None:
    """The resume half of the registered-source identity guard."""

    _register(registrations, lambda _: None, _unused_resume)
    target = _Target()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    with bind_tool_call_execution_context(replace(_context(), run_id=None)):
        settlement = await tool.resume_user_interaction(
            interaction_id=_gated_interaction_id("slack", "interaction-1"),
            response="approve",
        )

    assert settlement is not None
    assert settlement.status == "failed"
    assert target.calls == []


@pytest.mark.asyncio
async def test_legacy_resume_returning_a_non_settlement_is_rejected(
    registrations: list[Any],
) -> None:
    """A never-gated tool may return a settlement or None - nothing else."""

    class BadResumeTarget(_Target):
        async def resume_user_interaction(
            self, *, interaction_id: str, response: str
        ) -> Any:
            return {"success": True}

    _register(registrations, lambda _: None, _unused_resume)
    target = BadResumeTarget()
    (tool,) = gate_mcp_tools([target], connection={"id": 41})

    with bind_tool_call_execution_context(_context()):
        with pytest.raises(TypeError, match="ToolInteractionSettlement or None"):
            await tool.resume_user_interaction(
                interaction_id="host-owned-interaction", response="Continue"
            )
