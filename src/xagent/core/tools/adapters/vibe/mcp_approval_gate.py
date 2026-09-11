"""Host-scoped approval boundary for deferred MCP calls.

The module has no web or ORM dependencies. A host registers async hooks for
one exact task source, and the MCP loader wraps both direct and sandboxed tools
at their common host-side boundary. With no matching registration the wrapper
is a transparent pass-through.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import RLock
from typing import Any, Literal, Optional, Type
from uuid import uuid4

from pydantic import BaseModel

from ...user_interaction import WAITING_FOR_USER_STATUS, ToolInteractionSettlement
from .base import AbstractBaseTool, ToolMetadata
from .connector_runtime import ConnectorRef

logger = logging.getLogger(__name__)

GateDecisionValue = Literal["allow", "require_approval", "deny"]
ExecutionPattern = Literal["react", "dag"]

_GATE_FAILURE = {
    "success": False,
    "status": "error",
    "error": "The connector approval gate is unavailable. The call was not sent.",
}
_UNSUPPORTED_PATTERN = {
    "success": False,
    "status": "denied",
    "error": "Deferred MCP approval is not supported for this execution pattern.",
}


@dataclass(frozen=True)
class ToolCallExecutionContext:
    """Stable identity of the execution slot issuing one MCP call."""

    task_source: str | None
    task_id: str | None
    run_id: str | None
    turn_id: str | None
    tool_call_id: str
    pattern: ExecutionPattern
    react_step_id: str | None = None
    dag_step_id: str | None = None

    def __post_init__(self) -> None:
        if not self.tool_call_id:
            raise ValueError("tool_call_id must not be empty")

    def is_complete(self) -> bool:
        """Whether the host has every identity needed for a durable decision."""

        return bool(
            self.task_source
            and self.task_id
            and self.run_id
            and self.turn_id
            and (
                (self.pattern == "react" and self.react_step_id)
                or (self.pattern == "dag" and self.dag_step_id)
            )
        )


@dataclass(frozen=True)
class GatedCall:
    """Canonical, immutable snapshot presented to a host approval hook."""

    connector_ref: ConnectorRef
    tool_name: str
    canonical_arguments_json: str
    arguments_sha256: str
    execution_context: ToolCallExecutionContext

    @classmethod
    def from_arguments(
        cls,
        *,
        connector_ref: ConnectorRef,
        tool_name: str,
        arguments: Mapping[str, Any],
        execution_context: ToolCallExecutionContext,
    ) -> GatedCall:
        canonical = json.dumps(
            dict(arguments),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        decoded = json.loads(canonical)
        if not isinstance(decoded, dict):
            raise ValueError("MCP tool arguments must encode a JSON object")
        return cls(
            connector_ref=connector_ref,
            tool_name=tool_name,
            canonical_arguments_json=canonical,
            arguments_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            execution_context=execution_context,
        )

    @property
    def arguments(self) -> dict[str, Any]:
        """Return a fresh deep copy of the canonical argument snapshot."""

        decoded: Any = json.loads(self.canonical_arguments_json)
        if not isinstance(decoded, dict):
            raise ValueError("canonical MCP arguments no longer encode an object")
        return decoded


@dataclass(frozen=True)
class GateDecision:
    """Decision returned by the host before connector dispatch."""

    decision: GateDecisionValue
    interaction_id: str | None = None
    message: str = ""

    def __post_init__(self) -> None:
        if self.decision not in {"allow", "require_approval", "deny"}:
            raise ValueError(f"invalid gate decision: {self.decision!r}")
        if self.decision == "require_approval" and not self.interaction_id:
            raise ValueError("require_approval needs an interaction_id")

    @classmethod
    def allow(cls) -> GateDecision:
        return cls(decision="allow")

    @classmethod
    def require_approval(
        cls, interaction_id: str, *, message: str = ""
    ) -> GateDecision:
        return cls(
            decision="require_approval",
            interaction_id=interaction_id,
            message=message,
        )

    @classmethod
    def deny(cls, *, message: str = "") -> GateDecision:
        return cls(decision="deny", message=message)


GateHook = Callable[[GatedCall], Any]
GateResumeHook = Callable[..., Any]


@dataclass(frozen=True)
class MCPApprovalGateRegistration:
    """Opaque handle used to remove one exact scoped registration."""

    task_source: str
    registration_id: str


@dataclass(frozen=True)
class MCPApprovalReplayContext:
    """Identity bound while an approved payload uses the normal connector path."""

    interaction_id: str
    arguments_sha256: str
    execution_context: ToolCallExecutionContext


@dataclass(frozen=True)
class _RegisteredHooks:
    handle: MCPApprovalGateRegistration
    gate: GateHook
    resume: GateResumeHook
    timeout_seconds: float


_REGISTRATIONS: dict[str, _RegisteredHooks] = {}
_REGISTRATIONS_LOCK = RLock()
_CURRENT_TOOL_CALL_CONTEXT: ContextVar[ToolCallExecutionContext | None] = ContextVar(
    "xagent_mcp_gate_tool_call_context", default=None
)
_CURRENT_REPLAY_CONTEXT: ContextVar[MCPApprovalReplayContext | None] = ContextVar(
    "xagent_mcp_gate_replay_context", default=None
)


def register_mcp_approval_gate(
    *,
    task_source: str,
    gate: GateHook,
    resume: GateResumeHook,
    timeout_seconds: float = 10.0,
) -> MCPApprovalGateRegistration:
    """Register async hooks for one task source; duplicate scopes are refused."""

    if (
        not isinstance(task_source, str)
        or not task_source
        or task_source.strip() != task_source
    ):
        raise ValueError("task_source must be a non-empty trimmed string")
    if not callable(gate) or not callable(resume):
        raise TypeError("gate and resume hooks must be callable")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    handle = MCPApprovalGateRegistration(task_source, str(uuid4()))
    with _REGISTRATIONS_LOCK:
        if task_source in _REGISTRATIONS:
            raise RuntimeError(
                f"an MCP approval gate is already registered for {task_source!r}"
            )
        _REGISTRATIONS[task_source] = _RegisteredHooks(
            handle=handle,
            gate=gate,
            resume=resume,
            timeout_seconds=float(timeout_seconds),
        )
    return handle


def unregister_mcp_approval_gate(handle: MCPApprovalGateRegistration) -> bool:
    """Remove only the registration identified by ``handle``."""

    with _REGISTRATIONS_LOCK:
        current = _REGISTRATIONS.get(handle.task_source)
        if current is None or current.handle != handle:
            return False
        del _REGISTRATIONS[handle.task_source]
        return True


def _registration_for(task_source: str | None) -> _RegisteredHooks | None:
    with _REGISTRATIONS_LOCK:
        return _REGISTRATIONS.get(task_source) if task_source is not None else None


def _has_registrations() -> bool:
    with _REGISTRATIONS_LOCK:
        return bool(_REGISTRATIONS)


@contextmanager
def bind_tool_call_execution_context(
    context: ToolCallExecutionContext,
) -> Iterator[None]:
    """Bind one call identity across async hook and connector tasks."""

    token = _CURRENT_TOOL_CALL_CONTEXT.set(context)
    try:
        yield
    finally:
        _CURRENT_TOOL_CALL_CONTEXT.reset(token)


def current_tool_call_execution_context() -> ToolCallExecutionContext | None:
    return _CURRENT_TOOL_CALL_CONTEXT.get()


def current_mcp_approval_replay_context() -> MCPApprovalReplayContext | None:
    return _CURRENT_REPLAY_CONTEXT.get()


async def _call_async_hook(
    hook: Callable[..., Any], timeout_seconds: float, /, *args: Any, **kwargs: Any
) -> Any:
    returned = hook(*args, **kwargs)
    if not inspect.isawaitable(returned):
        raise TypeError("MCP approval gate hooks must be async")
    return await asyncio.wait_for(returned, timeout=timeout_seconds)


def _connector_ref(connection: Mapping[str, Any]) -> ConnectorRef | None:
    connector_id = connection.get("id")
    if type(connector_id) is not int or connector_id <= 0:
        return None
    return ConnectorRef("mcp", connector_id)


async def _call_resume_hook(
    hook: Callable[..., Any],
    timeout_seconds: float,
    dispatch_started: asyncio.Event,
    /,
    **kwargs: Any,
) -> Any:
    """Bound only pre-dispatch resume policy; never cancel a started write."""

    returned = hook(**kwargs)
    if not inspect.isawaitable(returned):
        raise TypeError("MCP approval gate hooks must be async")

    hook_task = asyncio.ensure_future(returned)
    dispatch_wait = asyncio.create_task(dispatch_started.wait())
    try:
        done, _ = await asyncio.wait(
            {hook_task, dispatch_wait},
            timeout=timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if hook_task in done:
            return hook_task.result()
        if dispatch_wait in done or dispatch_started.is_set():
            # The host has entered the one-shot executor. Connector latency is
            # no longer approval-policy latency and must not be cancelled by
            # the short gate deadline: cancellation cannot prove a remote
            # write did not commit.
            return await hook_task

        hook_task.cancel()
        try:
            await hook_task
        except asyncio.CancelledError:
            pass
        raise TimeoutError("MCP approval resume hook timed out before dispatch")
    finally:
        if not dispatch_wait.done():
            dispatch_wait.cancel()


class MCPApprovalGateTool(AbstractBaseTool):
    """Gate an MCP tool before either direct or sandboxed dispatch."""

    def __init__(
        self,
        target: AbstractBaseTool,
        *,
        connector_ref: ConnectorRef | None,
    ) -> None:
        self._target = target
        self._connector_ref = connector_ref

    @property
    def target(self) -> AbstractBaseTool:
        return self._target

    @property
    def name(self) -> str:
        return self._target.name

    @property
    def description(self) -> str:
        return self._target.description

    @property
    def tags(self) -> list[str]:
        return self._target.tags

    @property
    def metadata(self) -> ToolMetadata:
        metadata = self._target.metadata
        if not _has_registrations():
            return metadata
        # ReAct fans one response out to every interaction paused in a
        # concurrent batch. A gated tool must therefore revoke the scheduler's
        # concurrency/idempotency declaration while any gate can be active.
        return metadata.model_copy(
            update={"read_only": False, "concurrency_safe": False}
        )

    @property
    def is_sandboxed(self) -> bool:
        return bool(getattr(self._target, "is_sandboxed", False))

    def args_type(self) -> Type[BaseModel]:
        return self._target.args_type()

    def return_type(self) -> Type[BaseModel]:
        return self._target.return_type()

    def state_type(self) -> Optional[Type[BaseModel]]:
        return self._target.state_type()

    def return_value_as_string(self, value: Any) -> str:
        return self._target.return_value_as_string(value)

    def is_async(self) -> bool:
        return True

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        context = current_tool_call_execution_context()
        registration = _registration_for(context.task_source if context else None)
        if registration is None:
            if (context is None or not context.task_source) and _has_registrations():
                raise RuntimeError(
                    f"MCP tool {self.name} requires async approval evaluation"
                )
            return self._target.run_json_sync(args)
        raise RuntimeError(f"MCP tool {self.name} requires async approval evaluation")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        context = current_tool_call_execution_context()
        registration = _registration_for(context.task_source if context else None)
        if registration is None:
            if (context is None or not context.task_source) and _has_registrations():
                return dict(_GATE_FAILURE)
            return await self._target.run_json_async(args)
        if context is None or not context.is_complete() or self._connector_ref is None:
            return dict(_GATE_FAILURE)

        try:
            call = GatedCall.from_arguments(
                connector_ref=self._connector_ref,
                tool_name=self.name,
                arguments=args,
                execution_context=context,
            )
            decision = await _call_async_hook(
                registration.gate,
                registration.timeout_seconds,
                call,
            )
            if not isinstance(decision, GateDecision):
                raise TypeError("gate hook returned an invalid decision")
        except Exception:
            logger.warning(
                "MCP approval gate failed closed for tool=%s task_id=%s tool_call_id=%s",
                self.name,
                context.task_id,
                context.tool_call_id,
                exc_info=True,
            )
            return dict(_GATE_FAILURE)

        if decision.decision == "allow":
            return await self._target.run_json_async(call.arguments)
        if decision.decision == "deny":
            return {
                "success": False,
                "status": "denied",
                "error": decision.message or "The connector call was denied.",
            }
        if context.pattern != "react":
            return dict(_UNSUPPORTED_PATTERN)
        return {
            "success": False,
            "status": WAITING_FOR_USER_STATUS,
            "interaction_id": decision.interaction_id,
            "message": decision.message or f"Approve running {self.name}?",
            "message_type": "confirmation",
            "interactions": [
                {
                    "type": "confirm",
                    "field": "approve",
                    "label": "Approve",
                    "options": [
                        {"label": "Approve", "value": "approve"},
                        {"label": "Reject", "value": "reject"},
                    ],
                }
            ],
        }

    async def resume_user_interaction(
        self,
        *,
        interaction_id: str,
        response: str,
    ) -> ToolInteractionSettlement | None:
        context = current_tool_call_execution_context()
        registration = _registration_for(context.task_source if context else None)
        if registration is None:
            if (context is None or not context.task_source) and _has_registrations():
                return ToolInteractionSettlement.failed(result=dict(_GATE_FAILURE))
            target_resume = getattr(self._target, "resume_user_interaction", None)
            if not callable(target_resume):
                return None
            resumed = target_resume(interaction_id=interaction_id, response=response)
            resumed = await resumed if inspect.isawaitable(resumed) else resumed
            if resumed is not None and not isinstance(
                resumed, ToolInteractionSettlement
            ):
                raise TypeError(
                    "resume_user_interaction must return "
                    "ToolInteractionSettlement or None"
                )
            return resumed
        if context is None or not context.is_complete() or self._connector_ref is None:
            return ToolInteractionSettlement.failed(result=dict(_GATE_FAILURE))

        connector_ref = self._connector_ref
        active = True
        used = False
        dispatch_started = asyncio.Event()

        async def executor(arguments: Mapping[str, Any]) -> Any:
            nonlocal used
            if not active or used:
                raise RuntimeError("approval replay executor is no longer available")
            used = True
            canonical = GatedCall.from_arguments(
                connector_ref=connector_ref,
                tool_name=self.name,
                arguments=arguments,
                execution_context=context,
            )
            replay = MCPApprovalReplayContext(
                interaction_id=interaction_id,
                arguments_sha256=canonical.arguments_sha256,
                execution_context=context,
            )
            token = _CURRENT_REPLAY_CONTEXT.set(replay)
            try:
                # The host resume hook must persist its EXECUTING/dispatch
                # marker before awaiting this executor. From this point on,
                # an exception is conservatively ambiguous unless the host
                # returns a persisted known outcome.
                dispatch_started.set()
                return await self._target.run_json_async(canonical.arguments)
            finally:
                _CURRENT_REPLAY_CONTEXT.reset(token)

        try:
            result = await _call_resume_hook(
                registration.resume,
                registration.timeout_seconds,
                dispatch_started,
                interaction_id=interaction_id,
                response=response,
                connector_ref=connector_ref,
                tool_name=self.name,
                execution_context=context,
                executor=executor,
            )
            if not isinstance(result, ToolInteractionSettlement):
                raise TypeError(
                    "MCP approval resume hook must return ToolInteractionSettlement"
                )
            return result
        except Exception:
            logger.warning(
                "MCP approval resume failed closed for tool=%s task_id=%s "
                "tool_call_id=%s interaction_id=%s",
                self.name,
                context.task_id,
                context.tool_call_id,
                interaction_id,
                exc_info=True,
            )
            if dispatch_started.is_set():
                return ToolInteractionSettlement.dispatch_unknown(
                    error=(
                        "The connector call may have reached the external system. "
                        "Automatic retry is disabled."
                    )
                )
            return ToolInteractionSettlement.failed(result=dict(_GATE_FAILURE))
        finally:
            active = False

    async def setup(self, task_id: Optional[str] = None) -> None:
        await self._target.setup(task_id)

    async def teardown(self, task_id: Optional[str] = None) -> None:
        await self._target.teardown(task_id)

    async def save_state_json(self) -> Mapping[str, Any]:
        return await self._target.save_state_json()

    async def load_state_json(self, state: Mapping[str, Any]) -> None:
        await self._target.load_state_json(state)


def gate_mcp_tools(
    tools: Sequence[AbstractBaseTool],
    *,
    connection: Mapping[str, Any] | None = None,
    connector_ref: ConnectorRef | None = None,
) -> list[AbstractBaseTool]:
    """Wrap tools where direct and sandboxed MCP transports converge."""

    ref = connector_ref or _connector_ref(connection or {})
    return [MCPApprovalGateTool(tool, connector_ref=ref) for tool in tools]
