"""Host-scoped approval boundary for deferred MCP calls.

The module has no web or ORM dependencies. A host registers async hooks for
one exact task source, and the MCP loader wraps both direct and sandboxed tools
at their common host-side boundary.

Scoping contract
----------------
**An unregistered or absent ``task_source`` is not gated.** A call whose source
has no registration dispatches to the wrapped tool exactly as it did before the
wrapper existed, and :attr:`MCPApprovalGateTool.metadata` reports the wrapped
tool's own metadata. Fail-closed behavior applies only to a call whose source
*is* registered: such a call must present a complete execution identity and a
connector ref, or it is refused before dispatch.

Hosts that need gating must therefore bind ``task_source`` at their own entry
point. A host that registers a gate for ``"slack"`` but forgets to bind
``task_source="slack"`` on its executions gets *no* approval prompt; it does
not get a fleet-wide outage. This is deliberate: the wrapper sits on the single
loader boundary shared by every execution entry point in the process, so a
global fail-closed rule would take every other host's MCP traffic - read-only
calls included - down with it the moment any one tenant registered a gate.

The one place the wrapper still fails closed without a matching registration is
:meth:`MCPApprovalGateTool.resume_user_interaction` for an interaction that
*this wrapper itself* gated at pause time. That fact is carried durably in the
issued interaction id (see :func:`_gated_interaction_id`), so an approval that
loses its source binding between pause and resume is refused instead of being
replayed unauthorized. An interaction the gate never gated resumes through the
legacy path untouched.
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
# How long a cancelled resume waits for an already-dispatched connector write
# to finish before giving up on observing it. Bounded so an external cancel
# cannot be blocked indefinitely by a hung remote call.
_POST_DISPATCH_DRAIN_SECONDS = 5.0


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


# Durable marker prefix for an interaction id this wrapper issued. The id is
# the ONLY gate-owned value that survives a full checkpoint round-trip: ReAct
# copies it verbatim into the pending-response entry, serializes that entry in
# ``pattern_state``, restores it into a rebuilt process, and hands it back to
# ``resume_user_interaction(interaction_id=...)``. Nothing else the gate can
# write at pause time reaches resume (the published request payload projects a
# fixed key set, and process memory does not survive a rebuild), so the source
# this call was gated for rides along inside the id itself.
_GATED_INTERACTION_PREFIX = "xgate"
_GATED_INTERACTION_SEPARATOR = ":"


def _gated_interaction_id(task_source: str, host_interaction_id: str) -> str:
    """Stamp the gating source onto a host-issued interaction id.

    The source is length-prefixed rather than merely delimited: a registered
    ``task_source`` may legitimately contain the separator, and a plain
    three-way split would then recover a *prefix* of the real source and
    silently match a different registration.
    """

    return _GATED_INTERACTION_SEPARATOR.join(
        (
            _GATED_INTERACTION_PREFIX,
            str(len(task_source)),
            task_source + host_interaction_id,
        )
    )


def _gated_interaction_source(interaction_id: str) -> tuple[str, str] | None:
    """Recover ``(task_source, host_interaction_id)`` from a gated id.

    Returns ``None`` for any id this wrapper did not issue, which is what
    keeps a never-gated interaction on the legacy resume path. This function
    parses untrusted-shaped input and must never raise: it is called at the top
    of ``resume_user_interaction``, outside any try, and ReAct invokes that
    callback outside its own rollback block, so an exception here would escape
    ``_deliver_pending_tool_interaction_responses`` with the pending entry
    un-popped - i.e. a permanently stuck task.
    """

    parts = interaction_id.split(_GATED_INTERACTION_SEPARATOR, 2)
    if len(parts) != 3 or parts[0] != _GATED_INTERACTION_PREFIX:
        return None
    _, raw_length, payload = parts
    # ``isdecimal`` rather than ``isdigit``: the latter accepts superscripts
    # and other numeric-but-not-decimal code points ("²".isdigit() is True)
    # that int() then rejects with ValueError.
    if not raw_length.isdecimal():
        return None
    try:
        length = int(raw_length)
    except ValueError:  # pragma: no cover - belt and braces behind isdecimal
        return None
    if length <= 0 or length > len(payload):
        return None
    return payload[:length], payload[length:]


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


def _ensure_awaitable(returned: Any) -> Any:
    """Reject a hook that is not async, before anything is scheduled."""

    if not inspect.isawaitable(returned):
        raise TypeError("MCP approval gate hooks must be async")
    return returned


async def _call_async_hook(
    hook: Callable[..., Any], timeout_seconds: float, /, *args: Any, **kwargs: Any
) -> Any:
    returned = _ensure_awaitable(hook(*args, **kwargs))
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
    """Bound only pre-dispatch resume policy; never cancel a started write.

    Cancellation semantics, post-dispatch
    -------------------------------------
    Once the host has entered the executor, this function's job is to make sure
    the connector write is *observed*, not to keep the caller alive. An external
    cancellation that arrives after dispatch therefore still propagates: the
    drained hook's settlement, if it produced one, is deliberately **discarded**
    and ``CancelledError`` is re-raised, so a write that demonstrably completed
    is rolled back to "pending" from the runtime's point of view.

    That is the correct trade. The caller is being torn down (task cancel, lease
    loss); there is no longer a ledger to project a settlement onto, and
    smuggling a return value out of a cancelled frame would suppress the
    cancellation the host asked for. Correctness on replay comes from
    idempotency instead: the host resume hook must persist its EXECUTING /
    dispatch marker *before* awaiting the executor - which the module docstring
    already requires - so a later replay recognizes the in-flight attempt rather
    than issuing a second write.
    """

    returned = _ensure_awaitable(hook(**kwargs))

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
            #
            # SHIELDED, and that is load-bearing. A bare ``await hook_task``
            # here is the frame an external cancel lands on while the write is
            # in flight: it would tear straight through into ``hook_task``,
            # killing the host hook inside ``await executor(...)`` mid-RPC.
            # ``hook_task`` would then already be done-and-cancelled by the
            # time the ``finally`` runs, so the bounded drain and its warning
            # below would be skipped entirely - the write interrupted with no
            # settlement and no log line. The shield lets the cancellation
            # unwind to the ``finally`` with the task still alive, which is
            # the only state in which that drain can do its job.
            return await asyncio.shield(hook_task)

        hook_task.cancel()
        await _drain(hook_task)
        raise TimeoutError("MCP approval resume hook timed out before dispatch")
    finally:
        if not dispatch_wait.done():
            dispatch_wait.cancel()
        # An EXTERNAL cancellation (task cancel, lease loss) unwinds through
        # here while ``hook_task`` may be mid connector write. Leaving it
        # detached lets that write land after the caller believes the task was
        # cancelled, with no settlement recorded anywhere.
        #
        # Pre-dispatch there is nothing to preserve, so cancel and drain.
        # Post-dispatch the write is already in flight and cancelling proves
        # nothing about it, so the drain is bounded and shielded: bounded
        # because a `finally` that awaits forever would make the enclosing
        # cancellation un-cancellable, shielded because the drain itself is
        # running under an active CancelledError and would otherwise be
        # interrupted immediately.
        if not hook_task.done():
            if not dispatch_started.is_set():
                hook_task.cancel()
                await _drain(hook_task)
            else:
                await _drain(
                    asyncio.shield(hook_task), timeout=_POST_DISPATCH_DRAIN_SECONDS
                )
                if not hook_task.done():
                    logger.warning(
                        "MCP approval resume hook is still dispatching after "
                        "cancellation; the connector write may still land."
                    )


async def _drain(awaitable: Any, *, timeout: float | None = None) -> None:
    """Await ``awaitable`` to completion, swallowing its outcome.

    Used only on paths that already have a settlement to report: the point is
    that no connector write is left running unobserved, not to surface the
    drained task's own result or error.
    """

    try:
        if timeout is None:
            await awaitable
        else:
            await asyncio.wait_for(awaitable, timeout=timeout)
    except (Exception, asyncio.CancelledError):
        logger.debug("drained MCP approval resume hook task", exc_info=True)


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
        # ReAct fans one user response out to every interaction paused in a
        # concurrent batch, so two gated calls batched together can both be
        # settled by a single approval. A tool that can pause for approval
        # must therefore revoke the scheduler's concurrency declaration.
        #
        # Scoping this to the call's own task_source is not possible here:
        # ``metadata`` is a property read by ``_tool_is_concurrency_safe``
        # (react.py:3513) while the scheduler is *planning* a batch, outside
        # any bound ToolCallExecutionContext, so there is no call whose source
        # could be consulted. Degrading only ``concurrency_safe`` keeps the
        # blast radius to batching: that field is, per the contract comment on
        # ToolMetadata (base.py:83-86), "the only field the scheduler reads".
        #
        # ``read_only`` is deliberately left as the wrapped tool declares it.
        # Nothing in the agent or scheduler layer reads ``metadata.read_only``
        # (only base.py:190-194 consumes it, to derive ``concurrency_safe``
        # when a tool declares no explicit value) so forcing it to False buys
        # no safety, while lying about a read-only tool's nature would
        # mis-describe every wrapped MCP tool to any future consumer.
        return metadata.model_copy(update={"concurrency_safe": False})

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
            # Unregistered or absent source: not this gate's business. Dispatch
            # exactly as the unwrapped tool would.
            return self._target.run_json_sync(args)
        # The call's own source IS registered, so it must be evaluated. The
        # gate hooks are async and there is no running loop to await them on.
        raise RuntimeError(f"MCP tool {self.name} requires async approval evaluation")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        context = current_tool_call_execution_context()
        registration = _registration_for(context.task_source if context else None)
        if registration is None:
            # Unregistered or absent source: transparent pass-through.
            return await self._target.run_json_async(args)
        # From here the call's own source is registered, so every remaining
        # exit fails closed.
        if context is None or not context.is_complete() or self._connector_ref is None:
            return dict(_GATE_FAILURE)
        if context.pattern != "react":
            # Checked BEFORE the hook: a decision issued for an unsupported
            # pattern would leave the host holding a persisted, clickable
            # approval prompt that nothing can ever resume.
            return dict(_UNSUPPORTED_PATTERN)

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
        return {
            "success": False,
            "status": WAITING_FOR_USER_STATUS,
            # Stamped with the gating source so a resume that arrives without a
            # matching binding is refused instead of replayed unauthorized.
            "interaction_id": _gated_interaction_id(
                registration.handle.task_source, decision.interaction_id or ""
            ),
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
        # What this wrapper itself recorded at pause time, carried durably in
        # the interaction id. ``None`` means the gate never gated this call.
        gated = _gated_interaction_source(interaction_id)

        if (
            registration is None
            or gated is None
            or gated[0] != registration.handle.task_source
        ):
            if gated is not None:
                # This interaction WAS gated, for source ``gated[0]``, but the
                # resume context does not present that source (it lost its
                # binding, or arrived through a different entry point). Replaying
                # the approved write here would dispatch it outside the policy
                # that authorized it, so refuse. ``failed`` is correct rather
                # than ``dispatch_unknown``: nothing was dispatched.
                logger.warning(
                    "Refusing to resume a gated MCP interaction without its "
                    "gating source. tool=%s gated_source=%r resume_source=%r "
                    "interaction_id=%r",
                    self.name,
                    gated[0],
                    context.task_source if context else None,
                    interaction_id,
                )
                return ToolInteractionSettlement.failed(
                    result=dict(_GATE_FAILURE),
                    error=(
                        "This connector call was approved under a different "
                        "execution identity and was not sent."
                    ),
                )
            # Never gated by this wrapper: legacy path, byte-for-byte what the
            # unwrapped tool would do. Returning ``None`` for a target with no
            # callback keeps ReAct's free-text replan behavior unchanged.
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

        # The gating source and the resume source agree. Hand the host back the
        # id it issued, not the wrapper's stamped one.
        host_interaction_id = gated[1]
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
                interaction_id=host_interaction_id,
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
                interaction_id=host_interaction_id,
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
