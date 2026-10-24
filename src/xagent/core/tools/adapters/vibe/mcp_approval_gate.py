"""Task- and tool-scoped approval boundary for deferred MCP calls.

The module has no web or ORM dependencies. A host registers policy hooks for
one exact task source, mints immutable authority for a resolved task, and the
MCP loader wraps matcher-selected direct and sandboxed tools at their common
host-side boundary. Legacy source-wide registrations remain supported until
their callers migrate to explicit :class:`MCPApprovalGateResolution` values.

Scoping contract
----------------
Explicit ``OFF`` preserves the exact raw tools and metadata. Explicit ``ON``
requires live authority and applies the static matcher before the per-call
classifier. ``UNKNOWN`` maps to a rollback fence: while the live matcher is
available only its selected tools are refused, and approved resumes remain
pending for recovery. Without a live matcher ``UNKNOWN`` passes raw tools
through because widening uncertainty into an all-tools outage is forbidden.
Stale or forged explicit authority fails closed; when a live matcher exists,
that closure is limited to matcher-selected tools.

Omitting a resolution and scope preserves the legacy path. An unregistered or
absent legacy ``task_source`` is not gated, so unrelated traffic continues to
dispatch exactly as it did before the wrapper existed.

The one place the wrapper still fails closed without a matching registration is
:meth:`MCPApprovalGateTool.resume_user_interaction` for an interaction that
*this wrapper itself* gated at pause time. That fact is carried durably in the
issued interaction id (see :func:`_gated_interaction_id`), so an approval that
loses its source binding between pause and resume is refused instead of being
replayed unauthorized. An interaction the gate never gated resumes through the
legacy path untouched.

Host contract
-------------
Two obligations the gate cannot enforce for the host. Both are load-bearing;
neither is checked at runtime, so a host that ignores them gets silent
incorrectness rather than an error.

1. **Persist and acknowledge a dispatch marker before awaiting the executor.**
   The resume hook receives a callable :class:`MCPApprovalReplayExecutor` under
   the existing ``executor`` keyword. After durably recording that this
   interaction has entered dispatch (an ``EXECUTING`` row, an outbox entry -
   whatever its ledger calls it), the host must synchronously call
   :meth:`MCPApprovalReplayExecutor.acknowledge_dispatch` before awaiting the
   executor. From that acknowledgement onward the gate treats the call as
   possibly committed, even if cancellation lands before executor entry. Old
   integrations remain safe and source-compatible: entering the executor
   performs the same acknowledgement automatically. Only the host's own marker
   lets a later replay tell "never dispatched" from "may have landed"; without
   it, a replay can issue a second external write.

2. **Compare the approved argument hash before dispatching.** The gate freezes
   the approved arguments at pause time and exposes
   :attr:`GatedCall.arguments_sha256`; at resume it recomputes the same digest
   over whatever the host passes to the executor and publishes it on
   :class:`MCPApprovalReplayContext`. **The gate does not compare them** - it
   has no durable copy of the pause-time value, by design: the host's approval
   ledger is the single authoritative record of what was approved, and a second
   persisted copy here would be a divergence path, not a safety net. The host
   must therefore compare the replay context's digest against the value its own
   ledger froze at approval time, and refuse on mismatch. Until it does, the
   "approved call == executed call" property is *not* enforced anywhere.

This core deliberately does not define provider aggregation, source taxonomy,
delegation propagation, web configuration, or production enrollment. Those
layers consume the public resolution and scope contract without being coupled
back into the generic gate.
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
from threading import BoundedSemaphore
from threading import Event as ThreadEvent
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
MCPApprovalGateResolutionState = Literal["on", "off", "unknown"]

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
_NESTED_APPROVAL_UNAVAILABLE = {
    "success": False,
    "status": "denied",
    "failure_code": "nested_delegation_not_approvable",
    "error": "Deferred MCP approval is unavailable in delegated execution.",
}
_ROLLBACK_FENCE_REGISTRATION_ID = "rollback-fence"
# Default for how long the gate will wait to OBSERVE a connector write that
# has already been dispatched - both on the ordinary success path and when
# draining after a cancellation. Deliberately not the registration's
# ``timeout_seconds``: that one bounds approval *policy* latency before
# anything is sent, and is typically small, whereas this bounds a remote call
# that is already in flight and cannot be un-sent. Bounded either way, so a
# hung connector can neither park a resume forever nor block an external
# cancel.
#
# This is a default, not a universal ceiling: no supported MCP transport
# guarantees every call completes within 30s (the direct adapter awaits
# ``session.call_tool`` with no per-call timeout, and streamable-HTTP/SSE
# sessions default their own read timeout to five minutes - see
# ``sessions.py``). A host whose registered task source dispatches to
# connectors that legitimately run longer must override this via
# ``register_mcp_approval_gate(..., dispatch_observe_seconds=...)`` rather
# than have valid in-flight writes settle as ``dispatch_unknown`` at this
# default cutoff.
_DISPATCH_OBSERVE_SECONDS = 30.0
# How long the gate waits for a connector call it has given up observing to
# actually unwind once cancelled. This is a *teardown* budget, not a response
# budget: what it covers is the ``async with create_session(...)`` exit -
# closing the socket and, for stdio, terminating and reaping the child
# process - not the remote's answer, which by this point is never coming.
_DISPATCH_CLEANUP_SECONDS = 5.0

# Cleanup tasks for dispatches the gate stopped observing. Held in a
# module-level set purely so the event loop keeps a strong reference: a bare
# ``ensure_future`` result can be garbage collected mid-flight, which would
# abandon exactly the teardown this set exists to guarantee.
_ORPHANED_DISPATCH_CLEANUPS: set["asyncio.Task[None]"] = set()
# Synchronous classifiers run outside the event loop. Bound detached workers so
# one broken host policy cannot grow an unbounded executor queue.
_SYNC_CLASSIFIER_SLOTS = BoundedSemaphore(4)


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
    """Canonical, immutable snapshot presented to a host approval hook.

    ``arguments_sha256`` is the approval's identity: the host is expected to
    freeze it alongside whatever it shows the approver, and to compare it again
    at resume. See "Host contract" in the module docstring - the gate publishes
    both digests but deliberately never compares them itself.
    """

    connector_ref: ConnectorRef
    tool_name: str
    canonical_arguments_json: str
    arguments_sha256: str
    execution_context: ToolCallExecutionContext
    # Exact MCP-advertised name used by the static matcher. ``None`` preserves
    # compatibility for hosts constructing snapshots outside the MCP loader.
    raw_tool_name: str | None = None

    @classmethod
    def from_arguments(
        cls,
        *,
        connector_ref: ConnectorRef,
        tool_name: str,
        arguments: Mapping[str, Any],
        execution_context: ToolCallExecutionContext,
        raw_tool_name: str | None = None,
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
            raw_tool_name=raw_tool_name,
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
GateClassifier = Callable[[GatedCall], Any]
GateToolMatcher = Callable[[ConnectorRef, str], bool]


@dataclass(frozen=True)
class MCPApprovalGateRegistration:
    """Opaque handle used to remove one exact scoped registration."""

    task_source: str
    registration_id: str


@dataclass(frozen=True)
class MCPApprovalGateScope:
    """Trusted task-scoped authority minted from one live registration."""

    task_source: str
    registration_id: str


@dataclass(frozen=True)
class MCPApprovalGateResolution:
    """Explicit immutable ON/OFF/UNKNOWN task approval resolution.

    ``None`` is deliberately not one of the states. Callers may still omit a
    resolution entirely to preserve the legacy integration path, but a runtime
    provider that answered must return one of these three exact values.
    """

    state: MCPApprovalGateResolutionState
    task_source: str
    scope: MCPApprovalGateScope | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        _validate_task_source(self.task_source)
        if self.state == "on":
            if (
                not isinstance(self.scope, MCPApprovalGateScope)
                or self.scope.task_source != self.task_source
                or self.reason is not None
            ):
                raise ValueError(
                    "an ON approval resolution requires a same-source scope and no reason"
                )
            return
        if self.state == "off":
            if self.scope is not None or self.reason is not None:
                raise ValueError(
                    "an OFF approval resolution cannot carry scope or reason"
                )
            return
        if self.state == "unknown":
            if self.scope is not None:
                raise ValueError("an UNKNOWN approval resolution cannot carry a scope")
            if (
                not isinstance(self.reason, str)
                or not self.reason
                or self.reason.strip() != self.reason
            ):
                raise ValueError(
                    "an UNKNOWN approval resolution requires a non-empty trimmed reason"
                )
            return
        raise ValueError(f"invalid MCP approval resolution state: {self.state!r}")

    @classmethod
    def on(cls, scope: MCPApprovalGateScope) -> "MCPApprovalGateResolution":
        if not isinstance(scope, MCPApprovalGateScope):
            raise TypeError("scope must be an MCPApprovalGateScope")
        return cls(state="on", task_source=scope.task_source, scope=scope)

    @classmethod
    def off(cls, *, task_source: str) -> "MCPApprovalGateResolution":
        return cls(state="off", task_source=task_source)

    @classmethod
    def unknown(cls, *, task_source: str, reason: str) -> "MCPApprovalGateResolution":
        return cls(state="unknown", task_source=task_source, reason=reason)

    @property
    def approval_scope(self) -> MCPApprovalGateScope | None:
        """Map the explicit resolution onto the core loader authority seam."""

        if self.state == "on":
            return self.scope
        if self.state == "unknown":
            return mcp_approval_gate_rollback_fence(task_source=self.task_source)
        return None

    @property
    def signature_identity(self) -> tuple[str, str, str | None, str | None]:
        """Stable primitive identity for tool caches and rebuild signatures."""

        return (
            self.state,
            self.task_source,
            self.scope.registration_id if self.scope is not None else None,
            self.reason,
        )


class MCPApprovalGateTemporarilyUnavailable(RuntimeError):
    """A gated approval remains pending until trusted authority recovers."""


@dataclass(frozen=True)
class MCPApprovalReplayContext:
    """Identity bound while an approved payload uses the normal connector path.

    Read it with :func:`current_mcp_approval_replay_context` from inside a
    connector call to learn which approval authorized the dispatch in flight.

    ``arguments_sha256`` is recomputed here over the arguments the host is
    actually about to send. **The gate never compares it to the approved
    value** - it keeps no durable copy of that value, because the host's
    approval ledger is the authoritative record (see "Host contract" in the
    module docstring). A host that wants the "approved call == executed call"
    guarantee must compare this digest against the one its ledger froze at
    approval time and refuse on mismatch; nothing in this module does it.
    """

    interaction_id: str
    arguments_sha256: str
    execution_context: ToolCallExecutionContext


class MCPApprovalReplayExecutor:
    """One-shot replay callable with an explicit durable-dispatch boundary.

    A durable host calls :meth:`acknowledge_dispatch` synchronously immediately
    after committing its dispatch marker and before awaiting this object. The
    acknowledgement does not dispatch the connector by itself; it tells the
    wrapper that automatic retry authority has ended, so cancellation, timeout,
    or a hook failure in the small marker-to-executor window is classified as
    possibly dispatched rather than definitely not sent.

    Calling the executor directly remains supported for legacy hooks and
    performs the acknowledgement immediately before connector dispatch. The
    capability is created for one resume interaction and becomes unusable when
    that resume finishes, so it cannot opt another interaction into dispatch.
    """

    def __init__(
        self,
        *,
        execute: Callable[[Mapping[str, Any]], Any],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._execute = execute
        self._loop = loop
        self._dispatch_acknowledged = ThreadEvent()
        self._dispatch_notification = asyncio.Event()
        self._lifecycle_lock = RLock()
        self._active = True

    def acknowledge_dispatch(self) -> None:
        """Mark the host's durable transition into non-retryable dispatch.

        The call is idempotent while this capability is active and is safe from
        a host worker thread. The thread event is authoritative for immediate
        classification; the loop notification only wakes an async waiter.
        """

        with self._lifecycle_lock:
            if not self._active:
                raise RuntimeError("approval replay executor is no longer available")
            if self._dispatch_acknowledged.is_set():
                return
            self._dispatch_acknowledged.set()
            self._loop.call_soon_threadsafe(self._dispatch_notification.set)

    @property
    def dispatch_acknowledged(self) -> bool:
        """Whether the host has durably surrendered automatic retry authority."""

        return self._dispatch_acknowledged.is_set()

    async def wait_for_dispatch_acknowledgement(self) -> None:
        if self.dispatch_acknowledged:
            return
        await self._dispatch_notification.wait()

    async def __call__(self, arguments: Mapping[str, Any]) -> Any:
        with self._lifecycle_lock:
            if not self._active:
                raise RuntimeError("approval replay executor is no longer available")
        return await _ensure_awaitable(self._execute(arguments))

    def _deactivate(self) -> None:
        with self._lifecycle_lock:
            self._active = False


@dataclass(frozen=True)
class _RegisteredHooks:
    handle: MCPApprovalGateRegistration
    gate: GateHook
    resume: GateResumeHook
    timeout_seconds: float
    dispatch_observe_seconds: float
    tool_matcher: GateToolMatcher | None = None
    classifier: GateClassifier | None = None


_REGISTRATIONS: dict[str, _RegisteredHooks] = {}
_REGISTRATION_GENERATIONS: dict[str, int] = {}
_REGISTRATIONS_LOCK = RLock()
_CURRENT_TOOL_CALL_CONTEXT: ContextVar[ToolCallExecutionContext | None] = ContextVar(
    "xagent_mcp_gate_tool_call_context", default=None
)
_CURRENT_REPLAY_CONTEXT: ContextVar[MCPApprovalReplayContext | None] = ContextVar(
    "xagent_mcp_gate_replay_context", default=None
)


def _validate_task_source(task_source: str) -> None:
    if (
        not isinstance(task_source, str)
        or not task_source
        or task_source.strip() != task_source
    ):
        raise ValueError("task_source must be a non-empty trimmed string")


def register_mcp_approval_gate(
    *,
    task_source: str,
    gate: GateHook,
    resume: GateResumeHook,
    timeout_seconds: float = 10.0,
    dispatch_observe_seconds: float | None = None,
    tool_matcher: GateToolMatcher | None = None,
    classifier: GateClassifier | None = None,
) -> MCPApprovalGateRegistration:
    """Register async hooks for one task source; duplicate scopes are refused.

    ``dispatch_observe_seconds`` bounds how long the gate will wait to OBSERVE
    a connector write *this registration* has already dispatched, once its
    resume hook enters the one-shot executor (see ``_DISPATCH_OBSERVE_SECONDS``
    and :func:`_call_resume_hook`). It is deliberately independent from
    ``timeout_seconds``, which only bounds pre-dispatch approval *policy*
    latency and is typically small. Leave it unset to use the module default;
    pass an explicit value when this task source's connectors are known to run
    longer (or shorter) than that default - for example to match a connector's
    own configured call timeout - so a valid in-flight write is not discarded
    as ``dispatch_unknown`` at an unrelated cutoff.
    """

    _validate_task_source(task_source)
    if not callable(gate) or not callable(resume):
        raise TypeError("gate and resume hooks must be callable")
    if (tool_matcher is None) != (classifier is None):
        raise ValueError("tool_matcher and classifier must be supplied together")
    if tool_matcher is not None and not callable(tool_matcher):
        raise TypeError("tool_matcher must be callable")
    if classifier is not None and not callable(classifier):
        raise TypeError("classifier must be callable")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    if dispatch_observe_seconds is not None and (
        not math.isfinite(dispatch_observe_seconds) or dispatch_observe_seconds <= 0
    ):
        raise ValueError("dispatch_observe_seconds must be finite and positive")
    resolved_observe_seconds = (
        float(dispatch_observe_seconds)
        if dispatch_observe_seconds is not None
        else _DISPATCH_OBSERVE_SECONDS
    )
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
            dispatch_observe_seconds=resolved_observe_seconds,
            tool_matcher=tool_matcher,
            classifier=classifier,
        )
        _REGISTRATION_GENERATIONS[task_source] = (
            _REGISTRATION_GENERATIONS.get(task_source, 0) + 1
        )
    return handle


def mcp_approval_gate_scope(
    handle: MCPApprovalGateRegistration,
) -> MCPApprovalGateScope:
    """Mint task authority from one active matcher/classifier registration."""

    with _REGISTRATIONS_LOCK:
        current = _REGISTRATIONS.get(handle.task_source)
        if current is None or current.handle != handle:
            raise ValueError("approval gate registration is not active")
        if current.tool_matcher is None or current.classifier is None:
            raise ValueError("legacy source-wide registrations have no task scope")
    return MCPApprovalGateScope(handle.task_source, handle.registration_id)


def mcp_approval_gate_rollback_fence(*, task_source: str) -> MCPApprovalGateScope:
    """Represent UNKNOWN authority without confusing it with explicit OFF."""

    _validate_task_source(task_source)
    return MCPApprovalGateScope(task_source, _ROLLBACK_FENCE_REGISTRATION_ID)


def unregister_mcp_approval_gate(handle: MCPApprovalGateRegistration) -> bool:
    """Remove only the registration identified by ``handle``."""

    with _REGISTRATIONS_LOCK:
        current = _REGISTRATIONS.get(handle.task_source)
        if current is None or current.handle != handle:
            return False
        del _REGISTRATIONS[handle.task_source]
        _REGISTRATION_GENERATIONS[handle.task_source] = (
            _REGISTRATION_GENERATIONS.get(handle.task_source, 0) + 1
        )
        return True


def mcp_approval_gate_registration_generation(task_source: str) -> int:
    """Return the monotonic approval-policy generation for one task source."""

    _validate_task_source(task_source)
    with _REGISTRATIONS_LOCK:
        return _REGISTRATION_GENERATIONS.get(task_source, 0)


def _registration_for(task_source: str | None) -> _RegisteredHooks | None:
    with _REGISTRATIONS_LOCK:
        return _REGISTRATIONS.get(task_source) if task_source is not None else None


def _registration_for_scope(
    scope: MCPApprovalGateScope | None,
) -> _RegisteredHooks | None:
    if not isinstance(scope, MCPApprovalGateScope):
        return None
    with _REGISTRATIONS_LOCK:
        current = _REGISTRATIONS.get(scope.task_source)
        if (
            current is None
            or current.handle.registration_id != scope.registration_id
            or current.tool_matcher is None
            or current.classifier is None
        ):
            return None
        return current


def _scoped_registration_for_source(
    task_source: str | None,
) -> _RegisteredHooks | None:
    registration = _registration_for(task_source)
    if (
        registration is None
        or registration.tool_matcher is None
        or registration.classifier is None
    ):
        return None
    return registration


def has_approval_gate_for_source(task_source: str | None) -> bool:
    """Backward-compatible alias for legacy source-wide gate discovery."""

    return has_legacy_approval_gate_for_source(task_source)


def has_legacy_approval_gate_for_source(task_source: str | None) -> bool:
    """Whether calls bound to ``task_source`` use the legacy source-wide gate.

    Public because the host wiring has to answer one question this module
    cannot answer for it: may a *delegated* run materialize connectors that
    the delegating run would have had to get approved? A nested run builds its
    own execution context and cannot pause for approval, so the wiring uses
    this to refuse those connectors rather than let them dispatch outside the
    parent's scope. ``None`` is never gated, matching dispatch.
    """

    registration = _registration_for(task_source)
    return registration is not None and registration.tool_matcher is None


def has_scoped_approval_gate_for_source(task_source: str | None) -> bool:
    """Whether ``task_source`` has a live task/tool-scoped registration."""

    return _scoped_registration_for_source(task_source) is not None


def _has_legacy_registrations() -> bool:
    with _REGISTRATIONS_LOCK:
        return any(item.tool_matcher is None for item in _REGISTRATIONS.values())


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
    callback outside its own rollback block. An accidental parser exception
    would therefore abort the turn with the pending entry un-popped. Only the
    explicit transient-authority exception uses that retry behavior on purpose.
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


def is_gate_issued_interaction_id(interaction_id: str) -> bool:
    """Return True when ``interaction_id`` has the shape of a gate-issued id.

    This is a shape check only: there is no MAC and no registry of issued
    ids behind it, so any string built in the gate's format is accepted.
    The prefix and length parsing lives only in
    :func:`_gated_interaction_source`; code outside this module that needs
    to recognise a gate-issued id must call this function instead of
    parsing the id itself. Never raises for a ``str`` input: it only
    delegates to :func:`_gated_interaction_source`, which never raises for
    one.
    """

    return _gated_interaction_source(interaction_id) is not None


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


async def _call_classifier(
    classifier: GateClassifier,
    timeout_seconds: float,
    call: GatedCall,
) -> Any:
    """Run a side-effect-free classifier without blocking the event loop."""

    if inspect.iscoroutinefunction(classifier) or inspect.iscoroutinefunction(
        getattr(type(classifier), "__call__", None)
    ):
        return await asyncio.wait_for(classifier(call), timeout=timeout_seconds)

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    limiter = _SYNC_CLASSIFIER_SLOTS
    while not limiter.acquire(blocking=False):
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError("synchronous approval classifier capacity exhausted")
        await asyncio.sleep(min(0.005, remaining))

    task = asyncio.create_task(asyncio.to_thread(classifier, call))

    def release_slot(completed: asyncio.Task[Any]) -> None:
        limiter.release()
        if not completed.cancelled():
            completed.exception()

    task.add_done_callback(release_slot)

    def discard_late_awaitable(completed: asyncio.Task[Any]) -> None:
        if completed.cancelled():
            return
        try:
            late_result = completed.result()
        except Exception:
            return
        if inspect.iscoroutine(late_result):
            late_result.close()

    try:
        remaining = max(0.0, deadline - loop.time())
        returned = await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
    except (TimeoutError, asyncio.CancelledError):
        task.add_done_callback(discard_late_awaitable)
        raise
    if not inspect.isawaitable(returned):
        return returned
    remaining = max(0.0, deadline - loop.time())
    return await asyncio.wait_for(returned, timeout=remaining)


def _connector_ref(connection: Mapping[str, Any]) -> ConnectorRef | None:
    connector_id = connection.get("id")
    if type(connector_id) is not int or connector_id <= 0:
        return None
    return ConnectorRef("mcp", connector_id)


async def _call_resume_hook(
    hook: Callable[..., Any],
    timeout_seconds: float,
    executor: MCPApprovalReplayExecutor,
    /,
    *,
    dispatch_observe_seconds: float,
    **kwargs: Any,
) -> Any:
    """Bound only pre-dispatch resume policy; never cancel a started write.

    Cancellation semantics, post-dispatch
    -------------------------------------
    Once the host has acknowledged its durable marker (or entered the executor,
    which acknowledges automatically), this function's job is to make sure the
    possibly-started connector write is *observed*, not to keep the caller alive.
    An external cancellation that arrives after acknowledgement therefore still
    propagates: the drained hook's settlement, if it produced one, is deliberately
    **discarded** and ``CancelledError`` is re-raised.

    That is the correct trade. The caller is being torn down (task cancel, lease
    loss); there is no longer a ledger to project a settlement onto, and
    smuggling a return value out of a cancelled frame would suppress the
    cancellation the host asked for. Correctness on replay comes from
    idempotency instead: the host resume hook must persist its EXECUTING /
    dispatch marker and acknowledge it *before* awaiting the executor - which
    the module docstring already requires - so a later replay recognizes the
    in-flight attempt rather than issuing a second write.

    ``dispatch_observe_seconds`` is spent exactly once per dispatched call,
    never twice. A monotonic deadline is recorded the moment the gate starts
    observing (right before the first post-dispatch ``wait_for``); if an
    external cancellation interrupts that wait before it times out, the
    ``finally`` drain below reuses the SAME deadline and passes only the
    remaining time, rather than re-arming a fresh full budget. Without this, a
    cancellation landing near the end of the first wait would let the caller
    stay blocked for close to two full budgets back to back.
    """

    returned = _ensure_awaitable(hook(**kwargs))

    hook_task = asyncio.ensure_future(returned)
    dispatch_wait = asyncio.create_task(executor.wait_for_dispatch_acknowledgement())
    # Set once the observation budget has already been spent waiting on this
    # same dispatched call, so the cleanup path does not spend it again.
    observed = False
    # Monotonic wall-clock deadline for the observation budget, set once when
    # the gate first starts waiting on the dispatched call below. ``None``
    # until then: cancellation arriving before dispatch is even observed has
    # nothing to share a deadline with, so that path keeps its own full
    # budget (see the ``finally`` branch below).
    observe_deadline: float | None = None
    try:
        done, _ = await asyncio.wait(
            {hook_task, dispatch_wait},
            timeout=timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if hook_task in done:
            return hook_task.result()
        if dispatch_wait in done or executor.dispatch_acknowledged:
            # The host has durably surrendered retry authority. Connector
            # latency is no longer approval-policy latency and must not be
            # cancelled by the short gate deadline: cancellation cannot prove
            # a remote write did not commit.
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
            #
            # BOUNDED as well: a hung connector RPC would otherwise park the
            # resume forever with no backstop, since the registration deadline
            # has already been spent by this point. The TimeoutError raised on
            # expiry reaches the caller with ``dispatch_acknowledged`` set,
            # which makes it settle as ``dispatch_unknown`` rather than a clean
            # failure - the write may well have landed.
            observe_deadline = (
                asyncio.get_running_loop().time() + dispatch_observe_seconds
            )
            try:
                return await asyncio.wait_for(
                    asyncio.shield(hook_task), timeout=dispatch_observe_seconds
                )
            except TimeoutError:
                # The observation budget is spent; do not let the ``finally``
                # spend a second one waiting for the same hung call.
                observed = True
                raise

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
            if not executor.dispatch_acknowledged:
                hook_task.cancel()
                await _drain(hook_task)
            elif not observed:
                # Cancellation interrupted the observation wait above before
                # it timed out. Share its deadline instead of re-arming a
                # fresh ``dispatch_observe_seconds``: with ``observe_deadline``
                # unset (cancellation raced ahead of the assignment above, or
                # arrived before this call ever started observing) the full
                # budget still applies, matching the pre-existing behavior.
                remaining = dispatch_observe_seconds
                if observe_deadline is not None:
                    remaining = max(
                        0.0, observe_deadline - asyncio.get_running_loop().time()
                    )
                await _drain(asyncio.shield(hook_task), timeout=remaining)
                if not hook_task.done():
                    _terminate_unobserved(hook_task, dispatch_observe_seconds)
            else:
                # Budget already spent on the success path above.
                _terminate_unobserved(hook_task, dispatch_observe_seconds)


def _terminate_unobserved(
    hook_task: "asyncio.Future[Any]", dispatch_observe_seconds: float
) -> None:
    """Cancel and drain a dispatched call the gate has stopped observing.

    Bounded observation is not bounded *resource* lifetime. Only the ``shield``
    wrapper is cancelled when the observation budget expires, so without this
    the real ``hook_task`` keeps running: its MCP session, its socket and -
    for the stdio transport, which has no read deadline of its own - its child
    process stay alive for as long as the remote stays silent. One stalled
    connector then costs one task, one socket and one process *per approval*,
    with nothing to reclaim them.

    Cancelling here does not violate the "never cancel a started write"
    invariant. That invariant protects the *observation* of a write that might
    still answer; by this point the full observation budget has already been
    spent and the caller has already been handed its outcome (``TimeoutError``
    with the durable dispatch boundary acknowledged, which
    :meth:`MCPApprovalGateTool.resume_user_interaction` projects as
    ``dispatch_unknown``). Cancelling cannot change that outcome, and the write
    may still have landed either way - which is exactly what
    ``dispatch_unknown`` already says. What cancelling does change is that the
    transport unwinds instead of leaking.

    The drain runs as its own task because both call sites are ``finally``
    paths that may be unwinding under an active ``CancelledError``; awaiting
    there would either be interrupted immediately or make the enclosing
    cancellation un-cancellable. It is bounded by
    ``_DISPATCH_CLEANUP_SECONDS`` so a connector that ignores cancellation
    cannot park the cleanup either, and the done-callback attached first keeps
    a late failure traceable to this interaction instead of surfacing as a
    bare "Task exception was never retrieved".
    """

    logger.warning(
        "MCP approval resume hook is still dispatching after %.0fs; "
        "cancelling it - the connector write may still have landed.",
        dispatch_observe_seconds,
    )

    def _retrieve(task: "asyncio.Future[Any]") -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.warning(
                "MCP approval resume hook failed after the gate stopped "
                "observing it: %r",
                error,
            )

    hook_task.add_done_callback(_retrieve)
    hook_task.cancel()
    cleanup = asyncio.ensure_future(
        _drain(hook_task, timeout=_DISPATCH_CLEANUP_SECONDS)
    )
    _ORPHANED_DISPATCH_CLEANUPS.add(cleanup)
    cleanup.add_done_callback(_ORPHANED_DISPATCH_CLEANUPS.discard)


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
        scoped_registration: _RegisteredHooks | None = None,
        approval_supported: bool = True,
        invalid_scope: bool = False,
        transient_fence: bool = False,
        raw_tool_name: str | None = None,
    ) -> None:
        self._target = target
        self._connector_ref = connector_ref
        self._scoped_registration = scoped_registration
        self._approval_supported = approval_supported
        self._invalid_scope = invalid_scope
        self._transient_fence = transient_fence
        self._raw_tool_name = raw_tool_name

    def _registration(
        self, context: ToolCallExecutionContext | None
    ) -> _RegisteredHooks | None:
        if self._scoped_registration is not None:
            current = _registration_for(self._scoped_registration.handle.task_source)
            if current is None or current.handle != self._scoped_registration.handle:
                return None
            return current
        registration = _registration_for(context.task_source if context else None)
        return (
            registration
            if registration is not None and registration.tool_matcher is None
            else None
        )

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
        if (
            self._scoped_registration is None
            and not self._invalid_scope
            and not self._transient_fence
            and not _has_legacy_registrations()
        ):
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

    @property
    def category(self) -> Any:
        """Delegate the tool category (MCP tools set it as an attribute)."""

        return getattr(self._target, "category", None)

    def __getattr__(self, name: str) -> Any:
        """Delegate optional runtime capabilities to the wrapped tool.

        Mirrors ``OutputFilteredToolWrapper.__getattr__``. Call sites across the
        codebase duck-type tools with ``getattr(tool, "<cap>", None)`` and skip
        the feature when it is absent, so without this a wrapped tool loses
        those capabilities *silently* rather than loudly - e.g. ``source_server``
        (mcp_tools.py), which MCP tools set as a plain attribute.

        Only reached for attributes normal lookup did not find, so every
        property and method defined above keeps priority. Private names are
        refused so this never masks a missing internal during ``__init__``.

        The class-attribute guard is not decoration. Python falls back to
        ``__getattr__`` when a *property getter raises AttributeError*, not
        only when the attribute is absent, so a naive delegation silently
        swallows a fault inside one of this class's own properties and answers
        with the wrapped tool's raw value instead. That is exactly how a broken
        ``metadata`` would start returning the target's UNDEGRADED metadata -
        re-enabling concurrent batching for a gated tool - with no error
        anywhere. If the name is defined on the class, the descriptor ran and
        failed, so surface that rather than paper over it.
        """

        if name.startswith("_"):
            raise AttributeError(name)
        if hasattr(type(self), name):
            raise AttributeError(
                f"{type(self).__name__}.{name} raised AttributeError; refusing "
                "to mask it by delegating to the wrapped tool"
            )
        try:
            target = object.__getattribute__(self, "_target")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(target, name)

    def is_async(self) -> bool:
        return True

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        if self._invalid_scope or self._transient_fence:
            raise RuntimeError(f"MCP tool {self.name} has no usable approval authority")
        context = current_tool_call_execution_context()
        registration = self._registration(context)
        if self._scoped_registration is not None and registration is None:
            raise RuntimeError(f"MCP tool {self.name} has a stale approval scope")
        if registration is None:
            # Unregistered or absent source: not this gate's business. Dispatch
            # exactly as the unwrapped tool would.
            return self._target.run_json_sync(args)
        # The call's own source IS registered, so it must be evaluated. The
        # gate hooks are async and there is no running loop to await them on.
        raise RuntimeError(f"MCP tool {self.name} requires async approval evaluation")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        if self._invalid_scope or self._transient_fence:
            return dict(_GATE_FAILURE)
        context = current_tool_call_execution_context()
        registration = self._registration(context)
        if self._scoped_registration is not None and registration is None:
            return dict(_GATE_FAILURE)
        if registration is None:
            # Unregistered or absent source: transparent pass-through.
            return await self._target.run_json_async(args)
        if self._scoped_registration is not None and (
            context is None
            or (
                context.task_source != registration.handle.task_source
                and not (not self._approval_supported and context.task_source is None)
            )
        ):
            return dict(_GATE_FAILURE)
        # Scoped registrations classify first. Only the gate hook below may
        # create a durable prompt; the classifier must remain side-effect free.
        if context is None or self._connector_ref is None:
            return dict(_GATE_FAILURE)
        try:
            call = GatedCall.from_arguments(
                connector_ref=self._connector_ref,
                tool_name=self.name,
                arguments=args,
                execution_context=context,
                raw_tool_name=self._raw_tool_name,
            )
            protected = True
            if registration.classifier is not None:
                protected = await _call_classifier(
                    registration.classifier,
                    registration.timeout_seconds,
                    call,
                )
                if type(protected) is not bool:
                    raise TypeError("approval classifier must return bool")
        except Exception:
            logger.warning(
                "MCP approval classifier failed closed for tool=%s task_id=%s "
                "tool_call_id=%s",
                self.name,
                context.task_id,
                context.tool_call_id,
                exc_info=True,
            )
            return dict(_GATE_FAILURE)

        if not protected:
            return await self._target.run_json_async(args)
        if not self._approval_supported:
            return dict(_NESTED_APPROVAL_UNAVAILABLE)
        if not context.is_complete():
            return dict(_GATE_FAILURE)
        if context.pattern != "react":
            # Checked BEFORE the hook: a decision issued for an unsupported
            # pattern would leave the host holding a persisted, clickable
            # approval prompt that nothing can ever resume.
            return dict(_UNSUPPORTED_PATTERN)

        try:
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
            # ``call.arguments``, NOT the caller's ``args``. This is deliberate
            # and load-bearing: the canonical snapshot is the exact payload the
            # hook was shown and that ``arguments_sha256`` covers, so
            # dispatching it is what makes "the approved call is the executed
            # call" true even if the hook mutated the caller's dict after
            # deciding. Dispatching ``args`` instead would reopen that gap to
            # buy back only tuple/non-str-key fidelity, which cannot arise on
            # the MCP path (arguments reach it as parsed JSON), and would not
            # help with NaN/Infinity: ``from_arguments`` rejects those at
            # snapshot time, before this branch is reached.
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
            # No ``options`` here: ``confirm`` is not in TYPES_REQUIRING_OPTIONS
            # (interaction_types.py), the write-side validator refuses options
            # on it outright (``options_forbidden``), and the renderer draws a
            # confirm as a boolean switch that ignores them. A host that needs
            # a richer prompt renders its own (the Slack approval UI does) and
            # only needs a stable interaction to resume.
            "interactions": [
                {
                    "type": "confirm",
                    "field": "approve",
                    "label": "Approve",
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
        # What this wrapper itself recorded at pause time, carried durably in
        # the interaction id. ``None`` means the gate never gated this call.
        gated = _gated_interaction_source(interaction_id)
        if self._transient_fence and gated is not None:
            fence_source = (
                self._scoped_registration.handle.task_source
                if self._scoped_registration is not None
                else None
            )
            if (
                gated[0] == fence_source
                and context is not None
                and context.task_source == fence_source
            ):
                # Raising keeps ReAct's existing callback-delivery transaction
                # intact: the pending response is not popped or terminally
                # projected, so a later turn may retry after authority recovers.
                raise MCPApprovalGateTemporarilyUnavailable(
                    "MCP approval authority is temporarily unavailable"
                )
        registration = self._registration(context)

        if (
            self._invalid_scope
            or registration is None
            or gated is None
            or gated[0] != registration.handle.task_source
            or (
                self._scoped_registration is not None
                and (
                    context is None
                    or context.task_source != registration.handle.task_source
                )
            )
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
        used = False

        async def execute(arguments: Mapping[str, Any]) -> Any:
            nonlocal used
            if used:
                raise RuntimeError("approval replay executor is no longer available")
            used = True
            canonical = GatedCall.from_arguments(
                connector_ref=connector_ref,
                tool_name=self.name,
                arguments=arguments,
                execution_context=context,
                raw_tool_name=self._raw_tool_name,
            )
            replay = MCPApprovalReplayContext(
                interaction_id=host_interaction_id,
                arguments_sha256=canonical.arguments_sha256,
                execution_context=context,
            )
            token = _CURRENT_REPLAY_CONTEXT.set(replay)
            try:
                # Legacy hooks enter the same conservative boundary immediately
                # before connector dispatch. Durable hooks acknowledge earlier,
                # immediately after their marker commits, closing the otherwise
                # unobservable marker-to-executor window.
                executor.acknowledge_dispatch()
                return await self._target.run_json_async(canonical.arguments)
            finally:
                _CURRENT_REPLAY_CONTEXT.reset(token)

        executor = MCPApprovalReplayExecutor(
            execute=execute,
            loop=asyncio.get_running_loop(),
        )

        try:
            result = await _call_resume_hook(
                registration.resume,
                registration.timeout_seconds,
                executor,
                dispatch_observe_seconds=registration.dispatch_observe_seconds,
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
            if executor.dispatch_acknowledged:
                return ToolInteractionSettlement.dispatch_unknown(
                    error=(
                        "The connector call may have reached the external system. "
                        "Automatic retry is disabled."
                    )
                )
            return ToolInteractionSettlement.failed(result=dict(_GATE_FAILURE))
        finally:
            executor._deactivate()

    async def setup(self, task_id: Optional[str] = None) -> None:
        await self._target.setup(task_id)

    async def teardown(self, task_id: Optional[str] = None) -> None:
        await self._target.teardown(task_id)

    async def save_state_json(self) -> Mapping[str, Any]:
        return await self._target.save_state_json()

    async def load_state_json(self, state: Mapping[str, Any]) -> None:
        await self._target.load_state_json(state)


def _raw_mcp_tool_name(tool: AbstractBaseTool) -> str | None:
    """Recover the MCP-advertised name without invoking dynamic attributes."""

    current: Any = tool
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        try:
            attributes = object.__getattribute__(current, "__dict__")
        except (AttributeError, TypeError):
            break
        if not isinstance(attributes, dict):
            break

        mcp_tool = attributes.get("mcp_tool")
        try:
            mcp_attributes = object.__getattribute__(mcp_tool, "__dict__")
        except (AttributeError, TypeError):
            mcp_attributes = None
        raw_name = (
            mcp_attributes.get("name") if isinstance(mcp_attributes, dict) else None
        )
        if isinstance(raw_name, str) and raw_name:
            return raw_name
        current = attributes.get("_target")
    return None


def gate_mcp_tools(
    tools: Sequence[AbstractBaseTool],
    *,
    connection: Mapping[str, Any] | None = None,
    connector_ref: ConnectorRef | None = None,
    approval_scope: MCPApprovalGateScope | None = None,
    approval_resolution: MCPApprovalGateResolution | None = None,
    approval_supported: bool = True,
) -> list[AbstractBaseTool]:
    """Wrap exactly the MCP tools governed by explicit task authority.

    No scope preserves legacy source-wide behavior. A valid scope wraps only
    matcher-selected tools. Rollback fences use the live registration matcher
    but keep approved resume pending; stale or forged authority closes selected
    tools. UNKNOWN without a live matcher cannot safely infer protected targets
    and therefore never expands into an all-tools outage.
    """

    if approval_resolution is not None:
        if not isinstance(approval_resolution, MCPApprovalGateResolution):
            raise TypeError("approval_resolution must be an MCPApprovalGateResolution")
        if approval_scope is not None:
            raise ValueError(
                "approval_scope and approval_resolution are mutually exclusive"
            )
        if approval_resolution.state == "off":
            return list(tools)
        approval_scope = approval_resolution.approval_scope
    if type(approval_supported) is not bool:
        raise TypeError("approval_supported must be bool")

    ref = connector_ref or _connector_ref(connection or {})
    if approval_scope is None:
        return [
            MCPApprovalGateTool(
                tool,
                connector_ref=ref,
                approval_supported=approval_supported,
                raw_tool_name=_raw_mcp_tool_name(tool),
            )
            for tool in tools
        ]

    if not isinstance(approval_scope, MCPApprovalGateScope):
        return [
            MCPApprovalGateTool(
                tool,
                connector_ref=ref,
                approval_supported=approval_supported,
                invalid_scope=True,
                raw_tool_name=_raw_mcp_tool_name(tool),
            )
            for tool in tools
        ]

    live_registration = _scoped_registration_for_source(approval_scope.task_source)
    transient_fence = approval_scope.registration_id == _ROLLBACK_FENCE_REGISTRATION_ID
    registration = _registration_for_scope(approval_scope)

    if live_registration is None:
        if transient_fence:
            # UNKNOWN may not disable unrelated reads when no live matcher can
            # identify the protected subset. Higher layers retain the UNKNOWN
            # resolution and retry authority discovery on the next turn.
            return list(tools)
        return [
            MCPApprovalGateTool(
                tool,
                connector_ref=ref,
                approval_supported=approval_supported,
                invalid_scope=True,
                raw_tool_name=_raw_mcp_tool_name(tool),
            )
            for tool in tools
        ]

    if ref is None:
        if transient_fence:
            return list(tools)
        return [
            MCPApprovalGateTool(
                tool,
                connector_ref=None,
                scoped_registration=live_registration,
                approval_supported=approval_supported,
                invalid_scope=not transient_fence,
                transient_fence=transient_fence,
                raw_tool_name=_raw_mcp_tool_name(tool),
            )
            for tool in tools
        ]

    matcher = live_registration.tool_matcher
    if matcher is None:  # pragma: no cover - guarded by helper invariant
        raise AssertionError("scoped registration lost its tool matcher")

    wrapped: list[AbstractBaseTool] = []
    for tool in tools:
        raw_tool_name = _raw_mcp_tool_name(tool)
        if raw_tool_name is None:
            logger.warning(
                "MCP approval raw tool name unavailable; failing closed for "
                "tool=%s connector=%s",
                tool.name,
                ref,
            )
            wrapped.append(
                MCPApprovalGateTool(
                    tool,
                    connector_ref=ref,
                    scoped_registration=live_registration,
                    approval_supported=approval_supported,
                    invalid_scope=not transient_fence,
                    transient_fence=transient_fence,
                )
            )
            continue
        try:
            matches = matcher(ref, raw_tool_name)
            if type(matches) is not bool:
                raise TypeError("approval tool matcher must return bool")
        except Exception:
            logger.warning(
                "MCP approval tool matcher failed closed for tool=%s connector=%s",
                tool.name,
                ref,
                exc_info=True,
            )
            wrapped.append(
                MCPApprovalGateTool(
                    tool,
                    connector_ref=ref,
                    scoped_registration=live_registration,
                    approval_supported=approval_supported,
                    invalid_scope=not transient_fence,
                    transient_fence=transient_fence,
                    raw_tool_name=raw_tool_name,
                )
            )
            continue
        if not matches:
            wrapped.append(tool)
            continue
        wrapped.append(
            MCPApprovalGateTool(
                tool,
                connector_ref=ref,
                scoped_registration=live_registration,
                approval_supported=approval_supported,
                invalid_scope=registration is None and not transient_fence,
                transient_fence=transient_fence,
                raw_tool_name=raw_tool_name,
            )
        )
    return wrapped
