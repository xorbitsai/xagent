from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .trace import TraceAction, TraceCategory, TraceEventType, TraceScope

CHECKPOINT_TYPE = "agent_execution_checkpoint"
LEGACY_CHECKPOINT_TYPES = frozenset({"agent_v2_execution_checkpoint"})
READABLE_CHECKPOINT_TYPES = frozenset({CHECKPOINT_TYPE, *LEGACY_CHECKPOINT_TYPES})
CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_READER_METHODS = (
    "load_latest_checkpoint",
    "get_latest_checkpoint",
    "latest_checkpoint",
)


CHECKPOINT_EVENT_TYPE = TraceEventType(
    TraceScope.SYSTEM,
    TraceAction.UPDATE,
    TraceCategory.GENERAL,
)


def checkpoint_execution_id(data: dict[str, Any]) -> str:
    """Canonical execution id of a checkpoint event payload.

    ``_event_payload`` writes ``root_execution_id`` and ``execution_id``
    identically today; this helper is the single place that defines the
    precedence (root first, then the flat field, then the snapshot's own id)
    so readers, pruning, and the storage encoder cannot drift apart.
    """
    snapshot = data.get("snapshot")
    nested = snapshot.get("execution_id") if isinstance(snapshot, dict) else None
    return str(
        data.get("root_execution_id") or data.get("execution_id") or nested or ""
    )


class CheckpointPersistenceError(RuntimeError):
    """Raised when a checkpoint cannot be durably persisted."""


class CheckpointReadError(RuntimeError):
    """Base for checkpoint read failures that must not collapse to absence.

    ``None`` from a checkpoint reader means "queried successfully, nothing
    found" — an authoritative fact callers may act on (e.g. build a fresh
    context). Anything that prevented that determination raises one of the
    subclasses below instead, so a transient or refused read can never be
    mistaken for a checkpoint that genuinely does not exist.
    """


class CheckpointUnavailableError(CheckpointReadError):
    """Raised when a checkpoint read could not be completed.

    Covers infrastructure failures only: session checkout, query
    execution, and generic per-row decode errors such as a failed blob
    prefetch. Rows that decode as permanently unreadable are classified
    by the corrupt error once the matching set is exhausted.
    """


class CheckpointCorruptError(CheckpointReadError):
    """Raised when matching checkpoint rows exist but none are usable.

    Distinct from ``CheckpointUnavailableError``: the read completed and
    the candidate set was proven exhausted, but every row is permanently
    undecodable or shaped without a payload. This is a terminal state, not
    a retryable one.
    """


class CheckpointAccessRefusedError(CheckpointReadError):
    """Raised when a reader is not authoritative for the requested partition.

    The checkpoint may exist, but this reader's partition (run binding,
    build scope) is not the one allowed to observe it right now. Distinct
    from absence: callers must not treat a refusal as "no checkpoint" and
    fall back to building fresh state.

    ``reason`` discriminates *why* the read was refused, so consumers can
    report an accurate message instead of one generic sentence for every
    case: ``"lease_mismatch"`` (an active lease exists but is not bound to
    this reader), ``"active_run"`` (a different run is in progress under
    its own lease), ``"superseded_legacy"`` (a tagged run has already
    superseded the untagged/legacy partition this reader is confined to),
    or ``"run_provenance_unavailable"`` (the checkpoint pointer names a row
    that exists and decodes fine but has no run-partition field to check,
    and no other readable row was found either, so this reader cannot prove
    it is allowed to observe it). Defaults to ``"active_run"``, the most
    common case, so existing call sites that do not pass ``reason`` keep
    working unchanged.
    """

    def __init__(self, message: str, *, reason: str = "active_run") -> None:
        super().__init__(message)
        self.reason = reason


async def read_latest_checkpoint_payload(
    reader: Any,
    execution_id: str,
) -> Any:
    """Read through the first checkpoint capability exposed by ``reader``.

    Method names are compatibility aliases, not independent data sources. Once
    a reader exposes the preferred available method, its result is
    authoritative, including ``None`` for "no checkpoint". Falling through
    after that valid empty result can invoke deprecated aliases or combine
    inconsistent reader implementations.
    """

    for method_name in CHECKPOINT_READER_METHODS:
        method = getattr(reader, method_name, None)
        if not callable(method):
            continue
        payload = method(execution_id)
        if inspect.isawaitable(payload):
            payload = await payload
        return payload
    return None


@dataclass
class TraceCheckpointStore:
    """Durable checkpoint adapter backed by the tracer event pipeline.

    This object intentionally exposes the same method names that AgentRunner and
    PatternRuntime already probe for: checkpoint/write_checkpoint for writes and
    load_latest_checkpoint for reads.
    """

    tracer: Any
    require_persisted: bool = True

    async def checkpoint(self, **payload: Any) -> str | None:
        return await self.save(payload)

    async def write_checkpoint(self, payload: dict[str, Any]) -> str | None:
        return await self.save(payload)

    async def trace_event(self, *args: Any, **kwargs: Any) -> Any:
        """Forward non-checkpoint trace events to the wrapped tracer."""
        trace_event = getattr(self.tracer, "trace_event", None)
        if not callable(trace_event):
            return None
        result = trace_event(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def save(self, payload: dict[str, Any]) -> str | None:
        execution_id = self._execution_id(payload)
        event_payload = self._event_payload(payload, execution_id=execution_id)

        event_id = await self._call_checkpoint_writer(payload, event_payload)
        if event_id is None:
            raise CheckpointPersistenceError(
                "Tracer does not expose a durable checkpoint write API."
            )
        return str(event_id)

    async def load_latest_checkpoint(
        self,
        execution_id: str,
    ) -> dict[str, Any] | None:
        payload = await read_latest_checkpoint_payload(self.tracer, execution_id)
        return self._unwrap_checkpoint_payload(payload)

    def get_latest_checkpoint(self, execution_id: str) -> dict[str, Any] | None:
        raise NotImplementedError("Use async load_latest_checkpoint().")

    def latest_checkpoint(self, execution_id: str) -> dict[str, Any] | None:
        raise NotImplementedError("Use async load_latest_checkpoint().")

    async def _call_checkpoint_writer(
        self,
        payload: dict[str, Any],
        event_payload: dict[str, Any],
    ) -> str | None:
        for method_name in ("save_checkpoint", "checkpoint", "write_checkpoint"):
            method = getattr(self.tracer, method_name, None)
            if not callable(method):
                continue
            result = (
                method(payload)
                if method_name == "write_checkpoint"
                else method(**payload)
            )
            if inspect.isawaitable(result):
                result = await result
            return str(result) if result is not None else None

        trace_event = getattr(self.tracer, "trace_event", None)
        if callable(trace_event):
            if self.require_persisted and not self._supports_kwarg(
                trace_event,
                "require_persisted",
            ):
                raise CheckpointPersistenceError(
                    "Tracer.trace_event() cannot guarantee checkpoint persistence."
                )
            result = trace_event(
                self._checkpoint_trace_event_type(trace_event),
                task_id=event_payload["root_execution_id"],
                data=event_payload,
                require_persisted=self.require_persisted,
            )
            if inspect.isawaitable(result):
                result = await result
            return str(result) if result is not None else None

        return None

    def _event_payload(
        self,
        payload: dict[str, Any],
        *,
        execution_id: str,
    ) -> dict[str, Any]:
        metadata = payload.get("metadata")
        sequence = metadata.get("sequence") if isinstance(metadata, dict) else None
        return {
            "checkpoint_type": CHECKPOINT_TYPE,
            "snapshot_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "root_execution_id": execution_id,
            "execution_id": execution_id,
            "sequence": sequence,
            "status": payload.get("status"),
            "label": payload.get("label"),
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "snapshot": dict(payload),
        }

    def _validate_snapshot(self, container: dict[str, Any]) -> dict[str, Any]:
        """Return the readable checkpoint's snapshot, or raise if absent.

        A container whose ``checkpoint_type`` is readable but that carries no
        ``snapshot`` dict claims to be a checkpoint while holding no payload
        -- corrupt, not absent.
        """
        snapshot = container.get("snapshot")
        if isinstance(snapshot, dict):
            return dict(snapshot)
        raise CheckpointCorruptError(
            "Checkpoint payload has a readable checkpoint_type but no snapshot."
        )

    def _unwrap_checkpoint_payload(self, payload: Any) -> dict[str, Any] | None:
        # ``None`` is the reader's authoritative "no checkpoint" -- pass it
        # through unchanged. Anything else that isn't a recognized shape is
        # corrupt, not absent: a caller must not treat a malformed payload
        # as "build fresh state".
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise CheckpointCorruptError(
                "Checkpoint reader returned a non-dict, non-None payload."
            )
        if payload.get("checkpoint_type") in READABLE_CHECKPOINT_TYPES:
            return self._validate_snapshot(payload)
        data = payload.get("data")
        if (
            isinstance(data, dict)
            and data.get("checkpoint_type") in READABLE_CHECKPOINT_TYPES
        ):
            return self._validate_snapshot(data)
        if payload.get("type") == "checkpoint" or "context" in payload:
            return dict(payload)
        raise CheckpointCorruptError(
            "Checkpoint payload shape is not recognized by any reader."
        )

    def _execution_id(self, payload: dict[str, Any]) -> str:
        execution_id = payload.get("execution_id")
        if not execution_id:
            raise CheckpointPersistenceError(
                "Checkpoint payload is missing execution_id."
            )
        return str(execution_id)

    def _supports_kwarg(self, method: Any, name: str) -> bool:
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return False
        return name in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    def _checkpoint_trace_event_type(self, trace_event: Any) -> Any:
        del trace_event
        return CHECKPOINT_EVENT_TYPE
