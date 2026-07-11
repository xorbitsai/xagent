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
        for method_name in (
            "load_latest_checkpoint",
            "get_latest_checkpoint",
            "latest_checkpoint",
        ):
            method = getattr(self.tracer, method_name, None)
            if not callable(method):
                continue
            payload = method(execution_id)
            if inspect.isawaitable(payload):
                payload = await payload
            return self._unwrap_checkpoint_payload(payload)
        return None

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

    def _unwrap_checkpoint_payload(self, payload: Any) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        if payload.get("checkpoint_type") in READABLE_CHECKPOINT_TYPES:
            snapshot = payload.get("snapshot")
            return dict(snapshot) if isinstance(snapshot, dict) else None
        data = payload.get("data")
        if (
            isinstance(data, dict)
            and data.get("checkpoint_type") in READABLE_CHECKPOINT_TYPES
        ):
            snapshot = data.get("snapshot")
            return dict(snapshot) if isinstance(snapshot, dict) else None
        if payload.get("type") == "checkpoint" or "context" in payload:
            return dict(payload)
        return None

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
