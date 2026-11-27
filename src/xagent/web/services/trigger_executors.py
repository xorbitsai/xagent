"""Host-executed trigger runs.

By default the engine turns every accepted trigger event into a hidden agent
task and starts it. A host application can instead claim the runs of triggers
whose ``executor`` field names a kind it registered here. The engine still
owns verification, audit, rate limiting, deduplication and run bookkeeping;
only the execution step is handed to the host, exactly once per run.

The module is deliberately generic: a kind is an opaque string chosen by the
host, and the executor receives the normalized event, not provider details.
"""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class HostTriggerRun:
    """Everything a host executor needs to run one claimed trigger run."""

    run_id: int
    trigger_id: int
    user_id: int
    executor: str
    trigger_type: str
    provider: str | None
    resource_id: str | None
    event_type: str | None
    source_event_id: str | None
    received_at: datetime | None
    payload: dict[str, Any]
    prompt: str
    test: bool


# Return normally -> the run is recorded COMPLETED; raise -> FAILED with the
# exception text. The callable may run for a long time; the engine never blocks
# the provider acknowledgement on it.
HostTriggerExecutor = Callable[[HostTriggerRun], Awaitable[None]]

_lock = threading.Lock()
_executors: dict[str, HostTriggerExecutor] = {}


class TriggerExecutorAlreadyRegisteredError(ValueError):
    """A different executor is already registered for this kind."""


def register_trigger_executor(
    kind: str, executor: HostTriggerExecutor, *, replace: bool = False
) -> None:
    """Register ``executor`` for triggers whose ``executor`` field is ``kind``."""
    kind = (kind or "").strip()
    if not kind:
        raise ValueError("Trigger executor kind must be a non-empty string")
    with _lock:
        existing = _executors.get(kind)
        if existing is not None and existing is not executor and not replace:
            raise TriggerExecutorAlreadyRegisteredError(
                f"Trigger executor already registered for kind {kind!r}"
            )
        _executors[kind] = executor


def unregister_trigger_executor(kind: str) -> None:
    with _lock:
        _executors.pop(kind, None)


def get_trigger_executor(kind: str | None) -> HostTriggerExecutor | None:
    if not kind:
        return None
    with _lock:
        return _executors.get(kind)


def registered_trigger_executor_kinds() -> list[str]:
    with _lock:
        return sorted(_executors)
