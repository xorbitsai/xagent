from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class ExecutionComponent(Protocol):
    def clone(self) -> "ExecutionComponent": ...

    def to_dict(self) -> dict[str, Any]: ...


@dataclass
class WorkspaceComponent:
    """Sandbox and workspace state for an execution."""

    workspace_id: str | None = None
    workspace_path: str | None = None
    cwd: str | None = None
    state: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "WorkspaceComponent":
        return WorkspaceComponent(
            workspace_id=self.workspace_id,
            workspace_path=self.workspace_path,
            cwd=self.cwd,
            state=copy.deepcopy(self.state),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "workspace_path": self.workspace_path,
            "cwd": self.cwd,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkspaceComponent":
        return cls(
            workspace_id=data.get("workspace_id"),
            workspace_path=data.get("workspace_path"),
            cwd=data.get("cwd"),
            state=data.get("state", {}),
        )


@dataclass
class MemoryComponent:
    """Memory session state for an execution."""

    session_id: str | None = None
    snapshot: dict[str, Any] | None = None

    def clone(self) -> "MemoryComponent":
        snapshot = (
            dict(self.snapshot) if isinstance(self.snapshot, dict) else self.snapshot
        )
        return MemoryComponent(session_id=self.session_id, snapshot=snapshot)

    def to_dict(self) -> dict[str, Any]:
        return {"session_id": self.session_id, "snapshot": self.snapshot}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryComponent":
        return cls(
            session_id=data.get("session_id"),
            snapshot=data.get("snapshot"),
        )


@dataclass
class SpillRegistryComponent:
    """Files this execution stored oversized tool results into."""

    records: list[dict[str, Any]] = field(default_factory=list)

    def clone(self) -> "SpillRegistryComponent":
        return SpillRegistryComponent(records=copy.deepcopy(self.records))

    def to_dict(self) -> dict[str, Any]:
        return {"records": self.records}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpillRegistryComponent":
        raw = data.get("records")
        return cls(
            records=[r for r in raw if isinstance(r, dict)]
            if isinstance(raw, list)
            else []
        )


@dataclass
class LoadedToolsComponent:
    """Deferred connector tools whose schemas this execution has loaded.

    ``names`` is the set whose schemas are sent, in load order, so the tool
    list only ever grows at its end. ``pending`` holds names a load_tools call
    asked for in the current model response: they become part of ``names`` at
    the start of the next iteration, so a call in the same response as its
    load is refused instead of running with arguments written blind.
    ``last_scanned_at`` is the timestamp of the newest user message already
    scanned for connector names to pre-load. A timestamp, not a count, so a
    compaction that shrinks the transcript cannot hide a later message.
    """

    names: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    last_scanned_at: str | None = None

    def clone(self) -> "LoadedToolsComponent":
        return LoadedToolsComponent(
            names=list(self.names),
            pending=list(self.pending),
            last_scanned_at=self.last_scanned_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "pending": list(self.pending),
            "last_scanned_at": self.last_scanned_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoadedToolsComponent":
        def _names(key: str) -> list[str]:
            raw = data.get(key)
            if not isinstance(raw, list):
                return []
            return list(dict.fromkeys(n for n in raw if isinstance(n, str)))

        scanned = data.get("last_scanned_at")
        return cls(
            names=_names("names"),
            pending=_names("pending"),
            last_scanned_at=scanned if isinstance(scanned, str) else None,
        )


@dataclass
class GenericComponent:
    """Fallback component for unknown serialized component payloads."""

    data: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "GenericComponent":
        return GenericComponent(data=dict(self.data))

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)


COMPONENT_LOADERS: dict[str, Callable[[dict[str, Any]], ExecutionComponent]] = {
    "workspace": WorkspaceComponent.from_dict,
    "memory": MemoryComponent.from_dict,
    "spilled_results": SpillRegistryComponent.from_dict,
    "loaded_tools": LoadedToolsComponent.from_dict,
}


def clone_component(component: ExecutionComponent) -> ExecutionComponent:
    return component.clone()
