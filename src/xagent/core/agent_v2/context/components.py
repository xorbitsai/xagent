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
}


def clone_component(component: ExecutionComponent) -> ExecutionComponent:
    return component.clone()
