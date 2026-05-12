from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ExecutionStatus(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_FOR_USER = "waiting_for_user"
    INTERRUPTED = "interrupted"
    REPLANNING = "replanning"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _status_value(status: ExecutionStatus | str) -> str:
    return status.value if isinstance(status, ExecutionStatus) else str(status)


@dataclass
class ExecutionFrame:
    frame_id: str
    root_execution_id: str
    pattern_type: str
    status: ExecutionStatus | str
    context: dict[str, Any]
    pattern_state: dict[str, Any]
    parent_frame_id: str | None = None
    children: list[str] = field(default_factory=list)
    active_child_id: str | None = None
    tool_calls: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "parent_frame_id": self.parent_frame_id,
            "root_execution_id": self.root_execution_id,
            "pattern_type": self.pattern_type,
            "status": _status_value(self.status),
            "context": dict(self.context),
            "pattern_state": dict(self.pattern_state),
            "children": list(self.children),
            "active_child_id": self.active_child_id,
            "tool_calls": dict(self.tool_calls),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionFrame":
        return cls(
            frame_id=str(data["frame_id"]),
            parent_frame_id=data.get("parent_frame_id"),
            root_execution_id=str(data["root_execution_id"]),
            pattern_type=str(data["pattern_type"]),
            status=str(data["status"]),
            context=dict(data.get("context", {})),
            pattern_state=dict(data.get("pattern_state", {})),
            children=list(data.get("children", [])),
            active_child_id=data.get("active_child_id"),
            tool_calls=dict(data.get("tool_calls", {})),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ExecutionSnapshot:
    root_execution_id: str
    status: ExecutionStatus | str
    frames: dict[str, ExecutionFrame]
    active_frame_ids: list[str] = field(default_factory=list)
    control_state: dict[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=_utcnow)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_execution_id": self.root_execution_id,
            "status": _status_value(self.status),
            "frames": {
                frame_id: frame.to_dict() for frame_id, frame in self.frames.items()
            },
            "active_frame_ids": list(self.active_frame_ids),
            "control_state": dict(self.control_state),
            "updated_at": self.updated_at.isoformat(),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionSnapshot":
        frames: dict[str, ExecutionFrame] = {}
        for frame_id, frame_payload in data.get("frames", {}).items():
            frame = ExecutionFrame.from_dict(frame_payload)
            if frame.frame_id != frame_id:
                raise ValueError(f"Frame key mismatch: {frame_id} vs {frame.frame_id}")
            frames[frame_id] = frame
        return cls(
            root_execution_id=str(data["root_execution_id"]),
            status=str(data["status"]),
            frames=frames,
            active_frame_ids=list(data.get("active_frame_ids", [])),
            control_state=dict(data.get("control_state", {})),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at")
            else _utcnow(),
            schema_version=int(data.get("schema_version", 1)),
        )
