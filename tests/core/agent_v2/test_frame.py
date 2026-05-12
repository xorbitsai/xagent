from __future__ import annotations

from xagent.core.agent_v2 import ExecutionFrame, ExecutionSnapshot, ExecutionStatus


def test_execution_snapshot_roundtrip() -> None:
    root = ExecutionFrame(
        frame_id="exec-1:dag",
        root_execution_id="exec-1",
        pattern_type="dag",
        status=ExecutionStatus.RUNNING,
        context={"execution_id": "exec-1"},
        pattern_state={"status": "running"},
        children=["exec-1:dag_step:step_1"],
        active_child_id="exec-1:dag_step:step_1",
    )
    child = ExecutionFrame(
        frame_id="exec-1:dag_step:step_1",
        parent_frame_id="exec-1:dag",
        root_execution_id="exec-1",
        pattern_type="react",
        status=ExecutionStatus.INTERRUPTED,
        context={"execution_id": "exec-1_child"},
        pattern_state={"pending_tool_calls": []},
        metadata={"dag_step_id": "step_1"},
    )
    snapshot = ExecutionSnapshot(
        root_execution_id="exec-1",
        status=ExecutionStatus.INTERRUPTED,
        frames={root.frame_id: root, child.frame_id: child},
        active_frame_ids=[root.frame_id, child.frame_id],
        control_state={"planned_user_message_count": 1},
    )

    restored = ExecutionSnapshot.from_dict(snapshot.to_dict())

    assert restored.root_execution_id == "exec-1"
    assert restored.status == "interrupted"
    assert restored.frames[root.frame_id].active_child_id == child.frame_id
    assert restored.frames[child.frame_id].metadata["dag_step_id"] == "step_1"
