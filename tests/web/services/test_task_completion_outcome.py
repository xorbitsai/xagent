"""Semantic completion survives settlement, reconnect, and later turns."""

import pytest
from sqlalchemy.orm import sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from tests.web.services.task_database_shared import task_id as task_id_fixture
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.services import task_coordinator_service as coordinator
from xagent.web.services import task_execution as execution
from xagent.web.services import task_lease_service as leases
from xagent.web.services import task_orchestrator as orchestrator
from xagent.web.services import task_stream_snapshot as snapshots
from xagent.web.services.client_error_messages import CLIENT_SAFE_TASK_FAILURE
from xagent.web.services.execution_result_projection import (
    project_execution_result_for_channel,
)
from xagent.web.services.managed_task_lease import (
    claim_managed_task_lease_isolated,
    finalize_managed_task_lease_result,
    finalize_managed_task_lease_result_isolated,
)
from xagent.web.services.task_command_execution import (
    _load_task_command_routing_snapshot,
)
from xagent.web.services.task_execution_controller import (
    TaskControlState,
    apply_task_control_transition,
    control_state_for_status,
    task_control_snapshot,
)

engine = engine_fixture
task_id = task_id_fixture


@pytest.mark.parametrize("route", ["initial", "resume", "channel"])
@pytest.mark.parametrize("storage_version", [1, 2])
@pytest.mark.parametrize(
    "result_status,success,final_status,outcome",
    [
        ("completed", True, TaskStatus.COMPLETED, outcome)
        for outcome in ("completed", "partial", "blocked", None, "invalid")
    ]
    + [
        ("failed", False, TaskStatus.FAILED, "partial"),
        ("waiting_for_user", True, TaskStatus.WAITING_FOR_USER, "partial"),
        ("interrupted", True, TaskStatus.PAUSED, "partial"),
    ],
)
def test_settled_outcome_and_next_run(
    engine,
    task_id,
    monkeypatch,
    route,
    storage_version,
    result_status,
    success,
    final_status,
    outcome,
):
    factory = sessionmaker(engine)
    monkeypatch.setattr(execution, "get_session_local", lambda: factory)
    monkeypatch.setattr(snapshots, "get_session_local", lambda: factory)
    with factory() as db:
        task = db.get(Task, task_id)
        task.conversation_storage_version = storage_version
        task.status = TaskStatus.COMPLETED
        task.control_state = TaskControlState.COMPLETED.value
        task.completion_outcome = "partial"
        uid = task.user_id
        db.commit()
        lease = leases.acquire_task_lease(
            db, task_id, runner_id="outcome-test", new_run=True
        )
        assert lease is not None
        db.refresh(task)
        assert task.completion_outcome is None
    result = {
        "status": result_status,
        "success": success,
        "output": "Delivered answer",
        "completion_outcome": outcome,
    }
    empty = execution._PreparedTaskFileOutputs((), (), ())
    if route == "initial":
        finalized = execution._finalize_task_execution_result_isolated(
            task_id=task_id,
            task_user_id=uid,
            pre_run_status=TaskStatus.RUNNING,
            result=result,
            expected_run_id=lease.run_id,
            task_lease=lease,
            resolved_scope_segments=(),
            prepared_outputs=empty,
        )
        transported = finalized.broadcast_meta["completion_outcome"]
    elif route == "resume":
        finalized = execution._finalize_resumed_task(
            task_id,
            status=result_status,
            success=success,
            output=result["output"],
            task_owner_user_id=uid,
            result=result,
            task_lease=lease,
            prepared_outputs=empty,
        )
        transported = finalized["completion_outcome"]
    else:
        projection = project_execution_result_for_channel(result)
        with factory() as db:
            assert finalize_managed_task_lease_result(
                db,
                lease,
                status=projection.task_status,
                assistant_content=projection.transcript_content,
                execution_result=result,
            )
        transported = projection.completion_outcome
    expected = (
        outcome
        if final_status == TaskStatus.COMPLETED and outcome != "invalid"
        else None
    )
    assert transported == expected
    with factory() as db:
        task = db.get(Task, task_id)
        assert task.status == final_status
        assert task.completion_outcome == expected
        if final_status == TaskStatus.COMPLETED and route != "channel":
            assert task.output == "Delivered answer"
    assert (
        snapshots.load_task_stream_snapshots([task_id])[0]["completion_outcome"]
        == expected
    )
    with factory() as db:
        task = db.get(Task, task_id)
        apply_task_control_transition(
            task, TaskControlState.RUNNING, status=TaskStatus.RUNNING, new_run=True
        )
        db.commit()
    assert (
        snapshots.load_task_stream_snapshots([task_id])[0]["completion_outcome"] is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["persisted", "channel"])
@pytest.mark.parametrize("outcome", ["completed", "partial", "blocked", None])
async def test_history_and_rest_read_same_outcome(
    engine, task_id, monkeypatch, outcome, route
):
    from xagent.web.api import chat, websocket
    from xagent.web.models import database
    from xagent.web.models.user import User

    factory = sessionmaker(engine)

    monkeypatch.setattr(database, "_SessionLocal", factory)
    # No cross-test response cache; the production cache is keyed by updated_at.
    monkeypatch.setattr(chat, "cache_get", lambda key: None)
    monkeypatch.setattr(websocket, "cache_get", lambda key: None)
    with factory() as db:
        task = db.get(Task, task_id)
        task.status = TaskStatus.COMPLETED
        task.completion_outcome = outcome
        uid = task.user_id
        db.commit()
        if route == "channel":
            lease = leases.acquire_task_lease(
                db, task_id, runner_id="channel-history-test", new_run=True
            )
            assert lease is not None
            assert finalize_managed_task_lease_result(
                db,
                lease,
                status=TaskStatus.COMPLETED,
                assistant_content="Channel answer",
                execution_result={"completion_outcome": outcome},
            )
        owner = db.get(User, uid)
        assert (await chat.get_task(task_id, db, owner))[
            "completion_outcome"
        ] == outcome
        assert (await chat.get_task_status(task_id, db, owner))[
            "completion_outcome"
        ] == outcome
        routing = _load_task_command_routing_snapshot(db, task)
        assert routing.task_info["completion_outcome"] == outcome
    history = websocket._load_historical_stream_snapshot_sync(
        task_id, actor_user_id=uid, actor_is_admin=False
    )
    info = next(
        event for event in history.events if event.get("event_type") == "task_info"
    )
    assert info["data"]["completion_outcome"] == outcome


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_version", [1, 2])
@pytest.mark.parametrize("first_outcome", ["partial", "blocked"])
async def test_channel_continuation_updates_outcome_and_rejects_late_result(
    engine, task_id, monkeypatch, storage_version, first_outcome
):
    from xagent.web.models import database
    from xagent.web.models.chat_message import TaskChatMessage

    factory = sessionmaker(engine)
    monkeypatch.setattr(database, "_SessionLocal", factory)
    monkeypatch.setattr(snapshots, "get_session_local", lambda: factory)
    with factory() as db:
        db.get(Task, task_id).conversation_storage_version = storage_version
        db.commit()

    for outcome in (first_outcome, "completed"):
        managed = await claim_managed_task_lease_isolated(task_id)
        assert managed is not None
        try:
            assert (
                snapshots.load_task_stream_snapshots([task_id])[0]["completion_outcome"]
                is None
            )
            assert await managed.finalize_result(
                status=TaskStatus.COMPLETED,
                assistant_content=outcome,
                execution_result={"success": True, "completion_outcome": outcome},
            )
        finally:
            await managed.close()
        assert (
            snapshots.load_task_stream_snapshots([task_id])[0]["completion_outcome"]
            == outcome
        )
        if outcome == first_outcome:
            first_lease = managed.lease

    # A delayed result from the earlier run cannot replace the new completion.
    assert not await finalize_managed_task_lease_result_isolated(
        first_lease,
        status=TaskStatus.COMPLETED,
        assistant_content="Late answer",
        execution_result={"completion_outcome": "blocked"},
    )
    with factory() as db:
        assert db.get(Task, task_id).completion_outcome == "completed"
        assert [
            message.content
            for message in db.query(TaskChatMessage)
            .filter_by(task_id=task_id, role="assistant")
            .order_by(TaskChatMessage.id)
        ] == [first_outcome, "completed"]


@pytest.mark.parametrize(
    "status",
    [
        TaskStatus.RUNNING,
        TaskStatus.PAUSED,
        TaskStatus.WAITING_FOR_USER,
        TaskStatus.FAILED,
    ],
)
def test_noncompleted_transition_clears_outcome(status):
    task = Task(id=1, status=TaskStatus.COMPLETED, completion_outcome="partial")
    apply_task_control_transition(task, TaskControlState.RUNNING, status=status)
    assert task.completion_outcome is None


@pytest.mark.parametrize(
    "status",
    [
        TaskStatus.RUNNING,
        TaskStatus.PAUSED,
        TaskStatus.WAITING_FOR_USER,
        TaskStatus.FAILED,
    ],
)
def test_persisted_noncompleted_transition_clears_outcome(engine, task_id, status):
    factory = sessionmaker(engine)
    with factory() as db:
        task = db.get(Task, task_id)
        task.status = TaskStatus.COMPLETED
        task.completion_outcome = "partial"
        db.commit()
        # Exercise the SQL UPDATE path without new_run also clearing the field.
        apply_task_control_transition(
            task, control_state_for_status(status), status=status
        )
        db.commit()
    with factory() as db:
        task = db.get(Task, task_id)
        assert task.status == status
        assert task.completion_outcome is None


@pytest.mark.parametrize("writer", ["coordinator", "orchestrator"])
def test_execution_start_writers_clear_outcome(engine, task_id, monkeypatch, writer):
    factory = sessionmaker(engine)
    with factory() as db:
        task = db.get(Task, task_id)
        task.status = TaskStatus.COMPLETED
        task.control_state = TaskControlState.COMPLETED.value
        task.completion_outcome = "partial"
        db.commit()
        if writer == "coordinator":
            snapshot = task_control_snapshot(task)
            lease = coordinator.acquire_task_lease_no_commit(
                db, task_id, runner_id="outcome-test"
            )
            assert lease is not None
            db.commit()
            db.refresh(task)
            # Owning a task alone does not start a new execution.
            assert task.completion_outcome == "partial"
            assert (
                coordinator.begin_task_execution_no_commit(
                    db, lease, expected=snapshot, new_run=True
                )
                is not None
            )
        else:
            monkeypatch.setattr(orchestrator, "enqueues_task_turns", lambda: False)
            orchestrator._accept_turn_no_commit(
                db,
                task_id,
                task.user_id,
                payload=orchestrator.TaskTurnPayload("Follow up"),
                kind=orchestrator.TurnKind.APPEND,
            )
        db.commit()
    with factory() as db:
        task = db.get(Task, task_id)
        assert task.status == TaskStatus.RUNNING
        assert task.completion_outcome is None


_PROJECTED_MODEL_FAILURE = (
    "Model provider call failed (403 provider_code_4204): Model is decommissioned"
)
_MODEL_ERROR_403 = {
    "kind": "access_denied",
    "status_code": 403,
    "provider_code": "provider_code_4204",
    "message": "Model is decommissioned",
}
_OWNER_DIAGNOSTIC_403 = (
    "OpenAI API error (403): Model is decommissioned | "
    "provider_raw=RAW_MARKER sk-live-ECHOEDKEY1234567890"
)
_OWNER_DIAGNOSTIC_403_STORED = (
    "OpenAI API error (403): Model is decommissioned | provider_raw=RAW_MARKER sk-***"
)
_RAW_400 = "OpenAI bad request (400): plan rejected RAW_MARKER"
_QUOTA_REASON = "Monthly quota reached for this agent."
_QUOTA_DETAILS = {"limit": 5, "used": 5}


def _model_failure_result(scenario: str) -> dict:
    if scenario == "forbidden-403":
        return {
            "status": "error",
            "success": False,
            "output": "All 1 patterns failed",
            "error": "All 1 patterns failed",
            "diagnostic_error": _OWNER_DIAGNOSTIC_403,
            "model_error": dict(_MODEL_ERROR_403),
        }
    if scenario == "bad-request-400":
        # Plan generation puts the provider text in ``output`` and ``error``.
        return {
            "status": "error",
            "success": False,
            "output": _RAW_400,
            "error": _RAW_400,
            "diagnostic_error": _RAW_400,
            "model_error": {
                "kind": "bad_request",
                "status_code": 400,
                "provider_code": "invalid_request",
                "message": "plan rejected RAW_MARKER",
            },
        }
    return {
        "status": "quota_exceeded",
        "success": False,
        "output": _QUOTA_REASON,
        "error": _QUOTA_REASON,
        "error_code": "quota_exceeded",
        "error_details": dict(_QUOTA_DETAILS),
        "diagnostic_error": _OWNER_DIAGNOSTIC_403,
        "model_error": dict(_MODEL_ERROR_403),
    }


def _finalize_failed_result(
    engine, task_id, monkeypatch, route, result, storage_version
):
    """Run one finalizer and return its client-facing fields as a dict."""
    factory = sessionmaker(engine)
    monkeypatch.setattr(execution, "get_session_local", lambda: factory)
    with factory() as db:
        task = db.get(Task, task_id)
        task.conversation_storage_version = storage_version
        task.status = TaskStatus.COMPLETED
        task.control_state = TaskControlState.COMPLETED.value
        uid = task.user_id
        db.commit()
        lease = leases.acquire_task_lease(
            db, task_id, runner_id="failure-test", new_run=True
        )
        assert lease is not None
    empty = execution._PreparedTaskFileOutputs((), (), ())
    if route == "initial":
        finalized = execution._finalize_task_execution_result_isolated(
            task_id=task_id,
            task_user_id=uid,
            pre_run_status=TaskStatus.RUNNING,
            result=result,
            expected_run_id=lease.run_id,
            task_lease=lease,
            resolved_scope_segments=(),
            prepared_outputs=empty,
        )
        fields = {
            "output": finalized.ai_response,
            "error_code": finalized.error_code,
            "error_details": finalized.error_details,
        }
    else:
        finalized = execution._finalize_resumed_task(
            task_id,
            status=str(result.get("status") or ""),
            success=bool(result.get("success", False)),
            output=str(result.get("output") or result.get("error") or ""),
            task_owner_user_id=uid,
            result=result,
            task_lease=lease,
            prepared_outputs=empty,
        )
        fields = {
            "output": finalized["output"],
            "error_code": finalized["error_code"],
            "error_details": finalized["error_details"],
        }
    with factory() as db:
        task = db.get(Task, task_id)
        assert task.status == TaskStatus.FAILED
        rows = (
            db.query(TaskChatMessage)
            .filter(TaskChatMessage.task_id == task_id)
            .filter(TaskChatMessage.role == "assistant")
            .all()
        )
        assert len(rows) == 1
        fields["error_message"] = task.error_message
        fields["row"] = (rows[0].content, rows[0].message_type)
        if storage_version == 2:
            events = (
                db.query(TaskExecutionEvent)
                .filter(TaskExecutionEvent.task_id == task_id)
                .filter(TaskExecutionEvent.kind == "assistant_message")
                .all()
            )
            assert len(events) == 1
            assert (
                events[0].payload["content"],
                events[0].payload["message_type"],
            ) == fields["row"]
    fields["repr"] = repr(finalized)
    return fields


@pytest.mark.parametrize("route", ["initial", "resume"])
@pytest.mark.parametrize("storage_version", [1, 2])
def test_failed_model_error_reaches_clients_as_the_projection_and_owners_as_full_text(
    engine, task_id, monkeypatch, route, storage_version
):
    fields = _finalize_failed_result(
        engine,
        task_id,
        monkeypatch,
        route,
        _model_failure_result("forbidden-403"),
        storage_version,
    )

    assert fields["error_message"] == _OWNER_DIAGNOSTIC_403_STORED
    assert fields["output"] == _PROJECTED_MODEL_FAILURE
    assert fields["error_code"] == "model_error"
    assert fields["error_details"] == {
        "kind": "access_denied",
        "status_code": 403,
        "provider_code": "provider_code_4204",
        "message": _PROJECTED_MODEL_FAILURE,
    }
    assert fields["row"] == (CLIENT_SAFE_TASK_FAILURE, "task_failure")
    assert "RAW_MARKER" not in fields["repr"]
    assert "RAW_MARKER" not in repr(fields["row"])


@pytest.mark.parametrize("route", ["initial", "resume"])
@pytest.mark.parametrize("storage_version", [1, 2])
def test_failed_http_400_keeps_the_generic_sentence_even_when_output_holds_the_raw_text(
    engine, task_id, monkeypatch, route, storage_version
):
    fields = _finalize_failed_result(
        engine,
        task_id,
        monkeypatch,
        route,
        _model_failure_result("bad-request-400"),
        storage_version,
    )

    assert fields["error_message"] == _RAW_400
    assert fields["output"] == CLIENT_SAFE_TASK_FAILURE
    assert fields["error_code"] is None
    assert fields["error_details"] is None
    assert fields["row"] == (CLIENT_SAFE_TASK_FAILURE, "task_failure")
    assert "RAW_MARKER" not in fields["repr"]
    assert "RAW_MARKER" not in repr(fields["row"])


@pytest.mark.parametrize("route", ["initial", "resume"])
@pytest.mark.parametrize("storage_version", [1, 2])
def test_failed_quota_gate_code_takes_precedence_over_a_recorded_model_error(
    engine, task_id, monkeypatch, route, storage_version
):
    fields = _finalize_failed_result(
        engine,
        task_id,
        monkeypatch,
        route,
        _model_failure_result("quota-gate"),
        storage_version,
    )

    assert fields["error_message"] == _QUOTA_REASON
    assert fields["output"] == _QUOTA_REASON
    assert fields["error_code"] == "quota_exceeded"
    assert fields["error_details"] == _QUOTA_DETAILS
    assert fields["row"] == (CLIENT_SAFE_TASK_FAILURE, "task_failure")
    assert "RAW_MARKER" not in fields["repr"]


@pytest.mark.parametrize("route", ["initial", "resume"])
@pytest.mark.parametrize("storage_version", [1, 2])
def test_failed_result_without_a_model_error_is_unchanged(
    engine, task_id, monkeypatch, route, storage_version
):
    fields = _finalize_failed_result(
        engine,
        task_id,
        monkeypatch,
        route,
        {
            "status": "error",
            "success": False,
            "output": "display text",
            "error": "plain failure text",
        },
        storage_version,
    )

    assert fields["error_message"] == "plain failure text"
    assert fields["output"] == "display text"
    assert fields["error_code"] is None
    assert fields["error_details"] is None
    assert fields["row"] == (CLIENT_SAFE_TASK_FAILURE, "task_failure")


@pytest.mark.parametrize("route", ["initial", "resume"])
@pytest.mark.parametrize("storage_version", [1, 2])
def test_failed_model_error_stays_out_of_the_next_turn_model_context(
    engine, task_id, monkeypatch, route, storage_version
):
    from xagent.web.services.chat_history_service import load_task_transcript_window

    _finalize_failed_result(
        engine,
        task_id,
        monkeypatch,
        route,
        _model_failure_result("forbidden-403"),
        storage_version,
    )

    with sessionmaker(engine)() as db:
        messages = load_task_transcript_window(db, task_id).messages

    for message in messages:
        assert "Model provider call failed" not in message["content"]
    if storage_version == 1:
        assistant_contents = [
            message["content"] for message in messages if message["role"] == "assistant"
        ]
        assert assistant_contents == [CLIENT_SAFE_TASK_FAILURE]
