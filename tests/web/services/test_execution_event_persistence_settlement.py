"""V2 settlement after an authoritative execution-event commit failure."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from tests.core.agent.test_react import FakeLLM, FakeTool
from tests.web.services.test_execution_event_recovery import tracer_for
from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_execution_event_writer import engine as engine_fixture
from tests.web.services.test_task_execution_event_writer import (
    facts,
)
from tests.web.services.test_task_execution_event_writer import (
    task_id as task_id_fixture,
)
from xagent.core.agent import ExecutionContext, PatternRuntime, ReActPattern
from xagent.core.agent.checkpoint import (
    CheckpointUnavailableError,
    TraceCheckpointStore,
)
from xagent.web.models.task import Task, TaskStatus
from xagent.web.services import task_execution_event_recovery as event_recovery
from xagent.web.services import task_orchestrator as orchestrator
from xagent.web.services.managed_task_lease import start_managed_task_lease
from xagent.web.services.task_event_trace_handler import get_event_type_mapping
from xagent.web.services.task_lease_recovery import (
    TASK_UNKNOWN_TOOL_EFFECT_ERROR as LEASE_EXPIRED_UNKNOWN_EFFECT_ERROR,
)
from xagent.web.services.task_lease_recovery import (
    recover_task_lease_candidate_no_commit,
)
from xagent.web.services.task_lease_service import (
    TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR,
    acquire_task_lease,
    bind_task_lease_context,
    get_expired_task_lease_candidates,
    utc_now,
)
from xagent.web.services.task_orchestrator import (
    TaskTurnPayload,
    _schedule_bg,
    settle_task_lease_isolated,
)
from xagent.web.tracing import ExecutionEventTraceAdapter

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture


@pytest.fixture(autouse=True)
def _clear_bg_manager():
    """``_schedule_bg`` registers its handle globally; do not leak it."""
    from xagent.web.services.task_execution import background_task_manager

    background_task_manager.running_tasks.clear()
    yield
    background_task_manager.running_tasks.clear()


def _tool_turn_llm() -> FakeLLM:
    return FakeLLM(
        responses=[
            {
                "content": "calculate",
                "tool_calls": [
                    {
                        "id": "call1",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                        },
                    }
                ],
            },
            {"content": "4", "done": True},
        ]
    )


def _fail_once(monkeypatch, kind: str) -> list[str]:
    """Fail the first commit of ``kind`` without writing it; later ones pass."""

    original = ExecutionEventTraceAdapter._save_to_database
    failed: list[str] = []

    async def save(self, event):
        if not failed and get_event_type_mapping(event) == kind:
            failed.append(kind)
            raise OperationalError("COMMIT", {}, Exception("connection reset"))
        return await original(self, event)

    monkeypatch.setattr(ExecutionEventTraceAdapter, "_save_to_database", save)
    return failed


def _react_execution(tid: int, tool: FakeTool):
    """A real ReAct turn writing V2 facts through the strict event writer."""

    async def execute(**kwargs):
        lease = kwargs.get("task_lease") or kwargs["lease"]
        with bind_task_lease_context(lease):
            runtime = PatternRuntime(tracer=TraceCheckpointStore(tracer_for(tid)))
            context = ExecutionContext(execution_id=str(tid), system_prompt="Calc")
            context.add_user_message("2+2", metadata={"turn_id": "turn"})
            await ReActPattern(max_iterations=3).run(
                context=context,
                tools=[tool],
                runtime=runtime,
                llm=_tool_turn_llm(),
            )

    return execute


async def _run_scheduled_turn(
    tid: int, lease, execute, *, channel: bool = False
) -> list[dict[str, Any]]:
    published: list[dict[str, Any]] = []

    async def publish(event, _task_id):
        published.append(event)

    with (
        patch.object(orchestrator, "run_task_lease_heartbeat", new=AsyncMock()),
        patch.object(
            orchestrator, "load_task_setup_snapshot_sync", return_value=MagicMock()
        ),
        patch.object(
            orchestrator, "resolve_execution_scope", return_value=None, create=True
        ),
        patch(
            "xagent.web.services.task_execution.execute_task_background",
            new=execute,
        ),
        patch(
            "xagent.web.services.shared_channel_execution.execute_channel_background",
            new=execute,
        ),
        patch(
            "xagent.web.services.task_event_display.publish_task_result", new=publish
        ),
        patch.object(orchestrator, "_get_agent_manager", return_value=MagicMock()),
    ):
        await _schedule_bg(
            task_id=tid,
            task_owner_user_id=1,
            task_source="sdk",
            task_lease=lease,
            payload=TaskTurnPayload("2+2"),
            force_fresh=False,
            context=None,
            channel_command=MagicMock() if channel else None,
        )
    return published


def _assert_unknown_effect_settlement(factory, tid: int, lease, error: str) -> None:
    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == TaskStatus.FAILED
        assert task.runner_id is None
        assert task.error_message == error
        rows = facts(db, tid)
        kinds = [row.kind for row in rows]
        assert kinds.count("tool_execution_start") == 1
        assert "tool_execution_end" not in kinds
        settled = [row for row in rows if row.kind == "execution_settled"]
        assert len(settled) == 1
        assert settled[0].run_id == lease.run_id
        assert settled[0].payload == {"status": "failed", "result": {"error": error}}


def _assert_client_safe_failure_transcript(factory, tid: int) -> None:
    """The diagnostic stays on the row; readers get the client-safe line."""
    from xagent.web.models.chat_message import TaskChatMessage
    from xagent.web.services.client_error_messages import CLIENT_SAFE_TASK_FAILURE

    with factory() as db:
        lines = [
            row.content
            for row in db.query(TaskChatMessage)
            .filter(TaskChatMessage.task_id == tid, TaskChatMessage.role == "assistant")
            .order_by(TaskChatMessage.id)
        ]
    assert lines[-1] == CLIENT_SAFE_TASK_FAILURE
    assert not [
        line for line in lines if TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR in line
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("pause_switch", ["true", "false"], ids=["pause", "legacy"])
@pytest.mark.parametrize("channel", [False, True], ids=["web", "channel"])
async def test_lost_tool_result_settles_as_unknown_effect(
    canonical, monkeypatch, caplog, channel, pause_switch
):
    """The same FAILED unknown-effect settlement whether the interruption
    is decided (switch on) or classified the legacy way (switch off)."""
    from xagent.web.services.client_error_messages import CLIENT_SAFE_TASK_FAILURE

    monkeypatch.setenv("XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED", pause_switch)
    caplog.set_level("WARNING", logger="xagent.web.services.task_lease_service")
    factory, tid = canonical
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    failed = _fail_once(monkeypatch, "tool_execution_end")
    tool = FakeTool()

    published = await _run_scheduled_turn(
        tid, lease, _react_execution(tid, tool), channel=channel
    )

    # The tool ran once, its result commit failed, and the DB is fine again.
    assert failed == ["tool_execution_end"]
    assert len(tool.calls) == 1
    _assert_unknown_effect_settlement(
        factory, tid, lease, TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR
    )
    # One client-safe terminal event; no success, no operator detail.
    assert [event["type"] for event in published] == ["task_error"]
    assert published[0]["error"] == CLIENT_SAFE_TASK_FAILURE
    assert "Conversation event commit failed" not in str(published)
    assert TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR not in str(published)
    _assert_client_safe_failure_transcript(factory, tid)
    assert [
        record.getMessage()
        for record in caplog.records
        if "unknown tool effect" in record.getMessage()
    ] == [
        f"Task {tid} run {lease.run_id} settles as an unknown tool effect "
        "(verdict unknown_tool_effect)"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("pause_switch", ["false", None])
async def test_unstarted_tool_keeps_generic_persistence_failure(
    canonical, monkeypatch, pause_switch
):
    """No started attempt means no unknown effect: the tool never ran.

    With interruption pauses switched off -- explicitly, or by default with
    automatic resume off, which the unset switch follows -- the run fails as
    it always has.
    """

    if pause_switch is None:
        monkeypatch.delenv("XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED", raising=False)
        monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_ENABLED", "false")
    else:
        monkeypatch.setenv("XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED", pause_switch)
    factory, tid = canonical
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    failed = _fail_once(monkeypatch, "tool_execution_start")
    tool = FakeTool()

    await _run_scheduled_turn(tid, lease, _react_execution(tid, tool))

    assert failed == ["tool_execution_start"]
    assert tool.calls == []
    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == TaskStatus.FAILED
        assert task.error_message.startswith(
            "setup/run error: ExecutionEventPersistenceError"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("pause_switch", ["true", None])
async def test_unstarted_tool_persistence_failure_pauses_the_run(
    canonical, monkeypatch, pause_switch
):
    """With the pause switch on -- explicitly, or by default with automatic
    resume on -- the same failure pauses the run: it has a checkpoint."""

    if pause_switch is None:
        monkeypatch.delenv("XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED", raising=False)
        monkeypatch.delenv("XAGENT_TASK_AUTO_RESUME_ENABLED", raising=False)
    else:
        monkeypatch.setenv("XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED", pause_switch)
    factory, tid = canonical
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    failed = _fail_once(monkeypatch, "tool_execution_start")
    tool = FakeTool()

    await _run_scheduled_turn(tid, lease, _react_execution(tid, tool))

    assert failed == ["tool_execution_start"]
    assert tool.calls == []
    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == TaskStatus.PAUSED
        assert task.control_state == "paused"
        assert task.runner_id is None
        assert task.error_message is None


@pytest.mark.asyncio
@pytest.mark.parametrize("pause_switch", ["true", "false"], ids=["pause", "legacy"])
async def test_unreadable_classification_defers_to_lease_recovery(
    canonical, monkeypatch, pause_switch
):
    """Settlement that cannot classify keeps the lease; recovery classifies.

    Switch on, the interruption decision cannot read the checkpoint; switch
    off, the legacy unknown-effect classification cannot.
    """

    monkeypatch.setenv("XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED", pause_switch)
    factory, tid = canonical
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    _fail_once(monkeypatch, "tool_execution_end")
    tool = FakeTool()
    original_read = event_recovery.read_event_checkpoint

    def unavailable(*args, **kwargs):
        raise CheckpointUnavailableError("recovery read failed")

    from xagent.web.services import task_lease_service

    degradations: list[tuple[str, str]] = []
    monkeypatch.setattr(
        task_lease_service,
        "register_degradation",
        lambda signal, detail: degradations.append((signal, detail)),
    )
    monkeypatch.setattr(event_recovery, "read_event_checkpoint", unavailable)
    published = await _run_scheduled_turn(tid, lease, _react_execution(tid, tool))

    assert published == []
    assert degradations == [
        (
            task_lease_service.CHECKPOINT_RECOVERY_UNAVAILABLE,
            f"task_id={tid}: CheckpointUnavailableError",
        )
    ]
    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == TaskStatus.RUNNING
        assert task.runner_id == lease.runner_id
        assert not [row for row in facts(db, tid) if row.kind == "execution_settled"]
        task.lease_expires_at = utc_now() - timedelta(seconds=5)
        db.commit()

    monkeypatch.setattr(event_recovery, "read_event_checkpoint", original_read)
    with factory() as db:
        candidate = get_expired_task_lease_candidates(db, cutoff=utc_now(), limit=1)[0]
        assert (
            recover_task_lease_candidate_no_commit(
                db, candidate, recovered_at=utc_now()
            )
            == TaskStatus.FAILED
        )
        db.commit()
    assert len(tool.calls) == 1
    _assert_unknown_effect_settlement(
        factory, tid, lease, LEASE_EXPIRED_UNKNOWN_EFFECT_ERROR
    )


@pytest.mark.asyncio
async def test_managed_lease_failure_settles_as_unknown_effect(canonical, monkeypatch):
    """Inline channel transports settle a raised run through the managed lease."""

    factory, tid = canonical
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    _fail_once(monkeypatch, "tool_execution_end")
    tool = FakeTool()
    with patch(
        "xagent.web.services.managed_task_lease.run_task_lease_heartbeat",
        new=AsyncMock(),
    ):
        managed = start_managed_task_lease(lease)
        # The channel bots' error path: the run raised, no result to project.
        with pytest.raises(Exception) as caught:
            await _react_execution(tid, tool)(task_lease=lease)
        assert await managed.finalize_result(
            status=TaskStatus.FAILED, error_message=str(caught.value)
        )
    assert len(tool.calls) == 1
    _assert_unknown_effect_settlement(
        factory, tid, lease, TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR
    )
    _assert_client_safe_failure_transcript(factory, tid)


def test_legacy_task_settlement_is_not_classified(canonical):
    factory, tid = canonical
    with factory() as db:
        db.get(Task, tid).conversation_storage_version = 1
        db.commit()
        lease = acquire_task_lease(db, tid, new_run=True)
    assert settle_task_lease_isolated(
        lease,
        error_message="setup/run error: RuntimeError: boom",
        classify_unknown_tool_effect=True,
    )
    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == TaskStatus.FAILED
        assert task.error_message == "setup/run error: RuntimeError: boom"


async def _run_turn_with_broken_finalize_commit(
    canonical,
    monkeypatch,
    outcome: dict[str, Any],
    *,
    applied: bool,
    storage_version: int = 2,
    before_read_back: Callable[[list[dict[str, Any]], Any], None] | None = None,
    read_back_error: Exception | None = None,
    rollback_error: bool = False,
) -> tuple[Any, list[dict[str, Any]], list[bool]]:
    """Run one real turn whose result COMMIT raises once.

    Only the finalize session's commit of the result is broken: the session
    that stages the result fact is the one whose next commit raises.
    ``applied`` decides whether the server applied that COMMIT before the
    driver raised (a lost acknowledgement) or rolled it back. With
    ``rollback_error`` the following rollback raises too and the session is
    unusable afterwards. Returns the lease, the published events, and each
    staged-output settlement's ``metadata_committed`` flag.
    """

    from xagent.web.models import database
    from xagent.web.services import task_execution

    factory, tid = canonical
    monkeypatch.setattr(database, "_SessionLocal", factory)
    with factory() as db:
        db.get(Task, tid).conversation_storage_version = storage_version
        db.commit()
        lease = acquire_task_lease(db, tid, new_run=True)

    target: list[Session] = []
    broken: list[str] = []
    poisoned: list[Session] = []
    original_commit = Session.commit
    original_rollback = Session.rollback
    original_execute = Session.execute
    original_stage = task_execution.stage_result_fact_no_commit

    def stage(db, task, result):
        target[:] = [db]
        return original_stage(db, task, result)

    def broken_commit(self):
        if not (target and target[0] is self):
            return original_commit(self)
        target.clear()
        broken.append("finalize")
        if applied:
            original_commit(self)
        if rollback_error:
            target.append(self)
        raise OperationalError("COMMIT", {}, Exception("connection lost"))

    def broken_rollback(self):
        if target and target[0] is self:
            target.clear()
            original_rollback(self)
            poisoned.append(self)
            raise OperationalError("ROLLBACK", {}, Exception("connection lost"))
        return original_rollback(self)

    def poisoned_execute(self, *args, **kwargs):
        if any(session is self for session in poisoned):
            raise OperationalError("SELECT", {}, Exception("connection lost"))
        return original_execute(self, *args, **kwargs)

    published: list[dict[str, Any]] = []
    original_read_back = task_execution._finalization_witness_committed

    def read_back(witness):
        if before_read_back is not None:
            before_read_back(published, witness)
        if read_back_error is not None:
            raise read_back_error
        return original_read_back(witness)

    settled_outputs: list[bool] = []
    original_settle_outputs = task_execution._settle_prepared_task_file_outputs

    def settle_outputs(prepared, *, metadata_committed, **kwargs):
        settled_outputs.append(metadata_committed)
        return original_settle_outputs(
            prepared, metadata_committed=metadata_committed, **kwargs
        )

    monkeypatch.setattr(Session, "commit", broken_commit)
    monkeypatch.setattr(Session, "rollback", broken_rollback)
    monkeypatch.setattr(Session, "execute", poisoned_execute)
    monkeypatch.setattr(task_execution, "stage_result_fact_no_commit", stage)
    monkeypatch.setattr(task_execution, "_finalization_witness_committed", read_back)
    monkeypatch.setattr(
        task_execution, "_settle_prepared_task_file_outputs", settle_outputs
    )
    manager = MagicMock()
    manager.get_agent_for_task = AsyncMock(return_value=MagicMock())
    manager.execute_task = AsyncMock(return_value=dict(outcome))

    async def publish(event, _task_id):
        published.append(event)

    with (
        patch.object(orchestrator, "run_task_lease_heartbeat", new=AsyncMock()),
        patch.object(
            orchestrator, "resolve_execution_scope", return_value=None, create=True
        ),
        patch.object(orchestrator, "_get_agent_manager", return_value=manager),
        patch("xagent.web.services.task_events.publish_task_event", new=publish),
        patch.object(task_execution, "publish_task_event", new=publish),
    ):
        await _schedule_bg(
            task_id=tid,
            task_owner_user_id=1,
            task_source="sdk",
            task_lease=lease,
            payload=TaskTurnPayload("2+2"),
            force_fresh=False,
            context=None,
        )

    assert broken == ["finalize"]
    manager.execute_task.assert_awaited_once()
    return lease, published, settled_outputs


def _terminal_events(published: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        event
        for event in published
        if event.get("type") in {"task_completed", "task_error"}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        {"success": True, "output": "4"},
        {"success": False, "output": "", "error": "model refused"},
    ],
    ids=["completed", "failed"],
)
async def test_finalize_lost_ack_publishes_committed_terminal_once(
    canonical, monkeypatch, outcome
):
    """COMMIT reached the server but the driver raised: announce what committed."""

    factory, tid = canonical
    lease, published, _ = await _run_turn_with_broken_finalize_commit(
        canonical, monkeypatch, outcome, applied=True
    )

    expected = TaskStatus.COMPLETED if outcome["success"] else TaskStatus.FAILED
    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == expected
        assert task.runner_id is None
        settled = [row for row in facts(db, tid) if row.kind == "execution_settled"]
        assert len(settled) == 1
        state_version = int(task.state_version)
    terminal = _terminal_events(published)
    assert len(terminal) == 1
    assert terminal[0]["type"] == "task_completed"
    assert terminal[0]["task"]["status"] == expected.value
    assert terminal[0]["run_id"] == lease.run_id
    assert terminal[0]["state_version"] == state_version
    if outcome["success"]:
        # The V2 display path ran: the committed assistant frame precedes it.
        assert terminal[0]["message_id"] == published[-2]["data"]["message_id"]


@pytest.mark.asyncio
async def test_finalize_rolled_back_commit_is_never_reported_as_success(
    canonical, monkeypatch
):
    factory, tid = canonical
    lease, published, _ = await _run_turn_with_broken_finalize_commit(
        canonical, monkeypatch, {"success": True, "output": "4"}, applied=False
    )

    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == TaskStatus.FAILED
        assert task.runner_id is None
        assert task.error_message.startswith("setup/run error: OperationalError")
        settled = [row for row in facts(db, tid) if row.kind == "execution_settled"]
        assert [row.payload["status"] for row in settled] == ["failed"]
    terminal = _terminal_events(published)
    assert [event["type"] for event in terminal] == ["task_error"]
    assert terminal[0]["run_id"] == lease.run_id


@pytest.mark.asyncio
async def test_finalize_lost_ack_publishes_committed_wait_once(canonical, monkeypatch):
    factory, tid = canonical
    lease, published, _ = await _run_turn_with_broken_finalize_commit(
        canonical,
        monkeypatch,
        {"success": True, "status": "waiting_for_user", "output": "Which file?"},
        applied=True,
    )

    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == TaskStatus.WAITING_FOR_USER
        assert task.runner_id is None
        state_version = int(task.state_version)
    assert _terminal_events(published) == []
    info = [event for event in published if event.get("event_type") == "task_info"]
    assert len(info) == 1
    assert info[0]["data"]["status"] == TaskStatus.WAITING_FOR_USER.value
    assert info[0]["data"]["state_version"] == state_version
    assert info[0]["data"]["run_id"] == lease.run_id


def _coordinator_cancel_commits(factory, tid: int) -> None:
    """Commit what a coordinator-context external cancel commits.

    Mirrors ``_finalize_external_cancel_sync``: in coordinator context the
    ownership values are empty, so the row turns FAILED at the next state
    version with the lease fence intact, plus the interruption line and (V2)
    its own result fact under the same key a FAILED finalize would use.
    """

    from xagent.web.services.assistant_history_safety import (
        CLIENT_SAFE_FAILURE_MESSAGE_TYPE,
    )
    from xagent.web.services.chat_history_service import (
        persist_assistant_message_no_commit,
    )
    from xagent.web.services.external_task_cancel import (
        EXTERNAL_CANCEL_ERROR_MESSAGE,
        EXTERNAL_TURN_INTERRUPTED_MESSAGE,
    )
    from xagent.web.services.task_execution_controller import TaskControlState
    from xagent.web.services.task_execution_event_writer import (
        stage_result_fact_no_commit,
    )

    with factory() as db:
        task = db.get(Task, tid)
        # The finalize COMMIT really rolled back: the run still owns the row.
        assert task.status == TaskStatus.RUNNING
        assert task.runner_id is not None
        version = int(task.state_version or 0)
        task.status = TaskStatus.FAILED
        task.control_state = TaskControlState.FAILED.value
        task.state_version = version + 1
        task.error_message = EXTERNAL_CANCEL_ERROR_MESSAGE
        db.flush()
        persist_assistant_message_no_commit(
            db,
            task_id=tid,
            user_id=int(task.user_id),
            content=EXTERNAL_TURN_INTERRUPTED_MESSAGE,
            message_type=CLIENT_SAFE_FAILURE_MESSAGE_TYPE,
            content_is_reconciled=True,
        )
        stage_result_fact_no_commit(
            db,
            task,
            {"status": "cancelled", "error": EXTERNAL_CANCEL_ERROR_MESSAGE},
        )
        db.commit()


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_version", [2, 1], ids=["v2", "v1"])
async def test_rolled_back_failure_is_not_confirmed_by_a_racing_cancel(
    canonical, monkeypatch, storage_version
):
    """A cancel committing the same FAILED version is not this result's commit."""

    from xagent.web.services.external_task_cancel import (
        EXTERNAL_CANCEL_ERROR_MESSAGE,
    )

    factory, tid = canonical
    cancels: list[str] = []

    def cancel(published, _witness):
        _coordinator_cancel_commits(factory, tid)
        cancels.append("cancel")
        # The cancel announces its own outcome after its commit.
        published.append({"type": "task_error", "source": "cancel"})

    _, published, settled_outputs = await _run_turn_with_broken_finalize_commit(
        canonical,
        monkeypatch,
        {"success": False, "output": "", "error": "model refused"},
        applied=False,
        storage_version=storage_version,
        before_read_back=cancel,
    )

    assert cancels == ["cancel"]
    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == TaskStatus.FAILED
        assert task.error_message == EXTERNAL_CANCEL_ERROR_MESSAGE
        settled = [row for row in facts(db, tid) if row.kind == "execution_settled"]
        if storage_version == 2:
            assert [row.payload["result"] for row in settled] == [
                {"status": "cancelled", "error": EXTERNAL_CANCEL_ERROR_MESSAGE}
            ]
        else:
            assert settled == []
    # Only the cancel's terminal event; the rolled-back result is never sent.
    assert [event.get("source") for event in _terminal_events(published)] == ["cancel"]
    # The rolled-back result's staged outputs were compensated.
    assert settled_outputs == [False]


@pytest.mark.asyncio
async def test_failed_read_back_reraises_commit_error_without_success(
    canonical, monkeypatch
):
    factory, tid = canonical
    _, published, settled_outputs = await _run_turn_with_broken_finalize_commit(
        canonical,
        monkeypatch,
        {"success": True, "output": "4"},
        applied=True,
        read_back_error=OperationalError("SELECT", {}, Exception("still down")),
    )

    # Undecidable: the commit error is the run's failure, so nothing claims
    # the committed completion. The row keeps what the server committed.
    with factory() as db:
        assert db.get(Task, tid).status == TaskStatus.COMPLETED
    assert [event for event in published if event.get("type") == "task_completed"] == []
    assert settled_outputs == [False]


@pytest.mark.asyncio
async def test_rollback_error_after_lost_ack_still_publishes_committed_once(
    canonical, monkeypatch
):
    factory, tid = canonical
    _, published, settled_outputs = await _run_turn_with_broken_finalize_commit(
        canonical,
        monkeypatch,
        {"success": True, "output": "4"},
        applied=True,
        rollback_error=True,
    )

    with factory() as db:
        assert db.get(Task, tid).status == TaskStatus.COMPLETED
    terminal = _terminal_events(published)
    assert [event["task"]["status"] for event in terminal] == ["completed"]
    assert settled_outputs == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        {"success": False, "status": "interrupted", "output": ""},
        {"success": True, "injection_outcome_unknown": True, "output": "4"},
    ],
    ids=["interrupted", "preserve_unknown_result"],
)
async def test_finalize_lost_ack_publishes_committed_pause_once(
    canonical, monkeypatch, outcome
):
    factory, tid = canonical
    _, published, settled_outputs = await _run_turn_with_broken_finalize_commit(
        canonical, monkeypatch, outcome, applied=True
    )

    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == TaskStatus.PAUSED
        settled = [row for row in facts(db, tid) if row.kind == "execution_settled"]
        assert [row.payload["status"] for row in settled] == ["paused"]
        if outcome.get("injection_outcome_unknown"):
            assert task.output == "4"
        state_version = int(task.state_version)
    assert _terminal_events(published) == []
    info = [event for event in published if event.get("event_type") == "task_info"]
    assert len(info) == 1
    assert info[0]["data"]["status"] == TaskStatus.PAUSED.value
    assert info[0]["data"]["state_version"] == state_version
    assert settled_outputs == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        {"success": True, "output": "4"},
        {"success": False, "output": "", "error": "model refused"},
    ],
    ids=["completed", "failed"],
)
async def test_legacy_finalize_lost_ack_publishes_committed_terminal_once(
    canonical, monkeypatch, outcome
):
    """V1 proves its commit by the transcript row the finalize inserted."""

    factory, tid = canonical
    _, published, settled_outputs = await _run_turn_with_broken_finalize_commit(
        canonical, monkeypatch, outcome, applied=True, storage_version=1
    )

    expected = TaskStatus.COMPLETED if outcome["success"] else TaskStatus.FAILED
    with factory() as db:
        assert db.get(Task, tid).status == expected
    terminal = _terminal_events(published)
    assert [event["type"] for event in terminal] == ["task_completed"]
    assert terminal[0]["task"]["status"] == expected.value
    assert settled_outputs == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "status"),
    [
        (
            {"success": True, "status": "waiting_for_user", "output": "Which?"},
            TaskStatus.WAITING_FOR_USER,
        ),
        ({"success": False, "status": "interrupted", "output": ""}, TaskStatus.PAUSED),
    ],
    ids=["waiting_for_user", "interrupted"],
)
async def test_legacy_control_lost_ack_without_witness_is_not_confirmed(
    canonical, monkeypatch, outcome, status
):
    """A V1 wait/pause inserts no row of its own, so nothing can prove it."""

    factory, tid = canonical
    _, published, settled_outputs = await _run_turn_with_broken_finalize_commit(
        canonical, monkeypatch, outcome, applied=True, storage_version=1
    )

    with factory() as db:
        assert db.get(Task, tid).status == status
    assert [e for e in published if e.get("event_type") == "task_info"] == []
    assert _terminal_events(published) == []
    assert settled_outputs == [False]


def test_settle_of_already_terminal_row_skips_classification(canonical, monkeypatch):
    factory, tid = canonical
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
        # A completion committed under the still-held fence.
        db.get(Task, tid).status = TaskStatus.COMPLETED
        db.commit()
    reads: list[Any] = []
    original = orchestrator.run_has_unknown_tool_effect

    def classify(db, lease_):
        reads.append(lease_)
        return original(db, lease_)

    monkeypatch.setattr(orchestrator, "run_has_unknown_tool_effect", classify)
    assert not settle_task_lease_isolated(
        lease,
        error_message="setup/run error: RuntimeError: boom",
        classify_unknown_tool_effect=True,
    )
    assert reads == []
    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == TaskStatus.COMPLETED
        assert task.error_message is None


def _count_classification_reads(monkeypatch) -> list[int]:
    from xagent.web.services import task_lease_service

    reads: list[int] = []
    original = task_lease_service._resolve_event_checkpoint_recovery

    def resolve(db, *, task_id, run_id):
        reads.append(task_id)
        return original(db, task_id=task_id, run_id=run_id)

    monkeypatch.setattr(
        task_lease_service, "_resolve_event_checkpoint_recovery", resolve
    )
    return reads


@pytest.mark.parametrize("path", ["release", "channel_claim"])
def test_managed_abandonment_of_fresh_claim_is_a_plain_failure(
    canonical, monkeypatch, path
):
    """No started tool attempt: one classification read, no unknown effect."""

    from xagent.web.services.channel_runtime import (
        _ChannelTaskClaimSnapshot,
        _compensate_channel_task_claim_sync,
    )
    from xagent.web.services.client_error_messages import CLIENT_SAFE_TASK_FAILURE
    from xagent.web.services.managed_task_lease import (
        _release_managed_task_lease_sync,
    )

    factory, tid = canonical
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    reads = _count_classification_reads(monkeypatch)

    if path == "release":
        assert _release_managed_task_lease_sync(lease)
    else:
        monkeypatch.setattr(
            "xagent.web.services.channel_runtime.get_session_local", lambda: factory
        )
        assert _compensate_channel_task_claim_sync(
            _ChannelTaskClaimSnapshot(
                user_id=1, task_id=tid, is_new_task=True, lease=lease
            )
        )

    assert reads == [tid]
    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == TaskStatus.FAILED
        assert task.runner_id is None
        assert task.error_message == CLIENT_SAFE_TASK_FAILURE
        settled = [row for row in facts(db, tid) if row.kind == "execution_settled"]
        assert [row.payload["result"] for row in settled] == [{"error": None}]


def _hold_witness_commit_in_flight(factory, witness, *, hold: float):
    """Insert ``witness``'s row under the task lock and commit it later.

    Stands for the server still applying the finalize COMMIT when the
    read-back starts: the row exists only once the lock holder commits.
    """

    import threading
    import time

    from sqlalchemy import update

    from xagent.web.models.task_execution_event import TaskExecutionEvent
    from xagent.web.services.task_execution_event_store import (
        lock_task_execution_events_no_commit,
    )

    locked = threading.Event()
    errors: list[BaseException] = []

    def run():
        try:
            with factory() as db:
                sequence = lock_task_execution_events_no_commit(db, witness.task_id)
                db.execute(
                    update(Task)
                    .where(Task.id == witness.task_id)
                    .values(conversation_event_sequence=sequence + 1)
                )
                db.add(
                    TaskExecutionEvent(
                        event_id=witness.event_id,
                        task_id=witness.task_id,
                        scope_id=witness.scope_id,
                        idempotency_key=witness.idempotency_key,
                        sequence=sequence + 1,
                        kind="execution_settled",
                        payload_version=1,
                        payload={},
                        occurred_at=utc_now(),
                    )
                )
                db.flush()
                locked.set()
                time.sleep(hold)
                db.commit()
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)
            locked.set()

    thread = threading.Thread(target=run)
    thread.start()
    assert locked.wait(10)
    assert not errors
    return thread, errors


@pytest.mark.asyncio
async def test_read_back_waits_for_an_in_flight_commit(canonical, monkeypatch):
    """The read-back takes the task lock first, so it sees a COMMIT in flight."""

    factory, _tid = canonical
    threads: list[Any] = []

    def in_flight(_published, witness):
        threads.append(_hold_witness_commit_in_flight(factory, witness, hold=0.5))

    _, published, settled_outputs = await _run_turn_with_broken_finalize_commit(
        canonical,
        monkeypatch,
        {"success": True, "output": "4"},
        applied=False,
        before_read_back=in_flight,
    )
    thread, errors = threads[0]
    thread.join(10)
    assert not errors
    assert [event["type"] for event in _terminal_events(published)] == [
        "task_completed"
    ]
    assert settled_outputs == [True]


@pytest.mark.asyncio
async def test_read_back_lock_timeout_reraises_commit_error(canonical, monkeypatch):
    from xagent.web.services import task_execution

    factory, tid = canonical
    if "postgresql" not in str(factory.kw["bind"].url):
        pytest.skip("lock_timeout bounds the PostgreSQL wait only")
    monkeypatch.setattr(task_execution, "_FINALIZATION_READ_BACK_LOCK_TIMEOUT", "100ms")
    threads: list[Any] = []

    def in_flight(_published, witness):
        threads.append(_hold_witness_commit_in_flight(factory, witness, hold=2))

    _, published, settled_outputs = await _run_turn_with_broken_finalize_commit(
        canonical,
        monkeypatch,
        {"success": True, "output": "4"},
        applied=False,
        before_read_back=in_flight,
    )
    thread, errors = threads[0]
    thread.join(10)
    assert not errors
    # Undecidable within the bound: no completion is claimed.
    assert [e for e in published if e.get("type") == "task_completed"] == []
    assert settled_outputs == [False]
    with factory() as db:
        assert db.get(Task, tid).status == TaskStatus.FAILED


def test_result_fact_witness_names_only_a_row_this_call_inserted(canonical):
    from xagent.web.services.task_execution_event_writer import (
        fact_witness_committed_no_commit,
        stage_result_fact_no_commit,
    )

    factory, tid = canonical
    with factory() as db:
        task = db.get(Task, tid)
        task.run_id = "run-1"
        task.status = TaskStatus.FAILED
        witness = stage_result_fact_no_commit(db, task, {"error": "boom"})
        db.commit()
        assert witness is not None
        assert fact_witness_committed_no_commit(db, witness)
        # A replay reuses the committed row: it proves nothing about this call.
        assert stage_result_fact_no_commit(db, task, {"error": "boom"}) is None
        db.rollback()
        task = db.get(Task, tid)
        task.conversation_storage_version = 1
        assert stage_result_fact_no_commit(db, task, {"error": "boom"}) is None
