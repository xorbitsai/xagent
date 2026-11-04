"""Stage E: real V2 entry paths must not fetch compatibility content."""

import pytest
import sqlalchemy as sa

from tests.shared.execution_event_read_guard import forbid_legacy_content
from tests.web.services.test_execution_event_recovery import tracer_for
from tests.web.services.test_task_event_display import fact
from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_execution_event_writer import engine as engine_fixture
from tests.web.services.test_task_execution_event_writer import (
    task_id as task_id_fixture,
)
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import (
    DAGExecution,
    Task,
    TraceCheckpointBlob,
    TraceEvent,
    TraceMessageBlob,
)
from xagent.web.services import (
    task_command_execution,
    task_existing_command,
    task_start_consumer,
)
from xagent.web.services.chat_history_service import (
    claim_user_message_delivery_no_commit,
    inspect_user_message_delivery,
    mark_user_message_delivery,
    persist_user_message_no_commit,
)

base_canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture


@pytest.fixture
def canonical(base_canonical, monkeypatch):
    factory, tid = base_canonical
    for module in (task_command_execution, task_existing_command, task_start_consumer):
        monkeypatch.setattr(module, "get_session_local", lambda: factory)
    return factory, tid


@pytest.mark.parametrize(
    "column",
    [
        TaskChatMessage.content,
        TaskChatMessage.attachments,
        TaskChatMessage.interactions,
        TraceEvent.data,
        DAGExecution.current_plan,
        DAGExecution.skipped_steps,
        TraceMessageBlob.message_data,
        TraceCheckpointBlob.blob_data,
    ],
)
def test_content_tripwire_rejects_projection_and_full_row(canonical, column):
    factory, _ = canonical
    with factory() as db, forbid_legacy_content(db.bind):
        with pytest.raises(AssertionError, match="Legacy content read"):
            db.execute(sa.select(column))
        with pytest.raises(AssertionError, match="Legacy content read"):
            db.execute(sa.select(TaskChatMessage))
        db.execute(sa.select(TaskChatMessage.delivery_status))


def test_v2_input_retry_and_delivery_use_fact_content(canonical):
    factory, tid = canonical
    with factory() as db:
        owner = db.get(Task, tid).user_id
        claim = claim_user_message_delivery_no_commit(
            db,
            tid,
            owner,
            "accepted",
            attachments=[{"file_id": "file-1"}],
            turn_id="turn",
        )
        assert claim.claimed
        db.commit()
        # Keep the control row but make its content contradict the fact.
        db.execute(
            sa.update(TaskChatMessage)
            .where(TaskChatMessage.task_id == tid)
            .values(content="obsolete", attachments=[{"file_id": "wrong"}])
        )
        db.commit()
    # A fresh session makes ORM identity-map reuse unable to hide a read.
    with factory() as db, forbid_legacy_content(db.bind):
        retry = inspect_user_message_delivery(
            db, tid, "accepted", attachments=[{"file_id": "file-1"}], turn_id="turn"
        )
        assert retry is not None and retry.payload_matches and retry.pending
        changed = inspect_user_message_delivery(
            db, tid, "obsolete", attachments=[{"file_id": "wrong"}], turn_id="turn"
        )
        assert changed is not None and not changed.payload_matches
        reused = claim_user_message_delivery_no_commit(
            db,
            tid,
            owner,
            "accepted",
            attachments=[{"file_id": "file-1"}],
            turn_id="turn",
        )
        assert not reused.claimed and reused.payload_matches
        assert (
            mark_user_message_delivery(
                db, task_id=tid, turn_id="turn", status="completed"
            ).outcome
            == "updated"
        )
        db.commit()
    with factory() as db, forbid_legacy_content(db.bind):
        completed = inspect_user_message_delivery(
            db, tid, "accepted", attachments=[{"file_id": "file-1"}], turn_id="turn"
        )
        assert completed is not None and not completed.pending
        assert completed.message.delivery_status == "completed"
        from xagent.web.services.task_command_execution import (
            _load_command_message_delivery_status,
        )

        assert _load_command_message_delivery_status(tid, "turn") == "completed"
        assert _load_command_message_delivery_status(tid, "absent") is None


def test_v2_persistence_retry_returns_event_content(canonical):
    factory, tid = canonical
    with factory() as db:
        owner = db.get(Task, tid).user_id
        first = persist_user_message_no_commit(
            db, tid, owner, "accepted", attachments=[], turn_id="turn"
        )
        db.commit()
        identity = first.id
    with factory() as db, forbid_legacy_content(db.bind):
        again = persist_user_message_no_commit(
            db, tid, owner, "accepted", attachments=[], turn_id="turn"
        )
        assert again.id == identity
        assert again.content == "accepted"
        assert again.attachments == []


@pytest.mark.asyncio
async def test_all_v2_readers_share_facts_with_legacy_content_blocked(
    canonical, monkeypatch
):
    from xagent.core.agent.checkpoint import TraceCheckpointStore
    from xagent.core.agent.context import ExecutionContext
    from xagent.web.api import chat, websocket, workforces
    from xagent.web.api.conversation_logs import get_conversation_log_detail
    from xagent.web.api.v1 import tasks
    from xagent.web.models.agent import Agent
    from xagent.web.models.user import User
    from xagent.web.models.workforce import Workforce, WorkforceRun
    from xagent.web.services import task_setup_snapshot as setup

    factory, tid = canonical
    with factory() as db:
        task = db.get(Task, tid)
        owner = task.user_id
        task.source = "sdk"
        task.is_visible = False
        agent = Agent(name="acceptance", user_id=owner)
        db.add(agent)
        db.flush()
        task.agent_id = agent.id
        workforce = Workforce(
            owner_user_id=owner,
            scope_id=str(owner),
            name="acceptance",
            manager_agent_id=agent.id,
        )
        db.add(workforce)
        db.flush()
        workforce_id = workforce.id
        db.add(
            WorkforceRun(
                workforce_id=workforce.id, task_id=tid, user_id=owner, snapshot={}
            )
        )
        persist_user_message_no_commit(db, tid, owner, "earlier", turn_id="old")
        db.commit()
    context = ExecutionContext(execution_id=str(tid))
    context.add_message("user", "earlier", metadata={"turn_id": "old"})
    checkpoint = {
        "execution_id": str(tid),
        "context": context.to_dict(),
        "pattern_state": {},
        "label": "ready",
    }
    store = TraceCheckpointStore(tracer_for(tid))
    await store.save(checkpoint)
    with factory() as db:
        fact(
            db,
            tid,
            "assistant_message",
            "answer",
            {"content": "answer", "message_type": "assistant_response"},
            flat=True,
        )
        fact(
            db,
            tid,
            "react_task_start",
            "child-start",
            {
                "source": "xagent-agent-tool-child",
                "worker_task_id": "child",
                "agent_name": "Worker",
            },
            scope_id="child",
        )
        persist_user_message_no_commit(db, tid, owner, "current", turn_id="current")
        db.execute(
            sa.update(TaskChatMessage)
            .where(TaskChatMessage.task_id == tid)
            .values(content="wrong legacy body")
        )
        db.commit()
    monkeypatch.setattr(setup, "get_session_local", lambda: factory)
    monkeypatch.setattr("xagent.web.models.database._SessionLocal", factory)
    monkeypatch.setattr(websocket, "cache_get", lambda *_: None)
    monkeypatch.setattr(websocket, "cache_set", lambda *_a, **_k: None)
    monkeypatch.setattr(tasks, "get_session_local", lambda: factory)
    monkeypatch.setattr(
        tasks,
        "_resolve_task_or_404",
        lambda task_id, _principal, db: db.get(Task, task_id),
    )
    with forbid_legacy_content(factory.kw["bind"]):
        loaded = setup.load_task_setup_snapshot_sync(
            tid, owner, before_turn_id="current"
        )
        assert [m["content"] for m in loaded.conversation_history] == [
            "earlier",
            "answer",
        ]
        assert await store.load_latest_checkpoint(str(tid)) == checkpoint
        replay = websocket._load_historical_stream_snapshot_sync(
            tid, actor_user_id=owner, actor_is_admin=False
        )
        assert "earlier" in repr(replay.events) and "answer" in repr(replay.events)
        assert "wrong legacy body" not in repr(replay.events)
        steps = tasks._load_task_steps_snapshot(tid, None)
        assert steps.storage_version == 2
        with factory() as db:
            user = db.get(User, owner)
            detail = await get_conversation_log_detail(tid, db=db, user=user)
            assert [m["content"] for m in detail["transcript"]] == [
                "earlier",
                "answer",
                "current",
            ]
            child = chat.get_task_agent_execution(tid, "child", db=db, user=user)
            grouped = await workforces.get_workforce_agent_execution(
                workforce_id, tid, "child", db=db, user=user
            )
            assert [e["event_id"] for e in child["trace_events"]] == ["child-start"]
            assert grouped == child


def test_v2_assistant_retry_uses_committed_fact(canonical):
    from xagent.web.services.chat_history_service import (
        persist_assistant_message_no_commit,
    )
    from xagent.web.services.task_execution_event_writer import append_fact_no_commit

    factory, tid = canonical
    with factory() as db:
        owner = db.get(Task, tid).user_id
        fact = append_fact_no_commit(
            db,
            task_id=tid,
            kind="assistant_message",
            key="answer",
            payload={
                "content": "committed answer",
                "attachments": [],
                "interactions": None,
            },
        )
        event_id = fact.event_id
        first = persist_assistant_message_no_commit(
            db,
            tid,
            owner,
            "committed answer",
            content_is_reconciled=True,
            execution_event_id=event_id,
        )
        db.commit()
        identity = first.id
    with factory() as db, forbid_legacy_content(db.bind):
        retry = persist_assistant_message_no_commit(
            db,
            tid,
            owner,
            "committed answer",
            content_is_reconciled=True,
            execution_event_id=event_id,
        )
        assert retry.id == identity
        assert retry.content == "committed answer"


def test_existing_execution_accepts_input_with_its_command(canonical, monkeypatch):
    from types import SimpleNamespace

    from xagent.web.models.task_command import TaskExecutionCommand
    from xagent.web.models.task_execution_event import TaskExecutionEvent
    from xagent.web.services import task_existing_command as existing

    factory, tid = canonical
    monkeypatch.setattr(
        existing,
        "get_task_event_bridge",
        lambda: SimpleNamespace(require_ready=lambda: None),
    )
    monkeypatch.setattr(existing, "notify_task_command_dispatcher", lambda: None)
    with factory() as db:
        owner = db.get(Task, tid).user_id
    existing.enqueue_existing_execution(
        task_id=tid,
        task_owner_user_id=owner,
        task_description="existing description",
        context={},
        actor_user_id=owner,
    )
    with factory() as db:
        command = db.query(TaskExecutionCommand).one()
        accepted = db.query(TaskExecutionEvent).filter_by(kind="input_accepted").one()
        assert accepted.turn_id == command.command_id
        assert accepted.run_id == command.target_run_id
        assert accepted.payload["content"] == "existing description"


def test_completed_turn_settlement_does_not_read_legacy_answer(canonical):
    from xagent.web.models.task import TaskStatus
    from xagent.web.services.task_execution_event_writer import append_fact_no_commit
    from xagent.web.services.task_orchestrator import finish_turn

    factory, tid = canonical
    with factory() as db:
        task = db.get(Task, tid)
        task.status = TaskStatus.COMPLETED
        task.run_id = "current"
        task.error_message = "stale"
        for run, content in (("current", "committed answer"), ("old", "old answer")):
            append_fact_no_commit(
                db,
                task_id=tid,
                run_id=run,
                kind="assistant_message",
                key=run,
                payload={"content": content, "message_type": "assistant_response"},
            )
        db.add(
            TaskChatMessage(
                task_id=tid,
                user_id=task.user_id,
                role="assistant",
                content="wrong projection",
                message_type="assistant_response",
            )
        )
        db.commit()
    with factory() as db, forbid_legacy_content(db.bind):
        assert finish_turn(db, tid)
        task = db.get(Task, tid)
        assert task.output == "committed answer"
        assert task.error_message is None


def test_content_tripwire_checks_aliases_predicates_and_actual_bound_version(canonical):
    from sqlalchemy.orm import aliased, defer

    factory, tid = canonical
    with factory() as db, forbid_legacy_content(db.bind):
        alias = aliased(TaskChatMessage)
        with pytest.raises(AssertionError, match="Legacy content read"):
            db.execute(sa.select(alias.content))
        with pytest.raises(AssertionError, match="Legacy content read"):
            db.execute(
                sa.select(TaskChatMessage.id).where(TaskChatMessage.content == "secret")
            )
        with pytest.raises(AssertionError, match="Legacy content read"):
            db.execute(
                sa.select(TraceEvent.id).where(
                    TraceEvent.data["content"].as_string() == "secret"
                )
            )
        db.execute(
            sa.select(TraceEvent.id).where(
                TraceEvent.data["checkpoint_type"].as_string() == "state"
            )
        )
        with pytest.raises(AssertionError, match="Legacy content read"):
            db.execute(
                sa.select(TraceEvent.id).where(
                    TraceEvent.data["content"].as_string() == "secret"
                )
            )
        db.execute(
            sa.select(TaskChatMessage).options(
                defer(TaskChatMessage.content),
                defer(TaskChatMessage.attachments),
                defer(TaskChatMessage.interactions),
            )
        )
        for version in (1, 2):
            statement = sa.select(TaskChatMessage).where(
                TaskChatMessage.task_id.in_(
                    sa.select(Task.id).where(
                        Task.conversation_storage_version == version
                    )
                )
            )
            if version == 1:
                db.execute(statement)
            else:
                with pytest.raises(AssertionError, match="Legacy content read"):
                    db.execute(statement)


def test_ambiguous_command_acceptance_uses_event_content(canonical):
    from xagent.web.models.task import TaskStatus
    from xagent.web.services.task_command_execution import (
        _reconcile_command_acceptance_graph,
    )

    factory, tid = canonical
    with factory() as db:
        task = db.get(Task, tid)
        owner = task.user_id
        task.status = TaskStatus.RUNNING
        task.run_id = "run"
        claim_user_message_delivery_no_commit(
            db, tid, owner, "accepted", turn_id="turn"
        )
        db.commit()
        db.execute(
            sa.update(TaskChatMessage)
            .where(TaskChatMessage.task_id == tid)
            .values(content="obsolete")
        )
        db.commit()
    with forbid_legacy_content(factory.kw["bind"]):
        assert _reconcile_command_acceptance_graph(
            task_id=tid,
            task_owner_user_id=owner,
            turn_id="turn",
            content="accepted",
            file_ids=[],
            expected_run_id="run",
            expected_status=TaskStatus.RUNNING,
        )


@pytest.mark.asyncio
async def test_checkpoint_pruning_preserves_event_recovery_and_blob_cache(
    canonical, monkeypatch
):
    from xagent.core.agent.checkpoint import TraceCheckpointStore
    from xagent.web.models.task import TraceCheckpointBlob, TraceEvent, TraceMessageBlob
    from xagent.web.models.task_execution_event import TaskExecutionEvent
    from xagent.web.services import trace_handlers

    factory, tid = canonical
    monkeypatch.setattr(trace_handlers, "get_checkpoint_history_limit", lambda: 1)
    store = TraceCheckpointStore(tracer_for(tid))
    checkpoints = []
    for index in range(3):
        snapshot = {
            "execution_id": str(tid),
            "context": {
                "messages": [
                    {"role": "user", "content": f"message-{index}-" + "x" * 4000}
                ],
                "metadata": {"memory": "m" * 4000},
            },
            "pattern_state": {},
            "label": f"checkpoint-{index}",
        }
        checkpoints.append(snapshot)
        with forbid_legacy_content(factory.kw["bind"]):
            await store.save(snapshot)
    with factory() as db:
        assert db.query(TraceEvent).count() == 1
        events = (
            db.query(TaskExecutionEvent)
            .filter_by(kind="recovery_state")
            .order_by(TaskExecutionEvent.sequence)
            .all()
        )
        assert len(events) == 3
        payloads = [event.payload for event in events]
        assert db.query(TraceMessageBlob).count() == 3
        assert db.query(TraceCheckpointBlob).count() > 0
        db.execute(sa.delete(TraceEvent))
        db.execute(sa.delete(TraceMessageBlob))
        db.execute(sa.delete(TraceCheckpointBlob))
        db.commit()
    with forbid_legacy_content(factory.kw["bind"]):
        assert await store.load_latest_checkpoint(str(tid)) == checkpoints[-1]
    with factory() as db:
        assert [
            event.payload
            for event in db.query(TaskExecutionEvent)
            .filter_by(kind="recovery_state")
            .order_by(TaskExecutionEvent.sequence)
        ] == payloads


def test_existing_execution_rolls_back_input_if_command_staging_fails(
    canonical, monkeypatch
):
    from types import SimpleNamespace

    from xagent.web.models.task_command import TaskExecutionCommand
    from xagent.web.models.task_execution_event import TaskExecutionEvent

    factory, tid = canonical
    monkeypatch.setattr(
        task_existing_command,
        "get_task_event_bridge",
        lambda: SimpleNamespace(require_ready=lambda: None),
    )

    def fail(*args, **kwargs):
        raise RuntimeError("command staging failed")

    monkeypatch.setattr(task_existing_command, "stage_task_start_command", fail)
    with factory() as db:
        owner = db.get(Task, tid).user_id
        previous = (db.get(Task, tid).status, db.get(Task, tid).state_version)
    with pytest.raises(RuntimeError, match="command staging failed"):
        task_existing_command.enqueue_existing_execution(
            task_id=tid,
            task_owner_user_id=owner,
            task_description="existing",
            context={},
            actor_user_id=owner,
        )
    with factory() as db:
        assert db.query(TaskExecutionEvent).count() == 0
        assert db.query(TaskExecutionCommand).count() == 0
        assert (db.get(Task, tid).status, db.get(Task, tid).state_version) == previous


@pytest.mark.asyncio
@pytest.mark.parametrize("shared", [False, True])
async def test_existing_execution_applies_current_input_once(
    canonical, monkeypatch, shared
):
    from unittest.mock import AsyncMock

    from xagent.web.models.task_execution_event import TaskExecutionEvent
    from xagent.web.services import (
        task_event_display,
        task_execution,
        task_orchestrator,
        task_setup_snapshot,
    )
    from xagent.web.services.task_lease_service import acquire_task_lease_isolated

    factory, tid = canonical
    monkeypatch.setattr(task_setup_snapshot, "get_session_local", lambda: factory)
    monkeypatch.setattr(task_orchestrator, "resolve_execution_scope", lambda *_: None)
    monkeypatch.setattr(task_orchestrator, "_get_agent_manager", lambda: None)
    monkeypatch.setattr(task_event_display, "publish_task_result", AsyncMock())
    monkeypatch.setattr(
        task_orchestrator, "get_shared_task_execution_enabled", lambda: shared
    )
    observed = []
    payload = task_orchestrator.TaskTurnPayload("current description")
    with factory() as db:
        owner = db.get(Task, tid).user_id
        persist_user_message_no_commit(db, tid, owner, "earlier", turn_id="earlier")
        db.commit()
    from xagent.core.agent.checkpoint import TraceCheckpointStore

    await TraceCheckpointStore(tracer_for(tid)).save(
        {
            "execution_id": str(tid),
            "context": {
                "messages": [
                    {
                        "role": "user",
                        "content": "earlier",
                        "metadata": {"turn_id": "earlier"},
                    }
                ]
            },
            "pattern_state": {},
        }
    )
    lease = None
    if shared:
        # Shared START already owns the lease and accepted fact at handoff.
        from xagent.web.services.task_execution_event_writer import (
            append_fact_no_commit,
        )

        lease = acquire_task_lease_isolated(tid)
        with factory() as db:
            append_fact_no_commit(
                db,
                task_id=tid,
                run_id=lease.run_id,
                turn_id=payload.turn_id,
                kind="input_accepted",
                key=f"message:{payload.turn_id}",
                payload={
                    "role": "user",
                    "content": payload.transcript_message,
                    "turn_id": payload.turn_id,
                },
            )
            db.commit()

    async def execute(**kwargs):
        with factory() as db:
            facts = (
                db.query(TaskExecutionEvent)
                .filter_by(kind="input_accepted", turn_id=payload.turn_id)
                .all()
            )
            observed.append(
                (
                    [
                        m["content"]
                        for m in kwargs["task_setup_snapshot"].conversation_history
                    ],
                    [f.payload["content"] for f in facts],
                    kwargs["context"]["turn_id"],
                )
            )

    monkeypatch.setattr(task_execution, "execute_task_background", execute)
    with forbid_legacy_content(factory.kw["bind"]):
        if shared:
            handle = task_orchestrator._schedule_bg(
                task_id=tid,
                task_owner_user_id=owner,
                task_source=None,
                task_lease=lease,
                payload=payload,
                force_fresh=False,
                context={},
            )
        else:
            handle = await task_orchestrator.TaskTurnOrchestrator.schedule_existing_task_execution(
                task_id=tid,
                task_owner_user_id=owner,
                task_source=None,
                payload=payload,
                context={},
            )
        await handle
    assert observed == [(["earlier"], ["current description"], payload.turn_id)]


@pytest.mark.parametrize("damage", ["missing", "version", "content"])
def test_delivery_inspection_refuses_missing_or_invalid_facts(canonical, damage):
    from xagent.web.models.task_execution_event import TaskExecutionEvent

    factory, tid = canonical
    with factory() as db:
        owner = db.get(Task, tid).user_id
        persist_user_message_no_commit(db, tid, owner, "accepted", turn_id="turn")
        db.commit()
        event = db.query(TaskExecutionEvent).filter_by(kind="input_accepted").one()
        if damage == "missing":
            db.delete(event)
        elif damage == "version":
            event.payload_version = 99
        else:
            event.payload = {"content": None}
        db.commit()
    with factory() as db, forbid_legacy_content(db.bind):
        with pytest.raises(
            ValueError, match="Missing accepted input|Unsupported chat fact"
        ):
            inspect_user_message_delivery(
                db, tid, "accepted", attachments=None, turn_id="turn"
            )


def test_local_existing_input_fences_replaced_lease_and_rolls_back(
    canonical, monkeypatch
):
    from xagent.web.models.task_execution_event import TaskExecutionEvent
    from xagent.web.services import task_execution_event_writer as writer
    from xagent.web.services.task_lease_service import (
        TaskLeaseLostError,
        acquire_task_lease_isolated,
    )
    from xagent.web.services.task_orchestrator import (
        TaskTurnPayload,
        _accept_existing_event_input_sync,
    )

    factory, tid = canonical
    with factory() as db:
        owner = db.get(Task, tid).user_id
    lease = acquire_task_lease_isolated(tid)
    payload = TaskTurnPayload("existing")
    original = writer.append_fact_no_commit

    def fail_after_append(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("acceptance write failed")

    monkeypatch.setattr(writer, "append_fact_no_commit", fail_after_append)
    with pytest.raises(RuntimeError, match="acceptance write failed"):
        _accept_existing_event_input_sync(lease, owner, payload)
    with factory() as db:
        assert db.query(TaskExecutionEvent).count() == 0
        db.get(Task, tid).lease_attempt_id = "replacement"
        db.commit()
    monkeypatch.setattr(writer, "append_fact_no_commit", original)
    with pytest.raises(TaskLeaseLostError):
        _accept_existing_event_input_sync(lease, owner, payload)
    with factory() as db:
        assert db.query(TaskExecutionEvent).count() == 0
        assert db.get(Task, tid).lease_attempt_id == "replacement"


@pytest.mark.asyncio
async def test_monitor_uses_v1_branch_only_for_legacy_content(canonical):
    from tests.web.services.test_task_event_display import (
        test_monitor_counts_each_task_source_once_and_isolates_child_scope,
    )

    factory, _ = canonical
    with forbid_legacy_content(factory.kw["bind"]):
        await test_monitor_counts_each_task_source_once_and_isolates_child_scope(
            canonical
        )


def test_monitor_tripwire_checks_cached_source_version(canonical):
    from xagent.web.services.task_event_metrics import monitoring_trace_source

    factory, _ = canonical
    statement = sa.select(monitoring_trace_source())
    with factory() as db, forbid_legacy_content(db.bind):
        db.execute(statement)
        params = statement.compile(db.bind).params
        key = next(
            key for key in params if key.startswith("conversation_storage_version_")
        )
        params[key] = 2
        with pytest.raises(AssertionError, match="Legacy content read"):
            db.execute(statement, params)


@pytest.mark.parametrize("reader", ["model", "display"])
def test_fixed_horizon_excludes_another_connection_commit_between_pages(
    canonical, monkeypatch, reader
):
    from xagent.web.services.task_event_context_service import load_task_event_context
    from xagent.web.services.task_event_display import load_event_display_snapshot

    factory, tid = canonical
    with factory() as db:
        for index in range(105):
            fact(
                db,
                tid,
                "assistant_message",
                f"message-{index}",
                {"content": f"answer-{index}", "message_type": "assistant_response"},
                flat=True,
            )
        db.commit()
    original = factory.class_.scalars
    appended = False

    def scalars(session, statement, *args, **kwargs):
        nonlocal appended
        result = original(session, statement, *args, **kwargs)
        if not appended and "ORDER BY task_execution_events.sequence" in str(statement):
            # Finish the current cursor, then commit on a separate connection
            # before the caller can request its next page. No sleep/race timing.
            page = list(result)
            appended = True
            with factory() as writer:
                fact(
                    writer,
                    tid,
                    "assistant_message",
                    "later",
                    {"content": "later", "message_type": "assistant_response"},
                    flat=True,
                )
                writer.commit()
            return iter(page)
        return result

    monkeypatch.setattr(factory.class_, "scalars", scalars)
    with factory() as db, forbid_legacy_content(db.bind):
        if reader == "model":
            view = load_task_event_context(db, tid)
            messages = view.messages
        else:
            view = load_event_display_snapshot(db, tid)
            assert view.horizon == 105
            messages = view.messages
        assert [message["content"] for message in messages] == [
            f"answer-{index}" for index in range(105)
        ]
    assert appended
    with factory() as db:
        assert db.get(Task, tid).conversation_event_sequence == 106
