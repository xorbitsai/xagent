"""Row-level redaction and purge on both SQLite and PostgreSQL.

``redact_task_rows`` and ``purge_task_rows`` share their content-row delete,
whose statement order exists for foreign keys and the active-interaction
CHECK. Both are run here against both dialects with foreign keys enforced,
starting from a task whose checkpoint pointer and active interaction both
name its trace row -- the shape that order exists for.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from tests.web.services.task_interaction_schema_shared import make_row
from xagent.web.models.agent import Agent
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.task import (
    DAGExecution,
    Task,
    TaskConnectorRuntimeContext,
    TaskStatus,
    TraceCheckpointBlob,
    TraceEvent,
    TraceMessageBlob,
)
from xagent.web.models.task_channel_delivery import TaskChannelDelivery
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.models.trigger import AgentTrigger, TriggerRun, TriggerRunStatus
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.models.user_channel import UserChannel
from xagent.web.services import channel_delivery
from xagent.web.services.connector_runtime import prepare_append_connector_runtime
from xagent.web.services.task_deletion import purge_task_rows, redact_task_rows
from xagent.web.services.task_file_lifecycle import lock_attachment_task
from xagent.web.services.task_runtime import PROMPT_CONTEXT_AGENT_CONFIG_KEY

engine = engine_fixture

NOW = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)
UPDATED_AT = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

#: Routing and authorization keys survive; prompt context and the creator IP
#: are content.
AGENT_CONFIG = {
    "auth_mode": "widget",
    "guest_id": "guest-1",
    PROMPT_CONTEXT_AGENT_CONFIG_KEY: {"text": "secret"},
    "widget_client_ip": "203.0.113.7",
}

CONTENT_TABLES = (
    TraceEvent,
    TraceMessageBlob,
    TraceCheckpointBlob,
    DAGExecution,
    TaskChatMessage,
    TaskExecutionEvent,
    TaskInteractionRequest,
)


@pytest.fixture
def sessions(engine) -> sessionmaker[Session]:
    """Sessions shaped like production's: ``autoflush=False``."""
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False)


def _seed(db: Session) -> tuple[int, int, int]:
    """A completed task with content, an anchored active interaction, a file,
    a trigger run and a channel delivery still owed.

    Returns ``(task_id, owner_id, channel_id)``.
    """
    user = User(username="owner", password_hash="unused")
    db.add(user)
    db.flush()
    agent = Agent(user_id=user.id, name="agent")
    channel = UserChannel(
        user_id=user.id, channel_type="slack", channel_name="ws", config={}
    )
    db.add_all([agent, channel])
    db.flush()
    task = Task(
        user_id=user.id,
        agent_id=agent.id,
        title="secret",
        input="secret",
        output="secret",
        status=TaskStatus.COMPLETED,
        input_tokens=7,
        output_tokens=3,
        agent_config=AGENT_CONFIG,
    )
    db.add(task)
    db.flush()
    trigger = AgentTrigger(
        user_id=user.id, agent_id=agent.id, type="scheduled", name="t", config={}
    )
    db.add(trigger)
    db.flush()
    command = TaskExecutionCommand(
        task_id=task.id,
        command_id="turn-1",
        kind="message",
        payload={"content": "secret", "channel": {"id": channel.id}},
        status="completed",
    )
    trace = TraceEvent(
        task_id=task.id,
        event_id="evt-1",
        event_type="agent_execution_checkpoint",
        timestamp=NOW,
        data={"content": "secret"},
    )
    db.add_all([command, trace])
    db.flush()
    db.add_all(
        [
            TraceMessageBlob(
                task_id=task.id,
                execution_id="exec",
                message_hash="m",
                message_data={},
                message_bytes=2,
            ),
            TraceCheckpointBlob(
                task_id=task.id,
                execution_id="exec",
                blob_kind="messages",
                blob_hash="c",
                blob_data={},
                blob_bytes=2,
            ),
            DAGExecution(task_id=task.id),
            TaskChatMessage(
                task_id=task.id,
                user_id=user.id,
                role="user",
                content="secret",
                message_type="user_message",
            ),
            TaskExecutionEvent(
                event_id="exec-event-1",
                task_id=task.id,
                sequence=1,
                scope_id="root",
                idempotency_key="k1",
                kind="assistant_message",
                payload_version=1,
                payload={"content": "secret"},
                occurred_at=NOW,
            ),
            TaskInteractionRequest(
                **make_row(task_id=task.id, resume_trace_event_id=trace.id, db=db)
            ),
            # No durable key and a path outside the uploads root: the
            # redaction deletes the row and has no bytes to touch.
            UploadedFile(
                user_id=user.id,
                task_id=task.id,
                filename="a.txt",
                storage_path="/nonexistent/a.txt",
                file_size=1,
            ),
            TriggerRun(
                trigger_id=trigger.id,
                task_id=task.id,
                status=TriggerRunStatus.COMPLETED.value,
                idempotency_key=f"run-{task.id}",
                payload_snapshot={"text": "secret"},
            ),
            # A reply still owed to the channel. The delivery worker reads
            # its destination out of the command payload the redaction
            # blanks.
            TaskChannelDelivery(
                command_id=command.id, channel_id=channel.id, destination={}
            ),
        ]
    )
    db.execute(
        sa.update(Task)
        .where(Task.id == task.id)
        .values(
            last_checkpoint_event_id="evt-1",
            last_checkpoint_trace_event_id=trace.id,
            interaction_protocol_version=1,
            updated_at=UPDATED_AT,
        )
    )
    db.commit()
    return int(task.id), int(user.id), int(channel.id)


def _redact(sessions: sessionmaker[Session], task_id: int, now: datetime) -> None:
    with sessions() as db:
        task = lock_attachment_task(db, task_id, deleting=True)
        assert task is not None
        redact_task_rows(db, task, now=now)
        db.commit()


def _utc(value: datetime) -> datetime:
    """A stored timestamp as an aware UTC value on either dialect.

    SQLite hands back naive UTC; PostgreSQL an aware value in the session's
    zone, which ``replace`` would mislabel rather than convert.
    """
    return (
        value.replace(tzinfo=timezone.utc)
        if value.tzinfo is None
        else value.astimezone(timezone.utc)
    )


def _count(db: Session, model: type, task_id: int) -> int:
    return int(
        db.execute(
            sa.select(sa.func.count())
            .select_from(model)
            .where(model.task_id == task_id)  # type: ignore[attr-defined]
        ).scalar_one()
    )


def test_redact_task_rows_keeps_the_task_and_removes_its_content(sessions) -> None:
    with sessions() as db:
        task_id, owner_id, _ = _seed(db)

    _redact(sessions, task_id, NOW)
    # Idempotency: a re-run removes nothing and keeps the first trace stamp.
    _redact(sessions, task_id, NOW + timedelta(days=1))

    with sessions() as db:
        assert {
            model.__tablename__: _count(db, model, task_id) for model in CONTENT_TABLES
        } == {model.__tablename__: 0 for model in CONTENT_TABLES}
        assert (
            db.query(UploadedFile).filter(UploadedFile.user_id == owner_id).count() == 0
        )
        task = db.get(Task, task_id)
        assert task is not None
        assert (task.title, task.input, task.output) == ("", None, None)
        assert (task.input_tokens, task.output_tokens) == (7, 3)
        assert task.agent_config == {"auth_mode": "widget", "guest_id": "guest-1"}
        assert _utc(task.updated_at) == UPDATED_AT
        assert _utc(task.traces_expired_at) == NOW
        [command] = db.query(TaskExecutionCommand).filter_by(task_id=task_id).all()
        assert command.payload == {}
        [run] = db.query(TriggerRun).filter_by(task_id=task_id).all()
        assert run.payload_snapshot is None


def test_redaction_discards_the_pending_channel_delivery(sessions) -> None:
    """A blanked command's delivery must leave the delivery worker's queue.

    ``channel_delivery._pending`` picks pending deliveries of settled
    commands, and the claim then reads the channel out of the command
    payload; with that payload blanked, a still-pending row would fail its
    claim on every pass and starve the rest of the channel's queue.
    """
    with sessions() as db:
        task_id, _, channel_id = _seed(db)
        command_id = int(db.query(TaskChannelDelivery.command_id).one()[0])

    with patch.object(channel_delivery, "get_session_local", return_value=sessions):
        assert channel_delivery._pending(channel_id) == [command_id]
        _redact(sessions, task_id, NOW)
        assert channel_delivery._pending(channel_id) == []

    with sessions() as db:
        [delivery] = db.query(TaskChannelDelivery).all()
        assert (delivery.status, delivery.claim_token) == ("discarded", None)


def test_purge_task_rows_still_deletes_the_task_and_detaches_files(sessions) -> None:
    with sessions() as db:
        task_id, owner_id, _ = _seed(db)

    with sessions() as db:
        assert purge_task_rows(db, task_id=task_id, detached_reason="task_deleted")
        db.commit()

    with sessions() as db:
        assert db.get(Task, task_id) is None
        assert all(_count(db, model, task_id) == 0 for model in CONTENT_TABLES)
        [file] = db.query(UploadedFile).filter(UploadedFile.user_id == owner_id).all()
        assert (file.task_id, file.detached_reason) == (None, "task_deleted")


def test_a_redacted_connector_bound_task_can_take_another_turn(sessions) -> None:
    """The connector runtime context survives: the next turn is checked
    against it, and without it a required context key could never be
    satisfied again (resending it is refused as a change)."""
    with sessions() as db:
        owner = User(username="connector-owner", password_hash="unused")
        db.add(owner)
        db.flush()
        server = MCPServer(
            name="runtime",
            description="runtime",
            managed="external",
            transport="streamable_http",
            url="https://example.com/mcp",
            runtime_input_schema={
                "context": {"account_id": {"type": "string", "required": True}}
            },
            runtime_bindings=[
                {
                    "source": {"input_type": "context", "key": "account_id"},
                    "target": {"target_type": "mcp_meta", "key": "account_id"},
                }
            ],
        )
        agent = Agent(
            user_id=owner.id,
            name="connector agent",
            instructions="Use tools.",
            tool_categories=["mcp"],
        )
        db.add_all([server, agent])
        db.flush()
        db.add(
            UserMCPServer(
                user_id=owner.id,
                mcpserver_id=server.id,
                is_owner=True,
                can_edit=True,
                can_delete=True,
                is_active=True,
            )
        )
        ref = {"connector_type": "mcp", "connector_id": int(server.id)}
        task = Task(
            user_id=owner.id,
            agent_id=agent.id,
            title="secret",
            source="sdk",
            status=TaskStatus.COMPLETED,
            connector_runtime_selected_refs=[ref],
        )
        db.add(task)
        db.flush()
        db.add(
            TaskConnectorRuntimeContext(
                task_id=task.id,
                connector_type="mcp",
                connector_id=server.id,
                context={"account_id": "acct-1"},
            )
        )
        db.commit()
        task_id, owner_id, agent_id = int(task.id), int(owner.id), int(agent.id)

    _redact(sessions, task_id, NOW)

    with sessions() as db:
        task = db.get(Task, task_id)
        agent = db.get(Agent, agent_id)
        assert task is not None and agent is not None
        for payload_items in (
            None,
            [{"connector_ref": ref, "context": {"account_id": "acct-1"}}],
        ):
            plan = prepare_append_connector_runtime(
                db=db,
                agent=agent,
                task=task,
                connector_user_id=owner_id,
                payload_items=payload_items,
            )
            assert plan.ephemeral_by_ref == {}
