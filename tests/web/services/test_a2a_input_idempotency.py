"""First A2A inputs retain their identity across durable acceptance retries."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import Mock

import pytest
from sqlalchemy.orm import sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from xagent.web.models import database
from xagent.web.models.agent import Agent
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services import (
    a2a_task_read,
    task_event_bridge,
    task_orchestrator,
    task_start,
)

engine = engine_fixture


@pytest.fixture
def ingress(engine, monkeypatch):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(task_event_bridge, "_bridge", Mock())
    monkeypatch.setattr(
        task_orchestrator,
        "_schedule_bg",
        Mock(side_effect=AssertionError("local execution")),
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    for module in (database, task_start, a2a_task_read):
        monkeypatch.setattr(module, "get_session_local", lambda: sessions)
    with sessions() as db:
        user = User(username="owner", password_hash="unused")
        db.add(user)
        db.flush()
        agent = Agent(user_id=user.id, name="A2A")
        db.add(agent)
        db.commit()
        args = dict(
            agent_id=agent.id,
            task_owner_user_id=user.id,
            agent_execution_mode="balanced",
            text="hello",
            message_id="message-1",
            key_prefix="key-one",
            context_id=None,
            task_id=None,
        )
    yield sessions, args
    Base.metadata.drop_all(engine)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status", [TaskStatus.PENDING, TaskStatus.COMPLETED, TaskStatus.FAILED]
)
async def test_first_message_replay_returns_original_task_without_scheduling(
    ingress, status
):
    sessions, args = ingress
    first = await task_start.start_a2a_turn(**args)
    with sessions() as db:
        db.get(Task, first.id).status = status
        db.commit()
    replay = await task_start.start_a2a_turn(**args)
    assert replay.id == first.id
    assert replay.status == status
    with sessions() as db:
        assert db.query(Task).count() == 1
        assert db.query(TaskExecutionCommand).count() == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", [{"text": "different"}, {"context_id": "different-context"}]
)
async def test_same_message_with_changed_input_is_conflict(ingress, change):
    sessions, args = ingress
    await task_start.start_a2a_turn(**args)
    with pytest.raises(task_start.TaskStartRejected, match="a2a_input_conflict"):
        await task_start.start_a2a_turn(**(args | change))
    with sessions() as db:
        assert db.query(Task).count() == 1


@pytest.mark.asyncio
async def test_different_message_identity_is_a_new_task(ingress):
    _, args = ingress
    first = await task_start.start_a2a_turn(**args)
    second = await task_start.start_a2a_turn(**(args | {"message_id": "message-2"}))
    assert first.id != second.id


@pytest.mark.asyncio
async def test_same_message_is_isolated_by_agent_and_owner(ingress):
    sessions, args = ingress
    first = await task_start.start_a2a_turn(**args)
    with sessions() as db:
        another = Agent(user_id=args["task_owner_user_id"], name="another")
        db.add(another)
        db.commit()
        agent_id = another.id
    second = await task_start.start_a2a_turn(**(args | {"agent_id": agent_id}))
    with sessions() as db:
        owner = User(username="new-owner", password_hash="unused")
        db.add(owner)
        db.flush()
        db.get(Agent, args["agent_id"]).user_id = owner.id
        db.commit()
        owner_id = owner.id
    third = await task_start.start_a2a_turn(**(args | {"task_owner_user_id": owner_id}))
    assert len({first.id, second.id, third.id}) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("delete_command", [False, True])
async def test_deleted_input_target_remains_a_tombstone(ingress, delete_command):
    from xagent.web.models.task_input_receipt import TaskInputReceipt

    sessions, args = ingress
    first = await task_start.start_a2a_turn(**args)
    with sessions() as db:
        if delete_command:
            db.query(TaskExecutionCommand).delete(synchronize_session=False)
        else:
            db.query(Task).filter_by(id=first.id).delete(synchronize_session=False)
        db.commit()
        assert db.query(TaskInputReceipt).count() == 1
    with pytest.raises(task_start.TaskStartRejected, match="a2a_input_unavailable"):
        await task_start.start_a2a_turn(**args)
    with sessions() as db:
        assert db.query(TaskExecutionCommand).count() == 0
        assert db.query(Task).count() == int(delete_command)


@pytest.mark.asyncio
async def test_owner_transfer_does_not_expose_original_input(ingress):
    sessions, args = ingress
    first = await task_start.start_a2a_turn(**args)
    with sessions() as db:
        owner = User(username="replacement", password_hash="unused")
        db.add(owner)
        db.flush()
        db.get(Task, first.id).user_id = owner.id
        db.commit()
    with pytest.raises(task_start.TaskStartRejected, match="a2a_input_unavailable"):
        await task_start.start_a2a_turn(**args)


def test_concurrent_first_requests_commit_only_one_graph(ingress, engine):
    from sqlalchemy import event

    from xagent.web.models.chat_message import TaskChatMessage
    from xagent.web.models.task_input_receipt import TaskInputReceipt

    sessions, args = ingress
    barrier = Barrier(2)

    def before_insert(_conn, _cursor, statement, _parameters, _context, _many):
        if statement.startswith("INSERT INTO task_input_receipts"):
            barrier.wait(timeout=10)

    event.listen(engine, "before_cursor_execute", before_insert)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(task_start._prepare_a2a_turn_sync, **args)
                for _ in range(2)
            ]
            results = [future.result(timeout=20) for future in futures]
    finally:
        event.remove(engine, "before_cursor_execute", before_insert)
    ids = [
        result.task.id
        if isinstance(result, task_start._A2ATurnPreparation)
        else result.id
        for result in results
    ]
    assert ids[0] == ids[1]
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 1
        assert db.query(Task).count() == 1
        assert db.query(TaskExecutionCommand).count() == 1
        assert db.query(TaskChatMessage).count() == 1


@pytest.mark.asyncio
async def test_acceptance_failure_rolls_back_receipt_and_task(ingress, monkeypatch):
    from xagent.web.models.task_input_receipt import TaskInputReceipt

    sessions, args = ingress
    original = task_orchestrator.TaskTurnOrchestrator.claim_created_turn_no_commit
    monkeypatch.setattr(
        task_orchestrator.TaskTurnOrchestrator,
        "claim_created_turn_no_commit",
        Mock(side_effect=RuntimeError("reject admission")),
    )
    with pytest.raises(RuntimeError, match="reject admission"):
        await task_start.start_a2a_turn(**args)
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 0
        assert db.query(Task).count() == 0
        assert db.query(TaskExecutionCommand).count() == 0
    monkeypatch.setattr(
        task_orchestrator.TaskTurnOrchestrator, "claim_created_turn_no_commit", original
    )
    await task_start.start_a2a_turn(**args)


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", [False, True])
async def test_uncertain_commit_reconciles_only_durable_receipt(
    ingress, engine, monkeypatch, committed
):
    from sqlalchemy.orm import Session

    from xagent.web.models.task_input_receipt import TaskInputReceipt

    sessions, args = ingress
    failed = False

    class UncertainSession(Session):
        def commit(self):
            nonlocal failed
            if not failed:
                failed = True
                if committed:
                    super().commit()
                raise RuntimeError("commit acknowledgement lost")
            return super().commit()

    uncertain = sessionmaker(bind=engine, class_=UncertainSession)
    monkeypatch.setattr(task_start, "get_session_local", lambda: uncertain)
    if committed:
        first = await task_start.start_a2a_turn(**args)
    else:
        with pytest.raises(RuntimeError, match="commit acknowledgement lost"):
            await task_start.start_a2a_turn(**args)
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == int(committed)
        assert db.query(Task).count() == int(committed)
    retry = await task_start.start_a2a_turn(**args)
    if committed:
        assert retry.id == first.id
    with sessions() as db:
        assert db.query(TaskInputReceipt).count() == 1
        assert db.query(TaskExecutionCommand).count() == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["hello", "different input"])
async def test_different_key_has_independent_message_identity(ingress, text):
    sessions, args = ingress
    first = await task_start.start_a2a_turn(**args)
    other = args | {"key_prefix": "key-two", "text": text}
    second = await task_start.start_a2a_turn(**other)
    replay = await task_start.start_a2a_turn(**other)
    assert second.id != first.id
    assert replay.id == second.id
    with sessions() as db:
        assert db.query(Task).count() == 2
        assert db.query(TaskExecutionCommand).count() == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_context", [None, "caller-context"])
async def test_retry_with_returned_context_preserves_original_input(
    ingress, initial_context
):
    sessions, args = ingress
    args = args | {"context_id": initial_context}
    first = await task_start.start_a2a_turn(**args)
    returned = task_start.task_context_id(first)
    retry = args | {"context_id": returned}
    replay = await task_start.start_a2a_turn(**retry)
    assert replay.id == first.id
    for change in ({"text": "changed"}, {"context_id": "wrong-context"}):
        with pytest.raises(task_start.TaskStartRejected, match="a2a_input_conflict"):
            await task_start.start_a2a_turn(**(retry | change))
    if initial_context is not None:
        with pytest.raises(task_start.TaskStartRejected, match="a2a_input_conflict"):
            await task_start.start_a2a_turn(**(args | {"context_id": None}))
    with sessions() as db:
        assert db.query(Task).count() == 1
        db.query(Task).delete(synchronize_session=False)
        db.commit()
    with pytest.raises(task_start.TaskStartRejected, match="a2a_input_unavailable"):
        await task_start.start_a2a_turn(**retry)
