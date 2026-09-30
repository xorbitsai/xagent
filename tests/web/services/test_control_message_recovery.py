"""A restored pending tool call reuses its committed outbound effect."""

import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

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
from xagent.core.agent.checkpoint import ExecutionEventPersistenceError
from xagent.core.agent.context import ExecutionContext
from xagent.core.agent.pattern.react.react import ReActPattern
from xagent.core.agent.runtime import PatternRuntime
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task, TraceEvent
from xagent.web.services import task_execution as websocket
from xagent.web.services.task_execution_event_store import ExecutionEventConflict
from xagent.web.services.task_lease_service import (
    acquire_task_lease,
    bind_task_lease_context,
)

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture


@pytest.fixture
def outbound(canonical, monkeypatch):
    factory, task_id = canonical
    monkeypatch.setattr(websocket, "get_db", lambda: iter([factory()]))
    broadcasts = AsyncMock()
    monkeypatch.setattr(websocket, "publish_task_event", broadcasts)
    handler = websocket.make_agent_outbound_handler(task_id, authoritative=True)
    return factory, task_id, handler, broadcasts


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,wait",
    [("send_message", False), ("send_message", True), ("ask_user_question", True)],
)
@pytest.mark.parametrize(
    "window", ["before_commit", "before_broadcast", "before_checkpoint"]
)
async def test_pending_control_call_recovers_original_message(
    outbound, monkeypatch, name, wait, window
):
    factory, tid, handler, broadcasts = outbound
    pattern = ReActPattern()
    pattern.pending_tool_calls = [
        {
            "id": "provider-call",
            "name": name,
            "args": {
                "message": "Choose a file",
                "expect_response": wait,
                "interactions": [],
            },
            "tool_attempt_id": "attempt-1",
            "assistant_message_id": "batch-1",
        }
    ]
    saved = deepcopy(pattern.get_state())
    runtime = PatternRuntime(execution_id=str(tid), outbound_message_handler=handler)
    runtime.active_react_step_id = "old-step"
    runtime.active_turn_id = "old-turn"
    runtime.checkpoint = AsyncMock()
    with factory() as db:
        db.get(Task, tid).run_id = "old-run"
        db.commit()
    with monkeypatch.context() as failure:
        if window == "before_commit":
            failure.setattr(
                "xagent.web.services.task_execution_event_writer.append_task_execution_event_no_commit",
                lambda *a, **kw: (_ for _ in ()).throw(OSError("crash")),
            )
        elif window == "before_broadcast":
            broadcasts.side_effect = OSError("crash")
        else:
            runtime.checkpoint.side_effect = OSError("crash")
        with pytest.raises((OSError, ExecutionEventPersistenceError)):
            await pattern._execute_pending_tool_calls(
                context=ExecutionContext(), tools=[], llm=None, runtime=runtime
            )
    broadcasts.side_effect = None
    original_id = runtime.outbound_messages[0]["event_id"]
    if window == "before_commit":
        with factory() as db:
            assert facts(db, tid) == []
        broadcasts.assert_not_awaited()
    with factory() as db:
        db.get(Task, tid).run_id = "new-run"
        db.commit()
    restored = ReActPattern()
    restored.load_state(saved)
    resumed = PatternRuntime(execution_id=str(tid), outbound_message_handler=handler)
    resumed.active_react_step_id = "new-step"
    resumed.active_turn_id = "new-turn"
    resumed.checkpoint = AsyncMock()
    await restored._execute_pending_tool_calls(
        context=ExecutionContext(), tools=[], llm=None, runtime=resumed
    )
    assert resumed.outbound_messages[0]["event_id"] == original_id
    assert restored.pending_tool_calls == []
    if wait:
        assert restored.waiting_for_user_request["event_id"] == original_id
    with factory() as db:
        messages = [
            event
            for event in facts(db, tid)
            if event.kind in {"agent_message", "agent_progress"}
        ]
        assert len(messages) == 1
        fact = messages[0]
        assert fact.tool_attempt_id == "attempt-1"
        assert fact.assistant_message_id == "batch-1"
        assert fact.run_id == ("new-run" if window == "before_commit" else "old-run")
        assert db.query(TaskChatMessage).count() == int(wait)
        assert db.query(TraceEvent).count() == 1
        sent = broadcasts.await_args.args[0]
        assert sent["event_id"] == original_id
        assert sent["data"] == fact.payload["data"]
        assert (
            resumed.outbound_messages[0]["metadata"] == fact.payload["data"]["metadata"]
        )


@pytest.mark.asyncio
async def test_replay_rejects_changed_business_content_but_allows_new_attempt(outbound):
    factory, tid, handler, broadcasts = outbound
    runtime = PatternRuntime(outbound_message_handler=handler)

    async def send(message, attempt):
        return await runtime.send_message(
            message=message,
            expect_response=True,
            metadata={"tool_attempt_id": attempt, "assistant_message_id": "batch-1"},
        )

    first = await send("same text", "attempt-1")
    with pytest.raises(ExecutionEventPersistenceError) as error:
        await send("different text", "attempt-1")
    assert isinstance(error.value.__cause__, ExecutionEventConflict)
    second = await send("same text", "attempt-2")
    assert first["event_id"] != second["event_id"]
    assert broadcasts.await_count == 2
    with factory() as db:
        assert len([e for e in facts(db, tid) if e.kind == "agent_message"]) == 2


@pytest.mark.asyncio
async def test_aggregated_tool_question_reuses_ordered_sources(outbound):
    factory, tid, handler, broadcasts = outbound
    pairs = [
        (
            {
                "id": f"call-{n}",
                "name": f"tool-{n}",
                "tool_attempt_id": f"attempt-{n}",
                "assistant_message_id": "batch",
            },
            {"message": "Choose", "interactions": []},
        )
        for n in range(2)
    ]
    identities = []
    for step in ["first", "recovered"]:
        pattern = ReActPattern()
        runtime = PatternRuntime(outbound_message_handler=handler)
        runtime.active_react_step_id = step
        runtime.checkpoint = AsyncMock()
        await pattern._pause_for_tool_results(
            waiting_pairs=deepcopy(pairs), context=ExecutionContext(), runtime=runtime
        )
        identities.append(pattern.waiting_for_user_request["event_id"])
    assert identities[0] == identities[1]
    with factory() as db:
        message = next(e for e in facts(db, tid) if e.kind == "agent_message")
        assert [
            s["tool_attempt_id"]
            for s in message.payload["data"]["metadata"]["tool_calls"]
        ] == ["attempt-0", "attempt-1"]
        assert db.query(TaskChatMessage).count() == 1


@pytest.mark.asyncio
async def test_concurrent_control_replay_commits_one_effect(outbound):
    factory, tid, handler, broadcasts = outbound

    async def send():
        runtime = PatternRuntime(outbound_message_handler=handler)
        return await runtime.send_message(
            message="Choose",
            expect_response=True,
            metadata={"tool_attempt_id": "attempt", "assistant_message_id": "batch"},
        )

    first, second = await asyncio.gather(send(), send())
    assert first == second
    assert broadcasts.await_count == 2
    with factory() as db:
        assert len([e for e in facts(db, tid) if e.kind == "agent_message"]) == 1
        assert db.query(TaskChatMessage).count() == 1
        assert db.query(TraceEvent).count() == 1


@pytest.mark.asyncio
async def test_committed_control_replay_still_rejects_stale_lease(outbound):
    factory, tid, handler, broadcasts = outbound
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    runtime = PatternRuntime(outbound_message_handler=handler)
    with bind_task_lease_context(lease):
        await runtime.send_message(
            message="Choose",
            expect_response=True,
            metadata={"tool_attempt_id": "attempt"},
        )
        with factory() as db:
            db.get(Task, tid).run_id = "replacement-run"
            db.commit()
        with pytest.raises(ExecutionEventPersistenceError):
            await runtime.send_message(
                message="Choose",
                expect_response=True,
                metadata={"tool_attempt_id": "attempt"},
            )
    broadcasts.assert_awaited_once()
    with factory() as db:
        assert len([e for e in facts(db, tid) if e.kind == "agent_message"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent", [False, True])
@pytest.mark.parametrize("wait", [False, True])
async def test_outbound_replay_projects_source_fact(outbound, concurrent, wait):
    factory, tid, handler, broadcasts = outbound
    payload = {
        "event_id": "question-1",
        "message": "Choose A or B",
        "expect_response": wait,
        "metadata": {},
    }
    if concurrent:
        await asyncio.gather(handler(deepcopy(payload)), handler(deepcopy(payload)))
    else:
        await handler(deepcopy(payload))
        await handler(deepcopy(payload))
    with factory() as db:
        rows = facts(db, tid)
        assert len(rows) == 1
        assert rows[0].kind == ("agent_message" if wait else "agent_progress")
        assert db.query(TraceEvent).count() == 1
        chats = db.query(TaskChatMessage).all()
        assert len(chats) == int(wait)
        if wait:
            assert chats[0].execution_event_id == rows[0].event_id
            assert chats[0].content == payload["message"]
    assert broadcasts.await_args_list[0].args == broadcasts.await_args_list[1].args
    with pytest.raises(ExecutionEventPersistenceError):
        await handler({**payload, "message": "Changed content"})
    await handler({**payload, "event_id": "question-2"})
    with factory() as db:
        assert len(facts(db, tid)) == 2
        assert db.query(TaskChatMessage).count() == 2 * int(wait)
        assert db.query(TraceEvent).count() == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("window", ["before_commit", "before_broadcast"])
async def test_outbound_projection_failure_recovery(outbound, monkeypatch, window):
    from xagent.web.services import chat_history_service

    factory, tid, handler, broadcasts = outbound
    payload = {
        "event_id": "question-1",
        "message": "Choose",
        "expect_response": True,
        "metadata": {},
    }
    original = chat_history_service.persist_assistant_message_no_commit

    def fail_after_projection(*args, **kwargs):
        original(*args, **kwargs)
        args[0].flush()
        raise OSError("crash after projection")

    with monkeypatch.context() as failure:
        if window == "before_commit":
            failure.setattr(
                chat_history_service,
                "persist_assistant_message_no_commit",
                fail_after_projection,
            )
        else:
            broadcasts.side_effect = OSError("crash before broadcast")
        with pytest.raises((OSError, ExecutionEventPersistenceError)):
            await handler(deepcopy(payload))
    broadcasts.side_effect = None
    if window == "before_commit":
        broadcasts.assert_not_awaited()
        with factory() as db:
            assert facts(db, tid) == []
            assert db.query(TaskChatMessage).count() == 0
            assert db.query(TraceEvent).count() == 0
    await handler(deepcopy(payload))
    with factory() as db:
        rows = facts(db, tid)
        assert len(rows) == 1
        assert db.query(TaskChatMessage).one().execution_event_id == rows[0].event_id
        assert db.query(TraceEvent).count() == 1
