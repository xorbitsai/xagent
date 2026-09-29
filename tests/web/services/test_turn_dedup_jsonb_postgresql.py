"""The accepted-turn record keeps acceptance order through PostgreSQL JSONB.

JSONB does not preserve object key order, so a record whose eviction order
depended on dict insertion order would, after a reload, evict whichever ids
sort first rather than the oldest turns. These tests drive the runner through
the real database checkpoint handler.
"""

import os
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from tests.core.agent.test_runner import CheckpointingPattern, FakeWorkspaceManager
from xagent.core.agent import Agent, ExecutionContext
from xagent.core.agent import runner as runner_module
from xagent.core.agent.checkpoint import TraceCheckpointStore
from xagent.core.agent.runner import (
    ACCEPTED_TURN_IDS_METADATA_KEY,
    AgentRunner,
    UserMessageInjectionConflictError,
    UserMessageInjectionOutcome,
)
from xagent.core.agent.trace import Tracer
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services import trace_handlers
from xagent.web.services.task_lease_service import TaskLease, bind_task_lease_context
from xagent.web.services.trace_database import TraceDatabaseRuntime

pytestmark = [pytest.mark.postgresql, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def database_store(monkeypatch):
    url = os.getenv("XAGENT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    schema = "turn_dedup_test_" + uuid.uuid4().hex
    admin = create_engine(url)
    with admin.begin() as db:
        db.execute(text(f'CREATE SCHEMA "{schema}"'))
    source = create_engine(
        make_url(url).update_query_dict({"options": f"-csearch_path={schema}"})
    )
    runtime = None
    try:
        Base.metadata.create_all(source)
        with Session(source) as db:
            user = User(username="turn-dedup-test", password_hash="unused")
            db.add(user)
            db.flush()
            task = Task(
                user_id=user.id,
                title="Turn dedup",
                description="Turn dedup",
                status=TaskStatus.RUNNING,
                runner_id="runner",
                run_id="run",
                lease_attempt_id="attempt",
            )
            db.add(task)
            db.flush()
            task_id = int(task.id)
            db.commit()
        runtime = TraceDatabaseRuntime(source, use_async=False, limit=1)
        monkeypatch.setattr(
            trace_handlers, "get_trace_database_runtime", lambda: runtime
        )

        def sessions():
            with Session(source, autoflush=False) as db:
                yield db

        monkeypatch.setattr(trace_handlers, "get_db", sessions)
        tracer = Tracer()
        tracer.add_handler(trace_handlers.DatabaseTraceHandler(task_id))
        with bind_task_lease_context(TaskLease(task_id, "runner", "run", "attempt")):
            yield TraceCheckpointStore(tracer), source
    finally:
        if runtime is not None:
            await runtime.close()
        source.dispose()
        with admin.begin() as db:
            db.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def _runner(store: TraceCheckpointStore, tmp_path: Path) -> AgentRunner:
    return AgentRunner(
        agent=Agent(name="writer", patterns=[CheckpointingPattern({"success": True})]),
        tracer=store,
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )


async def test_jsonb_reload_evicts_oldest_turns(database_store, tmp_path, monkeypatch):
    store, source = database_store
    with source.connect() as connection:
        # The premise: JSONB hands objects back re-keyed, lists untouched.
        reordered = connection.execute(
            text("""SELECT '{"zzzz": 1, "a": 2}'::jsonb""")
        ).scalar_one()
        assert list(reordered) == ["a", "zzzz"]

    monkeypatch.setattr(runner_module, "MAX_ACCEPTED_TURN_IDS", 3)
    execution_id = "exec-jsonb-order"
    runner = _runner(store, tmp_path)
    await runner.run(
        task="Original task",
        execution_id=execution_id,
        metadata={"turn_id": "zzzz-start"},
    )
    # Later ids sort first under JSONB's key order (length, then bytes).
    turn_ids = ["yyy-1", "xx-2", "w-3", "v4"]
    for index, turn_id in enumerate(turn_ids):
        runner.context_manager.remove_context(execution_id)
        result = await runner.inject_user_message(
            execution_id, f"turn {index}", turn_id=turn_id, request_interrupt=False
        )
        assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH

    payload = await store.load_latest_checkpoint(execution_id)
    assert payload is not None
    stored = ExecutionContext.from_dict(payload["context"])
    record = stored.metadata[ACCEPTED_TURN_IDS_METADATA_KEY]
    assert [entry[0] for entry in record] == ["xx-2", "w-3", "v4"]

    runner.context_manager.remove_context(execution_id)
    retry = await runner.inject_user_message(
        execution_id, "turn 1", turn_id="xx-2", request_interrupt=False
    )
    assert retry.outcome is UserMessageInjectionOutcome.POSTED_REPLAY
    with pytest.raises(UserMessageInjectionConflictError):
        await runner.inject_user_message(
            execution_id, "turn 9", turn_id="w-3", request_interrupt=False
        )
