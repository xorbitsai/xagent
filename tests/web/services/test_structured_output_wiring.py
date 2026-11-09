"""The two execution entry points read the format stored on a task and hand it
to the agent only when the task has one."""

from __future__ import annotations

import asyncio
import copy
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from xagent.core.agent.structured_output import (
    StructuredOutputFormat,
    _attach_structured_output,
)
from xagent.core.agent.trace import Tracer
from xagent.web.api import websocket as websocket_api
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services import task_events
from xagent.web.services import task_execution as task_execution_service
from xagent.web.services.response_format import RESPONSE_FORMAT_CONFIG_KEY
from xagent.web.services.task_execution import (
    background_task_manager,
    execute_resume_background,
    execute_task_background,
)
from xagent.web.services.task_lease_service import TaskLease, acquire_task_lease
from xagent.web.services.task_setup_snapshot import load_task_setup_snapshot_sync

INVALID = StructuredOutputFormat.INVALID
KWARG = "structured_output_format"
STORED_WARNING = "stored response_format is not valid"
ANSWER = "The answer is 42."
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
    "additionalProperties": False,
}


def _stored(**changes: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "type": "json_schema",
        "name": "result",
        "schema": copy.deepcopy(SCHEMA),
        "description": "A result.",
    }
    value.update(changes)
    return value


# (label, a stored value that is broken, the rule the WARNING names)
BROKEN = [
    ("name-is-int", _stored(name=1), "N5"),
    ("extra-key", {**_stored(), "strict": True}, "stored_shape"),
    (
        "missing-description",
        {k: v for k, v in _stored().items() if k != "description"},
        "stored_shape",
    ),
    ("type-is-text", _stored(type="text"), "stored_shape"),
    ("a-string", "x", "stored_shape"),
]

# agent_config values that mean "this task has no format"
NO_FORMAT = [
    ("no-key", {"other": 1}),
    ("not-a-dict", ["response_format"]),
    ("key-is-none", {RESPONSE_FORMAT_CONFIG_KEY: None}),
]


class ScriptedModel:
    """A chat model that replays a script of replies and counts its calls."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls = 0
        self.model_name = "fake-model"

    async def chat(self, **kwargs: Any) -> Any:
        self.calls += 1
        return self.script.pop(0)


def _submit(arguments: str) -> dict[str, Any]:
    return {
        "type": "tool_call",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "submit_result", "arguments": arguments},
            }
        ],
    }


def _good_model() -> ScriptedModel:
    return ScriptedModel([_submit('{"name": "Bo"}')])


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'structured_output_wiring.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


def _task(db, agent_config: Any, status: TaskStatus) -> tuple[int, int]:
    owner = User(username="owner", password_hash="x")
    db.add(owner)
    db.commit()
    db.refresh(owner)
    task = Task(
        user_id=owner.id,
        title="t",
        description="d",
        status=status,
        execution_mode="balanced",
        source="sdk",
        agent_config=agent_config,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return int(task.id), int(owner.id)


def _read_task(db, task_id: int) -> tuple[Any, Any]:
    db.expire_all()
    task = db.query(Task).filter(Task.id == task_id).one()
    return task.status, task.structured_output_record


def _config(stored: Any) -> dict[str, Any]:
    return {RESPONSE_FORMAT_CONFIG_KEY: stored}


def _wiring_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if STORED_WARNING in r.getMessage()]


class _NoopAgentService:
    def set_conversation_history(
        self, history, *, watermark=None, event_watermark=None
    ):
        pass

    def set_execution_context_messages(self, messages):
        pass

    def set_recovered_skill_context(self, skill_context):
        pass


class _NoopBackgroundTaskManager:
    async def wait_for_previous(self, task_id):
        pass

    def cleanup_task(self, task_id):
        pass


class _NoopBroadcastManager:
    async def broadcast_to_task(self, event, task_id):
        pass


async def _run_new_turn(
    db, monkeypatch: pytest.MonkeyPatch, agent_config: Any, llm: ScriptedModel
) -> tuple[int, dict[str, Any]]:
    """Run one real new turn; the agent manager is a stand-in that runs the real
    structured-output step on its result."""
    task_id, owner_id = _task(db, agent_config, TaskStatus.COMPLETED)
    with get_db_session() as lease_db:
        lease = acquire_task_lease(lease_db, task_id, new_run=True)
    assert lease is not None
    return task_id, await run_turn(monkeypatch, task_id, owner_id, lease, llm)


async def run_turn(
    monkeypatch: pytest.MonkeyPatch,
    task_id: int,
    owner_id: int,
    lease: TaskLease,
    llm: ScriptedModel,
    message: str = "first turn",
) -> dict[str, Any]:
    """Run ``execute_task_background`` for a task that holds ``lease`` and
    return the keywords the agent manager received."""
    snapshot = load_task_setup_snapshot_sync(task_id, owner_id)
    assert snapshot is not None
    seen: dict[str, Any] = {}

    class AgentManager:
        async def get_agent_for_task(self, task_id, db, **kwargs):
            return _NoopAgentService()

        async def execute_task(self, **kwargs):
            seen.update(kwargs)
            result = {
                "success": True,
                "status": "completed",
                "output": ANSWER,
                "file_outputs": [],
            }
            return await _attach_structured_output(
                result,
                kwargs.get(KWARG),
                llm=llm,
                task_id=kwargs["task_id"],
                run_id=kwargs["context"].get("run_id"),
                tracer=Tracer(),
            )

    with monkeypatch.context() as scoped:
        scoped.setattr(
            task_execution_service,
            "background_task_manager",
            _NoopBackgroundTaskManager(),
        )
        scoped.setattr(websocket_api, "manager", _NoopBroadcastManager())
        await execute_task_background(
            task_id=task_id,
            user_message=message,
            context={},
            agent_manager=AgentManager(),
            task_owner_user_id=owner_id,
            llm_user_message=None,
            task_setup_snapshot=snapshot,
            expected_run_id=lease.run_id,
            task_lease=lease,
            resolved_execution_scope=None,
        )
    return seen


def get_db_session():
    from xagent.web.models.database import get_session_local

    return get_session_local()()


async def _run_resume(
    db,
    monkeypatch: pytest.MonkeyPatch,
    agent_config: Any,
    llm: ScriptedModel,
    *,
    resume_stub: AsyncMock | None = None,
) -> tuple[int, dict[str, Any], AsyncMock]:
    """Run one real resume; ``resume_execution_by_id`` runs the real
    structured-output step on its result and records the keywords it got."""
    task_id, owner_id = _task(db, agent_config, TaskStatus.PAUSED)
    seen: dict[str, Any] = {}

    async def resume(execution_id: str, **kwargs: Any) -> dict[str, Any]:
        seen.clear()
        seen.update(kwargs)
        result = {
            "success": True,
            "status": "completed",
            "output": ANSWER,
            "agent_result": {"context": MagicMock(messages=[])},
        }
        attached = await _attach_structured_output(
            result,
            kwargs.get(KWARG),
            llm=llm,
            task_id=execution_id,
            run_id=kwargs["metadata"]["run_id"],
            tracer=Tracer(),
        )
        assert attached is not None
        return attached

    agent = MagicMock()
    agent.resume_execution_by_id = resume_stub or AsyncMock(side_effect=resume)
    sink = AsyncMock()
    monkeypatch.setattr(task_events, "_task_event_sink", sink)
    current = asyncio.current_task()
    assert current is not None
    background_task_manager.resume_tasks[task_id] = current
    try:
        await execute_resume_background(
            task_id=task_id, agent_service=agent, task_owner_user_id=owner_id
        )
    finally:
        background_task_manager.resume_tasks.pop(task_id, None)
    return task_id, seen, sink


# --- a new turn --------------------------------------------------------------


async def test_a_new_turn_hands_the_stored_format_to_the_agent_and_extracts(
    db_session, monkeypatch
) -> None:
    stored = _stored()
    llm = _good_model()

    task_id, seen = await _run_new_turn(db_session, monkeypatch, _config(stored), llm)

    fmt = seen[KWARG]
    assert isinstance(fmt, StructuredOutputFormat) and fmt is not INVALID
    assert (fmt.name, fmt.description, fmt.schema) == (
        stored["name"],
        stored["description"],
        stored["schema"],
    )
    status, record = _read_task(db_session, task_id)
    assert status == TaskStatus.COMPLETED
    assert record["status"] == "ready"
    assert record["output"] == {"name": "Bo"}
    assert llm.calls == 1


@pytest.mark.parametrize(
    ("stored", "rule"),
    [(s, r) for _, s, r in BROKEN],
    ids=[label for label, _, _ in BROKEN],
)
async def test_a_new_turn_with_a_broken_stored_format_fails_without_a_model(
    db_session, monkeypatch, caplog, stored, rule
) -> None:
    llm = _good_model()

    with caplog.at_level(logging.WARNING):
        task_id, seen = await _run_new_turn(
            db_session, monkeypatch, _config(stored), llm
        )

    assert seen[KWARG] is INVALID
    status, record = _read_task(db_session, task_id)
    assert status == TaskStatus.COMPLETED
    assert record["status"] == "failed"
    assert record["reason"]["code"] == "invalid_stored_format"
    assert llm.calls == 0
    warnings = _wiring_warnings(caplog)
    assert len(warnings) == 1
    assert f"rule={rule} at " in warnings[0]


@pytest.mark.parametrize(
    "agent_config", [c for _, c in NO_FORMAT], ids=[n for n, _ in NO_FORMAT]
)
async def test_a_new_turn_without_a_format_passes_no_keyword(
    db_session, monkeypatch, agent_config
) -> None:
    _, seen = await _run_new_turn(db_session, monkeypatch, agent_config, _good_model())

    assert KWARG not in seen


async def test_a_new_turn_does_not_change_the_stored_format(
    db_session, monkeypatch
) -> None:
    stored = _stored()
    kept = copy.deepcopy(stored)

    _, seen = await _run_new_turn(
        db_session, monkeypatch, _config(stored), _good_model()
    )

    assert seen[KWARG].schema == kept["schema"]
    assert stored == kept


# --- a resumed run -----------------------------------------------------------


async def test_a_resumed_run_hands_the_stored_format_to_the_agent_and_extracts(
    db_session, monkeypatch
) -> None:
    stored = _stored()
    llm = _good_model()

    task_id, seen, _ = await _run_resume(db_session, monkeypatch, _config(stored), llm)

    fmt = seen[KWARG]
    assert isinstance(fmt, StructuredOutputFormat) and fmt is not INVALID
    assert (fmt.name, fmt.description, fmt.schema) == (
        stored["name"],
        stored["description"],
        stored["schema"],
    )
    status, record = _read_task(db_session, task_id)
    assert status == TaskStatus.COMPLETED
    assert record["status"] == "ready"
    assert llm.calls == 1


@pytest.mark.parametrize(
    ("stored", "rule"),
    [(s, r) for _, s, r in BROKEN],
    ids=[label for label, _, _ in BROKEN],
)
async def test_a_resumed_run_with_a_broken_stored_format_fails_without_a_model(
    db_session, monkeypatch, caplog, stored, rule
) -> None:
    llm = _good_model()

    with caplog.at_level(logging.WARNING):
        task_id, seen, _ = await _run_resume(
            db_session, monkeypatch, _config(stored), llm
        )

    assert seen[KWARG] is INVALID
    status, record = _read_task(db_session, task_id)
    assert status == TaskStatus.COMPLETED
    assert record["status"] == "failed"
    assert record["reason"]["code"] == "invalid_stored_format"
    assert llm.calls == 0
    warnings = _wiring_warnings(caplog)
    assert len(warnings) == 1
    assert f"rule={rule} at " in warnings[0]


@pytest.mark.parametrize(
    "agent_config", [c for _, c in NO_FORMAT], ids=[n for n, _ in NO_FORMAT]
)
async def test_a_resumed_run_without_a_format_passes_no_keyword(
    db_session, monkeypatch, agent_config
) -> None:
    task_id, seen, _ = await _run_resume(
        db_session, monkeypatch, agent_config, _good_model()
    )

    assert KWARG not in seen
    assert _read_task(db_session, task_id)[1] is None


async def test_a_resumed_run_does_not_change_the_stored_format(
    db_session, monkeypatch
) -> None:
    stored = _stored()
    kept = copy.deepcopy(stored)

    _, seen, _ = await _run_resume(
        db_session, monkeypatch, _config(stored), _good_model()
    )

    assert seen[KWARG].schema == kept["schema"]
    assert stored == kept


def test_the_read_of_a_resumed_run_finds_nothing_for_another_run(db_session) -> None:
    task_id, _ = _task(db_session, _config(_stored()), TaskStatus.RUNNING)
    run_id = "run-a"
    db_session.query(Task).filter(Task.id == task_id).update({"run_id": run_id})
    db_session.commit()
    read = task_execution_service._read_resumed_task_agent_config_sync

    assert read(task_id, run_id) == (True, _config(_stored()))
    assert read(task_id, "another-run") == (False, None)


async def test_a_resumed_run_that_is_no_longer_its_run_passes_no_keyword(
    db_session, monkeypatch, caplog
) -> None:
    monkeypatch.setattr(
        task_execution_service,
        "_read_resumed_task_agent_config_sync",
        lambda task_id, run_id: (False, None),
    )
    llm = _good_model()

    with caplog.at_level(logging.WARNING):
        task_id, seen, _ = await _run_resume(
            db_session, monkeypatch, _config(_stored()), llm
        )

    assert KWARG not in seen
    assert llm.calls == 0
    matches = [
        r for r in caplog.records if "no task row matches this run" in r.getMessage()
    ]
    assert len(matches) == 1
    assert matches[0].levelno == logging.WARNING
    assert f"task_id={task_id} " in matches[0].getMessage()


async def test_a_failed_read_of_a_resumed_run_is_treated_as_no_format(
    db_session, monkeypatch, caplog
) -> None:
    def fail(task_id: int, run_id: str) -> Any:
        raise OperationalError("select", {}, Exception("connection lost"))

    monkeypatch.setattr(
        task_execution_service, "_read_resumed_task_agent_config_sync", fail
    )
    llm = _good_model()

    with caplog.at_level(logging.WARNING):
        task_id, seen, _ = await _run_resume(
            db_session, monkeypatch, _config(_stored()), llm
        )

    assert KWARG not in seen
    assert llm.calls == 0
    status, record = _read_task(db_session, task_id)
    assert status == TaskStatus.COMPLETED
    assert record is None
    run_id = seen["metadata"]["run_id"]
    matches = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and "OperationalError" in r.getMessage()
        and f"task_id={task_id}" in r.getMessage()
        and f"run_id={run_id}" in r.getMessage()
    ]
    assert len(matches) == 1
    assert matches[0].exc_info is not None
    assert "connection lost" not in matches[0].getMessage()


async def test_a_pool_timeout_during_the_read_of_a_resumed_run_is_not_swallowed(
    db_session, monkeypatch, caplog
) -> None:
    def exhausted(task_id: int, run_id: str) -> Any:
        try:
            raise PoolTimeoutError("pool exhausted")
        except PoolTimeoutError as cause:
            raise RuntimeError("checkout failed") from cause

    monkeypatch.setattr(
        task_execution_service, "_read_resumed_task_agent_config_sync", exhausted
    )
    stub = AsyncMock(return_value={"success": True, "status": "completed"})

    with caplog.at_level(logging.WARNING):
        _, _, sink = await _run_resume(
            db_session, monkeypatch, _config(_stored()), _good_model(), resume_stub=stub
        )

    stub.assert_not_awaited()
    assert any(
        "component=resume database pool checkout timed out" in r.getMessage()
        and r.levelno == logging.ERROR
        for r in caplog.records
    )
    assert not any(
        isinstance(call.args[0], dict) and call.args[0].get("type") == "task_error"
        for call in sink.call_args_list
    )
