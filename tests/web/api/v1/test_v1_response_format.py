"""``response_format`` on the v1 task create and append endpoints."""

from __future__ import annotations

import asyncio
import contextvars
import copy
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from sqlalchemy import event

from xagent.web.api.v1 import tasks as v1_tasks
from xagent.web.models.database import get_engine
from xagent.web.models.task import Task, TaskStatus
from xagent.web.schemas.v1 import AppendMessageResponse, CreateTaskResponse
from xagent.web.services import task_auto_resume, task_event_bridge, task_start
from xagent.web.services.response_format import RESPONSE_FORMAT_CONFIG_KEY

from ..conftest import _admin_headers, _direct_db_session, client
from .structured_output_shared import bearer, create_agent_with_key
from .test_workforces import _create_active_workforce, _create_workforce_key

pytestmark = pytest.mark.usefixtures("_test_db")

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
    "additionalProperties": False,
}


@pytest.fixture(autouse=True)
def mock_schedule_bg():
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ) as mocked:
        yield mocked


def _format(name: str = "answer", schema: Any = None, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": name,
        "schema": copy.deepcopy(SCHEMA) if schema is None else schema,
    }
    body.update(extra)
    return {"type": "json_schema", "json_schema": body}


def _another_agent_with_key() -> tuple[int, str]:
    headers = _admin_headers()
    agent = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": "another response format agent",
            "description": "test",
            "instructions": "you are another test agent",
            "execution_mode": "balanced",
        },
    )
    assert agent.status_code == 200, agent.text
    agent_id = int(agent.json()["id"])
    key = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
    assert key.status_code == 200, key.text
    return agent_id, key.json()["full_key"]


def _create(key: str, agent_id: int, **extra: Any) -> Any:
    body: dict[str, Any] = {
        "agent_id": agent_id,
        "message": {"role": "user", "content": "first turn"},
    }
    body.update(extra)
    return client.post("/v1/chat/tasks", headers=bearer(key), json=body)


def _append(key: str, agent_id: int, task_id: int, **extra: Any) -> Any:
    body: dict[str, Any] = {
        "agent_id": agent_id,
        "message": {"role": "user", "content": "next turn"},
    }
    body.update(extra)
    return client.post(
        f"/v1/chat/tasks/{task_id}/messages", headers=bearer(key), json=body
    )


def _raw_agent_config(task_id: int) -> Any:
    db = _direct_db_session()
    try:
        return db.query(Task).filter(Task.id == task_id).one().agent_config
    finally:
        db.close()


def _stored_format(task_id: int) -> Any:
    config = _raw_agent_config(task_id)
    return config.get(RESPONSE_FORMAT_CONFIG_KEY) if isinstance(config, dict) else None


def _set_status(task_id: int, status: TaskStatus) -> None:
    db = _direct_db_session()
    try:
        db.query(Task).filter(Task.id == task_id).one().status = status
        db.commit()
    finally:
        db.close()


def _task_count() -> int:
    db = _direct_db_session()
    try:
        return db.query(Task).count()
    finally:
        db.close()


def _completed_task(key: str, agent_id: int, **extra: Any) -> int:
    response = _create(key, agent_id, **extra)
    assert response.status_code == 202, response.text
    task_id = int(response.json()["task_id"])
    _set_status(task_id, TaskStatus.COMPLETED)
    return task_id


def _echo(response: Any) -> tuple[str, str]:
    body = response.json()
    return body["response_format_type"], body["response_format_name"]


def _error(response: Any) -> tuple[str, str]:
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    return error["code"], error["message"]


# --- create -----------------------------------------------------------------------


def test_create_stores_the_schema_as_sent_and_echoes_the_format() -> None:
    agent_id, key = create_agent_with_key()

    response = _create(key, agent_id, response_format=_format(description="the answer"))

    assert response.status_code == 202, response.text
    assert _echo(response) == ("json_schema", "answer")
    stored = _stored_format(int(response.json()["task_id"]))
    assert stored == {
        "type": "json_schema",
        "name": "answer",
        "schema": SCHEMA,
        "description": "the answer",
    }


@pytest.mark.parametrize(
    "extra",
    [{}, {"response_format": None}, {"response_format": {"type": "text"}}],
    ids=["absent", "null", "text"],
)
def test_create_without_a_format_leaves_the_task_config_untouched(
    extra: dict[str, Any],
) -> None:
    agent_id, key = create_agent_with_key()
    baseline = _create(key, agent_id)
    assert baseline.status_code == 202, baseline.text

    response = _create(key, agent_id, **extra)

    assert response.status_code == 202, response.text
    assert _echo(response) == ("text", "")
    task_id = int(response.json()["task_id"])
    assert _raw_agent_config(task_id) == _raw_agent_config(
        int(baseline.json()["task_id"])
    )
    assert _stored_format(task_id) is None


def test_create_keeps_the_prompt_context_next_to_the_format() -> None:
    agent_id, key = create_agent_with_key()

    response = _create(
        key,
        agent_id,
        prompt_context={"date": "2026-10-19"},
        response_format=_format(),
    )

    assert response.status_code == 202, response.text
    config = _raw_agent_config(int(response.json()["task_id"]))
    assert set(config) == {"prompt_context", RESPONSE_FORMAT_CONFIG_KEY}


def test_a_format_that_breaks_a_rule_is_refused_before_a_task_exists() -> None:
    agent_id, key = create_agent_with_key()
    before = _task_count()

    response = _create(
        key,
        agent_id,
        response_format=_format(
            schema={
                "type": "object",
                "properties": {"a": {"type": "integer", "enum": [True]}},
                "required": ["a"],
                "additionalProperties": False,
            }
        ),
    )

    code, message = _error(response)
    assert code == "invalid_input"
    assert message.startswith("S4 at /properties/a: ")
    assert _task_count() == before


def test_a_format_that_is_not_an_object_is_the_outer_rule_not_a_body_error() -> None:
    agent_id, key = create_agent_with_key()
    before = _task_count()

    response = _create(key, agent_id, response_format="x")

    code, message = _error(response)
    assert code == "invalid_input"
    assert message.startswith("N1 at response_format: ")
    assert _task_count() == before


def test_a_lone_surrogate_property_name_is_refused_as_ascii_text() -> None:
    agent_id, key = create_agent_with_key()
    before = _task_count()
    schema = (
        '{"type": "object", "properties": {"\\ud800": {"type": "string"}}, '
        '"required": ["\\ud800"], "additionalProperties": false}'
    )
    body = (
        f'{{"agent_id": {agent_id}, '
        '"message": {"role": "user", "content": "x"}, '
        '"response_format": {"type": "json_schema", "json_schema": '
        f'{{"name": "answer", "schema": {schema}}}}}}}'
    )

    response = client.post(
        "/v1/chat/tasks",
        headers={**bearer(key), "Content-Type": "application/json"},
        content=body,
    )

    assert response.status_code == 422, response.text
    assert response.content.decode("utf-8").isascii()
    assert response.json()["error"]["message"].startswith("S1 at /properties")
    assert _task_count() == before


def test_the_agent_check_comes_before_the_format_check() -> None:
    agent_id, key = create_agent_with_key()

    response = _create(key, agent_id + 1000, response_format="x")

    assert response.status_code == 404, response.text


def test_a_request_body_nested_two_thousand_levels_deep_is_a_client_error() -> None:
    agent_id, key = create_agent_with_key()
    depth = 2000
    body = (
        f'{{"agent_id": {agent_id}, '
        '"message": {"role": "user", "content": "x"}, '
        f'"response_format": {"[" * depth}{"]" * depth}}}'
    )

    response = client.post(
        "/v1/chat/tasks",
        headers={**bearer(key), "Content-Type": "application/json"},
        content=body,
    )

    assert 400 <= response.status_code < 500, response.text


# --- append -----------------------------------------------------------------------


def test_append_with_a_new_format_replaces_the_old_one() -> None:
    agent_id, key = create_agent_with_key()
    task_id = _completed_task(key, agent_id, response_format=_format(name="one"))

    response = _append(key, agent_id, task_id, response_format=_format(name="two"))

    assert response.status_code == 202, response.text
    assert _echo(response) == ("json_schema", "two")
    assert _stored_format(task_id)["name"] == "two"


def test_append_with_text_clears_the_format() -> None:
    agent_id, key = create_agent_with_key()
    task_id = _completed_task(key, agent_id, response_format=_format())

    response = _append(key, agent_id, task_id, response_format={"type": "text"})

    assert response.status_code == 202, response.text
    assert _echo(response) == ("text", "")
    assert _stored_format(task_id) is None


@pytest.mark.parametrize(
    "extra", [{}, {"response_format": None}], ids=["absent", "null"]
)
def test_append_without_a_format_keeps_the_stored_one(extra: dict[str, Any]) -> None:
    agent_id, key = create_agent_with_key()
    task_id = _completed_task(key, agent_id, response_format=_format(name="kept"))
    before = _raw_agent_config(task_id)

    response = _append(key, agent_id, task_id, **extra)

    assert response.status_code == 202, response.text
    assert _echo(response) == ("json_schema", "kept")
    assert _raw_agent_config(task_id) == before


def test_append_without_any_format_echoes_text() -> None:
    agent_id, key = create_agent_with_key()
    task_id = _completed_task(key, agent_id)

    response = _append(key, agent_id, task_id)

    assert response.status_code == 202, response.text
    assert _echo(response) == ("text", "")
    assert _raw_agent_config(task_id) is None


def test_append_sets_a_format_and_a_prompt_context_together() -> None:
    agent_id, key = create_agent_with_key()
    task_id = _completed_task(key, agent_id)

    response = _append(
        key,
        agent_id,
        task_id,
        prompt_context={"date": "2026-10-19"},
        response_format=_format(),
    )

    assert response.status_code == 202, response.text
    assert set(_raw_agent_config(task_id)) == {
        "prompt_context",
        RESPONSE_FORMAT_CONFIG_KEY,
    }


def test_an_invalid_format_on_append_changes_nothing() -> None:
    agent_id, key = create_agent_with_key()
    task_id = _completed_task(key, agent_id, response_format=_format(name="kept"))
    before = _raw_agent_config(task_id)

    response = _append(key, agent_id, task_id, response_format=[1])

    assert _error(response)[1].startswith("N1 at response_format: ")
    assert _raw_agent_config(task_id) == before


@pytest.mark.parametrize(
    ("case", "status", "code"),
    [
        ("busy", TaskStatus.RUNNING, "task_busy"),
        ("needs-reply", TaskStatus.WAITING_FOR_USER, "interaction_response_required"),
        ("other-agent", TaskStatus.COMPLETED, "task_not_found"),
    ],
)
def test_a_refused_append_leaves_the_stored_format_alone(
    case: str, status: TaskStatus, code: str
) -> None:
    agent_id, key = create_agent_with_key()
    task_id = _completed_task(key, agent_id, response_format=_format(name="kept"))
    _set_status(task_id, status)
    before = _stored_format(task_id)
    caller_id, caller_key = (
        _another_agent_with_key() if case == "other-agent" else (agent_id, key)
    )

    response = _append(
        caller_key, caller_id, task_id, response_format=_format(name="refused")
    )

    assert response.status_code in (404, 409), response.text
    assert response.json()["error"]["code"] == code
    assert _stored_format(task_id) == before
    assert before["name"] == "kept"


# --- the echo comes from the claiming transaction ----------------------------------


_AFTER_APPEND_RETURNS: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "after_append_returns", default=False
)


@contextmanager
def _statements_after_append_returns() -> Iterator[list[str]]:
    """The SELECTs from ``tasks`` that the handler sends once
    ``append_sdk_turn`` has returned.

    ``wrapped`` sets a context variable when ``append_sdk_turn`` returns, and
    ``record`` only counts statements issued while it is set.
    ``asyncio.to_thread`` and ``create_task`` copy the current context, so a
    read the handler makes through ``run_db_io_cancellation_safe`` on a worker
    thread is recorded, while background execution threads started earlier or
    outside this request keep the default value and are not.
    """
    seen: list[str] = []
    real = task_start.append_sdk_turn

    async def wrapped(**kwargs: Any) -> Any:
        result = await real(**kwargs)
        _AFTER_APPEND_RETURNS.set(True)
        return result

    def record(_conn: Any, _cursor: Any, statement: str, *_args: Any) -> None:
        if _AFTER_APPEND_RETURNS.get():
            if re.match(r"\s*SELECT\b", statement, re.I) and re.search(
                r"\bFROM\s+tasks\b", statement, re.I
            ):
                seen.append(statement)

    engine = get_engine()
    event.listen(engine, "before_cursor_execute", record)
    try:
        with patch.object(v1_tasks.task_start_service, "append_sdk_turn", wrapped):
            yield seen
    finally:
        event.remove(engine, "before_cursor_execute", record)


def test_the_handler_does_not_read_the_task_again_to_echo_the_format() -> None:
    agent_id, key = create_agent_with_key()
    task_id = _completed_task(key, agent_id)

    with _statements_after_append_returns() as reads:
        response = _append(key, agent_id, task_id, response_format=_format(name="now"))

    assert response.status_code == 202, response.text
    assert _echo(response) == ("json_schema", "now")
    assert reads == []


def test_the_echo_equals_the_format_on_the_task_after_commit() -> None:
    agent_id, key = create_agent_with_key()
    task_id = _completed_task(key, agent_id, response_format=_format(name="one"))
    steps: list[tuple[dict[str, Any], tuple[str, str]]] = [
        ({"response_format": _format(name="two")}, ("json_schema", "two")),
        ({}, ("json_schema", "two")),
        ({"response_format": {"type": "text"}}, ("text", "")),
    ]

    for extra, expected in steps:
        _set_status(task_id, TaskStatus.COMPLETED)
        response = _append(key, agent_id, task_id, **extra)
        assert response.status_code == 202, response.text
        stored = _stored_format(task_id)
        assert _echo(response) == expected
        assert (stored["name"] if stored else "") == expected[1]


def _result_fields() -> dict[str, Any]:
    return {
        "task_id": 1,
        "agent_id": 2,
        "status": TaskStatus.RUNNING,
        "accepted_at": datetime.now(timezone.utc),
        "run_id": "run-1",
        "state_version": 1,
        "control_state": "running",
    }


def test_the_start_result_must_be_given_the_echo_fields() -> None:
    with pytest.raises(TypeError):
        task_start.TaskStartResult(**_result_fields())  # type: ignore[call-arg]

    result = task_start.TaskStartResult(
        **_result_fields(), response_format_type="text", response_format_name=""
    )
    assert result.dispatch_class is None


def test_the_prepared_append_must_be_given_the_echo_fields() -> None:
    fields: dict[str, Any] = {
        "task_id": 1,
        "agent_id": 2,
        "task_owner_user_id": 3,
        "payload": MagicMock(),
        "claimed_turn": MagicMock(),
    }

    with pytest.raises(TypeError):
        task_start._PreparedAppendTurn(**fields)  # type: ignore[call-arg]

    prepared = task_start._PreparedAppendTurn(
        **fields, response_format_type="text", response_format_name=""
    )
    assert prepared.response_format_name == ""


def test_a_downstream_can_build_the_responses_without_the_echo_fields() -> None:
    created = CreateTaskResponse(
        task_id=1,
        agent_id=2,
        status="running",
        created_at=datetime.now(timezone.utc),
        run_id="run-1",
        state_version=1,
        control_state="running",
    )
    appended = AppendMessageResponse(
        task_id=1,
        agent_id=2,
        status="running",
        accepted_at=datetime.now(timezone.utc),
        run_id="run-1",
        state_version=1,
        control_state="running",
    )

    for response in (created, appended):
        assert response.response_format_type == "text"
        assert response.response_format_name == ""


# --- shared execution ---------------------------------------------------------------


@pytest.fixture
def combined_until_the_append(monkeypatch: pytest.MonkeyPatch) -> None:
    """The task is created and completed in combined mode, so no START is
    pending; ``_go_shared`` then switches the append itself to shared mode."""
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "false")


def _go_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(task_event_bridge, "get_task_event_bridge", lambda: Mock())
    monkeypatch.setattr(
        task_auto_resume, "notify_task_command_dispatcher", lambda: None
    )


@pytest.mark.parametrize(
    ("before", "change", "expected"),
    [
        (None, _format(name="two"), ("json_schema", "two")),
        ("one", _format(name="two"), ("json_schema", "two")),
        ("one", {"type": "text"}, ("text", "")),
    ],
    ids=["set", "replace", "clear"],
)
@pytest.mark.usefixtures("combined_until_the_append")
def test_a_shared_append_stores_the_format_it_echoes(
    monkeypatch: pytest.MonkeyPatch,
    before: str | None,
    change: dict[str, Any],
    expected: tuple[str, str],
) -> None:
    agent_id, key = create_agent_with_key()
    extra = {} if before is None else {"response_format": _format(name=before)}
    task_id = _completed_task(key, agent_id, **extra)
    _go_shared(monkeypatch)

    response = _append(key, agent_id, task_id, response_format=change)

    assert response.status_code == 202, response.text
    assert _echo(response) == expected
    stored = _stored_format(task_id)
    assert (stored["name"] if stored else "") == expected[1]


@pytest.mark.usefixtures("combined_until_the_append")
def test_a_refused_shared_append_leaves_the_stored_format_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_id, key = create_agent_with_key()
    task_id = _completed_task(key, agent_id, response_format=_format(name="kept"))
    _set_status(task_id, TaskStatus.RUNNING)
    _go_shared(monkeypatch)

    response = _append(key, agent_id, task_id, response_format=_format(name="two"))

    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "task_busy"
    assert _stored_format(task_id)["name"] == "kept"


@pytest.mark.usefixtures("combined_until_the_append")
def test_a_shared_append_stores_a_prompt_context_sent_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_id, key = create_agent_with_key()
    task_id = _completed_task(key, agent_id)
    _go_shared(monkeypatch)

    response = _append(key, agent_id, task_id, prompt_context={"date": "2026-10-19"})

    assert response.status_code == 202, response.text
    assert _echo(response) == ("text", "")
    assert set(_raw_agent_config(task_id)) == {"prompt_context"}


@pytest.mark.usefixtures("combined_until_the_append")
@pytest.mark.parametrize("shared", [True, False], ids=["shared", "combined"])
def test_the_echo_follows_a_format_changed_after_the_task_was_loaded(
    monkeypatch: pytest.MonkeyPatch, shared: bool
) -> None:
    agent_id, key = create_agent_with_key()
    task_id = _completed_task(key, agent_id, response_format=_format(name="F1"))
    if shared:
        _go_shared(monkeypatch)
    real = task_start._resolve_turn_files

    def another_append_completes_first(**kwargs: Any) -> Any:
        db = _direct_db_session()
        try:
            task = db.query(Task).filter(Task.id == task_id).one()
            config = dict(task.agent_config)
            config[RESPONSE_FORMAT_CONFIG_KEY] = dict(
                config[RESPONSE_FORMAT_CONFIG_KEY], name="F2"
            )
            task.agent_config = config
            task.status = TaskStatus.COMPLETED
            db.commit()
        finally:
            db.close()
        return real(**kwargs)

    monkeypatch.setattr(
        task_start, "_resolve_turn_files", another_append_completes_first
    )

    response = _append(key, agent_id, task_id)

    assert response.status_code == 202, response.text
    assert _stored_format(task_id)["name"] == "F2"
    assert _echo(response) == ("json_schema", "F2")


# --- workforce keys ---------------------------------------------------------------


def _workforce_task() -> tuple[int, int, str]:
    headers = _admin_headers()
    workforce_id = _create_active_workforce(headers)
    key = _create_workforce_key(headers, workforce_id)
    run = client.post(
        f"/v1/workforces/{workforce_id}/runs",
        headers=bearer(key),
        json={"message": {"role": "user", "content": "first"}},
    )
    assert run.status_code == 202, run.text
    return workforce_id, int(run.json()["task_id"]), key


def _workforce_append(workforce_id: int, task_id: int, key: str, **extra: Any) -> Any:
    body: dict[str, Any] = {
        "workforce_id": workforce_id,
        "message": {"role": "user", "content": "next"},
    }
    body.update(extra)
    return client.post(
        f"/v1/chat/tasks/{task_id}/messages", headers=bearer(key), json=body
    )


N8_MESSAGE = (
    "N8 at response_format: Structured output is not supported for workforce keys."
)


@pytest.mark.parametrize(
    "value", [{"type": "text"}, "x", _format()], ids=["text", "not-an-object", "format"]
)
def test_a_workforce_key_cannot_set_or_clear_a_format(value: Any) -> None:
    workforce_id, task_id, key = _workforce_task()

    response = _workforce_append(workforce_id, task_id, key, response_format=value)

    assert _error(response) == ("invalid_input", N8_MESSAGE)
    assert _raw_agent_config(task_id) is None or (
        RESPONSE_FORMAT_CONFIG_KEY not in _raw_agent_config(task_id)
    )


def test_the_workforce_check_comes_after_the_prompt_context_one() -> None:
    workforce_id, task_id, key = _workforce_task()

    response = _workforce_append(
        workforce_id,
        task_id,
        key,
        prompt_context={"date": "2026-10-19"},
        response_format="x",
    )

    assert _error(response) == (
        "invalid_input",
        "prompt_context is not supported for workforce keys.",
    )


def test_a_workforce_key_may_send_null() -> None:
    workforce_id, task_id, key = _workforce_task()
    _set_status(task_id, TaskStatus.COMPLETED)
    from xagent.web.models.workforce import WorkforceRun

    db = _direct_db_session()
    try:
        run_id = (_raw_agent_config(task_id) or {}).get("workforce_run_id")
        if isinstance(run_id, int):
            db.query(WorkforceRun).filter(WorkforceRun.id == run_id).update(
                {"status": "completed"}
            )
            db.commit()
    finally:
        db.close()

    response = _workforce_append(workforce_id, task_id, key, response_format=None)

    assert response.status_code == 202, response.text
    assert _echo(response) == ("text", "")


def test_the_async_handler_checks_the_format_off_the_event_loop() -> None:
    seen: list[bool] = []
    real = v1_tasks.normalize_response_format

    def spy(raw: Any) -> Any:
        try:
            asyncio.get_running_loop()
            seen.append(True)
        except RuntimeError:
            seen.append(False)
        return real(raw)

    agent_id, key = create_agent_with_key()
    with patch.object(v1_tasks, "normalize_response_format", spy):
        response = _create(key, agent_id, response_format=_format())

    assert response.status_code == 202, response.text
    assert seen == [False]
