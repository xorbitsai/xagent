"""``GET /v1/features``: what a key may do with structured output."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests.web.services import test_structured_output_wiring as wiring
from xagent.web.models.task import Task, TaskStatus
from xagent.web.services.task_lease_service import TaskLease

from ..conftest import _admin_headers, _direct_db_session, client
from .structured_output_shared import bearer, create_agent_with_key
from .test_auth import _create_personal_key
from .test_v1_response_format import (
    N8_MESSAGE,
    SCHEMA,
    _error,
    _format,
    _raw_agent_config,
    _set_status,
    _stored_format,
    _workforce_append,
    _workforce_task,
)
from .test_workforces import _manager_agent_id

pytestmark = pytest.mark.usefixtures("_test_db")


@pytest.fixture(autouse=True)
def mock_schedule_bg():
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ) as mocked:
        yield mocked


def _personal_key() -> str:
    return _create_personal_key()[0]


def _agent_key() -> str:
    return create_agent_with_key()[1]


def _workforce_key() -> str:
    return _workforce_task()[2]


@pytest.mark.parametrize(
    ("make_key", "features"),
    [
        (_personal_key, ["structured_output"]),
        (_agent_key, ["structured_output"]),
        (_workforce_key, []),
    ],
    ids=["personal", "agent", "workforce"],
)
def test_the_features_follow_the_kind_of_key(make_key: Any, features: list) -> None:
    response = client.get("/v1/features", headers=bearer(make_key()))

    assert response.status_code == 200, response.text
    assert response.json() == {"features": features}


def test_an_invalid_key_is_refused_like_everywhere_else() -> None:
    response = client.get("/v1/features", headers=bearer("xag_nope_nope"))

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


def _task_owner(task_id: int) -> tuple[int, TaskLease]:
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        lease = TaskLease(
            task_id=task_id,
            runner_id=str(task.runner_id),
            run_id=task.run_id,
            attempt_id=task.lease_attempt_id,
        )
        return int(task.user_id), lease
    finally:
        db.close()


async def _run_turn(
    monkeypatch: pytest.MonkeyPatch, task_id: int, llm: wiring.ScriptedModel
) -> dict[str, Any]:
    owner_id, lease = _task_owner(task_id)
    return await wiring.run_turn(monkeypatch, task_id, owner_id, lease, llm, "next")


async def test_a_workforce_task_keeps_its_format_across_a_workforce_key_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workforce_id, task_id, workforce_key = _workforce_task()
    headers = _admin_headers()
    manager_key = client.post(
        f"/api/agents/{_manager_agent_id(workforce_id)}/api-key", headers=headers
    ).json()["full_key"]
    manager_id = _manager_agent_id(workforce_id)
    _set_status(task_id, TaskStatus.COMPLETED)

    appended = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=bearer(manager_key),
        json={
            "agent_id": manager_id,
            "message": {"role": "user", "content": "next"},
            "response_format": _format(),
        },
    )
    assert appended.status_code == 202, appended.text
    stored = _stored_format(task_id)
    assert stored is not None and stored["schema"] == SCHEMA

    model = wiring.ScriptedModel([wiring._submit('{"a": "x", "b": 1}')])
    await _run_turn(monkeypatch, task_id, model)

    got = client.get(f"/v1/chat/tasks/{task_id}", headers=bearer(workforce_key))
    assert got.status_code == 200, got.text
    assert got.json()["structured_output_status"] == "ready"
    assert got.json()["structured_output"] == {"a": "x", "b": 1}

    _set_status(task_id, TaskStatus.COMPLETED)
    again = _workforce_append(workforce_id, task_id, workforce_key)
    assert again.status_code == 202, again.text
    model = wiring.ScriptedModel([wiring._submit('{"a": "y", "b": 2}')])
    seen = await _run_turn(monkeypatch, task_id, model)

    fmt = seen[wiring.KWARG]
    assert (fmt.name, fmt.description) == (stored["name"], stored["description"])
    assert fmt.schema == stored["schema"]
    assert "[Output requirement]" in seen["task"]
    assert model.calls == 1
    ready = client.get(f"/v1/chat/tasks/{task_id}", headers=bearer(workforce_key))
    assert ready.json()["structured_output_status"] == "ready"

    _set_status(task_id, TaskStatus.COMPLETED)
    refused = _workforce_append(
        workforce_id, task_id, workforce_key, response_format={"type": "text"}
    )
    assert _error(refused) == ("invalid_input", N8_MESSAGE)
    assert _raw_agent_config(task_id)["response_format"] == stored
