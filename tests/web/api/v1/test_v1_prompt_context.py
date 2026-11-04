"""Run-key prompt context on the v1 task create and append endpoints."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from xagent.web.api import websocket as websocket_api
from xagent.web.models.task import Task, TaskStatus
from xagent.web.schemas.chat import TaskCreateRequest as ChatTaskCreateRequest
from xagent.web.schemas.chat import WidgetTaskCreateRequest
from xagent.web.schemas.v1 import AppendMessageRequest, CreateTaskRequest, ReplyRequest
from xagent.web.services import task_execution as task_execution_service
from xagent.web.services.file_turn import (
    PROMPT_CONTEXT_MAX_DATA_BYTES,
    build_prompt_context_block,
    prompt_context_data_size,
)
from xagent.web.services.task_execution import execute_task_background
from xagent.web.services.task_lease_service import TaskLease
from xagent.web.services.task_setup_snapshot import load_task_setup_snapshot_sync

from ..conftest import _admin_headers, _direct_db_session, client

pytestmark = pytest.mark.usefixtures("_test_db")

DECLARATIONS = [
    {"name": "plan.tier", "max_length": 2, "description": "Current plan tier"},
]


@pytest.fixture(autouse=True)
def mock_schedule_bg():
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ) as mocked:
        yield mocked


def _create_agent_with_key(
    declarations: list[dict[str, Any]] | None = None,
) -> tuple[int, str]:
    headers = _admin_headers()
    body: dict[str, Any] = {
        "name": "v1 prompt context agent",
        "description": "test",
        "instructions": "you are a test agent",
        "execution_mode": "balanced",
    }
    if declarations is not None:
        body["prompt_context_fields"] = declarations
    agent_resp = client.post("/api/agents", headers=headers, json=body)
    assert agent_resp.status_code == 200, agent_resp.text
    agent_id = agent_resp.json()["id"]
    key_resp = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
    assert key_resp.status_code == 200, key_resp.text
    return agent_id, key_resp.json()["full_key"]


def _bearer(full_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {full_key}"}


def _task_agent_config(task_id: int) -> dict[str, Any]:
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        return dict(task.agent_config or {})
    finally:
        db.close()


def _force_task_status(task_id: int, status: TaskStatus) -> None:
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.status = status
        db.commit()
    finally:
        db.close()


def _create_task(
    full_key: str,
    agent_id: int,
    *,
    prompt_context: dict[str, Any] | None = None,
) -> int:
    body: dict[str, Any] = {
        "agent_id": agent_id,
        "message": {"role": "user", "content": "first turn"},
    }
    if prompt_context is not None:
        body["prompt_context"] = prompt_context
    response = client.post("/v1/chat/tasks", headers=_bearer(full_key), json=body)
    assert response.status_code == 202, response.text
    return int(response.json()["task_id"])


def _append(
    full_key: str,
    agent_id: int,
    task_id: int,
    *,
    prompt_context: dict[str, Any] | None = None,
) -> Any:
    body: dict[str, Any] = {
        "agent_id": agent_id,
        "message": {"role": "user", "content": "next turn"},
    }
    if prompt_context is not None:
        body["prompt_context"] = prompt_context
    return client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json=body,
    )


def test_create_stores_signed_context_for_the_first_turn() -> None:
    agent_id, full_key = _create_agent_with_key(DECLARATIONS)

    task_id = _create_task(
        full_key,
        agent_id,
        prompt_context={
            "teacher.name": "Teacher Wang",
            "plan.tier": "premium",
            "period": 3,
        },
    )

    config = _task_agent_config(task_id)["prompt_context"]
    assert config == {
        "data": {
            "teacher.name": "Teacher Wang",
            "plan.tier": "premium",
            "period": 3,
        },
        "signed": True,
        "descriptions": {"plan.tier": "Current plan tier"},
    }
    block = build_prompt_context_block(config)
    assert block is not None
    assert "Signed by the integrator's backend." in block
    assert "Field meanings (set by the agent owner):" in block
    assert "- plan.tier: Current plan tier" in block
    assert '"teacher.name": "Teacher Wang"' in block


def test_create_keeps_values_a_field_declaration_would_drop() -> None:
    agent_id, full_key = _create_agent_with_key(DECLARATIONS)

    task_id = _create_task(
        full_key,
        agent_id,
        prompt_context={"plan.tier": "premium", "undeclared": "kept"},
    )

    assert _task_agent_config(task_id)["prompt_context"] == {
        "data": {"plan.tier": "premium", "undeclared": "kept"},
        "signed": True,
        "descriptions": {"plan.tier": "Current plan tier"},
    }


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


async def _run_agent_turn(
    monkeypatch: pytest.MonkeyPatch,
    task_id: int,
    user_message: str,
    captured: dict[str, Any],
) -> None:
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        owner_id = int(task.user_id)
        lease = TaskLease(
            task_id=task_id,
            runner_id=str(task.runner_id),
            run_id=task.run_id,
            attempt_id=task.lease_attempt_id,
        )
    finally:
        db.close()
    # The production path into ``execute_task_background``: the snapshot has
    # to carry the stored ``agent_config`` for the context to reach the turn.
    snapshot = load_task_setup_snapshot_sync(task_id, owner_id)
    assert snapshot is not None

    class AgentManager:
        async def get_agent_for_task(self, task_id, db, **kwargs):
            return _NoopAgentService()

        async def execute_task(self, **kwargs):
            captured["agent_task"] = kwargs.get("task")
            return {"success": True, "output": "ok", "file_outputs": []}

    monkeypatch.setattr(
        task_execution_service,
        "background_task_manager",
        _NoopBackgroundTaskManager(),
    )
    monkeypatch.setattr(websocket_api, "manager", _NoopBroadcastManager())
    await execute_task_background(
        task_id=task_id,
        user_message=user_message,
        context={},
        agent_manager=AgentManager(),
        task_owner_user_id=owner_id,
        llm_user_message=None,
        task_setup_snapshot=snapshot,
        expected_run_id=lease.run_id,
        task_lease=lease,
        resolved_execution_scope=None,
    )


async def test_create_context_reaches_the_first_agent_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(
        full_key,
        agent_id,
        prompt_context={"teacher.name": "Teacher Wang", "date": "2026-10-20"},
    )
    captured: dict[str, Any] = {}

    await _run_agent_turn(monkeypatch, task_id, "first turn", captured)

    block = build_prompt_context_block(_task_agent_config(task_id)["prompt_context"])
    assert block is not None
    assert captured["agent_task"] == f"first turn\n\n{block}"


async def test_append_context_reaches_the_next_agent_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(
        full_key,
        agent_id,
        prompt_context={"date": "2026-10-19"},
    )
    _force_task_status(task_id, TaskStatus.COMPLETED)

    response = _append(
        full_key,
        agent_id,
        task_id,
        prompt_context={"date": "2026-10-20", "teacher": "Teacher Li"},
    )
    assert response.status_code == 202, response.text
    captured: dict[str, Any] = {}

    await _run_agent_turn(monkeypatch, task_id, "next turn", captured)

    block = build_prompt_context_block(_task_agent_config(task_id)["prompt_context"])
    assert block is not None
    assert captured["agent_task"] == f"next turn\n\n{block}"
    assert "2026-10-19" not in captured["agent_task"]
    assert "Teacher Li" in captured["agent_task"]


def test_create_without_context_leaves_the_task_config_alone() -> None:
    agent_id, full_key = _create_agent_with_key()

    task_id = _create_task(full_key, agent_id)

    assert "prompt_context" not in _task_agent_config(task_id)


def test_create_with_empty_context_stays_a_noop() -> None:
    agent_id, full_key = _create_agent_with_key()

    task_id = _create_task(full_key, agent_id, prompt_context={})

    assert "prompt_context" not in _task_agent_config(task_id)


def test_append_uses_the_new_context_for_that_turn() -> None:
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(
        full_key,
        agent_id,
        prompt_context={"date": "2026-10-19", "teacher": "Teacher Zhang"},
    )
    _force_task_status(task_id, TaskStatus.COMPLETED)

    response = _append(
        full_key,
        agent_id,
        task_id,
        prompt_context={"date": "2026-10-20", "teacher": "Teacher Li"},
    )

    assert response.status_code == 202, response.text
    config = _task_agent_config(task_id)["prompt_context"]
    assert config == {
        "data": {"date": "2026-10-20", "teacher": "Teacher Li"},
        "signed": True,
    }
    block = build_prompt_context_block(config)
    assert block is not None
    assert "2026-10-20" in block
    assert "Teacher Li" in block
    assert "2026-10-19" not in block
    assert "Teacher Zhang" not in block


def test_append_keeps_values_a_field_declaration_would_drop() -> None:
    agent_id, full_key = _create_agent_with_key(DECLARATIONS)
    task_id = _create_task(full_key, agent_id)
    _force_task_status(task_id, TaskStatus.COMPLETED)

    response = _append(
        full_key,
        agent_id,
        task_id,
        prompt_context={"plan.tier": "premium"},
    )

    assert response.status_code == 202, response.text
    assert _task_agent_config(task_id)["prompt_context"] == {
        "data": {"plan.tier": "premium"},
        "signed": True,
        "descriptions": {"plan.tier": "Current plan tier"},
    }


def test_append_context_preserves_other_task_config_keys() -> None:
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.agent_config = {"keep_me": "yes"}
        db.commit()
    finally:
        db.close()
    _force_task_status(task_id, TaskStatus.COMPLETED)

    response = _append(
        full_key,
        agent_id,
        task_id,
        prompt_context={"date": "2026-10-20"},
    )

    assert response.status_code == 202, response.text
    config = _task_agent_config(task_id)
    assert config["keep_me"] == "yes"
    assert config["prompt_context"] == {
        "data": {"date": "2026-10-20"},
        "signed": True,
    }


def test_append_without_context_replays_the_stored_value() -> None:
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(
        full_key,
        agent_id,
        prompt_context={"date": "2026-10-19"},
    )
    _force_task_status(task_id, TaskStatus.COMPLETED)

    response = _append(full_key, agent_id, task_id)

    assert response.status_code == 202, response.text
    assert _task_agent_config(task_id)["prompt_context"] == {
        "data": {"date": "2026-10-19"},
        "signed": True,
    }


def test_append_with_explicit_null_context_replays_the_stored_value() -> None:
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(
        full_key,
        agent_id,
        prompt_context={"date": "2026-10-19"},
    )
    _force_task_status(task_id, TaskStatus.COMPLETED)

    response = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "next turn"},
            "prompt_context": None,
        },
    )

    assert response.status_code == 202, response.text
    assert _task_agent_config(task_id)["prompt_context"] == {
        "data": {"date": "2026-10-19"},
        "signed": True,
    }


async def test_append_empty_context_clears_it_for_the_next_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(
        full_key,
        agent_id,
        prompt_context={"date": "2026-10-19", "teacher": "Teacher Zhang"},
    )
    _force_task_status(task_id, TaskStatus.COMPLETED)

    response = _append(full_key, agent_id, task_id, prompt_context={})

    assert response.status_code == 202, response.text
    assert "prompt_context" not in _task_agent_config(task_id)
    captured: dict[str, Any] = {}
    await _run_agent_turn(monkeypatch, task_id, "next turn", captured)
    assert captured["agent_task"] == "next turn"
    assert "2026-10-19" not in captured["agent_task"]
    assert "Teacher Zhang" not in captured["agent_task"]


def test_append_non_empty_context_replaces_without_merging_keys() -> None:
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(
        full_key,
        agent_id,
        prompt_context={"date": "2026-10-19", "teacher": "Teacher Zhang"},
    )
    _force_task_status(task_id, TaskStatus.COMPLETED)

    response = _append(
        full_key,
        agent_id,
        task_id,
        prompt_context={"teacher": "Teacher Li"},
    )

    assert response.status_code == 202, response.text
    assert _task_agent_config(task_id)["prompt_context"] == {
        "data": {"teacher": "Teacher Li"},
        "signed": True,
    }


def test_append_rejects_a_multi_key_payload_over_the_total_byte_cap() -> None:
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_task_status(task_id, TaskStatus.COMPLETED)

    response = _append(
        full_key,
        agent_id,
        task_id,
        prompt_context={"first": "x" * 2500, "second": "y" * 2500},
    )

    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "invalid_input"
    assert "prompt_context" not in _task_agent_config(task_id)


@pytest.mark.parametrize(
    "prompt_context",
    [
        pytest.param(["a"], id="list"),
        pytest.param("text", id="string"),
        pytest.param(3, id="number"),
        pytest.param(True, id="boolean"),
        pytest.param({"nested": {"a": 1}}, id="nested-object"),
        pytest.param({"none": None}, id="none-value"),
        pytest.param({"list": [1]}, id="list-value"),
        pytest.param({"too_big": "x" * 5000}, id="over-size"),
        pytest.param({"": "v"}, id="empty-key"),
        pytest.param({"has space": "v"}, id="key-with-space"),
        pytest.param({"line\nbreak": "v"}, id="key-with-newline"),
        pytest.param({"1leading_digit": "v"}, id="key-with-leading-digit"),
        pytest.param({"k" * 65: "v"}, id="key-over-64-characters"),
    ],
)
def test_create_rejects_malformed_or_oversized_context(
    prompt_context, mock_schedule_bg
):
    agent_id, full_key = _create_agent_with_key()

    response = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "first turn"},
            "prompt_context": prompt_context,
        },
    )

    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "invalid_input"
    assert mock_schedule_bg.call_count == 0


def test_create_rejects_non_finite_numbers_from_raw_json() -> None:
    agent_id, full_key = _create_agent_with_key()
    body = (
        '{{"agent_id": {agent_id}, "message": {{"role": "user", "content": '
        '"first turn"}}, "prompt_context": {{"nan": NaN}}}}'
    ).format(agent_id=agent_id)

    response = client.post(
        "/v1/chat/tasks",
        headers={**_bearer(full_key), "Content-Type": "application/json"},
        content=body,
    )

    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "invalid_input"


@pytest.mark.parametrize(
    "prompt_context",
    [
        pytest.param({"nested": {"a": 1}}, id="nested-object"),
        pytest.param({"too_big": "x" * 5000}, id="over-size"),
        pytest.param({"has space": "v"}, id="key-with-space"),
        pytest.param({"k" * 65: "v"}, id="key-over-64-characters"),
    ],
)
def test_append_rejects_malformed_or_oversized_context(prompt_context) -> None:
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_task_status(task_id, TaskStatus.COMPLETED)

    response = _append(full_key, agent_id, task_id, prompt_context=prompt_context)

    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "invalid_input"
    assert "prompt_context" not in _task_agent_config(task_id)


def _context_with_rendered_size(size: int) -> dict[str, Any]:
    context = {"k": "x" * (size - 9)}
    assert prompt_context_data_size(context) == size
    return context


def test_create_accepts_exactly_the_byte_limit_and_rejects_one_more() -> None:
    agent_id, full_key = _create_agent_with_key()
    limit = PROMPT_CONTEXT_MAX_DATA_BYTES

    accepted = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "first turn"},
            "prompt_context": _context_with_rendered_size(limit),
        },
    )
    rejected = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "first turn"},
            "prompt_context": _context_with_rendered_size(limit + 1),
        },
    )

    assert accepted.status_code == 202, accepted.text
    assert rejected.status_code == 422, rejected.text
    assert rejected.json()["error"]["code"] == "invalid_input"


def test_append_accepts_exactly_the_byte_limit_and_rejects_one_more() -> None:
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_task_status(task_id, TaskStatus.COMPLETED)
    limit = PROMPT_CONTEXT_MAX_DATA_BYTES

    rejected = _append(
        full_key,
        agent_id,
        task_id,
        prompt_context=_context_with_rendered_size(limit + 1),
    )
    assert rejected.status_code == 422, rejected.text
    assert "prompt_context" not in _task_agent_config(task_id)

    accepted = _append(
        full_key, agent_id, task_id, prompt_context=_context_with_rendered_size(limit)
    )
    assert accepted.status_code == 202, accepted.text
    assert _task_agent_config(task_id)["prompt_context"]["data"] == (
        _context_with_rendered_size(limit)
    )


@pytest.mark.parametrize("status", [TaskStatus.RUNNING, TaskStatus.WAITING_FOR_USER])
def test_rejected_append_leaves_the_stored_context_unchanged(
    status: TaskStatus,
) -> None:
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id, prompt_context={"date": "2026-10-19"})
    before = _task_agent_config(task_id)["prompt_context"]
    _force_task_status(task_id, status)

    response = _append(
        full_key, agent_id, task_id, prompt_context={"date": "2026-10-20"}
    )

    assert response.status_code == 409, response.text
    assert _task_agent_config(task_id)["prompt_context"] == before


def test_prompt_context_stays_off_the_other_entries() -> None:
    assert "prompt_context" in CreateTaskRequest.model_fields
    assert "prompt_context" in AppendMessageRequest.model_fields
    assert "prompt_context" in WidgetTaskCreateRequest.model_fields
    assert "prompt_context" not in ChatTaskCreateRequest.model_fields
    assert "prompt_context" not in ReplyRequest.model_fields
