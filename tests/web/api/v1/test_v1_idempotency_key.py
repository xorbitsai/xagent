"""``Idempotency-Key`` on ``POST /v1/chat/tasks`` and the append endpoint."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from xagent.web.models.api_idempotency_key import ApiIdempotencyKey
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task, TaskStatus
from xagent.web.services import api_idempotency

from ..conftest import _admin_headers, _direct_db_session, client

pytestmark = pytest.mark.usefixtures("_test_db")


@pytest.fixture(autouse=True)
def mock_schedule_bg():
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ) as mocked:
        yield mocked


def _create_agent_with_key(name: str = "idempotency agent") -> tuple[int, str]:
    headers = _admin_headers()
    agent_resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": name,
            "description": "test",
            "instructions": "you are a test agent",
            "execution_mode": "balanced",
        },
    )
    assert agent_resp.status_code == 200, agent_resp.text
    agent_id = agent_resp.json()["id"]
    key_resp = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
    assert key_resp.status_code == 200, key_resp.text
    return agent_id, key_resp.json()["full_key"]


def _headers(full_key: str, idempotency_key: str | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {full_key}"}
    if idempotency_key is not None:
        headers["Idempotency-Key"] = idempotency_key
    return headers


def _create(
    full_key: str,
    agent_id: int,
    idempotency_key: str | None = None,
    content: str = "first turn",
) -> Any:
    return client.post(
        "/v1/chat/tasks",
        headers=_headers(full_key, idempotency_key),
        json={"agent_id": agent_id, "message": {"role": "user", "content": content}},
    )


def _append(
    full_key: str,
    agent_id: int,
    task_id: int,
    idempotency_key: str | None = None,
    content: str = "next turn",
) -> Any:
    return client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_headers(full_key, idempotency_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": content},
        },
    )


def _task_count() -> int:
    db = _direct_db_session()
    try:
        return db.query(Task).count()
    finally:
        db.close()


def _user_message_count(task_id: int) -> int:
    db = _direct_db_session()
    try:
        return (
            db.query(TaskChatMessage)
            .filter(TaskChatMessage.task_id == task_id, TaskChatMessage.role == "user")
            .count()
        )
    finally:
        db.close()


def _set_status(task_id: int, status: TaskStatus) -> None:
    db = _direct_db_session()
    try:
        db.query(Task).filter(Task.id == task_id).one().status = status
        db.commit()
    finally:
        db.close()


def _only_row() -> dict[str, Any]:
    db = _direct_db_session()
    try:
        row = db.query(ApiIdempotencyKey).one()
        return {
            "id": row.id,
            "state": row.state,
            "task_id": row.task_id,
            "fingerprint": row.request_fingerprint,
        }
    finally:
        db.close()


def _patch_row(**values: Any) -> None:
    db = _direct_db_session()
    try:
        row = db.query(ApiIdempotencyKey).one()
        for name, value in values.items():
            setattr(row, name, value)
        db.commit()
    finally:
        db.close()


def test_without_header_every_create_starts_a_task() -> None:
    agent_id, key = _create_agent_with_key()

    first = _create(key, agent_id)
    second = _create(key, agent_id)

    assert first.status_code == 202 and second.status_code == 202
    assert first.json()["task_id"] != second.json()["task_id"]
    assert _task_count() == 2
    db = _direct_db_session()
    try:
        assert db.query(ApiIdempotencyKey).count() == 0
    finally:
        db.close()


def test_same_key_and_body_replays_the_original_task() -> None:
    agent_id, key = _create_agent_with_key()

    first = _create(key, agent_id, "retry-1")
    replay = _create(key, agent_id, "retry-1")

    assert first.status_code == 202
    assert "idempotent-replayed" not in first.headers
    assert replay.status_code == 202
    assert replay.headers["Idempotent-Replayed"] == "true"
    assert replay.json() == first.json()
    assert _task_count() == 1
    assert _only_row()["state"] == "completed"


def test_same_key_with_different_body_is_422() -> None:
    agent_id, key = _create_agent_with_key()
    assert _create(key, agent_id, "retry-2", content="one").status_code == 202

    changed = _create(key, agent_id, "retry-2", content="two")

    assert changed.status_code == 422
    assert changed.json()["error"]["code"] == "idempotency_key_reused"
    assert _task_count() == 1


def _create_body_dump(agent_id: int, content: str) -> dict[str, Any]:
    from xagent.web.schemas.v1 import CreateTaskRequest

    return CreateTaskRequest(
        agent_id=agent_id, message={"role": "user", "content": content}
    ).model_dump(mode="json")


def test_retry_while_first_request_is_in_flight_is_409() -> None:
    agent_id, key = _create_agent_with_key()
    # The first request reserved the key and has not finished.
    in_flight = api_idempotency.acquire_sync(
        scope_kind="agent",
        scope_id=agent_id,
        operation="create_task",
        key="retry-3",
        fingerprint=api_idempotency.request_fingerprint(
            _create_body_dump(agent_id, "first turn")
        ),
    )
    assert isinstance(in_flight, api_idempotency.IdempotencyClaim)

    retry = _create(key, agent_id, "retry-3")

    assert retry.status_code == 409
    assert retry.json()["error"]["code"] == "idempotency_key_in_progress"
    assert retry.headers["Retry-After"]
    assert _task_count() == 0


def test_expired_key_can_be_reused_with_a_different_body() -> None:
    agent_id, key = _create_agent_with_key()
    first = _create(key, agent_id, "retry-4", content="one")
    assert first.status_code == 202
    _patch_row(expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))

    again = _create(key, agent_id, "retry-4", content="two")

    assert again.status_code == 202
    assert again.json()["task_id"] != first.json()["task_id"]
    assert "idempotent-replayed" not in again.headers
    assert _task_count() == 2


def test_stale_reservation_that_created_nothing_is_taken_over() -> None:
    agent_id, key = _create_agent_with_key()
    api_idempotency.acquire_sync(
        scope_kind="agent",
        scope_id=agent_id,
        operation="create_task",
        key="retry-5",
        fingerprint=api_idempotency.request_fingerprint(
            _create_body_dump(agent_id, "first turn")
        ),
    )
    _patch_row(lease_expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))

    response = _create(key, agent_id, "retry-5")

    assert response.status_code == 202
    assert _task_count() == 1
    assert _only_row()["task_id"] == response.json()["task_id"]


def test_accepted_task_with_unrecorded_response_is_not_started_again() -> None:
    agent_id, key = _create_agent_with_key()
    first = _create(key, agent_id, "retry-6")
    task_id = first.json()["task_id"]
    # The process died after the create committed, before the response was
    # recorded.
    _patch_row(
        state="in_progress",
        response_body=None,
        response_status=None,
        lease_expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )

    retry = _create(key, agent_id, "retry-6")

    assert retry.status_code == 409
    error = retry.json()["error"]
    assert error["code"] == "idempotency_conflict"
    assert error["details"] == {"task_id": task_id}
    assert _task_count() == 1


def test_rejected_create_releases_the_key() -> None:
    agent_id, key = _create_agent_with_key()

    rejected = _create(key, agent_id + 999, "retry-7")
    assert rejected.status_code == 404
    db = _direct_db_session()
    try:
        assert db.query(ApiIdempotencyKey).count() == 0
    finally:
        db.close()

    # The same key now works for the corrected request.
    accepted = _create(key, agent_id, "retry-7")
    assert accepted.status_code == 202
    assert _task_count() == 1


def test_keys_are_scoped_to_the_credential_owner() -> None:
    agent_a, key_a = _create_agent_with_key("agent a")
    agent_b, key_b = _create_agent_with_key("agent b")

    first = _create(key_a, agent_a, "shared-key")
    second = _create(key_b, agent_b, "shared-key")

    assert first.status_code == 202 and second.status_code == 202
    assert first.json()["task_id"] != second.json()["task_id"]
    assert _task_count() == 2


def test_replay_fails_when_the_original_task_was_deleted() -> None:
    agent_id, key = _create_agent_with_key()
    first = _create(key, agent_id, "retry-8")
    task_id = first.json()["task_id"]
    _set_status(task_id, TaskStatus.COMPLETED)
    db = _direct_db_session()
    try:
        db.query(Task).filter(Task.id == task_id).delete()
        db.commit()
    finally:
        db.close()

    retry = _create(key, agent_id, "retry-8")

    assert retry.status_code == 409
    assert retry.json()["error"]["code"] == "idempotency_conflict"


@pytest.mark.parametrize("value", ["x" * 256, "has space", "tab\tkey"])
def test_malformed_key_is_422(value: str) -> None:
    agent_id, key = _create_agent_with_key()

    response = client.post(
        "/v1/chat/tasks",
        headers={
            "Authorization": f"Bearer {key}",
            "Idempotency-Key": value.encode("utf-8").decode("latin-1"),
        },
        json={"agent_id": agent_id, "message": {"role": "user", "content": "hi"}},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "idempotency_key_invalid"
    assert _task_count() == 0


def test_append_same_key_replays_without_a_second_turn() -> None:
    agent_id, key = _create_agent_with_key()
    task_id = _create(key, agent_id).json()["task_id"]
    _set_status(task_id, TaskStatus.COMPLETED)

    first = _append(key, agent_id, task_id, "append-1")
    assert first.status_code == 202, first.text
    messages_after_first = _user_message_count(task_id)
    # Without replay protection this retry would be rejected as task_busy or,
    # once the turn finished, start another turn.
    _set_status(task_id, TaskStatus.COMPLETED)
    replay = _append(key, agent_id, task_id, "append-1")

    assert replay.status_code == 202
    assert replay.headers["Idempotent-Replayed"] == "true"
    assert replay.json() == first.json()
    assert _user_message_count(task_id) == messages_after_first


def test_append_same_key_with_different_body_is_422() -> None:
    agent_id, key = _create_agent_with_key()
    task_id = _create(key, agent_id).json()["task_id"]
    _set_status(task_id, TaskStatus.COMPLETED)
    assert _append(key, agent_id, task_id, "append-2", "one").status_code == 202
    _set_status(task_id, TaskStatus.COMPLETED)

    changed = _append(key, agent_id, task_id, "append-2", "two")

    assert changed.status_code == 422
    assert changed.json()["error"]["code"] == "idempotency_key_reused"


def test_append_key_is_scoped_to_its_task() -> None:
    agent_id, key = _create_agent_with_key()
    task_a = _create(key, agent_id).json()["task_id"]
    task_b = _create(key, agent_id).json()["task_id"]
    for task_id in (task_a, task_b):
        _set_status(task_id, TaskStatus.COMPLETED)

    on_a = _append(key, agent_id, task_a, "append-3")
    on_b = _append(key, agent_id, task_b, "append-3")

    assert on_a.status_code == 202 and on_b.status_code == 202
    assert on_a.json()["task_id"] == task_a
    assert on_b.json()["task_id"] == task_b


def test_append_without_header_is_unchanged() -> None:
    agent_id, key = _create_agent_with_key()
    task_id = _create(key, agent_id).json()["task_id"]
    _set_status(task_id, TaskStatus.COMPLETED)

    first = _append(key, agent_id, task_id)
    _set_status(task_id, TaskStatus.COMPLETED)
    second = _append(key, agent_id, task_id)

    assert first.status_code == 202 and second.status_code == 202
    assert "idempotent-replayed" not in second.headers
