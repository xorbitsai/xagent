"""The web UI's task detail for a task the retention purge expired (#2565).

``GET /api/chat/task/{id}`` serves a task to its owner and to admins. A task
retention expired answers ``410 task_expired`` to exactly those callers and
the existing 404 to everyone else.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.shared.auth_database import auth_db_override
from xagent.web.api.chat import chat_router
from xagent.web.models.auth_database import get_auth_db
from xagent.web.models.database import get_db
from xagent.web.models.expired_task import ExpiredTaskTombstone
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User

from .conftest import (
    _admin_headers,
    _direct_db_session,
    _override_get_db,
    _register_second_user,
)

pytestmark = pytest.mark.usefixtures("_test_db")

# The shared test app does not mount the chat router; this one does, on the
# same database, so the conftest's auth helpers' tokens work against it.
_chat_app = FastAPI()
_chat_app.include_router(chat_router)
_chat_app.dependency_overrides[get_db] = _override_get_db
_chat_app.dependency_overrides[get_auth_db] = auth_db_override(_override_get_db)
client = TestClient(_chat_app, raise_server_exceptions=False)

_EXPIRED_AT = datetime(2026, 9, 1, 8, 30, 0, tzinfo=timezone.utc)
#: Far above any id a test creates, so no live task holds it by accident.
_UNUSED_TASK_ID = 910_000


def _user_id(username: str) -> int:
    db = _direct_db_session()
    try:
        return int(db.query(User).filter(User.username == username).one().id)
    finally:
        db.close()


def _insert_tombstone(*, task_id: int, user_id: int, is_visible: bool) -> None:
    db = _direct_db_session()
    try:
        db.add(
            ExpiredTaskTombstone(
                task_id=task_id,
                user_id=user_id,
                source="internal",
                is_visible=is_visible,
                is_channel_plumbing=False,
                task_created_at=_EXPIRED_AT,
                expired_at=_EXPIRED_AT,
            )
        )
        db.commit()
    finally:
        db.close()


def _assert_expired(response, task_id: int) -> None:
    assert response.status_code == 410, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "task_expired"
    assert detail["task_id"] == task_id
    assert datetime.fromisoformat(detail["expired_at"]) == _EXPIRED_AT


def test_expired_task_detail_is_disclosed_to_the_owner_and_admins_only() -> None:
    admin = _admin_headers()
    owner = _register_second_user(username="chatexpiredowner")
    stranger = _register_second_user(username="chatexpiredstranger")
    task_id = _UNUSED_TASK_ID + 1
    # This route serves visible and hidden tasks alike, so visibility is not
    # part of its predicate.
    _insert_tombstone(
        task_id=task_id, user_id=_user_id("chatexpiredowner"), is_visible=True
    )

    _assert_expired(client.get(f"/api/chat/task/{task_id}", headers=owner), task_id)
    _assert_expired(client.get(f"/api/chat/task/{task_id}", headers=admin), task_id)

    denied = client.get(f"/api/chat/task/{task_id}", headers=stranger)
    assert denied.status_code == 404, denied.text
    assert denied.json() == {"detail": "Task not found"}


def test_live_task_wins_over_a_tombstone_with_the_same_id() -> None:
    admin = _admin_headers()
    admin_id = _user_id("admin")
    db = _direct_db_session()
    try:
        task = Task(
            user_id=admin_id,
            title="Live task",
            description="Live task",
            status=TaskStatus.COMPLETED,
        )
        db.add(task)
        db.commit()
        task_id = int(task.id)
    finally:
        db.close()
    _insert_tombstone(task_id=task_id, user_id=admin_id, is_visible=True)

    response = client.get(f"/api/chat/task/{task_id}", headers=admin)

    assert response.status_code == 200, response.text
    assert response.json()["title"] == "Live task"
