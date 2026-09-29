"""The web UI's task detail and status for a task the retention purge expired (#2565).

``GET /api/chat/task/{id}`` and ``GET /api/chat/task/{id}/status`` both serve
a task to its owner and to admins, through the same owner-or-admin scope
(``_raise_task_expired_or_not_found`` in ``api/chat.py``). A task retention
expired answers ``410 task_expired`` to exactly those callers and the
existing 404 to everyone else, on both routes alike -- unlike ``PUT``/
``DELETE``, which stay ``404`` (see the comments at their not-found raises).
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.shared.auth_database import auth_db_override
from xagent.web.api.chat import chat_router
from xagent.web.models.auth_database import get_auth_db
from xagent.web.models.database import get_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User

from .conftest import (
    _admin_headers,
    _direct_db_session,
    _override_get_db,
    _register_second_user,
)
from .expired_task_shared import assert_task_expired, insert_tombstone

pytestmark = pytest.mark.usefixtures("_test_db")

# The shared test app does not mount the chat router; this one does, on the
# same database, so the conftest's auth helpers' tokens work against it.
_chat_app = FastAPI()
_chat_app.include_router(chat_router)
_chat_app.dependency_overrides[get_db] = _override_get_db
_chat_app.dependency_overrides[get_auth_db] = auth_db_override(_override_get_db)
client = TestClient(_chat_app, raise_server_exceptions=False)

#: Far above any id a test creates, so no live task holds it by accident.
_UNUSED_TASK_ID = 910_000

#: The two task-id routes ``_raise_task_expired_or_not_found`` guards.
_TASK_ID_ROUTES = ["", "/status"]


def _user_id(username: str) -> int:
    db = _direct_db_session()
    try:
        return int(db.query(User).filter(User.username == username).one().id)
    finally:
        db.close()


@pytest.mark.parametrize("route_suffix", _TASK_ID_ROUTES, ids=["detail", "status"])
def test_expired_task_detail_is_disclosed_to_the_owner_and_admins_only(
    route_suffix: str,
) -> None:
    admin = _admin_headers()
    owner = _register_second_user(username="chatexpiredowner")
    stranger = _register_second_user(username="chatexpiredstranger")
    task_id = _UNUSED_TASK_ID + 1
    # This route serves visible and hidden tasks alike, so visibility is not
    # part of its predicate.
    insert_tombstone(
        task_id=task_id, user_id=_user_id("chatexpiredowner"), is_visible=True
    )

    assert_task_expired(
        client.get(f"/api/chat/task/{task_id}{route_suffix}", headers=owner), task_id
    )
    assert_task_expired(
        client.get(f"/api/chat/task/{task_id}{route_suffix}", headers=admin), task_id
    )

    denied = client.get(f"/api/chat/task/{task_id}{route_suffix}", headers=stranger)
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
    insert_tombstone(task_id=task_id, user_id=admin_id, is_visible=True)

    response = client.get(f"/api/chat/task/{task_id}", headers=admin)

    assert response.status_code == 200, response.text
    assert response.json()["title"] == "Live task"
