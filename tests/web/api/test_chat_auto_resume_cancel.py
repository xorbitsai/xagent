"""``POST /api/chat/task/{id}/auto-resume/cancel``: a person on the web UI
cancels the automatic resume a paused task waits for.

The task's owner or an admin may; anyone else reads 404 and an expired task
410, as for ``GET /api/chat/task/{id}``. A settled answer is 200 with an
``outcome`` and the task as GET reads it; an SDK task is 409
``auto_resume_not_cancellable`` and a cancel that kept losing a race 409
``auto_resume_cancel_raced``. A cancel's ``notice`` is the message broadcast, and the task's
cached responses are dropped so the next GET reads the new fence.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.shared.auth_database import auth_db_override
from xagent.web.api.chat import chat_router
from xagent.web.models.auth_database import get_auth_db
from xagent.web.models.database import get_db, get_session_local
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_auto_recovery import TaskAutoRecovery, TaskRecoveryEvent
from xagent.web.models.user import User
from xagent.web.services import task_events, task_orchestrator
from xagent.web.services import web_auto_resume_cancel as web_cancel
from xagent.web.services.hot_path_cache import (
    InMemoryTTLCache,
    set_cache_backend_for_testing,
    web_task_detail_key,
)
from xagent.web.services.task_auto_resume import AutoResumeCancel

from .conftest import (
    _admin_headers,
    _direct_db_session,
    _override_get_db,
    _register_second_user,
)
from .expired_task_shared import assert_task_expired, insert_tombstone

pytestmark = pytest.mark.usefixtures("_test_db")

_chat_app = FastAPI()
_chat_app.include_router(chat_router)
_chat_app.dependency_overrides[get_db] = _override_get_db
_chat_app.dependency_overrides[get_auth_db] = auth_db_override(_override_get_db)
client = TestClient(_chat_app, raise_server_exceptions=False)

RUN_ID = "run-current"
STATE_VERSION = 7
INTERRUPTED_AT = datetime(2026, 10, 1, 8, 30, tzinfo=timezone.utc)
STOPPED = {
    "status": "stopped",
    "stop_reason": "cancelled",
    "reason": "lease_expired",
    "attempt": 0,
    "next_attempt_at": None,
}


@pytest.fixture
def published(monkeypatch) -> list[dict[str, Any]]:
    """The broadcast messages, each with the row as committed when sent."""
    messages: list[dict[str, Any]] = []

    async def publish(message: dict[str, Any], task_id: int) -> None:
        with get_session_local()() as db:
            row = db.get(TaskAutoRecovery, task_id)
            messages.append({**message, "_row": (row.state, row.state_detail)})

    monkeypatch.setattr(task_events, "publish_task_event", publish)
    return messages


@pytest.fixture
def cache() -> Iterator[InMemoryTTLCache]:
    backend = InMemoryTTLCache()
    set_cache_backend_for_testing(backend)
    yield backend
    set_cache_backend_for_testing(None)


def _user_id(username: str) -> int:
    db = _direct_db_session()
    try:
        return int(db.query(User).filter(User.username == username).one().id)
    finally:
        db.close()


def _task(
    owner: str = "bob",
    *,
    row: dict[str, Any] | None = None,
    control_state: str = "paused",
    source: str | None = None,
) -> int:
    """A PAUSED task of ``owner`` at ``RUN_ID``/``STATE_VERSION`` with a
    ``scheduled`` recovery row (``row`` overrides; ``{"state": None}``: no
    row)."""
    db = _direct_db_session()
    try:
        task = Task(
            user_id=_user_id(owner),
            title="Interrupted task",
            description="Interrupted task",
            status=TaskStatus.PAUSED,
            control_state=control_state,
            run_id=RUN_ID,
            state_version=STATE_VERSION,
            source=source,
        )
        db.add(task)
        db.commit()
        task_id = int(task.id)
        values = {
            "run_id": RUN_ID,
            "reason": "lease_expired",
            "state": "scheduled",
            "paused_state_version": STATE_VERSION,
            "interrupted_at": INTERRUPTED_AT,
            "episode_started_at": INTERRUPTED_AT,
            "next_attempt_at": INTERRUPTED_AT,
            **(row or {}),
        }
        if values["state"] is not None:
            db.add(TaskAutoRecovery(task_id=task_id, **values))
            db.commit()
        return task_id
    finally:
        db.close()


def _row(task_id: int) -> tuple[str, str | None, int] | None:
    db = _direct_db_session()
    try:
        row = db.get(TaskAutoRecovery, task_id)
        version = int(db.get(Task, task_id).state_version)
        return None if row is None else (row.state, row.state_detail, version)
    finally:
        db.close()


def _post(task_id: int, headers: dict[str, str], body: Any = None):
    url = f"/api/chat/task/{task_id}/auto-resume/cancel"
    if body is None:
        return client.post(url, headers=headers)
    return client.post(url, headers=headers, json=body)


@pytest.fixture
def bob() -> dict[str, str]:
    _admin_headers()
    return _register_second_user("bob")


# --------------------------------------------------------------------------
# A1. The owner cancels; the answer, the broadcast and GET agree
# --------------------------------------------------------------------------


def test_the_owner_cancels_and_the_notice_is_the_broadcast(bob, published):
    task_id = _task()

    response = _post(task_id, bob)

    assert response.status_code == 200, response.text
    body = response.json()
    assert {k: body[k] for k in ("task_id", "outcome", "status", "control_state")} == {
        "task_id": task_id,
        "outcome": "cancelled",
        "status": "paused",
        "control_state": "paused",
    }
    assert (body["run_id"], body["state_version"]) == (RUN_ID, STATE_VERSION + 1)
    assert body["auto_recovery"] == {
        "reason": "lease_expired",
        "interrupted_at": "2026-10-01T08:30:00+00:00",
        "run_id": RUN_ID,
        "state_version": STATE_VERSION + 1,
        "auto_resume": STOPPED,
    }
    (message,) = published
    # Broadcast after the commit, and the very message returned.
    assert message.pop("_row") == ("manual", "user_cancelled")
    assert body["notice"] == message
    assert message["type"] == "task_recovery_update"
    assert message["auto_resume"] == STOPPED
    assert _row(task_id) == ("manual", "user_cancelled", STATE_VERSION + 1)


def test_a_get_right_after_reads_the_new_fence(bob, published, cache):
    task_id = _task()
    before = client.get(f"/api/chat/task/{task_id}", headers=bob).json()
    assert before["state_version"] == STATE_VERSION
    assert cache.get_json(web_task_detail_key(task_id)) is not None

    assert _post(task_id, bob).json()["outcome"] == "cancelled"

    # The task row's ``updated_at`` (the cache's key) has whole-second
    # resolution on SQLite: only the dropped entry keeps GET from serving
    # the old version next to the new fence.
    assert cache.get_json(web_task_detail_key(task_id)) is None
    after = client.get(f"/api/chat/task/{task_id}", headers=bob).json()
    assert after["state_version"] == STATE_VERSION + 1
    assert after["auto_recovery"]["state_version"] == STATE_VERSION + 1
    assert after["auto_recovery"]["auto_resume"] == STOPPED


def test_a_second_cancel_reads_already_cancelled(bob, published):
    task_id = _task()
    assert _post(task_id, bob).json()["outcome"] == "cancelled"

    response = _post(task_id, bob)

    assert response.status_code == 200
    body = response.json()
    assert (body["outcome"], body["notice"]) == ("already_cancelled", None)
    assert body["auto_recovery"]["auto_resume"] == STOPPED
    assert len(published) == 1


# --------------------------------------------------------------------------
# A2. Who may cancel
# --------------------------------------------------------------------------


def test_another_user_reads_404_and_nothing_changes(bob, published):
    task_id = _task()
    carol = _register_second_user("carol")

    response = _post(task_id, carol)

    assert response.status_code == 404
    assert response.json() == {"detail": "Task not found"}
    assert _row(task_id) == ("scheduled", None, STATE_VERSION)
    assert published == []


def test_another_user_never_reaches_the_task_lock(bob, monkeypatch):
    """Checked before the locked transaction: no one queues on (or learns
    of) another user's task row lock."""
    task_id = _task()
    carol = _register_second_user("carol")

    async def unexpected(**_kwargs):
        raise AssertionError("the cancel must not run")

    monkeypatch.setattr("xagent.web.api.chat.cancel_auto_resume_for_web", unexpected)

    assert _post(task_id, carol).status_code == 404


def test_an_unknown_task_reads_404(bob):
    assert _post(999_999, bob).status_code == 404


def test_unauthenticated_reads_401(bob):
    task_id = _task()

    assert _post(task_id, {}).status_code == 401
    assert _row(task_id) == ("scheduled", None, STATE_VERSION)


def test_an_expired_task_reads_410_to_its_owner_only(bob):
    insert_tombstone(task_id=4242, user_id=_user_id("bob"))

    assert_task_expired(_post(4242, bob), 4242)
    assert _post(4242, _register_second_user("carol")).status_code == 404


def test_an_admin_cancels_another_users_task(bob, published):
    task_id = _task()

    response = _post(task_id, _admin_headers())

    assert response.json()["outcome"] == "cancelled"
    db = _direct_db_session()
    try:
        event = db.query(TaskRecoveryEvent).filter_by(task_id=task_id).one()
        assert event.detail["actor_user_id"] == _user_id("admin")
    finally:
        db.close()


def test_a_service_answering_missing_reads_404(bob, monkeypatch):
    task_id = _task()

    async def gone(**_kwargs):
        return web_cancel.WebAutoResumeCancelResult(
            web_cancel.WebAutoResumeCancelOutcome.MISSING
        )

    monkeypatch.setattr("xagent.web.api.chat.cancel_auto_resume_for_web", gone)

    assert _post(task_id, bob).status_code == 404


def test_a_task_expired_between_the_check_and_the_lock_reads_410(bob, monkeypatch):
    """Retention purged the task after the unlocked check: its owner is told
    it expired, as GET would."""
    task_id = _task()
    attempt = web_cancel._cancel_once_no_commit

    def purged_first(db, **kwargs):
        purge = _direct_db_session()
        try:
            purge.query(TaskAutoRecovery).filter_by(task_id=task_id).delete()
            purge.query(Task).filter_by(id=task_id).delete()
            purge.commit()
        finally:
            purge.close()
        insert_tombstone(task_id=task_id, user_id=_user_id("bob"))
        return attempt(db, **kwargs)

    monkeypatch.setattr(web_cancel, "_cancel_once_no_commit", purged_first)

    assert_task_expired(_post(task_id, bob), task_id)


# --------------------------------------------------------------------------
# A3. Nothing to cancel, or not here
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("task", "body", "outcome", "control_state"),
    [
        pytest.param(
            {"control_state": "resume_requested"},
            None,
            "resume_in_progress",
            "resume_requested",
            id="handing_off",
        ),
        pytest.param(
            {"row": {"state": None}}, None, "not_pending", "paused", id="no_row"
        ),
        pytest.param(
            {"row": {"state": "exhausted"}},
            None,
            "not_pending",
            "paused",
            id="exhausted",
        ),
        pytest.param(
            {}, {"run_id": "run-earlier"}, "not_pending", "paused", id="other_run"
        ),
    ],
)
def test_nothing_to_cancel_is_an_outcome(
    bob, published, task, body, outcome, control_state
):
    task_id = _task(**task)
    before = _row(task_id)

    response = _post(task_id, bob, body)

    assert response.status_code == 200, response.text
    answer = response.json()
    assert (answer["outcome"], answer["control_state"], answer["notice"]) == (
        outcome,
        control_state,
        None,
    )
    assert answer["state_version"] == STATE_VERSION
    assert _row(task_id) == before
    assert published == []


def test_the_run_seen_is_cancelled(bob, published):
    task_id = _task()

    assert _post(task_id, bob, {"run_id": RUN_ID}).json()["outcome"] == "cancelled"


def test_an_sdk_task_is_not_cancellable(bob, published):
    task_id = _task(source="sdk")

    response = _post(task_id, bob)

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "auto_resume_not_cancellable"
    assert _row(task_id) == ("scheduled", None, STATE_VERSION)
    assert published == []


def test_a_cancel_that_keeps_racing_reads_409_cancel_raced(bob, published, monkeypatch):
    task_id = _task()
    monkeypatch.setattr(
        web_cancel,
        "cancel_pending_auto_resume_no_commit",
        lambda *_args, **_kwargs: AutoResumeCancel(cancelled=False, raced=True),
    )

    response = _post(task_id, bob)

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "auto_resume_cancel_raced"
    assert _row(task_id) == ("scheduled", None, STATE_VERSION)


@pytest.mark.parametrize(
    "body",
    [{"run_id": RUN_ID, "state_version": 7}, {"run_id": "r" * 65}, {"run_id": ""}],
    ids=["unknown_field", "run_id_too_long", "run_id_empty"],
)
def test_an_invalid_body_reads_422(bob, body):
    task_id = _task()

    assert _post(task_id, bob, body).status_code == 422
    assert _row(task_id) == ("scheduled", None, STATE_VERSION)


def test_a_failing_cancel_reads_500_and_writes_nothing(bob, published, monkeypatch):
    task_id = _task()

    def failing(db, task, **_kwargs):
        # A write staged before the failure must not be kept.
        db.query(Task).filter_by(id=task.id).update({"state_version": 99})
        raise RuntimeError("cancel failed")

    monkeypatch.setattr(web_cancel, "cancel_pending_auto_resume_no_commit", failing)

    response = _post(task_id, bob)

    assert response.status_code == 500
    assert "cancel failed" not in response.text
    assert _row(task_id) == ("scheduled", None, STATE_VERSION)
    assert published == []


def test_a_failed_cache_drop_still_answers_the_cancel(bob, published, monkeypatch):
    """After the commit only best-effort work runs: the person is still told
    the cancel happened, with its notice, and it is broadcast."""
    task_id = _task()

    def failing(_task_id):
        raise RuntimeError("cache down")

    monkeypatch.setattr(task_orchestrator, "invalidate_task_cache", failing)

    response = _post(task_id, bob)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["outcome"] == "cancelled"
    assert body["state_version"] == STATE_VERSION + 1
    (message,) = published
    message.pop("_row")
    assert body["notice"] == message
