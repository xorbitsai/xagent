"""``auto_recovery`` on the web UI's task detail and status.

``GET /api/chat/task/{id}`` and ``GET /api/chat/task/{id}/status`` report the
interruption that left a PAUSED task as it stands: its ``task_auto_recovery``
row, only while the row still names the task's current run and
``state_version``. The row is not cleared when the run is resumed, so a stale
row must read as ``null``; a FAILED run's row is not shown, and the
operator-only fields never leave the server. History replay carries the same
view, and its reasserted pause says why the run stopped.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

from tests.shared.auth_database import auth_db_override
from xagent.web.api.chat import chat_router
from xagent.web.models.auth_database import get_auth_db
from xagent.web.models.database import get_db, release_db_connection_if_clean
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_auto_recovery import TaskAutoRecovery
from xagent.web.models.user import User
from xagent.web.services.hot_path_cache import (
    InMemoryTTLCache,
    set_cache_backend_for_testing,
)
from xagent.web.services.task_auto_recovery import current_auto_recovery_view

from .conftest import _admin_headers, _direct_db_session, _override_get_db

pytestmark = pytest.mark.usefixtures("_test_db")

_chat_app = FastAPI()
_chat_app.include_router(chat_router)
_chat_app.dependency_overrides[get_db] = _override_get_db
_chat_app.dependency_overrides[get_auth_db] = auth_db_override(_override_get_db)
client = TestClient(_chat_app, raise_server_exceptions=False)

ROUTES = pytest.mark.parametrize(
    "route_suffix", ["", "/status"], ids=["detail", "status"]
)

RUN_ID = "run-current"
STATE_VERSION = 7
INTERRUPTED_AT = datetime(2026, 10, 1, 8, 30, tzinfo=timezone.utc)
SCHEDULED = {
    "state": "scheduled",
    "next_attempt_at": datetime(2026, 10, 1, 8, 31, 10, tzinfo=timezone.utc),
}


def _paused_task(
    *,
    row: dict | None,
    status: TaskStatus = TaskStatus.PAUSED,
    conversation_storage_version: int = 1,
) -> int:
    """A ``status`` (PAUSED) task at ``RUN_ID``/``STATE_VERSION``, with a
    recovery row built from ``row`` overrides (``None``: no row)."""
    db = _direct_db_session()
    try:
        admin_id = int(db.query(User).filter(User.username == "admin").one().id)
        task = Task(
            user_id=admin_id,
            title="Interrupted task",
            description="Interrupted task",
            status=status,
            control_state=status.value,
            run_id=RUN_ID,
            state_version=STATE_VERSION,
            conversation_storage_version=conversation_storage_version,
        )
        db.add(task)
        db.commit()
        task_id = int(task.id)
        if row is not None:
            db.add(
                TaskAutoRecovery(
                    task_id=task_id,
                    **{
                        "run_id": RUN_ID,
                        "reason": "lease_expired",
                        "state": "manual",
                        "state_detail": "operator-detail",
                        "paused_state_version": STATE_VERSION,
                        "interrupted_at": INTERRUPTED_AT,
                        "episode_started_at": INTERRUPTED_AT,
                        "total_resumes": 2,
                        "last_command_id": "command-1",
                        "last_error": "operator-only diagnostic",
                        **row,
                    },
                )
            )
            db.commit()
        return task_id
    finally:
        db.close()


def _auto_recovery(task_id: int, route_suffix: str) -> object:
    response = client.get(
        f"/api/chat/task/{task_id}{route_suffix}", headers=_admin_headers()
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "auto_recovery" in body
    return body["auto_recovery"]


@ROUTES
def test_current_interruption_is_shown_with_its_fence(route_suffix: str) -> None:
    _admin_headers()
    task_id = _paused_task(row={})

    assert _auto_recovery(task_id, route_suffix) == {
        "reason": "lease_expired",
        "interrupted_at": "2026-10-01T08:30:00+00:00",
        "run_id": RUN_ID,
        "state_version": STATE_VERSION,
        # A recorded ``manual`` row: nothing automatic to show.
        "auto_resume": None,
    }


@ROUTES
def test_scheduled_resume_is_shown_as_its_notice(route_suffix: str) -> None:
    _admin_headers()
    task_id = _paused_task(row=SCHEDULED)

    view = _auto_recovery(task_id, route_suffix)

    assert view == {
        "reason": "lease_expired",
        "interrupted_at": "2026-10-01T08:30:00+00:00",
        "run_id": RUN_ID,
        "state_version": STATE_VERSION,
        "auto_resume": {
            "status": "scheduled",
            "stop_reason": None,
            "reason": "lease_expired",
            "attempt": 3,
            "next_attempt_at": "2026-10-01T08:31:10+00:00",
        },
    }


@ROUTES
@pytest.mark.parametrize(
    "stale",
    [
        {"paused_state_version": STATE_VERSION - 1},
        {"run_id": "run-earlier"},
    ],
    ids=["state_version", "run_id"],
)
def test_stale_interruption_is_hidden(route_suffix: str, stale: dict) -> None:
    """A row an earlier pause left behind: the task has resumed since."""
    _admin_headers()
    task_id = _paused_task(row=stale)

    assert _auto_recovery(task_id, route_suffix) is None


@ROUTES
@pytest.mark.parametrize("state", ["manual", "disabled"])
def test_failed_run_row_is_not_shown(route_suffix: str, state: str) -> None:
    """Settlement records a FAILED run's interruption fenced at the FAILED
    ``state_version``; it is bookkeeping, not a pause to explain."""
    _admin_headers()
    task_id = _paused_task(row={"state": state}, status=TaskStatus.FAILED)

    assert _auto_recovery(task_id, route_suffix) is None


@ROUTES
def test_unreadable_row_reads_as_null(route_suffix: str, caplog) -> None:
    """The view is informational: a failed read must not fail the request."""
    _admin_headers()
    task_id = _paused_task(row={})
    db = _direct_db_session()
    try:
        db.execute(text("DROP TABLE task_auto_recovery"))
        db.commit()
    finally:
        db.close()

    with caplog.at_level(logging.WARNING):
        assert _auto_recovery(task_id, route_suffix) is None
    assert "Reading the auto-recovery row" in caplog.text


def test_failed_read_leaves_the_session_usable_and_clean(caplog) -> None:
    """The read runs in a SAVEPOINT: after it fails the caller's transaction
    still runs statements (PostgreSQL would otherwise refuse them), and a
    successful read leaves the session releasable as read-only."""
    _admin_headers()
    task_id = _paused_task(row={})
    db = _direct_db_session()
    try:
        task = db.get(Task, task_id)
        assert current_auto_recovery_view(db, task) is not None
        assert release_db_connection_if_clean(db)

        db.execute(text("DROP TABLE task_auto_recovery"))
        db.commit()
        db.expunge_all()
        task = db.get(Task, task_id)
        with caplog.at_level(logging.WARNING):
            assert current_auto_recovery_view(db, task) is None
        assert "Reading the auto-recovery row" in caplog.text
        assert db.query(Task.id).filter(Task.id == task_id).scalar() == task_id
    finally:
        db.close()


@ROUTES
def test_task_without_interruption_reports_null(route_suffix: str) -> None:
    _admin_headers()
    task_id = _paused_task(row=None)

    assert _auto_recovery(task_id, route_suffix) is None


class _JsonCache(InMemoryTTLCache):
    """Round-trips values through JSON, as the Redis backend does, and
    records the keys written."""

    def __init__(self) -> None:
        super().__init__()
        self.sets: list[str] = []

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        self.sets.append(key)
        super().set_json(key, json.loads(json.dumps(value)), ttl_seconds)

    def get_json(self, key: str) -> Any | None:
        value = super().get_json(key)
        return None if value is None else json.loads(json.dumps(value))


@pytest.fixture
def _response_cache() -> Iterator[_JsonCache]:
    backend = _JsonCache()
    set_cache_backend_for_testing(backend)
    yield backend
    set_cache_backend_for_testing(None)


@ROUTES
@pytest.mark.usefixtures("_response_cache")
def test_cached_response_reads_the_row_afresh(route_suffix: str) -> None:
    """The response cache is keyed by the task row; the recovery row is not
    part of it, so a cache hit still reports the row as it stands."""
    _admin_headers()
    task_id = _paused_task(row=None)
    assert _auto_recovery(task_id, route_suffix) is None

    db = _direct_db_session()
    try:
        db.add(
            TaskAutoRecovery(
                task_id=task_id,
                run_id=RUN_ID,
                reason="llm_unavailable",
                state="manual",
                paused_state_version=STATE_VERSION,
                interrupted_at=INTERRUPTED_AT,
                episode_started_at=INTERRUPTED_AT,
            )
        )
        db.commit()
    finally:
        db.close()

    assert _auto_recovery(task_id, route_suffix) == {
        "reason": "llm_unavailable",
        "interrupted_at": "2026-10-01T08:30:00+00:00",
        "run_id": RUN_ID,
        "state_version": STATE_VERSION,
        "auto_resume": None,
    }


def _move_row(task_id: int, **values: object) -> None:
    """A sweeper transition: it writes the recovery row, never ``tasks``."""
    db = _direct_db_session()
    try:
        updated_at = db.get(Task, task_id).updated_at
        row = db.get(TaskAutoRecovery, task_id)
        for name, value in values.items():
            setattr(row, name, value)
        db.commit()
        assert db.get(Task, task_id).updated_at == updated_at
    finally:
        db.close()


def _status_and_attempt(view: object) -> tuple[str, str | None, int]:
    assert isinstance(view, dict)
    auto_resume = view["auto_resume"]
    return (auto_resume["status"], auto_resume["stop_reason"], auto_resume["attempt"])


@ROUTES
def test_cached_response_follows_the_sweeper(
    route_suffix: str, _response_cache: _JsonCache
) -> None:
    """The sweeper's transitions leave the task row (the response cache's
    key) alone; each read is a cache hit and still reports the row as it
    stands."""
    _admin_headers()
    task_id = _paused_task(row=SCHEDULED)
    assert _status_and_attempt(_auto_recovery(task_id, route_suffix)) == (
        "scheduled",
        None,
        3,
    )

    _move_row(task_id, state="dispatched", total_resumes=3)
    assert _status_and_attempt(_auto_recovery(task_id, route_suffix)) == (
        "resuming",
        None,
        3,
    )

    _move_row(task_id, state="exhausted", next_attempt_at=None)
    assert _status_and_attempt(_auto_recovery(task_id, route_suffix)) == (
        "stopped",
        "limit_reached",
        3,
    )
    # Written once, by the first read; the others were hits.
    assert len(_response_cache.sets) == 1


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (
            {},
            {
                "reason": "lease_expired",
                "interrupted_at": "2026-10-01T08:30:00+00:00",
                "run_id": RUN_ID,
                "state_version": STATE_VERSION,
                "auto_resume": None,
            },
        ),
        (
            SCHEDULED,
            {
                "reason": "lease_expired",
                "interrupted_at": "2026-10-01T08:30:00+00:00",
                "run_id": RUN_ID,
                "state_version": STATE_VERSION,
                "auto_resume": {
                    "status": "scheduled",
                    "stop_reason": None,
                    "reason": "lease_expired",
                    "attempt": 3,
                    "next_attempt_at": "2026-10-01T08:31:10+00:00",
                },
            },
        ),
        ({"paused_state_version": STATE_VERSION - 1}, None),
        (None, None),
    ],
    ids=["current", "scheduled", "stale", "none"],
)
def test_history_task_info_carries_the_same_view(row, expected) -> None:
    """The web UI loads a task from its history replay's ``task_info``, not
    from the detail endpoint, so that frame carries the same view."""
    from xagent.web.api.websocket import _history_task_info

    _admin_headers()
    task_id = _paused_task(row=row)
    db = _direct_db_session()
    try:
        task = db.get(Task, task_id)
        info = _history_task_info(
            db, task, task_id, auto_recovery=current_auto_recovery_view(db, task)
        )
    finally:
        db.close()

    assert info["data"]["auto_recovery"] == expected


@pytest.mark.parametrize(
    "conversation_storage_version", [2, 1], ids=["event_history", "legacy_history"]
)
@pytest.mark.parametrize(
    ("row", "reason"),
    [({}, "lease_expired"), ({"run_id": "run-earlier"}, None), (None, None)],
    ids=["current", "stale", "none"],
)
def test_reasserted_history_pause_says_why(
    conversation_storage_version: int, row, reason
) -> None:
    """Replayed activity reads as running to the client and drops the pause's
    reason; the pause history reasserts last carries it again."""
    from xagent.web.api.websocket import _load_historical_stream_snapshot_sync

    _admin_headers()
    task_id = _paused_task(
        row=row, conversation_storage_version=conversation_storage_version
    )
    db = _direct_db_session()
    try:
        admin_id = int(db.query(User).filter(User.username == "admin").one().id)
    finally:
        db.close()

    snapshot = _load_historical_stream_snapshot_sync(
        task_id, actor_user_id=admin_id, actor_is_admin=True
    )

    assert snapshot is not None
    paused = [event for event in snapshot.events if event["type"] == "task_paused"]
    assert len(paused) == 1
    assert paused[0].get("interruption_reason") == reason
    assert paused[-1] is snapshot.events[-1]


def _history_auto_resume(task_id: int, admin_id: int) -> object:
    from xagent.web.api.websocket import _load_historical_stream_snapshot_sync

    snapshot = _load_historical_stream_snapshot_sync(
        task_id, actor_user_id=admin_id, actor_is_admin=True
    )
    assert snapshot is not None
    (task_info,) = [e for e in snapshot.events if e.get("event_type") == "task_info"]
    auto_recovery = task_info["data"]["auto_recovery"]
    return auto_recovery and auto_recovery["auto_resume"]["status"]


@pytest.mark.parametrize(
    "conversation_storage_version", [2, 1], ids=["event_history", "legacy_history"]
)
def test_history_replay_cache_follows_the_sweeper(
    conversation_storage_version: int, _response_cache: _JsonCache
) -> None:
    """The legacy replay's cache is keyed on the task row, which the
    sweeper's transitions leave alone, so the view is part of its key; the
    event replay's key holds the whole ``task_info`` already. A repeat read
    with nothing changed is still a hit."""
    stored = _response_cache.sets
    _admin_headers()
    task_id = _paused_task(
        row=SCHEDULED, conversation_storage_version=conversation_storage_version
    )
    db = _direct_db_session()
    try:
        admin_id = int(db.query(User).filter(User.username == "admin").one().id)
    finally:
        db.close()

    assert _history_auto_resume(task_id, admin_id) == "scheduled"
    assert _history_auto_resume(task_id, admin_id) == "scheduled"
    assert len(stored) == 1

    _move_row(task_id, state="dispatched", total_resumes=3)
    assert _history_auto_resume(task_id, admin_id) == "resuming"
    assert len(stored) == 2

    _move_row(task_id, paused_state_version=STATE_VERSION - 1)
    assert _history_auto_resume(task_id, admin_id) is None
