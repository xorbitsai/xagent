"""The v1 API's answer for a task the retention policy expired (#2565).

Every ``/v1`` route addressed by a task id answers ``410 task_expired`` for a
task retention expired, but only to a key that could have seen the task while
it existed -- the tombstone is held to ``resolve_sdk_task``'s own predicate
(``source == "sdk"`` plus the key's agent or workforce). Every other caller
keeps the ``404 task_not_found`` a missing task gets, so the 410 discloses
nothing a 200 would not have.

Also covers the trace-only half: ``/steps`` on a live task whose trace was
expired carries ``steps_expired`` / ``steps_expired_at``, and the steps cache
cannot serve an entry written before the purge without that flag.

Tombstones are inserted directly for most cases, which pins the predicate
without depending on the purge's eligibility rules; one test per key type
runs the real purge end to end.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.expired_task import ExpiredTaskTombstone
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.models.workforce import WorkforceRun
from xagent.web.services.hot_path_cache import (
    InMemoryTTLCache,
    set_cache_backend_for_testing,
)
from xagent.web.services.task_retention_purge import (
    RetentionPurgeAction,
    purge_task,
)

from ..conftest import (
    _admin_headers,
    _direct_db_session,
    _register_second_user,
    client,
)
from .test_workforces import (
    _create_active_workforce,
    _create_workforce_key,
)

pytestmark = pytest.mark.usefixtures("_test_db")

EXPIRED_AT = datetime(2026, 9, 1, 3, 0, 0, tzinfo=UTC)
#: An id no test ever creates a live task under.
UNUSED_TASK_ID = 9_000_001


@pytest.fixture(autouse=True)
def mock_schedule_bg():
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ) as mocked:
        yield mocked


def _bearer(full_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {full_key}"}


def _create_agent_with_key(headers: dict[str, str] | None = None) -> tuple[int, str]:
    headers = headers or _admin_headers()
    agent_resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": "v1 expired test agent",
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


def _create_task(full_key: str, agent_id: int) -> int:
    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={"agent_id": agent_id, "message": {"role": "user", "content": "hi"}},
    )
    assert resp.status_code == 202, resp.text
    return resp.json()["task_id"]


def _admin_user_id() -> int:
    db = _direct_db_session()
    try:
        return int(db.query(User.id).filter(User.username == "admin").scalar())
    finally:
        db.close()


def _insert_tombstone(
    task_id: int,
    *,
    agent_id: int | None = None,
    workforce_id: int | None = None,
    source: str = "sdk",
) -> None:
    db = _direct_db_session()
    try:
        db.add(
            ExpiredTaskTombstone(
                task_id=task_id,
                user_id=_admin_user_id(),
                agent_id=agent_id,
                workforce_id=workforce_id,
                source=source,
                is_visible=False,
                is_channel_plumbing=False,
                task_created_at=EXPIRED_AT - timedelta(days=400),
                expired_at=EXPIRED_AT,
            )
        )
        db.commit()
    finally:
        db.close()


def _purge_conversation(task_id: int) -> datetime:
    """Run the real retention purge on ``task_id``; return its ``now``.

    Makes the task eligible first, the same way the trace-expiry steps test
    in ``test_tasks.py`` does: terminal, no lease, no owed command, and a
    transcript as old as its activity anchor.
    """
    base = datetime(2025, 1, 1, tzinfo=UTC)
    now = base + timedelta(days=500)
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.status = TaskStatus.COMPLETED
        task.last_activity_at = base
        task.lease_expires_at = None
        db.query(TaskChatMessage).filter(TaskChatMessage.task_id == task_id).update(
            {TaskChatMessage.created_at: base}, synchronize_session=False
        )
        db.query(TaskExecutionCommand).filter(
            TaskExecutionCommand.task_id == task_id
        ).delete(synchronize_session=False)
        db.commit()
        assert (
            purge_task(db, task_id, now=now, conversation_days=365, trace_days=90)
            is RetentionPurgeAction.PURGED_CONVERSATION
        )
    finally:
        db.close()
    return now


def _as_utc(value: str) -> datetime:
    """SQLite drops the offset on the way back; PostgreSQL keeps it."""
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _call(route: str, task_id: int, full_key: str, owner: dict):
    """Hit one task-id route with a body valid for ``owner``'s key type."""
    headers = _bearer(full_key)
    message = {"role": "user", "content": "again"}
    if route == "get":
        return client.get(f"/v1/chat/tasks/{task_id}", headers=headers)
    if route == "steps":
        return client.get(f"/v1/chat/tasks/{task_id}/steps", headers=headers)
    if route == "events":
        return client.get(f"/v1/chat/tasks/{task_id}/events", headers=headers)
    if route == "append":
        return client.post(
            f"/v1/chat/tasks/{task_id}/messages",
            headers=headers,
            json={**owner, "message": message},
        )
    if route == "reply":
        return client.post(
            f"/v1/chat/tasks/{task_id}/reply",
            headers=headers,
            json={**owner, "command_id": "cmd-expired", "message": message},
        )
    if route == "upload":
        return client.post(
            f"/v1/chat/files?task_id={task_id}",
            headers=headers,
            files={"files": ("note.txt", b"hello", "text/plain")},
        )
    raise AssertionError(route)


TASK_ROUTES = ["get", "steps", "events", "append", "reply", "upload"]


def _assert_expired(resp, task_id: int, expired_at: datetime = EXPIRED_AT) -> None:
    assert resp.status_code == 410, resp.text
    error = resp.json()["error"]
    assert error["code"] == "task_expired"
    assert error["details"]["task_id"] == task_id
    assert _as_utc(error["details"]["expired_at"]) == expired_at


def _assert_not_found(resp) -> None:
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "task_not_found"
    assert "details" not in resp.json()["error"]


# ===== REST: agent-bound key =====


@pytest.mark.parametrize("route", TASK_ROUTES)
def test_owned_expired_task_returns_410_on_every_task_route(route):
    agent_id, full_key = _create_agent_with_key()
    _insert_tombstone(UNUSED_TASK_ID, agent_id=agent_id)

    resp = _call(route, UNUSED_TASK_ID, full_key, {"agent_id": agent_id})

    _assert_expired(resp, UNUSED_TASK_ID)


def test_the_real_purge_turns_a_404_candidate_into_a_410():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)

    now = _purge_conversation(task_id)

    _assert_expired(
        client.get(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key)),
        task_id,
        expired_at=now,
    )


@pytest.mark.parametrize("route", TASK_ROUTES)
def test_a_tombstone_of_another_agent_stays_404(route):
    """Non-disclosure: the id's existence is not revealed to another key."""
    owner_agent_id, _owner_key = _create_agent_with_key()
    other_agent_id, other_key = _create_agent_with_key(_register_second_user())
    _insert_tombstone(UNUSED_TASK_ID, agent_id=owner_agent_id)

    resp = _call(route, UNUSED_TASK_ID, other_key, {"agent_id": other_agent_id})

    _assert_not_found(resp)


@pytest.mark.parametrize("route", TASK_ROUTES)
def test_a_non_sdk_tombstone_stays_404(route):
    """The live lookup never served a Web UI task to an SDK key; nor does this."""
    agent_id, full_key = _create_agent_with_key()
    _insert_tombstone(UNUSED_TASK_ID, agent_id=agent_id, source="web")

    resp = _call(route, UNUSED_TASK_ID, full_key, {"agent_id": agent_id})

    _assert_not_found(resp)


def test_a_task_deleted_by_its_owner_stays_404():
    """User-initiated deletion writes no tombstone, so it keeps answering 404."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    db = _direct_db_session()
    try:
        db.query(Task).filter(Task.id == task_id).delete()
        db.commit()
    finally:
        db.close()

    _assert_not_found(
        client.get(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))
    )


def test_a_live_task_wins_over_a_tombstone_with_the_same_id():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _insert_tombstone(task_id, agent_id=agent_id)

    resp = client.get(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))

    assert resp.status_code == 200, resp.text
    assert resp.json()["task_id"] == task_id


def test_a_live_task_the_key_cannot_see_is_404_even_over_an_owned_tombstone():
    """The live row holds the id, so the tombstone never speaks for it."""
    owner_agent_id, owner_key = _create_agent_with_key()
    task_id = _create_task(owner_key, owner_agent_id)
    other_agent_id, other_key = _create_agent_with_key(_register_second_user())
    _insert_tombstone(task_id, agent_id=other_agent_id)

    _assert_not_found(
        client.get(f"/v1/chat/tasks/{task_id}", headers=_bearer(other_key))
    )


# ===== REST: workforce-bound key =====


@pytest.mark.parametrize("route", TASK_ROUTES)
def test_workforce_key_gets_410_for_its_own_expired_task(route):
    headers = _admin_headers()
    workforce_id = _create_active_workforce(headers, name="Expired WF")
    full_key = _create_workforce_key(headers, workforce_id)
    _insert_tombstone(UNUSED_TASK_ID, workforce_id=workforce_id)

    resp = _call(route, UNUSED_TASK_ID, full_key, {})

    _assert_expired(resp, UNUSED_TASK_ID)


def test_workforce_key_gets_410_after_the_real_purge_clears_the_run_pointer():
    """The purge SETs ``WorkforceRun.task_id`` NULL, which is what the live
    lookup joins through; the tombstone's own ``workforce_id`` is what still
    proves the binding."""
    headers = _admin_headers()
    workforce_id = _create_active_workforce(headers, name="Purged WF")
    full_key = _create_workforce_key(headers, workforce_id)
    run = client.post(
        f"/v1/workforces/{workforce_id}/runs",
        headers=_bearer(full_key),
        json={"message": {"role": "user", "content": "go"}},
    )
    assert run.status_code == 202, run.text
    task_id = run.json()["task_id"]

    now = _purge_conversation(task_id)

    db = _direct_db_session()
    try:
        run_row = db.query(WorkforceRun).filter(
            WorkforceRun.id == run.json()["workforce_run_id"]
        )
        assert run_row.one().task_id is None
    finally:
        db.close()
    _assert_expired(
        client.get(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key)),
        task_id,
        expired_at=now,
    )


def test_a_tombstone_of_another_workforce_stays_404():
    headers = _admin_headers()
    wf_a = _create_active_workforce(headers, name="Expired WF A")
    wf_b = _create_active_workforce(headers, name="Expired WF B")
    key_b = _create_workforce_key(headers, wf_b)
    _insert_tombstone(UNUSED_TASK_ID, workforce_id=wf_a)

    _assert_not_found(
        client.get(f"/v1/chat/tasks/{UNUSED_TASK_ID}", headers=_bearer(key_b))
    )


# ===== /steps: trace-only expiry =====


def _insert_trace_event(task_id: int, event_id: str, content: str) -> None:
    db = _direct_db_session()
    try:
        db.add(
            TraceEvent(
                task_id=task_id,
                event_id=event_id,
                event_type="ai_message",
                timestamp=datetime(2026, 9, 20, 12, 0, 0, tzinfo=UTC),
                data={"content": content},
            )
        )
        db.commit()
    finally:
        db.close()


def _set_traces_expired_at(task_id: int, value: datetime | None) -> None:
    db = _direct_db_session()
    try:
        db.query(Task).filter(Task.id == task_id).update(
            {Task.traces_expired_at: value}, synchronize_session=False
        )
        db.commit()
    finally:
        db.close()


def test_steps_on_an_untouched_task_report_not_expired():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)

    body = client.get(
        f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key)
    ).json()

    assert body["steps_expired"] is False
    assert body["steps_expired_at"] is None


def test_steps_expired_with_no_steps_left():
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _set_traces_expired_at(task_id, EXPIRED_AT)

    resp = client.get(f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key))

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["steps"] == []
    assert body["steps_expired"] is True
    assert _as_utc(body["steps_expired_at"]) == EXPIRED_AT


def test_steps_expired_coexists_with_steps_from_a_later_turn():
    """A trace-expired task can take new turns; the flag then means "may be
    incomplete", not "empty"."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _set_traces_expired_at(task_id, EXPIRED_AT)
    _insert_trace_event(task_id, "evt-after-expiry", "later turn")

    body = client.get(
        f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key)
    ).json()

    assert [step["data"]["content"] for step in body["steps"]] == ["later turn"]
    assert body["steps_expired"] is True
    assert _as_utc(body["steps_expired_at"]) == EXPIRED_AT


def test_steps_cache_entry_from_before_the_purge_is_not_served():
    """``max_event_id`` alone would call this entry fresh: the purge moved
    only ``traces_expired_at``. The cache must compare both."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _insert_trace_event(task_id, "evt-kept", "still here")

    set_cache_backend_for_testing(InMemoryTTLCache())
    try:
        warm = client.get(
            f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key)
        ).json()
        assert warm["steps_expired"] is False

        _set_traces_expired_at(task_id, EXPIRED_AT)
        after = client.get(
            f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key)
        ).json()
        assert after["steps_expired"] is True
        assert _as_utc(after["steps_expired_at"]) == EXPIRED_AT
        assert after["steps"] == warm["steps"]

        # A second, later purge restamps the column; the entry cached under
        # the first stamp is stale again.
        later = EXPIRED_AT + timedelta(days=30)
        _set_traces_expired_at(task_id, later)
        restamped = client.get(
            f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key)
        ).json()
        assert _as_utc(restamped["steps_expired_at"]) == later
    finally:
        set_cache_backend_for_testing(None)


def test_steps_cache_entry_without_the_expiry_key_still_serves_untouched_tasks():
    """An entry written before this field existed has no
    ``traces_expired_at`` key; for a task retention never touched that is
    still a hit, so a deploy does not cold-start every steps cache entry."""
    from xagent.web.api.v1 import tasks as v1_tasks
    from xagent.web.services.hot_path_cache import cache_get, cache_set, task_steps_key

    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _insert_trace_event(task_id, "evt-legacy", "legacy")

    set_cache_backend_for_testing(InMemoryTTLCache())
    try:
        client.get(f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key))
        entry = dict(cache_get(task_steps_key(task_id)))
        entry.pop("traces_expired_at")
        entry["response"] = {
            k: v
            for k, v in entry["response"].items()
            if k not in ("steps_expired", "steps_expired_at")
        }
        cache_set(task_steps_key(task_id), entry, ttl_seconds=60)

        with patch.object(
            v1_tasks,
            "map_trace_events_to_public_steps",
            side_effect=AssertionError("expected a cache hit"),
        ):
            body = client.get(
                f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key)
            ).json()
        assert body["steps_expired"] is False
        assert body["steps_expired_at"] is None
    finally:
        set_cache_backend_for_testing(None)
