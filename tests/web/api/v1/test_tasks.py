"""Integration tests for /v1/chat/tasks/* endpoints.

Endpoints covered:
  - POST /v1/chat/tasks
  - POST /v1/chat/tasks/{id}/messages
  - GET  /v1/chat/tasks/{id}
  - GET  /v1/chat/tasks/{id}/steps

Tests mock the background-execution kickoff so the suite doesn't need
to spin up an actual AgentService / LLM. The behaviors under test are
HTTP shape + DB rows + which background helper was called with which
arguments -- not the LLM call itself. The steps endpoint exercises
real :class:`TraceEvent` rows inserted directly into the test DB to
drive the mapping.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.services.hot_path_cache import (
    InMemoryTTLCache,
    set_cache_backend_for_testing,
)

from ..conftest import _admin_headers, _direct_db_session, client

# Opt this file into the shared conftest ``_test_db`` fixture; see the
# note in test_agent_api_keys.py for why we use ``usefixtures`` with a
# string name rather than importing the fixture.
pytestmark = pytest.mark.usefixtures("_test_db")


# ===== helpers =====


def _create_agent_with_key() -> tuple[int, str]:
    """Create one agent under the admin user + generate its API key.

    Returns: (agent_id, full_key)
    """
    headers = _admin_headers()
    agent_resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": "v1 tasks test agent",
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


def _bearer(full_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {full_key}"}


# We mock ``TaskTurnOrchestrator._schedule_bg`` (the actual bg coroutine
# spawn) rather than the public ``start_new_turn`` / ``append_turn`` so
# the orchestrator's claim + persist logic still runs and tests can
# verify DB writes (atomic claim flipped status, user messages got
# persisted, etc.). Only the asyncio.create_task / agent execution is
# stubbed.
#
# Scope: file-local. ``autouse=True`` means every test in this module
# gets the mock automatically. GET-only tests (which never call the
# orchestrator) are unaffected; POST tests assert on ``await_count`` /
# ``await_args``. Other test files (e.g. test_steps_mapping.py,
# test_auth.py) are NOT affected because pytest fixture scoping is
# per-module.
#
# Fixture name kept as ``mock_start_task`` to minimize churn across the
# existing test surface; conceptually it now mocks "bg scheduling".
@pytest.fixture(autouse=True)
def mock_start_task():
    # Patch the lease-aware bg scheduler so the orchestrator's atomic
    # claim + transcript persist logic still runs against a real DB;
    # only the asyncio.create_task / agent execution is stubbed.
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ) as mocked:
        yield mocked


# ===== POST /v1/chat/tasks =====


def test_create_task_happy_path(mock_start_task):
    """Returns 202 + task_id, writes hidden SDK Task + input,
    persists first user message, kicks off background, and leaves the
    task readable through the SDK API surface.
    """
    agent_id, full_key = _create_agent_with_key()

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "first user message"},
        },
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["agent_id"] == agent_id
    # POST atomically claims RUNNING before returning 202, so the
    # response body reports the post-claim state, not 'pending'.
    assert body["status"] == "running"
    assert "task_id" in body
    assert "created_at" in body
    task_id = body["task_id"]

    # DB: Task row exists, owned by admin user, source='sdk', input set.
    # POST atomically claims RUNNING before returning 202, so the row
    # is already RUNNING from the moment the response lands.
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        assert task is not None
        assert task.agent_id == agent_id
        assert task.source == "sdk"
        assert task.is_visible is False
        assert task.input == "first user message"
        assert task.status == TaskStatus.RUNNING

        # task_chat_messages: one user-role message written
        from xagent.web.models.chat_message import TaskChatMessage

        msgs = (
            db.query(TaskChatMessage).filter(TaskChatMessage.task_id == task_id).all()
        )
        assert len(msgs) == 1
        assert msgs[0].role == "user"
        assert msgs[0].content == "first user message"
    finally:
        db.close()

    sdk_task = client.get(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))
    assert sdk_task.status_code == 200, sdk_task.text
    assert sdk_task.json()["task_id"] == task_id

    # Background kickoff was called exactly once for this task. The
    # scheduler receives a ``TaskTurnPayload`` carrying both transcript
    # and execution channels.
    assert mock_start_task.call_count == 1
    kwargs = mock_start_task.call_args.kwargs
    assert kwargs["task_id"] == task_id
    assert kwargs["payload"].transcript_message == "first user message"


def test_create_task_missing_authorization_returns_401(mock_start_task):
    """No Authorization header -> 401 invalid_api_key envelope."""
    agent_id, _key = _create_agent_with_key()

    resp = client.post(
        "/v1/chat/tasks",
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "hi"},
        },
    )
    assert resp.status_code == 401
    body = resp.json()
    assert body["error"]["code"] == "invalid_api_key"
    # No DB side effects
    assert mock_start_task.call_count == 0


def test_create_task_agent_id_mismatch_returns_404(mock_start_task):
    """body.agent_id != authed agent.id -> 404 agent_not_found."""
    _agent_id, full_key = _create_agent_with_key()

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": 999999,  # not the bound agent
            "message": {"role": "user", "content": "hi"},
        },
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "agent_not_found"
    assert mock_start_task.call_count == 0


def test_create_task_empty_message_returns_422(mock_start_task):
    """Empty message.content fails Pydantic min_length=1.

    The /v1/* path rewrites the FastAPI default
    ``{"detail": [...]}`` shape into the SDK envelope so clients can
    pin against ``body.error.code == 'invalid_input'`` for 422 just
    like they do for the other error codes.
    """
    agent_id, full_key = _create_agent_with_key()

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": ""},
        },
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "invalid_input"
    assert "detail" not in body  # legacy FastAPI shape must not leak
    assert mock_start_task.call_count == 0


def test_create_task_wrong_role_returns_422(mock_start_task):
    """role != 'user' fails Pydantic Literal check -> 422 with the
    SDK envelope shape, not FastAPI's default ``{"detail": [...]}``."""
    agent_id, full_key = _create_agent_with_key()

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "assistant", "content": "hi"},
        },
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "invalid_input"
    assert "detail" not in body
    assert mock_start_task.call_count == 0


def test_create_task_revoked_key_returns_401(mock_start_task):
    """Revoked key can't create tasks -> 401 invalid_api_key."""
    agent_id, full_key = _create_agent_with_key()
    # Revoke the key via the admin endpoint
    admin = _admin_headers()
    revoke = client.delete(f"/api/agents/{agent_id}/api-key", headers=admin)
    assert revoke.status_code == 200
    assert revoke.json()["revoked"] is True

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "hi"},
        },
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "invalid_api_key"
    assert mock_start_task.call_count == 0


def test_create_task_cross_user_agent_returns_404(mock_start_task):
    """Bob's key cannot target Alice's agent_id -> 404 agent_not_found.

    Defense: the key is bound to agent X. Putting agent_id=Y in the
    body where Y != X always returns 404 regardless of whether Y
    exists, owned by a different user, etc.
    """
    # Admin (alice) creates agent A and a key for it.
    alice_agent_id, _alice_key = _create_agent_with_key()

    # Register bob and create agent B + key, then have bob attempt to
    # POST against alice's agent_id using bob's own key.
    from ..conftest import _register_second_user

    bob_headers = _register_second_user()
    bob_agent = client.post(
        "/api/agents",
        headers=bob_headers,
        json={
            "name": "bob agent",
            "description": "test",
            "instructions": "test",
            "execution_mode": "balanced",
        },
    ).json()
    bob_agent_id = bob_agent["id"]
    bob_key = client.post(
        f"/api/agents/{bob_agent_id}/api-key", headers=bob_headers
    ).json()["full_key"]

    # Bob's key + Alice's agent_id in body -> 404
    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(bob_key),
        json={
            "agent_id": alice_agent_id,
            "message": {"role": "user", "content": "hi"},
        },
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "agent_not_found"
    assert mock_start_task.call_count == 0


# ===== Shared helper for E tests: create a task via POST then return its id =====


def _create_task(full_key: str, agent_id: int, content: str = "hello") -> int:
    """Drive POST /v1/chat/tasks and return the resulting task_id."""
    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={"agent_id": agent_id, "message": {"role": "user", "content": content}},
    )
    assert resp.status_code == 202, resp.text
    return resp.json()["task_id"]


# ===== POST /v1/chat/tasks/{task_id}/messages =====


def _force_task_status(task_id: int, status: TaskStatus) -> None:
    """Bypass the bg coroutine and flip a task to a desired status.

    Tests in this file mock out the bg scheduling so a freshly-created
    task stays at PENDING forever. The orchestrator's ``append_turn``
    only accepts terminal statuses (COMPLETED / FAILED) -- which is
    correct production behavior -- so tests that exercise the
    append-happy path need to push the task to COMPLETED first.
    """
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = status
        db.commit()
    finally:
        db.close()


def test_append_message_happy_path(mock_start_task):
    """Returns 202 + accepted_at, persists new user message, kicks off bg."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id, content="first turn")
    # Mark the previous turn as COMPLETED so append_turn's atomic claim
    # passes (PENDING is rejected as busy because it means the create's
    # bg run hasn't finished yet).
    _force_task_status(task_id, TaskStatus.COMPLETED)
    mock_start_task.reset_mock()  # discard the create-task call so we count just the append

    resp = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "second turn"},
        },
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["task_id"] == task_id
    assert body["agent_id"] == agent_id
    # The atomic claim in the endpoint already flipped the row to
    # RUNNING; the response must mirror that so back-to-back POST+GET
    # don't show contradictory statuses to SDK clients.
    assert body["status"] == "running"
    assert "accepted_at" in body

    # Two user messages now exist for this task; task.input is the latest
    from xagent.web.models.chat_message import TaskChatMessage

    db = _direct_db_session()
    try:
        msgs = (
            db.query(TaskChatMessage)
            .filter(TaskChatMessage.task_id == task_id)
            .order_by(TaskChatMessage.id)
            .all()
        )
        assert len(msgs) == 2
        assert [m.content for m in msgs] == ["first turn", "second turn"]
        task = db.query(Task).filter(Task.id == task_id).first()
        assert task is not None
        assert task.input == "second turn"
    finally:
        db.close()

    assert mock_start_task.call_count == 1


def test_append_message_to_running_task_returns_409(mock_start_task):
    """Appending to a RUNNING task is rejected as task_busy."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    mock_start_task.reset_mock()

    # Flip status to RUNNING directly so we don't have to actually run
    # the agent service in tests.
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = TaskStatus.RUNNING
        db.commit()
    finally:
        db.close()

    resp = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json={"agent_id": agent_id, "message": {"role": "user", "content": "hello"}},
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "task_busy"
    # No new background kickoff happened
    assert mock_start_task.call_count == 0


def test_append_message_claims_slot_atomically(mock_start_task):
    """Successful append flips task.status to RUNNING in the same
    transaction as the input write, so a concurrent POST can't pass
    the busy check and both kick off background tasks.

    We verify the post-state directly: after one successful append,
    task.status == RUNNING and a second POST to the same task gets
    409 even though the bg coroutine hasn't run yet (mocked out).
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id, content="t1")
    # Push to COMPLETED so the first append is allowed (PENDING is busy).
    _force_task_status(task_id, TaskStatus.COMPLETED)
    mock_start_task.reset_mock()

    # First append succeeds and atomically claims the slot.
    r1 = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json={"agent_id": agent_id, "message": {"role": "user", "content": "t2"}},
    )
    assert r1.status_code == 202

    # task.status was flipped to RUNNING inside the endpoint, even
    # though the (mocked) bg coroutine never ran. This is the
    # mechanism that defeats the TOCTOU race.
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        assert task is not None
        assert task.status == TaskStatus.RUNNING
    finally:
        db.close()

    # Second append (the would-be losing concurrent request) hits the
    # claim filter and 409s.
    r2 = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json={"agent_id": agent_id, "message": {"role": "user", "content": "t3"}},
    )
    assert r2.status_code == 409
    assert r2.json()["error"]["code"] == "task_busy"
    # Only one bg kickoff total (from the winning first append).
    assert mock_start_task.call_count == 1


def test_create_then_append_race_returns_409(mock_start_task):
    """Regression: create-then-immediate-append must 409, not race.

    The old append_turn used ``status != RUNNING`` which let PENDING
    slip through. A client could POST /v1/chat/tasks, get a PENDING
    task back, and immediately POST /messages; both the create's bg
    coroutine and the append's bg coroutine would then race to run
    the same task. The orchestrator's terminal-state-only filter
    closes this: PENDING (= "first turn scheduled but not started")
    is treated as busy.
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id, content="first")
    # NOT calling _force_task_status here — task stays in PENDING
    # exactly as it would right after the SDK's create response is
    # returned but before the bg coroutine ran.
    mock_start_task.reset_mock()

    resp = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "second"},
        },
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "task_busy"
    # No second bg kickoff should have happened
    assert mock_start_task.call_count == 0


def test_append_message_bg_inflight_does_not_corrupt_task_state(mock_start_task):
    """Regression: when ``append_turn`` refuses because a previous bg
    coroutine is still in flight, the DB row must NOT have been mutated.

    The bug scenario: previous turn flipped status to COMPLETED but the
    bg coroutine is still in tail cleanup (``_sync_sdk_columns`` hasn't
    returned), so ``background_task_manager.running_tasks[task_id]`` is
    still a not-done asyncio.Task. A new append should be refused as
    busy, and the DB row should still report COMPLETED + the original
    input — not RUNNING + new input.

    If the inflight check happened *after* the atomic UPDATE (the old
    ordering), the row would be RUNNING + new_input even on 409
    rejection. The old runner's _sync_sdk_columns would then see
    RUNNING and flip the row to FAILED with a placeholder error
    message, corrupting an otherwise successful past turn.
    """
    import asyncio

    from xagent.web.api.websocket import background_task_manager

    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id, content="first turn")
    _force_task_status(task_id, TaskStatus.COMPLETED)
    mock_start_task.reset_mock()

    # Snapshot DB state right before the append attempt.
    db_before = _direct_db_session()
    try:
        task_before = db_before.query(Task).filter(Task.id == task_id).first()
        original_status = task_before.status
        original_input = task_before.input
    finally:
        db_before.close()

    # Plant a not-done asyncio.Task in the bg manager registry to
    # simulate "previous runner is still cleaning up".
    loop = asyncio.new_event_loop()
    try:

        async def _never_done() -> None:
            await asyncio.sleep(3600)

        fake_inflight = loop.create_task(_never_done())
        background_task_manager.running_tasks[task_id] = fake_inflight

        try:
            resp = client.post(
                f"/v1/chat/tasks/{task_id}/messages",
                headers=_bearer(full_key),
                json={
                    "agent_id": agent_id,
                    "message": {"role": "user", "content": "second"},
                },
            )
            assert resp.status_code == 409
            assert resp.json()["error"]["code"] == "task_busy"
            assert mock_start_task.call_count == 0

            # The critical assertion: DB row was NOT mutated by the
            # refused append. status stays terminal, input unchanged.
            db_after = _direct_db_session()
            try:
                task_after = db_after.query(Task).filter(Task.id == task_id).first()
                assert task_after.status == original_status, (
                    f"task.status was corrupted on refused append: "
                    f"{original_status} -> {task_after.status}"
                )
                assert task_after.input == original_input, (
                    f"task.input was overwritten on refused append: "
                    f"{original_input!r} -> {task_after.input!r}"
                )
            finally:
                db_after.close()
        finally:
            # Clean up the fake registry entry so other tests aren't
            # affected.
            background_task_manager.running_tasks.pop(task_id, None)
            fake_inflight.cancel()
    finally:
        loop.close()


def test_append_message_to_missing_task_returns_404(mock_start_task):
    """Appending to a task that doesn't exist -> 404 task_not_found."""
    agent_id, full_key = _create_agent_with_key()
    resp = client.post(
        "/v1/chat/tasks/9999999/messages",
        headers=_bearer(full_key),
        json={"agent_id": agent_id, "message": {"role": "user", "content": "hi"}},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "task_not_found"
    assert mock_start_task.call_count == 0


def test_append_message_to_other_agents_task_returns_404(mock_start_task):
    """Bob can't append to Alice's task even if he knows the id."""
    alice_agent_id, alice_key = _create_agent_with_key()
    alice_task_id = _create_task(alice_key, alice_agent_id)
    mock_start_task.reset_mock()

    from ..conftest import _register_second_user

    bob_headers = _register_second_user()
    bob_agent = client.post(
        "/api/agents",
        headers=bob_headers,
        json={
            "name": "bob agent",
            "description": "test",
            "instructions": "test",
            "execution_mode": "balanced",
        },
    ).json()
    bob_agent_id = bob_agent["id"]
    bob_key = client.post(
        f"/api/agents/{bob_agent_id}/api-key", headers=bob_headers
    ).json()["full_key"]

    resp = client.post(
        f"/v1/chat/tasks/{alice_task_id}/messages",
        headers=_bearer(bob_key),
        json={"agent_id": bob_agent_id, "message": {"role": "user", "content": "hi"}},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "task_not_found"
    assert mock_start_task.call_count == 0


def test_append_message_body_agent_id_mismatch_returns_404(mock_start_task):
    """body.agent_id != authed agent.id -> 404 agent_not_found.

    Distinct from cross-agent task ownership (which is task_not_found) --
    here the task IS the caller's, but the body claims a different
    agent_id; consistent with POST /v1/chat/tasks behavior.
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    mock_start_task.reset_mock()

    resp = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json={"agent_id": 999999, "message": {"role": "user", "content": "hi"}},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "agent_not_found"
    assert mock_start_task.call_count == 0


# ===== GET /v1/chat/tasks/{task_id} =====


def test_get_task_running_right_after_create(mock_start_task):
    """A fresh SDK task is visible as ``status='running'`` immediately
    after POST returns: the atomic claim commits the status flip before
    202, so an immediate GET sees RUNNING + input set + output/error
    null + completed_at null.
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id, content="seed input")

    resp = client.get(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["task_id"] == task_id
    assert body["agent_id"] == agent_id
    assert body["status"] == "running"
    assert body["input"] == "seed input"
    assert body["output"] is None
    assert body["error"] is None
    assert "created_at" in body
    assert body["completed_at"] is None


def test_get_task_completed_returns_output(mock_start_task):
    """Completed task: output populated, completed_at set."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)

    # Flip status + write output to simulate completed background turn
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = TaskStatus.COMPLETED
        task.output = "final answer"
        db.commit()
    finally:
        db.close()

    resp = client.get(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["output"] == "final answer"
    assert body["completed_at"] is not None


def test_get_task_failed_returns_error(mock_start_task):
    """Failed task: error populated, completed_at set."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = TaskStatus.FAILED
        task.error_message = "agent crashed"
        db.commit()
    finally:
        db.close()

    resp = client.get(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failed"
    assert body["error"] == "agent crashed"
    assert body["output"] is None
    assert body["completed_at"] is not None


def test_get_missing_task_returns_404(mock_start_task):
    _agent_id, full_key = _create_agent_with_key()
    resp = client.get("/v1/chat/tasks/9999999", headers=_bearer(full_key))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "task_not_found"


def test_get_other_agents_task_returns_404(mock_start_task):
    """Cross-agent task access -> 404 (not leaking existence)."""
    alice_agent_id, alice_key = _create_agent_with_key()
    alice_task_id = _create_task(alice_key, alice_agent_id)

    from ..conftest import _register_second_user

    bob_headers = _register_second_user()
    bob_agent = client.post(
        "/api/agents",
        headers=bob_headers,
        json={
            "name": "bob agent",
            "description": "test",
            "instructions": "test",
            "execution_mode": "balanced",
        },
    ).json()
    bob_agent_id = bob_agent["id"]
    bob_key = client.post(
        f"/api/agents/{bob_agent_id}/api-key", headers=bob_headers
    ).json()["full_key"]

    resp = client.get(f"/v1/chat/tasks/{alice_task_id}", headers=_bearer(bob_key))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "task_not_found"


# ===== F: GET /v1/chat/tasks/{task_id}/steps =====


def _insert_trace_event(
    *,
    task_id: int,
    event_type: str,
    event_id: str,
    timestamp: datetime,
    data: dict,
    step_id: str | None = None,
    build_id: str | None = None,
) -> None:
    """Insert one TraceEvent row directly via the test DB.

    Bypasses the production trace handler (which runs through asyncio
    + thread pool) so tests can assert on the GET /steps surface
    without spinning up the agent runtime.
    """
    db = _direct_db_session()
    try:
        ev = TraceEvent(
            task_id=task_id,
            event_id=event_id,
            event_type=event_type,
            timestamp=timestamp,
            step_id=step_id,
            build_id=build_id,
            data=data,
        )
        db.add(ev)
        db.commit()
    finally:
        db.close()


def test_get_steps_returns_mapped_steps_in_order(mock_start_task):
    """Insert react_action + tool_execution + ai_message + filtered
    llm_call events, GET /steps, assert 3 public steps in started_at
    order with correct types and statuses.
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)

    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    # Public step 1: thinking phase=action (react_action_start/end)
    _insert_trace_event(
        task_id=task_id,
        event_type="react_action_start",
        event_id="evt-1",
        timestamp=base.replace(second=1),
        step_id="step-A",
        data={},
    )
    _insert_trace_event(
        task_id=task_id,
        event_type="react_action_end",
        event_id="evt-2",
        timestamp=base.replace(second=2),
        step_id="step-A",
        data={},
    )

    # Filtered: llm_call_start / end -- must not appear
    _insert_trace_event(
        task_id=task_id,
        event_type="llm_call_start",
        event_id="evt-3",
        timestamp=base.replace(second=3),
        step_id="step-A",
        data={},
    )
    _insert_trace_event(
        task_id=task_id,
        event_type="llm_call_end",
        event_id="evt-4",
        timestamp=base.replace(second=4),
        step_id="step-A",
        data={},
    )

    # Public step 2: tool_call (execute_python)
    _insert_trace_event(
        task_id=task_id,
        event_type="tool_execution_start",
        event_id="evt-5",
        timestamp=base.replace(second=5),
        step_id="step-A",
        data={
            "tool_name": "execute_python",
            "tool_args": {"code": "print(1)"},
            "tool_execution_id": "tx-1",
        },
    )
    _insert_trace_event(
        task_id=task_id,
        event_type="tool_execution_end",
        event_id="evt-6",
        timestamp=base.replace(second=6),
        step_id="step-A",
        data={
            "tool_name": "execute_python",
            "tool_args": {"code": "print(1)"},
            "tool_execution_id": "tx-1",
            "result": {"output": "1\n"},
            "success": True,
        },
    )

    # Public step 3: message role=assistant (ai_message)
    _insert_trace_event(
        task_id=task_id,
        event_type="ai_message",
        event_id="evt-7",
        timestamp=base.replace(second=7),
        data={"content": "Here's the result"},
    )

    resp = client.get(f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["task_id"] == task_id
    assert body["agent_id"] == agent_id
    steps = body["steps"]
    assert len(steps) == 3

    # Assert ordering and types
    assert steps[0]["type"] == "thinking"
    assert steps[0]["data"]["phase"] == "action"
    assert steps[0]["status"] == "completed"

    assert steps[1]["type"] == "tool_call"
    assert steps[1]["data"]["name"] == "execute_python"
    assert steps[1]["data"]["args"] == {"code": "print(1)"}
    assert steps[1]["data"]["result"] == {"output": "1\n"}

    assert steps[2]["type"] == "message"
    assert steps[2]["data"] == {"role": "assistant", "content": "Here's the result"}


def test_get_steps_task_not_found_returns_404(mock_start_task):
    """Non-existent task_id -> 404 task_not_found."""
    _agent_id, full_key = _create_agent_with_key()
    resp = client.get("/v1/chat/tasks/9999999/steps", headers=_bearer(full_key))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "task_not_found"


def test_get_steps_other_agents_task_returns_404(mock_start_task):
    """Cross-agent steps access -> 404 (not leaking existence)."""
    alice_agent_id, alice_key = _create_agent_with_key()
    alice_task_id = _create_task(alice_key, alice_agent_id)

    from ..conftest import _register_second_user

    bob_headers = _register_second_user()
    bob_agent = client.post(
        "/api/agents",
        headers=bob_headers,
        json={
            "name": "bob agent steps",
            "description": "test",
            "instructions": "test",
            "execution_mode": "balanced",
        },
    ).json()
    bob_agent_id = bob_agent["id"]
    bob_key = client.post(
        f"/api/agents/{bob_agent_id}/api-key", headers=bob_headers
    ).json()["full_key"]

    resp = client.get(f"/v1/chat/tasks/{alice_task_id}/steps", headers=_bearer(bob_key))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "task_not_found"


def test_get_steps_empty_task_returns_empty_array(mock_start_task):
    """Task with no trace events yet -> 200 + empty steps array."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)

    resp = client.get(f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key))
    assert resp.status_code == 200
    body = resp.json()
    assert body["task_id"] == task_id
    assert body["agent_id"] == agent_id
    assert body["steps"] == []


def test_get_steps_ignores_worker_build_trace_events(mock_start_task):
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)

    _insert_trace_event(
        task_id=task_id,
        event_type="tool_execution_start",
        event_id="worker-trace-1",
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        step_id="worker-step",
        build_id="agent_123_abcd1234",
        data={
            "tool_name": "worker_tool",
            "tool_execution_id": "worker-call-1",
            "worker_task_id": "agent_123_abcd1234",
        },
    )

    resp = client.get(f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key))

    assert resp.status_code == 200, resp.text
    assert resp.json()["steps"] == []


def test_get_steps_cache_reuses_mapping_until_trace_event_changes(mock_start_task):
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    _insert_trace_event(
        task_id=task_id,
        event_type="ai_message",
        event_id="evt-cache-1",
        timestamp=base,
        data={"content": "cached"},
    )

    set_cache_backend_for_testing(InMemoryTTLCache())
    try:
        from xagent.web.api.v1 import _step_mapping

        with patch(
            "xagent.web.api.v1.tasks.map_trace_events_to_public_steps",
            wraps=_step_mapping.map_trace_events_to_public_steps,
        ) as mapper:
            first = client.get(
                f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key)
            )
            second = client.get(
                f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key)
            )

            assert first.status_code == 200, first.text
            assert second.status_code == 200, second.text
            assert first.json() == second.json()
            assert mapper.call_count == 1

            _insert_trace_event(
                task_id=task_id,
                event_type="ai_message",
                event_id="evt-cache-2",
                timestamp=base.replace(second=1),
                data={"content": "new"},
            )
            third = client.get(
                f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key)
            )

            assert third.status_code == 200, third.text
            assert mapper.call_count == 2
            assert len(third.json()["steps"]) == 2
    finally:
        set_cache_backend_for_testing(None)


# ===== source filtering: SDK API surface only sees source="sdk" tasks =====


def _insert_internal_task(agent_id: int) -> int:
    """Manually INSERT a task under ``agent_id`` with source != "sdk".

    Tests for the SDK source filter need a task that lives under the
    same agent but was created by the Web UI / internal paths, not
    via POST /v1/chat/tasks. Since the SDK create endpoint always
    writes source="sdk", we bypass it and craft the row directly.
    """
    from xagent.web.models.agent import Agent
    from xagent.web.models.task import Task, TaskStatus
    from xagent.web.models.user import User

    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        assert agent is not None
        # Reuse the agent's owner so user_id stays consistent.
        user = db.query(User).filter(User.id == agent.user_id).first()
        assert user is not None
        task = Task(
            user_id=user.id,
            title="internal task",
            description="created via web ui, not sdk",
            status=TaskStatus.COMPLETED,
            agent_id=agent.id,
            input="internal user message",
            source="internal",
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        return int(task.id)
    finally:
        db.close()


def test_get_task_returns_404_for_non_sdk_source(mock_start_task):
    """A task created by the Web UI / internal path (source != "sdk")
    under the same agent must NOT be readable through GET /v1/chat/tasks/{id}.

    Without the source filter, an SDK API key could enumerate / read
    the user's own Web UI conversations whenever they happen to live
    under the same agent.
    """
    agent_id, full_key = _create_agent_with_key()
    internal_task_id = _insert_internal_task(agent_id)
    mock_start_task.reset_mock()

    resp = client.get(f"/v1/chat/tasks/{internal_task_id}", headers=_bearer(full_key))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "task_not_found"


def test_append_message_returns_404_for_non_sdk_source(mock_start_task):
    """POST /v1/chat/tasks/{id}/messages on a non-SDK task must 404
    with task_not_found — the SDK key shouldn't be able to mutate
    Web UI conversations even if it knows the task id."""
    agent_id, full_key = _create_agent_with_key()
    internal_task_id = _insert_internal_task(agent_id)
    mock_start_task.reset_mock()

    resp = client.post(
        f"/v1/chat/tasks/{internal_task_id}/messages",
        headers=_bearer(full_key),
        json={"agent_id": agent_id, "message": {"role": "user", "content": "hi"}},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "task_not_found"
    assert mock_start_task.call_count == 0


def test_get_steps_returns_404_for_non_sdk_source(mock_start_task):
    """GET /v1/chat/tasks/{id}/steps on a non-SDK task must 404 so
    the SDK can't enumerate Web UI step traces under the same agent."""
    agent_id, full_key = _create_agent_with_key()
    internal_task_id = _insert_internal_task(agent_id)

    resp = client.get(
        f"/v1/chat/tasks/{internal_task_id}/steps", headers=_bearer(full_key)
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "task_not_found"


# ===== Latest-turn snapshot invariant on SDK append =====


def test_append_message_clears_stale_output_for_sdk_caller(mock_start_task):
    """An SDK append on a previously completed task immediately clears
    the stored ``output`` and ``error_message`` so a GET right after
    the append sees a clean latest-turn snapshot. Without this clearing,
    the response would mix the new turn's status / input with the
    previous turn's output — an internally contradictory snapshot for
    SDK consumers polling the task.
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id, content="first")

    # Plant a completed-state row with prior output / error_message
    # populated, as if the first turn had finished successfully.
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = TaskStatus.COMPLETED
        task.output = "first answer"
        task.error_message = "stale error from prior failure"
        db.commit()
    finally:
        db.close()

    mock_start_task.reset_mock()

    resp = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json={"agent_id": agent_id, "message": {"role": "user", "content": "second"}},
    )
    assert resp.status_code == 202, resp.text
    assert mock_start_task.call_count == 1

    # After the response returns, an immediate GET must see:
    #   - status = running (atomic transition committed)
    #   - input = the new turn's message
    #   - output = NULL (stale prior-turn output cleared)
    #   - error_message = NULL (stale prior error cleared)
    db = _direct_db_session()
    try:
        task_after = db.query(Task).filter(Task.id == task_id).first()
        assert task_after.status == TaskStatus.RUNNING
        assert task_after.input == "second"
        assert task_after.output is None
        assert task_after.error_message is None
    finally:
        db.close()
