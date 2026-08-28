"""Integration tests for ``POST /v1/chat/tasks/{task_id}/reply``.

Mirrors the structure of ``tests/web/api/test_a2a_api.py``'s
input-required resume tests, since ``task_reply.py`` copies that
resume's mechanics. Where a2a's test patches
``xagent.web.api.a2a._schedule_waiting_a2a_resume`` to avoid spinning
up a real background execution, these tests patch the analogous
``xagent.web.api.v1.task_reply._schedule_waiting_reply_resume``.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.core.agent.checkpoint import (
    CheckpointAccessRefusedError,
    CheckpointCorruptError,
    CheckpointReadError,
    CheckpointUnavailableError,
)
from xagent.web.api.v1 import task_reply as task_reply_module
from xagent.web.models.agent import Agent
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.schemas.v1 import ReplyRequest
from xagent.web.services.task_execution_controller import TaskControlState
from xagent.web.services.task_lease_service import TaskLease, current_task_lease

from ..conftest import _admin_headers, _direct_db_session, client

pytestmark = pytest.mark.usefixtures("_test_db")


# ===== helpers =====


_agent_name_counter = 0


def _create_agent_with_key() -> tuple[int, str]:
    global _agent_name_counter
    _agent_name_counter += 1
    headers = _admin_headers()
    agent_resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": f"v1 reply test agent {_agent_name_counter}",
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


@pytest.fixture(autouse=True)
def mock_start_task():
    """Stub the leaf that starts a real background turn for the seed POST.

    Only the initial ``POST /v1/chat/tasks`` used to create fixtures goes
    through this path; the reply endpoint itself never calls it.
    """
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ) as mocked:
        yield mocked


def _create_task(full_key: str, agent_id: int, content: str = "hello") -> int:
    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={"agent_id": agent_id, "message": {"role": "user", "content": content}},
    )
    assert resp.status_code == 202, resp.text
    return resp.json()["task_id"]


def _create_waiting_task(
    full_key: str,
    agent_id: int,
    *,
    run_id: str | None = "run-original",
    content: str = "hello",
) -> int:
    """Create a task and force it straight to waiting_for_user with a run_id.

    Mirrors what a real ask_user_question turn leaves behind: RUNNING ->
    WAITING_FOR_USER, lease released, run_id retained.
    """
    task_id = _create_task(full_key, agent_id, content=content)
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.status = TaskStatus.WAITING_FOR_USER
        task.control_state = TaskControlState.WAITING_FOR_USER.value
        task.run_id = run_id
        task.runner_id = None
        task.lease_expires_at = None
        task.last_heartbeat_at = None
        db.commit()
    finally:
        db.close()
    return task_id


def _insert_question_message(task_id: int, *, content: str = "Continue?") -> None:
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        db.add(
            TaskChatMessage(
                task_id=task_id,
                user_id=int(task.user_id),
                role="assistant",
                content=content,
                message_type="question",
                turn_id=f"question-{task_id}",
            )
        )
        db.commit()
    finally:
        db.close()


def _patch_agent_service(post_user_message: AsyncMock):
    agent_service = MagicMock()
    agent_service.post_user_message = post_user_message
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
    return patch(
        "xagent.web.api.chat.get_agent_manager",
        return_value=agent_manager,
    ), agent_service


def _reply_body(agent_id: int, content: str = "yes, continue") -> dict:
    return {"agent_id": agent_id, "message": {"role": "user", "content": content}}


def _seed_active_interaction_row(
    task_id: int, *, run_id: str, idempotency_key: str
) -> int:
    """One legal active TaskInteractionRequest row, for the legacy resume
    close test below. Mirrors test_a2a_api.py's identically-purposed
    helper, with origin="sdk" instead of "a2a" to match this endpoint's
    Task.source == "sdk" scoping."""
    db = _direct_db_session()
    try:
        anchor = TraceEvent(
            task_id=task_id,
            event_id=f"anchor-{idempotency_key}",
            event_type="agent_execution_checkpoint",
            timestamp=datetime.now(timezone.utc),
            data={},
        )
        db.add(anchor)
        db.flush()
        row = TaskInteractionRequest(
            task_id=task_id,
            run_id=run_id,
            kind="clarification",
            protocol_version=1,
            status="active",
            active_slot=1,
            origin="sdk",
            request_payload={"prompt": "example"},
            request_idempotency_key=idempotency_key,
            resume_trace_event_id=int(anchor.id),
            resume_event_id="resume-event-1",
            resume_execution_id="resume-execution-1",
            resume_locator_format="trace_event_pk_v1",
            resume_checkpoint_type="agent_execution_checkpoint",
            resume_run_partition=run_id,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return int(row.id)
    finally:
        db.close()


# ===== happy path =====


def test_reply_happy_path_resumes_the_same_run(mock_start_task):
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_waiting_task(full_key, agent_id, run_id="run-original")
    _insert_question_message(task_id)

    post_user_message = AsyncMock(return_value=True)
    agent_patch, agent_service = _patch_agent_service(post_user_message)
    with (
        agent_patch,
        patch(
            "xagent.web.api.v1.task_reply._schedule_waiting_reply_resume",
            new=AsyncMock(),
        ) as schedule_resume,
    ):
        resp = client.post(
            f"/v1/chat/tasks/{task_id}/reply",
            headers=_bearer(full_key),
            json=_reply_body(agent_id),
        )

    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["task_id"] == task_id
    assert body["agent_id"] == agent_id
    assert body["status"] == "running"
    assert body["run_id"] == "run-original"

    agent_service.post_user_message.assert_awaited_once()
    call_kwargs = agent_service.post_user_message.call_args.kwargs
    assert call_kwargs["execution_message"] == "yes, continue"
    assert call_kwargs["display_message"] == "yes, continue"
    assert call_kwargs["request_interrupt"] is False
    assert call_kwargs["turn_id"].startswith(f"v1:reply:{task_id}:")
    schedule_resume.assert_called_once()
    scheduled_lease = schedule_resume.call_args.kwargs["task_lease"]
    assert scheduled_lease.run_id == "run-original"

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        assert task.status == TaskStatus.RUNNING
        assert task.run_id == "run-original"
        assert task.input == "yes, continue"
        assert task.output is None
        assert task.error_message is None
    finally:
        db.close()


# ===== status matrix =====


@pytest.mark.parametrize(
    ("status", "expected_code"),
    [
        (TaskStatus.PENDING, "no_pending_interaction"),
        (TaskStatus.RUNNING, "task_busy"),
        (TaskStatus.PAUSED, "no_pending_interaction"),
        (TaskStatus.COMPLETED, "no_pending_interaction"),
        (TaskStatus.FAILED, "no_pending_interaction"),
    ],
)
def test_reply_status_matrix(mock_start_task, status: TaskStatus, expected_code: str):
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.status = status
        db.commit()
    finally:
        db.close()

    resp = client.post(
        f"/v1/chat/tasks/{task_id}/reply",
        headers=_bearer(full_key),
        json=_reply_body(agent_id),
    )
    assert resp.status_code == 409, resp.text
    assert resp.json()["error"]["code"] == expected_code


def test_reply_waiting_status_is_not_rejected_by_the_status_gate(mock_start_task):
    """The waiting row of the status matrix: WAITING_FOR_USER must fall
    through the status gate (not be rejected there) -- verified via the
    lease-acquisition dimension being reached instead."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_waiting_task(full_key, agent_id)
    _insert_question_message(task_id)

    post_user_message = AsyncMock(return_value=True)
    agent_patch, _ = _patch_agent_service(post_user_message)
    with (
        agent_patch,
        patch(
            "xagent.web.api.v1.task_reply._schedule_waiting_reply_resume",
            new=AsyncMock(),
        ),
    ):
        resp = client.post(
            f"/v1/chat/tasks/{task_id}/reply",
            headers=_bearer(full_key),
            json=_reply_body(agent_id),
        )
    assert resp.status_code == 202, resp.text


# ===== ownership =====


def test_reply_with_body_agent_id_mismatch_returns_404(mock_start_task):
    """The request body's agent_id must match the key-bound agent; a
    mismatch is 404 agent_not_found (not 403, so the existence of the
    other agent isn't leaked). Same key throughout -- this exercises
    body validation, not cross-key ownership (see the dedicated
    cross-key test below for that)."""
    agent_id, full_key = _create_agent_with_key()
    other_agent_id, _other_key = _create_agent_with_key()
    task_id = _create_waiting_task(full_key, agent_id)
    _insert_question_message(task_id)

    resp = client.post(
        f"/v1/chat/tasks/{task_id}/reply",
        headers=_bearer(full_key),
        json=_reply_body(other_agent_id),
    )
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "agent_not_found"


def test_reply_with_a_different_agents_key_returns_404_task_not_found(
    mock_start_task,
):
    """A key bound to a DIFFERENT agent than the task's owner must not
    reach the task at all -- 404 task_not_found from
    _resolve_task_or_404's ownership predicate, not the body-validation
    404 the test above exercises. Body is self-consistent (the second
    agent's own id) so only the ownership predicate can be at fault."""
    agent_id, full_key = _create_agent_with_key()
    other_agent_id, other_full_key = _create_agent_with_key()
    task_id = _create_waiting_task(full_key, agent_id)
    _insert_question_message(task_id)

    resp = client.post(
        f"/v1/chat/tasks/{task_id}/reply",
        headers=_bearer(other_full_key),
        json=_reply_body(other_agent_id),
    )
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "task_not_found"


def test_reply_to_non_sdk_source_task_returns_404(mock_start_task):
    agent_id, full_key = _create_agent_with_key()
    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        task = Task(
            user_id=int(agent.user_id),
            title="web ui task",
            status=TaskStatus.WAITING_FOR_USER,
            control_state=TaskControlState.WAITING_FOR_USER.value,
            run_id="run-web",
            agent_id=agent_id,
            source="web",
            is_visible=True,
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = int(task.id)
    finally:
        db.close()

    resp = client.post(
        f"/v1/chat/tasks/{task_id}/reply",
        headers=_bearer(full_key),
        json=_reply_body(agent_id),
    )
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "task_not_found"


def test_reply_to_unknown_task_returns_404(mock_start_task):
    _agent_id, full_key = _create_agent_with_key()
    resp = client.post(
        "/v1/chat/tasks/999999999/reply",
        headers=_bearer(full_key),
        json=_reply_body(_agent_id),
    )
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "task_not_found"


# ===== files rejected =====


def test_reply_with_files_returns_422(mock_start_task):
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_waiting_task(full_key, agent_id)
    _insert_question_message(task_id)

    resp = client.post(
        f"/v1/chat/tasks/{task_id}/reply",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "here", "files": ["file-1"]},
        },
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "invalid_input"


def test_reply_missing_agent_id_returns_422(mock_start_task):
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_waiting_task(full_key, agent_id)
    _insert_question_message(task_id)

    resp = client.post(
        f"/v1/chat/tasks/{task_id}/reply",
        headers=_bearer(full_key),
        json={"message": {"role": "user", "content": "here"}},
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "invalid_input"


# ===== checkpoint missing (endpoint layer) =====


def test_reply_checkpoint_missing_is_fail_closed(mock_start_task):
    """post_user_message returns False (no checkpoint found) -> 409
    interaction_not_resumable, task restored to waiting_for_user with
    zero residue: no runner_id, no lease, run_id unchanged, no error."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_waiting_task(full_key, agent_id, run_id="run-legacy")
    _insert_question_message(task_id)

    observed_lease: dict[str, TaskLease] = {}

    async def post_user_message(*_args, **_kwargs) -> bool:
        lease = current_task_lease()
        assert lease is not None
        assert lease.run_id == "run-legacy"
        observed_lease["lease"] = lease
        verify_db = _direct_db_session()
        try:
            leased = verify_db.query(Task).filter(Task.id == task_id).one()
            assert leased.status == TaskStatus.RUNNING
            assert leased.run_id == lease.run_id
        finally:
            verify_db.close()
        return False

    agent_patch, _ = _patch_agent_service(AsyncMock(side_effect=post_user_message))
    with agent_patch:
        resp = client.post(
            f"/v1/chat/tasks/{task_id}/reply",
            headers=_bearer(full_key),
            json=_reply_body(agent_id),
        )

    assert resp.status_code == 409, resp.text
    assert resp.json()["error"]["code"] == "interaction_not_resumable"
    assert observed_lease, "post_user_message stub never ran"

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        assert task.status == TaskStatus.WAITING_FOR_USER
        assert task.control_state == TaskControlState.WAITING_FOR_USER.value
        assert task.runner_id is None
        assert task.lease_expires_at is None
        assert task.run_id == "run-legacy"
        assert task.error_message is None
    finally:
        db.close()


# ===== legacy resume interaction close =====


def test_reply_closes_the_legacy_resume_interaction_row_on_successful_injection(
    mock_start_task,
):
    """The success path this wiring exists for: once the reply's fence
    UPDATE lands (``_update_reply_input_sync``), the run's active
    interaction row is retired (``terminated`` /
    ``answered_via_legacy_resume``) and the task's protocol marker is
    cleared back to NULL in the same commit -- mirrors
    test_a2a_api.py's
    test_message_send_closes_the_legacy_resume_interaction_row_on_successful_injection.

    Reverting the close wiring in ``_update_reply_input_sync`` turns this
    red: the row stays "active" and the marker stays 1.
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_waiting_task(full_key, agent_id, run_id="run-close-success")
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.interaction_protocol_version = 1
        db.commit()
    finally:
        db.close()
    row_id = _seed_active_interaction_row(
        task_id, run_id="run-close-success", idempotency_key="reply-close-success-q1"
    )

    post_user_message = AsyncMock(return_value=True)
    agent_patch, agent_service = _patch_agent_service(post_user_message)
    with (
        agent_patch,
        patch(
            "xagent.web.api.v1.task_reply._schedule_waiting_reply_resume",
            new=AsyncMock(),
        ),
    ):
        resp = client.post(
            f"/v1/chat/tasks/{task_id}/reply",
            headers=_bearer(full_key),
            json=_reply_body(agent_id),
        )

    assert resp.status_code == 202, resp.text
    agent_service.post_user_message.assert_awaited_once()

    db = _direct_db_session()
    try:
        row = (
            db.query(TaskInteractionRequest)
            .filter(TaskInteractionRequest.id == row_id)
            .one()
        )
        assert row.status == "terminated"
        assert row.terminal_reason == "answered_via_legacy_resume"
        refreshed = db.query(Task).filter(Task.id == task_id).one()
        assert refreshed.interaction_protocol_version is None
    finally:
        db.close()


# A fabricated id, not the seeded row's -- test_reply_reads_the_interaction_
# row_before_injecting hands this to the close instead of the real row id,
# so a site that re-read the row at close time would hand the close the
# real id and fail there instead.
_OBSERVED_INTERACTION_ID = 4321


def test_reply_reads_the_interaction_row_before_injecting(mock_start_task):
    """The close is keyed on the row observed *before* the injection,
    and only the ordering makes that true -- see task_interaction_close's
    module docstring. Moving the read after the injection leaves the whole
    change doing nothing while the row-level assertions in the test above
    stay green. The observed value is a fabricated id, not the seeded row's,
    so a site that re-read the row at close time would hand the close the
    real id and fail here."""

    agent_id, full_key = _create_agent_with_key()
    task_id = _create_waiting_task(full_key, agent_id, run_id="run-close-order")
    # Kept real and distinct from _OBSERVED_INTERACTION_ID: a site that
    # re-read the row at close time (instead of using the id observed before
    # injection) would hand the close this real id and fail the assertion
    # below.
    _seed_active_interaction_row(
        task_id, run_id="run-close-order", idempotency_key="reply-close-order-q1"
    )

    order: list[str] = []

    def record_read(_task_id: int) -> int:
        order.append("read")
        return _OBSERVED_INTERACTION_ID

    async def record_injection(*_args: object, **_kwargs: object) -> bool:
        order.append("inject")
        return True

    agent_patch, _agent_service = _patch_agent_service(
        AsyncMock(side_effect=record_injection)
    )
    with (
        agent_patch,
        patch(
            "xagent.web.api.v1.task_reply._schedule_waiting_reply_resume",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.v1.task_reply.active_interaction_id_sync",
            side_effect=record_read,
        ),
        patch(
            "xagent.web.api.v1.task_reply.close_legacy_resume_interaction",
            return_value=1,
        ) as close_mock,
    ):
        resp = client.post(
            f"/v1/chat/tasks/{task_id}/reply",
            headers=_bearer(full_key),
            json=_reply_body(agent_id),
        )

    assert resp.status_code == 202, resp.text
    assert order == ["read", "inject"]
    close_mock.assert_called_once()
    assert close_mock.call_args.kwargs["task_id"] == task_id
    assert close_mock.call_args.kwargs["run_id"] == "run-close-order"
    assert close_mock.call_args.kwargs["interaction_id"] == _OBSERVED_INTERACTION_ID


def test_update_reply_input_rolls_back_the_interaction_close_with_the_fence() -> None:
    """Mirrors test_a2a_api.py's
    test_update_a2a_resume_input_rolls_back_the_interaction_close_with_the_fence
    for the reply site's own fence-and-close pair: the fence UPDATE and the
    legacy resume interaction close are one atomic write in
    ``_update_reply_input_sync`` too. A fence miss (ownership changed under
    this lease) must roll back both together, not close the row while
    rejecting the input.

    What this actually pins is that the close must never commit
    independently of the host transaction -- reordering the two statements
    within that same transaction changes nothing observable, because a
    rollback undoes every statement issued since the last commit regardless
    of program order. Turning this red needs two changes at once, the same
    pair the a2a-side test calls out: the close must move ahead of the
    fence's early return (unreachable there today, since a fence miss
    returns before the close ever runs) and it must commit independently of
    this function's session.
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_waiting_task(full_key, agent_id, run_id="run-reply-atomicity")
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        # _create_waiting_task leaves the task in waiting_for_user with no
        # lease held (mirrors a real ask_user_question turn). This test
        # needs a live RUNNING lease for the fence UPDATE to have a chance
        # of matching, under a runner_id the stale lease below deliberately
        # does not share -- the fence's WHERE clause requires an exact
        # match, so that lease has already lost the race by the time the
        # write is attempted.
        task.status = TaskStatus.RUNNING
        task.runner_id = "current-runner"
        task.interaction_protocol_version = 1
        db.commit()
    finally:
        db.close()
    row_id = _seed_active_interaction_row(
        task_id, run_id="run-reply-atomicity", idempotency_key="reply-atomicity-q1"
    )

    stale_lease = TaskLease(
        task_id=task_id, runner_id="a-different-runner", run_id="run-reply-atomicity"
    )
    updated = task_reply_module._update_reply_input_sync(
        stale_lease, "attempted text", row_id
    )

    assert updated is False
    db = _direct_db_session()
    try:
        row = (
            db.query(TaskInteractionRequest)
            .filter(TaskInteractionRequest.id == row_id)
            .one()
        )
        assert row.status == "active"
        refreshed = db.query(Task).filter(Task.id == task_id).one()
        assert refreshed.interaction_protocol_version == 1
        # Unchanged from _create_task's original content, not overwritten
        # with the rejected fence write's "attempted text".
        assert refreshed.input == "hello"
    finally:
        db.close()


def test_reply_checkpoint_missing_restore_clears_an_unpaired_marker(mock_start_task):
    """The restore (abandonment) branch's compensation cleanup: when
    post_user_message returns False, the prelease is released back to
    waiting_for_user via ``_restore_reply_prelease_sync``, which must also
    reconcile a marker that no longer names any active row -- there is no
    active interaction row staged for this run, so the NOT EXISTS guard in
    ``clear_interaction_marker_if_unpaired`` matches and the marker clears.

    Reverting the ``clear_interaction_marker_if_unpaired`` call in
    ``_restore_reply_prelease_sync`` turns this red: the marker stays 1
    even though the task is back in waiting_for_user with no active row.
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_waiting_task(full_key, agent_id, run_id="run-unpaired-marker")
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.interaction_protocol_version = 1
        db.commit()
    finally:
        db.close()

    agent_patch, _ = _patch_agent_service(AsyncMock(return_value=False))
    with agent_patch:
        resp = client.post(
            f"/v1/chat/tasks/{task_id}/reply",
            headers=_bearer(full_key),
            json=_reply_body(agent_id),
        )

    assert resp.status_code == 409, resp.text
    assert resp.json()["error"]["code"] == "interaction_not_resumable"

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        assert task.status == TaskStatus.WAITING_FOR_USER
        assert task.interaction_protocol_version is None
    finally:
        db.close()


# ===== checkpoint read errors =====


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code"),
    [
        (
            CheckpointCorruptError("all matching rows undecodable"),
            409,
            "interaction_not_resumable",
        ),
        (
            CheckpointAccessRefusedError("refused", reason="superseded_legacy"),
            409,
            "interaction_not_resumable",
        ),
        (
            CheckpointAccessRefusedError("refused", reason="lease_mismatch"),
            409,
            "task_busy",
        ),
        (
            CheckpointAccessRefusedError("refused", reason="active_run"),
            409,
            "task_busy",
        ),
        (
            CheckpointUnavailableError("checkpoint store unavailable"),
            503,
            "temporarily_unavailable",
        ),
    ],
)
def test_reply_checkpoint_read_error_maps_to_distinct_code(
    mock_start_task,
    error: Exception,
    expected_status: int,
    expected_code: str,
):
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_waiting_task(
        full_key, agent_id, run_id=f"run-{type(error).__name__}"
    )
    _insert_question_message(task_id)

    agent_patch, _ = _patch_agent_service(AsyncMock(side_effect=error))
    with agent_patch:
        resp = client.post(
            f"/v1/chat/tasks/{task_id}/reply",
            headers=_bearer(full_key),
            json=_reply_body(agent_id),
        )

    assert resp.status_code == expected_status, resp.text
    assert resp.json()["error"]["code"] == expected_code

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        assert task.status == TaskStatus.WAITING_FOR_USER
        assert task.control_state == TaskControlState.WAITING_FOR_USER.value
        assert task.runner_id is None
        assert task.lease_expires_at is None
    finally:
        db.close()


def test_reply_unknown_checkpoint_read_error_subclass_is_treated_as_retryable(
    mock_start_task,
):
    """A future CheckpointReadError subclass this dispatch does not
    recognize must fall to the conservative 503 branch, not silently
    collapse into the terminal interaction_not_resumable code."""

    class _FutureCheckpointError(CheckpointReadError):
        pass

    agent_id, full_key = _create_agent_with_key()
    task_id = _create_waiting_task(full_key, agent_id)
    _insert_question_message(task_id)

    agent_patch, _ = _patch_agent_service(
        AsyncMock(side_effect=_FutureCheckpointError("unknown failure mode"))
    )
    with agent_patch:
        resp = client.post(
            f"/v1/chat/tasks/{task_id}/reply",
            headers=_bearer(full_key),
            json=_reply_body(agent_id),
        )

    assert resp.status_code == 503, resp.text
    assert resp.json()["error"]["code"] == "temporarily_unavailable"

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        assert task.status == TaskStatus.WAITING_FOR_USER
        assert task.control_state == TaskControlState.WAITING_FOR_USER.value
        assert task.runner_id is None
        assert task.lease_expires_at is None
    finally:
        db.close()


# ===== concurrent reply race =====


@pytest.mark.asyncio
async def test_concurrent_reply_race_exactly_one_winner(mock_start_task):
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_waiting_task(full_key, agent_id, run_id="run-race")
    _insert_question_message(task_id)

    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        owner_user_id = int(agent.user_id)
    finally:
        db.close()

    post_user_message = AsyncMock(return_value=True)
    agent_patch, _ = _patch_agent_service(post_user_message)

    from xagent.web.api.v1.deps import (
        AgentPrincipalSnapshot,
        ApiKeyPrincipal,
        RuntimeApiKeySnapshot,
    )

    principal = ApiKeyPrincipal(
        key=RuntimeApiKeySnapshot(key_prefix="xag_test"),
        agent=AgentPrincipalSnapshot(
            id=agent_id,
            user_id=owner_user_id,
            execution_mode="balanced",
            status="published",
            origin="user_created",
        ),
    )

    async def attempt(content: str) -> object:
        try:
            return await task_reply_module.reply_to_task(
                task_id=task_id,
                request=ReplyRequest(
                    agent_id=agent_id,
                    message={"role": "user", "content": content},
                ),
                principal=principal,
            )
        except Exception as exc:  # noqa: BLE001
            return exc

    with (
        agent_patch,
        patch(
            "xagent.web.api.v1.task_reply._schedule_waiting_reply_resume",
            new=AsyncMock(),
        ),
    ):
        result_a, result_b = await asyncio.gather(
            attempt("answer A"), attempt("answer B")
        )

    from xagent.web.api.v1.errors import V1ApiError

    results = [result_a, result_b]
    successes = [r for r in results if not isinstance(r, Exception)]
    failures = [r for r in results if isinstance(r, Exception)]
    assert len(successes) == 1, (result_a, result_b)
    assert len(failures) == 1, (result_a, result_b)
    assert isinstance(failures[0], V1ApiError)
    assert failures[0].code.value == "task_busy"

    db = _direct_db_session()
    try:
        # post_user_message is mocked (no real transcript write here), but
        # the winner must be exactly one: exactly one successful claim,
        # exactly one input write.
        task = db.query(Task).filter(Task.id == task_id).one()
        assert task.status == TaskStatus.RUNNING
        assert task.input in ("answer A", "answer B")
    finally:
        db.close()
    assert post_user_message.await_count == 1


# ===== run_id NULL historical waiting row =====


def test_reply_untagged_checkpoint_is_not_resumed_without_an_exact_run(mock_start_task):
    """A legacy waiting task with no run_id: the prelease mints a new run
    (no run fence to preserve), post_user_message finds no checkpoint
    fenced to that fresh run, and the endpoint fails closed."""
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_waiting_task(full_key, agent_id, run_id=None)
    _insert_question_message(task_id)

    async def post_user_message(*_args, **_kwargs) -> bool:
        lease = current_task_lease()
        assert lease is not None
        assert lease.run_id is not None
        verify_db = _direct_db_session()
        try:
            leased = verify_db.query(Task).filter(Task.id == task_id).one()
            assert leased.run_id == lease.run_id
        finally:
            verify_db.close()
        return False

    agent_patch, _ = _patch_agent_service(AsyncMock(side_effect=post_user_message))
    with agent_patch:
        resp = client.post(
            f"/v1/chat/tasks/{task_id}/reply",
            headers=_bearer(full_key),
            json=_reply_body(agent_id),
        )

    assert resp.status_code == 409, resp.text
    assert resp.json()["error"]["code"] == "interaction_not_resumable"
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        assert task.status == TaskStatus.WAITING_FOR_USER
        assert task.runner_id is None
        assert task.lease_expires_at is None
        assert task.run_id is not None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_reply_resume_binds_the_coordinator_to_the_leased_run() -> None:
    """The v1 reply scheduler must register its coordinator against its run.

    The coordinator is the process-local evidence a duplicate RESUME command
    is allowed to complete against. Bound to the wrong run -- or to no run at
    all -- it would let a command for a different run be recorded as an
    idempotent success.
    """

    from xagent.web.api import websocket as websocket_api

    real_manager = websocket_api.BackgroundTaskManager()
    lease = TaskLease(task_id=4242, runner_id="runner-x", run_id="run-reply")
    resume_gate = asyncio.Event()

    async def execute_resume_background(**_kwargs) -> None:
        await resume_gate.wait()

    with (
        patch.object(websocket_api, "background_task_manager", real_manager),
        patch.object(
            websocket_api,
            "execute_resume_background",
            side_effect=execute_resume_background,
        ),
    ):
        await task_reply_module._schedule_waiting_reply_resume(
            task_id=4242,
            agent_service=MagicMock(),
            task_owner_user_id=1,
            task_lease=lease,
            heartbeat_stop=asyncio.Event(),
            heartbeat_task=asyncio.ensure_future(asyncio.sleep(0)),
        )
        try:
            assert (
                real_manager.resume_admission_state(4242, expected_run_id="run-reply")
                is websocket_api.ResumeReservationOutcome.COORDINATOR_RUNNING
            )
            assert (
                real_manager.resume_admission_state(
                    4242, expected_run_id="some-other-run"
                )
                is websocket_api.ResumeReservationOutcome.RESERVATION_HELD
            )
        finally:
            resume_gate.set()
            coordinator = real_manager.resume_tasks[4242]
            await asyncio.wait_for(coordinator, timeout=5)
