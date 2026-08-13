"""Integration tests for ``POST /v1/chat/tasks/{task_id}/reply``.

Mirrors the structure of ``tests/web/api/test_a2a_api.py``'s
input-required resume tests, since ``task_reply.py`` copies that
resume's mechanics. Where a2a's test patches
``xagent.web.api.a2a._schedule_waiting_a2a_resume`` to avoid spinning
up a real background execution, these tests patch the analogous
``xagent.web.api.v1.task_reply._schedule_waiting_reply_resume``.
"""

import asyncio
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
from xagent.web.models.task import Task, TaskStatus
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
