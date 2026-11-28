"""A chat message that read no active question, racing a finalizer's publication.

A message handler reads the task's active interaction before it injects the
user's answer. When a finalizer publishes its question between that read and
the answer's close, the close is bound to the empty read and matches nothing,
so the published question stays active although the answer is already
accepted. The run then resumes and asks again.

These cells pause the real handler right after its read, publish the first
question through each of the three finalizers, complete the same accepted
answer (injected live, or deferred to the resumed execution), and let the
resumed run publish its second question through the real finalizer. The
first question must end retired and the second must be the one the read
surface returns. The agent and the websocket and background task managers are
doubles; the handler, the interaction read, the close,
``execute_resume_background`` and the finalizers are real.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.web.services.test_finalizer_publication import _witness_retries  # noqa: F401
from tests.web.services.test_finalizer_publication import database  # noqa: F401
from tests.web.services.test_finalizer_publication import (
    FINALIZERS,
    QUESTION,
    SECOND_EVENT_ID,
    SECOND_QUESTION,
    _finalize,
    _native,
    _prepare,
    _rows,
    _second_question_result,
    _seed_round,
    _waiting_result,
)
from xagent.core.agent.runner import UserMessageInjectionOutcome
from xagent.web.api.websocket import _make_command_reply
from xagent.web.models.database import get_session_local
from xagent.web.models.task import Task
from xagent.web.models.user import User
from xagent.web.services import task_lease_service as leases
from xagent.web.services.task_command_execution import handle_task_message
from xagent.web.services.task_execution import (
    ResumeReservationOutcome,
    execute_resume_background,
)
from xagent.web.services.task_interaction_close import (
    ActiveInteractionAbsent,
    ActiveInteractionFound,
    active_interaction_id_sync,
)
from xagent.web.services.task_interaction_read import get_pending_interaction_question


def _pending_question(task_id: int) -> str | None:
    with get_session_local()() as db:
        pending = get_pending_interaction_question(db, db.get(Task, task_id))
        return pending[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("finalizer", FINALIZERS)
@pytest.mark.parametrize("injection", ["live", "deferred"])
async def test_an_answer_accepted_before_publication_lets_the_next_question_retire_it(
    database,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    finalizer: str,
    injection: str,
) -> None:
    _native(monkeypatch)
    round_ = _seed_round()
    _prepare(finalizer, round_)
    registration = leases._get_task_lease_heartbeat_manager().register(round_.lease)
    observed: list[str] = []

    def read_then_publish(task_id: int) -> Any:
        read = active_interaction_id_sync(task_id)
        if observed:
            # Only the first read is the one the finalizer races.
            return read
        observed.append(type(read).__name__)
        assert isinstance(read, ActiveInteractionAbsent)
        _finalize(finalizer, round_, _waiting_result())
        return read

    outcomes = (
        [UserMessageInjectionOutcome.POSTED_FRESH]
        if injection == "live"
        else [
            UserMessageInjectionOutcome.NOT_POSTED,
            UserMessageInjectionOutcome.POSTED_FRESH,
        ]
    )
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    agent.post_user_message = AsyncMock(side_effect=outcomes)
    agent.resume_execution_by_id = AsyncMock(
        side_effect=lambda *_a, **_k: _second_question_result(round_)
    )
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(), send_personal_message=AsyncMock()
    )
    background = MagicMock()
    background.try_reserve_resume.return_value = ResumeReservationOutcome.RESERVED
    previous_execution = MagicMock()
    background.running_tasks.get.return_value = previous_execution
    resume_call = AsyncMock()

    with get_session_local()() as db:
        owner = db.get(User, round_.user_id)
        db.expunge(owner)
    try:
        with (
            patch(
                "xagent.web.services.agent_service_manager.get_agent_manager",
                return_value=MagicMock(
                    get_agent_for_task=AsyncMock(return_value=agent)
                ),
            ),
            patch("xagent.web.api.websocket.manager", ws_manager),
            patch(
                "xagent.web.services.task_execution.background_task_manager", background
            ),
            patch(
                "xagent.web.services.task_execution.execute_resume_background",
                resume_call,
            ),
            patch(
                "xagent.web.services.task_command_execution.active_interaction_id_sync",
                side_effect=read_then_publish,
            ),
        ):
            await handle_task_message(
                _make_command_reply(MagicMock()),
                round_.task_id,
                {
                    "message": "Use staging",
                    "client_message_id": f"race-{finalizer}-{injection}",
                    "user": owner,
                    "files": [],
                },
            )

        # The answer is accepted and the handler is done, but the close was
        # bound to the empty read: the first question is still the active row
        # and the one the read surface returns.
        assert observed == ["ActiveInteractionAbsent"]
        resume_call.assert_called_once()
        resume_kwargs = dict(resume_call.call_args.kwargs)
        assert resume_kwargs.pop("previous_task") is previous_execution
        after_handler = _rows(round_.task_id)
        assert [(status, reason) for _, status, reason, _ in after_handler] == [
            ("active", None)
        ]
        assert _pending_question(round_.task_id) == QUESTION
        assert agent.post_user_message.await_count == 1
        if injection == "deferred":
            assert resume_kwargs["pending_user_message"]["interaction_id"] is None

        with (
            patch("xagent.web.api.websocket.manager", ws_manager),
            patch(
                "xagent.web.services.task_execution.background_task_manager.promote_resume_task"
            ),
        ):
            await execute_resume_background(**resume_kwargs)
    finally:
        await registration.close()
        await leases._get_task_lease_heartbeat_manager().wait_until_idle()

    assert agent.post_user_message.await_count == len(outcomes)
    agent.resume_execution_by_id.assert_awaited_once()
    first_id, _, _, first_key = after_handler[0]
    rows = _rows(round_.task_id)
    assert [(status, reason, key) for _, status, reason, key in rows] == [
        ("terminated", "run_superseded", first_key),
        ("active", None, SECOND_EVENT_ID),
    ]
    assert rows[0][0] == first_id
    assert _pending_question(round_.task_id) == SECOND_QUESTION
    with get_session_local()() as db:
        assert db.get(Task, round_.task_id).interaction_protocol_version == 1
    read = active_interaction_id_sync(round_.task_id)
    assert isinstance(read, ActiveInteractionFound)
    assert read.interaction_id == rows[1][0]


@pytest.mark.asyncio
async def test_a_resume_starts_only_after_the_previous_execution_ended(
    database,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executions under one coordinator share an attempt id, so the database
    cannot tell an older finalizer from a newer one. What keeps the older one
    from retiring the newer one's question is this wait: the resume claims
    nothing, and so settles nothing, until the previous execution has ended."""
    _native(monkeypatch)
    round_ = _seed_round()
    release = asyncio.Event()
    previous_task = asyncio.create_task(release.wait())
    acquire = MagicMock(return_value=None)

    with (
        patch("xagent.web.services.task_execution._acquire_resume_task_lease", acquire),
        patch(
            "xagent.web.services.task_execution.background_task_manager.promote_resume_task"
        ),
    ):
        resume = asyncio.create_task(
            execute_resume_background(
                task_id=round_.task_id,
                agent_service=MagicMock(),
                task_owner_user_id=round_.user_id,
                expected_run_id=round_.lease.run_id,
                previous_task=previous_task,
            )
        )
        for _ in range(10):
            await asyncio.sleep(0)
        acquire.assert_not_called()
        release.set()
        await resume
    acquire.assert_called_once()
