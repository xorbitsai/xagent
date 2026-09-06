"""Structured command-outcome fields on terminal ``agent_error`` frames.

Issue #1500: a client deciding whether an accepted clarification reply is
safe to resend must read a structured, command-correlated outcome instead of
parsing human-readable error text. These tests pin the wire contract of the
live terminal broadcast: the identity-bearing first-party frame carries
``outcome``, ``resend_safe`` and ``message_code``, the anonymous external
frames keep their pinned minimal shape, and a first-party MESSAGE terminal
states what is provable about application instead of restating the condition
the command was last waiting on.

Field names deliberately match the durable terminal-event projection planned
in #1904 (``outcome``/``resend_safe``), so the frontend contract survives the
switch from this live broadcast to durable delivery.
"""

from unittest.mock import AsyncMock, patch

import pytest

from xagent.web.api import websocket as websocket_api
from xagent.web.api.websocket import (
    ClientVisibleTaskCommandDeferred,
    execute_durable_task_command,
)
from xagent.web.services.external_task_input import (
    EXTERNAL_INPUT_NOT_APPLIED_MESSAGE,
)
from xagent.web.services.task_command_terminal_events import (
    FIRST_PARTY_MESSAGE_NOT_APPLIED_MESSAGE,
    FIRST_PARTY_MESSAGE_UNCONFIRMED_MESSAGE,
    TerminalTaskEventDraft,
    TerminalTaskEventMessageCode,
    bind_terminal_event_draft,
)
from xagent.web.services.task_command_transport import (
    MAX_COMMAND_FAILURES,
    ClaimedTaskCommand,
    TaskCommandDeferred,
    TaskCommandKind,
    TaskCommandRejected,
    max_command_defers,
)


def _message_command(
    *,
    payload: dict | None = None,
    defer_count: int = 0,
    failure_count: int = 0,
    kind: TaskCommandKind = TaskCommandKind.MESSAGE,
) -> ClaimedTaskCommand:
    return ClaimedTaskCommand(
        id=1,
        task_id=7,
        actor_user_id=None,
        command_id="clarification-reply-1",
        kind=kind,
        payload=payload if payload is not None else {},
        target_run_id="run-1",
        attempt_count=defer_count + 1,
        failure_count=failure_count,
        defer_count=defer_count,
    )


async def _run_terminal(command: ClaimedTaskCommand, error: BaseException) -> dict:
    """Drive one exhausted execution and return the broadcast frame."""

    with (
        patch.object(
            websocket_api,
            "_execute_durable_task_command",
            new=AsyncMock(side_effect=error),
        ),
        patch.object(
            websocket_api.manager,
            "broadcast_to_task",
            new=AsyncMock(),
        ) as broadcast,
    ):
        with pytest.raises(type(error)):
            await execute_durable_task_command(command)
        broadcast.assert_awaited_once()
        frame, task_id = broadcast.await_args.args
        assert task_id == command.task_id
        return frame


@pytest.mark.asyncio
async def test_exhausted_resend_safe_deferral_broadcasts_not_applied_outcome() -> None:
    command = _message_command(defer_count=max_command_defers() - 1)
    error = TaskCommandDeferred(
        "Message clarification-reply-1 is waiting for the live-control resume slot",
        resend_safe=True,
    )

    frame = await _run_terminal(command, error)

    assert frame["type"] == "agent_error"
    assert frame["outcome"] == "failed"
    assert frame["resend_safe"] is True
    assert frame["message_code"] == "task_command_deferred"
    assert frame["command_id"] == "clarification-reply-1"
    assert frame["command_kind"] == "message"
    assert frame["message"] == FIRST_PARTY_MESSAGE_NOT_APPLIED_MESSAGE
    # The durable projection's disambiguators travel too: an operator retry
    # can send one command_id through a terminal broadcast twice, and a
    # consumer must be able to tell the two outcomes apart.
    assert frame["task_run_id"] == "run-1"
    assert frame["outcome_version"] == command.attempt_count
    # Exhaustive: a leaked extra field or a silently dropped identity field
    # on the first-party frame is a wire-contract change.
    assert set(frame) == {
        "type",
        "message",
        "outcome",
        "resend_safe",
        "message_code",
        "command_kind",
        "task_id",
        "command_id",
        "task_run_id",
        "outcome_version",
        "timestamp",
    }


@pytest.mark.asyncio
async def test_exhausted_unsafe_deferral_broadcasts_unconfirmed_outcome() -> None:
    command = _message_command(defer_count=max_command_defers() - 1)
    error = TaskCommandDeferred(
        "Message clarification-reply-1 is waiting for runtime injection",
        resend_safe=False,
    )

    frame = await _run_terminal(command, error)

    assert frame["outcome"] == "failed"
    assert frame["resend_safe"] is False
    assert frame["message_code"] == "task_command_deferred"
    assert frame["message"] == FIRST_PARTY_MESSAGE_UNCONFIRMED_MESSAGE


@pytest.mark.asyncio
async def test_message_terminal_wording_never_restates_the_wait_reason() -> None:
    """The exhaustion notice must not read "failed: ... is waiting for ...".

    ``ClientVisibleTaskCommandDeferred`` specifically: a plain
    ``TaskCommandDeferred`` already fell into the generic redacted fallback
    before this fix, so only the client-visible subclass - the class the
    real deferral raise sites use - exercises the regression this test pins.
    """

    command = _message_command(defer_count=max_command_defers() - 1)
    # Constructed exactly as the real raise sites construct it: message only.
    # (ClientVisibleError's __init__ wins the MRO and takes no resend_safe;
    # the attribute falls back to TaskCommandDeferred's unsafe default.)
    error = ClientVisibleTaskCommandDeferred(
        "Message clarification-reply-1 is waiting for the live-control resume slot",
    )

    frame = await _run_terminal(command, error)

    assert "waiting" not in frame["message"]
    assert "resume slot" not in frame["message"]


@pytest.mark.asyncio
async def test_generic_message_failure_broadcasts_unconfirmed_outcome() -> None:
    command = _message_command(failure_count=MAX_COMMAND_FAILURES - 1)
    error = RuntimeError("worker exploded mid-injection")

    frame = await _run_terminal(command, error)

    assert frame["outcome"] == "failed"
    assert frame["resend_safe"] is False
    assert frame["message_code"] == "task_command_failed"
    assert frame["message"] == FIRST_PARTY_MESSAGE_UNCONFIRMED_MESSAGE
    assert "exploded" not in frame["message"]


@pytest.mark.asyncio
async def test_non_message_terminal_frame_gains_fields_keeps_wording() -> None:
    command = _message_command(
        kind=TaskCommandKind.PAUSE,
        failure_count=MAX_COMMAND_FAILURES - 1,
    )
    error = RuntimeError("boom")

    frame = await _run_terminal(command, error)

    assert frame["outcome"] == "failed"
    assert frame["resend_safe"] is False
    assert frame["message_code"] == "task_command_failed"
    assert frame["message"].startswith("Task command pause failed:")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind",
    [TaskCommandKind.PAUSE, TaskCommandKind.RESUME, TaskCommandKind.CANCEL],
)
async def test_non_message_deferral_exhaustion_carries_structured_fields(
    kind: TaskCommandKind,
) -> None:
    """Every first-party kind's exhausted deferral carries the same fields.

    ``resend_safe`` stays ``False`` here by design - it is a proof of
    non-application whose only producer is the MESSAGE contention deferral,
    not a retryability rating for these idempotent kinds.
    """

    command = _message_command(kind=kind, defer_count=max_command_defers() - 1)
    error = ClientVisibleTaskCommandDeferred(
        f"{kind.value.title()} command clarification-reply-1 is waiting "
        "for the active task lease owner",
    )

    frame = await _run_terminal(command, error)

    assert frame["outcome"] == "failed"
    assert frame["resend_safe"] is False
    assert frame["message_code"] == "task_command_deferred"
    assert frame["command_kind"] == kind.value
    assert frame["command_id"] == "clarification-reply-1"
    assert frame["message"].startswith(f"Task command {kind.value} failed:")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("defer_count", "failure_count", "error"),
    [
        (
            max_command_defers() - 2,
            0,
            TaskCommandDeferred("still waiting", resend_safe=False),
        ),
        (0, MAX_COMMAND_FAILURES - 2, RuntimeError("transient")),
    ],
    ids=["deferral-below-boundary", "failure-below-boundary"],
)
async def test_one_step_below_the_terminal_boundary_broadcasts_nothing(
    defer_count: int,
    failure_count: int,
    error: BaseException,
) -> None:
    """A non-terminal round is silent; only exhaustion broadcasts."""

    command = _message_command(
        defer_count=defer_count,
        failure_count=failure_count,
    )

    with (
        patch.object(
            websocket_api,
            "_execute_durable_task_command",
            new=AsyncMock(side_effect=error),
        ),
        patch.object(
            websocket_api.manager,
            "broadcast_to_task",
            new=AsyncMock(),
        ) as broadcast,
    ):
        with pytest.raises(type(error)):
            await execute_durable_task_command(command)
        broadcast.assert_not_awaited()


@pytest.mark.asyncio
async def test_broadcast_reads_the_bound_draft_not_the_exception() -> None:
    """The frame's fields come from the dispatcher-bound draft.

    Differential pin ahead of durable replay (#1904): today every draft is
    derived mechanically from its exception, so a regression that read
    ``error.resend_safe`` directly would pass every other test unchanged.
    This one binds a draft that contradicts the exception and asserts the
    draft wins.
    """

    command = _message_command()
    error = TaskCommandDeferred("deferred", resend_safe=False)
    bind_terminal_event_draft(
        error,
        TerminalTaskEventDraft(
            message_code=TerminalTaskEventMessageCode.TASK_COMMAND_DEFERRED,
            resend_safe=True,
        ),
    )

    with patch.object(
        websocket_api.manager,
        "broadcast_to_task",
        new=AsyncMock(),
    ) as broadcast:
        await websocket_api._broadcast_terminal_command_error(command, error)

    frame, task_id = broadcast.await_args.args
    assert task_id == command.task_id
    assert frame["resend_safe"] is True
    assert frame["message"] == FIRST_PARTY_MESSAGE_NOT_APPLIED_MESSAGE


@pytest.mark.asyncio
async def test_external_scope_terminal_frame_stays_identity_free() -> None:
    """The anonymous external frame gains no structured outcome fields.

    A retry decision needs the ``command_id`` the external frames withhold,
    so the structured fields would be undecidable noise there; the frame
    keeps its pinned minimal shape and its proof-aware wording.
    """

    command = _message_command(payload={"scope": "external"})
    error = TaskCommandRejected(
        "principal revoked",
        reason="revoked_principal",
    )

    frame = await _run_terminal(command, error)

    assert set(frame) == {"type", "message", "task_id", "timestamp"}
    assert frame["message"] == EXTERNAL_INPUT_NOT_APPLIED_MESSAGE
