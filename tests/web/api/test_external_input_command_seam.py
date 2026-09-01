"""External-scope MESSAGE commands: routing, registration, and the boundary.

The seam under test lets an embedding application execute external task
input through the durable command transport. Every test drives one edge:
the dispatcher adapter routes an external-scope MESSAGE to the registered
executor and nothing else, a deferral crosses the adapter untouched so the
transport's defer budget (not the failure budget) absorbs lease contention,
a deployment without a registered core refuses terminally, a scoped command
never runs the first-party chat core, and a client frame cannot mint a
scoped command at the WebSocket enqueue boundary.
"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from xagent.web.api import websocket as websocket_api
from xagent.web.models.agent import Agent
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.task_command_terminal_event import TaskCommandTerminalEvent
from xagent.web.services.external_task_input import (
    EXTERNAL_INPUT_NOT_APPLIED_MESSAGE,
    EXTERNAL_INPUT_UNCONFIRMED_MESSAGE,
    execute_external_task_input_command,
    register_external_task_input_executor,
    registered_external_task_input_executor,
    unregister_external_task_input_executor,
)
from xagent.web.services.task_command_transport import (
    COMMAND_COMPLETED,
    COMMAND_FAILED,
    COMMAND_PENDING,
    MAX_COMMAND_DEFERS,
    MAX_COMMAND_FAILURES,
    ClaimedTaskCommand,
    TaskCommandDeferred,
    TaskCommandKind,
    TaskCommandRejected,
    dispatch_one_task_command,
    enqueue_task_command,
    max_command_defers,
)
from xagent.web.services.task_execution_controller import TaskControlState

from .conftest import _admin_headers, _direct_db_session, client

pytestmark = pytest.mark.usefixtures("_test_db")


@pytest.fixture(autouse=True)
def _clean_executor_registry():
    unregister_external_task_input_executor()
    yield
    unregister_external_task_input_executor()


def _claimed_message_command(
    *,
    task_id: int = 101,
    payload: dict[str, Any],
    command_id: str = "ext-input-1",
    defer_count: int = 0,
) -> ClaimedTaskCommand:
    return ClaimedTaskCommand(
        id=1,
        task_id=task_id,
        actor_user_id=None,
        command_id=command_id,
        kind=TaskCommandKind.MESSAGE,
        payload=payload,
        target_run_id=None,
        attempt_count=1,
        defer_count=defer_count,
    )


def _create_agent() -> tuple[int, int]:
    headers = _admin_headers()
    response = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": "External input agent",
            "description": "External input test agent",
            "instructions": "You are an external input test agent.",
            "execution_mode": "balanced",
        },
    )
    assert response.status_code == 200, response.text
    agent_id = int(response.json()["id"])
    db = _direct_db_session()
    try:
        owner_user_id = int(db.query(Agent).filter(Agent.id == agent_id).one().user_id)
    finally:
        db.close()
    return agent_id, owner_user_id


def _create_waiting_task(*, agent_id: int, owner_user_id: int) -> int:
    db = _direct_db_session()
    try:
        task = Task(
            user_id=owner_user_id,
            title="external input task",
            status=TaskStatus.WAITING_FOR_USER,
            control_state=TaskControlState.RUNNING.value,
            run_id="run-external",
            state_version=4,
            agent_id=agent_id,
            source="external",
            is_visible=False,
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        return int(task.id)
    finally:
        db.close()


def _forbid_first_party_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fail(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError(
            "the first-party chat core must not run for a scoped MESSAGE"
        )

    monkeypatch.setattr(websocket_api, "_handle_chat_message_unserialized", _fail)


# ---------------------------------------------------------------------------
# Registration semantics
# ---------------------------------------------------------------------------


def test_registering_the_same_executor_twice_is_a_no_op() -> None:
    async def executor(_command: ClaimedTaskCommand) -> None:
        return None

    register_external_task_input_executor(executor)
    register_external_task_input_executor(executor)
    assert registered_external_task_input_executor() is executor


def test_registering_a_different_executor_is_refused() -> None:
    async def first(_command: ClaimedTaskCommand) -> None:
        return None

    async def second(_command: ClaimedTaskCommand) -> None:
        return None

    register_external_task_input_executor(first)
    with pytest.raises(ValueError, match="already registered"):
        register_external_task_input_executor(second)
    assert registered_external_task_input_executor() is first


def test_unregister_returns_the_executor_and_frees_the_slot() -> None:
    async def first(_command: ClaimedTaskCommand) -> None:
        return None

    async def second(_command: ClaimedTaskCommand) -> None:
        return None

    register_external_task_input_executor(first)
    assert unregister_external_task_input_executor() is first
    register_external_task_input_executor(second)
    assert registered_external_task_input_executor() is second


def test_a_non_callable_executor_is_refused() -> None:
    with pytest.raises(TypeError, match="callable"):
        register_external_task_input_executor("not-callable")  # type: ignore[arg-type]
    assert registered_external_task_input_executor() is None


# ---------------------------------------------------------------------------
# Dispatcher adapter routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_external_scope_message_routes_to_the_registered_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter hands the claimed command to the seam verbatim.

    The actor load, origin resolution, and delivery-ledger probes of the
    first-party arm belong to first-party rows; none of them may run for an
    external-scope command, so the routing must happen before all of them
    (``actor_user_id=None`` would otherwise fail the actor load).
    """

    _forbid_first_party_chat(monkeypatch)
    observed: list[ClaimedTaskCommand] = []

    async def executor(command: ClaimedTaskCommand) -> dict[str, Any]:
        observed.append(command)
        return {"delivered": True}

    register_external_task_input_executor(executor)
    command = _claimed_message_command(
        payload={"scope": "external", "message": "the answer"}
    )

    result = await websocket_api._execute_durable_task_command(command)

    assert observed == [command]
    assert result == {"delivered": True}


@pytest.mark.asyncio
async def test_external_scope_message_deferral_crosses_the_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deferral must reach the dispatcher as ``TaskCommandDeferred``.

    That exception type is what routes the disposition to the defer budget
    instead of the failure budget; wrapping or swallowing it would consume
    terminal failure attempts on a condition that clears by itself.
    """

    _forbid_first_party_chat(monkeypatch)

    async def executor(_command: ClaimedTaskCommand) -> None:
        raise TaskCommandDeferred("the parked run still holds the resume lease")

    register_external_task_input_executor(executor)
    command = _claimed_message_command(
        payload={"scope": "external", "message": "the answer"}
    )

    with pytest.raises(TaskCommandDeferred, match="resume lease"):
        await websocket_api._execute_durable_task_command(command)


@pytest.mark.asyncio
async def test_external_scope_message_without_a_registered_core_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No registered core means no retry can succeed: reject, don't defer.

    An indefinitely PENDING command would also block every later command
    for the same task behind the per-task FIFO.
    """

    _forbid_first_party_chat(monkeypatch)
    command = _claimed_message_command(
        payload={"scope": "external", "message": "the answer"}
    )

    with pytest.raises(TaskCommandRejected, match="no external input execution core"):
        await websocket_api._execute_durable_task_command(command)


@pytest.mark.asyncio
async def test_foreign_scope_message_is_rejected_terminally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scope naming no execution core must not run the first-party core."""

    _forbid_first_party_chat(monkeypatch)

    async def executor(_command: ClaimedTaskCommand) -> None:
        raise AssertionError("the external core must not run for a foreign scope")

    register_external_task_input_executor(executor)

    for scope_value in ("workforce", {"nested": "dict"}):
        command = _claimed_message_command(
            payload={"scope": scope_value, "message": "x"}
        )
        with pytest.raises(TaskCommandRejected, match="has no execution core"):
            await websocket_api._execute_durable_task_command(command)


@pytest.mark.asyncio
async def test_scopeless_message_stays_on_the_first_party_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pre-external payload shape keeps its first-party execution core."""

    async def seam_must_not_run(_command: ClaimedTaskCommand) -> None:
        raise AssertionError("the external seam must not run for a scopeless MESSAGE")

    register_external_task_input_executor(seam_must_not_run)

    first_party_calls: list[int] = []

    async def record_chat(_ws: Any, task_id: int, _data: dict[str, Any]) -> None:
        first_party_calls.append(task_id)

    monkeypatch.setattr(websocket_api, "_handle_chat_message_unserialized", record_chat)
    monkeypatch.setattr(
        websocket_api,
        "_load_command_actor",
        lambda _actor_id: SimpleNamespace(id=7, is_admin=False),
    )
    monkeypatch.setattr(
        websocket_api,
        "_load_command_message_delivery_status",
        lambda _task_id, _turn_id: "dispatched",
    )

    command = ClaimedTaskCommand(
        id=1,
        task_id=55,
        actor_user_id=7,
        command_id="chat-1",
        kind=TaskCommandKind.MESSAGE,
        payload={"message": "hello"},
        target_run_id=None,
        attempt_count=1,
    )

    result = await websocket_api._execute_durable_task_command(command)

    assert first_party_calls == [55]
    assert result == {"task_id": 55, "command_id": "chat-1", "kind": "message"}


# ---------------------------------------------------------------------------
# The seam itself
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_seam_rejects_with_unsupported_scope_reason_when_unregistered() -> None:
    command = _claimed_message_command(payload={"scope": "external", "message": "x"})
    with pytest.raises(TaskCommandRejected) as excinfo:
        await execute_external_task_input_command(command)
    assert excinfo.value.reason == "unsupported_scope"


# ---------------------------------------------------------------------------
# First-party WebSocket enqueue boundary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_frames_cannot_name_a_command_scope() -> None:
    """``scope`` is server-owned: a client frame carrying it is refused.

    Without this boundary a first-party chat frame could route its MESSAGE
    command into the external execution core with a forged identity payload.
    Loud rejection rather than a silent strip, so the sender learns the
    frame was not accepted as written.
    """

    for kind in (
        TaskCommandKind.MESSAGE,
        TaskCommandKind.PAUSE,
        TaskCommandKind.RESUME,
    ):
        with pytest.raises(websocket_api.ClientVisibleValidationError, match="scope"):
            await websocket_api._enqueue_websocket_task_command(
                task_id=1,
                message_data={
                    "user": SimpleNamespace(id=7, is_admin=False),
                    "message": "hello",
                    "scope": "external",
                },
                kind=kind,
                command_id="forged-1",
            )


# ---------------------------------------------------------------------------
# End to end through the durable transport
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deferred_external_input_spends_the_defer_budget_not_failures() -> None:
    """Contention defers durably and the retry converges on one delivery.

    This is the transport chain an embedding application relies on: enqueue
    while the parked run holds the resume lease, observe a defer (status
    back to PENDING, ``defer_count`` up, ``failure_count`` untouched), then
    complete on the next claim once the lease is free.
    """

    agent_id, owner_user_id = _create_agent()
    task_id = _create_waiting_task(agent_id=agent_id, owner_user_id=owner_user_id)

    attempts: list[int] = []

    async def executor(command: ClaimedTaskCommand) -> dict[str, Any]:
        attempts.append(command.attempt_count)
        if len(attempts) == 1:
            raise TaskCommandDeferred("the parked run still holds the resume lease")
        return {"delivered": True}

    register_external_task_input_executor(executor)

    db = _direct_db_session()
    try:
        enqueued = enqueue_task_command(
            db,
            task_id=task_id,
            actor_user_id=owner_user_id,
            command_id="ext-input-e2e",
            kind=TaskCommandKind.MESSAGE,
            payload={"scope": "external", "message": "the answer"},
        )
        assert enqueued.created
    finally:
        db.close()

    processed = await dispatch_one_task_command(
        websocket_api.execute_durable_task_command
    )
    assert processed is True

    db = _direct_db_session()
    try:
        row = (
            db.query(TaskExecutionCommand)
            .filter(TaskExecutionCommand.command_id == "ext-input-e2e")
            .one()
        )
        assert row.status == COMMAND_PENDING
        assert int(row.defer_count or 0) == 1
        assert int(row.failure_count or 0) == 0
        # The defer backoff parks the claim briefly; expire it so the second
        # dispatch below can claim without sleeping through real time.
        row.claim_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        db.commit()
    finally:
        db.close()

    processed = await dispatch_one_task_command(
        websocket_api.execute_durable_task_command
    )
    assert processed is True

    db = _direct_db_session()
    try:
        row = (
            db.query(TaskExecutionCommand)
            .filter(TaskExecutionCommand.command_id == "ext-input-e2e")
            .one()
        )
        assert row.status == COMMAND_COMPLETED
        assert row.result == {"delivered": True}
        assert int(row.failure_count or 0) == 0
    finally:
        db.close()

    assert attempts == [1, 2]


@pytest.mark.asyncio
async def test_exhausted_external_deferral_withholds_command_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An external audience never sees durable command identity.

    The terminal rebind in ``execute_durable_task_command`` runs *after* the
    executor's own draft, so the disclosure decision has to live in
    ``_terminal_command_event_draft`` itself: an exhausted external-scope
    deferral must persist ``include_command_identity=False`` (same policy as
    the external cancel) while still carrying the exception's own
    ``resend_safe`` evidence. The live ``agent_error`` frame follows the
    same rule: identity withheld and a purpose-built terminal sentence,
    pinned here because the exhausted-deferral path broadcasts through the
    same chokepoint.
    """

    from unittest.mock import AsyncMock, MagicMock

    agent_id, owner_user_id = _create_agent()
    task_id = _create_waiting_task(agent_id=agent_id, owner_user_id=owner_user_id)

    async def executor(_command: ClaimedTaskCommand) -> dict[str, Any]:
        raise TaskCommandDeferred(
            "the parked run still holds the resume lease",
            resend_safe=True,
        )

    register_external_task_input_executor(executor)

    broadcast_manager = MagicMock()
    broadcast_manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", broadcast_manager)

    db = _direct_db_session()
    try:
        enqueued = enqueue_task_command(
            db,
            task_id=task_id,
            actor_user_id=owner_user_id,
            command_id="ext-input-exhausted",
            kind=TaskCommandKind.MESSAGE,
            payload={"scope": "external", "message": "the answer"},
        )
        assert enqueued.created
        # Spend all but the last unit of the defer budget so the next
        # deferral is the terminal one -- max_command_defers() real round
        # trips would prove nothing more about the disposition under test.
        db.query(TaskExecutionCommand).filter(
            TaskExecutionCommand.command_id == "ext-input-exhausted"
        ).update(
            {TaskExecutionCommand.defer_count: max_command_defers() - 1},
            synchronize_session=False,
        )
        db.commit()
    finally:
        db.close()

    processed = await dispatch_one_task_command(
        websocket_api.execute_durable_task_command
    )
    assert processed is True

    payloads = [
        call.args[0] for call in broadcast_manager.broadcast_to_task.await_args_list
    ]
    assert [payload["type"] for payload in payloads] == ["agent_error"]
    (exhausted_payload,) = payloads
    assert exhausted_payload["message"] == EXTERNAL_INPUT_NOT_APPLIED_MESSAGE
    assert "command_id" not in exhausted_payload
    assert "command_kind" not in exhausted_payload

    db = _direct_db_session()
    try:
        row = (
            db.query(TaskExecutionCommand)
            .filter(TaskExecutionCommand.command_id == "ext-input-exhausted")
            .one()
        )
        assert row.status == COMMAND_FAILED
        event = (
            db.query(TaskCommandTerminalEvent)
            .filter(TaskCommandTerminalEvent.task_command_id == int(row.id))
            .one()
        )
        assert event.outcome == "failed"
        assert event.message_code == "task_command_deferred"
        assert event.resend_safe is True
        assert event.include_command_identity is False
    finally:
        db.close()


@pytest.mark.asyncio
async def test_defer_exhaustion_boundary_uses_the_effective_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both budget consumers must use the effective budget, not the constant.

    With a raised TTL the budget is 2*TTL, so the historical constant's
    boundary (defer count 60) must stay non-terminal: the wrapper must not
    broadcast and the transport must keep the row PENDING. At the effective
    boundary the command terminalizes -- and because an exhausted
    ``resend_safe=False`` deferral proves nothing about the downstream
    handoff, the broadcast must assert only uncertainty, never
    non-application.
    """

    from unittest.mock import AsyncMock, MagicMock

    monkeypatch.setenv("XAGENT_TASK_LEASE_TTL_SECONDS", "300")
    budget = max_command_defers()
    assert budget == 600

    agent_id, owner_user_id = _create_agent()
    task_id = _create_waiting_task(agent_id=agent_id, owner_user_id=owner_user_id)

    async def executor(_command: ClaimedTaskCommand) -> dict[str, Any]:
        raise TaskCommandDeferred("the delivery handoff is still pending")

    register_external_task_input_executor(executor)

    broadcast_manager = MagicMock()
    broadcast_manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", broadcast_manager)

    def seed_defer_count(count: int) -> None:
        # The command's own claim_expires_at, not a next-attempt column --
        # this transport has none -- gates whether a PENDING row is
        # claimable (``_claim_availability_predicate``). The first seed
        # relies on the fresh row's NULL default; the second clears the
        # one-second backoff the prior deferral armed so the row is
        # claimable again without sleeping through real time.
        db = _direct_db_session()
        try:
            db.query(TaskExecutionCommand).filter(
                TaskExecutionCommand.command_id == "ext-input-boundary"
            ).update(
                {
                    TaskExecutionCommand.defer_count: count,
                    TaskExecutionCommand.claim_expires_at: None,
                },
                synchronize_session=False,
            )
            db.commit()
        finally:
            db.close()

    db = _direct_db_session()
    try:
        enqueued = enqueue_task_command(
            db,
            task_id=task_id,
            actor_user_id=owner_user_id,
            command_id="ext-input-boundary",
            kind=TaskCommandKind.MESSAGE,
            payload={"scope": "external", "message": "the answer"},
        )
        assert enqueued.created
    finally:
        db.close()

    # The historical constant's boundary: one deferral past count 59 must
    # spend budget, not terminalize.
    seed_defer_count(MAX_COMMAND_DEFERS - 1)
    processed = await dispatch_one_task_command(
        websocket_api.execute_durable_task_command
    )
    assert processed is True
    broadcast_manager.broadcast_to_task.assert_not_awaited()

    db = _direct_db_session()
    try:
        row = (
            db.query(TaskExecutionCommand)
            .filter(TaskExecutionCommand.command_id == "ext-input-boundary")
            .one()
        )
        assert row.status == COMMAND_PENDING
        assert int(row.defer_count or 0) == MAX_COMMAND_DEFERS
        assert (
            db.query(TaskCommandTerminalEvent)
            .filter(TaskCommandTerminalEvent.task_command_id == int(row.id))
            .count()
            == 0
        )
    finally:
        db.close()

    # The effective boundary: the six-hundredth deferral is terminal, and
    # its broadcast asserts only uncertainty.
    seed_defer_count(budget - 1)
    processed = await dispatch_one_task_command(
        websocket_api.execute_durable_task_command
    )
    assert processed is True

    payloads = [
        call.args[0] for call in broadcast_manager.broadcast_to_task.await_args_list
    ]
    assert [payload["type"] for payload in payloads] == ["agent_error"]
    (exhausted_payload,) = payloads
    assert exhausted_payload["message"] == EXTERNAL_INPUT_UNCONFIRMED_MESSAGE
    assert "command_id" not in exhausted_payload
    assert "command_kind" not in exhausted_payload

    db = _direct_db_session()
    try:
        row = (
            db.query(TaskExecutionCommand)
            .filter(TaskExecutionCommand.command_id == "ext-input-boundary")
            .one()
        )
        assert row.status == COMMAND_FAILED
        event = (
            db.query(TaskCommandTerminalEvent)
            .filter(TaskCommandTerminalEvent.task_command_id == int(row.id))
            .one()
        )
        assert event.outcome == "failed"
        assert event.resend_safe is False
    finally:
        db.close()


@pytest.mark.asyncio
async def test_exhausted_generic_failure_reports_an_unconfirmed_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A generic failure proves nothing about the downstream handoff.

    The failure-budget exhaustion arm broadcasts through the same external
    branch; a worker may have injected the answer before the exception, so
    the audience must get the uncertainty sentence, never categorical
    non-application.
    """

    from unittest.mock import AsyncMock, MagicMock

    agent_id, owner_user_id = _create_agent()
    task_id = _create_waiting_task(agent_id=agent_id, owner_user_id=owner_user_id)

    async def executor(_command: ClaimedTaskCommand) -> dict[str, Any]:
        raise RuntimeError("the runtime broke mid-delivery")

    register_external_task_input_executor(executor)

    broadcast_manager = MagicMock()
    broadcast_manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", broadcast_manager)

    db = _direct_db_session()
    try:
        enqueued = enqueue_task_command(
            db,
            task_id=task_id,
            actor_user_id=owner_user_id,
            command_id="ext-input-generic-failure",
            kind=TaskCommandKind.MESSAGE,
            payload={"scope": "external", "message": "the answer"},
        )
        assert enqueued.created
        db.query(TaskExecutionCommand).filter(
            TaskExecutionCommand.command_id == "ext-input-generic-failure"
        ).update(
            {TaskExecutionCommand.failure_count: MAX_COMMAND_FAILURES - 1},
            synchronize_session=False,
        )
        db.commit()
    finally:
        db.close()

    processed = await dispatch_one_task_command(
        websocket_api.execute_durable_task_command
    )
    assert processed is True

    payloads = [
        call.args[0] for call in broadcast_manager.broadcast_to_task.await_args_list
    ]
    assert [payload["type"] for payload in payloads] == ["agent_error"]
    (failure_payload,) = payloads
    assert failure_payload["message"] == EXTERNAL_INPUT_UNCONFIRMED_MESSAGE
    assert "command_id" not in failure_payload
    assert "command_kind" not in failure_payload

    db = _direct_db_session()
    try:
        row = (
            db.query(TaskExecutionCommand)
            .filter(TaskExecutionCommand.command_id == "ext-input-generic-failure")
            .one()
        )
        assert row.status == COMMAND_FAILED
    finally:
        db.close()


@pytest.mark.asyncio
async def test_rejected_external_input_broadcasts_and_keeps_the_bound_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A terminal rejection must surface, and the executor's draft must win.

    The seam executor is the one MESSAGE handler with no notification
    channel of its own: without the broadcast, a deferred answer that is
    later terminally rejected vanishes silently while the task stays parked
    (xorbitsai/xagent-saas#952 B2). And the rejection disposition must
    persist the draft the executor bound -- hardcoding a neutral one
    silently discarded its identity-withholding (#952 M3) -- proven here
    through the real dispatch path, not by reading back the attribute.

    The bound draft uses ``message_code=None`` rather than the more obvious
    ``TASK_COMMAND_FAILED``: the fallback ``_terminal_command_event_draft``
    derives exactly ``task_command_failed``/``resend_safe=False``/identity
    withheld for this exact command (a rejection on an external-scope
    MESSAGE), so a fallback-equivalent bound value would pass even if
    precedence were broken and the fallback ran instead. ``None`` is a value
    ``TerminalTaskEventDraft`` accepts but the fallback never produces, so
    only the bound draft winning can explain the assertion below.
    """

    from unittest.mock import AsyncMock, MagicMock

    from xagent.web.services.task_command_terminal_events import (
        TerminalTaskEventDraft,
        bind_terminal_event_draft,
    )

    agent_id, owner_user_id = _create_agent()
    task_id = _create_waiting_task(agent_id=agent_id, owner_user_id=owner_user_id)

    async def executor(_command: ClaimedTaskCommand) -> dict[str, Any]:
        rejection = TaskCommandRejected(
            "the continuation was refused", reason="continuation_refused"
        )
        bind_terminal_event_draft(
            rejection,
            TerminalTaskEventDraft(
                message_code=None,
                resend_safe=False,
                include_command_identity=False,
            ),
        )
        raise rejection

    register_external_task_input_executor(executor)

    broadcast_manager = MagicMock()
    broadcast_manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", broadcast_manager)

    db = _direct_db_session()
    try:
        enqueued = enqueue_task_command(
            db,
            task_id=task_id,
            actor_user_id=owner_user_id,
            command_id="ext-input-rejected",
            kind=TaskCommandKind.MESSAGE,
            payload={"scope": "external", "message": "the answer"},
        )
        assert enqueued.created
    finally:
        db.close()

    processed = await dispatch_one_task_command(
        websocket_api.execute_durable_task_command
    )
    assert processed is True

    payloads = [
        call.args[0] for call in broadcast_manager.broadcast_to_task.await_args_list
    ]
    assert [payload["type"] for payload in payloads] == ["agent_error"]

    (rejection_payload,) = payloads
    # The live frame mirrors the persisted-event disclosure rule: the
    # external audience never sees durable command identity, and the generic
    # fallback's "Please try again." would be false for the non-retryable
    # rejections this broadcast exists to surface.
    assert rejection_payload["message"] == EXTERNAL_INPUT_NOT_APPLIED_MESSAGE
    assert "command_id" not in rejection_payload
    assert "command_kind" not in rejection_payload

    db = _direct_db_session()
    try:
        row = (
            db.query(TaskExecutionCommand)
            .filter(TaskExecutionCommand.command_id == "ext-input-rejected")
            .one()
        )
        assert row.status == COMMAND_FAILED
        event = (
            db.query(TaskCommandTerminalEvent)
            .filter(TaskCommandTerminalEvent.task_command_id == int(row.id))
            .one()
        )
        assert event.outcome == "failed"
        assert event.message_code is None
        assert event.include_command_identity is False
    finally:
        db.close()


@pytest.mark.asyncio
async def test_rejected_external_input_without_a_bound_draft_gets_the_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An executor that binds nothing still gets a classified terminal event.

    The rejection arm derives the standard draft only when the executor
    bound none, so the fallback must classify the outcome
    (``task_command_failed``) and withhold identity for the external scope
    -- the other half of the precedence contract pinned above.
    """

    from unittest.mock import AsyncMock, MagicMock

    agent_id, owner_user_id = _create_agent()
    task_id = _create_waiting_task(agent_id=agent_id, owner_user_id=owner_user_id)

    async def executor(_command: ClaimedTaskCommand) -> dict[str, Any]:
        raise TaskCommandRejected(
            "the continuation was refused", reason="continuation_refused"
        )

    register_external_task_input_executor(executor)

    broadcast_manager = MagicMock()
    broadcast_manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", broadcast_manager)

    db = _direct_db_session()
    try:
        enqueued = enqueue_task_command(
            db,
            task_id=task_id,
            actor_user_id=owner_user_id,
            command_id="ext-input-rejected-unbound",
            kind=TaskCommandKind.MESSAGE,
            payload={"scope": "external", "message": "the answer"},
        )
        assert enqueued.created
    finally:
        db.close()

    processed = await dispatch_one_task_command(
        websocket_api.execute_durable_task_command
    )
    assert processed is True

    payloads = [
        call.args[0] for call in broadcast_manager.broadcast_to_task.await_args_list
    ]
    assert [payload["type"] for payload in payloads] == ["agent_error"]
    (rejection_payload,) = payloads
    assert rejection_payload["message"] == EXTERNAL_INPUT_NOT_APPLIED_MESSAGE

    db = _direct_db_session()
    try:
        row = (
            db.query(TaskExecutionCommand)
            .filter(TaskExecutionCommand.command_id == "ext-input-rejected-unbound")
            .one()
        )
        assert row.status == COMMAND_FAILED
        event = (
            db.query(TaskCommandTerminalEvent)
            .filter(TaskCommandTerminalEvent.task_command_id == int(row.id))
            .one()
        )
        assert event.outcome == "failed"
        assert event.message_code == "task_command_failed"
        assert event.include_command_identity is False
    finally:
        db.close()


@pytest.mark.asyncio
async def test_rejection_broadcast_failure_does_not_supersede_the_disposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed broadcast must not turn a terminal rejection into a retry.

    ``broadcast_to_task`` re-raises non-connection errors (its control-state
    snapshot read can fail), and an exception escaping the rejection arm
    would supersede ``TaskCommandRejected``: the dispatcher's generic
    disposition would then spend the failure budget on a command whose
    outcome is already durably decided, and the executor-bound draft would
    be lost with it. The broadcast is fire-and-forget for the disposition:
    one dispatch must leave the command terminally failed with the bound
    draft persisted.
    """

    from unittest.mock import AsyncMock, MagicMock

    from xagent.web.services.task_command_terminal_events import (
        TerminalTaskEventDraft,
        bind_terminal_event_draft,
    )

    agent_id, owner_user_id = _create_agent()
    task_id = _create_waiting_task(agent_id=agent_id, owner_user_id=owner_user_id)

    async def executor(_command: ClaimedTaskCommand) -> dict[str, Any]:
        rejection = TaskCommandRejected(
            "the continuation was refused", reason="continuation_refused"
        )
        bind_terminal_event_draft(
            rejection,
            TerminalTaskEventDraft(
                message_code=None,
                resend_safe=False,
                include_command_identity=False,
            ),
        )
        raise rejection

    register_external_task_input_executor(executor)

    broadcast_manager = MagicMock()
    broadcast_manager.broadcast_to_task = AsyncMock(
        side_effect=Exception("the control-state snapshot read failed")
    )
    monkeypatch.setattr(websocket_api, "manager", broadcast_manager)

    db = _direct_db_session()
    try:
        enqueued = enqueue_task_command(
            db,
            task_id=task_id,
            actor_user_id=owner_user_id,
            command_id="ext-input-broadcast-broke",
            kind=TaskCommandKind.MESSAGE,
            payload={"scope": "external", "message": "the answer"},
        )
        assert enqueued.created
    finally:
        db.close()

    processed = await dispatch_one_task_command(
        websocket_api.execute_durable_task_command
    )
    assert processed is True
    broadcast_manager.broadcast_to_task.assert_awaited()

    db = _direct_db_session()
    try:
        row = (
            db.query(TaskExecutionCommand)
            .filter(TaskExecutionCommand.command_id == "ext-input-broadcast-broke")
            .one()
        )
        assert row.status == COMMAND_FAILED
        event = (
            db.query(TaskCommandTerminalEvent)
            .filter(TaskCommandTerminalEvent.task_command_id == int(row.id))
            .one()
        )
        assert event.outcome == "failed"
        assert event.message_code is None
        assert event.include_command_identity is False
    finally:
        db.close()
