"""``settle_task_interaction``: a host settles a waiting task without a client."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from xagent.web.api import websocket as ws_api
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.models.user import User
from xagent.web.services import task_command_execution as execution
from xagent.web.services import task_command_transport as transport
from xagent.web.services.task_event_display import project_event_message
from xagent.web.services.task_execution import (
    ClientVisibleError,
    ClientVisibleValidationError,
)
from xagent.web.services.task_execution_event_writer import append_fact_no_commit
from xagent.web.services.task_interaction_settle import (
    SettleOutcome,
    settle_task_interaction,
    task_waits_on_interaction_sync,
)

from .conftest import _admin_headers, _direct_db_session
from .test_conversation_logs import _create_task_row, _user_id

pytestmark = pytest.mark.usefixtures("_test_db")


@pytest.fixture(autouse=True)
def _admin_exists(_test_db):
    _admin_headers()


@pytest.fixture
def dispatched(monkeypatch):
    calls = []

    async def fake(handler, *, command_db_id):
        calls.append(command_db_id)

    monkeypatch.setattr(transport, "dispatch_task_command_promptly", fake)
    return calls


def _owner() -> User:
    with _direct_db_session() as db:
        user = db.query(User).filter(User.id == _user_id("admin")).one()
        db.expunge(user)
        return user


STAMP = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
RUN_ID = "run-1"


def _task(
    status: TaskStatus,
    *,
    waiting_on: list[str] | None = None,
    storage_version: int = 1,
) -> int:
    """A task row; ``waiting_on`` records a waiting message naming those ids."""
    task_id = _create_task_row(user_id=_user_id("admin"), title="Connect")
    with _direct_db_session() as db:
        db.query(Task).filter(Task.id == task_id).update(
            {
                Task.status: status,
                Task.conversation_storage_version: storage_version,
                Task.run_id: RUN_ID,
            }
        )
        if waiting_on is not None:
            data = {
                "message": "Connect your account",
                "message_type": "question",
                "expect_response": True,
                "metadata": {"interaction_ids": waiting_on},
            }
            if storage_version == 2:
                append_fact_no_commit(
                    db,
                    task_id=task_id,
                    kind="agent_message",
                    key="outbound:w1",
                    run_id=RUN_ID,
                    payload={"data": data, "protocol_event_id": "w1"},
                    occurred_at=STAMP,
                )
            else:
                db.add(
                    TraceEvent(
                        task_id=task_id,
                        event_id="w1",
                        event_type="agent_message",
                        timestamp=STAMP,
                        data=data,
                    )
                )
        db.commit()
    return task_id


def _waiting(*interaction_ids: str, storage_version: int = 1) -> int:
    return _task(
        TaskStatus.WAITING_FOR_USER,
        waiting_on=list(interaction_ids),
        storage_version=storage_version,
    )


def _commands(task_id: int):
    with _direct_db_session() as db:
        return [
            (row.command_id, dict(row.payload))
            for row in db.query(TaskExecutionCommand).filter_by(task_id=task_id)
        ]


@pytest.mark.parametrize("storage_version", [1, 2])
async def test_a_waiting_task_queues_one_hidden_message_with_no_websocket(
    dispatched, storage_version
) -> None:
    task_id = _waiting("abc-123", storage_version=storage_version)
    outcome = await settle_task_interaction(
        task_id=task_id, actor_user=_owner(), interaction_id="abc-123"
    )
    assert outcome is SettleOutcome.QUEUED
    [(command_id, payload)] = _commands(task_id)
    assert command_id.startswith("settle:")
    assert payload["message"] == "[interaction settled]"
    assert payload["message_metadata"] == {
        "display": "none",
        "kind": "interaction_settled",
        "interaction_id": "abc-123",
    }
    assert len(dispatched) == 1


@pytest.mark.parametrize("interaction_id", ["x", "a" * 64, "Tool_call-0123456789"])
async def test_every_valid_interaction_id_fits_a_command_id(
    dispatched, interaction_id
) -> None:
    task_id = _waiting(interaction_id)
    outcome = await settle_task_interaction(
        task_id=task_id, actor_user=_owner(), interaction_id=interaction_id
    )
    assert outcome is SettleOutcome.QUEUED
    [(command_id, _)] = _commands(task_id)
    assert transport.COMMAND_ID_PATTERN.fullmatch(command_id)


async def test_the_host_chooses_kind_and_text(dispatched) -> None:
    task_id = _waiting("i1")
    await settle_task_interaction(
        task_id=task_id,
        actor_user=_owner(),
        interaction_id="i1",
        kind="connector_settled",
        text="[connector settled]",
    )
    [(_, payload)] = _commands(task_id)
    assert payload["message"] == "[connector settled]"
    assert payload["message_metadata"]["kind"] == "connector_settled"


async def test_a_repeated_call_is_a_replay(dispatched) -> None:
    task_id = _waiting("same")
    first = await settle_task_interaction(
        task_id=task_id, actor_user=_owner(), interaction_id="same"
    )
    again = await settle_task_interaction(
        task_id=task_id, actor_user=_owner(), interaction_id="same"
    )
    assert (first, again) == (SettleOutcome.QUEUED, SettleOutcome.DUPLICATE)
    assert len(_commands(task_id)) == 1
    assert len(dispatched) == 1


def _set_status(task_id: int, status: TaskStatus) -> None:
    with _direct_db_session() as db:
        db.query(Task).filter(Task.id == task_id).update({Task.status: status})
        db.commit()


def _finish_commands(task_id: int, status: str) -> None:
    with _direct_db_session() as db:
        db.query(TaskExecutionCommand).filter_by(task_id=task_id).update(
            {TaskExecutionCommand.status: status}
        )
        db.commit()


async def test_a_repeat_after_the_task_resumed_is_still_a_duplicate(dispatched) -> None:
    task_id = _waiting("same")
    await settle_task_interaction(
        task_id=task_id, actor_user=_owner(), interaction_id="same"
    )
    _set_status(task_id, TaskStatus.RUNNING)
    outcome = await settle_task_interaction(
        task_id=task_id, actor_user=_owner(), interaction_id="same"
    )
    assert outcome is SettleOutcome.DUPLICATE


async def test_a_settle_that_ran_without_resuming_the_task_can_be_retried(
    dispatched,
) -> None:
    """The first command was dropped or failed: its id must not stay taken."""
    task_id = _waiting("same")
    await settle_task_interaction(
        task_id=task_id, actor_user=_owner(), interaction_id="same"
    )
    _finish_commands(task_id, "completed")  # dropped at execution, still waiting
    retry = await settle_task_interaction(
        task_id=task_id, actor_user=_owner(), interaction_id="same"
    )
    assert retry is SettleOutcome.QUEUED
    _finish_commands(task_id, "failed")
    again = await settle_task_interaction(
        task_id=task_id, actor_user=_owner(), interaction_id="same"
    )
    assert again is SettleOutcome.QUEUED
    ids = [command_id for command_id, _ in _commands(task_id)]
    assert len(set(ids)) == 3 == len(dispatched)


async def test_a_retry_while_a_settle_is_still_queued_is_a_duplicate(
    dispatched,
) -> None:
    task_id = _waiting("same")
    await settle_task_interaction(
        task_id=task_id, actor_user=_owner(), interaction_id="same"
    )
    _finish_commands(task_id, "pending")
    retry = await settle_task_interaction(
        task_id=task_id, actor_user=_owner(), interaction_id="same"
    )
    assert retry is SettleOutcome.DUPLICATE
    assert len(_commands(task_id)) == 1


@pytest.mark.parametrize("status", [TaskStatus.COMPLETED, TaskStatus.RUNNING])
async def test_a_task_that_is_not_waiting_queues_nothing(dispatched, status) -> None:
    task_id = _task(status, waiting_on=["x1"])
    outcome = await settle_task_interaction(
        task_id=task_id, actor_user=_owner(), interaction_id="x1"
    )
    assert outcome is SettleOutcome.NOT_WAITING
    assert _commands(task_id) == []
    assert dispatched == []


@pytest.mark.parametrize("storage_version", [1, 2])
async def test_a_settle_for_another_interaction_queues_nothing(
    dispatched, storage_version
) -> None:
    """The task waits on Y; a settle for X must not touch it."""
    task_id = _waiting("question-y", storage_version=storage_version)
    outcome = await settle_task_interaction(
        task_id=task_id, actor_user=_owner(), interaction_id="question-x"
    )
    assert outcome is SettleOutcome.NOT_WAITING
    assert _commands(task_id) == []
    assert dispatched == []


async def test_a_batch_of_requests_can_be_settled_one_by_one(dispatched) -> None:
    task_id = _waiting("tool-a", "tool-b")
    first = await settle_task_interaction(
        task_id=task_id, actor_user=_owner(), interaction_id="tool-b"
    )
    assert first is SettleOutcome.QUEUED
    [(_, payload)] = _commands(task_id)
    assert payload["message_metadata"]["interaction_id"] == "tool-b"


async def test_a_wait_without_recorded_ids_cannot_be_settled(dispatched) -> None:
    """A plain question (or a wait from before ids were recorded)."""
    plain = _task(TaskStatus.WAITING_FOR_USER, waiting_on=[])
    never_recorded = _task(TaskStatus.WAITING_FOR_USER)
    for task_id in (plain, never_recorded):
        outcome = await settle_task_interaction(
            task_id=task_id, actor_user=_owner(), interaction_id="x1"
        )
        assert outcome is SettleOutcome.NOT_WAITING
        assert _commands(task_id) == []


async def test_only_the_latest_waiting_message_counts(dispatched) -> None:
    task_id = _waiting("old-question")
    with _direct_db_session() as db:
        db.add(
            TraceEvent(
                task_id=task_id,
                event_id="w2",
                event_type="agent_message",
                timestamp=STAMP,
                data={
                    "expect_response": True,
                    "metadata": {"interaction_ids": ["new-question"]},
                },
            )
        )
        db.commit()
    assert not task_waits_on_interaction_sync(task_id, "old-question")
    assert task_waits_on_interaction_sync(task_id, "new-question")


async def test_a_waiting_question_behind_many_result_cards_is_still_found(
    dispatched,
) -> None:
    """Cards are agent messages that ask nothing; they must not push the
    waiting question out of the lookup (concurrent DAG steps can emit many)."""
    task_id = _waiting("the-question")
    with _direct_db_session() as db:
        for n in range(25):
            db.add(
                TraceEvent(
                    task_id=task_id,
                    event_id=f"card-{n}",
                    event_type="agent_message",
                    timestamp=STAMP,
                    data={
                        "message": f"Card {n}",
                        "message_type": "result_card",
                        "expect_response": False,
                        "metadata": {"display": "result_card"},
                    },
                )
            )
        db.commit()
    assert task_waits_on_interaction_sync(task_id, "the-question")
    assert not task_waits_on_interaction_sync(task_id, "another")


async def test_only_the_owner_or_an_admin_may_settle(dispatched) -> None:
    task_id = _task(TaskStatus.WAITING_FOR_USER)
    stranger = SimpleNamespace(id=987654, is_admin=False)
    with pytest.raises(ClientVisibleError):
        await settle_task_interaction(
            task_id=task_id, actor_user=stranger, interaction_id="x1"
        )
    with pytest.raises(ClientVisibleValidationError):
        await settle_task_interaction(
            task_id=987655, actor_user=_owner(), interaction_id="x1"
        )
    assert _commands(task_id) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"interaction_id": "bad id"},
        {"interaction_id": "x" * 65},
        {"interaction_id": "ok", "kind": "Bad Kind"},
        {"interaction_id": "ok", "text": "  "},
        {"interaction_id": "ok", "text": "y" * 201},
    ],
)
async def test_malformed_arguments_are_refused(dispatched, kwargs) -> None:
    task_id = _task(TaskStatus.WAITING_FOR_USER)
    with pytest.raises(ValueError):
        await settle_task_interaction(task_id=task_id, actor_user=_owner(), **kwargs)
    assert _commands(task_id) == []


async def _enqueue_frame(task_id: int, frame: dict):
    return await ws_api._enqueue_websocket_task_command(
        task_id=task_id,
        message_data={"user": _owner(), **frame},
        kind=transport.TaskCommandKind.MESSAGE,
    )


async def test_a_client_frame_cannot_set_message_metadata(dispatched) -> None:
    task_id = _task(TaskStatus.WAITING_FOR_USER)
    with pytest.raises(ClientVisibleValidationError):
        await _enqueue_frame(
            task_id,
            {
                "message": "hi",
                "message_metadata": {"display": "none", "kind": "interaction_settled"},
            },
        )
    assert _commands(task_id) == []


async def test_an_old_frame_with_metadata_still_works_and_stores_none(
    dispatched,
) -> None:
    task_id = _task(TaskStatus.WAITING_FOR_USER)
    enqueued = await _enqueue_frame(
        task_id,
        {
            "message": "hi",
            "client_message_id": "c-1",
            "metadata": {"display": "none", "kind": "x", "request_id": "r"},
        },
    )
    assert enqueued is not None and enqueued.created
    [(_, payload)] = _commands(task_id)
    assert "metadata" not in payload and "message_metadata" not in payload
    assert payload["message"] == "hi"


def _settle_frame(interaction_id: str = "z") -> dict:
    return {
        "message": "[interaction settled]",
        "user": _owner(),
        "client_message_id": f"settle:{interaction_id}",
        "message_metadata": {
            "display": "none",
            "kind": "interaction_settled",
            "interaction_id": interaction_id,
        },
    }


async def test_execution_applies_a_settle_for_the_open_interaction(
    monkeypatch,
) -> None:
    task_id = _waiting("z")
    prepare = MagicMock(side_effect=RuntimeError("reached"))
    monkeypatch.setattr(execution, "_prepare_task_message_sync", prepare)
    with pytest.raises(RuntimeError, match="reached"):
        await execution.handle_task_message(AsyncMock(), task_id, _settle_frame())
    prepare.assert_called_once()
    assert prepare.call_args.kwargs["message_metadata"] == {
        "display": "none",
        "kind": "interaction_settled",
        "interaction_id": "z",
    }


async def test_execution_drops_a_settle_when_the_task_is_no_longer_waiting(
    monkeypatch,
) -> None:
    task_id = _task(TaskStatus.RUNNING, waiting_on=["z"])  # a real answer won
    prepare = MagicMock(side_effect=AssertionError("must not run"))
    monkeypatch.setattr(execution, "_prepare_task_message_sync", prepare)
    await execution.handle_task_message(AsyncMock(), task_id, _settle_frame())
    prepare.assert_not_called()
    with _direct_db_session() as db:
        assert db.query(TaskChatMessage).filter_by(task_id=task_id).count() == 0


async def test_execution_drops_a_settle_for_another_interaction_untouched(
    monkeypatch,
) -> None:
    """Host settles X while the task waits on Y: nothing is injected, the live
    question is not closed and no message row appears, so Y's real answer is
    still the only thing that resumes the task."""
    task_id = _waiting("question-y")
    prepare = MagicMock(side_effect=AssertionError("must not run"))
    monkeypatch.setattr(execution, "_prepare_task_message_sync", prepare)
    close = MagicMock(side_effect=AssertionError("must not close the live question"))
    monkeypatch.setattr(execution, "close_legacy_resume_interaction_sync", close)
    await execution.handle_task_message(
        AsyncMock(), task_id, _settle_frame("question-x")
    )
    prepare.assert_not_called()
    close.assert_not_called()
    with _direct_db_session() as db:
        assert db.query(TaskChatMessage).filter_by(task_id=task_id).count() == 0
        task = db.query(Task).filter(Task.id == task_id).one()
        assert task.status == TaskStatus.WAITING_FOR_USER
        assert task.input is None or "settled" not in str(task.input)


def test_event_backed_replay_carries_the_settle_metadata() -> None:
    task_id = _task(TaskStatus.RUNNING, storage_version=2)
    with _direct_db_session() as db:
        fact = append_fact_no_commit(
            db,
            task_id=task_id,
            kind="input_accepted",
            key="in-1",
            run_id=RUN_ID,
            turn_id="t-1",
            payload={
                "content": "[interaction settled]",
                "message_type": "user",
                "message_metadata": {
                    "display": "none",
                    "kind": "interaction_settled",
                    "interaction_id": "z",
                },
            },
            occurred_at=STAMP,
        )
        db.commit()
        event = db.query(TaskExecutionEvent).filter_by(id=fact.id).one()
        projected = project_event_message(event)
    assert projected["message_metadata"]["interaction_id"] == "z"


async def test_a_client_frame_cannot_use_the_settle_command_prefix(dispatched) -> None:
    task_id = _waiting("z")
    with pytest.raises(ClientVisibleValidationError):
        await ws_api._enqueue_websocket_task_command(
            task_id=task_id,
            message_data={"user": _owner(), "message": "hi"},
            kind=transport.TaskCommandKind.MESSAGE,
            command_id="settle:abcdef",
        )
    assert _commands(task_id) == []
    # The host's own settle is unaffected.
    outcome = await settle_task_interaction(
        task_id=task_id, actor_user=_owner(), interaction_id="z"
    )
    assert outcome is SettleOutcome.QUEUED


async def test_a_retried_settle_after_the_task_left_waiting_still_reaches_prepare(
    monkeypatch,
) -> None:
    """Attempt 1 injected the settle (a delivery row exists, the task is no
    longer waiting) and the worker died before marking it dispatched. The retry
    must not be dropped by the interaction guard: reconciling the delivery
    happens in ``_prepare_task_message_sync``."""
    task_id = _task(TaskStatus.RUNNING, waiting_on=["z"])
    with _direct_db_session() as db:
        db.add(
            TaskChatMessage(
                task_id=task_id,
                user_id=_user_id("admin"),
                role="user",
                content="[interaction settled]",
                message_type="user_message",
                turn_id="settle:z",
                delivery_status="pending",
            )
        )
        db.commit()
    prepare = MagicMock(side_effect=RuntimeError("reached"))
    monkeypatch.setattr(execution, "_prepare_task_message_sync", prepare)
    with pytest.raises(RuntimeError, match="reached"):
        await execution.handle_task_message(AsyncMock(), task_id, _settle_frame())
    prepare.assert_called_once()
