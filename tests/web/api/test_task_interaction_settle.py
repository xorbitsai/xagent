"""``settle_task_interaction``: a host settles a waiting task without a client."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from xagent.web.api import websocket as ws_api
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services import task_command_execution as execution
from xagent.web.services import task_command_transport as transport
from xagent.web.services.task_execution import (
    ClientVisibleError,
    ClientVisibleValidationError,
)
from xagent.web.services.task_interaction_settle import (
    SettleOutcome,
    settle_task_interaction,
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


def _task(status: TaskStatus) -> int:
    task_id = _create_task_row(user_id=_user_id("admin"), title="Connect")
    with _direct_db_session() as db:
        db.query(Task).filter(Task.id == task_id).update({Task.status: status})
        db.commit()
    return task_id


def _commands(task_id: int):
    with _direct_db_session() as db:
        return [
            (row.command_id, dict(row.payload))
            for row in db.query(TaskExecutionCommand).filter_by(task_id=task_id)
        ]


async def test_a_waiting_task_queues_one_hidden_message_with_no_websocket(
    dispatched,
) -> None:
    task_id = _task(TaskStatus.WAITING_FOR_USER)
    outcome = await settle_task_interaction(
        task_id=task_id, actor_user=_owner(), interaction_id="abc-123"
    )
    assert outcome is SettleOutcome.QUEUED
    [(command_id, payload)] = _commands(task_id)
    assert command_id == "settle:abc-123"
    assert payload["message"] == "[interaction settled]"
    assert payload["message_metadata"] == {
        "hidden": True,
        "kind": "interaction_settled",
        "interaction_id": "abc-123",
    }
    assert len(dispatched) == 1


async def test_the_host_chooses_kind_and_text(dispatched) -> None:
    task_id = _task(TaskStatus.WAITING_FOR_USER)
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
    task_id = _task(TaskStatus.WAITING_FOR_USER)
    first = await settle_task_interaction(
        task_id=task_id, actor_user=_owner(), interaction_id="same"
    )
    again = await settle_task_interaction(
        task_id=task_id, actor_user=_owner(), interaction_id="same"
    )
    assert (first, again) == (SettleOutcome.QUEUED, SettleOutcome.DUPLICATE)
    assert len(_commands(task_id)) == 1
    assert len(dispatched) == 1


async def test_a_repeat_after_the_task_resumed_is_still_a_duplicate(dispatched) -> None:
    task_id = _task(TaskStatus.WAITING_FOR_USER)
    await settle_task_interaction(
        task_id=task_id, actor_user=_owner(), interaction_id="same"
    )
    with _direct_db_session() as db:
        db.query(Task).filter(Task.id == task_id).update(
            {Task.status: TaskStatus.RUNNING}
        )
        db.commit()
    outcome = await settle_task_interaction(
        task_id=task_id, actor_user=_owner(), interaction_id="same"
    )
    assert outcome is SettleOutcome.DUPLICATE


@pytest.mark.parametrize("status", [TaskStatus.COMPLETED, TaskStatus.RUNNING])
async def test_a_task_that_is_not_waiting_queues_nothing(dispatched, status) -> None:
    task_id = _task(status)
    outcome = await settle_task_interaction(
        task_id=task_id, actor_user=_owner(), interaction_id="x1"
    )
    assert outcome is SettleOutcome.NOT_WAITING
    assert _commands(task_id) == []
    assert dispatched == []


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
                "message_metadata": {"hidden": True, "kind": "interaction_settled"},
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
            "metadata": {"hidden": True, "kind": "x", "request_id": "r"},
        },
    )
    assert enqueued is not None and enqueued.created
    [(_, payload)] = _commands(task_id)
    assert "metadata" not in payload and "message_metadata" not in payload
    assert payload["message"] == "hi"


def _settle_frame() -> dict:
    return {
        "message": "[interaction settled]",
        "user": _owner(),
        "client_message_id": "settle:z",
        "message_metadata": {
            "hidden": True,
            "kind": "interaction_settled",
            "interaction_id": "z",
        },
    }


async def test_execution_applies_a_settle_while_the_task_is_waiting(
    monkeypatch,
) -> None:
    task_id = _task(TaskStatus.WAITING_FOR_USER)
    prepare = MagicMock(side_effect=RuntimeError("reached"))
    monkeypatch.setattr(execution, "_prepare_task_message_sync", prepare)
    with pytest.raises(RuntimeError, match="reached"):
        await execution.handle_task_message(AsyncMock(), task_id, _settle_frame())
    prepare.assert_called_once()


async def test_execution_drops_a_settle_when_the_task_is_no_longer_waiting(
    monkeypatch,
) -> None:
    task_id = _task(TaskStatus.RUNNING)  # a real answer won the race
    prepare = MagicMock(side_effect=AssertionError("must not run"))
    monkeypatch.setattr(execution, "_prepare_task_message_sync", prepare)
    await execution.handle_task_message(AsyncMock(), task_id, _settle_frame())
    prepare.assert_not_called()
    with _direct_db_session() as db:
        assert db.query(TaskChatMessage).filter_by(task_id=task_id).count() == 0
