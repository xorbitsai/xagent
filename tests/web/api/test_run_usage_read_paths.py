"""The usage stored on an answer row never leaves the server.

``message_metadata["run_usage"]`` is written for the embedding application to
read from the database. The replay, the conversation log detail and the model
context all read the same rows and must not carry it.
"""

import json
from typing import Any

import pytest

from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task
from xagent.web.services.chat_history_service import (
    persist_assistant_message_no_commit,
)
from xagent.web.services.task_conversation_context_service import (
    load_task_conversation_context_sync,
)
from xagent.web.services.turn_settlement import (
    TurnModelUsage,
    TurnUsage,
    turn_usage_payload,
)

from .conftest import _admin_headers, _direct_db_session, client
from .test_conversation_logs import _create_task_row, _user_id

pytestmark = pytest.mark.usefixtures("_test_db")

ANSWER = "the answer text"
RUN_USAGE = {
    "run_usage": turn_usage_payload(
        TurnUsage(17, 6, 2, 0, 0, (TurnModelUsage("7", "model-a", 17, 6),))
    )
}


def _task_with_answer(storage_version: int) -> tuple[int, int]:
    user_id = _user_id("admin")
    task_id = _create_task_row(
        user_id=user_id,
        title="Usage",
        output_text=ANSWER,
        source="sdk",
        is_visible=False,
    )
    with _direct_db_session() as db:
        db.query(Task).filter(Task.id == task_id).update(
            {Task.conversation_storage_version: storage_version, Task.run_id: "run-1"}
        )
        db.commit()
        if storage_version == 1:
            db.add(
                TaskChatMessage(
                    task_id=task_id,
                    user_id=user_id,
                    role="user",
                    content="question",
                    message_type="chat",
                    turn_id="turn-1",
                )
            )
        persist_assistant_message_no_commit(
            db,
            task_id,
            user_id,
            ANSWER,
            message_type="assistant_response",
            turn_id="turn-1",
            source_event_id="answer-1",
            message_metadata=RUN_USAGE,
        )
        db.commit()
        stored = (
            db.query(TaskChatMessage)
            .filter(
                TaskChatMessage.task_id == task_id, TaskChatMessage.role == "assistant"
            )
            .one()
        )
        assert stored.message_metadata == RUN_USAGE
    return user_id, task_id


def _replayed_events(task_id: int, user_id: int) -> list[dict[str, Any]]:
    from xagent.web.api.websocket import _load_historical_stream_snapshot_sync

    snapshot = _load_historical_stream_snapshot_sync(
        task_id, actor_user_id=user_id, actor_is_admin=True
    )
    assert snapshot is not None
    return list(snapshot.events)


def _output_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


@pytest.mark.parametrize(
    "surface",
    ["web_replay_v1", "web_replay_v2", "conversation_log_detail", "model_context"],
)
def test_run_usage_never_leaves_server(surface: str) -> None:
    headers = _admin_headers()
    version = 1 if surface == "web_replay_v1" else 2
    user_id, task_id = _task_with_answer(version)

    output: Any
    if surface in ("web_replay_v1", "web_replay_v2"):
        output = _replayed_events(task_id, user_id)
    elif surface == "conversation_log_detail":
        response = client.get(f"/api/conversation-logs/{task_id}", headers=headers)
        assert response.status_code == 200, response.text
        output = response.text
    else:
        with _direct_db_session() as db:
            output = load_task_conversation_context_sync(db, task_id)

    text = _output_text(output)
    # The surface really read the answer row.
    assert ANSWER in text
    assert "run_usage" not in text
