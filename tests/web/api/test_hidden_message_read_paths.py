"""Read paths that must not show a hidden message (a client's resume signal).

The first-party chat socket replays hidden messages with their ``metadata`` so
its own client can hide them. Surfaces read by owners auditing a conversation
or by SDK consumers (whose client may predate the flag) omit them instead.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from xagent.web.api.v1._step_mapping import map_trace_events_to_public_steps
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.services.message_metadata import is_display_none_user_event

from .conftest import _admin_headers, _direct_db_session, client
from .test_agent_preview_history import create_agent
from .test_conversation_logs import (
    _attach_trigger_run,
    _create_agent_row,
    _create_task_row,
    _user_id,
)

pytestmark = pytest.mark.usefixtures("_test_db")

HIDDEN = {"display": "none", "kind": "connector_settled", "interaction_id": "i-1"}
STAMP = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _event(event_type: str, data: dict) -> SimpleNamespace:
    return SimpleNamespace(
        event_type=event_type,
        data=data,
        step_id=None,
        task_id="t1",
        event_id=f"evt-{event_type}-{data.get('message') or data.get('content')}",
        timestamp=STAMP,
    )


def test_is_display_none_user_event_needs_a_user_message_with_the_flag() -> None:
    assert is_display_none_user_event("user_message", {"metadata": HIDDEN})
    assert not is_display_none_user_event("user_message", {"metadata": {"kind": "x"}})
    assert not is_display_none_user_event("user_message", {"message": "hi"})
    assert not is_display_none_user_event("user_message", None)
    assert not is_display_none_user_event("ai_message", {"metadata": HIDDEN})


def test_public_steps_omit_hidden_user_messages_but_keep_the_reply() -> None:
    steps = map_trace_events_to_public_steps(
        [
            _event("user_message", {"message": "hello"}),
            _event(
                "user_message",
                {"message": "[connector settled]", "metadata": HIDDEN},
            ),
            _event("ai_message", {"content": "done"}),
        ]
    )
    assert [(s["data"]["role"], s["data"]["content"]) for s in steps] == [
        ("user", "hello"),
        ("assistant", "done"),
    ]


def test_conversation_log_detail_omits_hidden_messages_and_their_trace_twin() -> None:
    headers = _admin_headers()
    user_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=user_id, name="Audit Agent")
    task_id = _create_task_row(
        user_id=user_id,
        title="Connect",
        source="trigger",
        is_visible=False,
        agent_id=agent_id,
        agent_config={"trigger_type": "webhook"},
    )
    _attach_trigger_run(
        user_id=user_id,
        agent_id=agent_id,
        task_id=task_id,
        trigger_type="webhook",
        source_event_id="evt-1",
    )
    with _direct_db_session() as db:
        for role, content, metadata in (
            ("user", "hello", None),
            ("user", "[connector settled]", HIDDEN),
            ("assistant", "connected", None),
        ):
            db.add(
                TaskChatMessage(
                    task_id=task_id,
                    user_id=user_id,
                    role=role,
                    content=content,
                    message_type="chat",
                    message_metadata=metadata,
                )
            )
        for event_id, data in (
            ("u1", {"message": "hello"}),
            ("u2", {"message": "[connector settled]", "metadata": HIDDEN}),
        ):
            db.add(
                TraceEvent(
                    task_id=task_id,
                    event_id=event_id,
                    event_type="user_message",
                    timestamp=STAMP,
                    data=data,
                )
            )
        db.commit()

    response = client.get(f"/api/conversation-logs/{task_id}", headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert [m["content"] for m in body["transcript"]] == ["hello", "connected"]
    assert [e["event_id"] for e in body["trace_events"]] == ["u1"]


def test_preview_sample_is_never_a_hidden_message() -> None:
    from xagent.web.api.agents import get_agent_preview_task  # noqa: F401

    headers = _admin_headers()
    agent = create_agent(headers)
    with _direct_db_session() as db:
        task = Task(
            user_id=agent["user_id"],
            title="Sample",
            status=TaskStatus.COMPLETED,
            is_visible=False,
            agent_config={
                "is_preview": True,
                "preview_agent_id": agent["id"],
                "preview_config_key": "k",
            },
        )
        db.add(task)
        db.flush()
        for content, metadata in (
            ("[connector settled]", HIDDEN),
            ("the real sample", None),
        ):
            db.add(
                TaskChatMessage(
                    task_id=task.id,
                    user_id=agent["user_id"],
                    role="user",
                    message_type="message",
                    content=content,
                    message_metadata=metadata,
                )
            )
        db.commit()

    response = client.get(f"/api/agents/{agent['id']}/preview-task", headers=headers)
    assert response.status_code == 200, response.text
    assert response.json()["message"] == "the real sample"


def _admin_id() -> int:
    return _user_id("admin")


def test_websocket_replay_of_a_legacy_task_carries_message_metadata() -> None:
    from xagent.web.api.websocket import _load_historical_stream_snapshot_sync

    _admin_headers()
    user_id = _admin_id()
    task_id = _create_task_row(user_id=user_id, title="Connect")
    with _direct_db_session() as db:
        for content, metadata in (("hello", None), ("[connector settled]", HIDDEN)):
            db.add(
                TaskChatMessage(
                    task_id=task_id,
                    user_id=user_id,
                    role="user",
                    content=content,
                    message_type="chat",
                    message_metadata=metadata,
                )
            )
        db.commit()
    snapshot = _load_historical_stream_snapshot_sync(
        task_id, actor_user_id=user_id, actor_is_admin=True
    )
    assert snapshot is not None
    replayed = {
        event["data"]["content"]: event["data"]
        for event in snapshot.events
        if event.get("event_type") == "user_message"
    }
    assert replayed["[connector settled]"]["metadata"] == HIDDEN
    assert "metadata" not in replayed["hello"]


def _trigger_task(*, storage_version: int) -> tuple[int, int]:
    user_id = _admin_id()
    agent_id = _create_agent_row(user_id=user_id, name="Audit Agent")
    task_id = _create_task_row(
        user_id=user_id,
        title="Connect",
        source="trigger",
        is_visible=False,
        agent_id=agent_id,
        agent_config={"trigger_type": "webhook"},
    )
    _attach_trigger_run(
        user_id=user_id,
        agent_id=agent_id,
        task_id=task_id,
        trigger_type="webhook",
        source_event_id="evt-1",
    )
    with _direct_db_session() as db:
        db.query(Task).filter(Task.id == task_id).update(
            {Task.conversation_storage_version: storage_version, Task.run_id: "run-1"}
        )
        db.commit()
    return user_id, task_id


def _add_v2_input(task_id: int, key: str, content: str, metadata, at: datetime):
    from xagent.web.services.task_execution_event_writer import append_fact_no_commit

    payload = {"content": content, "message_type": "user"}
    if metadata:
        payload["message_metadata"] = metadata
    with _direct_db_session() as db:
        append_fact_no_commit(
            db,
            task_id=task_id,
            kind="input_accepted",
            key=key,
            run_id="run-1",
            turn_id=key,
            payload=payload,
            occurred_at=at,
        )
        db.commit()


def test_conversation_log_v2_detail_and_list_leave_out_hidden_messages() -> None:
    headers = _admin_headers()
    _, task_id = _trigger_task(storage_version=2)
    _add_v2_input(task_id, "u1", "hello", None, STAMP)
    _add_v2_input(task_id, "u2", "[connector settled]", HIDDEN, STAMP.replace(hour=13))

    detail = client.get(f"/api/conversation-logs/{task_id}", headers=headers)
    assert detail.status_code == 200, detail.text
    transcript = detail.json()["transcript"]
    assert [m["content"] for m in transcript] == ["hello"]
    assert all("message_metadata" not in m for m in transcript)
    summary = detail.json()["log"]
    assert summary["message_count"] == 1
    assert summary["last_activity_at"].startswith("2026-01-01T12:00:00")

    (listed,) = client.get("/api/conversation-logs", headers=headers).json()["logs"]
    assert listed["message_count"] == 1
    assert listed["last_activity_at"].startswith("2026-01-01T12:00:00")


def test_conversation_log_v1_list_count_and_activity_ignore_hidden_messages() -> None:
    headers = _admin_headers()
    user_id, task_id = _trigger_task(storage_version=1)
    with _direct_db_session() as db:
        for content, metadata, at in (
            ("hello", None, STAMP),
            ("[connector settled]", HIDDEN, STAMP.replace(hour=13)),
        ):
            db.add(
                TaskChatMessage(
                    task_id=task_id,
                    user_id=user_id,
                    role="user",
                    content=content,
                    message_type="chat",
                    message_metadata=metadata,
                    created_at=at,
                )
            )
        db.commit()
    (listed,) = client.get("/api/conversation-logs", headers=headers).json()["logs"]
    assert listed["message_count"] == 1
    assert listed["last_activity_at"].startswith("2026-01-01T12:00:00")


def test_a_result_card_message_is_displayed_in_the_list_count() -> None:
    headers = _admin_headers()
    user_id, task_id = _trigger_task(storage_version=1)
    with _direct_db_session() as db:
        for metadata in (None, {"display": "result_card"}, HIDDEN):
            db.add(
                TaskChatMessage(
                    task_id=task_id,
                    user_id=user_id,
                    role="assistant" if metadata != HIDDEN else "user",
                    content="x",
                    message_type="chat",
                    message_metadata=metadata,
                    created_at=STAMP,
                )
            )
        db.commit()
    (listed,) = client.get("/api/conversation-logs", headers=headers).json()["logs"]
    assert listed["message_count"] == 2


CARD = {
    "title": "Open deals",
    "source": {"name": "Cosmos", "app_id": None, "logo": None},
    "items": ["Contoso"],
    "footnote": None,
}


def test_public_steps_skip_a_result_card_message() -> None:
    """SDK readers get no step for a card: it is not a question and has no step type.

    This pins existing behaviour the card relies on (``_is_question_agent_message``
    drops any agent message that asks nothing), not a change made for cards.
    """
    steps = map_trace_events_to_public_steps(
        [
            _event(
                "agent_message",
                {
                    "message": "Open deals",
                    "message_type": "result_card",
                    "expect_response": False,
                    "visible": True,
                    "metadata": {"display": "result_card", "result_card": CARD},
                },
            ),
            _event("ai_message", {"content": "done"}),
        ]
    )
    assert [s["data"]["content"] for s in steps] == ["done"]


def _add_v2_agent_message(task_id: int, key: str, data: dict, at: datetime):
    from xagent.web.services.task_execution_event_writer import append_fact_no_commit

    with _direct_db_session() as db:
        append_fact_no_commit(
            db,
            task_id=task_id,
            kind="agent_message",
            key=key,
            run_id="run-1",
            turn_id="t1",
            payload={"data": data, "protocol_event_id": key},
            occurred_at=at,
        )
        db.commit()


def _card_data(**overrides) -> dict:
    data = {
        "message": CARD["title"],
        "message_type": "result_card",
        "expect_response": False,
        "visible": True,
        "metadata": {"display": "result_card", "result_card": CARD},
    }
    data.update(overrides)
    return data


def test_v2_conversation_log_counts_a_card_like_the_detail_view_and_v1() -> None:
    headers = _admin_headers()
    _, task_id = _trigger_task(storage_version=2)
    _add_v2_input(task_id, "u1", "hello", None, STAMP)
    _add_v2_agent_message(task_id, "card", _card_data(), STAMP.replace(hour=14))
    # A plain progress message and a message that only claims to be a card
    # are not part of the conversation.
    _add_v2_agent_message(
        task_id,
        "progress",
        {"message": "working", "message_type": "info", "expect_response": False},
        STAMP.replace(hour=15),
    )
    _add_v2_agent_message(
        task_id,
        "fake",
        _card_data(message="fake", metadata={}),
        STAMP.replace(hour=16),
    )

    detail = client.get(f"/api/conversation-logs/{task_id}", headers=headers)
    assert detail.status_code == 200, detail.text
    assert [m["content"] for m in detail.json()["transcript"]] == [
        "hello",
        CARD["title"],
    ]
    assert detail.json()["log"]["message_count"] == 2
    (listed,) = client.get("/api/conversation-logs", headers=headers).json()["logs"]
    assert listed["message_count"] == 2
    assert listed["last_activity_at"].startswith("2026-01-01T14:00:00")


def test_websocket_replay_of_a_legacy_task_carries_a_result_card() -> None:
    from xagent.web.api.websocket import _load_historical_stream_snapshot_sync

    _admin_headers()
    user_id = _admin_id()
    task_id = _create_task_row(user_id=user_id, title="Card")
    metadata = {"display": "result_card", "result_card": CARD}
    with _direct_db_session() as db:
        for content, message_metadata in (
            ("Open deals", metadata),
            ("Malformed card", {"display": "result_card"}),
        ):
            db.add(
                TaskChatMessage(
                    task_id=task_id,
                    user_id=user_id,
                    role="assistant",
                    content=content,
                    message_type="result_card",
                    message_metadata=message_metadata,
                )
            )
        db.commit()
    snapshot = _load_historical_stream_snapshot_sync(
        task_id, actor_user_id=user_id, actor_is_admin=True
    )
    assert snapshot is not None
    replayed = {
        event["data"]["content"]: event["data"]
        for event in snapshot.events
        if event.get("event_type") == "agent_message"
    }
    assert replayed["Open deals"]["message_type"] == "result_card"
    assert replayed["Open deals"]["metadata"] == metadata
    assert "metadata" not in replayed["Malformed card"]
