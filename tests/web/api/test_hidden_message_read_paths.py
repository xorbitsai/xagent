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
from xagent.web.services.message_metadata import is_hidden_user_event

from .conftest import _admin_headers, _direct_db_session, client
from .test_agent_preview_history import create_agent
from .test_conversation_logs import (
    _attach_trigger_run,
    _create_agent_row,
    _create_task_row,
    _user_id,
)

pytestmark = pytest.mark.usefixtures("_test_db")

HIDDEN = {"hidden": True, "kind": "connector_settled", "interaction_id": "i-1"}
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


def test_is_hidden_user_event_needs_a_user_message_with_the_flag() -> None:
    assert is_hidden_user_event("user_message", {"metadata": HIDDEN})
    assert not is_hidden_user_event("user_message", {"metadata": {"kind": "x"}})
    assert not is_hidden_user_event("user_message", {"message": "hi"})
    assert not is_hidden_user_event("user_message", None)
    assert not is_hidden_user_event("ai_message", {"metadata": HIDDEN})


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
