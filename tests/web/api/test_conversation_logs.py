from __future__ import annotations

import csv
import gc
import io
import logging
import re
import secrets
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, NamedTuple

import pytest
from sqlalchemy import and_, column, event, exists, literal_column, select, text
from sqlalchemy.orm import Session, aliased
from sqlalchemy.types import Boolean

from xagent.web.api import conversation_logs
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import get_engine
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.trigger import AgentTrigger, TriggerRun, TriggerRunStatus
from xagent.web.models.user import User
from xagent.web.services import conversation_log_sources as external_source_hooks
from xagent.web.services.task_runtime import (
    MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY,
)

from .conftest import (
    _admin_headers,
    _direct_db_session,
    _register_second_user,
    client,
)
from .expired_task_shared import (
    assert_task_expired,
    insert_tombstone,
    purge_conversation,
)

pytestmark = pytest.mark.usefixtures("_test_db")


@contextmanager
def _capture_sql_statements():
    statements: list[tuple[str, Any]] = []
    engine = get_engine()

    def before_cursor_execute(
        conn: Any,
        cursor: Any,
        statement: str,
        parameters: Any,
        context: Any,
        executemany: bool,
    ) -> None:
        statements.append((statement, parameters))

    event.listen(engine, "before_cursor_execute", before_cursor_execute)
    try:
        yield statements
    finally:
        event.remove(engine, "before_cursor_execute", before_cursor_execute)


def _user_id(username: str) -> int:
    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.username == username).one()
        return int(user.id)
    finally:
        db.close()


def _create_agent_row(
    *,
    user_id: int,
    name: str,
    status: AgentStatus = AgentStatus.PUBLISHED,
    widget_enabled: bool = True,
    allowed_domains: list[str] | None = None,
    share_enabled: bool = False,
    share_token: str | None = None,
) -> int:
    db = _direct_db_session()
    try:
        agent = Agent(
            user_id=user_id,
            name=name,
            description=f"{name} description",
            instructions=f"{name} instructions",
            execution_mode="balanced",
            status=status,
            widget_enabled=widget_enabled,
            allowed_domains=allowed_domains or ["example.com"],
            share_enabled=share_enabled,
            share_token=share_token,
            widget_key=f"wk-{secrets.token_urlsafe(24)}" if widget_enabled else None,
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        return int(agent.id)
    finally:
        db.close()


def _widget_key_for(agent_id: int) -> str:
    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        assert agent is not None and agent.widget_key
        return str(agent.widget_key)
    finally:
        db.close()


def _create_task_row(
    *,
    user_id: int,
    title: str,
    source: str = "internal",
    is_visible: bool = True,
    agent_id: int | None = None,
    description: str | None = None,
    input_text: str | None = None,
    output_text: str | None = None,
    agent_config: dict[str, Any] | None = None,
    channel_name: str | None = None,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> int:
    db = _direct_db_session()
    try:
        task_kwargs: dict[str, Any] = {
            "user_id": user_id,
            "title": title,
            "description": description or title,
            "status": TaskStatus.COMPLETED,
            "source": source,
            "is_visible": is_visible,
            "agent_id": agent_id,
            "input": input_text,
            "output": output_text,
            "agent_config": agent_config,
            "channel_name": channel_name,
            "input_tokens": 3,
            "output_tokens": 5,
            "total_tokens": 8,
            "llm_calls": 1,
        }
        if created_at is not None:
            task_kwargs["created_at"] = created_at
        if updated_at is not None:
            task_kwargs["updated_at"] = updated_at
        task = Task(**task_kwargs)
        db.add(task)
        db.commit()
        db.refresh(task)
        return int(task.id)
    finally:
        db.close()


def _attach_trigger_run(
    *,
    user_id: int,
    agent_id: int,
    task_id: int,
    trigger_type: str,
    source_event_id: str,
) -> tuple[int, int]:
    db = _direct_db_session()
    try:
        trigger = AgentTrigger(
            user_id=user_id,
            agent_id=agent_id,
            type=trigger_type,
            name=f"{trigger_type} trigger",
            enabled=True,
            config={},
            webhook_token=f"token-{task_id}" if trigger_type == "webhook" else None,
            secret_hash="$2b$hidden",
        )
        db.add(trigger)
        db.flush()
        run = TriggerRun(
            trigger_id=int(trigger.id),
            task_id=task_id,
            status=TriggerRunStatus.COMPLETED.value,
            source_event_id=source_event_id,
            payload_snapshot={"subject": source_event_id},
            idempotency_key=f"{trigger_type}:{source_event_id}",
        )
        db.add(run)
        db.commit()
        return int(trigger.id), int(run.id)
    finally:
        db.close()


def _add_chat_message(
    *,
    task_id: int,
    user_id: int,
    role: str,
    content: str,
    message_type: str = "chat",
    created_at: datetime | None = None,
) -> None:
    db = _direct_db_session()
    try:
        message_kwargs: dict[str, Any] = {
            "task_id": task_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "message_type": message_type,
        }
        if created_at is not None:
            message_kwargs["created_at"] = created_at
        db.add(TaskChatMessage(**message_kwargs))
        db.commit()
    finally:
        db.close()


def _add_compact_event(
    *,
    task_id: int,
    event_id: str,
    timestamp: datetime,
    original_tokens: int,
    compacted_tokens: int,
) -> None:
    db = _direct_db_session()
    try:
        db.add(
            TraceEvent(
                task_id=task_id,
                event_id=event_id,
                event_type="action_end_compact",
                timestamp=timestamp,
                step_id="react_x",
                data={
                    "original_tokens": original_tokens,
                    "compacted_tokens": compacted_tokens,
                    "compression_ratio": "0.8%",
                    "success": True,
                },
            )
        )
        db.commit()
    finally:
        db.close()


def _authenticate_widget_guest(
    *,
    agent_id: int,
    guest_id: str = "guest-1",
    origin: str = "https://example.com",
) -> dict[str, str]:
    del origin
    response = client.post(
        "/api/widget/auth",
        json={"widget_key": _widget_key_for(agent_id), "guest_id": guest_id},
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _authenticate_share_guest(share_token: str) -> dict[str, str]:
    response = client.post("/api/share/auth", json={"share_token": share_token})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_conversation_logs_list_maps_sources_counts_filters_and_access_scope() -> None:
    admin_headers = _admin_headers()
    bob_headers = _register_second_user(username="bob")
    admin_id = _user_id("admin")
    bob_id = _user_id("bob")
    alpha_agent_id = _create_agent_row(user_id=admin_id, name="Alpha Agent")
    beta_agent_id = _create_agent_row(user_id=admin_id, name="Beta Agent")
    bob_agent_id = _create_agent_row(user_id=bob_id, name="Bob Agent")

    rest_task_id = _create_task_row(
        user_id=admin_id,
        title="REST lead intake",
        description="Lead intake from API",
        input_text="needle from api input",
        output_text="lead accepted",
        source="sdk",
        is_visible=False,
        agent_id=alpha_agent_id,
    )
    webhook_task_id = _create_task_row(
        user_id=admin_id,
        title="Webhook crm event",
        source="trigger",
        is_visible=False,
        agent_id=alpha_agent_id,
        agent_config={"trigger_type": "webhook"},
        input_text="crm payload",
    )
    _attach_trigger_run(
        user_id=admin_id,
        agent_id=alpha_agent_id,
        task_id=webhook_task_id,
        trigger_type="webhook",
        source_event_id="evt-webhook",
    )
    scheduled_task_id = _create_task_row(
        user_id=admin_id,
        title="Scheduled daily digest",
        source="trigger",
        is_visible=False,
        agent_id=alpha_agent_id,
        agent_config={"trigger_type": "scheduled"},
    )
    _attach_trigger_run(
        user_id=admin_id,
        agent_id=alpha_agent_id,
        task_id=scheduled_task_id,
        trigger_type="scheduled",
        source_event_id="evt-scheduled",
    )
    widget_task_id = _create_task_row(
        user_id=admin_id,
        title="Widget visitor",
        source="widget",
        is_visible=False,
        agent_id=beta_agent_id,
        agent_config={"guest_id": "guest-1"},
        channel_name="Web Widget",
    )
    share_task_id = _create_task_row(
        user_id=admin_id,
        title="Share visitor",
        source="shared_link",
        is_visible=False,
        agent_id=beta_agent_id,
        agent_config={"auth_mode": "share", "share_agent_id": beta_agent_id},
        channel_name="Shared Agent",
    )
    _create_task_row(
        user_id=admin_id,
        title="Visible SDK should stay out",
        source="sdk",
        is_visible=True,
        agent_id=alpha_agent_id,
    )
    _create_task_row(
        user_id=admin_id,
        title="Hidden internal should stay out",
        source="internal",
        is_visible=False,
        agent_id=alpha_agent_id,
    )
    bob_task_id = _create_task_row(
        user_id=bob_id,
        title="Bob REST task",
        source="sdk",
        is_visible=False,
        agent_id=bob_agent_id,
    )

    response = client.get("/api/conversation-logs", headers=admin_headers)
    assert response.status_code == 200, response.text
    body = response.json()
    returned_ids = {item["task_id"] for item in body["logs"]}
    assert returned_ids == {
        rest_task_id,
        webhook_task_id,
        widget_task_id,
        share_task_id,
        bob_task_id,
    }
    assert scheduled_task_id not in returned_ids
    assert body["source_counts"] == {
        "all": 5,
        "widget": 1,
        "rest_api": 2,
        "shared_link": 1,
        "webhook": 1,
    }
    labels_by_id = {item["task_id"]: item["source_label"] for item in body["logs"]}
    assert labels_by_id[rest_task_id] == "REST API"
    assert labels_by_id[webhook_task_id] == "Webhook"

    filtered = client.get(
        f"/api/conversation-logs?source=rest_api&agent_id={alpha_agent_id}&search=needle",
        headers=admin_headers,
    )
    assert filtered.status_code == 200, filtered.text
    assert [item["task_id"] for item in filtered.json()["logs"]] == [rest_task_id]
    assert filtered.json()["pagination"]["total"] == 1

    bob_response = client.get("/api/conversation-logs", headers=bob_headers)
    assert bob_response.status_code == 200, bob_response.text
    assert [item["task_id"] for item in bob_response.json()["logs"]] == [bob_task_id]


def test_conversation_logs_list_rejects_unsupported_source() -> None:
    response = client.get(
        "/api/conversation-logs?source=internal",
        headers=_admin_headers(),
    )

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "Unsupported conversation source"


def test_conversation_log_detail_returns_read_only_transcript_and_audit_metadata() -> (
    None
):
    headers = _admin_headers()
    user_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=user_id, name="Audit Agent")
    task_id = _create_task_row(
        user_id=user_id,
        title="Webhook audit",
        source="trigger",
        is_visible=False,
        agent_id=agent_id,
        input_text="incoming payload",
        output_text="processed payload",
        agent_config={"trigger_type": "webhook"},
    )
    trigger_id, run_id = _attach_trigger_run(
        user_id=user_id,
        agent_id=agent_id,
        task_id=task_id,
        trigger_type="webhook",
        source_event_id="evt-42",
    )
    _add_chat_message(
        task_id=task_id,
        user_id=user_id,
        role="user",
        content="Please handle this event",
    )
    _add_chat_message(
        task_id=task_id,
        user_id=user_id,
        role="assistant",
        content="Event handled",
    )

    response = client.get(f"/api/conversation-logs/{task_id}", headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["log"]["task_id"] == task_id
    assert body["log"]["source"] == "webhook"
    assert body["metadata"]["task"]["input"] == "incoming payload"
    assert body["metadata"]["task"]["output"] == "processed payload"
    assert body["metadata"]["trigger"] == {
        "trigger_id": trigger_id,
        "trigger_run_id": run_id,
        "trigger_type": "webhook",
        "source_event_id": "evt-42",
        "status": TriggerRunStatus.COMPLETED.value,
        "test": False,
    }
    assert "webhook_secret" not in str(body["metadata"])
    assert [message["role"] for message in body["transcript"]] == [
        "user",
        "assistant",
    ]
    assert [message["content"] for message in body["transcript"]] == [
        "Please handle this event",
        "Event handled",
    ]
    assert body["read_only"] is True


def test_conversation_log_detail_interleaves_compaction_notices() -> None:
    headers = _admin_headers()
    user_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=user_id, name="Compact Agent")
    task_id = _create_task_row(
        user_id=user_id,
        title="Compaction log",
        source="widget",
        is_visible=False,
        agent_id=agent_id,
    )
    base = datetime(2026, 7, 7, 6, 31, 0, tzinfo=timezone.utc)
    _add_chat_message(
        task_id=task_id,
        user_id=user_id,
        role="user",
        content="Do a long task",
        created_at=base,
    )
    _add_compact_event(
        task_id=task_id,
        event_id="compact-evt-1",
        timestamp=base.replace(second=18),
        original_tokens=56860,
        compacted_tokens=449,
    )
    _add_chat_message(
        task_id=task_id,
        user_id=user_id,
        role="assistant",
        content="All done",
        created_at=base.replace(second=53),
    )

    response = client.get(f"/api/conversation-logs/{task_id}", headers=headers)
    assert response.status_code == 200, response.text
    transcript = response.json()["transcript"]

    # Compaction notice sits between the user turn and the assistant reply.
    assert [item["message_type"] for item in transcript] == [
        "chat",
        "compaction",
        "chat",
    ]
    compaction = transcript[1]
    assert compaction["role"] == "system"
    assert compaction["compaction"]["original_tokens"] == 56860
    assert compaction["compaction"]["compacted_tokens"] == 449


def test_conversation_log_detail_rejects_non_owner_external_task() -> None:
    _admin_headers()
    bob_headers = _register_second_user(username="detailbob")
    admin_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=admin_id, name="Private External Agent")
    task_id = _create_task_row(
        user_id=admin_id,
        title="Admin REST task",
        source="sdk",
        is_visible=False,
        agent_id=agent_id,
    )

    response = client.get(f"/api/conversation-logs/{task_id}", headers=bob_headers)

    assert response.status_code == 404, response.text


def test_conversation_log_detail_returns_404_for_non_webhook_trigger_tasks() -> None:
    headers = _admin_headers()
    user_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=user_id, name="Scheduled Agent")
    task_id = _create_task_row(
        user_id=user_id,
        title="Scheduled digest",
        source="trigger",
        is_visible=False,
        agent_id=agent_id,
        agent_config={"trigger_type": "scheduled"},
    )
    _attach_trigger_run(
        user_id=user_id,
        agent_id=agent_id,
        task_id=task_id,
        trigger_type="scheduled",
        source_event_id="evt-scheduled",
    )

    response = client.get(f"/api/conversation-logs/{task_id}", headers=headers)

    assert response.status_code == 404, response.text


def test_conversation_log_detail_returns_public_context_for_widget_and_share_logs() -> (
    None
):
    headers = _admin_headers()
    user_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=user_id, name="Public Context Agent")
    widget_task_id = _create_task_row(
        user_id=user_id,
        title="Widget public context",
        source="widget",
        is_visible=False,
        agent_id=agent_id,
        agent_config={"guest_id": "guest-42", "widget_agent_id": agent_id},
        channel_name="Web Widget",
    )
    share_task_id = _create_task_row(
        user_id=user_id,
        title="Share public context",
        source="shared_link",
        is_visible=False,
        agent_id=agent_id,
        agent_config={"auth_mode": "share", "share_agent_id": agent_id},
        channel_name="Shared Agent",
    )

    widget_response = client.get(
        f"/api/conversation-logs/{widget_task_id}", headers=headers
    )
    share_response = client.get(
        f"/api/conversation-logs/{share_task_id}", headers=headers
    )

    assert widget_response.status_code == 200, widget_response.text
    assert widget_response.json()["metadata"]["public_context"] == {
        "guest_id": "guest-42",
        "auth_mode": "widget",
        "channel_name": "Web Widget",
        "widget_agent_id": agent_id,
    }
    assert share_response.status_code == 200, share_response.text
    assert share_response.json()["metadata"]["public_context"] == {
        "auth_mode": "share",
        "channel_name": "Shared Agent",
        "share_agent_id": agent_id,
    }


def test_public_widget_and_share_task_creation_classifies_hidden_external_logs() -> (
    None
):
    _admin_headers()
    owner_id = _user_id("admin")
    widget_agent_id = _create_agent_row(
        user_id=owner_id,
        name="Widget Agent",
        widget_enabled=True,
        allowed_domains=["example.com"],
    )
    share_agent_id = _create_agent_row(
        user_id=owner_id,
        name="Share Agent",
        share_enabled=True,
        share_token="share-token",
    )

    widget_headers = _authenticate_widget_guest(agent_id=widget_agent_id)
    widget_response = client.post(
        "/api/widget/chat/task/create",
        json={
            "title": "Widget conversation",
            "description": "Widget hello",
            "agent_id": widget_agent_id,
        },
        headers=widget_headers,
    )
    assert widget_response.status_code == 200, widget_response.text

    share_headers = _authenticate_share_guest("share-token")
    share_response = client.post(
        "/api/share/chat/task/create",
        json={
            "title": "Share conversation",
            "description": "Share hello",
            "agent_id": share_agent_id,
        },
        headers=share_headers,
    )
    assert share_response.status_code == 200, share_response.text

    db = _direct_db_session()
    try:
        widget_task = (
            db.query(Task).filter(Task.id == widget_response.json()["task_id"]).one()
        )
        share_task = (
            db.query(Task).filter(Task.id == share_response.json()["task_id"]).one()
        )
        assert widget_task.source == "widget"
        assert widget_task.is_visible is False
        assert share_task.source == "shared_link"
        assert share_task.is_visible is False
    finally:
        db.close()

    owner_headers = _admin_headers()
    logs_response = client.get("/api/conversation-logs", headers=owner_headers)
    assert logs_response.status_code == 200, logs_response.text
    logs = logs_response.json()["logs"]
    assert {item["source"] for item in logs} == {"widget", "shared_link"}


def test_conversation_logs_list_sorts_by_last_message_activity() -> None:
    headers = _admin_headers()
    user_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=user_id, name="Activity Agent")
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    stale_activity_task_id = _create_task_row(
        user_id=user_id,
        title="Recently updated row",
        source="sdk",
        is_visible=False,
        agent_id=agent_id,
        created_at=base.replace(hour=8),
        updated_at=base.replace(hour=11),
    )
    fresh_activity_task_id = _create_task_row(
        user_id=user_id,
        title="Recent conversation turn",
        source="sdk",
        is_visible=False,
        agent_id=agent_id,
        created_at=base.replace(hour=8),
        updated_at=base.replace(hour=9),
    )
    _add_chat_message(
        task_id=stale_activity_task_id,
        user_id=user_id,
        role="user",
        content="older turn",
        created_at=base.replace(hour=10),
    )
    _add_chat_message(
        task_id=fresh_activity_task_id,
        user_id=user_id,
        role="user",
        content="newer turn",
        created_at=base.replace(hour=12),
    )

    response = client.get("/api/conversation-logs", headers=headers)

    assert response.status_code == 200, response.text
    logs = response.json()["logs"]
    assert [item["task_id"] for item in logs[:2]] == [
        fresh_activity_task_id,
        stale_activity_task_id,
    ]
    assert logs[0]["last_activity_at"] == "2026-01-01T12:00:00+00:00"


WINDOW_START = datetime(2026, 3, 10, tzinfo=timezone.utc)
WINDOW_END = datetime(2026, 3, 11, tzinfo=timezone.utc)
WINDOW_PARAMS = {
    "active_from": "2026-03-10T00:00:00Z",
    "active_to": "2026-03-11T00:00:00Z",
}


def _listed_ids(headers: dict[str, str], **params: Any) -> list[int]:
    response = client.get("/api/conversation-logs", params=params, headers=headers)
    assert response.status_code == 200, response.text
    return [item["task_id"] for item in response.json()["logs"]]


def _displayed_activity_in_window(
    headers: dict[str, str], start: datetime, end: datetime
) -> set[int]:
    response = client.get(
        "/api/conversation-logs", params={"per_page": 100}, headers=headers
    )
    assert response.status_code == 200, response.text
    return {
        item["task_id"]
        for item in response.json()["logs"]
        if start <= datetime.fromisoformat(item["last_activity_at"]) < end
    }


@pytest.mark.parametrize("inside", [True, False], ids=["inside", "outside"])
@pytest.mark.parametrize("has_messages", [True, False], ids=["messages", "bare"])
def test_activity_window_matches_displayed_last_activity_for_v1_rows(
    has_messages: bool, inside: bool
) -> None:
    headers = _admin_headers()
    user_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=user_id, name="Window Agent")
    in_window = datetime(2026, 3, 10, 12, tzinfo=timezone.utc)
    before = datetime(2026, 3, 1, 12, tzinfo=timezone.utc)
    after = datetime(2026, 3, 12, 12, tzinfo=timezone.utc)
    target_time = in_window if inside else after
    opposite_time = before if inside else in_window

    target_id = _create_task_row(
        user_id=user_id,
        title="target",
        source="sdk",
        is_visible=False,
        agent_id=agent_id,
        created_at=before,
        # With messages, the message time decides; updated_at points the other
        # way so a filter that read it instead would disagree with the list.
        updated_at=opposite_time if has_messages else target_time,
    )
    if has_messages:
        _add_chat_message(
            task_id=target_id,
            user_id=user_id,
            role="user",
            content="hello",
            created_at=target_time,
        )
    # Controls on both sides of the window keep the result set non-trivial.
    control_in = _create_task_row(
        user_id=user_id,
        title="control in",
        source="sdk",
        is_visible=False,
        agent_id=agent_id,
        created_at=before,
        updated_at=in_window,
    )
    _create_task_row(
        user_id=user_id,
        title="control out",
        source="sdk",
        is_visible=False,
        agent_id=agent_id,
        created_at=before,
        updated_at=before,
    )

    expected = _displayed_activity_in_window(headers, WINDOW_START, WINDOW_END)
    assert (target_id in expected) is inside
    assert control_in in expected
    assert set(_listed_ids(headers, per_page=100, **WINDOW_PARAMS)) == expected


def test_activity_window_scopes_counts_agents_and_logs_together() -> None:
    headers = _admin_headers()
    user_id = _user_id("admin")
    alpha = _create_agent_row(user_id=user_id, name="Alpha")
    beta = _create_agent_row(user_id=user_id, name="Beta")
    gamma = _create_agent_row(user_id=user_id, name="Gamma")
    inside = datetime(2026, 3, 10, 12, tzinfo=timezone.utc)
    outside = datetime(2026, 3, 20, 12, tzinfo=timezone.utc)

    def make(title: str, source: str, agent_id: int, when: datetime) -> int:
        return _create_task_row(
            user_id=user_id,
            title=title,
            source=source,
            is_visible=False,
            agent_id=agent_id,
            created_at=outside,
            updated_at=when,
        )

    alpha_rest = make("alpha rest in", "sdk", alpha, inside)
    make("alpha widget out", "widget", alpha, outside)
    beta_widget = make("beta widget in", "widget", beta, inside)
    make("gamma rest out", "sdk", gamma, outside)
    make("gamma widget out", "widget", gamma, outside)

    response = client.get(
        "/api/conversation-logs", params=WINDOW_PARAMS, headers=headers
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert {item["task_id"] for item in body["logs"]} == {alpha_rest, beta_widget}
    assert body["source_counts"] == {
        "all": 2,
        "widget": 1,
        "rest_api": 1,
        "shared_link": 0,
        "webhook": 0,
    }
    assert {item["agent_id"] for item in body["agents"]} == {alpha, beta}
    assert body["pagination"]["total"] == 2


@pytest.mark.parametrize(
    ("row_offset_us", "bound", "included"),
    [(0, "start", True), (0, "end", False), (-1, "end", True), (-1, "start", False)],
    ids=["at-start", "at-end", "just-before-end", "just-before-start"],
)
def test_activity_window_is_half_open(
    row_offset_us: int, bound: str, included: bool
) -> None:
    headers = _admin_headers()
    user_id = _user_id("admin")
    edge = datetime(2026, 3, 10, 0, 0, 0, 123456, tzinfo=timezone.utc)
    row_time = edge + timedelta(microseconds=row_offset_us)
    task_id = _create_task_row(
        user_id=user_id,
        title="edge",
        source="sdk",
        is_visible=False,
        created_at=row_time,
        updated_at=row_time,
    )
    far = timedelta(days=1)
    if bound == "start":
        params = {
            "active_from": edge.isoformat(),
            "active_to": (edge + far).isoformat(),
        }
    else:
        params = {
            "active_from": (edge - far).isoformat(),
            "active_to": edge.isoformat(),
        }

    assert (task_id in _listed_ids(headers, **params)) is included


def test_activity_window_converts_offsets_to_utc() -> None:
    headers = _admin_headers()
    user_id = _user_id("admin")
    stored = datetime(2026, 11, 16, 1, 0, tzinfo=timezone.utc)
    task_id = _create_task_row(
        user_id=user_id,
        title="utc 01:00",
        source="sdk",
        is_visible=False,
        created_at=stored,
        updated_at=stored,
    )

    offset_hit = _listed_ids(headers, active_from="2026-11-16T08:30:00+08:00")
    utc_hit = _listed_ids(headers, active_from="2026-11-16T00:30:00Z")
    offset_miss = _listed_ids(headers, active_from="2026-11-16T09:30:00+08:00")
    utc_miss = _listed_ids(headers, active_from="2026-11-16T01:30:00Z")

    assert offset_hit == utc_hit == [task_id]
    assert offset_miss == utc_miss == []


@pytest.mark.parametrize(
    "params",
    [
        {"active_from": "2026-03-10T00:00:00"},
        {"active_to": "2026-03-10T00:00:00"},
        {"active_from": "2026-03-10T00:00:00Z", "active_to": "2026-03-10T00:00:00Z"},
        {"active_from": "2026-03-11T00:00:00Z", "active_to": "2026-03-10T00:00:00Z"},
        {"active_from": "yesterday"},
        {"active_to": "not-a-date"},
    ],
    ids=[
        "naive-from",
        "naive-to",
        "from-equals-to",
        "from-after-to",
        "unparseable-from",
        "unparseable-to",
    ],
)
def test_activity_window_rejects_invalid_input(params: dict[str, str]) -> None:
    response = client.get(
        "/api/conversation-logs", params=params, headers=_admin_headers()
    )

    assert response.status_code == 422, response.text


def test_list_sql_has_no_time_window_clause_without_time_parameters() -> None:
    headers = _admin_headers()
    user_id = _user_id("admin")
    _create_task_row(
        user_id=user_id,
        title="row",
        source="sdk",
        is_visible=False,
        updated_at=datetime(2026, 3, 10, 12, tzinfo=timezone.utc),
    )

    def summary_statements(**params: Any) -> list[str]:
        with _capture_sql_statements() as statements:
            response = client.get(
                "/api/conversation-logs", params=params, headers=headers
            )
        assert response.status_code == 200, response.text
        return [
            statement
            for statement, _parameters in statements
            if "GROUP BY" in statement and "count(tasks.id)" in statement
        ]

    without_window = summary_statements()
    with_window = summary_statements(**WINDOW_PARAMS)

    assert len(without_window) == 1
    assert "task_chat_messages" not in without_window[0]
    assert "task_execution_events" not in without_window[0]
    assert len(with_window) == 1
    assert "task_chat_messages" in with_window[0]
    assert "task_execution_events" in with_window[0]


EXPORT_URL = "/api/conversation-logs/export"
EXPORT_HEADER = [
    "conversation_id",
    "agent",
    "channel",
    "status",
    "title",
    "started_at",
    "last_activity_at",
    "message_count",
    "total_tokens",
    "first_message",
    "latest_answer",
    "error_message",
]
FORMULA_PREFIXES = ["=", "+", "-", "@", "\t", "\r"]


def _export(headers: dict[str, str], **params: Any) -> Any:
    return client.get(EXPORT_URL, params=params, headers=headers)


def _csv_rows(response: Any) -> list[list[str]]:
    text = response.content.decode("utf-8-sig")
    return list(csv.reader(io.StringIO(text, newline="")))


def _exported_by_column(response: Any) -> list[dict[str, str]]:
    header, *rows = _csv_rows(response)
    return [dict(zip(header, row)) for row in rows]


def _set_task_fields(task_id: int, **fields: Any) -> None:
    db = _direct_db_session()
    try:
        db.query(Task).filter(Task.id == task_id).update(fields)
        db.commit()
    finally:
        db.close()


def test_export_returns_csv_with_bom_and_fixed_header() -> None:
    headers = _admin_headers()
    user_id = _user_id("admin")
    _create_task_row(user_id=user_id, title="row", source="sdk", is_visible=False)

    response = _export(headers)

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/csv")
    assert response.headers["cache-control"] == "no-store"
    assert re.fullmatch(
        r'attachment; filename="conversation-logs-\d{8}-\d{6}\.csv"',
        response.headers["content-disposition"],
    )
    assert response.content.startswith(b"\xef\xbb\xbf")
    assert _csv_rows(response)[0] == EXPORT_HEADER
    assert len(_csv_rows(response)) == 2


def test_export_with_no_rows_is_only_the_header() -> None:
    response = _export(_admin_headers(), **WINDOW_PARAMS)

    assert response.status_code == 200, response.text
    assert _csv_rows(response) == [EXPORT_HEADER]


def test_export_columns_carry_the_listed_values() -> None:
    headers = _admin_headers()
    user_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=user_id, name="Export Agent")
    created = datetime(2026, 3, 1, 8, tzinfo=timezone.utc)
    last_message = datetime(2026, 3, 10, 12, tzinfo=timezone.utc)
    task_id = _create_task_row(
        user_id=user_id,
        title="Export title",
        description="first words",
        output_text="latest reply",
        source="widget",
        is_visible=False,
        agent_id=agent_id,
        created_at=created,
        updated_at=created,
    )
    _set_task_fields(task_id, error_message="boom")
    _add_chat_message(
        task_id=task_id,
        user_id=user_id,
        role="user",
        content="hi",
        created_at=created,
    )
    _add_chat_message(
        task_id=task_id,
        user_id=user_id,
        role="assistant",
        content="hello",
        created_at=last_message,
    )

    (listed,) = client.get("/api/conversation-logs", headers=headers).json()["logs"]
    (row,) = _exported_by_column(_export(headers))

    assert listed["message_count"] == 2
    assert row == {
        "conversation_id": str(task_id),
        "agent": "Export Agent",
        "channel": "Widget",
        "status": "completed",
        "title": "Export title",
        "started_at": "2026-03-01T08:00:00+00:00",
        "last_activity_at": "2026-03-10T12:00:00+00:00",
        "message_count": str(listed["message_count"]),
        "total_tokens": "8",
        "first_message": "first words",
        "latest_answer": "latest reply",
        "error_message": "boom",
    }


def test_export_prefixes_formula_like_text_in_every_text_column() -> None:
    headers = _admin_headers()
    user_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=user_id, name="=agent")
    task_id = _create_task_row(
        user_id=user_id,
        title="+title",
        description="-first",
        output_text="@answer",
        source="sdk",
        is_visible=False,
        agent_id=agent_id,
    )
    _set_task_fields(task_id, error_message="\t=error")

    (row,) = _exported_by_column(_export(headers))

    assert row["agent"] == "'=agent"
    assert row["title"] == "'+title"
    assert row["first_message"] == "'-first"
    assert row["latest_answer"] == "'@answer"
    assert row["error_message"] == "'\t=error"
    assert not row["channel"].startswith("'")
    assert not row["status"].startswith("'")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(f"{prefix}1+1", f"'{prefix}1+1") for prefix in FORMULA_PREFIXES]
    + [("plain text", "plain text"), ("a=b", "a=b"), ("", ""), (None, "")],
)
def test_csv_text_neutralises_formula_prefixes(
    value: str | None, expected: str
) -> None:
    assert conversation_logs._csv_text(value) == expected


def test_csv_line_guards_every_text_cell_and_leaves_numbers() -> None:
    cells: list[Any] = ["=x"] * len(EXPORT_HEADER)
    cells[EXPORT_HEADER.index("conversation_id")] = 7
    cells[EXPORT_HEADER.index("message_count")] = 0

    parsed = next(
        csv.reader(io.StringIO(conversation_logs._csv_line(cells), newline=""))
    )

    for name, cell in zip(EXPORT_HEADER, parsed):
        if name in ("conversation_id", "message_count"):
            assert not cell.startswith("'")
        else:
            assert cell == "'=x", name


def test_export_quotes_every_field_so_separators_cannot_split_cells() -> None:
    headers = _admin_headers()
    user_id = _user_id("admin")
    _create_task_row(
        user_id=user_id,
        title="safe;=1+1",
        description="first, second",
        output_text="plain text",
        source="sdk",
        is_visible=False,
    )

    response = _export(headers)

    text = response.content.decode("utf-8-sig")
    assert '"safe;=1+1"' in text
    assert '"first, second"' in text
    (row,) = _exported_by_column(response)
    assert row["title"] == "safe;=1+1"
    assert row["latest_answer"] == "plain text"


def test_export_has_the_same_rows_in_the_same_order_as_the_listing() -> None:
    headers = _admin_headers()
    user_id = _user_id("admin")
    alpha = _create_agent_row(user_id=user_id, name="Alpha")
    base = datetime(2026, 3, 10, tzinfo=timezone.utc)
    db = _direct_db_session()
    try:
        for index in range(105):
            # Groups of rows share one activity time so the id tie-break is
            # part of the order being compared.
            when = base + timedelta(minutes=index // 3)
            db.add(
                Task(
                    user_id=user_id,
                    title=f"bulk {index}",
                    description=f"needle {index}" if index % 2 else f"hay {index}",
                    status=TaskStatus.COMPLETED,
                    source="widget" if index % 5 == 0 else "sdk",
                    is_visible=False,
                    agent_id=alpha,
                    created_at=when,
                    updated_at=when,
                )
            )
        db.commit()
    finally:
        db.close()

    def listed(params: dict[str, Any]) -> list[str]:
        ids: list[str] = []
        page = 1
        while True:
            response = client.get(
                "/api/conversation-logs",
                params={**params, "per_page": 100, "page": page},
                headers=headers,
            )
            assert response.status_code == 200, response.text
            body = response.json()
            ids.extend(str(item["task_id"]) for item in body["logs"])
            if page >= body["pagination"]["total_pages"]:
                return ids
            page += 1

    filters: list[dict[str, Any]] = [
        {},
        {
            "source": "rest_api",
            "search": "needle",
            "agent_id": alpha,
            "active_from": "2026-03-10T00:05:00Z",
            "active_to": "2026-03-10T00:30:00Z",
        },
    ]
    for params in filters:
        expected = listed(params)
        assert expected
        exported = [
            row["conversation_id"]
            for row in _exported_by_column(_export(headers, **params))
        ]
        assert exported == expected
    assert len(listed({})) == 105


def test_export_rejects_an_unsupported_source_and_a_naive_time() -> None:
    headers = _admin_headers()

    assert _export(headers, source="internal").status_code == 400
    assert _export(headers, active_from="2026-03-10T00:00:00").status_code == 422


def test_export_is_scoped_to_the_callers_own_logs_unless_admin() -> None:
    admin_headers = _admin_headers()
    bob_headers = _register_second_user(username="bob")
    admin_id = _user_id("admin")
    bob_id = _user_id("bob")
    admin_task = _create_task_row(
        user_id=admin_id, title="admin row", source="sdk", is_visible=False
    )
    bob_task = _create_task_row(
        user_id=bob_id, title="bob row", source="sdk", is_visible=False
    )

    bob_ids = {r["conversation_id"] for r in _exported_by_column(_export(bob_headers))}
    admin_ids = {
        r["conversation_id"] for r in _exported_by_column(_export(admin_headers))
    }

    assert bob_ids == {str(bob_task)}
    assert admin_ids == {str(admin_task), str(bob_task)}


def test_export_requires_authentication() -> None:
    response = client.get(EXPORT_URL)

    assert response.status_code in (401, 403), response.text


class _RecordingSpool(tempfile.SpooledTemporaryFile):  # type: ignore[type-arg]
    instances: list[Any] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _RecordingSpool.instances.append(self)


@pytest.fixture
def _recorded_spools(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    _RecordingSpool.instances = []
    monkeypatch.setattr(
        conversation_logs.tempfile, "SpooledTemporaryFile", _RecordingSpool
    )
    return _RecordingSpool.instances


def _three_rows() -> None:
    user_id = _user_id("admin")
    for index in range(3):
        _create_task_row(
            user_id=user_id, title=f"row {index}", source="sdk", is_visible=False
        )


@pytest.mark.parametrize("reason", ["rows", "bytes"])
def test_export_over_a_limit_is_a_422_without_a_file_or_a_cursor_error(
    reason: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    _recorded_spools: list[Any],
) -> None:
    headers = _admin_headers()
    _three_rows()
    if reason == "rows":
        limit = 2
        monkeypatch.setattr(conversation_logs, "EXPORT_ROW_LIMIT", limit)
    else:
        header_bytes = len(conversation_logs._csv_line(EXPORT_HEADER).encode("utf-8"))
        limit = header_bytes + 10
        monkeypatch.setattr(conversation_logs, "EXPORT_BYTE_LIMIT", limit)

    with caplog.at_level(logging.DEBUG):
        response = _export(headers)
        # The result cursor must already be closed; a late close would run
        # after the connection went back to the pool.
        gc.collect()

    assert response.status_code == 422, response.text
    assert response.json()["detail"] == {
        "code": "conversation_log_export_too_large",
        "reason": reason,
        "limit": limit,
    }
    assert "text/csv" not in response.headers["content-type"]
    assert _recorded_spools and all(spool.closed for spool in _recorded_spools)
    assert "Error closing cursor" not in caplog.text


def test_export_exactly_at_the_row_limit_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = _admin_headers()
    _three_rows()
    monkeypatch.setattr(conversation_logs, "EXPORT_ROW_LIMIT", 3)

    response = _export(headers)

    assert response.status_code == 200, response.text
    assert len(_csv_rows(response)) == 4


def test_export_exactly_at_the_byte_limit_succeeds_and_one_byte_less_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = _admin_headers()
    _three_rows()
    # The limit counts the CSV text; the BOM is not part of it.
    size = len(_export(headers).content) - len(b"\xef\xbb\xbf")

    monkeypatch.setattr(conversation_logs, "EXPORT_BYTE_LIMIT", size)
    at_limit = _export(headers)
    monkeypatch.setattr(conversation_logs, "EXPORT_BYTE_LIMIT", size - 1)
    below_limit = _export(headers)

    assert at_limit.status_code == 200, at_limit.text
    assert len(_csv_rows(at_limit)) == 4
    assert below_limit.status_code == 422, below_limit.text


def test_export_releases_the_connection_before_the_first_byte_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = _admin_headers()
    _create_task_row(
        user_id=_user_id("admin"), title="row", source="sdk", is_visible=False
    )
    order: list[str] = []
    outstanding = 0
    original_iter = conversation_logs._iter_export_file
    pool_return_event = "checkin"  # codespell:ignore checkin

    def recording_iter(raw: Any) -> Any:
        for chunk in original_iter(raw):
            order.append(f"first-chunk(outstanding={outstanding})")
            yield chunk
            return

    def on_checkout(*_args: Any) -> None:
        nonlocal outstanding
        outstanding += 1
        order.append("checkout")

    def on_checkin(*_args: Any) -> None:
        nonlocal outstanding
        outstanding -= 1
        order.append("returned")

    monkeypatch.setattr(conversation_logs, "_iter_export_file", recording_iter)
    engine = get_engine()
    event.listen(engine, "checkout", on_checkout)
    event.listen(engine, pool_return_event, on_checkin)
    try:
        response = _export(headers)
    finally:
        event.remove(engine, "checkout", on_checkout)
        event.remove(engine, pool_return_event, on_checkin)

    assert response.status_code == 200, response.text
    first_chunk = next(i for i, name in enumerate(order) if "first-chunk" in name)
    assert "returned" in order[:first_chunk], order
    assert order[first_chunk] == "first-chunk(outstanding=0)", order


def test_conversation_logs_list_does_not_preload_off_page_messages() -> None:
    headers = _admin_headers()
    user_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=user_id, name="Paged Agent")

    for index in range(12):
        task_id = _create_task_row(
            user_id=user_id,
            title=f"Paged REST task {index}",
            source="sdk",
            is_visible=False,
            agent_id=agent_id,
        )
        _add_chat_message(
            task_id=task_id,
            user_id=user_id,
            role="user",
            content=f"message {index}",
        )

    with _capture_sql_statements() as statements:
        response = client.get(
            "/api/conversation-logs?page=2&per_page=5",
            headers=headers,
        )

    assert response.status_code == 200, response.text
    assert len(response.json()["logs"]) == 5

    message_query_param_counts = [
        len(parameters)
        for statement, parameters in statements
        if "FROM task_chat_messages" in statement
        and "task_chat_messages.task_id IN" in statement
        and isinstance(parameters, tuple)
    ]
    assert message_query_param_counts, (
        "Expected message queries to fire but none matched the SQL filter"
    )
    # At most five page task IDs plus the V1-only storage-version predicate.
    assert max(message_query_param_counts) <= 6


def test_conversation_logs_list_batches_trigger_type_lookup() -> None:
    headers = _admin_headers()
    user_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=user_id, name="Webhook Agent")

    for index in range(8):
        task_id = _create_task_row(
            user_id=user_id,
            title=f"Webhook task {index}",
            source="trigger",
            is_visible=False,
            agent_id=agent_id,
        )
        _attach_trigger_run(
            user_id=user_id,
            agent_id=agent_id,
            task_id=task_id,
            trigger_type="webhook",
            source_event_id=f"evt-{index}",
        )

    with _capture_sql_statements() as statements:
        response = client.get(
            "/api/conversation-logs?source=webhook&page=1&per_page=5",
            headers=headers,
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["logs"]) == 5
    assert body["pagination"]["total"] == 8

    trigger_lookup_queries = [
        statement
        for statement, _parameters in statements
        if "FROM trigger_runs" in statement or "FROM agent_triggers" in statement
    ]
    assert len(trigger_lookup_queries) <= 2


def test_detail_returns_trace_events() -> None:
    admin = _admin_headers()
    admin_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=admin_id, name="Trace Agent")
    task_id = _create_task_row(
        user_id=admin_id,
        title="Trace REST task",
        source="sdk",
        is_visible=False,
        agent_id=agent_id,
        input_text="hi",
        output_text="done",
    )

    db = _direct_db_session()
    try:
        db.add(
            TraceEvent(
                task_id=task_id,
                event_id="evt-1",
                event_type="tool_call_start",
                timestamp=datetime.now(timezone.utc),
                data={"tool_name": "search", "tool_args": {"q": "x"}},
            )
        )
        db.commit()
    finally:
        db.close()

    resp = client.get(f"/api/conversation-logs/{task_id}", headers=admin)
    assert resp.status_code == 200, resp.text
    events = resp.json()["trace_events"]
    assert len(events) == 1
    assert events[0]["event_type"] == "tool_call_start"
    assert events[0]["data"]["tool_name"] == "search"


def test_detail_on_a_trace_expired_task_is_empty_not_an_error() -> None:
    """Acceptance criterion from #2563: empty, not 500.

    Retention's trace path deletes a terminal task's ``trace_events`` while
    keeping the task and its transcript. This endpoint then authorizes a task
    that exists and reads a trace that does not, in two places -- the
    ``action_end_compact`` rows folded into the transcript, and the trace
    timeline itself. Both must degrade to nothing while the conversation the
    retention policy promised to keep is still served.
    """
    from datetime import timedelta

    from xagent.web.models.task_command import TaskExecutionCommand
    from xagent.web.services.task_retention_purge import (
        RetentionPurgeAction,
        purge_task,
    )

    admin = _admin_headers()
    admin_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=admin_id, name="Expired Trace Agent")
    task_id = _create_task_row(
        user_id=admin_id,
        title="Trace expiry task",
        source="sdk",
        is_visible=False,
        agent_id=agent_id,
        input_text="hi",
        output_text="done",
    )
    _add_chat_message(
        task_id=task_id,
        user_id=admin_id,
        role="user",
        content="the conversation retention keeps",
    )

    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    db = _direct_db_session()
    try:
        db.add_all(
            [
                TraceEvent(
                    task_id=task_id,
                    event_id="evt-timeline",
                    event_type="tool_call_start",
                    timestamp=base,
                    data={"tool_name": "search", "tool_args": {"q": "x"}},
                ),
                # The other reader: folded into the transcript rather than the
                # timeline, so it would go on rendering if only one were purged.
                TraceEvent(
                    task_id=task_id,
                    event_id="evt-compact",
                    event_type="action_end_compact",
                    timestamp=base,
                    data={"summary": "compacted"},
                ),
            ]
        )
        db.commit()
    finally:
        db.close()

    warm = client.get(f"/api/conversation-logs/{task_id}", headers=admin)
    assert warm.status_code == 200, warm.text
    # Both rows are in the timeline before the purge; the compact one is also
    # read separately into the transcript, which is why it is seeded here.
    assert len(warm.json()["trace_events"]) == 2
    assert warm.json()["trace_events_expired_at"] is None
    assert any(
        entry.get("message_type") == "compaction" for entry in warm.json()["transcript"]
    ), (
        "the compaction entry must be present before the purge for its absence to mean anything"
    )

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.status = TaskStatus.COMPLETED
        task.last_activity_at = base
        task.lease_expires_at = None
        # The transcript was written at wall-clock time, and the assessment
        # measures from the newest message as well as the stored anchor
        # (#2580), so it has to be as old as the anchor it stands beside.
        db.query(TaskChatMessage).filter(TaskChatMessage.task_id == task_id).update(
            {TaskChatMessage.created_at: base}, synchronize_session=False
        )
        db.commit()
        # Task creation stages a start command, and the eligibility predicate
        # counts a pending command as work still owed. Clearing it keeps this
        # test about the trace path rather than about eligibility.
        db.query(TaskExecutionCommand).filter(
            TaskExecutionCommand.task_id == task_id
        ).delete(synchronize_session=False)
        db.commit()
        assert (
            purge_task(
                db,
                task_id,
                now=base + timedelta(days=200),
                conversation_days=365,
                trace_days=90,
            )
            is RetentionPurgeAction.PURGED_TRACES
        )
    finally:
        db.close()

    resp = client.get(f"/api/conversation-logs/{task_id}", headers=admin)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["trace_events"] == []
    # Empty because retention removed it, not because there never was one.
    assert body["trace_events_expired_at"] is not None
    assert body["log"]["task_id"] == task_id
    # The conversation itself is what the shorter trace period exists to keep.
    transcript = body["transcript"]
    assert any(
        entry.get("content") == "the conversation retention keeps"
        for entry in transcript
    ), transcript
    # And the compact row folded into the transcript is gone with the trace,
    # rather than leaving a half-rendered entry behind. Transcript entries are
    # distinguished by ``message_type``; they carry no ``event_type`` at all,
    # so asserting on that field could never have failed.
    assert all(entry.get("message_type") != "compaction" for entry in transcript)


def test_detail_folds_raw_memory_reason_in_historical_checkpoint() -> None:
    admin = _admin_headers()
    admin_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=admin_id, name="Checkpoint Agent")
    task_id = _create_task_row(
        user_id=admin_id,
        title="Checkpoint REST task",
        source="sdk",
        is_visible=False,
        agent_id=agent_id,
    )
    raw_reason = "host resolver secret shard eu-3"

    db = _direct_db_session()
    try:
        db.add(
            TraceEvent(
                task_id=task_id,
                event_id="checkpoint-1",
                event_type="system_update_general",
                timestamp=datetime.now(timezone.utc),
                data={
                    "checkpoint_type": "agent_execution_checkpoint",
                    "snapshot": {
                        "context": {
                            "metadata": {
                                "memory_available": False,
                                "memory_availability_reason": raw_reason,
                            }
                        }
                    },
                },
            )
        )
        db.commit()
    finally:
        db.close()

    response = client.get(f"/api/conversation-logs/{task_id}", headers=admin)

    assert response.status_code == 200, response.text
    events = response.json()["trace_events"]
    assert len(events) == 1
    assert raw_reason not in repr(events)
    assert (
        events[0]["data"]["snapshot"]["context"]["metadata"][
            "memory_availability_reason"
        ]
        == "unavailable"
    )


def test_detail_includes_delegated_agent_traces_but_not_builder_traces() -> None:
    admin = _admin_headers()
    admin_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=admin_id, name="Workforce Manager")
    task_id = _create_task_row(
        user_id=admin_id,
        title="Workforce trace task",
        source="sdk",
        is_visible=False,
        agent_id=agent_id,
        input_text="coordinate workers",
    )

    db = _direct_db_session()
    try:
        now = datetime.now(timezone.utc)
        db.add_all(
            [
                TraceEvent(
                    task_id=task_id,
                    event_id="top-level",
                    event_type="react_task_start",
                    timestamp=now,
                    data={"pattern": "ReActPattern"},
                ),
                TraceEvent(
                    task_id=task_id,
                    event_id="delegated-child",
                    event_type="tool_execution_start",
                    timestamp=now,
                    build_id="agent_17_run",
                    data={
                        "source": "xagent-agent-tool-child",
                        "worker_task_id": "agent_17_run",
                        "agent_name": "Video Generation Agent",
                        "tool_name": "generate_video",
                        "authorization": "Bearer delegated-secret",
                    },
                ),
                TraceEvent(
                    task_id=task_id,
                    event_id="audit-only-child",
                    event_type="tool_execution_start",
                    timestamp=now,
                    build_id="agent_17_run",
                    data={
                        "source": "xagent-agent-tool-child",
                        "worker_task_id": "agent_17_run",
                        "__audit_only__": True,
                    },
                ),
                TraceEvent(
                    task_id=task_id,
                    event_id="builder-only",
                    event_type="tool_execution_start",
                    timestamp=now,
                    build_id="builder-session",
                    data={"source": "agent-builder", "tool_name": "internal"},
                ),
            ]
        )
        db.commit()
    finally:
        db.close()

    resp = client.get(f"/api/conversation-logs/{task_id}", headers=admin)
    assert resp.status_code == 200, resp.text
    events = resp.json()["trace_events"]
    assert [event["event_id"] for event in events] == [
        "top-level",
        "delegated-child",
    ]
    assert events[1]["data"]["authorization"] == "[REDACTED_RUNTIME_SECRET]"


def test_compaction_notice_sorts_between_messages_on_equal_timestamp() -> None:
    headers = _admin_headers()
    user_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=user_id, name="Tie Agent")
    task_id = _create_task_row(
        user_id=user_id,
        title="Tie-break log",
        source="widget",
        is_visible=False,
        agent_id=agent_id,
    )
    # User message, compaction event and assistant reply all share one timestamp,
    # so ordering is decided purely by the three-way kind tie-break.
    ts = datetime(2026, 7, 7, 6, 31, 0, tzinfo=timezone.utc)
    _add_chat_message(
        task_id=task_id, user_id=user_id, role="user", content="Q", created_at=ts
    )
    _add_compact_event(
        task_id=task_id,
        event_id="compact-tie",
        timestamp=ts,
        original_tokens=40000,
        compacted_tokens=300,
    )
    _add_chat_message(
        task_id=task_id, user_id=user_id, role="assistant", content="A", created_at=ts
    )

    response = client.get(f"/api/conversation-logs/{task_id}", headers=headers)
    assert response.status_code == 200, response.text
    transcript = response.json()["transcript"]
    assert [item["message_type"] for item in transcript] == [
        "chat",
        "compaction",
        "chat",
    ]


@pytest.fixture(autouse=True)
def _reset_external_task_hooks():
    hooks = external_source_hooks
    hooks.set_external_task_source_hook(None)
    hooks.set_external_task_context_hook(None)
    try:
        yield hooks
    finally:
        hooks.set_external_task_source_hook(None)
        hooks.set_external_task_context_hook(None)


def _widget_session_predicate(_db: Any) -> list[tuple[Any, str]]:
    return [
        (
            Task.agent_config["widget_session_id"].as_string().isnot(None),
            "widget",
        )
    ]


def test_unclassified_external_tasks_are_listed_under_rest_api() -> None:
    headers = _admin_headers()
    admin_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=admin_id, name="External Agent")
    external_task_id = _create_task_row(
        user_id=admin_id,
        title="Session transport task",
        source="external",
        is_visible=False,
        agent_id=agent_id,
    )
    sdk_task_id = _create_task_row(
        user_id=admin_id,
        title="SDK task",
        source="sdk",
        is_visible=False,
        agent_id=agent_id,
    )

    response = client.get("/api/conversation-logs", headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert {item["task_id"] for item in body["logs"]} == {
        external_task_id,
        sdk_task_id,
    }
    assert body["source_counts"] == {
        "all": 2,
        "widget": 0,
        "rest_api": 2,
        "shared_link": 0,
        "webhook": 0,
    }
    by_id = {item["task_id"]: item for item in body["logs"]}
    assert by_id[external_task_id]["source"] == "rest_api"
    assert by_id[external_task_id]["stored_source"] == "external"

    filtered = client.get("/api/conversation-logs?source=widget", headers=headers)
    assert filtered.status_code == 200, filtered.text
    assert filtered.json()["logs"] == []

    detail = client.get(f"/api/conversation-logs/{external_task_id}", headers=headers)
    assert detail.status_code == 200, detail.text
    assert detail.json()["log"]["source"] == "rest_api"
    assert detail.json()["metadata"]["public_context"] is None


def test_registered_hook_classifies_external_tasks_in_counts_filters_and_detail(
    _reset_external_task_hooks: Any,
) -> None:
    hooks = _reset_external_task_hooks
    headers = _admin_headers()
    admin_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=admin_id, name="Session Widget Agent")
    widget_task_id = _create_task_row(
        user_id=admin_id,
        title="Widget session visitor",
        source="external",
        is_visible=False,
        agent_id=agent_id,
        agent_config={"widget_session_id": "ws-1"},
    )
    api_task_id = _create_task_row(
        user_id=admin_id,
        title="Client application task",
        source="external",
        is_visible=False,
        agent_id=agent_id,
    )
    legacy_widget_task_id = _create_task_row(
        user_id=admin_id,
        title="Legacy widget visitor",
        source="widget",
        is_visible=False,
        agent_id=agent_id,
        # Carries the hook marker too: the hook must only ever see external
        # rows, so the legacy row keeps its agent_config-derived context.
        agent_config={"guest_id": "guest-legacy", "widget_session_id": "ws-legacy"},
        channel_name="Web Widget",
    )

    context_calls: list[tuple[int, str]] = []

    def _context_hook(_db: Any, task: Task, ui_source: str) -> dict[str, Any] | None:
        context_calls.append((int(task.id), ui_source))
        session_id = (task.agent_config or {}).get("widget_session_id")
        if ui_source != "widget" or not session_id:
            return None
        return {
            "auth_mode": "widget_session",
            "channel_name": "Session Widget",
            "widget_session_id": session_id,
            "widget_agent_id": int(task.agent_id),
        }

    def _source_hook(_db: Any) -> list[tuple[Any, str]]:
        return [
            # Only the legacy row carries guest_id. If the hook ever ran against
            # non-external rows this branch would relabel it Shareable Link.
            (Task.agent_config["guest_id"].as_string().isnot(None), "shared_link"),
            *_widget_session_predicate(_db),
        ]

    hooks.set_external_task_source_hook(_source_hook)
    hooks.set_external_task_context_hook(_context_hook)

    response = client.get("/api/conversation-logs", headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source_counts"] == {
        "all": 3,
        "widget": 2,
        "rest_api": 1,
        "shared_link": 0,
        "webhook": 0,
    }
    by_id = {item["task_id"]: item for item in body["logs"]}
    assert by_id[widget_task_id]["source"] == "widget"
    assert by_id[widget_task_id]["source_label"] == "Widget"
    assert by_id[api_task_id]["source"] == "rest_api"
    assert by_id[legacy_widget_task_id]["source"] == "widget"

    widget_filtered = client.get(
        "/api/conversation-logs?source=widget", headers=headers
    )
    assert widget_filtered.status_code == 200, widget_filtered.text
    assert {item["task_id"] for item in widget_filtered.json()["logs"]} == {
        widget_task_id,
        legacy_widget_task_id,
    }
    assert widget_filtered.json()["pagination"]["total"] == 2

    rest_filtered = client.get(
        "/api/conversation-logs?source=rest_api", headers=headers
    )
    assert rest_filtered.status_code == 200, rest_filtered.text
    assert [item["task_id"] for item in rest_filtered.json()["logs"]] == [api_task_id]

    widget_detail = client.get(
        f"/api/conversation-logs/{widget_task_id}", headers=headers
    )
    assert widget_detail.status_code == 200, widget_detail.text
    assert widget_detail.json()["log"]["source"] == "widget"
    assert widget_detail.json()["metadata"]["public_context"] == {
        "auth_mode": "widget_session",
        "channel_name": "Session Widget",
        "widget_session_id": "ws-1",
        "widget_agent_id": agent_id,
    }

    api_detail = client.get(f"/api/conversation-logs/{api_task_id}", headers=headers)
    assert api_detail.status_code == 200, api_detail.text
    assert api_detail.json()["log"]["source"] == "rest_api"
    assert api_detail.json()["metadata"]["public_context"] is None

    legacy_detail = client.get(
        f"/api/conversation-logs/{legacy_widget_task_id}", headers=headers
    )
    assert legacy_detail.status_code == 200, legacy_detail.text
    # The detail path gates the hook in Python; without that gate the guest_id
    # branch would relabel this row Shareable Link.
    assert legacy_detail.json()["log"]["source"] == "widget"
    assert legacy_detail.json()["metadata"]["public_context"] == {
        "guest_id": "guest-legacy",
        "auth_mode": "widget",
        "channel_name": "Web Widget",
        "widget_agent_id": None,
    }
    # The context hook only runs for external rows; legacy rows keep the
    # agent_config-derived context.
    assert {task_id for task_id, _ in context_calls} == {widget_task_id, api_task_id}


def test_source_hook_branches_with_unknown_ui_source_are_ignored(
    _reset_external_task_hooks: Any,
) -> None:
    hooks = _reset_external_task_hooks
    headers = _admin_headers()
    admin_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=admin_id, name="Unknown Source Agent")
    task_id = _create_task_row(
        user_id=admin_id,
        title="Mislabelled external task",
        source="external",
        is_visible=False,
        agent_id=agent_id,
        agent_config={"widget_session_id": "ws-2"},
    )

    hooks.set_external_task_source_hook(
        lambda _db: [
            (Task.agent_config["widget_session_id"].as_string().isnot(None), "slack")
        ]
    )

    response = client.get("/api/conversation-logs", headers=headers)
    assert response.status_code == 200, response.text
    assert response.json()["source_counts"]["rest_api"] == 1
    assert response.json()["source_counts"]["all"] == 1

    detail = client.get(f"/api/conversation-logs/{task_id}", headers=headers)
    assert detail.status_code == 200, detail.text
    assert detail.json()["log"]["source"] == "rest_api"


def test_failing_source_hook_degrades_to_rest_api_default(
    _reset_external_task_hooks: Any,
) -> None:
    hooks = _reset_external_task_hooks
    headers = _admin_headers()
    admin_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=admin_id, name="Broken Hook Agent")
    task_id = _create_task_row(
        user_id=admin_id,
        title="External task behind a broken classifier",
        source="external",
        is_visible=False,
        agent_id=agent_id,
    )

    def _broken_source_hook(_db: Any) -> list[tuple[Any, str]]:
        raise RuntimeError("deployment classifier is down")

    def _broken_context_hook(_db: Any, _task: Task, _ui: str) -> dict[str, Any]:
        raise RuntimeError("deployment context is down")

    hooks.set_external_task_source_hook(_broken_source_hook)
    hooks.set_external_task_context_hook(_broken_context_hook)

    response = client.get("/api/conversation-logs", headers=headers)
    assert response.status_code == 200, response.text
    assert response.json()["source_counts"]["rest_api"] == 1

    detail = client.get(f"/api/conversation-logs/{task_id}", headers=headers)
    assert detail.status_code == 200, detail.text
    assert detail.json()["log"]["source"] == "rest_api"
    assert detail.json()["metadata"]["public_context"] is None


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param("shared_link", "widget", id="alphabetical"),
        pytest.param("widget", "shared_link", id="reversed"),
    ],
)
def test_source_hook_branches_apply_in_returned_list_order(
    _reset_external_task_hooks: Any, first: str, second: str
) -> None:
    hooks = _reset_external_task_hooks
    headers = _admin_headers()
    admin_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=admin_id, name="Ordered Hook Agent")
    task_id = _create_task_row(
        user_id=admin_id,
        title="Matches both predicates",
        source="external",
        is_visible=False,
        agent_id=agent_id,
        agent_config={"widget_session_id": "ws-3", "share_token": "tok"},
    )

    # Both predicates match the row; the first listed pair must win
    # regardless of how the ui_source strings sort.
    hooks.set_external_task_source_hook(
        lambda _db: [
            (Task.agent_config["share_token"].as_string().isnot(None), first),
            (Task.agent_config["widget_session_id"].as_string().isnot(None), second),
        ]
    )

    response = client.get("/api/conversation-logs", headers=headers)
    assert response.status_code == 200, response.text
    assert response.json()["source_counts"][first] == 1
    assert response.json()["source_counts"][second] == 0

    detail = client.get(f"/api/conversation-logs/{task_id}", headers=headers)
    assert detail.status_code == 200, detail.text
    assert detail.json()["log"]["source"] == first


@pytest.mark.parametrize(
    "malformed_branches",
    [
        pytest.param(
            [
                (
                    Task.agent_config["widget_session_id"].as_string().isnot(None),
                    "widget",
                    1,
                )
            ],
            id="three-tuple",
        ),
        pytest.param(["widget"], id="bare-string-entry"),
        pytest.param([("tasks.source = 'external'", "widget")], id="string-predicate"),
        pytest.param(
            [
                (
                    Task.agent_config["widget_session_id"].as_string().isnot(None),
                    "webhook",
                )
            ],
            id="webhook-not-allowed",
        ),
        pytest.param(
            [
                (
                    Task.agent_config["widget_session_id"].as_string().isnot(None),
                    ["widget"],
                )
            ],
            id="unhashable-ui-source",
        ),
        # A bare cross-table comparison would cartesian-join the list query and
        # make the detail query return several rows.
        pytest.param([(Task.id == TriggerRun.task_id, "widget")], id="cross-table"),
        # Raw SQL declares no FROM entry, so it slips past the FROM check and
        # fails at execute time on every backend.
        pytest.param(
            [(literal_column("tasks.id = trigger_runs.task_id", Boolean), "widget")],
            id="literal-column",
        ),
        pytest.param(
            [
                (
                    and_(
                        Task.agent_config["widget_session_id"].as_string().isnot(None),
                        literal_column("tasks.id = trigger_runs.task_id", Boolean),
                    ),
                    "widget",
                )
            ],
            id="literal-column-nested",
        ),
        # column() is bound to no table either; it renders a bare identifier.
        pytest.param([(column("trigger_runs_flag", Boolean), "widget")], id="column"),
        # A VARCHAR expression. SQLite coerces it as a CASE condition (the
        # marker value below starts with a digit so it coerces truthy);
        # PostgreSQL rejects it at execute time.
        pytest.param(
            [(Task.agent_config["widget_session_id"].as_string(), "widget")],
            id="non-boolean",
        ),
    ],
)
def test_malformed_source_hook_entries_degrade_to_rest_api_default(
    _reset_external_task_hooks: Any, malformed_branches: list[Any]
) -> None:
    hooks = _reset_external_task_hooks
    headers = _admin_headers()
    admin_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=admin_id, name="Malformed Hook Agent")
    task_id = _create_task_row(
        user_id=admin_id,
        title="External task behind a malformed classifier",
        source="external",
        is_visible=False,
        agent_id=agent_id,
        agent_config={"widget_session_id": "7ws-4"},
    )
    # Two linked rows: a cross-table predicate that slipped through would
    # cartesian-join them, doubling the list and making the detail query
    # return two rows.
    for event_id in ("evt-a", "evt-b"):
        _attach_trigger_run(
            user_id=admin_id,
            agent_id=agent_id,
            task_id=task_id,
            trigger_type="scheduled",
            source_event_id=event_id,
        )

    hooks.set_external_task_source_hook(lambda _db: malformed_branches)

    response = client.get("/api/conversation-logs", headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source_counts"] == {
        "all": 1,
        "widget": 0,
        "rest_api": 1,
        "shared_link": 0,
        "webhook": 0,
    }
    assert [item["task_id"] for item in body["logs"]] == [task_id]

    detail = client.get(f"/api/conversation-logs/{task_id}", headers=headers)
    assert detail.status_code == 200, detail.text
    assert detail.json()["log"]["source"] == "rest_api"


def test_external_widget_row_without_deployment_context_has_no_public_context(
    _reset_external_task_hooks: Any,
) -> None:
    hooks = _reset_external_task_hooks
    headers = _admin_headers()
    admin_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=admin_id, name="Contextless Widget Agent")
    task_id = _create_task_row(
        user_id=admin_id,
        title="Widget session without context",
        source="external",
        is_visible=False,
        agent_id=agent_id,
        agent_config={"widget_session_id": "ws-5"},
    )
    hooks.set_external_task_source_hook(_widget_session_predicate)

    detail = client.get(f"/api/conversation-logs/{task_id}", headers=headers)
    assert detail.status_code == 200, detail.text
    assert detail.json()["log"]["source"] == "widget"
    # No agent_config-derived fallback: the session transport never sets those keys.
    assert detail.json()["metadata"]["public_context"] is None


def test_malformed_source_hook_entry_does_not_discard_valid_branches(
    _reset_external_task_hooks: Any,
) -> None:
    hooks = _reset_external_task_hooks
    headers = _admin_headers()
    admin_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=admin_id, name="Mixed Hook Agent")
    widget_task_id = _create_task_row(
        user_id=admin_id,
        title="Widget session next to a malformed entry",
        source="external",
        is_visible=False,
        agent_id=agent_id,
        agent_config={"widget_session_id": "ws-6"},
    )

    hooks.set_external_task_source_hook(
        lambda _db: [
            "not-a-pair",
            (Task.agent_config["widget_session_id"].as_string().isnot(None), "widget"),
            (Task.agent_config["share_token"].as_string().isnot(None), "widget", 1),
        ]
    )

    response = client.get("/api/conversation-logs", headers=headers)
    assert response.status_code == 200, response.text
    assert response.json()["source_counts"]["widget"] == 1
    assert response.json()["source_counts"]["rest_api"] == 0

    detail = client.get(f"/api/conversation-logs/{widget_task_id}", headers=headers)
    assert detail.status_code == 200, detail.text
    assert detail.json()["log"]["source"] == "widget"


@pytest.mark.parametrize(
    "build_predicate",
    [
        pytest.param(
            lambda: exists().where(TriggerRun.task_id == Task.id), id="exists"
        ),
        pytest.param(lambda: Task.id.in_(select(TriggerRun.task_id)), id="in-subquery"),
    ],
)
def test_self_contained_cross_table_predicates_classify_external_tasks(
    _reset_external_task_hooks: Any, build_predicate: Any
) -> None:
    hooks = _reset_external_task_hooks
    headers = _admin_headers()
    admin_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=admin_id, name="Cross Table Agent")
    linked_task_id = _create_task_row(
        user_id=admin_id,
        title="External task with a linked row",
        source="external",
        is_visible=False,
        agent_id=agent_id,
    )
    # Stands in for the deployment's widget-session linkage table.
    _attach_trigger_run(
        user_id=admin_id,
        agent_id=agent_id,
        task_id=linked_task_id,
        trigger_type="scheduled",
        source_event_id="evt-linked",
    )
    unlinked_task_id = _create_task_row(
        user_id=admin_id,
        title="External task without a linked row",
        source="external",
        is_visible=False,
        agent_id=agent_id,
    )

    hooks.set_external_task_source_hook(lambda _db: [(build_predicate(), "widget")])

    response = client.get("/api/conversation-logs", headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source_counts"] == {
        "all": 2,
        "widget": 1,
        "rest_api": 1,
        "shared_link": 0,
        "webhook": 0,
    }
    by_id = {item["task_id"]: item["source"] for item in body["logs"]}
    assert by_id == {linked_task_id: "widget", unlinked_task_id: "rest_api"}

    linked_detail = client.get(
        f"/api/conversation-logs/{linked_task_id}", headers=headers
    )
    assert linked_detail.status_code == 200, linked_detail.text
    assert linked_detail.json()["log"]["source"] == "widget"
    unlinked_detail = client.get(
        f"/api/conversation-logs/{unlinked_task_id}", headers=headers
    )
    assert unlinked_detail.status_code == 200, unlinked_detail.text
    assert unlinked_detail.json()["log"]["source"] == "rest_api"


def test_source_hook_returning_none_means_no_branches(
    _reset_external_task_hooks: Any, caplog: pytest.LogCaptureFixture
) -> None:
    hooks = _reset_external_task_hooks
    headers = _admin_headers()
    admin_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=admin_id, name="None Hook Agent")
    task_id = _create_task_row(
        user_id=admin_id,
        title="External task behind a None-returning hook",
        source="external",
        is_visible=False,
        agent_id=agent_id,
    )
    hooks.set_external_task_source_hook(lambda _db: None)

    with caplog.at_level(logging.WARNING, logger="xagent.web.api.conversation_logs"):
        response = client.get("/api/conversation-logs", headers=headers)
        detail = client.get(f"/api/conversation-logs/{task_id}", headers=headers)

    assert response.status_code == 200, response.text
    assert response.json()["source_counts"]["rest_api"] == 1
    assert detail.status_code == 200, detail.text
    assert detail.json()["log"]["source"] == "rest_api"
    assert not [
        r
        for r in caplog.records
        if r.name == "xagent.web.api.conversation_logs" and r.levelno >= logging.WARNING
    ]


def test_validator_rejects_non_boolean_and_cross_table_predicates(
    _reset_external_task_hooks: Any,
) -> None:
    from xagent.web.api.conversation_logs import (
        _validated_external_source_branches,
    )

    hooks = _reset_external_task_hooks
    valid = Task.agent_config["widget_session_id"].as_string().isnot(None)
    hooks.set_external_task_source_hook(
        lambda _db: [
            (Task.title, "widget"),
            (Task.agent_config["widget_session_id"].as_string(), "widget"),
            (Task.id == TriggerRun.task_id, "widget"),
            (aliased(Task).source == "external", "widget"),
            (exists().where(TriggerRun.task_id == Task.id), "shared_link"),
            (Task.is_visible, "rest_api"),
            (valid, "widget"),
        ]
    )

    db = _direct_db_session()
    try:
        branches = _validated_external_source_branches(db)
    finally:
        db.close()

    assert [ui_source for _, ui_source in branches] == [
        "shared_link",
        "rest_api",
        "widget",
    ]
    # A bare boolean ORM column is accepted once unwrapped to its Column.
    assert str(branches[1][0]) == "tasks.is_visible"
    assert branches[2][0] is valid


@contextmanager
def _count_session_rollbacks():
    count = {"rollbacks": 0}

    def _after_rollback(_session: Session) -> None:
        count["rollbacks"] += 1

    event.listen(Session, "after_rollback", _after_rollback)
    try:
        yield count
    finally:
        event.remove(Session, "after_rollback", _after_rollback)


def test_hook_db_failure_rolls_back_the_request_session(
    _reset_external_task_hooks: Any,
) -> None:
    hooks = _reset_external_task_hooks
    headers = _admin_headers()
    admin_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=admin_id, name="Bad Query Hook Agent")
    task_id = _create_task_row(
        user_id=admin_id,
        title="External task behind a hook that breaks its transaction",
        source="external",
        is_visible=False,
        agent_id=agent_id,
    )

    def _bad_query_source_hook(db: Session) -> list[tuple[Any, str]]:
        db.execute(text("SELECT 1 FROM no_such_table"))
        return []

    def _bad_query_context_hook(db: Session, _task: Task, _ui: str) -> None:
        db.execute(text("SELECT 1 FROM no_such_table"))
        return None

    hooks.set_external_task_source_hook(_bad_query_source_hook)
    hooks.set_external_task_context_hook(_bad_query_context_hook)

    # On PostgreSQL the failed statement aborts the request transaction, so the
    # fail-soft path must roll back before the endpoint issues its own queries.
    with _count_session_rollbacks() as count:
        response = client.get("/api/conversation-logs", headers=headers)
    assert response.status_code == 200, response.text
    assert response.json()["source_counts"]["rest_api"] == 1
    assert count["rollbacks"] >= 1

    with _count_session_rollbacks() as count:
        detail = client.get(f"/api/conversation-logs/{task_id}", headers=headers)
    assert detail.status_code == 200, detail.text
    assert detail.json()["log"]["source"] == "rest_api"
    assert detail.json()["metadata"]["public_context"] is None
    # One rollback for the source hook, one for the context hook.
    assert count["rollbacks"] >= 2


def test_mcp_actor_tasks_stay_out_of_conversation_logs() -> None:
    headers = _admin_headers()
    admin_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=admin_id, name="Actor Agent")
    actor_task_id = _create_task_row(
        user_id=admin_id,
        title="MCP OAuth negotiation turn",
        source="external",
        is_visible=False,
        agent_id=agent_id,
        agent_config={MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY: True},
    )
    session_task_id = _create_task_row(
        user_id=admin_id,
        title="Session transport conversation",
        source="external",
        is_visible=False,
        agent_id=agent_id,
        agent_config={MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY: False},
    )
    plain_task_id = _create_task_row(
        user_id=admin_id,
        title="Session transport conversation without the key",
        source="external",
        is_visible=False,
        agent_id=agent_id,
    )

    response = client.get("/api/conversation-logs", headers=headers)
    assert response.status_code == 200, response.text
    assert {item["task_id"] for item in response.json()["logs"]} == {
        session_task_id,
        plain_task_id,
    }
    assert response.json()["source_counts"]["all"] == 2

    detail = client.get(f"/api/conversation-logs/{actor_task_id}", headers=headers)
    assert detail.status_code == 404, detail.text


# --- Expired conversation logs (#2565) ---------------------------------------
#
# The purge refuses SQLite as a job, so most of these insert the tombstone the
# purge would have left; some run ``purge_task`` itself (via
# ``expired_task_shared.purge_conversation``) to pin the whole chain.
# Tombstone-only ids are far above anything a test creates, so no live task
# can hold them by accident.

_UNUSED_TASK_ID = 900_000


def _assert_not_found(response: Any) -> None:
    # Exactly the answer for an id that never existed, so the response cannot
    # be used to probe which ids retention expired.
    assert response.status_code == 404, response.text
    assert response.json() == {"detail": "Conversation log not found"}


def test_purged_conversation_log_detail_is_expired_and_absent_from_the_list() -> None:
    admin = _admin_headers()
    admin_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=admin_id, name="Expiring Agent")
    task_id = _create_task_row(
        user_id=admin_id,
        title="Conversation retention expires",
        source="widget",
        is_visible=False,
        agent_id=agent_id,
    )
    listed = client.get("/api/conversation-logs", headers=admin)
    assert listed.status_code == 200, listed.text
    assert [log["task_id"] for log in listed.json()["logs"]] == [task_id]

    now = purge_conversation(task_id)

    detail = client.get(f"/api/conversation-logs/{task_id}", headers=admin)
    assert_task_expired(detail, task_id, expired_at=now)

    # Lists are deliberately unchanged: an expired task is simply gone from
    # them, counts and agent filter options included.
    listed = client.get("/api/conversation-logs", headers=admin)
    assert listed.status_code == 200, listed.text
    assert listed.json()["logs"] == []
    assert listed.json()["source_counts"]["all"] == 0
    assert listed.json()["agents"] == []


def test_expired_log_detail_is_disclosed_to_the_owner_and_admins_only() -> None:
    admin = _admin_headers()
    owner = _register_second_user(username="expiredowner")
    stranger = _register_second_user(username="expiredstranger")
    task_id = _UNUSED_TASK_ID + 1
    insert_tombstone(task_id=task_id, user_id=_user_id("expiredowner"))

    assert_task_expired(
        client.get(f"/api/conversation-logs/{task_id}", headers=owner), task_id
    )
    # Admins read hidden external logs across users live, so they are told
    # about an expired one across users too.
    assert_task_expired(
        client.get(f"/api/conversation-logs/{task_id}", headers=admin), task_id
    )
    _assert_not_found(client.get(f"/api/conversation-logs/{task_id}", headers=stranger))


@pytest.mark.parametrize(
    ("tombstone", "reason"),
    [
        (
            {"source": "external", "is_channel_plumbing": True},
            "MCP actor/OAuth turns are channel plumbing, never a conversation log",
        ),
        (
            {"source": "sdk", "is_visible": True},
            "visible tasks are not hidden external conversation logs",
        ),
        (
            {"source": "internal"},
            "internal tasks are outside the external-source scope",
        ),
        (
            {"source": None},
            "a task without a source is outside the external-source scope",
        ),
        (
            {"source": "trigger", "trigger_type": "scheduled"},
            "a scheduled trigger task was never a Conversation Logs webhook entry",
        ),
        (
            {"source": "trigger", "trigger_type": "gmail"},
            "a Gmail trigger task was never a Conversation Logs webhook entry",
        ),
        (
            {"source": "trigger", "trigger_type": None},
            "a trigger task with no recorded trigger type cannot be told apart "
            "from a scheduled or Gmail one",
        ),
    ],
)
def test_expired_task_outside_the_log_scope_is_not_found(
    tombstone: dict[str, Any], reason: str
) -> None:
    admin = _admin_headers()
    task_id = _UNUSED_TASK_ID + 2
    insert_tombstone(task_id=task_id, user_id=_user_id("admin"), **tombstone)

    response = client.get(f"/api/conversation-logs/{task_id}", headers=admin)

    assert response.status_code == 404, reason
    _assert_not_found(response)


def test_expired_direct_and_external_sources_are_in_scope() -> None:
    admin = _admin_headers()
    admin_id = _user_id("admin")
    for offset, source in enumerate(("sdk", "widget", "shared_link", "external")):
        task_id = _UNUSED_TASK_ID + 10 + offset
        insert_tombstone(task_id=task_id, user_id=admin_id, source=source)
        assert_task_expired(
            client.get(f"/api/conversation-logs/{task_id}", headers=admin), task_id
        )


def test_expired_webhook_trigger_source_is_in_scope() -> None:
    """The stored ``trigger_type`` (#2565 follow-up) lets a webhook trigger
    tombstone answer 410 like every other in-scope source, instead of every
    ``trigger``-sourced tombstone falling back to 404.
    """
    admin = _admin_headers()
    admin_id = _user_id("admin")
    task_id = _UNUSED_TASK_ID + 20
    insert_tombstone(
        task_id=task_id, user_id=admin_id, source="trigger", trigger_type="webhook"
    )

    assert_task_expired(
        client.get(f"/api/conversation-logs/{task_id}", headers=admin), task_id
    )


def test_expired_webhook_trigger_conversation_log_is_reported_end_to_end() -> None:
    """A real purge of a webhook trigger task's conversation reports 410.

    Unlike ``test_expired_webhook_trigger_source_is_in_scope``, which inserts
    a tombstone directly, this drives the actual ``purge_task`` path so the
    trigger type it captures before nulling ``TriggerRun.task_id`` is what the
    detail route reads back.
    """
    admin = _admin_headers()
    admin_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=admin_id, name="Webhook Retention Agent")

    def _purge(*, trigger_type: str, source_event_id: str) -> tuple[int, datetime]:
        task_id = _create_task_row(
            user_id=admin_id,
            title=f"{trigger_type} trigger conversation",
            source="trigger",
            is_visible=False,
            agent_id=agent_id,
        )
        _attach_trigger_run(
            user_id=admin_id,
            agent_id=agent_id,
            task_id=task_id,
            trigger_type=trigger_type,
            source_event_id=source_event_id,
        )
        return task_id, purge_conversation(task_id)

    webhook_task_id, webhook_expired_at = _purge(
        trigger_type="webhook", source_event_id="evt-purge-webhook"
    )
    webhook_detail = client.get(
        f"/api/conversation-logs/{webhook_task_id}", headers=admin
    )
    assert_task_expired(webhook_detail, webhook_task_id, expired_at=webhook_expired_at)

    # Cheap to pin alongside: a scheduled trigger's purged conversation stays
    # not-found, since this page never served it live either.
    scheduled_task_id, _ = _purge(
        trigger_type="scheduled", source_event_id="evt-purge-scheduled"
    )
    _assert_not_found(
        client.get(f"/api/conversation-logs/{scheduled_task_id}", headers=admin)
    )


def test_live_log_wins_over_a_tombstone_with_the_same_id() -> None:
    admin = _admin_headers()
    admin_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=admin_id, name="Reused Id Agent")
    task_id = _create_task_row(
        user_id=admin_id,
        title="Live conversation",
        source="sdk",
        is_visible=False,
        agent_id=agent_id,
    )
    # SQLite hands a deleted id to the next insert, so a tombstone and a live
    # task can share one; the live task is what the id means now.
    insert_tombstone(task_id=task_id, user_id=admin_id)

    response = client.get(f"/api/conversation-logs/{task_id}", headers=admin)

    assert response.status_code == 200, response.text
    assert response.json()["log"]["title"] == "Live conversation"


def test_live_task_outside_the_log_scope_stays_not_found_despite_a_tombstone() -> None:
    """A live task the page does not serve is not an expired one.

    ``find_expired_task`` yields nothing while a live row holds the id, so a
    visible task that shares an id with a tombstone stays a plain 404.
    """
    admin = _admin_headers()
    admin_id = _user_id("admin")
    task_id = _create_task_row(
        user_id=admin_id, title="Visible task", source="sdk", is_visible=True
    )
    insert_tombstone(task_id=task_id, user_id=admin_id)

    _assert_not_found(client.get(f"/api/conversation-logs/{task_id}", headers=admin))


def test_tombstones_never_appear_in_the_list_or_its_counts() -> None:
    admin = _admin_headers()
    admin_id = _user_id("admin")
    agent_id = _create_agent_row(user_id=admin_id, name="Listed Agent")
    live_id = _create_task_row(
        user_id=admin_id,
        title="Still here",
        source="sdk",
        is_visible=False,
        agent_id=agent_id,
    )
    insert_tombstone(task_id=_UNUSED_TASK_ID + 20, user_id=admin_id, agent_id=agent_id)
    insert_tombstone(
        task_id=_UNUSED_TASK_ID + 21,
        user_id=admin_id,
        source="widget",
        agent_id=agent_id,
    )

    for source in ("all", "rest_api", "widget"):
        response = client.get(
            "/api/conversation-logs", params={"source": source}, headers=admin
        )
        assert response.status_code == 200, response.text
        body = response.json()
        expected = [live_id] if source in ("all", "rest_api") else []
        assert [log["task_id"] for log in body["logs"]] == expected
        assert body["source_counts"]["all"] == 1
        assert body["source_counts"]["rest_api"] == 1
        assert body["source_counts"]["widget"] == 0


class _Cell(NamedTuple):
    """One row of the live/tombstone parity matrix.

    ``trigger_runs`` is ``(trigger_type, source_event_id)`` pairs attached in
    order, oldest first, so the last entry is the newest ``TriggerRun``.
    """

    label: str
    source: str | None
    is_visible: bool
    agent_config: dict[str, Any] | None = None
    trigger_runs: tuple[tuple[str, str], ...] = ()


_PARITY_CELLS = [
    _Cell("visible_sdk", "sdk", True),
    _Cell("hidden_sdk", "sdk", False),
    _Cell("hidden_widget", "widget", False),
    _Cell("hidden_shared_link", "shared_link", False),
    _Cell("hidden_internal", "internal", False),
    _Cell("hidden_no_source", None, False),
    _Cell(
        "external_mcp_true",
        "external",
        False,
        agent_config={MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY: True},
    ),
    _Cell(
        "external_mcp_false",
        "external",
        False,
        agent_config={MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY: False},
    ),
    _Cell("external_mcp_absent", "external", False),
    _Cell(
        "trigger_webhook_run",
        "trigger",
        False,
        trigger_runs=(("webhook", "evt-matrix-webhook"),),
    ),
    _Cell(
        "trigger_scheduled_run",
        "trigger",
        False,
        trigger_runs=(("scheduled", "evt-matrix-scheduled"),),
    ),
    _Cell(
        "trigger_gmail_run",
        "trigger",
        False,
        trigger_runs=(("gmail", "evt-matrix-gmail"),),
    ),
    _Cell("trigger_no_run_no_config", "trigger", False),
    _Cell(
        "trigger_config_scheduled_run_webhook",
        "trigger",
        False,
        agent_config={"trigger_type": "scheduled"},
        trigger_runs=(("webhook", "evt-matrix-config-wins"),),
    ),
    _Cell(
        "trigger_two_runs_newest_wins",
        "trigger",
        False,
        trigger_runs=(
            ("webhook", "evt-matrix-older"),
            ("scheduled", "evt-matrix-newer"),
        ),
    ),
]


@pytest.mark.parametrize(
    "cell", _PARITY_CELLS, ids=[cell.label for cell in _PARITY_CELLS]
)
def test_live_and_tombstone_predicates_agree(cell: _Cell) -> None:
    """Matrix over task shape x caller, driven by the real purge (#2565).

    ``_apply_external_task_scope`` / ``_ui_source_for_task`` (the live gates)
    and ``_tombstone_in_external_task_scope`` / ``_tombstone_has_ui_source``
    (their tombstone mirrors) are two separate implementations of the same
    predicate. This checks agreement, not a hardcoded expectation per cell:
    for every caller, the post-purge status is 410 iff the live status was
    200, and otherwise the exact not-found body -- so the two can never
    silently drift apart. Whichever of the four functions changes, this test
    must change with it.
    """
    admin_headers = _admin_headers()
    owner_headers = _register_second_user(username=f"matrix-owner-{cell.label}")
    other_headers = _register_second_user(username=f"matrix-other-{cell.label}")
    owner_id = _user_id(f"matrix-owner-{cell.label}")

    agent_id = _create_agent_row(user_id=owner_id, name=f"Matrix Agent {cell.label}")
    task_id = _create_task_row(
        user_id=owner_id,
        title=f"Matrix cell {cell.label}",
        source=cell.source,
        is_visible=cell.is_visible,
        agent_id=agent_id,
        agent_config=cell.agent_config,
    )
    for trigger_type, source_event_id in cell.trigger_runs:
        _attach_trigger_run(
            user_id=owner_id,
            agent_id=agent_id,
            task_id=task_id,
            trigger_type=trigger_type,
            source_event_id=source_event_id,
        )

    callers = {
        "owner": owner_headers,
        "other": other_headers,
        "admin": admin_headers,
    }
    live_status: dict[str, bool] = {}
    for name, headers in callers.items():
        live = client.get(f"/api/conversation-logs/{task_id}", headers=headers)
        assert live.status_code in (200, 404), f"{cell.label}/{name}: {live.text}"
        live_status[name] = live.status_code == 200

    now = purge_conversation(task_id)

    for name, headers in callers.items():
        after = client.get(f"/api/conversation-logs/{task_id}", headers=headers)
        case = f"{cell.label}/{name}"
        if live_status[name]:
            assert_task_expired(after, task_id, expired_at=now)
        else:
            assert after.status_code == 404, f"{case}: {after.text}"
            _assert_not_found(after)
