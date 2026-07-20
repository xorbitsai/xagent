from __future__ import annotations

import secrets
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

import pytest
from sqlalchemy import event

from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import get_engine
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.trigger import AgentTrigger, TriggerRun, TriggerRunStatus
from xagent.web.models.user import User

from .conftest import (
    _admin_headers,
    _direct_db_session,
    _register_second_user,
    client,
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
    assert max(message_query_param_counts) <= 5


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
