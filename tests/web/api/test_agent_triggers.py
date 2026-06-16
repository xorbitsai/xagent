from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.trigger import AgentTrigger, TriggerRun, TriggerRunStatus
from xagent.web.services.task_orchestrator import TurnStarted, finish_turn
from xagent.web.services.triggers import (
    _compute_next_run_at,
    _start_prepared_trigger_run_id,
    dispatch_pending_trigger_runs,
    scan_due_scheduled_triggers,
)

from .conftest import _admin_headers, _direct_db_session, client

pytestmark = pytest.mark.usefixtures("_test_db")


@pytest.fixture(autouse=True)
def mock_bg_scheduler():
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ) as mocked:
        yield mocked


def _create_agent(headers: dict[str, str], name: str = "Trigger Agent") -> int:
    resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": name,
            "description": "test",
            "instructions": "You are a trigger test agent.",
            "execution_mode": "balanced",
        },
    )
    assert resp.status_code == 200, resp.text
    return int(resp.json()["id"])


def test_webhook_trigger_crud_returns_secret_once() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)

    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "webhook",
            "name": "Inbound webhook",
            "prompt_template": "payload={{payload}}",
            "config": {"source": "crm"},
        },
    )
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["type"] == "webhook"
    assert body["webhook_token"]
    assert body["webhook_secret"]

    db = _direct_db_session()
    try:
        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == body["id"]).one()
        assert str(trigger.secret_hash).startswith("$2")
        assert trigger.secret_hash != body["webhook_secret"]
    finally:
        db.close()

    listed = client.get(f"/api/agents/{agent_id}/triggers", headers=headers)
    assert listed.status_code == 200, listed.text
    assert len(listed.json()) == 1
    assert listed.json()[0]["webhook_secret"] is None

    patched = client.patch(
        f"/api/agents/{agent_id}/triggers/{body['id']}",
        headers=headers,
        json={"name": "Renamed webhook", "rotate_secret": True},
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["name"] == "Renamed webhook"
    assert patched.json()["webhook_secret"]


def test_trigger_test_run_creates_hidden_agent_task(mock_bg_scheduler) -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "webhook",
            "name": "Test webhook",
            "prompt_template": "Handle {{payload}}",
        },
    )
    trigger_id = created.json()["id"]

    fired = client.post(
        f"/api/agents/{agent_id}/triggers/{trigger_id}/test",
        headers=headers,
        json={"payload": {"subject": "hello"}, "source_event_id": "test-event"},
    )
    assert fired.status_code == 200, fired.text
    run_body = fired.json()["trigger_run"]
    assert run_body["status"] == TriggerRunStatus.RUNNING.value
    assert run_body["task_id"]
    assert fired.json()["duplicate"] is False
    assert mock_bg_scheduler.call_count == 1

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == run_body["task_id"]).one()
        assert task.agent_id == agent_id
        assert task.source == "trigger"
        assert task.is_visible is False
        assert task.status == TaskStatus.RUNNING
        assert "hello" in (task.description or "")
    finally:
        db.close()


def test_public_webhook_validates_secret_and_deduplicates(mock_bg_scheduler) -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "Public webhook"},
    )
    body = created.json()
    url = f"/api/triggers/webhook/{body['webhook_token']}"

    rejected = client.post(url, json={"subject": "hello"})
    assert rejected.status_code == 401

    event_headers = {
        "x-xagent-trigger-secret": body["webhook_secret"],
        "x-xagent-event-id": "evt-1",
    }
    first = client.post(url, headers=event_headers, json={"subject": "hello"})
    assert first.status_code == 200, first.text
    assert set(first.json()) == {"trigger_run_id", "status", "duplicate"}
    assert first.json()["duplicate"] is False

    second = client.post(url, headers=event_headers, json={"subject": "hello"})
    assert second.status_code == 200, second.text
    assert second.json()["duplicate"] is True
    assert second.json()["trigger_run_id"] == first.json()["trigger_run_id"]
    assert mock_bg_scheduler.call_count == 1

    db = _direct_db_session()
    try:
        assert db.query(TriggerRun).count() == 1
        assert db.query(Task).filter(Task.source == "trigger").count() == 1
    finally:
        db.close()


def test_public_webhook_invalid_utf8_body_falls_back_to_text(
    mock_bg_scheduler,
) -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "Invalid UTF-8 webhook"},
    )
    body = created.json()
    url = f"/api/triggers/webhook/{body['webhook_token']}"

    fired = client.post(
        url,
        headers={"x-xagent-trigger-secret": body["webhook_secret"]},
        content=b'\xff{"subject":"hello"}',
    )
    assert fired.status_code == 200, fired.text

    db = _direct_db_session()
    try:
        run_id = fired.json()["trigger_run_id"]
        run = db.query(TriggerRun).filter(TriggerRun.id == run_id).one()
        assert run.payload_snapshot == {"body": '\ufffd{"subject":"hello"}'}
    finally:
        db.close()

    assert mock_bg_scheduler.call_count == 1


def test_trigger_name_validation_rejects_empty_and_oversized() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)

    empty = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "   "},
    )
    assert empty.status_code == 400

    oversized = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "x" * 201},
    )
    assert oversized.status_code == 422

    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "Valid"},
    )
    assert created.status_code == 200, created.text

    patched = client.patch(
        f"/api/agents/{agent_id}/triggers/{created.json()['id']}",
        headers=headers,
        json={"name": " "},
    )
    assert patched.status_code == 400


def test_scheduled_next_run_skips_stale_intervals_without_iteration() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    stale_due_at = now - timedelta(days=3650)

    next_run_at = _compute_next_run_at(
        {"interval_seconds": 1},
        from_time=now,
        previous_due_at=stale_due_at,
        include_explicit=False,
    )

    assert next_run_at == now + timedelta(seconds=1)


def test_scheduled_scan_fires_due_trigger_and_advances_next_run(
    mock_bg_scheduler,
) -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Every minute",
            "config": {"interval_seconds": 60},
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = created.json()["id"]

    due_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    db = _direct_db_session()
    try:
        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
        trigger.next_run_at = due_at
        db.add(trigger)
        db.commit()

        runs = scan_due_scheduled_triggers(db, now=datetime.now(timezone.utc))
        assert len(runs) == 1
        db.refresh(trigger)
        assert trigger.next_run_at is not None
        next_run_at = trigger.next_run_at
        if next_run_at.tzinfo is None:
            next_run_at = next_run_at.replace(tzinfo=timezone.utc)
        assert next_run_at > due_at
        run = db.query(TriggerRun).filter(TriggerRun.id == runs[0].id).one()
        assert run.status == TriggerRunStatus.PENDING.value
        assert run.task_id is not None
        task = db.query(Task).filter(Task.id == run.task_id).one()
        assert task.agent_id == agent_id
        assert task.source == "trigger"
        assert task.is_visible is False
        assert task.status == TaskStatus.PENDING

        assert mock_bg_scheduler.call_count == 0
        assert asyncio.run(dispatch_pending_trigger_runs(db)) == 1
        db.refresh(run)
        db.refresh(task)
        assert run.status == TriggerRunStatus.RUNNING.value
        assert task.status == TaskStatus.RUNNING
    finally:
        db.close()

    assert mock_bg_scheduler.call_count == 1


def test_dispatch_claims_pending_trigger_run_once_under_concurrency() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Concurrent scheduled",
            "config": {"interval_seconds": 60},
        },
    )
    assert created.status_code == 200, created.text

    db = _direct_db_session()
    try:
        trigger = (
            db.query(AgentTrigger).filter(AgentTrigger.id == created.json()["id"]).one()
        )
        trigger.next_run_at = datetime.now(timezone.utc) - timedelta(seconds=5)
        db.add(trigger)
        db.commit()
        runs = scan_due_scheduled_triggers(db, now=datetime.now(timezone.utc))
        assert len(runs) == 1
        run_id = int(runs[0].id)
    finally:
        db.close()

    begin_calls = 0

    async def fake_begin_turn(**kwargs):
        nonlocal begin_calls
        begin_calls += 1
        await asyncio.sleep(0.05)

        async def done() -> None:
            return None

        return TurnStarted(
            task_id=int(kwargs["task_id"]),
            status=TaskStatus.RUNNING,
            updated_at=None,
            before_message_id=None,
            task_source="trigger",
            background_task=asyncio.create_task(done()),
        )

    async def start_twice() -> list[bool]:
        first, second = await asyncio.gather(
            _start_prepared_trigger_run_id(run_id),
            _start_prepared_trigger_run_id(run_id),
        )
        return [first, second]

    with patch(
        "xagent.web.services.triggers.TaskTurnOrchestrator.begin_turn",
        new=fake_begin_turn,
    ):
        results = asyncio.run(start_twice())

    assert results.count(True) == 1
    assert begin_calls == 1


def test_scheduled_scan_disables_one_shot_trigger(mock_bg_scheduler) -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    due_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "One shot",
            "config": {"next_run_at": due_at.isoformat()},
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = created.json()["id"]

    db = _direct_db_session()
    try:
        runs = scan_due_scheduled_triggers(db, now=datetime.now(timezone.utc))
        assert len(runs) == 1

        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
        assert trigger.enabled is False
        assert trigger.next_run_at is None
        run = db.query(TriggerRun).filter(TriggerRun.id == runs[0].id).one()
        assert run.status == TriggerRunStatus.PENDING.value
    finally:
        db.close()

    assert mock_bg_scheduler.call_count == 0


def test_finish_turn_syncs_trigger_run_status() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "Completion webhook"},
    )
    trigger_id = created.json()["id"]

    fired = client.post(
        f"/api/agents/{agent_id}/triggers/{trigger_id}/test",
        headers=headers,
        json={"payload": {"subject": "done"}},
    )
    run_body = fired.json()["trigger_run"]

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == run_body["task_id"]).one()
        task.status = TaskStatus.COMPLETED
        db.add(
            TaskChatMessage(
                task_id=int(task.id),
                user_id=int(task.user_id),
                role="assistant",
                content="done",
                message_type="assistant_message",
            )
        )
        db.add(task)
        db.commit()

        finish_turn(db, int(task.id))

        run = db.query(TriggerRun).filter(TriggerRun.id == run_body["id"]).one()
        assert run.status == TriggerRunStatus.COMPLETED.value
        assert run.finished_at is not None
        assert run.error_message is None
    finally:
        db.close()
