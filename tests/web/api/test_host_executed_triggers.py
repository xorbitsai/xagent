"""Host-executed trigger runs: a registered executor claims runs of triggers
whose ``executor`` field names its kind; everything else behaves as before."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

from xagent.web.models.task import Task
from xagent.web.models.trigger import AgentTrigger, TriggerRun, TriggerRunStatus
from xagent.web.models.user import User
from xagent.web.services import trigger_executors
from xagent.web.services.trigger_executors import (
    HostTriggerRun,
    TriggerExecutorAlreadyRegisteredError,
    register_trigger_executor,
    unregister_trigger_executor,
)
from xagent.web.services.triggers import (
    _start_host_trigger_run_id,
    create_agent_trigger,
    dispatch_pending_trigger_runs,
    fire_trigger,
    scan_due_scheduled_triggers,
)

from .conftest import _admin_headers, _direct_db_session, client
from .test_agent_triggers import _create_agent, _signed_webhook_headers

pytestmark = pytest.mark.usefixtures("_test_db")

CHANNEL_KINDS = ["slack", "teams", "feishu"]


@pytest.fixture(autouse=True)
def _isolated_registry() -> Iterator[None]:
    with trigger_executors._lock:
        saved = dict(trigger_executors._executors)
        trigger_executors._executors.clear()
    with patch("xagent.web.services.task_orchestrator._schedule_bg", new=MagicMock()):
        yield
    with trigger_executors._lock:
        trigger_executors._executors.clear()
        trigger_executors._executors.update(saved)


class Recorder:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[HostTriggerRun] = []
        self.error = error

    async def __call__(self, run: HostTriggerRun) -> None:
        self.calls.append(run)
        if self.error is not None:
            raise self.error


def _make_trigger(
    *,
    executor: str | None,
    trigger_type: str = "webhook",
    config: dict | None = None,
) -> int:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.username == "admin").one()
        trigger, _ = create_agent_trigger(
            db,
            user_id=int(user.id),
            agent_id=agent_id,
            trigger_type=trigger_type,
            name="host trigger",
            config=config,
            prompt_template="event={{payload}}",
            executor=executor,
        )
        return int(trigger.id)
    finally:
        db.close()


def _fire(trigger_id: int, **kwargs):
    db = _direct_db_session()
    try:
        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
        return asyncio.run(fire_trigger(db, trigger=trigger, **kwargs))
    finally:
        db.close()


def _run_row(run_id: int) -> TriggerRun:
    db = _direct_db_session()
    try:
        run = db.query(TriggerRun).filter(TriggerRun.id == run_id).one()
        db.expunge(run)
        return run
    finally:
        db.close()


@pytest.mark.parametrize("kind", CHANNEL_KINDS)
def test_host_executor_receives_event_once_and_run_completes(kind: str) -> None:
    recorder = Recorder()
    register_trigger_executor(kind, recorder)
    trigger_id = _make_trigger(executor=kind)

    run, created = _fire(
        trigger_id,
        event_payload={"subject": "hello"},
        source_event_id="evt-1",
        event_type="message",
        resource_id="res-1",
        wait_for_completion=True,
    )

    assert created is True
    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call.run_id == int(run.id)
    assert call.trigger_id == trigger_id
    assert call.executor == kind
    assert call.payload == {"subject": "hello"}
    assert call.source_event_id == "evt-1"
    assert call.event_type == "message"
    assert call.resource_id == "res-1"
    assert call.trigger_type == "webhook"
    assert call.test is False
    assert "hello" in call.prompt
    row = _run_row(int(run.id))
    assert row.status == TriggerRunStatus.COMPLETED.value
    assert row.task_id is None
    assert row.finished_at is not None
    db = _direct_db_session()
    try:
        assert db.query(Task).filter(Task.source == "trigger").count() == 0
    finally:
        db.close()


def test_host_executor_failure_is_recorded_on_run_and_trigger() -> None:
    recorder = Recorder(error=RuntimeError("boom"))
    register_trigger_executor("hostx", recorder)
    trigger_id = _make_trigger(executor="hostx")

    run, _ = _fire(
        trigger_id,
        event_payload={"a": 1},
        source_event_id="evt-fail",
        wait_for_completion=True,
    )

    row = _run_row(int(run.id))
    assert row.status == TriggerRunStatus.FAILED.value
    assert row.error_message == "RuntimeError: boom"
    db = _direct_db_session()
    try:
        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
        assert trigger.last_error == "RuntimeError: boom"
    finally:
        db.close()


def test_duplicate_event_executes_once() -> None:
    recorder = Recorder()
    register_trigger_executor("hostx", recorder)
    trigger_id = _make_trigger(executor="hostx")

    first, created_first = _fire(
        trigger_id,
        event_payload={"a": 1},
        source_event_id="evt-dup",
        wait_for_completion=True,
    )
    second, created_second = _fire(
        trigger_id,
        event_payload={"a": 1},
        source_event_id="evt-dup",
        wait_for_completion=True,
    )

    assert created_first is True and created_second is False
    assert int(first.id) == int(second.id)
    assert len(recorder.calls) == 1


def test_concurrent_claims_hand_the_run_over_exactly_once() -> None:
    recorder = Recorder()
    register_trigger_executor("hostx", recorder)
    trigger_id = _make_trigger(executor="hostx")
    # Prepared but not started: no executor registered while preparing is not
    # needed, so prepare through the unregistered path then start twice.
    unregister_trigger_executor("hostx")
    run, _ = _fire(trigger_id, event_payload={"a": 1}, source_event_id="evt-race")
    assert _run_row(int(run.id)).status == TriggerRunStatus.PENDING.value
    register_trigger_executor("hostx", recorder)

    async def start_twice() -> list[bool]:
        return list(
            await asyncio.gather(
                _start_host_trigger_run_id(
                    int(run.id), "hostx", wait_for_completion=True
                ),
                _start_host_trigger_run_id(
                    int(run.id), "hostx", wait_for_completion=True
                ),
            )
        )

    results = asyncio.run(start_twice())
    assert sorted(results) == [False, True]
    assert len(recorder.calls) == 1
    assert _run_row(int(run.id)).status == TriggerRunStatus.COMPLETED.value


def test_unregistered_kind_stays_pending_until_dispatcher_with_executor() -> None:
    trigger_id = _make_trigger(executor="hostx")
    run, created = _fire(
        trigger_id, event_payload={"a": 1}, source_event_id="evt-pending"
    )
    assert created is True
    row = _run_row(int(run.id))
    assert row.status == TriggerRunStatus.PENDING.value
    assert row.task_id is None

    async def dispatch() -> int:
        db = _direct_db_session()
        try:
            return await dispatch_pending_trigger_runs(db, wait_for_completion=True)
        finally:
            db.close()

    assert asyncio.run(dispatch()) == 0
    assert _run_row(int(run.id)).status == TriggerRunStatus.PENDING.value

    recorder = Recorder()
    register_trigger_executor("hostx", recorder)
    assert asyncio.run(dispatch()) == 1
    assert len(recorder.calls) == 1
    assert recorder.calls[0].payload == {"a": 1}
    assert _run_row(int(run.id)).status == TriggerRunStatus.COMPLETED.value
    # A second dispatch tick must not run it again.
    assert asyncio.run(dispatch()) == 0
    assert len(recorder.calls) == 1


def test_scheduled_trigger_dispatches_to_host_executor() -> None:
    recorder = Recorder()
    register_trigger_executor("hostx", recorder)
    trigger_id = _make_trigger(
        executor="hostx",
        trigger_type="scheduled",
        config={"interval_seconds": 3600},
    )
    db = _direct_db_session()
    try:
        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
        trigger.next_run_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        db.commit()
        runs = scan_due_scheduled_triggers(db)
        assert len(runs) == 1
        assert runs[0].task_id is None
        assert runs[0].status == TriggerRunStatus.PENDING.value
        assert asyncio.run(dispatch_pending_trigger_runs(db, wait_for_completion=True))
    finally:
        db.close()

    assert len(recorder.calls) == 1
    assert recorder.calls[0].event_type == "scheduled"
    assert recorder.calls[0].trigger_type == "scheduled"
    assert _run_row(recorder.calls[0].run_id).status == TriggerRunStatus.COMPLETED.value


def test_triggers_without_executor_keep_default_agent_task_path() -> None:
    recorder = Recorder()
    register_trigger_executor("hostx", recorder)
    trigger_id = _make_trigger(executor=None)

    run, created = _fire(
        trigger_id, event_payload={"a": 1}, source_event_id="evt-default"
    )

    assert created is True
    assert recorder.calls == []
    row = _run_row(int(run.id))
    assert row.task_id is not None
    db = _direct_db_session()
    try:
        assert db.query(Task).filter(Task.source == "trigger").count() == 1
    finally:
        db.close()


def test_webhook_callback_pipeline_hands_run_to_host_executor() -> None:
    recorder = Recorder()
    register_trigger_executor("hostx", recorder)
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "pipeline"},
    )
    body = created.json()
    db = _direct_db_session()
    try:
        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == body["id"]).one()
        trigger.executor = "hostx"
        db.commit()
    finally:
        db.close()

    url = f"/api/triggers/callback/webhook/{body['callback_id']}"
    raw_body = b'{"subject": "hello"}'
    event_headers = _signed_webhook_headers(
        body["webhook_secret"], raw_body, event_id="evt-pipe"
    )
    with client:
        first = client.post(url, headers=event_headers, content=raw_body)
        second = client.post(url, headers=event_headers, content=raw_body)

    assert first.status_code == 200, first.text
    assert first.json()["outcome"] == "accepted"
    assert second.json()["duplicates"] == 1
    run_id = first.json()["trigger_run_ids"][0]
    assert [call.run_id for call in recorder.calls] == [run_id]
    assert recorder.calls[0].payload == {"subject": "hello"}
    db = _direct_db_session()
    try:
        assert db.query(Task).filter(Task.source == "trigger").count() == 0
    finally:
        db.close()


def test_registry_rejects_conflicts_and_blank_kinds() -> None:
    first, second = Recorder(), Recorder()
    register_trigger_executor("hostx", first)
    register_trigger_executor("hostx", first)  # same callable is idempotent
    with pytest.raises(TriggerExecutorAlreadyRegisteredError):
        register_trigger_executor("hostx", second)
    register_trigger_executor("hostx", second, replace=True)
    assert trigger_executors.get_trigger_executor("hostx") is second
    with pytest.raises(ValueError):
        register_trigger_executor("  ", first)
    assert trigger_executors.get_trigger_executor(None) is None
