from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI

from tests.shared.db_teardown import drop_all_tables
from xagent.web import app as app_module
from xagent.web.models.agent import Agent
from xagent.web.models.database import (
    Base,
    configure_db,
    get_engine,
    get_session_local,
)
from xagent.web.models.trigger import AgentTrigger, TriggerType
from xagent.web.models.user import User
from xagent.web.startup_admission import register_host_startup_admission


def _patch_runtime_starts(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
) -> None:
    monkeypatch.setattr(
        app_module, "register_local_browser_runtime", lambda: events.append("runtime")
    )

    from xagent.web.api.websocket import background_task_manager

    monkeypatch.setattr(
        background_task_manager,
        "start_accepting",
        lambda: events.append("task admission"),
    )
    for name, event in (
        ("start_file_storage_startup_sync_task", "file sync"),
        ("start_trigger_dispatcher_task", "trigger dispatcher"),
        ("start_task_lease_recovery_task", "lease recovery"),
        ("start_uploaded_file_recovery_task", "file recovery"),
        ("start_orphan_upload_gc_task", "upload gc"),
    ):
        monkeypatch.setattr(
            app_module,
            name,
            lambda _app, event=event: events.append(event),
        )


@pytest.mark.asyncio
async def test_no_host_admission_preserves_runtime_startup_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_app = FastAPI()
    events: list[str] = []
    monkeypatch.setattr(app_module, "init_db", lambda: events.append("database"))
    _patch_runtime_starts(monkeypatch, events)

    await app_module._initialize_database_and_admit_runtime(test_app)

    assert events == [
        "database",
        "runtime",
        "task admission",
        "file sync",
        "trigger dispatcher",
        "lease recovery",
        "file recovery",
        "upload gc",
    ]


@pytest.mark.asyncio
async def test_host_admissions_run_after_database_and_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_app = FastAPI()
    events: list[str] = []
    monkeypatch.setattr(app_module, "init_db", lambda: events.append("database"))
    _patch_runtime_starts(monkeypatch, events)

    async def first() -> None:
        events.append("first admission")

    async def second() -> None:
        events.append("second admission")

    register_host_startup_admission(test_app, first)
    register_host_startup_admission(test_app, second)

    await app_module._initialize_database_and_admit_runtime(test_app)

    assert events[:5] == [
        "database",
        "first admission",
        "second admission",
        "runtime",
        "task admission",
    ]


@pytest.mark.asyncio
async def test_host_admission_stops_at_first_error_and_propagates_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_app = FastAPI()
    events: list[str] = []
    rejection = RuntimeError("host rejected startup")
    monkeypatch.setattr(app_module, "init_db", lambda: events.append("database"))
    _patch_runtime_starts(monkeypatch, events)

    async def first() -> None:
        events.append("first admission")

    async def reject() -> None:
        events.append("rejected admission")
        raise rejection

    async def never_runs() -> None:
        events.append("late admission")

    register_host_startup_admission(test_app, first)
    register_host_startup_admission(test_app, reject)
    register_host_startup_admission(test_app, never_runs)

    with pytest.raises(RuntimeError) as raised:
        await app_module._initialize_database_and_admit_runtime(test_app)

    assert raised.value is rejection
    assert events == ["database", "first admission", "rejected admission"]


@pytest.mark.asyncio
async def test_failed_admission_with_due_trigger_launches_no_background_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    database_url = f"sqlite:///{tmp_path / 'startup-admission.db'}"
    configure_db(database_url)
    Base.metadata.create_all(bind=get_engine())
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        user = User(username="admission-test", password_hash="hash")
        db.add(user)
        db.flush()
        agent = Agent(user_id=user.id, name="Admission test agent")
        db.add(agent)
        db.flush()
        due_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        trigger = AgentTrigger(
            user_id=user.id,
            agent_id=agent.id,
            type=TriggerType.SCHEDULED.value,
            name="Due before startup",
            config={"interval_seconds": 60},
            next_run_at=due_at,
        )
        db.add(trigger)
        db.commit()
        trigger_id = trigger.id

    test_app = FastAPI()
    runtime_starts: list[str] = []
    observed_due_trigger: list[int] = []

    def initialize_existing_database() -> None:
        configure_db(database_url)

    async def reject_after_checking_database() -> None:
        SessionLocal = get_session_local()
        with SessionLocal() as db:
            due = (
                db.query(AgentTrigger)
                .filter(
                    AgentTrigger.id == trigger_id,
                    AgentTrigger.next_run_at <= datetime.now(timezone.utc),
                )
                .one()
            )
            observed_due_trigger.append(due.id)
        raise RuntimeError("admission denied")

    def unexpected_create_task(coroutine):
        coroutine.close()
        raise AssertionError("startup launched a background task before admission")

    monkeypatch.setattr(app_module, "app", test_app)
    monkeypatch.setattr(app_module, "init_db", initialize_existing_database)
    monkeypatch.setattr(
        app_module, "validate_interaction_rollout_at_startup", lambda: None
    )
    monkeypatch.setattr(app_module.asyncio, "create_task", unexpected_create_task)
    _patch_runtime_starts(monkeypatch, runtime_starts)
    register_host_startup_admission(test_app, reject_after_checking_database)

    try:
        with pytest.raises(RuntimeError, match="admission denied"):
            await app_module.startup_event()

        assert observed_due_trigger == [trigger_id]
        assert runtime_starts == []
        with SessionLocal() as db:
            persisted = db.get(AgentTrigger, trigger_id)
            assert persisted is not None
            assert persisted.last_run_at is None
    finally:
        drop_all_tables(get_engine())
