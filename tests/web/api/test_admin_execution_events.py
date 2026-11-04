"""Admin execution-event diagnostics: correctness, auth, and payload secrecy."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from importlib import import_module
from types import SimpleNamespace
from typing import Any

import sqlalchemy as sa
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from tests.web.services.task_database_shared import engine as engine_fixture
from tests.web.services.task_database_shared import task_id as task_id_fixture
from xagent.web.api import admin_execution_events as api_module
from xagent.web.api.admin_execution_events import router
from xagent.web.auth_dependencies import get_current_user
from xagent.web.models.database import get_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_auto_recovery import TaskAutoRecovery, TaskRecoveryEvent
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.services import execution_event_diagnostics as diagnostics_module
from xagent.web.services.execution_event_diagnostics import (
    task_execution_event_diagnostics,
)
from xagent.web.services.ops_signals import (
    clear_degradation,
    register_degradation,
)
from xagent.web.services.task_execution_event_writer import append_fact_no_commit

engine = engine_fixture
task_id = task_id_fixture

SENTINEL = "T3A-SENTINEL"
T0 = datetime(2026, 9, 5, tzinfo=timezone.utc)


def fact(
    db: Session,
    task_id: int,
    kind: str,
    key: str,
    *,
    scope: str = "root",
    run: str | None = None,
    attempt: str | None = None,
) -> None:
    append_fact_no_commit(
        db,
        task_id=task_id,
        scope_id=scope,
        run_id=run,
        tool_attempt_id=attempt,
        assistant_message_id="batch-1" if attempt else None,
        kind=kind,
        key=key,
        payload={"data": {"secret": SENTINEL, "tool_params": {"q": SENTINEL}}},
        occurred_at=T0 + timedelta(seconds=len(key)),
    )


def make_v2(db: Session, task_id: int) -> None:
    task = db.get(Task, task_id)
    task.conversation_storage_version = 2
    db.flush()


def diagnose(engine, task_id: int) -> dict[str, Any]:
    with Session(engine) as db:
        result = task_execution_event_diagnostics(db, task_id)
    assert result is not None
    return result


def build_client(engine, *, is_admin: bool) -> TestClient:
    app = FastAPI()
    app.include_router(router)

    def test_db():
        with Session(engine) as db:
            yield db

    app.dependency_overrides[get_db] = test_db
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
        id=1, is_admin=is_admin
    )
    return TestClient(app)


def url(task_id: int) -> str:
    return f"/api/admin/execution-events/tasks/{task_id}"


def seed_two_runs(engine, task_id: int) -> None:
    with Session(engine) as db:
        make_v2(db, task_id)
        fact(db, task_id, "input_accepted", "in-1", run="r1")
        fact(db, task_id, "recovery_state", "rs-1", run="r1")
        fact(db, task_id, "execution_settled", "result-r1", run="r1")
        fact(db, task_id, "recovery_state", "rs-2", run="r2")
        fact(db, task_id, "agent_message", "msg-1", run="r2")
        fact(db, task_id, "recovery_state", "rs-child", scope="child-1", run="r2")
        fact(db, task_id, "assistant_message", "am-1", scope="child-1", run="r2")
        fact(db, task_id, "input_accepted", "in-2")
        db.commit()


def test_non_admin_is_forbidden(engine, task_id):
    response = build_client(engine, is_admin=False).get(url(task_id))
    assert response.status_code == 403
    assert response.json()["detail"] == "Admin access required"


def test_missing_task_is_404(engine, task_id):
    response = build_client(engine, is_admin=True).get(url(task_id + 1000))
    assert response.status_code == 404


def test_router_is_registered_in_the_app():
    app_module = import_module("xagent.web.app")
    # No ``with``: no lifespan startup. An unauthenticated call is refused by
    # the auth dependency; an unregistered path would be 404.
    response = TestClient(app_module.app).get(url(1))
    assert response.status_code in (401, 403)


def test_v2_task_counts_runs_scopes_and_recovery_states(engine, task_id):
    seed_two_runs(engine, task_id)
    with Session(engine) as db:
        task = db.get(Task, task_id)
        task.status = TaskStatus.RUNNING
        task.run_id = "r2"
        task.runner_id = "runner-a"
        task.lease_attempt_id = "lease-1"
        task.lease_expires_at = T0
        task.last_heartbeat_at = T0
        task.error_message = f"boom {SENTINEL}"
        db.commit()

    result = build_client(engine, is_admin=True).get(url(task_id)).json()

    assert result["storage"] == "v2"
    assert result["task"] == {
        "id": task_id,
        "conversation_storage_version": 2,
        "status": "running",
        "control_state": "idle",
        "run_id": "r2",
        "state_version": 0,
        "conversation_event_sequence": 8,
        "runner_id": "runner-a",
        "lease_attempt_id": "lease-1",
        "lease_expires_at": result["task"]["lease_expires_at"],
        "last_heartbeat_at": result["task"]["last_heartbeat_at"],
        "has_error_message": True,
    }
    assert result["task"]["lease_expires_at"].startswith("2026-09-05")
    assert "error_message" not in result["task"]

    events = result["events"]
    assert (events["total"], events["min_sequence"], events["max_sequence"]) == (
        8,
        1,
        8,
    )
    assert events["latest_occurred_at"] is not None
    assert events["by_scope_kind_truncated"] is False
    assert events["by_scope_kind"] == [
        {"scope_id": "child-1", "kind": "assistant_message", "count": 1},
        {"scope_id": "child-1", "kind": "recovery_state", "count": 1},
        {"scope_id": "root", "kind": "agent_message", "count": 1},
        {"scope_id": "root", "kind": "execution_settled", "count": 1},
        {"scope_id": "root", "kind": "input_accepted", "count": 2},
        {"scope_id": "root", "kind": "recovery_state", "count": 2},
    ]

    assert result["runs"]["truncated"] is False
    assert result["runs"]["items"] == [
        {
            "run_id": None,
            "event_count": 1,
            "min_sequence": 8,
            "max_sequence": 8,
            "settled": False,
        },
        {
            "run_id": "r2",
            "event_count": 4,
            "min_sequence": 4,
            "max_sequence": 7,
            "settled": False,
        },
        {
            "run_id": "r1",
            "event_count": 3,
            "min_sequence": 1,
            "max_sequence": 3,
            "settled": True,
        },
    ]
    assert result["recovery_states"] == {
        "items": [
            {"scope_id": "child-1", "max_sequence": 6},
            {"scope_id": "root", "max_sequence": 4},
        ],
        "truncated": False,
    }
    assert result["open_tool_attempts"] == {
        "open_count": 0,
        "truncated": False,
        "items": [],
    }
    assert result["auto_recovery"] == {
        "state": None,
        "events": [],
        "events_truncated": False,
    }


def test_open_tool_attempts_follow_the_recovery_reader_definition(engine, task_id):
    with Session(engine) as db:
        make_v2(db, task_id)
        # Start only: effect unknown.
        fact(db, task_id, "tool_execution_start", "tool:A:start", run="r1", attempt="A")
        # Start + end, start + failed: committed outcomes.
        fact(db, task_id, "tool_execution_start", "tool:B:start", run="r1", attempt="B")
        fact(db, task_id, "tool_execution_end", "tool:B:end", run="r1", attempt="B")
        fact(db, task_id, "tool_execution_start", "tool:C:start", run="r1", attempt="C")
        fact(db, task_id, "tool_execution_failed", "tool:C:fail", run="r1", attempt="C")
        # A later run's settlement end settles the original start, exactly as
        # read_committed_tool_outcome counts any end row for the attempt.
        fact(db, task_id, "tool_execution_start", "tool:D:start", run="r1", attempt="D")
        fact(
            db,
            task_id,
            "tool_execution_end",
            "tool:D:settlement:tool_execution_end",
            run="r2",
            attempt="D",
        )
        # A settlement start with no end of any kind stays open. The original
        # run id sorts after the settlement run's, so the reported run must
        # come from the start row, not from an aggregate over run ids.
        fact(db, task_id, "tool_execution_start", "tool:E:start", run="r9", attempt="E")
        fact(
            db,
            task_id,
            "tool_execution_start",
            "tool:E:settlement:tool_execution_start",
            run="r2",
            attempt="E",
        )
        # The same attempt id in another scope is a different attempt.
        fact(
            db,
            task_id,
            "tool_execution_start",
            "tool:A:start",
            scope="child-1",
            run="r1",
            attempt="A",
        )
        fact(
            db,
            task_id,
            "tool_execution_end",
            "tool:A:end",
            scope="child-1",
            run="r1",
            attempt="A",
        )
        # An end with no start is not a candidate.
        fact(db, task_id, "tool_execution_end", "tool:F:end", run="r1", attempt="F")
        # A non-tool fact never carries an attempt id and is ignored.
        fact(db, task_id, "agent_message", "msg-1", run="r1")
        db.commit()

    result = diagnose(engine, task_id)["open_tool_attempts"]

    assert result["open_count"] == 2
    assert result["truncated"] is False
    assert result["items"] == [
        {
            "tool_attempt_id": "E",
            "scope_id": "root",
            "run_id": "r9",
            "start_sequence": 8,
            "kinds": ["tool_execution_start"],
        },
        {
            "tool_attempt_id": "A",
            "scope_id": "root",
            "run_id": "r1",
            "start_sequence": 1,
            "kinds": ["tool_execution_start"],
        },
    ]


def test_open_tool_attempts_report_truncation(engine, task_id, monkeypatch):
    monkeypatch.setattr(diagnostics_module, "MAX_OPEN_TOOL_ATTEMPTS", 2)
    with Session(engine) as db:
        make_v2(db, task_id)
        for attempt in ("A", "B", "C"):
            fact(
                db,
                task_id,
                "tool_execution_start",
                f"tool:{attempt}:start",
                run="r1",
                attempt=attempt,
            )
        db.commit()

    result = diagnose(engine, task_id)["open_tool_attempts"]

    assert result["open_count"] == 3
    assert result["truncated"] is True
    assert [item["tool_attempt_id"] for item in result["items"]] == ["C", "B"]


def test_lists_report_truncation(engine, task_id, monkeypatch):
    for name in (
        "MAX_EVENT_GROUPS",
        "MAX_RUNS",
        "MAX_RECOVERY_STATE_SCOPES",
        "MAX_RECOVERY_EVENTS",
    ):
        monkeypatch.setattr(diagnostics_module, name, 1)
    seed_two_runs(engine, task_id)
    with Session(engine) as db:
        for event in ("scheduled", "auto_resumed"):
            db.add(TaskRecoveryEvent(task_id=task_id, run_id="r2", event=event))
        db.commit()

    result = diagnose(engine, task_id)

    assert len(result["events"]["by_scope_kind"]) == 1
    assert result["events"]["by_scope_kind_truncated"] is True
    assert len(result["runs"]["items"]) == 1
    assert result["runs"]["truncated"] is True
    assert len(result["recovery_states"]["items"]) == 1
    assert result["recovery_states"]["truncated"] is True
    assert [e["event"] for e in result["auto_recovery"]["events"]] == ["auto_resumed"]
    assert result["auto_recovery"]["events_truncated"] is True
    # Totals are aggregates, not list lengths, so they are not capped.
    assert result["events"]["total"] == 8


def test_auto_recovery_state_events_and_only_whether_last_error_is_set(engine, task_id):
    with Session(engine) as db:
        db.add(
            TaskAutoRecovery(
                task_id=task_id,
                run_id="r1",
                reason="runner_lost",
                state="scheduled",
                state_detail="backoff",
                paused_state_version=3,
                interrupted_at=T0,
                episode_started_at=T0,
                no_progress_resumes=1,
                total_resumes=2,
                next_attempt_at=T0,
                last_command_id="cmd-1",
                last_error=f"{SENTINEL} Bearer sk-not-a-real-token",
            )
        )
        db.add(TaskRecoveryEvent(task_id=task_id, run_id="r1", event="interrupted"))
        db.add(
            TaskRecoveryEvent(
                task_id=task_id,
                run_id="r1",
                event="scheduled",
                reason="runner_lost",
                attempt=1,
                next_attempt_at=T0,
                detail={"note": SENTINEL},
            )
        )
        db.commit()

    result = diagnose(engine, task_id)["auto_recovery"]

    state = result["state"]
    assert state["state"] == "scheduled"
    assert state["reason"] == "runner_lost"
    assert state["state_detail"] == "backoff"
    assert (state["no_progress_resumes"], state["total_resumes"]) == (1, 2)
    # The run's error text is the same content as error_message: hidden.
    assert state["has_last_error"] is True
    assert "last_error" not in state
    assert [e["event"] for e in result["events"]] == ["scheduled", "interrupted"]
    assert result["events"][0]["attempt"] == 1
    assert "detail" not in result["events"][0]
    assert SENTINEL not in json.dumps(result)


def test_v1_task_returns_empty_sections(engine, task_id):
    result = build_client(engine, is_admin=True).get(url(task_id)).json()

    assert result["storage"] == "v1"
    assert result["task"]["conversation_storage_version"] == 1
    assert result["task"]["has_error_message"] is False
    assert result["events"]["total"] == 0
    assert result["events"]["min_sequence"] is None
    assert result["events"]["by_scope_kind"] == []
    assert result["runs"]["items"] == []
    assert result["recovery_states"]["items"] == []
    assert result["open_tool_attempts"]["open_count"] == 0
    assert result["auto_recovery"]["state"] is None


def test_payload_content_never_appears_in_the_response(engine, task_id):
    seed_two_runs(engine, task_id)
    with Session(engine) as db:
        fact(db, task_id, "tool_execution_start", "tool:A:start", run="r2", attempt="A")
        task = db.get(Task, task_id)
        task.error_message = f"boom {SENTINEL}"
        db.add(
            TaskAutoRecovery(
                task_id=task_id,
                run_id="r2",
                reason="runner_lost",
                state="manual",
                paused_state_version=1,
                interrupted_at=T0,
                episode_started_at=T0,
                last_error=f"{SENTINEL} connection reset",
            )
        )
        db.add(
            TaskRecoveryEvent(
                task_id=task_id,
                run_id="r2",
                event="interrupted",
                detail={"note": SENTINEL},
            )
        )
        db.commit()
    with Session(engine) as db:
        stored = db.execute(
            sa.text(
                "SELECT count(*) FROM task_execution_events "
                "WHERE CAST(payload AS TEXT) LIKE :s"
            ),
            {"s": f"%{SENTINEL}%"},
        ).scalar()
    assert stored == 9

    response = build_client(engine, is_admin=True).get(url(task_id))

    assert response.status_code == 200
    assert SENTINEL not in response.text
    assert response.json()["task"]["has_error_message"] is True
    assert response.json()["open_tool_attempts"]["open_count"] == 1


def test_diagnostics_never_reference_the_payload_column(engine, task_id):
    seed_two_runs(engine, task_id)
    statements: list[str] = []

    def record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    sa.event.listen(engine, "before_cursor_execute", record)
    try:
        diagnose(engine, task_id)
    finally:
        sa.event.remove(engine, "before_cursor_execute", record)

    assert any("task_execution_events" in s for s in statements)
    assert not [s for s in statements if "payload" in s.lower()]


def test_database_error_returns_503_without_leaking_details(
    engine, task_id, monkeypatch, caplog
):
    def explode(db, requested_task_id):
        raise OperationalError("SELECT secret", {"p": SENTINEL}, Exception(SENTINEL))

    monkeypatch.setattr(api_module, "task_execution_event_diagnostics", explode)

    with caplog.at_level(logging.WARNING, logger=api_module.logger.name):
        response = build_client(engine, is_admin=True).get(url(task_id))

    assert response.status_code == 503
    assert SENTINEL not in response.text
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert f"task {task_id}" in messages
    assert "OperationalError" in messages
    assert SENTINEL not in messages
    assert not any(r.exc_info for r in caplog.records)


def test_degradations_are_reported_for_this_process(engine, task_id):
    register_degradation("checkpoint_load_unavailable", "task_id=1: detail")
    try:
        result = diagnose(engine, task_id)
    finally:
        clear_degradation("checkpoint_load_unavailable")

    assert "checkpoint_load_unavailable" in result["degradations"]


def test_event_sections_read_a_snapshot_at_the_task_horizon(engine, task_id):
    """Events above the horizon read first belong to a later snapshot.

    Each append advances the horizon in its own transaction, so a row above
    the horizon read by the request is one committed after that read. It is
    simulated here by inserting rows without advancing the horizon.
    """
    with Session(engine) as db:
        make_v2(db, task_id)
        fact(db, task_id, "tool_execution_start", "tool:A:start", run="r1", attempt="A")
        db.commit()
        horizon = db.get(Task, task_id).conversation_event_sequence
        for offset, (kind, key, attempt) in enumerate(
            [
                ("tool_execution_end", "tool:A:end", "A"),
                ("tool_execution_start", "tool:B:start", "B"),
                ("execution_settled", "settled:r1", None),
                ("recovery_state", "recovery:r1", None),
            ],
            start=1,
        ):
            db.add(
                TaskExecutionEvent(
                    event_id=f"later-{offset}",
                    task_id=task_id,
                    sequence=horizon + offset,
                    scope_id="root",
                    run_id="r1",
                    tool_attempt_id=attempt,
                    idempotency_key=key,
                    kind=kind,
                    payload_version=1,
                    payload={},
                    occurred_at=T0,
                )
            )
        db.commit()

    result = diagnose(engine, task_id)

    assert result["task"]["conversation_event_sequence"] == horizon
    assert result["events"]["total"] == horizon
    assert result["events"]["max_sequence"] == horizon
    assert result["runs"]["items"][0]["settled"] is False
    assert result["recovery_states"]["items"] == []
    # A's end and B's start are after the snapshot: A is still open, B absent.
    assert result["open_tool_attempts"]["open_count"] == 1
    assert [i["tool_attempt_id"] for i in result["open_tool_attempts"]["items"]] == [
        "A"
    ]
