"""Automatic resume after a crash, through real hosts and the real sweeper.

A worker (or a combined host) is killed with SIGKILL while its run waits on
the model. Lease recovery on the surviving hosts pauses the run and records
what automatic resume will do; the sweeper then dispatches the resume and a
replacement worker carries the same run to its end, unless something stopped
it first: a pause the user had requested, a message the user sent meanwhile,
a later tick of the same scheduled trigger, or the switch being off.

Backoff and polling are shortened so a resume is due about a second after the
lease (10s) expires.
"""

from __future__ import annotations

import json
import os
import signal
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from tests.e2e.shared_execution_harness import (
    SETTLE_TIMEOUT,
    auto_resume_commands,
    kill_host,
    model_calls,
    receive_event,
    recovery_state,
    restart_hosts,
    start_task,
    task_state,
    wait_for,
    ws_url,
)

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def _fast_auto_resume(monkeypatch):
    # Read by the hosts the shared_app fixture spawns after this.
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_SHORT_INITIAL_BACKOFF_SECONDS", "1")
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_POLL_SECONDS", "1")
    monkeypatch.delenv("XAGENT_TASK_AUTO_RESUME_ENABLED", raising=False)
    monkeypatch.delenv("XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED", raising=False)


def _assert_crash_is_resumed(app, task_id: int, run_id: str) -> None:
    settled = app.wait_task(task_id, timeout=SETTLE_TIMEOUT)
    assert settled["run_id"] == run_id
    assert "Shared E2E answer" in (settled["output"] or "")
    # The killed call and the resumed one: nothing ran twice.
    assert len(model_calls(app)) == 2
    row, events = recovery_state(task_id)
    assert row is not None
    assert (row["reason"], row["run_id"], row["total_resumes"]) == (
        "lease_expired",
        run_id,
        1,
    )
    assert events[:3] == ["interrupted", "scheduled", "auto_resumed"]
    assert len(auto_resume_commands(task_id)) == 1


# --------------------------------------------------------------------------
# E1 / E1': a killed run finishes without its user
# --------------------------------------------------------------------------


def test_killed_worker_run_resumes_automatically(shared_app):
    app = shared_app
    task_id = start_task(app)
    run_id = task_state(task_id)["run_id"]

    kill_host(app, app.processes[1])
    # The resumed call may answer; the killed one never returns.
    (app.root / "model-release").touch()
    app.start("worker")

    _assert_crash_is_resumed(app, task_id, run_id)
    assert {call["role"] for call in model_calls(app)} == {"worker"}


def test_killed_combined_host_run_resumes_automatically(shared_app):
    app = shared_app
    restart_hosts(app, "combined")
    task_id = start_task(app)
    run_id = task_state(task_id)["run_id"]

    kill_host(app, app.processes[0])
    (app.root / "model-release").touch()
    app.start("combined")

    _assert_crash_is_resumed(app, task_id, run_id)
    assert {call["role"] for call in model_calls(app)} == {"combined"}


# --------------------------------------------------------------------------
# E6: a pause the user requested is never undone
# --------------------------------------------------------------------------


def test_run_killed_while_pause_requested_stays_paused(shared_app):
    from xagent.web.models.database import get_session_local
    from xagent.web.models.task import Task

    app = shared_app
    task_id = start_task(app)
    worker = app.processes[1]
    # Freeze the owner so the pause cannot complete, then record the user's
    # request as the pause command would: the run dies PAUSE_REQUESTED.
    os.kill(worker.pid, signal.SIGSTOP)
    try:
        with get_session_local()() as db:
            task = db.get(Task, task_id)
            assert task.control_state == "running"
            task.control_state = "pause_requested"
            db.commit()
    finally:
        kill_host(app, worker)
    (app.root / "model-release").touch()
    app.start("worker")

    app.wait_task(task_id, status="paused", timeout=SETTLE_TIMEOUT)
    row, events = recovery_state(task_id)
    assert row is not None
    assert (row["reason"], row["state"]) == ("user_pause", "manual")
    # Well past the first backoff: nothing resumes it.
    time.sleep(5)
    assert task_state(task_id)["status"] == "paused"
    assert recovery_state(task_id)[1] == ["interrupted"] == events
    assert auto_resume_commands(task_id) == []
    assert len(model_calls(app)) == 1


# --------------------------------------------------------------------------
# E7: a message while the resume is scheduled wins
# --------------------------------------------------------------------------


def test_message_while_scheduled_makes_the_resume_stale(shared_app):
    from websockets.sync.client import connect

    app = shared_app
    # A wide window between recording and dispatch.
    restart_hosts(
        app,
        "web",
        "worker",
        XAGENT_TASK_AUTO_RESUME_SHORT_INITIAL_BACKOFF_SECONDS="20",
    )
    task_id = start_task(app)
    killed_run_id = task_state(task_id)["run_id"]
    kill_host(app, app.processes[1])
    (app.root / "model-release").touch()
    app.start("worker")

    wait_for(
        lambda: (recovery_state(task_id)[0] or {}).get("state") == "scheduled",
        app,
        "the scheduled resume",
    )
    with connect(ws_url(app, task_id), additional_headers=app.headers) as ws:
        receive_event(ws, "historical_data_complete")
        ws.send(
            json.dumps(
                {
                    "type": "chat",
                    "message": "Please continue",
                    "client_message_id": "while-scheduled",
                }
            )
        )
        receive_event(ws, "message_accepted")
    # The message starts a new run on the paused task.
    settled = app.wait_task(task_id, timeout=SETTLE_TIMEOUT)
    assert settled["run_id"] != killed_run_id

    # Once due, the sweeper finds the task moved on and stands down.
    wait_for(
        lambda: (recovery_state(task_id)[0] or {}).get("state") == "stale",
        app,
        "the stale resume",
    )
    _row, events = recovery_state(task_id)
    assert "auto_resumed" not in events
    assert auto_resume_commands(task_id) == []
    # The killed call and the user's continuation; never the sweeper's.
    assert len(model_calls(app)) == 2


# --------------------------------------------------------------------------
# E8: a later tick of the scheduled trigger supersedes the killed one
# --------------------------------------------------------------------------


def _due_now(trigger_id: int) -> None:
    from xagent.web.models.database import get_session_local
    from xagent.web.models.trigger import AgentTrigger

    with get_session_local()() as db:
        db.get(AgentTrigger, trigger_id).next_run_at = datetime.now(
            timezone.utc
        ) - timedelta(seconds=5)
        db.commit()


def _trigger_runs(trigger_id: int) -> list[dict[str, Any]]:
    from xagent.web.models.database import get_session_local
    from xagent.web.models.trigger import TriggerRun

    with get_session_local()() as db:
        return [
            {
                "id": run.id,
                "task_id": run.task_id,
                "status": run.status,
                "error": run.error_message,
                "started": run.started_at is not None,
            }
            for run in db.query(TriggerRun)
            .filter_by(trigger_id=trigger_id)
            .order_by(TriggerRun.id)
        ]


def test_later_scheduled_tick_supersedes_the_killed_one(shared_app):
    from xagent.web.services.task_auto_resume import (
        TASK_AUTO_RESUME_SUPERSEDED_TRIGGER_ERROR,
    )

    app = shared_app
    agent_id, _ = app.create_agent()
    response = app.client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=app.headers,
        json={
            "type": "scheduled",
            # In the rendered prompt, so each tick's model call waits.
            "name": "e2e:gate tick",
            "config": {"interval_seconds": 3600},
        },
    )
    assert response.status_code == 200, response.text
    trigger_id = response.json()["id"]
    _due_now(trigger_id)
    first = wait_for(
        lambda: next(
            (run for run in _trigger_runs(trigger_id) if run["task_id"]), None
        ),
        app,
        "the first tick",
    )
    wait_for(lambda: (app.root / "model-entered").exists(), app, "model call")

    kill_host(app, app.processes[1])
    # Before the worker restarts: the web host's dispatcher claims the next
    # tick on its own, so it counts as started well before the killed run's
    # lease expires, however long the replacement worker takes to boot.
    _due_now(trigger_id)
    app.start("worker")
    wait_for(
        lambda: [
            run
            for run in _trigger_runs(trigger_id)
            if run["id"] > first["id"] and run["started"] and run["status"] == "running"
        ],
        app,
        "the next tick to start",
    )

    wait_for(
        lambda: (
            (recovery_state(first["task_id"])[0] or {}).get("state") == "superseded"
        ),
        app,
        "the superseded resume",
    )
    assert task_state(first["task_id"])["status"] == "paused"
    assert "superseded" in recovery_state(first["task_id"])[1]
    (killed,) = [run for run in _trigger_runs(trigger_id) if run["id"] == first["id"]]
    assert (killed["status"], killed["error"]) == (
        "failed",
        TASK_AUTO_RESUME_SUPERSEDED_TRIGGER_ERROR,
    )
    assert auto_resume_commands(first["task_id"]) == []

    # The later tick is unaffected and completes.
    (app.root / "model-release").touch()
    later = [run for run in _trigger_runs(trigger_id) if run["id"] > first["id"]]
    app.wait_task(later[0]["task_id"], timeout=SETTLE_TIMEOUT)
    assert task_state(first["task_id"])["status"] == "paused"


# --------------------------------------------------------------------------
# E10a: with the switch off a killed run waits for its user
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "infra_pause", [None, "true"], ids=["infra-default", "infra-on"]
)
def test_switch_off_leaves_a_killed_run_paused(shared_app, infra_pause):
    # Unset, the infrastructure-failure pause switch follows automatic resume
    # (off). Explicitly on, it changes nothing here (E10b-ii): lease expiry
    # is recorded either way, and with automatic resume off it waits for the
    # user.
    app = shared_app
    switches = {"XAGENT_TASK_AUTO_RESUME_ENABLED": "false"}
    if infra_pause is not None:
        switches["XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED"] = infra_pause
    restart_hosts(app, "web", "worker", **switches)
    task_id = start_task(app)
    kill_host(app, app.processes[1])
    (app.root / "model-release").touch()
    app.start("worker")

    app.wait_task(task_id, status="paused", timeout=SETTLE_TIMEOUT)
    row, events = recovery_state(task_id)
    assert row is not None
    assert (row["state"], row["state_detail"]) == ("manual", "auto_resume_disabled")
    time.sleep(5)
    assert task_state(task_id)["status"] == "paused"
    assert recovery_state(task_id)[1] == events == ["interrupted"]
    assert auto_resume_commands(task_id) == []
    assert len(model_calls(app)) == 1
