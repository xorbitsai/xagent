"""Automatic resume after database faults, through real hosts on PostgreSQL.

The web host and the worker reach PostgreSQL through a controllable TCP link;
this process connects directly, so it can observe and arrange the database
while the hosts cannot reach it.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy as sa

from tests.e2e.shared_execution_harness import (
    SETTLE_TIMEOUT,
    assistant_messages,
    auto_resume_commands,
    model_calls,
    open_shared_app,
    recovery_state,
    start_task,
    task_state,
    wait_for,
)
from tests.e2e.tcp_link import ControllableTcpLink

pytestmark = [pytest.mark.e2e, pytest.mark.docker]

ANSWER = "Shared E2E answer"
ONE_RESUME = ["interrupted", "scheduled", "auto_resumed"]

# Armed per task and event type, this drops the connection that inserts the
# task's next matching trace event, once: the sequence spends its one shot
# even though the dropped transaction rolls back.
_DROP_CONNECTION_ON_CHECKPOINT = (
    "CREATE TABLE e2e_write_fault ("
    " task_id integer PRIMARY KEY, event_type text NOT NULL)",
    "CREATE SEQUENCE e2e_write_fault_hits",
    """
    CREATE FUNCTION e2e_drop_connection_on_checkpoint() RETURNS trigger
    LANGUAGE plpgsql AS $$
    BEGIN
      -- Nested IFs on purpose: SQL AND does not fix the evaluation order, and
      -- a nextval() on a row that does not match would spend the one shot.
      IF EXISTS (SELECT 1 FROM e2e_write_fault f
                 WHERE f.task_id = NEW.task_id
                   AND f.event_type = NEW.event_type) THEN
        IF nextval('e2e_write_fault_hits') = 1 THEN
          PERFORM pg_terminate_backend(pg_backend_pid());
          -- The pending terminate fires at this sleep's interrupt check.
          PERFORM pg_sleep(5);
        END IF;
      END IF;
      RETURN NEW;
    END $$
    """,
    "CREATE TRIGGER e2e_drop_connection BEFORE INSERT ON trace_events"
    " FOR EACH ROW EXECUTE FUNCTION e2e_drop_connection_on_checkpoint()",
)


@pytest.fixture(autouse=True)
def _fast_auto_resume(monkeypatch):
    # Read by the hosts the pg_shared_app fixture spawns after this.
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_POLL_SECONDS", "1")
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_SHORT_INITIAL_BACKOFF_SECONDS", "1")
    # A checkout must fail rather than wait out the default 30s pool timeout.
    monkeypatch.setenv("XAGENT_DB_POOL_TIMEOUT_SECONDS", "5")
    for name in (
        "XAGENT_TASK_AUTO_RESUME_ENABLED",
        "XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED",
        "XAGENT_TASK_AUTO_RESUME_MAX_TOTAL_PER_RUN",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def pg_link(postgres_url):
    link = ControllableTcpLink(postgres_url, default_port=5432)
    try:
        yield link
    finally:
        link.close()


@pytest.fixture
def pg_shared_app(tmp_path, monkeypatch, postgres_url, pg_link):
    with open_shared_app(
        tmp_path,
        monkeypatch,
        database_url=postgres_url,
        host_database_url=pg_link.url,
    ) as app:
        try:
            yield app
        finally:
            # The hosts must reach the database to shut down cleanly.
            pg_link.restore()


def _run_id(app, task_id: int) -> str:
    return wait_for(lambda: task_state(task_id)["run_id"], app, "the run")


def _execute(*statements: str, **parameters) -> list[tuple]:
    """Run statements on this process's direct connection; the last's rows."""
    from xagent.web.models.database import get_engine

    with get_engine().begin() as connection:
        for statement in statements:
            result = connection.execute(sa.text(statement), parameters)
        return list(result) if result.returns_rows else []


def _release_model(app) -> None:
    (app.root / "model-release").touch()


def _assert_answered_once(task_id: int) -> None:
    """One answer reached the history, and no failure bubble."""
    messages = assistant_messages(task_id)
    assert len([m for m in messages if ANSWER in m[1]]) == 1, messages
    assert not [m for m in messages if m[0] == "task_failure"], messages


def test_task_completes_on_postgresql_without_faults(pg_shared_app):
    app = pg_shared_app
    task_id = start_task(app, "e2e postgres")
    run_id = _run_id(app, task_id)

    settled = app.wait_task(task_id, run_id=run_id, timeout=SETTLE_TIMEOUT)
    assert ANSWER in (settled["output"] or "")
    assert settled["error"] is None
    assert len(model_calls(app)) == 1
    assert recovery_state(task_id) == (None, [])
    assert auto_resume_commands(task_id) == []


# --------------------------------------------------------------------------
# E2a: the connection drops mid-write and the database is back at once
# --------------------------------------------------------------------------


def test_connection_lost_mid_checkpoint_pauses_and_resumes(pg_shared_app):
    from xagent.core.agent.checkpoint import CHECKPOINT_EVENT_TYPE

    app = pg_shared_app
    checkpoint = str(CHECKPOINT_EVENT_TYPE)
    _execute(*_DROP_CONNECTION_ON_CHECKPOINT)
    task_id = start_task(app, "e2e:gate")
    run_id = _run_id(app, task_id)

    # The run's checkpoint before the model call is committed, so the filter
    # matches real rows: a mistyped one would make this test pass vacuously.
    ((written,),) = _execute(
        "SELECT count(*) FROM trace_events"
        " WHERE task_id = :task_id AND event_type = :event_type",
        task_id=task_id,
        event_type=checkpoint,
    )
    assert written >= 1
    _execute(
        "INSERT INTO e2e_write_fault (task_id, event_type)"
        " VALUES (:task_id, :event_type)",
        task_id=task_id,
        event_type=checkpoint,
    )
    _release_model(app)

    # The pause is brief (a 1s backoff); the recovery events prove it.
    settled = app.wait_task(task_id, run_id=run_id, timeout=SETTLE_TIMEOUT)
    assert ANSWER in (settled["output"] or "")
    assert settled["error"] is None
    # The answer whose checkpoint was lost, then the resumed run's.
    assert len(model_calls(app)) == 2
    row, events = recovery_state(task_id)
    assert row is not None
    assert (row["reason"], row["run_id"], row["total_resumes"]) == (
        "persistence_failure",
        run_id,
        1,
    )
    assert events == ONE_RESUME
    assert len(auto_resume_commands(task_id)) == 1
    _assert_answered_once(task_id)
    # The fault fired once, and a later checkpoint of the task went through.
    ((hits,),) = _execute("SELECT last_value FROM e2e_write_fault_hits")
    assert hits >= 2


# --------------------------------------------------------------------------
# E2b: the database stays unreachable past the lease TTL
# --------------------------------------------------------------------------


def _lease_state(task_id: int) -> dict:
    from xagent.web.models.database import get_session_local
    from xagent.web.models.task import Task

    with get_session_local()() as db:
        task = db.get(Task, task_id)
        return {
            "status": task.status.value,
            "runner_id": task.runner_id,
            "lease_expires_at": task.lease_expires_at,
        }


def _dispatcher_failure_lines(app) -> dict[str, int]:
    # Coupled to the log text in task_command_transport's dispatcher worker,
    # which exports no constant for it.
    return {
        path.name: path.read_text().count(
            "component=task-command-dispatcher command dispatch failed"
        )
        for path in app.root.glob("*.log")
    }


def test_database_unreachable_past_lease_ttl_resumes_after_recovery(
    pg_shared_app, pg_link
):
    from xagent.web.services.task_command_transport import (
        DISPATCHER_CONCURRENCY,
        DISPATCHER_FAILURE_LOG_INTERVAL_SECONDS,
    )

    app = pg_shared_app
    task_id = start_task(app, "e2e:gate")
    run_id = _run_id(app, task_id)

    pg_link.cut()
    cut_at = time.monotonic()
    # The answer's checkpoint and the settlement after it both fail.
    _release_model(app)

    def lease_lapsed():
        state = _lease_state(task_id)
        # Nobody could write: the run still holds its lease and is not FAILED.
        if state["status"] != "running" or state["runner_id"] is None:
            pytest.fail(
                f"Lease not held during the outage: {state}\n{app.diagnostics()}"
            )
        expires = state["lease_expires_at"]
        return expires < datetime.now(timezone.utc) - timedelta(seconds=1)

    wait_for(lease_lapsed, app, "the lease to lapse", timeout=30)
    pg_link.restore()
    outage_seconds = time.monotonic() - cut_at

    settled = app.wait_task(task_id, run_id=run_id, timeout=SETTLE_TIMEOUT)
    assert ANSWER in (settled["output"] or "")
    assert settled["error"] is None
    assert len(model_calls(app)) == 2
    row, events = recovery_state(task_id)
    assert row is not None
    # Pins today's behaviour: settlement is attempted once, while the
    # database is still down, so the lease TTL (not the failed write)
    # records the interruption.
    assert (row["reason"], row["run_id"], row["total_resumes"]) == (
        "lease_expired",
        run_id,
        1,
    )
    assert events == ONE_RESUME
    assert len(auto_resume_commands(task_id)) == 1
    _assert_answered_once(task_id)

    # Each dispatcher worker logs a failing claim at most once per interval,
    # and the worker host's dispatcher did see the outage. The bound holds
    # only while the outage fits in one interval.
    assert outage_seconds < DISPATCHER_FAILURE_LOG_INTERVAL_SECONDS
    lines = _dispatcher_failure_lines(app)
    worker_lines = [n for name, n in lines.items() if name.startswith("worker-")]
    assert worker_lines and all(
        1 <= n <= DISPATCHER_CONCURRENCY for n in worker_lines
    ), lines
    assert all(n <= DISPATCHER_CONCURRENCY for n in lines.values()), lines


# --------------------------------------------------------------------------
# E3: an outage shorter than the heartbeat's slack changes nothing
# --------------------------------------------------------------------------

# Under the lease TTL (10s) less one heartbeat (2s).
BRIEF_OUTAGE_SECONDS = 4


def test_brief_database_outage_during_a_model_call_is_absorbed(pg_shared_app, pg_link):
    app = pg_shared_app
    task_id = start_task(app, "e2e:gate")
    run_id = _run_id(app, task_id)

    pg_link.cut()
    # A heartbeat committing as the link was cut has landed by now.
    time.sleep(1)
    before = _lease_state(task_id)["lease_expires_at"]
    time.sleep(BRIEF_OUTAGE_SECONDS - 1)
    after = _lease_state(task_id)["lease_expires_at"]
    # The hosts could not renew the lease, which is still valid.
    assert after == before
    assert after > datetime.now(timezone.utc)
    pg_link.restore()
    _release_model(app)

    settled = app.wait_task(task_id, run_id=run_id, timeout=SETTLE_TIMEOUT)
    assert ANSWER in (settled["output"] or "")
    assert settled["error"] is None
    assert len(model_calls(app)) == 1
    assert recovery_state(task_id) == (None, [])
    assert auto_resume_commands(task_id) == []
    _assert_answered_once(task_id)
