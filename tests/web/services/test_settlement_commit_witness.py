"""Settlement COMMITs confirmed by their result fact's witness.

A settlement makes a fenced state transition, so a plain rerun after a lost
acknowledgement whose COMMIT applied would miss its own fence and report
nothing. Faults are raised at ``Session.commit`` (after or instead of the
real COMMIT) so each test knows whether the settlement applied; the read-back
must tell the two apart. Runs on SQLite and, when ``XAGENT_TEST_POSTGRES_URL``
is set, on PostgreSQL.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.exc import IntegrityError, OperationalError

from tests.web.services.test_execution_event_signals import (
    metric_reader as metric_reader_fixture,
)
from tests.web.services.test_execution_event_unknown_outcomes import (
    ATTEMPT,
)
from tests.web.services.test_execution_event_unknown_outcomes import (
    _capture as _capture_fixture,
)
from tests.web.services.test_execution_event_unknown_outcomes import (
    _deadlock,
    _run_with_unknown_tool,
    _settle_unknown_tool_effect,
    assert_none,
    assert_one,
    fields,
    records,
)
from tests.web.services.test_persistence_failure_pause import (
    AUTO_RESUME_SWITCH,
    PAUSE_SWITCH,
    RUN_ERROR,
    _assert_paused,
    _assistant_lines,
    _checkpoint,
    _persistence_interruption,
    _prepare_run,
    _projected_task,
    _settle,
    _settled_facts,
    _start_run,
)
from tests.web.services.test_provider_failure_pause import (
    LLM_RESULT,
    _completed_channel_command,
)
from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_execution_event_writer import engine as engine_fixture
from tests.web.services.test_task_execution_event_writer import (
    task_id as task_id_fixture,
)
from tests.web.services.test_trace_fact_commit_retry import (
    fail_commits,
    invalidated,
)
from tests.web.services.test_turn_runtime_settled_hook import (
    drive_channel_background,
)
from xagent.core.agent.interruption import InterruptionReason
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.services import db_write_retry
from xagent.web.services import managed_task_lease as managed
from xagent.web.services import task_execution_event_writer
from xagent.web.services import task_orchestrator as orchestrator
from xagent.web.services.channel_runtime import (
    _ChannelTaskClaimSnapshot,
    _compensate_channel_task_claim_sync,
)
from xagent.web.services.execution_result_projection import (
    interrupted_channel_result,
    project_execution_result_for_channel,
)
from xagent.web.services.task_execution_event_store import (
    lock_task_execution_events_no_commit,
)
from xagent.web.services.task_lease_service import (
    TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR,
)
from xagent.web.services.task_settlement_report import SettlementReport

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture
metric_reader = metric_reader_fixture
_capture = _capture_fixture


@pytest.fixture(autouse=True)
def _settlement_retries(monkeypatch):
    monkeypatch.setenv("XAGENT_DB_WRITE_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("XAGENT_DB_WRITE_RETRY_BASE_DELAY_MS", "0")
    monkeypatch.setenv("XAGENT_DB_WRITE_RETRY_MAX_DELAY_MS", "0")
    # The pause branch decides as these suites do: pause, leave it manual.
    monkeypatch.setenv(PAUSE_SWITCH, "true")
    monkeypatch.setenv(AUTO_RESUME_SWITCH, "false")


@pytest.fixture
def settle_attempts(monkeypatch):
    """Count the transactions ``settle_task_lease_isolated`` runs."""
    calls: list[int] = []
    original = orchestrator._settle_task_lease_attempt

    def attempt(*args, **kwargs):
        calls.append(kwargs["attempt"].number)
        return original(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "_settle_task_lease_attempt", attempt)
    return calls


@pytest.fixture
def read_backs(monkeypatch):
    """Record each witness read-back and its verdict."""
    verdicts: list[bool] = []
    original = task_execution_event_writer.fact_witness_committed

    def read_back(witness):
        verdict = original(witness)
        verdicts.append(verdict)
        return verdict

    monkeypatch.setattr(
        task_execution_event_writer, "fact_witness_committed", read_back
    )
    return verdicts


@pytest.fixture
def settles(monkeypatch):
    """Match the COMMIT of a transaction that staged a result fact."""
    staging: set[int] = set()
    stage = task_execution_event_writer.stage_result_fact_no_commit

    def staged(db, *args, **kwargs):
        staging.add(id(db))
        return stage(db, *args, **kwargs)

    monkeypatch.setattr(
        task_execution_event_writer, "stage_result_fact_no_commit", staged
    )

    def matches(db) -> bool:
        if id(db) not in staging:
            return False
        staging.discard(id(db))
        return True

    return matches


def _retries(metric_reader, site: str) -> dict[str, int]:
    """``outcome -> count`` of the retry counter for ``site``."""
    counts: dict[str, int] = {}
    data = metric_reader.get_metrics_data()
    for resource in data.resource_metrics if data else ():
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                if metric.name != "xagent.execution_event.write.retries":
                    continue
                for point in metric.data.data_points:
                    if point.attributes.get("site") == site:
                        outcome = str(point.attributes["outcome"])
                        counts[outcome] = counts.get(outcome, 0) + int(point.value)
    return counts


def _failed_once(factory, tid: int, lease) -> None:
    """One FAILED settlement with one failure transcript line and one fact."""
    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == TaskStatus.FAILED
        assert task.error_message == RUN_ERROR
        assert task.runner_id is None
    (fact,) = _settled_facts(factory, tid)
    assert fact["status"] == "failed"
    assert len(_assistant_lines(factory, tid)) == 1


# ----------------------------------------------------------- the error branch


def test_a_lost_settlement_ack_whose_commit_applied_is_confirmed(
    canonical, settles, monkeypatch, settle_attempts, read_backs, caplog, metric_reader
):
    factory, tid = canonical
    lease = _start_run(factory, tid)
    raised = fail_commits(monkeypatch, settles, committed=True)

    settled, report = _settle(lease)

    assert len(raised) == 1
    # Confirmed by the read-back, not by a rerun that would miss its fence.
    assert settled
    assert settle_attempts == [1]
    assert read_backs == [True]
    _failed_once(factory, tid, lease)
    with factory() as db:
        task = db.get(Task, tid)
        assert report.control_state["state_version"] == task.state_version
        assert report.control_state["run_id"] == lease.run_id
    assert_one(
        caplog,
        metric_reader,
        kind="settlement_ack_reconciled",
        task=tid,
        run_id=lease.run_id,
        ref=None,
        storage="v2",
        level=logging.INFO,
    )


def test_without_the_read_back_a_lost_settlement_ack_is_not_confirmed(
    canonical, settles, monkeypatch, settle_attempts
):
    """Reverse check of the test above: with the read-back answering
    "absent", the rerun misses its own fence and reports nothing."""
    factory, tid = canonical
    lease = _start_run(factory, tid)
    monkeypatch.setattr(
        task_execution_event_writer, "fact_witness_committed", lambda _w: False
    )
    fail_commits(monkeypatch, settles, committed=True)

    settled, report = _settle(lease)

    assert not settled
    assert report.control_state == {}
    assert settle_attempts == [1, 2]
    _failed_once(factory, tid, lease)


def test_a_deadlocked_settlement_commit_is_rerun_once(
    canonical, settles, monkeypatch, settle_attempts, read_backs, caplog, metric_reader
):
    factory, tid = canonical
    lease = _start_run(factory, tid)
    raised = fail_commits(monkeypatch, settles, committed=False, error=_deadlock)

    settled, report = _settle(lease)

    assert len(raised) == 1
    assert settled
    assert settle_attempts == [1, 2]
    assert read_backs == [False]
    _failed_once(factory, tid, lease)
    assert report.control_state["run_id"] == lease.run_id
    # A rerun is a retry, not an unknown outcome.
    assert_none(caplog, metric_reader)
    assert _retries(metric_reader, "settlement") == {"retried": 1, "recovered": 1}


def test_exhausted_settlement_reruns_raise_the_last_error(
    canonical, settles, monkeypatch, settle_attempts
):
    factory, tid = canonical
    lease = _start_run(factory, tid)
    raised = fail_commits(
        monkeypatch, settles, committed=False, times=3, error=_deadlock
    )

    with pytest.raises(OperationalError) as caught:
        _settle(lease)

    assert caught.value is raised[-1]
    assert settle_attempts == [1, 2, 3]
    with factory() as db:
        task = db.get(Task, tid)
        # Retained for TTL recovery, as before.
        assert task.status == TaskStatus.RUNNING
        assert (task.runner_id, task.run_id) == (lease.runner_id, lease.run_id)
    assert _settled_facts(factory, tid) == []


@pytest.mark.parametrize("committed", [True, False], ids=["applied", "rolled-back"])
def test_an_undecidable_settlement_read_back_raises_and_reports_unknown(
    canonical, settles, monkeypatch, settle_attempts, caplog, metric_reader, committed
):
    factory, tid = canonical
    lease = _start_run(factory, tid)

    def read_back_fails(_witness):
        raise OperationalError("SELECT", {}, Exception("connection refused"))

    monkeypatch.setattr(
        task_execution_event_writer, "fact_witness_committed", read_back_fails
    )
    raised = fail_commits(monkeypatch, settles, committed=committed)

    with pytest.raises(OperationalError) as caught:
        _settle(lease)

    assert caught.value is raised[0]
    assert settle_attempts == [1]
    assert len(_settled_facts(factory, tid)) == (1 if committed else 0)
    assert_one(
        caplog,
        metric_reader,
        kind="settlement_ack_unresolved",
        task=tid,
        run_id=lease.run_id,
        ref=None,
        storage="v2",
    )


def test_a_legacy_settlement_is_not_retried(canonical, monkeypatch, settle_attempts):
    """No fact witness: the commit error propagates as before."""
    factory, tid = canonical
    lease = _start_run(factory, tid, conversation_storage_version=1)
    raised = fail_commits(monkeypatch, lambda _db: True, committed=False)

    with pytest.raises(OperationalError) as caught:
        _settle(lease)

    assert caught.value is raised[0]
    assert settle_attempts == [1]


def test_a_non_retryable_settlement_commit_error_is_not_retried(
    canonical, settles, monkeypatch, settle_attempts, read_backs
):
    """With a witness, a non-retryable commit error still propagates as
    before: it is neither read back nor rerun."""
    factory, tid = canonical
    lease = _start_run(factory, tid)
    raised = fail_commits(
        monkeypatch,
        settles,
        committed=False,
        error=lambda: IntegrityError("COMMIT", {}, Exception("constraint")),
    )

    with pytest.raises(IntegrityError) as caught:
        _settle(lease)

    assert caught.value is raised[0]
    assert settle_attempts == [1]
    assert read_backs == []


@pytest.mark.parametrize("path", ["settle_decided", "settle_classified"])
async def test_a_lost_ack_settlement_reports_the_tool_effect_once(
    canonical, settles, monkeypatch, settle_attempts, read_backs, caplog, path
):
    """The confirmed COMMIT's post-commit step runs once: its unknown tool
    effect is reported once, after the reconciliation."""
    factory, tid = canonical
    lease = await _run_with_unknown_tool(factory, tid)
    fail_commits(monkeypatch, settles, committed=True)

    settled, _report = _settle_unknown_tool_effect(path, factory, tid, lease)

    assert settled
    assert settle_attempts == [1]
    assert read_backs == [True]
    with factory() as db:
        assert db.get(Task, tid).error_message == (
            TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR
        )
    outcomes = [fields(r.getMessage()) for r in records(caplog)]
    assert [o["kind"] for o in outcomes] == ["settlement_ack_reconciled", "tool_effect"]
    assert outcomes[1]["ref"] == ATTEMPT


# ----------------------------------------------------------- the pause branch


@pytest.mark.parametrize("committed", [True, False], ids=["lost-ack", "rolled-back"])
def test_a_failed_pause_commit_is_reconciled(
    canonical, settles, monkeypatch, settle_attempts, read_backs, committed
):
    factory, tid = canonical
    lease = _start_run(factory, tid)
    _checkpoint(factory, tid, lease.run_id)
    fail_commits(monkeypatch, settles, committed=committed)

    settled, report = _settle(lease, interruption=_persistence_interruption())

    assert settled
    assert report.paused_for is not None
    assert report.paused_for.value == "persistence_failure"
    assert report.control_state["status"] == "paused"
    assert report.recovery is not None
    assert settle_attempts == ([1] if committed else [1, 2])
    assert read_backs == [committed]
    row = _assert_paused(factory, tid, lease, reason="persistence_failure")
    assert report.control_state["state_version"] == row.paused_state_version
    (fact,) = _settled_facts(factory, tid)
    assert fact["status"] == "paused"
    assert _assistant_lines(factory, tid) == []


# ------------------------------------------------------- the managed _sync path


@pytest.mark.parametrize("committed", [True, False], ids=["lost-ack", "rolled-back"])
@pytest.mark.parametrize("path", ["finalize", "release"])
def test_a_failed_managed_settlement_commit_is_reconciled(
    canonical, settles, monkeypatch, read_backs, committed, path
):
    factory, tid = canonical
    lease = _start_run(factory, tid)
    fail_commits(monkeypatch, settles, committed=committed)

    if path == "finalize":
        settled = managed.finalize_managed_task_lease_result_sync(
            lease,
            status=TaskStatus.COMPLETED,
            assistant_content="done",
            execution_result={"success": True, "output": "done"},
        )
    else:
        settled = managed._release_managed_task_lease_sync(lease)

    assert settled
    assert read_backs == [committed]
    expected = TaskStatus.COMPLETED if path == "finalize" else TaskStatus.FAILED
    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == expected
        assert task.runner_id is None
    (fact,) = _settled_facts(factory, tid)
    assert fact["status"] == expected.value
    assert len(_assistant_lines(factory, tid)) == 1


def test_a_managed_lost_ack_without_the_read_back_is_not_confirmed(
    canonical, settles, monkeypatch
):
    """Reverse check: rerun after an applied COMMIT misses the lease fence."""
    factory, tid = canonical
    lease = _start_run(factory, tid)
    monkeypatch.setattr(
        task_execution_event_writer, "fact_witness_committed", lambda _w: False
    )
    fail_commits(monkeypatch, settles, committed=True)

    assert not managed.finalize_managed_task_lease_result_sync(
        lease, status=TaskStatus.COMPLETED, assistant_content="done"
    )
    assert len(_settled_facts(factory, tid)) == 1


# ------------------------------------------------------- the shared channel


@pytest.mark.parametrize("committed", [True, False], ids=["lost-ack", "rolled-back"])
def test_a_failed_shared_channel_pause_commit_is_reconciled(
    canonical, settles, monkeypatch, read_backs, committed
):
    """The shared channel leaf's settlement (``execute_channel_background``):
    a paused result is reported and its channel result stored once."""
    factory, _tid = canonical
    ids = _projected_task(factory)
    lease = _prepare_run(factory, ids)
    command_id, durable_result = _completed_channel_command(factory, ids, lease)
    projection = project_execution_result_for_channel(dict(LLM_RESULT))
    report = SettlementReport()
    fail_commits(monkeypatch, settles, committed=committed)

    assert managed.finalize_managed_task_lease_result_sync(
        lease,
        status=projection.task_status,
        assistant_content=projection.transcript_content,
        turn_id=ids["turn_id"],
        interactions=projection.interactions,
        message_type=projection.message_type,
        error_message=projection.diagnostic_error,
        execution_result=dict(LLM_RESULT),
        completion=(command_id, durable_result),
        settle_interruption=True,
        report=report,
    )

    assert read_backs == [committed]
    # What ``execute_channel_background`` hands ``publish_settlement_pause``.
    assert report.paused
    assert report.paused_for is InterruptionReason.LLM_UNAVAILABLE
    assert report.control_state["run_id"] == lease.run_id
    _assert_paused(factory, ids["task"], lease, reason="llm_unavailable")
    (fact,) = _settled_facts(factory, ids["task"])
    assert fact["status"] == "paused"
    with factory() as db:
        result = db.get(TaskExecutionCommand, command_id).result
    assert result["channel_result"] == interrupted_channel_result()


async def test_the_shared_channel_leaf_settles_through_the_witness_driver(
    monkeypatch,
):
    """``execute_channel_background`` settles in owned Sessions through
    ``finalize_managed_task_lease_result_sync``, with everything the call
    above reconciles."""
    calls: list[dict[str, Any]] = []

    def witnessed(lease, **kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(managed, "finalize_managed_task_lease_result_sync", witnessed)
    run_db_io = AsyncMock(return_value=True)
    await drive_channel_background(Mock(), "normal", monkeypatch, run_db_io=run_db_io)
    # The leaf's last off-loop call is its settlement.
    finalize = run_db_io.await_args_list[-1].args[0]
    assert finalize()
    (kwargs,) = calls
    assert kwargs["completion"] == (1, {})
    assert kwargs["settle_interruption"] is True
    assert isinstance(kwargs["report"], SettlementReport)


@pytest.mark.parametrize("committed", [True, False], ids=["lost-ack", "rolled-back"])
def test_a_failed_channel_claim_compensation_commit_is_reconciled(
    canonical, settles, monkeypatch, read_backs, committed
):
    factory, tid = canonical
    lease = _start_run(factory, tid)
    fail_commits(monkeypatch, settles, committed=committed)

    assert _compensate_channel_task_claim_sync(
        _ChannelTaskClaimSnapshot(user_id=1, task_id=tid, is_new_task=True, lease=lease)
    )

    assert read_backs == [committed]
    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == TaskStatus.FAILED
        assert task.runner_id is None
    assert len(_settled_facts(factory, tid)) == 1


# --------------------------------------------------------------- PostgreSQL


@pytest.fixture
def abandoned_task_lock(canonical, engine, monkeypatch):
    """Hold the task row lock after the failed COMMIT, as a dropped
    connection's abandoned transaction can."""
    if engine.dialect.name != "postgresql":
        pytest.skip("lock_timeout is a PostgreSQL setting")
    factory, tid = canonical
    monkeypatch.setattr(
        task_execution_event_writer, "_WITNESS_READ_BACK_LOCK_TIMEOUT", "200ms"
    )
    monkeypatch.setattr(db_write_retry, "_RERUN_LOCK_TIMEOUT", "200ms")
    holder = factory()
    yield lambda: lock_task_execution_events_no_commit(holder, tid)
    holder.rollback()
    holder.close()


def test_a_read_back_blocked_by_an_abandoned_lock_is_undecided(
    canonical,
    settles,
    abandoned_task_lock,
    monkeypatch,
    settle_attempts,
    caplog,
    metric_reader,
):
    """The read-back waits a bounded time for the task row lock, then the
    settlement fails as unresolved instead of guessing."""
    factory, tid = canonical
    lease = _start_run(factory, tid)
    raised = fail_commits(
        monkeypatch,
        settles,
        committed=False,
        error=invalidated,
        between=abandoned_task_lock,
    )

    with pytest.raises(OperationalError) as caught:
        _settle(lease)

    assert caught.value is raised[0]
    assert settle_attempts == [1]
    assert_one(
        caplog,
        metric_reader,
        kind="settlement_ack_unresolved",
        task=tid,
        run_id=lease.run_id,
        ref=None,
        storage="v2",
    )
