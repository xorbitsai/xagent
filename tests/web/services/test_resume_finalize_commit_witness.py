"""A resumed run's finalize COMMIT confirmed by the row it inserted.

``_finalize_resumed_task`` releases the lease in the same transaction as the
result, so after a lost acknowledgement whose COMMIT applied, only the result
fact (V2) or the legacy transcript row can prove the run settled. Faults are
raised at ``Session.commit`` (after or instead of the real COMMIT) so each
test knows whether the finalize applied; the read-back must tell the two
apart. Also pins R4: with automatic resume off, a resumed run is classified
for an unknown tool effect exactly as a new run is. Runs on SQLite and, when
``XAGENT_TEST_POSTGRES_URL`` is set, on PostgreSQL.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import OperationalError

from tests.web.services.test_execution_event_signals import (
    metric_reader as metric_reader_fixture,
)
from tests.web.services.test_execution_event_unknown_outcomes import (
    ATTEMPT,
    PERSISTENCE_RESULT,
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
    NO_OUTPUTS,
    PAUSE_SWITCH,
    PAUSED_FACT,
    _assert_paused,
    _assistant_lines,
    _checkpoint,
)
from tests.web.services.test_persistence_failure_pause import (
    _late_bound_sessions as _late_bound_sessions_fixture,
)
from tests.web.services.test_persistence_failure_pause import (
    _persistence_interruption,
    _resume_raising,
    _settled_facts,
    _start_run,
    _state,
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
from xagent.core.agent.checkpoint import (
    CheckpointUnavailableError,
    ExecutionEventPersistenceError,
)
from xagent.core.agent.interruption import InterruptionReason
from xagent.web.models.task import Task, TaskStatus
from xagent.web.services import task_execution
from xagent.web.services.client_error_messages import CLIENT_SAFE_TASK_FAILURE
from xagent.web.services.task_execution import (
    _finalize_resumed_task,
    _settle_resumed_task_lease,
)
from xagent.web.services.task_lease_service import (
    TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR,
)
from xagent.web.services.task_orchestrator import settle_task_lease_isolated

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture
metric_reader = metric_reader_fixture
_capture = _capture_fixture
_late_bound_sessions = _late_bound_sessions_fixture


@pytest.fixture(autouse=True)
def _switches(monkeypatch):
    # The pause branch decides as these suites do: pause, leave it manual.
    monkeypatch.setenv(PAUSE_SWITCH, "true")
    monkeypatch.setenv(AUTO_RESUME_SWITCH, "false")


@pytest.fixture
def finalizes(monkeypatch):
    """Match the COMMIT of a resumed finalize that released its lease."""
    releasing: set[int] = set()
    release = task_execution.release_task_lease_no_commit

    def released(db, *args, **kwargs):
        releasing.add(id(db))
        return release(db, *args, **kwargs)

    monkeypatch.setattr(task_execution, "release_task_lease_no_commit", released)

    def matches(db) -> bool:
        if id(db) not in releasing:
            return False
        releasing.discard(id(db))
        return True

    return matches


@pytest.fixture
def file_settlements(monkeypatch):
    """Record whether each prepared file output was kept or compensated."""
    kept: list[bool] = []
    original = task_execution._settle_prepared_task_file_outputs

    def settle(prepared, *, metadata_committed, **kwargs):
        kept.append(metadata_committed)
        return original(prepared, metadata_committed=metadata_committed, **kwargs)

    monkeypatch.setattr(task_execution, "_settle_prepared_task_file_outputs", settle)
    return kept


@pytest.fixture
def read_backs(monkeypatch):
    """Record each witness read-back and its verdict."""
    verdicts: list[bool] = []
    original = task_execution._finalization_witness_committed

    def read_back(witness):
        verdict = original(witness)
        verdicts.append(verdict)
        return verdict

    monkeypatch.setattr(task_execution, "_finalization_witness_committed", read_back)
    return verdicts


def _user_id(factory, tid: int) -> int:
    with factory() as db:
        return int(db.get(Task, tid).user_id)


# The resumed outcomes a finalize commits: (status, success, result).
COMPLETED = ("completed", True, {"success": True, "status": "completed"})
WAITING = ("waiting_for_user", False, {"success": False, "status": "waiting_for_user"})
PAUSED = ("failed", False, PERSISTENCE_RESULT)
OUTCOMES = {"completed": COMPLETED, "waiting_for_user": WAITING, "paused": PAUSED}
# A plain failure: no interruption reason, so it settles FAILED.
FAILED = ("failed", False, {"success": False, "status": "failed", "error": "boom"})


def _resume_finalize(factory, tid: int, lease, outcome) -> dict[str, Any]:
    status, success, result = outcome
    return _finalize_resumed_task(
        tid,
        status=status,
        success=success,
        output="done" if success else str(result.get("output", "")),
        task_owner_user_id=_user_id(factory, tid),
        result=dict(result, output="done") if success else dict(result),
        task_lease=lease,
        prepared_outputs=NO_OUTPUTS,
    )


def _assert_still_running(factory, tid: int, lease) -> None:
    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == TaskStatus.RUNNING
        assert task.runner_id == lease.runner_id
        assert task.run_id == lease.run_id
    assert _settled_facts(factory, tid) == []
    assert _assistant_lines(factory, tid) == []


# ------------------------------------------------- a lost ack that applied


@pytest.mark.parametrize("outcome", list(OUTCOMES))
def test_a_lost_resumed_finalize_ack_whose_commit_applied_is_confirmed(
    canonical,
    finalizes,
    file_settlements,
    read_backs,
    monkeypatch,
    caplog,
    metric_reader,
    outcome,
):
    factory, tid = canonical
    lease = _start_run(factory, tid)
    _checkpoint(factory, tid, lease.run_id)
    raised = fail_commits(monkeypatch, finalizes, committed=True)

    finalized = _resume_finalize(factory, tid, lease, OUTCOMES[outcome])

    assert len(raised) == 1
    assert read_backs == [True]
    # Returned exactly as a successful COMMIT returns.
    assert finalized["late_result"] is False
    assert finalized["lease_released"] is True
    expected_status = {
        "completed": "completed",
        "waiting_for_user": "waiting_for_user",
        "paused": "paused",
    }[outcome]
    assert finalized["final_status"] == expected_status
    report = finalized["report"]
    with factory() as db:
        task = db.get(Task, tid)
        assert task.status.value == expected_status
        assert task.runner_id is None
        assert report.control_state["state_version"] == task.state_version
        assert report.control_state["run_id"] == lease.run_id
    # The staged file outputs are kept, not compensated.
    assert file_settlements == [True]
    facts = _settled_facts(factory, tid)
    assert len(facts) == 1
    if outcome == "paused":
        assert report.paused_for is InterruptionReason.PERSISTENCE_FAILURE
        assert report.recovery is not None
        assert facts == [PAUSED_FACT]
        _assert_paused(factory, tid, lease, reason="persistence_failure")
        assert _assistant_lines(factory, tid) == []
    else:
        assert report.paused_for is None
        assert facts[0]["status"] == expected_status
        assert len(_assistant_lines(factory, tid)) == (
            1 if outcome == "completed" else 0
        )
    assert_one(
        caplog,
        metric_reader,
        kind="finalize_ack_reconciled",
        task=tid,
        run_id=lease.run_id,
        ref=None,
        storage="v2",
        level=logging.INFO,
    )


@pytest.mark.parametrize(
    ("outcome", "line"),
    [(COMPLETED, "done"), (FAILED, CLIENT_SAFE_TASK_FAILURE)],
    ids=["completed", "failed"],
)
def test_a_legacy_resumed_finalize_is_confirmed_by_its_transcript_row(
    canonical,
    finalizes,
    file_settlements,
    read_backs,
    monkeypatch,
    caplog,
    metric_reader,
    outcome,
    line,
):
    """An answer or a failure inserts a legacy transcript row; the failure's
    row has fixed content, so its id is what tells it apart."""
    factory, tid = canonical
    lease = _start_run(factory, tid, conversation_storage_version=1)
    fail_commits(monkeypatch, finalizes, committed=True)

    finalized = _resume_finalize(factory, tid, lease, outcome)

    assert read_backs == [True]
    assert finalized["lease_released"] is True
    assert finalized["final_status"] == outcome[0]
    assert file_settlements == [True]
    assert _assistant_lines(factory, tid) == [line]
    assert_one(
        caplog,
        metric_reader,
        kind="finalize_ack_reconciled",
        task=tid,
        run_id=lease.run_id,
        ref=None,
        storage="v1",
        level=logging.INFO,
    )


def test_a_legacy_resumed_wait_without_a_witness_stays_unresolved(
    canonical,
    finalizes,
    file_settlements,
    read_backs,
    monkeypatch,
    caplog,
    metric_reader,
):
    """A legacy wait inserts no row: nothing proves its COMMIT, so a lost
    acknowledgement is re-raised and reported unresolved even though it
    applied."""
    factory, tid = canonical
    lease = _start_run(factory, tid, conversation_storage_version=1)
    raised = fail_commits(monkeypatch, finalizes, committed=True)

    with pytest.raises(OperationalError) as caught:
        _resume_finalize(factory, tid, lease, WAITING)

    assert caught.value is raised[0]
    assert read_backs == []
    assert file_settlements == [False]
    assert_one(
        caplog,
        metric_reader,
        kind="finalize_ack_unresolved",
        task=tid,
        run_id=lease.run_id,
        ref=None,
        storage="v1",
    )


@pytest.mark.asyncio
async def test_a_lost_ack_unknown_tool_effect_is_reported_once(
    canonical, finalizes, file_settlements, read_backs, monkeypatch, caplog
):
    """The confirmed COMMIT's post-commit step runs once: its unknown tool
    effect is reported once, after the reconciliation."""
    factory, tid = canonical
    lease = await _run_with_unknown_tool(factory, tid)
    fail_commits(monkeypatch, finalizes, committed=True)

    finalized = _settle_unknown_tool_effect("resume_finalize", factory, tid, lease)

    assert finalized["lease_released"] is True
    assert finalized["final_status"] == "failed"
    assert read_backs == [True]
    assert file_settlements == [True]
    with factory() as db:
        assert db.get(Task, tid).error_message == (
            TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR
        )
    assert len(_settled_facts(factory, tid)) == 1
    outcomes = [fields(r.getMessage()) for r in records(caplog)]
    assert [o["kind"] for o in outcomes] == ["finalize_ack_reconciled", "tool_effect"]
    assert outcomes[1]["ref"] == ATTEMPT


# ------------------------------------------------- a commit that did not apply


@pytest.mark.parametrize("outcome", list(OUTCOMES))
@pytest.mark.parametrize(
    "error", [invalidated, _deadlock], ids=["lost-connection", "refused"]
)
def test_a_resumed_finalize_commit_that_did_not_apply_is_raised(
    canonical,
    finalizes,
    file_settlements,
    monkeypatch,
    caplog,
    metric_reader,
    outcome,
    error,
):
    factory, tid = canonical
    lease = _start_run(factory, tid)
    _checkpoint(factory, tid, lease.run_id)
    raised = fail_commits(monkeypatch, finalizes, committed=False, error=error)

    with pytest.raises(OperationalError) as caught:
        _resume_finalize(factory, tid, lease, OUTCOMES[outcome])

    assert caught.value is raised[0]
    # Rolled back: the files are compensated and the run is still owned.
    assert file_settlements == [False]
    _assert_still_running(factory, tid, lease)
    # The read-back decided "absent" (or the server refused): a plain failure.
    assert_none(caplog, metric_reader)


@pytest.mark.parametrize("outcome", [COMPLETED, FAILED], ids=["completed", "failed"])
def test_a_legacy_resumed_finalize_that_did_not_apply_is_raised(
    canonical,
    finalizes,
    file_settlements,
    read_backs,
    monkeypatch,
    caplog,
    metric_reader,
    outcome,
):
    """The legacy transcript row is read back and found absent: a plain
    failure, even after a dropped connection."""
    factory, tid = canonical
    lease = _start_run(factory, tid, conversation_storage_version=1)
    raised = fail_commits(monkeypatch, finalizes, committed=False, error=invalidated)

    with pytest.raises(OperationalError) as caught:
        _resume_finalize(factory, tid, lease, outcome)

    assert caught.value is raised[0]
    assert read_backs == [False]
    assert file_settlements == [False]
    _assert_still_running(factory, tid, lease)
    assert_none(caplog, metric_reader)


@pytest.mark.parametrize(
    "error", [invalidated, _deadlock], ids=["lost-connection", "refused"]
)
def test_a_failed_read_back_re_raises_the_commit_error(
    canonical, finalizes, file_settlements, monkeypatch, caplog, metric_reader, error
):
    factory, tid = canonical
    lease = _start_run(factory, tid)
    raised = fail_commits(monkeypatch, finalizes, committed=False, error=error)

    def read_back_fails(_witness):
        raise CheckpointUnavailableError("read-back unavailable")

    monkeypatch.setattr(
        task_execution, "_finalization_witness_committed", read_back_fails
    )

    with pytest.raises(OperationalError) as caught:
        _resume_finalize(factory, tid, lease, COMPLETED)

    assert caught.value is raised[0]
    assert file_settlements == [False]
    _assert_still_running(factory, tid, lease)
    if error is invalidated:
        # Only a dropped connection can hide an applied COMMIT.
        assert_one(
            caplog,
            metric_reader,
            kind="finalize_ack_unresolved",
            task=tid,
            run_id=lease.run_id,
            ref=None,
            storage="v2",
        )
    else:
        assert_none(caplog, metric_reader)


def test_without_the_read_back_an_applied_resumed_finalize_is_not_confirmed(
    canonical, finalizes, file_settlements, monkeypatch
):
    """Reverse check of the lost-ack tests: with the read-back answering
    "absent", an applied COMMIT is reported as a failure the caller cannot
    settle (the row is no longer RUNNING), and its files are compensated."""
    factory, tid = canonical
    lease = _start_run(factory, tid)
    monkeypatch.setattr(
        task_execution, "_finalization_witness_committed", lambda _w: False
    )
    fail_commits(monkeypatch, finalizes, committed=True)

    with pytest.raises(OperationalError):
        _resume_finalize(factory, tid, lease, COMPLETED)

    assert file_settlements == [False]
    with factory() as db:
        assert db.get(Task, tid).status == TaskStatus.COMPLETED


# ------------------------------------------- R4: automatic resume turned off


@pytest.fixture
def _automatic_resume_off(monkeypatch):
    # Unset, the pause switch follows automatic resume: settlement decides
    # every interruption as LEGACY.
    monkeypatch.delenv(PAUSE_SWITCH, raising=False)
    monkeypatch.setenv(AUTO_RESUME_SWITCH, "false")


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["new_run", "resumed_run"])
async def test_a_raised_run_is_classified_for_an_unknown_tool_effect(
    canonical, _automatic_resume_off, caplog, path
):
    """A run that raised after starting a tool whose outcome never committed
    fails as an unknown tool effect, resumed or not, with automatic resume
    off as well."""
    factory, tid = canonical
    lease = await _run_with_unknown_tool(factory, tid)
    interruption = _persistence_interruption()

    if path == "new_run":
        settled = settle_task_lease_isolated(
            lease,
            error_message="setup/run error: event write failed",
            classify_unknown_tool_effect=True,
            interruption=interruption,
        )
    else:
        settled = _settle_resumed_task_lease(
            lease,
            error_message="event write failed",
            interruption=interruption,
            classify_unknown_tool_effect=True,
        )

    assert settled
    task, row, events = _state(factory, tid)
    assert task.status == TaskStatus.FAILED
    assert task.error_message == TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR
    assert (row.reason, row.state) == ("unknown_tool_effect", "disabled")
    assert [(e.event, e.reason) for e in events] == [
        ("interrupted", "unknown_tool_effect")
    ]
    outcomes = [fields(r.getMessage()) for r in records(caplog)]
    assert [(o["kind"], o["ref"]) for o in outcomes] == [("tool_effect", ATTEMPT)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "explicit_cancel", [False, True], ids=["raised", "explicitly-cancelled"]
)
async def test_a_raised_resume_asks_settlement_to_classify(explicit_cancel):
    """The resume exception branch asks for the classification a new run's
    does, cancelled or not: a cancel does not undo a tool that ran."""
    settle_mock = MagicMock(return_value=True)

    await _resume_raising(
        ExecutionEventPersistenceError("event write failed"),
        settle_mock,
        explicit_cancel=explicit_cancel,
    )

    settle_mock.assert_called_once()
    assert settle_mock.call_args.kwargs["classify_unknown_tool_effect"] is True


@pytest.fixture
def _automatic_resume_on(monkeypatch):
    monkeypatch.setenv(PAUSE_SWITCH, "true")
    monkeypatch.setenv(AUTO_RESUME_SWITCH, "true")


@pytest.mark.asyncio
@pytest.mark.parametrize("classify", [False, True], ids=["unclassified", "classified"])
async def test_a_resume_settled_without_an_interruption_is_classified(
    canonical, _automatic_resume_on, caplog, classify
):
    """An explicit cancel, or an error with no interruption reason, settles
    with no interruption to resume from, so even with automatic resume on
    the flag decides whether a started tool with no committed outcome fails
    the run as an unknown tool effect."""
    factory, tid = canonical
    lease = await _run_with_unknown_tool(factory, tid)

    settled = _settle_resumed_task_lease(
        lease,
        error_message="Task was cancelled",
        interruption=None,
        classify_unknown_tool_effect=classify,
    )

    assert settled
    task, row, events = _state(factory, tid)
    assert task.status == TaskStatus.FAILED
    assert task.runner_id is None
    outcomes = [fields(r.getMessage()) for r in records(caplog)]
    if classify:
        assert task.error_message == TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR
        assert [(o["kind"], o["ref"]) for o in outcomes] == [("tool_effect", ATTEMPT)]
    else:
        assert task.error_message == "Task was cancelled"
        assert outcomes == []
    # No interruption decided: nothing to pause or resume from.
    assert row is None
    assert events == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["finalize", "resume_finalize"])
async def test_a_resumed_result_settles_like_a_new_run_result(
    canonical, _automatic_resume_off, path
):
    """With automatic resume off an unsuccessful result settles the same way
    for a new and a resumed run (LEGACY: failed as reported)."""
    factory, tid = canonical
    lease = await _run_with_unknown_tool(factory, tid)

    _settle_unknown_tool_effect(path, factory, tid, lease)

    task, row, events = _state(factory, tid)
    assert task.status == TaskStatus.FAILED
    assert task.error_message == PERSISTENCE_RESULT["error"]
    assert (row.reason, row.state) == ("persistence_failure", "disabled")
    assert [(e.event, e.reason) for e in events] == [
        ("interrupted", "persistence_failure")
    ]
