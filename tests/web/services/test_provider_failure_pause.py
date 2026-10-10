"""A run whose model provider stayed unavailable, or whose model kept producing
unusable output, rests PAUSED instead of FAILED.

``llm_unavailable`` follows ``XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED`` like
``persistence_failure``; ``model_output_invalid`` is recorded but never paused
for until the automatic-resume executor ships. Both reach settlement as an
unsuccessful result, so the result paths carry them: a new run's
(``_finalize_task_execution_result_isolated``), a resumed run's
(``_finalize_resumed_task``) and the shared channel leaf's
(``finalize_managed_task_lease_result``). Runs on SQLite and, when
``XAGENT_TEST_POSTGRES_URL`` is set, on PostgreSQL through the shared
``canonical`` fixture.
"""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import httpx
import openai
import pytest

from tests.core.agent.test_auto import decision_tool_response, plan_tool_response
from tests.core.agent.test_react import FakeLLM, FakeTool
from tests.core.agent.test_runner import FakeWorkspaceManager
from tests.web.services.channel_delivery_shared import (
    database_url as database_url_fixture,
)
from tests.web.services.channel_delivery_shared import selected as selected_fixture
from tests.web.services.coordinator_command_shared import claim_task_command
from tests.web.services.test_execution_event_persistence_settlement import (
    _run_scheduled_turn,
)
from tests.web.services.test_execution_event_recovery import tracer_for
from tests.web.services.test_persistence_failure_pause import (
    PERSISTENCE_RESULT,
    _assert_no_failure_in_model_context,
    _assert_paused,
    _assistant_lines,
    _break_decision,
    _checkpoint,
    _connection_reset,
    _expire_and_recover,
    _finalize,
    _finalize_and_release,
    _finalize_resumed,
)
from tests.web.services.test_persistence_failure_pause import (
    _late_bound_sessions as _late_bound_sessions_fixture,
)
from tests.web.services.test_persistence_failure_pause import (
    _prepare_run,
    _projected_task,
    _projection,
    _resume_raising,
    _schedule_failing_turn,
    _settled_facts,
    _start_run,
    _state,
    _without_interruption,
)
from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_execution_event_writer import engine as engine_fixture
from tests.web.services.test_task_execution_event_writer import (
    facts,
)
from tests.web.services.test_task_execution_event_writer import (
    task_id as task_id_fixture,
)
from tests.web.services.test_task_lease_expiry_interruption import (
    _fail_recording,
    _snapshot,
)
from xagent.core.agent import (
    Agent,
    AgentRunner,
    AutoPattern,
    ReActPattern,
)
from xagent.core.agent.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_TYPE,
    TraceCheckpointStore,
)
from xagent.core.agent.execution_adapter import INTERRUPTED_USER_MESSAGE
from xagent.core.agent.interruption import InterruptionReason
from xagent.core.agent.pattern.auto.auto import DECISION_TOOL_NAME
from xagent.core.agent.service import AgentService
from xagent.core.model.chat.error import retry_on
from xagent.core.model.chat.exceptions import LLMEmptyContentError, LLMTimeoutError
from xagent.core.retry import RetryWrapper
from xagent.core.retry.strategy import FixedDelay
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import get_session_local
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_auto_recovery import TaskAutoRecovery
from xagent.web.models.task_command import TaskExecutionCommand

# Imported here, not lazily: a module first imported while ``canonical``
# patches ``database.get_session_local`` would keep that patched factory.
from xagent.web.services import (
    agent_service_manager,
)
from xagent.web.services import shared_channel_execution as shared
from xagent.web.services import (
    task_auto_recovery,
    task_command_execution,
    task_coordinator_runtime,
    task_event_bridge,
    task_execution,
    task_orchestrator,
)
from xagent.web.services.client_error_messages import CLIENT_SAFE_TASK_FAILURE
from xagent.web.services.execution_result_projection import (
    interrupted_channel_result,
    project_execution_result_for_channel,
)
from xagent.web.services.managed_task_lease import (
    finalize_managed_task_lease_result,
)
from xagent.web.services.task_auto_recovery import (
    TASK_INTERRUPTION_PAUSED_TRIGGER_ERROR,
    InterruptionSettlementDeferred,
    settlement_interruption_for_failure,
    settlement_interruption_for_result,
    settlement_pause_enabled,
)
from xagent.web.services.task_execution import (
    _acquire_resume_task_lease,
    _settle_resumed_task_lease,
)
from xagent.web.services.task_execution_context_service import (
    TaskExecutionRecoverySnapshot,
)
from xagent.web.services.task_execution_event_writer import append_fact_no_commit
from xagent.web.services.task_lease_recovery import TASK_LEASE_PAUSED_TRIGGER_ERROR
from xagent.web.services.task_lease_service import (
    TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR,
    CheckpointRecoveryResolution,
    CheckpointRecoveryVerdict,
    TaskLease,
    bind_task_lease_context,
)
from xagent.web.services.task_orchestrator import (
    TaskTurnPayload,
    settle_task_lease_isolated,
)
from xagent.web.services.task_settlement_report import SettlementReport

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture
_late_bound_sessions = _late_bound_sessions_fixture
database_url = database_url_fixture
selected = selected_fixture

PAUSE_SWITCH = "XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED"
PAUSED_FACT = {"status": "paused", "result": {"error": None}}


@pytest.fixture(autouse=True)
def _switch_defaults(monkeypatch):
    # The infra switch is off by default until automatic resume ships; these
    # tests exercise it on (``infra_default`` cases pin the default).
    monkeypatch.setenv(PAUSE_SWITCH, "true")


# ------------------------------------------------- real runs, then resumed


def _calculator_call() -> dict[str, Any]:
    return {
        "content": "calculate",
        "tool_calls": [
            {
                "id": "call1",
                "function": {
                    "name": "calculator",
                    "arguments": '{"expression":"2+2"}',
                },
            }
        ],
    }


def _empty_final_answer() -> dict[str, Any]:
    """A ``final_answer`` call without an answer: twice ends the run."""
    return {
        "tool_calls": [
            {
                "id": "call_empty",
                "function": {"name": "final_answer", "arguments": '{"answer": ""}'},
            }
        ]
    }


def _final_answer(answer: str) -> dict[str, Any]:
    return {
        "tool_calls": [
            {
                "id": "call_final",
                "function": {
                    "name": "final_answer",
                    "arguments": '{"answer": "%s"}' % answer,
                },
            }
        ]
    }


class _UnavailableProvider:
    """A provider that keeps timing out, behind the real retry loop.

    Each ``chat`` call retries the request ``attempts`` times through
    ``RetryWrapper`` with the production retry predicate, then raises what
    the last attempt raised -- what a pattern sees once retries exhaust.
    """

    def __init__(self, before: list[Any], *, attempts: int = 3) -> None:
        self.before = list(before)
        self.attempts = attempts
        self.provider_requests = 0
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.before:
            return self.before.pop(0)
        provider = self

        class _Target:
            def invoke(self, *_args: Any, **_kwargs: Any) -> Any:
                raise AssertionError("async only")

            async def ainvoke(self, *_args: Any, **_kwargs: Any) -> Any:
                provider.provider_requests += 1
                raise LLMTimeoutError("provider timed out")

        return await RetryWrapper(
            _Target(),
            strategy=FixedDelay(0),
            max_retries=self.attempts,
            retry_on=retry_on,
        ).ainvoke()


def _runner(tid: int, pattern: Any, llm: Any, tool: FakeTool, tmp_path) -> AgentRunner:
    return AgentRunner(
        agent=Agent(name="provider", patterns=[pattern], tools=[tool], llm=llm),
        tracer=TraceCheckpointStore(tracer_for(tid)),
        workspace_manager=FakeWorkspaceManager(tmp_path),
    )


def _service_result(result: dict[str, Any]) -> dict[str, Any]:
    """The top-level fields ``AgentExecutionAdapter`` forwards to settlement."""
    return {
        key: result[key]
        for key in ("success", "status", "error", "output", "interruption_reason")
        if key in result
    }


def _react() -> ReActPattern:
    return ReActPattern(max_iterations=4)


def _auto() -> AutoPattern:
    return AutoPattern(react_pattern=_react())


async def _resume(factory, tid: int, lease: TaskLease, pattern, llm, tool, tmp_path):
    """Manual resume: a resume lease on the same run, then the runner."""
    with factory() as db:
        user_id = int(db.get(Task, tid).user_id)
    resume_lease = _acquire_resume_task_lease(
        tid, user_id, lease.run_id, refuse_terminal_status=True
    )
    assert resume_lease is not None and resume_lease.run_id == lease.run_id
    with bind_task_lease_context(resume_lease):
        return await _runner(tid, pattern, llm, tool, tmp_path).resume(
            str(tid), task="2+2"
        )


def _settle_new_run(factory, tid: int, lease: TaskLease, result: dict[str, Any]):
    finalized = _finalize(factory, tid, lease, _service_result(result))
    # The scheduler's settlement releases the paused lease.
    assert settle_task_lease_isolated(lease)
    return finalized


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", ["react", "auto"])
@pytest.mark.parametrize("gate", ["today", "executor"])
async def test_invalid_tool_protocol_resume_samples_the_model_again(
    canonical, monkeypatch, tmp_path, wrapped, gate
):
    """The run gives up on the tool protocol after a tool call.

    ``today``: settlement never pauses for it yet; the run fails as before
    and the row records why. ``executor``: with the deferral lifted, as the
    automatic-resume executor will lift it, the run pauses, and resuming it
    continues from its checkpoint with a NEW model call -- it does not replay
    the cached failure (Auto caches the child's result in its
    ``auto_after_child`` checkpoint) and does not re-run the tool."""

    if gate == "executor":
        monkeypatch.setattr(
            task_auto_recovery, "SETTLEMENT_PAUSE_DEFERRED_REASONS", frozenset()
        )
    factory, tid = canonical
    lease = _start_run(factory, tid)
    tool = FakeTool()
    decision = [decision_tool_response("react", "Needs a tool.")]
    first_llm = FakeLLM(
        responses=[
            *(decision if wrapped == "auto" else []),
            _calculator_call(),
            _empty_final_answer(),
            _empty_final_answer(),
        ]
    )
    pattern = _auto() if wrapped == "auto" else _react()
    with bind_task_lease_context(lease):
        result = await _runner(tid, pattern, first_llm, tool, tmp_path).run(
            task="2+2", execution_id=str(tid)
        )
    assert result["status"] == "invalid_tool_protocol"
    assert result["interruption_reason"] == "model_output_invalid"
    assert len(tool.calls) == 1
    # The checkpoint a resume loads: Auto's carries the cached child result.
    with bind_task_lease_context(lease):
        latest = await TraceCheckpointStore(tracer_for(tid)).load_latest_checkpoint(
            str(tid)
        )
    if wrapped == "auto":
        assert latest["label"] == "auto_after_child"
        cached = latest["pattern_state"]["last_result"]
        assert (cached["success"], cached["status"]) == (
            False,
            "invalid_tool_protocol",
        )
    else:
        assert latest["label"] == "invalid_tool_protocol"

    finalized = _settle_new_run(factory, tid, lease, result)

    if gate == "today":
        assert finalized.report.paused_for is None
        task, row, events = _state(factory, tid)
        assert task.status == TaskStatus.FAILED
        assert task.error_message == result["error"]
        assert (row.reason, row.state) == ("model_output_invalid", "manual")
        assert [(e.event, e.detail["task_status"]) for e in events] == [
            ("interrupted", "failed")
        ]
        return
    assert finalized.report.paused_for is InterruptionReason.MODEL_OUTPUT_INVALID
    _assert_paused(factory, tid, lease, reason="model_output_invalid")
    assert _settled_facts(factory, tid) == [PAUSED_FACT]
    assert _assistant_lines(factory, tid) == []
    _assert_no_failure_in_model_context(factory, tid)

    resume_llm = FakeLLM(responses=[_final_answer("4")])
    resumed = await _resume(
        factory,
        tid,
        lease,
        _auto() if wrapped == "auto" else _react(),
        resume_llm,
        tool,
        tmp_path,
    )

    assert resumed["success"], resumed
    assert resumed["output"] == "4"
    # One new ReAct call: no replayed failure and no fresh routing decision.
    assert len(resume_llm.calls) == 1
    offered = {schema["function"]["name"] for schema in resume_llm.calls[0]["tools"]}
    assert "final_answer" in offered and DECISION_TOOL_NAME not in offered
    assert len(tool.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", ["react", "auto"])
async def test_unavailable_provider_pauses_and_the_run_resumes(
    canonical, tmp_path, wrapped
):
    """The provider times out until the retry loop gives up; the run pauses
    as ``llm_unavailable`` and a manual resume against a healthy provider
    finishes it from its checkpoint without re-running the tool."""

    factory, tid = canonical
    lease = _start_run(factory, tid)
    tool = FakeTool()
    decision = [decision_tool_response("react", "Needs a tool.")]
    provider = _UnavailableProvider(
        [*(decision if wrapped == "auto" else []), _calculator_call()]
    )
    pattern = _auto() if wrapped == "auto" else _react()
    with bind_task_lease_context(lease):
        result = await _runner(tid, pattern, provider, tool, tmp_path).run(
            task="2+2", execution_id=str(tid)
        )
    assert result["success"] is False
    assert result["interruption_reason"] == "llm_unavailable"
    assert provider.provider_requests == provider.attempts
    assert len(tool.calls) == 1

    finalized = _settle_new_run(factory, tid, lease, result)

    assert finalized.report.paused_for is InterruptionReason.LLM_UNAVAILABLE
    row = _assert_paused(factory, tid, lease, reason="llm_unavailable")
    assert row.last_error == result["error"]
    assert _settled_facts(factory, tid) == [PAUSED_FACT]
    _assert_no_failure_in_model_context(factory, tid)

    healthy = FakeLLM(responses=[_final_answer("4")])
    resumed = await _resume(
        factory,
        tid,
        lease,
        _auto() if wrapped == "auto" else _react(),
        healthy,
        tool,
        tmp_path,
    )

    assert resumed["success"], resumed
    assert resumed["output"] == "4"
    assert len(healthy.calls) == 1
    assert len(tool.calls) == 1


# ------------------------------------------------------ DAG runs never pause


class _ProviderGoesDown:
    """Answers its scripted responses, then times out on every call."""

    model_name = "fake-llm"

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.responses:
            return self.responses.pop(0)
        raise LLMTimeoutError("provider timed out")


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", ["dag_plan_execute", "auto"])
@pytest.mark.parametrize("failing", ["planning", "step"])
async def test_dag_runs_never_pause_for_provider_failures(canonical, pattern, failing):
    """A DAG run (think mode, or Auto routed to it) swallows its provider
    failures into an ordinary failed result without an interruption reason,
    so settlement fails it as before (v1 leaves DAG LLM failures out)."""

    llm = _ProviderGoesDown(
        [
            *(
                [decision_tool_response("plan_execute", "Plan it.")]
                if pattern == "auto"
                else []
            ),
            *(
                [plan_tool_response([{"id": "answer", "task": "Answer directly"}])]
                if failing == "step"
                else []
            ),
        ]
    )
    service = AgentService(
        name=f"dag-{pattern}-{failing}",
        id=f"dag-{pattern}-{failing}",
        pattern=pattern,
        llm=llm,
        tools=[FakeTool()],
        tool_config=None,
    )
    service.allowed_skills = []

    result = await service.execute_task("Plan then answer", task_id="dag-run")

    assert result["success"] is False
    assert result["agent_result"]["failure_reason"] == (
        "step_failed" if failing == "step" else "plan_generation_error"
    )
    assert "interruption_reason" not in result
    assert settlement_interruption_for_result(result) is None

    factory, tid = canonical
    lease = _start_run(factory, tid)
    _checkpoint(factory, tid, lease.run_id)
    settled = {
        key: value
        for key, value in result.items()
        if key in {"success", "status", "error", "output"}
    }
    finalized = _settle_new_run(factory, tid, lease, settled)

    assert finalized.report.paused_for is None
    task, row, _events = _state(factory, tid)
    assert task.status == TaskStatus.FAILED
    assert row is None


# --------------------------------------------------------- switches, unit

LLM_RESULT = {
    "success": False,
    "status": "failed",
    "output": "All patterns failed",
    "error": "All 1 patterns failed or returned unsuccessful results.",
    "interruption_reason": "llm_unavailable",
}
MODEL_RESULT = {
    "success": False,
    "status": "invalid_tool_protocol",
    "output": "The model returned an invalid tool protocol response.",
    "error": (
        "The model returned an invalid tool protocol response after one repair attempt."
    ),
    "interruption_reason": "model_output_invalid",
}
RESULTS = {
    "llm_unavailable": LLM_RESULT,
    "model_output_invalid": MODEL_RESULT,
    "persistence_failure": PERSISTENCE_RESULT,
}


def test_settlement_acts_on_provider_and_model_output_failures():
    assert settlement_interruption_for_result(LLM_RESULT) is (
        InterruptionReason.LLM_UNAVAILABLE
    )
    assert settlement_interruption_for_result(MODEL_RESULT) is (
        InterruptionReason.MODEL_OUTPUT_INVALID
    )
    # The status alone names unusable output, with or without the runner's key.
    bare = {k: v for k, v in MODEL_RESULT.items() if k != "interruption_reason"}
    assert settlement_interruption_for_result(bare) is (
        InterruptionReason.MODEL_OUTPUT_INVALID
    )
    assert settlement_interruption_for_failure(LLMTimeoutError("timed out")) is (
        InterruptionReason.LLM_UNAVAILABLE
    )
    assert settlement_interruption_for_failure(LLMEmptyContentError("empty")) is (
        InterruptionReason.MODEL_OUTPUT_INVALID
    )
    # Quota wins over either.
    for result in (LLM_RESULT, MODEL_RESULT):
        assert (
            settlement_interruption_for_result({**result, "status": "quota_exceeded"})
            is None
        )
    # Still never: a lease expiry is TTL recovery's, not a settlement's.
    assert (
        settlement_interruption_for_result(
            {"success": False, "interruption_reason": "lease_expired"}
        )
        is None
    )


@pytest.mark.parametrize(
    ("infra", "enabled"),
    [
        ("true", {"persistence_failure", "llm_unavailable", "user_pause"}),
        ("false", set()),
        (None, set()),
    ],
    ids=["on", "off", "default"],
)
def test_settlement_switches_per_reason(monkeypatch, infra, enabled):
    """model_output_invalid never pauses until the executor ships, and the
    switch defaults off."""
    if infra is None:
        monkeypatch.delenv(PAUSE_SWITCH)
    else:
        monkeypatch.setenv(PAUSE_SWITCH, infra)
    names = {
        "persistence_failure": InterruptionReason.PERSISTENCE_FAILURE,
        "llm_unavailable": InterruptionReason.LLM_UNAVAILABLE,
        "model": InterruptionReason.MODEL_OUTPUT_INVALID,
        "user_pause": InterruptionReason.USER_PAUSE,
    }
    assert {
        name for name, reason in names.items() if settlement_pause_enabled(reason)
    } == enabled


# ------------------------------- result paths: S1, S3 and the shared channel S4

# PR4a covers ``persistence_failure`` on S1 and S3; S4 gains all three here.
_PATHS = ("result", "resumed_result", "channel")
_PATH_REASONS = [
    (path, reason)
    for path in _PATHS
    for reason in RESULTS
    if path == "channel" or reason != "persistence_failure"
]
# The reasons settlement pauses for today.
_PAUSE_PATH_REASONS = [
    case for case in _PATH_REASONS if case[1] != "model_output_invalid"
]
CASES = pytest.mark.parametrize(("path", "reason"), _PATH_REASONS)
PAUSE_CASES = pytest.mark.parametrize(("path", "reason"), _PAUSE_PATH_REASONS)


def _completed_channel_command(factory, ids, lease) -> tuple[int, dict[str, Any]]:
    """A completed channel START command of ``lease``'s run, as the worker
    leaves it before the leaf finalizes; returns the leaf's ``completion``."""
    with factory() as db:
        command = TaskExecutionCommand(
            task_id=ids["task"],
            command_id=f"channel-{ids['turn_id']}",
            kind="start",
            payload={},
            target_run_id=lease.run_id,
            status="completed",
            result={"run_id": lease.run_id, "lease_attempt_id": lease.attempt_id},
        )
        db.add(command)
        db.commit()
        ids["command"] = int(command.id)
    return ids["command"], {"success": False, "status": "failed"}


def _settle_channel(
    factory,
    ids,
    lease,
    result,
    *,
    settle_interruption: bool = True,
    report: SettlementReport | None = None,
    finalize_context: Any = None,
) -> None:
    """Settle ``result`` as the shared channel leaf does, completing a real
    channel command (its ``channel_result`` and the task output included)."""
    projection = project_execution_result_for_channel(dict(result))
    command_id, _placeholder = _completed_channel_command(factory, ids, lease)
    durable_result = {
        "success": projection.task_status != TaskStatus.FAILED,
        "status": str(result.get("status") or projection.task_status.value),
        "output": projection.transcript_content,
        "completion_outcome": projection.completion_outcome,
    }
    with factory() as db, finalize_context or contextlib.nullcontext():
        assert finalize_managed_task_lease_result(
            db,
            lease,
            status=projection.task_status,
            assistant_content=projection.transcript_content,
            turn_id=ids["turn_id"],
            interactions=projection.interactions,
            message_type=projection.message_type,
            error_message=projection.diagnostic_error,
            execution_result=dict(result),
            completion=(command_id, durable_result),
            settle_interruption=settle_interruption,
            report=report,
        )


def _full_projection(factory, ids, lease) -> dict[str, Any]:
    """PR4a's projection plus the channel command's ``channel_result``."""
    projection = _projection(factory, ids, lease)
    if "command" in ids:
        with factory() as db:
            result = db.get(TaskExecutionCommand, ids["command"]).result
        projection["channel_result"] = result.get("channel_result")
    return projection


def _settle_result(path: str, factory, ids, lease, result) -> None:
    if path == "result":
        _finalize_and_release(factory, ids, lease, dict(result))
    elif path == "resumed_result":
        assert _finalize_resumed(ids, lease, result)["lease_released"]
    else:
        _settle_channel(factory, ids, lease, result)


def _without_result_interruption(m: pytest.MonkeyPatch) -> None:
    """Settle as before this series: no result path sees an interruption."""
    m.setattr(task_execution, "settlement_interruption_for_result", lambda _r: None)
    m.setattr(task_auto_recovery, "settlement_interruption_for_result", lambda _r: None)


def _without_settled_reason(projection: dict[str, Any], reason: str):
    """A quota stop's settled fact keeps the result, reason included."""
    for fact in projection["settled"]:
        assert fact["result"].pop("interruption_reason") == reason
    return projection


def test_channel_pause_fills_the_report_after_the_commit(canonical):
    factory, _tid = canonical
    ids = _projected_task(factory)
    lease = _prepare_run(factory, ids)
    report = SettlementReport()

    _settle_channel(factory, ids, lease, LLM_RESULT, report=report)

    assert report.paused_for is InterruptionReason.LLM_UNAVAILABLE
    assert report.control_state["status"] == "paused"
    assert report.control_state["run_id"] == lease.run_id


def test_channel_pause_leaves_the_report_empty_when_the_commit_fails(canonical):
    factory, _tid = canonical
    ids = _projected_task(factory)
    lease = _prepare_run(factory, ids)
    report = SettlementReport()

    def failing_commit(self):
        raise RuntimeError("commit failed")

    with pytest.raises(RuntimeError, match="commit failed"):
        _settle_channel(
            factory,
            ids,
            lease,
            LLM_RESULT,
            report=report,
            finalize_context=patch("sqlalchemy.orm.Session.commit", failing_commit),
        )

    assert report == SettlementReport()


@PAUSE_CASES
def test_pause_projects_like_lease_recovery(canonical, path, reason):
    factory, _tid = canonical
    paused, recovered = _projected_task(factory), _projected_task(factory)
    paused_lease = _prepare_run(factory, paused)
    recovered_lease = _prepare_run(factory, recovered)

    _settle_result(path, factory, paused, paused_lease, RESULTS[reason])
    assert _expire_and_recover(factory, recovered) == TaskStatus.PAUSED

    left = _full_projection(factory, paused, paused_lease)
    right = _full_projection(factory, recovered, recovered_lease)
    if path == "channel":
        # Lease recovery leaves no channel result; a reader derives this one.
        assert left.pop("channel_result") == interrupted_channel_result()
    assert left.pop("trigger_run")[1] == TASK_INTERRUPTION_PAUSED_TRIGGER_ERROR
    assert right.pop("trigger_run")[1] == TASK_LEASE_PAUSED_TRIGGER_ERROR
    # Only lease recovery reconciles orphaned delivery rows.
    left.pop("delivery")
    right.pop("delivery")
    assert left == right
    assert left["status"] == TaskStatus.PAUSED
    assert left["output"] == "previous answer"
    assert left["assistant"] == []
    assert left["settled"] == [PAUSED_FACT]
    task, row, events = _state(factory, paused["task"])
    assert (row.reason, row.state) == (reason, "manual")
    assert row.paused_state_version == task.state_version
    assert row.last_error == RESULTS[reason]["error"]
    assert [(e.event, e.detail["task_status"]) for e in events] == [
        ("interrupted", "paused")
    ]
    _assert_no_failure_in_model_context(factory, paused["task"])


def _switch_case(m: pytest.MonkeyPatch, case: str) -> str:
    """Apply ``case``; return the task source it settles."""
    if case == "infra_off":
        m.setenv(PAUSE_SWITCH, "false")
    elif case in {"infra_default", "pause_requested"}:
        if case == "infra_default":
            m.delenv(PAUSE_SWITCH)
    return "sdk" if case == "ineligible" else "trigger"


@pytest.mark.parametrize(
    ("path", "reason", "case"),
    [
        (path, reason, case)
        for path, reason in _PAUSE_PATH_REASONS
        for case in ("infra_default", "infra_off", "ineligible")
    ],
)
def test_unpaused_runs_settle_exactly_as_before(
    canonical, monkeypatch, path, reason, case
):
    """The switch is off (or unset: it defaults off), or the task is
    ineligible: the same FAILED settlement as without an interruption, plus a
    ``disabled`` (or ``ineligible``) row."""

    source = _switch_case(monkeypatch, case)
    factory, _tid = canonical
    gated = _projected_task(factory, source=source)
    baseline = _projected_task(factory, source=source)
    gated_lease = _prepare_run(factory, gated)
    baseline_lease = _prepare_run(factory, baseline)

    _settle_result(path, factory, gated, gated_lease, RESULTS[reason])
    with monkeypatch.context() as m:
        _without_result_interruption(m)
        _settle_result(path, factory, baseline, baseline_lease, RESULTS[reason])

    gated_projection = _full_projection(factory, gated, gated_lease)
    assert gated_projection == _full_projection(factory, baseline, baseline_lease)
    assert gated_projection["status"] == TaskStatus.FAILED
    _task, row, events = _state(factory, gated["task"])
    expected = "ineligible" if case == "ineligible" else "disabled"
    assert (row.reason, row.state) == (reason, expected)
    assert [e.detail["task_status"] for e in events] == ["failed"]
    assert _state(factory, baseline["task"])[1] is None


@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize(
    "case", ["infra_on", "infra_off", "infra_default", "pause_requested", "ineligible"]
)
def test_model_output_invalid_never_pauses_yet(canonical, monkeypatch, path, case):
    """Until the automatic-resume executor ships, unusable model output
    settles exactly as without an interruption whatever the switch, the
    checkpoint or the control state say; the row records it as ``manual``
    (no switch would help), or ``ineligible``."""

    if case == "infra_off":
        monkeypatch.setenv(PAUSE_SWITCH, "false")
    source = _switch_case(monkeypatch, case)
    factory, _tid = canonical
    gated = _projected_task(factory, source=source)
    baseline = _projected_task(factory, source=source)
    gated_lease = _prepare_run(factory, gated)
    baseline_lease = _prepare_run(factory, baseline)
    if case == "pause_requested":
        with factory() as db:
            for ids in (gated, baseline):
                db.get(Task, ids["task"]).control_state = "pause_requested"
            db.commit()

    _settle_result(path, factory, gated, gated_lease, MODEL_RESULT)
    with monkeypatch.context() as m:
        _without_result_interruption(m)
        _settle_result(path, factory, baseline, baseline_lease, MODEL_RESULT)

    gated_projection = _full_projection(factory, gated, gated_lease)
    assert gated_projection == _full_projection(factory, baseline, baseline_lease)
    assert gated_projection["status"] == TaskStatus.FAILED
    _task, row, events = _state(factory, gated["task"])
    expected = "ineligible" if case == "ineligible" else "manual"
    assert (row.reason, row.state) == ("model_output_invalid", expected)
    assert [e.detail["task_status"] for e in events] == ["failed"]


@PAUSE_CASES
@pytest.mark.parametrize(
    "case",
    ["unknown_tool_effect", "not_recoverable", "pause_requested", "resume_requested"],
)
def test_paths_decide_every_verdict(canonical, monkeypatch, path, reason, case):
    factory, _tid = canonical
    ids = _projected_task(factory)
    lease = _prepare_run(factory, ids, checkpoint=case != "not_recoverable")
    if case == "unknown_tool_effect":
        monkeypatch.setattr(
            task_auto_recovery,
            "resolve_checkpoint_recovery_with_data",
            lambda _db, _candidate: CheckpointRecoveryResolution(
                CheckpointRecoveryVerdict.UNKNOWN_TOOL_EFFECT
            ),
        )
    if case in {"pause_requested", "resume_requested"}:
        with factory() as db:
            db.get(Task, ids["task"]).control_state = case
            db.commit()

    _settle_result(path, factory, ids, lease, RESULTS[reason])

    task, row, _events = _state(factory, ids["task"])
    projection = _full_projection(factory, ids, lease)
    if case in {"unknown_tool_effect", "not_recoverable"}:
        assert task.status == TaskStatus.FAILED
        assert task.error_message == (
            TASK_UNKNOWN_TOOL_EFFECT_SETTLEMENT_ERROR
            if case == "unknown_tool_effect"
            else RESULTS[reason]["error"]
        )
        assert (row.reason, row.state) == (case, "manual")
        assert [fact["status"] for fact in projection["settled"]] == ["failed"]
    else:
        assert (task.status, task.control_state) == (TaskStatus.PAUSED, "paused")
        assert task.runner_id is None
        assert (row.reason, row.state) == (
            "user_pause" if case == "pause_requested" else reason,
            "manual",
        )
        assert projection["settled"] == [PAUSED_FACT]
        assert projection["assistant"] == []


@CASES
@pytest.mark.parametrize("switch", ["on", "off"])
def test_quota_stop_wins_over_the_reason(canonical, monkeypatch, path, reason, switch):
    if switch == "off":
        monkeypatch.setenv(PAUSE_SWITCH, "false")
    factory, _tid = canonical
    quota = {**RESULTS[reason], "status": "quota_exceeded", "error_code": "q"}
    plain_quota = {k: v for k, v in quota.items() if k != "interruption_reason"}
    gated, baseline = _projected_task(factory), _projected_task(factory)
    gated_lease = _prepare_run(factory, gated)
    baseline_lease = _prepare_run(factory, baseline)

    _settle_result(path, factory, gated, gated_lease, quota)
    _settle_result(path, factory, baseline, baseline_lease, plain_quota)

    gated_projection = _without_settled_reason(
        _full_projection(factory, gated, gated_lease), reason
    )
    assert gated_projection == _full_projection(factory, baseline, baseline_lease)
    assert gated_projection["status"] == TaskStatus.FAILED
    assert _state(factory, gated["task"])[1] is None


@PAUSE_CASES
@pytest.mark.parametrize("failure", ["python", "database", "marker"])
def test_recording_failure_does_not_change_the_pause(
    canonical, monkeypatch, path, reason, failure
):
    factory, _tid = canonical
    recorded, unrecorded = _projected_task(factory), _projected_task(factory)
    recorded_lease = _prepare_run(factory, recorded)
    unrecorded_lease = _prepare_run(factory, unrecorded)

    _settle_result(path, factory, recorded, recorded_lease, RESULTS[reason])
    with monkeypatch.context() as m:
        _fail_recording(m, failure)
        _settle_result(path, factory, unrecorded, unrecorded_lease, RESULTS[reason])

    left = _full_projection(factory, recorded, recorded_lease)
    assert left == _full_projection(factory, unrecorded, unrecorded_lease)
    assert left["status"] == TaskStatus.PAUSED
    assert left["trigger_run"][1] == TASK_INTERRUPTION_PAUSED_TRIGGER_ERROR
    assert _state(factory, unrecorded["task"])[1:] == (None, [])


@PAUSE_CASES
@pytest.mark.parametrize("failure", ["eligibility", "checkpoint_read"])
def test_undecidable_interruption_keeps_the_lease_for_ttl(
    canonical, monkeypatch, path, reason, failure
):
    """A connectivity failure deciding the interruption defers it: nothing
    settles, the run stays RUNNING under its lease, and TTL recovery pauses
    it once the database answers again."""

    factory, _tid = canonical
    ids = _projected_task(factory)
    lease = _prepare_run(factory, ids)

    with monkeypatch.context() as m:
        _break_decision(m, failure, _connection_reset())
        with pytest.raises(InterruptionSettlementDeferred) as deferred:
            if path == "result":
                _finalize(factory, ids["task"], lease, dict(RESULTS[reason]))
            elif path == "resumed_result":
                _finalize_resumed(ids, lease, RESULTS[reason])
            else:
                _settle_channel(factory, ids, lease, RESULTS[reason])
        assert deferred.value.__cause__ is not None

    task, row, _events = _state(factory, ids["task"])
    assert task.status == TaskStatus.RUNNING
    assert (task.runner_id, task.run_id) == (lease.runner_id, lease.run_id)
    assert row is None
    assert _full_projection(factory, ids, lease)["settled"] == []

    assert _expire_and_recover(factory, ids) == TaskStatus.PAUSED
    _task, row, _events = _state(factory, ids["task"])
    assert (row.reason, row.state) == ("lease_expired", "manual")


@PAUSE_CASES
@pytest.mark.parametrize("failure", ["eligibility", "checkpoint_read"])
def test_reproducible_decision_fault_settles_as_before(
    canonical, monkeypatch, path, reason, failure
):
    """A fault that would recur on every attempt does not defer (TTL recovery
    would hit it too and leave the run RUNNING forever): the run settles
    exactly as without an interruption and releases its lease."""

    factory, _tid = canonical
    gated, baseline = _projected_task(factory), _projected_task(factory)
    gated_lease = _prepare_run(factory, gated)
    baseline_lease = _prepare_run(factory, baseline)

    with monkeypatch.context() as m:
        _break_decision(m, failure, TypeError("resolver bug"))
        _settle_result(path, factory, gated, gated_lease, RESULTS[reason])
    with monkeypatch.context() as m:
        _without_result_interruption(m)
        _settle_result(path, factory, baseline, baseline_lease, RESULTS[reason])

    gated_projection = _full_projection(factory, gated, gated_lease)
    assert gated_projection == _full_projection(factory, baseline, baseline_lease)
    assert gated_projection["status"] == TaskStatus.FAILED
    assert gated_projection["runner_id"] is None


def test_combined_channel_settlement_is_unchanged(canonical):
    """The combined bots finalize without deciding interruptions: their
    tasks are never eligible, so they fail as before, with no row."""

    factory, _tid = canonical
    gated, baseline = _projected_task(factory), _projected_task(factory)
    gated_lease = _prepare_run(factory, gated)
    baseline_lease = _prepare_run(factory, baseline)
    plain = {k: v for k, v in LLM_RESULT.items() if k != "interruption_reason"}

    _settle_channel(factory, gated, gated_lease, LLM_RESULT, settle_interruption=False)
    _settle_channel(factory, baseline, baseline_lease, plain, settle_interruption=False)

    gated_projection = _without_settled_reason(
        _full_projection(factory, gated, gated_lease), "llm_unavailable"
    )
    assert gated_projection == _full_projection(factory, baseline, baseline_lease)
    assert gated_projection["status"] == TaskStatus.FAILED
    assert _state(factory, gated["task"])[1] is None


# ------------------------------------------- S2: the rare raised provider error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "reason"),
    [(LLMTimeoutError("provider timed out"), "llm_unavailable")],
)
async def test_raised_provider_failure_pauses_on_the_exception_path(
    canonical, error, reason
):
    """The runner turns pattern failures into results, so these rarely raise
    past it; when one does, the exception path decides it the same way."""

    factory, _tid = canonical
    ids = _projected_task(factory)
    lease = _prepare_run(factory, ids)

    frames = await _schedule_failing_turn(factory, ids, lease, error)

    assert frames == ["task_paused"]
    task, row, _events = _state(factory, ids["task"])
    assert (task.status, task.control_state) == (TaskStatus.PAUSED, "paused")
    assert (row.reason, row.state) == (reason, "manual")


@pytest.mark.asyncio
async def test_raised_model_output_failure_fails_as_before(canonical, monkeypatch):
    factory, _tid = canonical
    gated, baseline = _projected_task(factory), _projected_task(factory)
    gated_lease = _prepare_run(factory, gated)
    baseline_lease = _prepare_run(factory, baseline)

    gated_frames = await _schedule_failing_turn(
        factory, gated, gated_lease, LLMEmptyContentError("empty response")
    )
    with monkeypatch.context() as m:
        _without_interruption(m)
        baseline_frames = await _schedule_failing_turn(
            factory, baseline, baseline_lease, LLMEmptyContentError("empty response")
        )

    gated_projection = _full_projection(factory, gated, gated_lease)
    assert gated_projection == _full_projection(factory, baseline, baseline_lease)
    assert gated_projection["status"] == TaskStatus.FAILED
    assert gated_frames == baseline_frames == ["task_error"]
    _task, row, _events = _state(factory, gated["task"])
    assert (row.reason, row.state) == ("model_output_invalid", "manual")


# ------------------------------------- S4: a shared channel turn, end to end


PREVIOUS_OUTPUT = "previous channel answer"


async def _run_shared_channel_turn(
    selected, monkeypatch, execution_result: dict[str, Any]
) -> TaskExecutionCommand:
    """Run one accepted channel turn through the worker's command executor.

    The fake agent writes a recoverable checkpoint for its run and gives the
    task a previous output, then returns ``execution_result``.
    """

    monkeypatch.setenv("XAGENT_TASK_EXECUTION_ROLE", "worker")
    monkeypatch.setattr(task_coordinator_runtime, "get_runner_id", lambda: "worker-1")
    # A registry an earlier test left on this loop is bound to its database.
    monkeypatch.setattr(task_coordinator_runtime, "_registry", None)
    assert task_coordinator_runtime.get_session_local is get_session_local
    with get_session_local()() as db:
        db.get(Task, selected.selection.task_id).conversation_storage_version = 2
        db.commit()
    command_id = shared._accept_channel_turn(
        selected, TaskTurnPayload("hello"), "ingress"
    )
    with get_session_local()() as db:
        command = await claim_task_command(
            db, runner_id="worker-1", command_db_id=command_id
        )
    service = SimpleNamespace(
        tracer=SimpleNamespace(add_handler=Mock(), remove_handler=Mock()),
        workspace=None,
        set_conversation_history=Mock(),
        set_execution_context_messages=Mock(),
        set_recovered_skill_context=Mock(),
    )

    async def execute(**kwargs: Any) -> dict[str, Any]:
        run_id = kwargs["task_lease"].run_id
        with get_session_local()() as db:
            append_fact_no_commit(
                db,
                task_id=selected.selection.task_id,
                kind="recovery_state",
                key=f"runtime:{run_id}",
                payload={
                    "data": {
                        "checkpoint_type": CHECKPOINT_TYPE,
                        "snapshot_schema_version": CHECKPOINT_SCHEMA_VERSION,
                        "execution_id": str(selected.selection.task_id),
                        "snapshot": _snapshot(messages=2, iteration=1),
                    }
                },
                run_id=run_id,
            )
            # The output the task carries from its previous turn when this
            # run settles (starting the run does not keep the stored one).
            db.get(Task, selected.selection.task_id).output = PREVIOUS_OUTPUT
            db.commit()
        return dict(execution_result)

    manager = SimpleNamespace(
        get_agent_for_task=AsyncMock(return_value=service),
        execute_task=AsyncMock(side_effect=execute),
    )
    monkeypatch.setattr(agent_service_manager, "get_agent_manager", lambda: manager)
    bridge = Mock()
    monkeypatch.setattr(task_event_bridge, "_bridge", bridge)
    monkeypatch.setattr(shared, "get_task_event_bridge", lambda: bridge)
    snapshot = SimpleNamespace(
        runtime_user=object(),
        task=SimpleNamespace(user_id=selected.selection.user_id, source="internal"),
        conversation_history=(),
        conversation_watermark=None,
        conversation_event_watermark=None,
        execution_recovery=TaskExecutionRecoverySnapshot(),
    )
    monkeypatch.setattr(
        task_orchestrator, "load_task_setup_snapshot_sync", lambda *a, **kw: snapshot
    )
    monkeypatch.setattr(task_orchestrator, "resolve_execution_scope", lambda *a: None)
    monkeypatch.setattr(
        task_execution,
        "background_task_manager",
        task_execution.BackgroundTaskManager(),
    )
    try:
        await task_command_execution.execute_durable_task_command(command)
        async with asyncio.timeout(10):
            while shared._read_channel_result(command.id, selected.run_id) is None:
                await asyncio.sleep(0.01)
            # The channel result commits with the settlement, before the run
            # broadcasts it; let the run finish rather than have shutdown
            # cancel it mid-broadcast.
            await asyncio.gather(
                *task_execution.background_task_manager.running_tasks.values()
            )
    finally:
        await task_coordinator_runtime.close_task_coordinators()
        await task_execution.background_task_manager.shutdown()
    return command


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["llm_unavailable", "persistence_failure"])
async def test_shared_channel_turn_pauses_and_its_channel_reads_interrupted(
    selected, monkeypatch, reason
):
    """The shared channel leaf pauses an eligible interrupted run. Its
    channel gets the same "interrupted" reply a lease-expiry pause produces
    (until a later change holds the reply across an automatic resume)."""

    published = _capture_task_events(monkeypatch)

    command = await _run_shared_channel_turn(selected, monkeypatch, RESULTS[reason])

    tid = selected.selection.task_id
    channel_result = shared._read_channel_result(command.id, selected.run_id)
    assert channel_result == interrupted_channel_result()
    assert (
        project_execution_result_for_channel(channel_result).visible_text
        == INTERRUPTED_USER_MESSAGE
    )
    with get_session_local()() as db:
        task = db.get(Task, tid)
        assert (task.status, task.control_state) == (TaskStatus.PAUSED, "paused")
        assert (task.run_id, task.runner_id) == (selected.run_id, None)
        assert task.lease_attempt_id is None
        assert task.error_message is None
        # A pause keeps the previous output, as lease recovery does.
        assert task.output == PREVIOUS_OUTPUT
        assert (
            db.get(TaskExecutionCommand, command.id).result["channel_result"]
            == interrupted_channel_result()
        )
        row = db.get(TaskAutoRecovery, tid)
        assert (row.reason, row.state, row.run_id) == (
            reason,
            "manual",
            selected.run_id,
        )
        assert row.paused_state_version == task.state_version
        assert row.progress_marker is not None
        assert (
            db.query(TaskChatMessage).filter_by(task_id=tid, role="assistant").all()
            == []
        )
        assert [
            fact.payload for fact in facts(db, tid) if fact.kind == "execution_settled"
        ] == [PAUSED_FACT]
        state_version = task.state_version
    # A web viewer of the task learns of the pause, with the committed
    # control identity and why the run stopped.
    paused = [event for event in published if event["type"] == "task_paused"]
    assert len(paused) == 1
    assert paused[0]["interruption_reason"] == reason
    assert (
        paused[0]["task_id"],
        paused[0]["status"],
        paused[0]["control_state"],
        paused[0]["run_id"],
        paused[0]["state_version"],
    ) == (tid, "paused", "paused", selected.run_id, state_version)
    assert RESULTS[reason]["error"] not in str(paused)


@pytest.mark.asyncio
async def test_shared_channel_pause_broadcast_failure_keeps_the_pause(
    selected, monkeypatch, caplog
):
    """The pause is committed before it is announced: a failed broadcast is
    only logged, and the run, its channel reply and its row stand."""

    publish = AsyncMock(side_effect=RuntimeError("event bus down"))
    monkeypatch.setattr("xagent.web.services.task_events.publish_task_event", publish)

    with caplog.at_level("WARNING", logger=task_orchestrator.__name__):
        command = await _run_shared_channel_turn(
            selected, monkeypatch, RESULTS["llm_unavailable"]
        )

    tid = selected.selection.task_id
    assert [
        call.args[0]["type"]
        for call in publish.await_args_list
        if call.args[0].get("type") == "task_paused"
    ] == ["task_paused"]
    assert "broadcast failed" in caplog.text
    assert (
        shared._read_channel_result(command.id, selected.run_id)
        == interrupted_channel_result()
    )
    with get_session_local()() as db:
        task = db.get(Task, tid)
        assert (task.status, task.control_state) == (TaskStatus.PAUSED, "paused")
        assert task.runner_id is None
        row = db.get(TaskAutoRecovery, tid)
        assert (row.reason, row.paused_state_version) == (
            "llm_unavailable",
            task.state_version,
        )


def _capture_task_events(monkeypatch) -> list[dict[str, Any]]:
    """Record every task event published from here on."""
    published: list[dict[str, Any]] = []

    async def publish(event: dict[str, Any], _task_id: int) -> None:
        published.append(event)

    monkeypatch.setattr("xagent.web.services.task_events.publish_task_event", publish)
    return published


def _leaf_failed_channel_result(result: dict[str, Any]) -> dict[str, Any]:
    """The channel result the leaf has always stored for a FAILED result."""
    projection = project_execution_result_for_channel(dict(result))
    return {
        "success": False,
        "status": str(result.get("status") or projection.task_status.value),
        "output": projection.transcript_content,
        "completion_outcome": projection.completion_outcome,
        "chat_response": {
            "message": projection.transcript_content,
            "interactions": projection.interactions,
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "switch"),
    [(reason, "false") for reason in sorted(RESULTS)]
    + [("model_output_invalid", "true")],
)
async def test_shared_channel_turn_that_does_not_pause_fails_as_before(
    selected, monkeypatch, reason, switch
):
    """Switch off, or a reason settlement does not pause for yet: the channel
    command, its result and the task output are what the leaf has always
    written for a FAILED result."""

    monkeypatch.setenv(PAUSE_SWITCH, switch)
    published = _capture_task_events(monkeypatch)

    command = await _run_shared_channel_turn(selected, monkeypatch, RESULTS[reason])
    assert [event for event in published if event["type"] == "task_paused"] == []

    tid = selected.selection.task_id
    channel_result = shared._read_channel_result(command.id, selected.run_id)
    assert channel_result == _leaf_failed_channel_result(RESULTS[reason])
    assert channel_result["output"] == CLIENT_SAFE_TASK_FAILURE
    with get_session_local()() as db:
        stored = db.get(TaskExecutionCommand, command.id).result
        assert stored == {
            "run_id": selected.run_id,
            "lease_attempt_id": stored["lease_attempt_id"],
            "channel_result": channel_result,
        }
        task = db.get(Task, tid)
        assert task.status == TaskStatus.FAILED
        assert task.error_message == RESULTS[reason]["error"]
        # A FAILED channel result clears the output, as it always has.
        assert task.output is None
        row = db.get(TaskAutoRecovery, tid)
        assert (row.reason, row.state) == (
            reason,
            "manual" if reason == "model_output_invalid" else "disabled",
        )
        assert [
            message.content
            for message in db.query(TaskChatMessage).filter_by(
                task_id=tid, role="assistant"
            )
        ] == [CLIENT_SAFE_TASK_FAILURE]


@pytest.mark.asyncio
async def test_shared_channel_deferred_settlement_retains_the_lease(canonical):
    """An undecidable channel settlement raises out of the leaf; the
    scheduler keeps the lease for TTL recovery, as on the other paths."""

    factory, tid = canonical
    lease = _start_run(factory, tid)

    async def execute(**_kwargs: Any) -> None:
        raise InterruptionSettlementDeferred("checkpoint not resolvable")

    published = await _run_scheduled_turn(tid, lease, execute, channel=True)

    assert published == []
    task, row, _events = _state(factory, tid)
    assert task.status == TaskStatus.RUNNING
    assert task.runner_id == lease.runner_id
    assert row is None


# --------------------------------------- S3: the resume exception path, LLM


@pytest.mark.asyncio
async def test_resume_exception_passes_an_llm_reason_and_announces_the_pause():
    """A provider failure that escapes a resumed run reaches the resume
    exception path's settlement as ``llm_unavailable``."""

    def settle(lease, **kwargs):
        assert kwargs["interruption"] is InterruptionReason.LLM_UNAVAILABLE
        report = kwargs["report"]
        report.paused_for = InterruptionReason.LLM_UNAVAILABLE
        report.control_state.update(status="paused", run_id=lease.run_id)
        return True

    settle_mock = Mock(side_effect=settle)
    publish_pause, frames = await _resume_raising(
        LLMTimeoutError("provider timed out"), settle_mock
    )

    settle_mock.assert_called_once()
    publish_pause.assert_awaited_once_with(
        42,
        SettlementReport(
            control_state={"status": "paused", "run_id": "run-a"},
            paused_for=InterruptionReason.LLM_UNAVAILABLE,
        ),
    )
    assert "task_error" not in frames


def test_resume_exception_settlement_pauses_for_llm_on_a_real_database(canonical):
    factory, _tid = canonical
    ids = _projected_task(factory)
    lease = _prepare_run(factory, ids)
    report = SettlementReport()

    assert _settle_resumed_task_lease(
        lease,
        error_message="setup/run error: LLMTimeoutError: provider timed out",
        interruption=InterruptionReason.LLM_UNAVAILABLE,
        report=report,
    )

    assert report.paused_for is InterruptionReason.LLM_UNAVAILABLE
    projection = _full_projection(factory, ids, lease)
    assert (projection["status"], projection["runner_id"]) == (TaskStatus.PAUSED, None)
    assert projection["settled"] == [PAUSED_FACT]
    _task, row, _events = _state(factory, ids["task"])
    assert (row.reason, row.state) == ("llm_unavailable", "manual")


# ------------------------------------ provider refusals are not interruptions


def _insufficient_quota() -> RuntimeError:
    """How openai.py re-raises an exhausted-quota rate limit."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    sdk_error = openai.RateLimitError(
        "You exceeded your current quota, please check your plan and billing",
        response=httpx.Response(429, request=request),
        body={"code": "insufficient_quota", "type": "insufficient_quota"},
    )
    error = RuntimeError(f"OpenAI rate limit exceeded: {sdk_error.message}")
    error.__cause__ = sdk_error
    return error


@pytest.mark.asyncio
async def test_exhausted_quota_settles_failed_with_the_switch_on(canonical, tmp_path):
    """A provider that refuses the account (quota, credential) is terminal:
    a real ReAct run carries no interruption reason and settles FAILED."""

    factory, tid = canonical
    lease = _start_run(factory, tid)
    _checkpoint(factory, tid, lease.run_id)
    tool = FakeTool()

    class _OutOfQuota:
        async def chat(self, **_kwargs: Any) -> Any:
            raise _insufficient_quota()

    with bind_task_lease_context(lease):
        result = await _runner(tid, _react(), _OutOfQuota(), tool, tmp_path).run(
            task="2+2", execution_id=str(tid)
        )
    assert result["success"] is False
    assert "interruption_reason" not in result

    finalized = _settle_new_run(factory, tid, lease, result)

    assert finalized.report.paused_for is None
    task, row, _events = _state(factory, tid)
    assert task.status == TaskStatus.FAILED
    assert row is None
