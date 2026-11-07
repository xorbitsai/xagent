"""The per-run settlement notification handed to the hosting application.

Delivery is detached from the entry point that reports the run: a hook that
fails or never returns is logged and counted, and never changes how the run
ended.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import ExitStack
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from sqlalchemy import event, update
from sqlalchemy.orm import Session

from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_execution_event_writer import engine as engine_fixture
from tests.web.services.test_task_execution_event_writer import (
    task_id as task_id_fixture,
)
from xagent.core.agent import (
    AgentExecutionAdapter,
    AgentExecutionConfig,
    PatternResult,
)
from xagent.core.agent.interruption import InterruptionReason
from xagent.web.models.task import Task, TaskStatus
from xagent.web.services import (
    agent_service_manager,
    shared_channel_execution,
)
from xagent.web.services import task_orchestrator as orchestrator
from xagent.web.services import (
    turn_settlement,
)
from xagent.web.services.client_error_messages import (
    CONNECTOR_RUNTIME_CLIENT_ERROR_CODES,
    ClientErrorCode,
)
from xagent.web.services.task_execution_context_service import (
    TaskExecutionRecoverySnapshot,
)
from xagent.web.services.task_lease_service import (
    TaskLease,
    TaskLeaseLostError,
    acquire_task_lease,
)
from xagent.web.services.task_orchestrator import TaskTurnPayload, _schedule_bg
from xagent.web.services.task_settlement_report import SettlementReport
from xagent.web.services.turn_settlement import (
    TurnFacts,
    TurnModelUsage,
    TurnSettlement,
    TurnUsage,
    build_error_code,
    build_turn_settlement,
    drain_turn_settlement_hooks,
    notify_turn_settled,
    report_turn_settled,
    set_turn_settled_hook,
    settled_outcome,
)

HOOK_METRIC = "xagent.turn_settlement.hook"
USAGE = TurnUsage(
    input_tokens=3,
    output_tokens=2,
    llm_calls=1,
    tool_calls=0,
    tool_errors=0,
    by_model=(TurnModelUsage("m-1", "model", 3, 2),),
)


@pytest.fixture(autouse=True)
def _no_hook_left_behind():
    set_turn_settled_hook(None)
    yield
    set_turn_settled_hook(None)
    for task in list(turn_settlement._inflight):
        task.cancel()
    turn_settlement._inflight.clear()


def _settlement(**overrides: Any) -> TurnSettlement:
    now = datetime.now(timezone.utc)
    facts = TurnFacts(started_at=now)
    values: dict[str, Any] = {
        "task_id": 7,
        "run_id": "run-7",
        "turn_id": "turn-7",
        "task_source": None,
        "outcome": settled_outcome("completed"),
        "ended_at": now,
    }
    values.update(overrides)
    return build_turn_settlement(facts, **values)


def _outcomes(counter: Any) -> list[str]:
    return [
        call.kwargs["attributes"]["outcome"]
        for call in counter.call_args_list
        if call.args and call.args[0] == HOOK_METRIC
    ]


async def _settle_inflight() -> None:
    await asyncio.gather(*list(turn_settlement._inflight), return_exceptions=True)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        *[(code.value, code.value) for code in ClientErrorCode],
        *[(code, code) for code in sorted(CONNECTOR_RUNTIME_CLIENT_ERROR_CODES)],
        (ClientErrorCode.MODEL_ERROR, "model_error"),
        ("quota_exceeded", "quota_exceeded"),
        ("garbage", "task_execution_failed"),
        (None, None),
        (123, "task_execution_failed"),
    ],
)
def test_build_error_code_is_a_closed_vocabulary(raw: Any, expected: Any) -> None:
    code = build_error_code(raw)

    assert code == expected
    assert code is None or type(code) is str


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "status": "paused",
                "failure_code": "quota_exceeded",
                "visible_text": "x",
            },
            ("paused", None, False),
        ),
        (
            {"status": "failed", "visible_text": ""},
            ("failed", "task_execution_failed", True),
        ),
        (
            {"status": "failed", "failure_code": "model_error", "visible_text": "msg"},
            ("failed", "model_error", False),
        ),
        (
            {"status": "failed", "failure_code": "free text", "visible_text": ""},
            ("failed", "task_execution_failed", True),
        ),
        (
            {
                "status": "failed",
                "failure_code": "model_error",
                "visible_text": "",
                "ignored_result": True,
            },
            ("failed", "task_execution_failed", False),
        ),
        (
            {"status": "completed", "visible_text": "  "},
            ("completed", None, True),
        ),
        ({"status": "completed"}, ("completed", None, False)),
        (
            {
                "status": "completed",
                "failure_code": "quota_exceeded",
                "visible_text": "",
                "ignored_result": True,
            },
            ("completed", None, False),
        ),
        (
            {"status": "waiting_for_user", "visible_text": ""},
            ("waiting_for_user", None, False),
        ),
        (
            {
                "status": "completed",
                "visible_text": "No output provided",
                "output_missing": True,
            },
            ("completed", None, True),
        ),
    ],
)
def test_settled_outcome_decides_code_and_emptiness(
    kwargs: dict[str, Any], expected: tuple[str, str | None, bool]
) -> None:
    outcome = settled_outcome(**kwargs)

    assert (
        outcome.status,
        outcome.error_code,
        outcome.empty_visible_answer,
    ) == expected


def test_completion_outcome_is_only_reported_for_completed_runs() -> None:
    facts = TurnFacts(
        started_at=datetime.now(timezone.utc), completion_outcome="answered"
    )
    kwargs: dict[str, Any] = {
        "task_id": 1,
        "run_id": None,
        "turn_id": None,
        "task_source": None,
        "ended_at": datetime.now(timezone.utc),
    }

    completed = build_turn_settlement(
        facts, outcome=settled_outcome("completed"), **kwargs
    )
    failed = build_turn_settlement(facts, outcome=settled_outcome("failed"), **kwargs)

    assert completed.completion_outcome == "answered"
    assert failed.completion_outcome is None


def test_absorb_result_keeps_only_allowlisted_fields() -> None:
    facts = TurnFacts(started_at=datetime.now(timezone.utc))
    first = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    facts.absorb_result(
        {
            "finish_reason": "length",
            "empty_final_answer": True,
            "failure_code": "not-an-allowlisted-code",
            "first_answer_delta_at": first.isoformat(),
            "model_error": {"kind": "rate_limited", "message": "provider text"},
        }
    )

    assert facts.llm_finish_reason == "length"
    assert facts.empty_final_answer is True
    assert facts.tool_failure_code is None
    assert facts.first_answer_delta_at == first
    assert facts.model_error_kind == "rate_limited"

    facts.absorb_result(
        {
            "first_answer_delta_at": "not a time",
            "model_error": {"kind": "invented"},
        }
    )
    assert facts.first_answer_delta_at is None
    assert facts.model_error_kind is None


def test_absorb_result_reads_real_adapter_output() -> None:
    first = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    raw = PatternResult(
        success=False,
        error="x",
        metadata={
            "status": "invalid_tool_protocol",
            "empty_final_answer": True,
            "finish_reason": "length",
            "failure_code": "authentication_required",
        },
    ).to_dict()
    # ``AgentRunner`` records the first answer delta in the raw result's metadata.
    raw["metadata"] = {"first_answer_delta_at": first.isoformat()}
    normalized = AgentExecutionAdapter(
        AgentExecutionConfig(name="a", pattern="react", llm=None)
    )._normalize_result(result=raw, execution_type="agent_react", execution_id="1")
    facts = TurnFacts(started_at=datetime.now(timezone.utc))

    facts.absorb_result(normalized)

    assert (
        facts.llm_finish_reason,
        facts.empty_final_answer,
        facts.tool_failure_code,
        facts.first_answer_delta_at,
    ) == ("length", True, "authentication_required", first)
    assert "finish_reason" not in normalized["metadata"]
    assert "first_answer_delta_at" not in normalized["metadata"]


def test_absorb_result_records_a_missing_output() -> None:
    adapter = AgentExecutionAdapter(
        AgentExecutionConfig(name="a", pattern="react", llm=None)
    )
    facts = TurnFacts(started_at=datetime.now(timezone.utc))

    facts.absorb_result(
        adapter._normalize_result(
            result={"success": True, "output": ""},
            execution_type="agent_react",
            execution_id="1",
        )
    )
    assert facts.output_missing is True

    facts.absorb_result(
        adapter._normalize_result(
            result={"success": True, "output": "ok"},
            execution_type="agent_react",
            execution_id="1",
        )
    )
    assert facts.output_missing is False


async def test_hook_receives_the_settlement_without_the_caller_waiting() -> None:
    received: list[TurnSettlement] = []
    release = asyncio.Event()

    async def hook(settlement: TurnSettlement) -> None:
        await release.wait()
        received.append(settlement)

    set_turn_settled_hook(hook)
    settlement = _settlement()
    with patch.object(turn_settlement, "increment_counter") as counter:
        notify_turn_settled(settlement)
        assert received == []
        assert len(turn_settlement._inflight) == 1

        release.set()
        await _settle_inflight()

    assert received == [settlement]
    assert turn_settlement._inflight == set()
    assert _outcomes(counter) == ["ok"]


async def test_sync_hook_is_supported() -> None:
    received: list[TurnSettlement] = []
    set_turn_settled_hook(received.append)

    with patch.object(turn_settlement, "increment_counter") as counter:
        notify_turn_settled(_settlement())
        await _settle_inflight()

    assert len(received) == 1
    assert _outcomes(counter) == ["ok"]


async def test_no_hook_is_counted_without_a_warning(caplog) -> None:
    with (
        caplog.at_level(logging.WARNING),
        patch.object(turn_settlement, "increment_counter") as counter,
    ):
        notify_turn_settled(_settlement())

    assert _outcomes(counter) == ["no_hook"]
    assert turn_settlement._inflight == set()
    assert [r for r in caplog.records if r.name == turn_settlement.__name__] == []


async def test_failing_hook_is_logged_and_counted(caplog) -> None:
    def hook(_settlement: TurnSettlement) -> None:
        raise RuntimeError("hook broke")

    set_turn_settled_hook(hook)
    with (
        caplog.at_level(logging.WARNING),
        patch.object(turn_settlement, "increment_counter") as counter,
    ):
        notify_turn_settled(_settlement(task_id=11, run_id="run-11"))
        await _settle_inflight()

    assert _outcomes(counter) == ["failed"]
    warnings = [r for r in caplog.records if r.name == turn_settlement.__name__]
    assert len(warnings) == 1
    assert "11" in warnings[0].getMessage()
    assert "run-11" in warnings[0].getMessage()


async def test_hook_that_never_returns_times_out(caplog, monkeypatch) -> None:
    monkeypatch.setattr(turn_settlement, "HOOK_TIMEOUT_SECONDS", 0.01)

    async def hook(_settlement: TurnSettlement) -> None:
        await asyncio.Event().wait()

    set_turn_settled_hook(hook)
    with (
        caplog.at_level(logging.WARNING),
        patch.object(turn_settlement, "increment_counter") as counter,
    ):
        notify_turn_settled(_settlement())
        await _settle_inflight()

    assert _outcomes(counter) == ["timeout"]
    assert any("timed out" in r.getMessage() for r in caplog.records)


async def test_drain_waits_for_deliveries_in_flight() -> None:
    release = asyncio.Event()
    finished: list[int] = []

    async def hook(settlement: TurnSettlement) -> None:
        await release.wait()
        finished.append(settlement.task_id)

    set_turn_settled_hook(hook)
    notify_turn_settled(_settlement(task_id=1))
    notify_turn_settled(_settlement(task_id=2))
    asyncio.get_running_loop().call_later(0.01, release.set)

    await drain_turn_settlement_hooks(timeout=5.0)

    assert sorted(finished) == [1, 2]
    assert turn_settlement._inflight == set()


async def test_drain_cancels_deliveries_past_its_timeout(monkeypatch) -> None:
    # The hook's own timeout is far away, so only the drain can stop it.
    monkeypatch.setattr(turn_settlement, "HOOK_TIMEOUT_SECONDS", 1000.0)
    cancelled: list[bool] = []

    async def hook(_settlement: TurnSettlement) -> None:
        try:
            await asyncio.sleep(1000)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    set_turn_settled_hook(hook)
    notify_turn_settled(_settlement())
    loop = asyncio.get_running_loop()
    started = loop.time()

    await asyncio.wait_for(drain_turn_settlement_hooks(timeout=0.1), timeout=5)

    assert loop.time() - started < 1
    assert cancelled == [True]
    assert turn_settlement._inflight == set()


async def test_drain_without_deliveries_returns_at_once() -> None:
    await asyncio.wait_for(drain_turn_settlement_hooks(timeout=60), timeout=1)


async def _drive_channel(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: TaskStatus = TaskStatus.COMPLETED,
    content: str = "ok",
    result: dict[str, Any] | None = None,
    usage: TurnUsage | None = None,
    committed: bool = True,
    pause: bool = False,
    finalize_gate: tuple[threading.Event, threading.Event] | None = None,
) -> dict[str, Any]:
    """Run ``execute_channel_background`` with its collaborators faked.

    The result write is the entry's own ``finalize`` closure around a fake
    ``finalize_managed_task_lease_result_sync``, which fills the report as a
    committed interruption pause does when ``pause`` is set, and, given
    ``finalize_gate``, signals the first event and then blocks on the second.
    Returns the keyword arguments ``execute_task`` received.
    """
    seen: dict[str, Any] = {}

    async def execute_task(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        if usage is not None:
            kwargs["turn_usage_sink"](usage)
        return dict(result or {"success": True, "output": content})

    tracer = SimpleNamespace(add_handler=Mock(), remove_handler=Mock())
    service = SimpleNamespace(
        tracer=tracer,
        set_conversation_history=Mock(),
        set_execution_context_messages=Mock(),
        set_recovered_skill_context=Mock(),
    )
    manager = Mock()
    manager.get_agent_for_task = AsyncMock(return_value=service)
    manager.execute_task = execute_task

    async def materialize(_recovery: Any) -> dict[str, Any]:
        return {}

    class Forwarder:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        async def close(self, *, drain: bool = False) -> None:
            return None

    monkeypatch.setattr(agent_service_manager, "get_agent_manager", lambda: manager)
    monkeypatch.setattr(
        "xagent.web.services.task_execution_context_service."
        "materialize_task_execution_recovery_state",
        materialize,
    )
    monkeypatch.setattr(shared_channel_execution, "ChannelProgressForwarder", Forwarder)
    monkeypatch.setattr(
        shared_channel_execution, "get_task_event_bridge", lambda: MagicMock()
    )
    monkeypatch.setattr(
        "xagent.web.services.execution_result_projection."
        "project_execution_result_for_channel",
        lambda _result: SimpleNamespace(
            task_status=status,
            transcript_content=content,
            interactions=[],
            message_type="text",
            diagnostic_error=None,
            completion_outcome=("answered" if status == TaskStatus.COMPLETED else None),
        ),
    )
    monkeypatch.setattr(
        shared_channel_execution, "channel_durable_result", lambda *_a: {}
    )

    def fake_finalize(_lease: Any, **kwargs: Any) -> bool:
        if finalize_gate is not None:
            started, release = finalize_gate
            started.set()
            release.wait()
        if pause:
            kwargs["report"].paused_for = InterruptionReason.LLM_UNAVAILABLE
        return committed

    monkeypatch.setattr(
        "xagent.web.services.managed_task_lease."
        "finalize_managed_task_lease_result_sync",
        fake_finalize,
    )
    real_run_db_io = shared_channel_execution.run_db_io_cancellation_safe

    async def run_db_io(work: Any) -> Any:
        # Every other database step is skipped; a result write that went
        # through here would still run for real, in a worker thread.
        return await real_run_db_io(work) if work.__name__ == "finalize" else None

    monkeypatch.setattr(
        shared_channel_execution, "run_db_io_cancellation_safe", run_db_io
    )
    monkeypatch.setattr(
        "xagent.web.services.task_orchestrator.publish_settlement_pause",
        AsyncMock(),
    )
    snapshot = SimpleNamespace(
        runtime_user=None,
        conversation_history=(),
        conversation_watermark=None,
        conversation_event_watermark=None,
        execution_recovery=TaskExecutionRecoverySnapshot(),
        task=SimpleNamespace(source="internal", user_id=5),
    )
    await shared_channel_execution.execute_channel_background(
        command=SimpleNamespace(task_id=42, command_id="cmd-1", id=1),
        lease=TaskLease(
            task_id=42, runner_id="runner-a", run_id="run-a", attempt_id="attempt-a"
        ),
        heartbeat_task=None,  # type: ignore[arg-type]
        snapshot=snapshot,
        payload=SimpleNamespace(
            for_agent="hello", transcript_message="hello", attachments=[], file_ids=()
        ),  # type: ignore[arg-type]
        turn_facts=TurnFacts(started_at=datetime.now(timezone.utc)),
    )
    return seen


def _capture() -> list[TurnSettlement]:
    received: list[TurnSettlement] = []
    set_turn_settled_hook(received.append)
    return received


async def test_channel_run_reports_its_committed_settlement_once(monkeypatch) -> None:
    received = _capture()

    seen = await _drive_channel(monkeypatch, usage=USAGE)
    await _settle_inflight()

    assert callable(seen["turn_usage_sink"])
    assert len(received) == 1
    settlement = received[0]
    assert settlement.task_id == 42
    assert settlement.run_id == "run-a"
    assert settlement.turn_id == "cmd-1"
    assert settlement.task_source == "internal"
    assert settlement.status == "completed"
    assert settlement.completion_outcome == "answered"
    assert settlement.error_code is None
    assert settlement.empty_visible_answer is False
    assert settlement.usage == USAGE


async def test_each_channel_run_reports_its_own_settlement_id(monkeypatch) -> None:
    received = _capture()

    await _drive_channel(monkeypatch)
    await _drive_channel(monkeypatch)
    await _settle_inflight()

    ids = [s.settlement_id for s in received]
    assert len(ids) == 2
    assert all(len(i) == 32 for i in ids)
    assert ids[0] != ids[1]


async def test_channel_failed_result_reports_a_closed_error_code(monkeypatch) -> None:
    received = _capture()

    await _drive_channel(
        monkeypatch,
        status=TaskStatus.FAILED,
        content="",
        result={"success": False, "error": "boom", "error_code": "free text"},
    )
    await _settle_inflight()

    assert [(s.status, s.error_code, s.empty_visible_answer) for s in received] == [
        ("failed", "task_execution_failed", True)
    ]


async def test_channel_failed_result_without_a_code_is_a_task_failure(
    monkeypatch,
) -> None:
    received = _capture()

    await _drive_channel(
        monkeypatch,
        status=TaskStatus.FAILED,
        result={"success": False, "error": "boom"},
    )
    await _settle_inflight()

    assert [(s.status, s.error_code) for s in received] == [
        ("failed", "task_execution_failed")
    ]


async def test_channel_run_whose_settlement_was_fenced_out_reports_nothing(
    monkeypatch,
) -> None:
    received = _capture()

    with pytest.raises(TaskLeaseLostError):
        await _drive_channel(monkeypatch, committed=False)
    await _settle_inflight()

    assert received == []


async def test_failing_hook_leaves_the_channel_run_unchanged(monkeypatch) -> None:
    def hook(_settlement: TurnSettlement) -> None:
        raise RuntimeError("hook broke")

    set_turn_settled_hook(hook)
    with patch.object(turn_settlement, "increment_counter") as counter:
        await _drive_channel(monkeypatch)
        await _settle_inflight()

    assert _outcomes(counter) == ["failed"]


async def test_channel_turn_paused_by_its_settlement_reports_paused(
    monkeypatch,
) -> None:
    received = _capture()

    await _drive_channel(
        monkeypatch,
        status=TaskStatus.FAILED,
        result={
            "success": False,
            "error": "x",
            "interruption_reason": "llm_unavailable",
        },
        pause=True,
    )
    await _settle_inflight()

    assert [(s.status, s.error_code) for s in received] == [("paused", None)]
    publish = cast(AsyncMock, orchestrator.publish_settlement_pause)
    publish.assert_awaited_once()


async def test_channel_model_failure_reports_the_presented_code(monkeypatch) -> None:
    received = _capture()

    await _drive_channel(
        monkeypatch,
        status=TaskStatus.FAILED,
        result={
            "success": False,
            "error": "x",
            "model_error": {
                "kind": "rate_limited",
                "status_code": 429,
                "provider_code": None,
                "message": "p",
            },
        },
    )
    await _settle_inflight()

    assert [s.error_code for s in received] == ["model_error"]


async def test_channel_interrupted_run_is_not_an_empty_answer(monkeypatch) -> None:
    received = _capture()

    await _drive_channel(monkeypatch, status=TaskStatus.PAUSED, content="")
    await _settle_inflight()

    assert [(s.status, s.empty_visible_answer) for s in received] == [("paused", False)]


async def test_channel_run_without_output_reports_an_empty_answer(monkeypatch) -> None:
    received = _capture()

    await _drive_channel(
        monkeypatch,
        content="No output provided",
        result={
            "success": True,
            "output": "No output provided",
            "output_missing": True,
        },
    )
    await _settle_inflight()

    assert [(s.status, s.empty_visible_answer) for s in received] == [
        ("completed", True)
    ]


async def test_channel_turn_cancelled_while_its_result_commits_is_reported(
    monkeypatch,
) -> None:
    received = _capture()
    started, release = threading.Event(), threading.Event()
    drive = asyncio.create_task(
        _drive_channel(monkeypatch, finalize_gate=(started, release))
    )

    try:
        assert await asyncio.to_thread(started.wait, 5)
        drive.cancel()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await drive
    await _settle_inflight()

    assert [(s.status, s.turn_id) for s in received] == [("completed", "cmd-1")]


# ---------------------------------------------------------------------------
# Scheduled turns: ``_schedule_bg`` reports once its own settlement commits.
# Every cell runs against real task and lease rows; the result finalizer and
# ``settle_task_lease_isolated`` are the real ones.
# ---------------------------------------------------------------------------

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture


@pytest.fixture
def _clear_bg_manager():
    """``_schedule_bg`` registers its handle globally; do not leak it."""
    from xagent.web.services.task_execution import background_task_manager

    background_task_manager.running_tasks.clear()
    yield
    background_task_manager.running_tasks.clear()


async def _schedule_turn(
    canonical: Any,
    manager: Any,
    *,
    task_source: str = "sdk",
    channel_command: Any = None,
    extra_patches: tuple[Any, ...] = (),
    drive: Callable[[asyncio.Task[None]], Awaitable[None]] | None = None,
) -> tuple[TaskLease, TaskTurnPayload, type[BaseException] | None]:
    # ``drive``, when given, is awaited with the scheduled runner before the
    # runner itself is awaited, so a test can act on the run in flight.
    from xagent.web.services import task_execution

    factory, tid = canonical
    with factory() as db:
        db.get(Task, tid).source = task_source
        db.commit()
        lease = acquire_task_lease(db, tid, new_run=True)
    payload = TaskTurnPayload("2+2")

    async def publish(_event: Any, _task_id: Any) -> None:
        return None

    raised: type[BaseException] | None = None
    with ExitStack() as stack:
        for active in (
            patch.object(orchestrator, "run_task_lease_heartbeat", new=AsyncMock()),
            patch.object(
                orchestrator, "resolve_execution_scope", return_value=None, create=True
            ),
            patch.object(orchestrator, "_get_agent_manager", return_value=manager),
            patch("xagent.web.services.task_events.publish_task_event", new=publish),
            patch.object(task_execution, "publish_task_event", new=publish),
            patch(
                "xagent.web.services.task_event_display.publish_task_result",
                new=publish,
            ),
            patch.object(orchestrator, "publish_settlement_pause", new=AsyncMock()),
            *extra_patches,
        ):
            stack.enter_context(active)
        runner = _schedule_bg(
            task_id=tid,
            task_owner_user_id=1,
            task_source=task_source,
            task_lease=lease,
            payload=payload,
            force_fresh=False,
            context=None,
            channel_command=channel_command,
        )
        if drive is not None:
            await drive(runner)
        try:
            await runner
        except BaseException as error:  # noqa: BLE001 -- the outcome is the datum
            raised = type(error)
    await _settle_inflight()
    return lease, payload, raised


def _result_manager(outcome: dict[str, Any] | BaseException) -> Any:
    manager = MagicMock()
    manager.get_agent_for_task = AsyncMock(return_value=MagicMock())
    if isinstance(outcome, BaseException):
        manager.execute_task = AsyncMock(side_effect=outcome)
    else:
        manager.execute_task = AsyncMock(return_value=dict(outcome))
    return manager


def _held_settlement(
    entered: asyncio.Event, release: threading.Event, results: list[bool]
) -> Callable[..., bool]:
    """The real scheduler settlement, held in its worker thread until
    ``release`` is set. ``entered`` is set once the thread is holding; the
    real settlement's own return value is recorded and returned unchanged."""
    loop = asyncio.get_running_loop()
    settle = orchestrator.settle_task_lease_isolated

    def held(*args: Any, **kwargs: Any) -> bool:
        loop.call_soon_threadsafe(entered.set)
        if not release.wait(5):
            raise AssertionError("the held settlement was never released")
        settled = settle(*args, **kwargs)
        results.append(settled)
        return settled

    return held


SCHEDULED_CELLS = {
    "normal": ({"success": True, "output": "4"}, "completed", None),
    "waiting_for_user": (
        {"success": True, "status": "waiting_for_user", "output": "Which file?"},
        "waiting_for_user",
        None,
    ),
    "failed_result": (
        {"success": False, "output": "", "error": "model refused"},
        "failed",
        "task_execution_failed",
    ),
    "exception": (RuntimeError("agent failed"), "failed", "task_execution_failed"),
    "cancellation": (
        asyncio.CancelledError(),
        "failed",
        "task_execution_failed",
    ),
}


@pytest.mark.usefixtures("_clear_bg_manager")
@pytest.mark.parametrize("path", list(SCHEDULED_CELLS))
async def test_scheduled_turn_reports_its_settlement_once(canonical, path) -> None:
    outcome, status, error_code = SCHEDULED_CELLS[path]
    received = _capture()

    lease, payload, raised = await _schedule_turn(canonical, _result_manager(outcome))

    factory, tid = canonical
    with factory() as db:
        task = db.get(Task, tid)
        assert task.status.value == status
        assert task.runner_id is None
    assert raised is (asyncio.CancelledError if path == "cancellation" else None)
    assert len(received) == 1
    settlement = received[0]
    assert (settlement.task_id, settlement.run_id, settlement.turn_id) == (
        tid,
        lease.run_id,
        payload.turn_id,
    )
    assert settlement.task_source == "sdk"
    assert settlement.status == status
    assert settlement.error_code == error_code
    assert settlement.started_at <= settlement.ended_at
    assert len(settlement.settlement_id) == 32


@pytest.mark.usefixtures("_clear_bg_manager")
async def test_scheduled_turn_whose_result_was_discarded_reports_the_kept_status(
    canonical,
) -> None:
    """The task row turned FAILED while the run was in flight, so its
    finalizer discards the completed result and keeps the terminal status."""
    factory, tid = canonical
    received = _capture()

    async def execute_task(**_kwargs: Any) -> dict[str, Any]:
        with factory() as db:
            db.get(Task, tid).status = TaskStatus.FAILED
            db.commit()
        return {"success": True, "output": "No output provided", "output_missing": True}

    manager = MagicMock()
    manager.get_agent_for_task = AsyncMock(return_value=MagicMock())
    manager.execute_task = execute_task

    await _schedule_turn(canonical, manager)

    assert [(s.status, s.error_code, s.empty_visible_answer) for s in received] == [
        ("failed", "task_execution_failed", False)
    ]


@pytest.mark.usefixtures("_clear_bg_manager")
async def test_scheduled_turn_without_output_reports_the_adapter_classification(
    canonical,
) -> None:
    adapter = AgentExecutionAdapter(
        AgentExecutionConfig(name="a", pattern="react", llm=None)
    )
    normalized = adapter._normalize_result(
        result={"success": True, "output": "", "finish_reason": "length"},
        execution_type="agent_react",
        execution_id="1",
    )
    received = _capture()

    await _schedule_turn(canonical, _result_manager(normalized))

    assert [
        (s.status, s.empty_visible_answer, s.llm_finish_reason) for s in received
    ] == [("completed", True, "length")]


@pytest.mark.usefixtures("_clear_bg_manager")
async def test_cancelled_external_turn_reports_external_interruption(
    canonical,
) -> None:
    from xagent.web.services.task_orchestrator import EXTERNAL_TASK_SOURCE

    received = _capture()

    await _schedule_turn(
        canonical,
        _result_manager(asyncio.CancelledError()),
        task_source=EXTERNAL_TASK_SOURCE,
    )

    assert [(s.status, s.error_code) for s in received] == [
        ("failed", "external_turn_interrupted")
    ]


@pytest.mark.usefixtures("_clear_bg_manager")
async def test_settlement_ready_short_circuit_reports_nothing(canonical) -> None:
    from xagent.web.services.task_lease_service import TaskLeaseRefreshState

    received = _capture()
    manager = _result_manager({"success": True, "output": "4"})

    await _schedule_turn(
        canonical,
        manager,
        extra_patches=(
            patch.object(
                orchestrator,
                "validate_preacquired_task_lease_isolated",
                return_value=TaskLeaseRefreshState.SETTLEMENT_READY,
            ),
        ),
    )

    manager.execute_task.assert_not_awaited()
    assert received == []


@pytest.mark.usefixtures("_clear_bg_manager")
async def test_scheduled_turn_reports_real_tracker_usage(canonical) -> None:
    """The real manager's tracker meters the run and the sink carries it."""
    from xagent.core.model.chat.token_context import add_token_usage
    from xagent.web.services.agent_service_manager import AgentServiceManager

    service = MagicMock()
    service.workspace = None

    async def execute(**_kwargs: Any) -> dict[str, Any]:
        add_token_usage(
            input_tokens=12,
            output_tokens=4,
            model="model-a",
            model_id="7",
            call_type="chat",
        )
        return {"success": True, "output": "4"}

    service.execute_task = execute
    manager = AgentServiceManager()
    manager.get_agent_for_task = AsyncMock(return_value=service)  # type: ignore[method-assign]
    received = _capture()

    await _schedule_turn(canonical, manager)

    assert len(received) == 1
    usage = received[0].usage
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens, usage.llm_calls) == (12, 4, 1)
    assert usage.by_model == (TurnModelUsage("7", "model-a", 12, 4),)


def _report_like_the_channel_entry(kwargs: dict[str, Any]) -> None:
    """What the channel entry reports once its result committed."""
    facts = kwargs["turn_facts"]
    assert isinstance(facts, TurnFacts)
    facts.outcome = settled_outcome("completed")
    report_turn_settled(
        facts,
        task_id=kwargs["command"].task_id,
        run_id=kwargs["lease"].run_id,
        turn_id="cmd",
        task_source=None,
        outcome=facts.outcome,
        ended_at=datetime.now(timezone.utc),
    )


@pytest.mark.usefixtures("_clear_bg_manager")
@pytest.mark.parametrize("coordinated", [False, True], ids=["local", "coordinator"])
async def test_channel_turn_is_reported_once_by_its_finalizer(
    canonical, monkeypatch, coordinated
) -> None:
    """A normal channel run is reported by its own finalizer only, even when
    the scheduler's settlement afterwards also returns True (coordinator
    context, modelled by reporting True for the fenced-out settlement)."""
    received = _capture()

    async def channel_run(**kwargs: Any) -> None:
        # What the channel entry does on success: its finalizer commits the
        # result and releases the lease, then it reports the run.
        from xagent.web.services.managed_task_lease import (
            finalize_managed_task_lease_result_sync,
        )

        assert await asyncio.to_thread(
            finalize_managed_task_lease_result_sync,
            kwargs["lease"],
            status=TaskStatus.COMPLETED,
            assistant_content="ok",
            turn_id="cmd",
        )
        _report_like_the_channel_entry(kwargs)

    settle = orchestrator.settle_task_lease_isolated
    calls: list[bool] = []

    def settle_spy(*args: Any, **kwargs: Any) -> bool:
        settled = settle(*args, **kwargs)
        calls.append(settled)
        return True if coordinated else settled

    await _schedule_turn(
        canonical,
        _result_manager({"success": True}),
        channel_command=SimpleNamespace(task_id=canonical[1], command_id="cmd"),
        extra_patches=(
            patch(
                "xagent.web.services.shared_channel_execution."
                "execute_channel_background",
                new=channel_run,
            ),
            patch.object(orchestrator, "settle_task_lease_isolated", new=settle_spy),
        ),
    )

    # Locally the finalizer released the lease, so the scheduler's settlement
    # is fenced out; in coordinator context it still returns True.
    assert calls == [False]
    assert [s.turn_id for s in received] == ["cmd"]


@pytest.mark.usefixtures("_clear_bg_manager")
async def test_channel_turn_that_raises_is_reported_by_the_scheduler(
    canonical,
) -> None:
    received = _capture()

    async def channel_run(**_kwargs: Any) -> None:
        raise RuntimeError("channel run failed")

    _lease, payload, _raised = await _schedule_turn(
        canonical,
        _result_manager({"success": True}),
        channel_command=SimpleNamespace(task_id=canonical[1], command_id="cmd"),
        extra_patches=(
            patch(
                "xagent.web.services.shared_channel_execution."
                "execute_channel_background",
                new=channel_run,
            ),
        ),
    )

    factory, tid = canonical
    with factory() as db:
        assert db.get(Task, tid).status == TaskStatus.FAILED
    assert [(s.status, s.error_code, s.turn_id) for s in received] == [
        ("failed", "task_execution_failed", payload.turn_id)
    ]


# ---------------------------------------------------------------------------
# Resumed runs: ``execute_resume_background`` reports from its cleanup, after
# either the result finalizer or the cleanup's own settlement committed.
# ---------------------------------------------------------------------------


def _resume_lease() -> TaskLease:
    return TaskLease(
        task_id=42, runner_id="runner-a", run_id="run-a", attempt_id="attempt-a"
    )


async def _drive_resume(
    monkeypatch: pytest.MonkeyPatch,
    outcome: dict[str, Any] | BaseException,
    *,
    final_status: TaskStatus = TaskStatus.COMPLETED,
    delivery_turn_id: str | None = "turn-r",
    settled: bool = True,
    trusted_task_source: str | None = "sdk",
    extra_patches: tuple[Any, ...] = (),
    preacquired_prior_status: TaskStatus | None = None,
    finalize_side_effect: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> type[BaseException] | None:
    from xagent.web.services.task_execution import execute_resume_background

    heartbeat_stop = asyncio.Event()

    async def heartbeat() -> Any:
        await heartbeat_stop.wait()
        return None

    heartbeat_task = asyncio.create_task(heartbeat())
    service = MagicMock()
    service.resume_execution_by_id = AsyncMock(
        side_effect=outcome if isinstance(outcome, BaseException) else None,
        return_value=outcome if isinstance(outcome, dict) else None,
    )
    monkeypatch.setattr(agent_service_manager, "get_agent_manager", lambda: Mock())
    finalized = {
        "task_title": "t",
        "task_description": "x",
        "task_execution_mode": "flash",
        "task_agent_id": None,
        "agent_name": None,
        "agent_logo_url": None,
        "final_status": final_status.value,
        "lease_released": True,
        "report": SettlementReport(),
        "normalized_outputs": [],
        "output": "ok" if final_status == TaskStatus.COMPLETED else "",
        "late_result": False,
        "completion_outcome": (
            "answered" if final_status == TaskStatus.COMPLETED else None
        ),
        "error_code": None,
    }
    raised: type[BaseException] | None = None
    with ExitStack() as stack:
        for active in (
            patch(
                "xagent.web.services.task_execution._acquire_resume_task_lease",
                new=MagicMock(),
            ),
            patch(
                "xagent.web.services.task_execution.run_task_lease_heartbeat",
                new=MagicMock(),
            ),
            patch(
                "xagent.web.services.task_execution._finalize_resumed_task",
                side_effect=(
                    None
                    if finalize_side_effect is None
                    else lambda *_a, **_k: finalize_side_effect(finalized)
                ),
                return_value=finalized,
            ),
            patch(
                "xagent.web.services.task_execution._settle_resumed_task_lease",
                return_value=settled,
            ),
            patch(
                "xagent.web.services.task_execution.mark_user_message_delivery_sync",
                return_value=None,
            ),
            patch(
                "xagent.web.services.task_execution.load_routine_history_turn_id",
                # No routine provenance: the delivery's own turn id.
                side_effect=lambda _task_id, fallback, **_kwargs: fallback,
            ),
            patch(
                "xagent.web.api.websocket.manager",
                MagicMock(broadcast_to_task=AsyncMock()),
            ),
            patch(
                "xagent.web.services.task_execution.background_task_manager"
                ".promote_resume_task"
            ),
            *extra_patches,
        ):
            stack.enter_context(active)
        try:
            await execute_resume_background(
                task_id=42,
                agent_service=service,
                task_owner_user_id=1,
                delivery_turn_id=delivery_turn_id,
                delivery_already_dispatched=True,
                expected_run_id="run-a",
                resolved_execution_scope=None,
                preacquired_lease=_resume_lease(),
                preacquired_heartbeat_stop=heartbeat_stop,
                preacquired_heartbeat_task=heartbeat_task,
                trusted_task_source=trusted_task_source,
                preacquired_prior_status=preacquired_prior_status,
            )
        except BaseException as error:  # noqa: BLE001 -- the outcome is the datum
            raised = type(error)
        finally:
            heartbeat_stop.set()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
    await _settle_inflight()
    return raised


RESUME_OK = {"success": True, "output": "ok", "status": "completed"}
RESUME_CELLS = {
    "normal": (RESUME_OK, TaskStatus.COMPLETED, None, "completed", None),
    "waiting": (
        {"success": True, "output": "Which?", "status": "waiting_for_user"},
        TaskStatus.WAITING_FOR_USER,
        None,
        "waiting_for_user",
        None,
    ),
    "exception": (
        RuntimeError("agent failed"),
        TaskStatus.COMPLETED,
        None,
        "failed",
        "task_execution_failed",
    ),
    "cancellation": (
        asyncio.CancelledError(),
        TaskStatus.COMPLETED,
        asyncio.CancelledError,
        "failed",
        "task_execution_failed",
    ),
}


@pytest.mark.parametrize("path", list(RESUME_CELLS))
async def test_resumed_run_reports_its_settlement_once(monkeypatch, path) -> None:
    outcome, final_status, expected_raise, status, error_code = RESUME_CELLS[path]
    received = _capture()

    raised = await _drive_resume(monkeypatch, outcome, final_status=final_status)

    assert raised is expected_raise
    assert len(received) == 1
    settlement = received[0]
    assert (settlement.task_id, settlement.run_id, settlement.turn_id) == (
        42,
        "run-a",
        "turn-r",
    )
    assert settlement.task_source == "sdk"
    assert len(settlement.settlement_id) == 32
    assert settlement.status == status
    assert settlement.error_code == error_code
    assert settlement.completion_outcome == (
        "answered" if status == "completed" else None
    )


async def test_cancelled_external_resume_reports_external_interruption(
    monkeypatch,
) -> None:
    from xagent.web.services.external_task_cancel import (
        EXTERNAL_TURN_INTERRUPTED_MESSAGE,
    )
    from xagent.web.services.task_orchestrator import EXTERNAL_TASK_SOURCE

    received = _capture()
    settle_calls: list[dict[str, Any]] = []

    def record_settlement(*_args: Any, **kwargs: Any) -> bool:
        settle_calls.append(kwargs)
        return True

    raised = await _drive_resume(
        monkeypatch,
        asyncio.CancelledError(),
        trusted_task_source=EXTERNAL_TASK_SOURCE,
        extra_patches=(
            patch(
                "xagent.web.services.task_execution._settle_resumed_task_lease",
                new=record_settlement,
            ),
        ),
    )

    assert raised is asyncio.CancelledError
    assert [c["error_message"] for c in settle_calls] == [
        EXTERNAL_TURN_INTERRUPTED_MESSAGE
    ]
    assert [(s.status, s.error_code) for s in received] == [
        ("failed", "external_turn_interrupted")
    ]


async def test_resume_without_delivery_turn_takes_the_result_turn(
    monkeypatch,
) -> None:
    received = _capture()
    result = {
        **RESUME_OK,
        "agent_result": {
            "context": {
                "messages": [
                    {"role": "user", "metadata": {"turn_id": "turn-from-result"}},
                    {"role": "assistant", "metadata": {}},
                ]
            }
        },
    }

    await _drive_resume(monkeypatch, result, delivery_turn_id=None)

    assert [(s.status, s.turn_id) for s in received] == [
        ("completed", "turn-from-result")
    ]


async def test_resume_whose_cleanup_settlement_was_fenced_out_reports_nothing(
    monkeypatch,
) -> None:
    received = _capture()

    await _drive_resume(monkeypatch, RuntimeError("agent failed"), settled=False)

    assert received == []


@pytest.mark.usefixtures("_clear_bg_manager")
async def test_channel_turn_cancelled_after_its_settlement_is_reported_once(
    canonical,
) -> None:
    """The channel entry reported, then the run was cancelled and the
    scheduler's settlement still returned True (coordinator context): the
    shared facts bag keeps it to one report."""
    received = _capture()

    async def channel_run(**kwargs: Any) -> None:
        from xagent.web.services.managed_task_lease import (
            finalize_managed_task_lease_result_sync,
        )

        assert await asyncio.to_thread(
            finalize_managed_task_lease_result_sync,
            kwargs["lease"],
            status=TaskStatus.COMPLETED,
            assistant_content="ok",
            turn_id="cmd",
        )
        _report_like_the_channel_entry(kwargs)
        raise asyncio.CancelledError()

    settle = orchestrator.settle_task_lease_isolated

    def settle_as_coordinator(*args: Any, **kwargs: Any) -> bool:
        settle(*args, **kwargs)
        return True

    _lease, _payload, raised = await _schedule_turn(
        canonical,
        _result_manager({"success": True}),
        channel_command=SimpleNamespace(task_id=canonical[1], command_id="cmd"),
        extra_patches=(
            patch(
                "xagent.web.services.shared_channel_execution."
                "execute_channel_background",
                new=channel_run,
            ),
            patch.object(
                orchestrator, "settle_task_lease_isolated", new=settle_as_coordinator
            ),
        ),
    )

    assert raised is asyncio.CancelledError
    assert [(s.status, s.turn_id) for s in received] == [("completed", "cmd")]


def test_report_turn_settled_reports_a_run_once() -> None:
    facts = TurnFacts(started_at=datetime.now(timezone.utc))
    kwargs: dict[str, Any] = {
        "task_id": 1,
        "run_id": "r",
        "turn_id": "t",
        "task_source": None,
        "outcome": settled_outcome("completed"),
        "ended_at": datetime.now(timezone.utc),
    }

    with patch.object(turn_settlement, "notify_turn_settled") as notify:
        assert report_turn_settled(facts, **kwargs) is True
        assert report_turn_settled(facts, **kwargs) is False

    assert notify.call_count == 1
    assert facts.notified is True


class _SinkTracker:
    """Stands in for ``TaskTracker``: completion hands the sink a usage."""

    quota_interrupt_reason = None

    def __init__(self, **_kwargs: Any) -> None:
        self.turn_usage_sink: Any = None

    async def start_tracking(self) -> None:
        return None

    async def interrupt_reason_for_quota(self) -> None:
        return None

    async def complete_tracking(self) -> None:
        if self.turn_usage_sink is not None:
            self.turn_usage_sink(USAGE)

    async def stop_periodic_updates(self) -> None:
        return None


@pytest.mark.parametrize("path", ["normal", "exception"])
async def test_resumed_run_reports_usage_from_either_completion_point(
    monkeypatch, path
) -> None:
    received = _capture()
    outcome = RESUME_OK if path == "normal" else RuntimeError("agent failed")

    await _drive_resume(
        monkeypatch,
        outcome,
        extra_patches=(
            patch("xagent.web.tracking.task_tracker.TaskTracker", new=_SinkTracker),
        ),
    )

    assert len(received) == 1
    assert received[0].usage == USAGE


async def test_resume_paused_for_an_unaccepted_input_reports_paused(
    monkeypatch,
) -> None:
    from xagent.web.services.task_execution import (
        _DeferredInjectionRejectedRetryableError,
    )

    received = _capture()
    pause = AsyncMock(return_value=True)

    await _drive_resume(
        monkeypatch,
        _DeferredInjectionRejectedRetryableError("not accepted", fenced=True),
        extra_patches=(
            patch.object(orchestrator, "pause_unknown_task_lease", new=pause),
        ),
    )

    pause.assert_awaited_once()
    assert [(s.status, s.error_code) for s in received] == [("paused", None)]


async def test_resume_without_any_turn_id_is_counted_and_still_reported(
    monkeypatch,
) -> None:
    received = _capture()

    with patch.object(turn_settlement, "increment_counter") as counter:
        await _drive_resume(monkeypatch, RESUME_OK, delivery_turn_id=None)
        await _settle_inflight()

    assert [(s.status, s.turn_id) for s in received] == [("completed", None)]
    missing = [
        call
        for call in counter.call_args_list
        if call.args and call.args[0] == "xagent.turn_settlement.missing_turn_id"
    ]
    assert len(missing) == 1
    assert _outcomes(counter) == ["ok"]


@pytest.mark.usefixtures("_clear_bg_manager")
async def test_scheduled_turn_cancelled_after_its_result_committed_is_reported(
    canonical,
) -> None:
    """The result committed COMPLETED, then the run was cancelled during its
    completion broadcast. The scheduler's settlement carries that error for a
    task already terminal, so it releases the lease but returns False. The
    committed outcome is still reported."""
    received = _capture()

    async def cancelled_broadcast(_event: Any, _task_id: Any) -> None:
        raise asyncio.CancelledError()

    settle = orchestrator.settle_task_lease_isolated
    calls: list[bool] = []

    def settle_spy(*args: Any, **kwargs: Any) -> bool:
        settled = settle(*args, **kwargs)
        calls.append(settled)
        return settled

    lease, payload, raised = await _schedule_turn(
        canonical,
        _result_manager({"success": True, "output": "4"}),
        extra_patches=(
            patch(
                "xagent.web.services.task_event_display.publish_task_result",
                new=cancelled_broadcast,
            ),
            patch.object(orchestrator, "settle_task_lease_isolated", new=settle_spy),
        ),
    )

    factory, tid = canonical
    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == TaskStatus.COMPLETED
        assert task.runner_id is None
    assert raised is asyncio.CancelledError
    assert calls == [False]
    assert [(s.status, s.error_code, s.turn_id) for s in received] == [
        ("completed", None, payload.turn_id)
    ]


@pytest.mark.usefixtures("_clear_bg_manager")
async def test_channel_result_committed_but_unreported_is_reported_by_scheduler(
    canonical,
) -> None:
    """The channel finalizer committed and recorded its status, but its own
    report failed inside the entry (which only logs that failure) and the
    entry returned normally. The scheduler reports the run once."""
    received = _capture()

    async def channel_run(**kwargs: Any) -> None:
        from xagent.web.services.managed_task_lease import (
            finalize_managed_task_lease_result_sync,
        )

        assert await asyncio.to_thread(
            finalize_managed_task_lease_result_sync,
            kwargs["lease"],
            status=TaskStatus.COMPLETED,
            assistant_content="ok",
            turn_id="cmd",
        )
        # The entry recorded the committed status, then its report raised
        # before ``report_turn_settled`` and was only logged.
        kwargs["turn_facts"].outcome = settled_outcome("completed")

    _lease, payload, _raised = await _schedule_turn(
        canonical,
        _result_manager({"success": True}),
        channel_command=SimpleNamespace(task_id=canonical[1], command_id="cmd"),
        extra_patches=(
            patch(
                "xagent.web.services.shared_channel_execution."
                "execute_channel_background",
                new=channel_run,
            ),
        ),
    )

    assert [(s.status, s.turn_id) for s in received] == [("completed", payload.turn_id)]


@pytest.mark.usefixtures("_clear_bg_manager")
async def test_scheduled_turn_failing_before_execution_is_reported(canonical) -> None:
    received = _capture()
    manager = _result_manager({"success": True, "output": "4"})

    await _schedule_turn(
        canonical,
        manager,
        extra_patches=(
            patch.object(
                orchestrator, "load_task_setup_snapshot_sync", return_value=None
            ),
        ),
    )

    factory, tid = canonical
    with factory() as db:
        assert db.get(Task, tid).status == TaskStatus.FAILED
    manager.execute_task.assert_not_awaited()
    assert [(s.status, s.error_code) for s in received] == [
        ("failed", "task_execution_failed")
    ]


@pytest.mark.usefixtures("_clear_bg_manager")
async def test_scheduled_turn_with_a_dedicated_failure_code_reports_it(
    canonical,
) -> None:
    from xagent.web.services.llm_utils import AutoModelUnavailableError

    received = _capture()

    await _schedule_turn(
        canonical, _result_manager(AutoModelUnavailableError("no model"))
    )

    assert [(s.status, s.error_code) for s in received] == [
        ("failed", "auto_model_unavailable")
    ]


@pytest.mark.usefixtures("_clear_bg_manager")
async def test_scheduled_turn_paused_by_its_settlement_reports_paused(
    canonical, monkeypatch
) -> None:
    """A recoverable infrastructure failure settles the run PAUSED."""
    from tests.core.agent.test_react import FakeTool
    from tests.web.services.test_execution_event_persistence_settlement import (
        _fail_once,
        _react_execution,
        _run_scheduled_turn,
    )

    monkeypatch.setenv("XAGENT_TASK_INFRA_FAILURE_PAUSE_ENABLED", "true")
    factory, tid = canonical
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    _fail_once(monkeypatch, "tool_execution_start")
    received = _capture()

    await _run_scheduled_turn(tid, lease, _react_execution(tid, FakeTool()))
    await _settle_inflight()

    with factory() as db:
        assert db.get(Task, tid).status == TaskStatus.PAUSED
    assert [(s.status, s.error_code) for s in received] == [("paused", None)]


@pytest.mark.usefixtures("_clear_bg_manager")
async def test_failing_hook_leaves_the_committed_task_unchanged(canonical) -> None:
    def hook(_settlement: TurnSettlement) -> None:
        raise RuntimeError("hook broke")

    set_turn_settled_hook(hook)
    with patch.object(turn_settlement, "increment_counter") as counter:
        await _schedule_turn(
            canonical, _result_manager({"success": True, "output": "4"})
        )

    factory, tid = canonical
    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == TaskStatus.COMPLETED
        assert task.runner_id is None
    assert _outcomes(counter) == ["failed"]


async def test_resume_late_result_reports_nothing(monkeypatch) -> None:
    received = _capture()
    late = {"late_result": True, "lease_released": False}

    await _drive_resume(
        monkeypatch,
        RESUME_OK,
        extra_patches=(
            patch(
                "xagent.web.services.task_execution._finalize_resumed_task",
                return_value=late,
            ),
        ),
    )

    assert received == []


async def test_resume_cancelled_while_its_result_commits_is_reported(
    monkeypatch,
) -> None:
    received = _capture()
    loop = asyncio.get_running_loop()
    holder: list[asyncio.Task[Any]] = []

    def cancel_while_writing(finalized: dict[str, Any]) -> dict[str, Any]:
        # Runs in the worker thread: the cancellation reaches the awaiting
        # coroutine while its result write is still in flight.
        loop.call_soon_threadsafe(holder[0].cancel)
        processed = threading.Event()
        loop.call_soon_threadsafe(processed.set)
        processed.wait()
        return finalized

    drive = asyncio.create_task(
        _drive_resume(
            monkeypatch,
            RESUME_OK,
            settled=False,
            finalize_side_effect=cancel_while_writing,
        )
    )
    holder.append(drive)

    assert await drive is asyncio.CancelledError
    assert [(s.status, s.error_code) for s in received] == [("completed", None)]


async def test_resume_without_output_reports_the_adapter_classification(
    monkeypatch,
) -> None:
    received = _capture()

    await _drive_resume(
        monkeypatch,
        {
            **RESUME_OK,
            "output_missing": True,
            "finish_reason": "length",
        },
    )

    assert [
        (s.status, s.empty_visible_answer, s.llm_finish_reason) for s in received
    ] == [("completed", True, "length")]


async def test_resume_finding_the_task_terminal_reports_the_kept_status(
    monkeypatch,
) -> None:
    received = _capture()
    kept = {
        "late_result": True,
        "lease_released": False,
        "kept_terminal_status": "failed",
    }

    await _drive_resume(
        monkeypatch,
        RESUME_OK,
        extra_patches=(
            patch(
                "xagent.web.services.task_execution._finalize_resumed_task",
                return_value=kept,
            ),
        ),
    )

    assert [(s.status, s.error_code, s.empty_visible_answer) for s in received] == [
        ("failed", "task_execution_failed", False)
    ]


@pytest.mark.usefixtures("_clear_bg_manager")
async def test_scheduled_turn_cancelled_again_during_its_settlement_is_reported(
    canonical,
) -> None:
    """A cancel lands while the run executes and the scheduler settles the run
    FAILED. ``cancel_task`` stops waiting while that settlement is still being
    written and cancels the runner again. The settlement still commits, so the
    run is reported once, and the cancellation propagates after the report."""
    from xagent.web.services.task_execution import background_task_manager

    received = _capture()
    entered, release, settled = asyncio.Event(), threading.Event(), []
    run_started = asyncio.Event()

    async def run_until_cancelled(**_kwargs: Any) -> dict[str, Any]:
        run_started.set()
        await asyncio.Event().wait()
        return {}

    manager = _result_manager({})
    manager.execute_task = run_until_cancelled
    cancel_requested: list[bool] = []

    async def cancel_twice(runner: asyncio.Task[None]) -> None:
        await asyncio.wait_for(run_started.wait(), 5)
        cancel = asyncio.create_task(background_task_manager.cancel_task(canonical[1]))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert runner.cancelling() == 1
            # cancel_task's own 0.5 s wait expires while the settlement is
            # held, and it cancels the runner a second time.
            async with asyncio.timeout(5):
                while runner.cancelling() < 2:
                    await asyncio.sleep(0.01)
        finally:
            release.set()
        cancel_requested.append((await cancel).requested)

    _lease, payload, raised = await _schedule_turn(
        canonical,
        manager,
        extra_patches=(
            patch.object(
                orchestrator,
                "settle_task_lease_isolated",
                new=_held_settlement(entered, release, settled),
            ),
        ),
        drive=cancel_twice,
    )

    factory, tid = canonical
    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == TaskStatus.FAILED
        assert task.runner_id is None
    assert cancel_requested == [True]
    assert settled == [True]
    assert raised is asyncio.CancelledError
    assert [(s.status, s.error_code, s.turn_id) for s in received] == [
        ("failed", "task_execution_failed", payload.turn_id)
    ]


@pytest.mark.usefixtures("_clear_bg_manager")
@pytest.mark.parametrize(
    ("already_terminal", "settled_value", "reported"),
    [
        (False, True, [("failed", "task_execution_failed")]),
        (True, False, []),
    ],
    ids=["settlement_committed", "task_already_terminal"],
)
async def test_scheduled_turn_cancelled_during_its_settlement_reports_what_it_committed(
    canonical, already_terminal, settled_value, reported
) -> None:
    """The run failed, and one cancel lands while the scheduler writes its
    FAILED settlement. A settlement that committed is reported. One that found
    the task already terminal (another writer finished it) committed nothing
    for this run and is not reported. The cancel changes neither."""
    received = _capture()
    entered, release, settled = asyncio.Event(), threading.Event(), []
    factory, tid = canonical

    async def cancel_once(runner: asyncio.Task[None]) -> None:
        try:
            await asyncio.wait_for(entered.wait(), 5)
            if already_terminal:
                with factory() as db:
                    db.get(Task, tid).status = TaskStatus.COMPLETED
                    db.commit()
            runner.cancel()
        finally:
            release.set()

    _lease, _payload, raised = await _schedule_turn(
        canonical,
        _result_manager(RuntimeError("agent failed")),
        extra_patches=(
            patch.object(
                orchestrator,
                "settle_task_lease_isolated",
                new=_held_settlement(entered, release, settled),
            ),
        ),
        drive=cancel_once,
    )

    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == (
            TaskStatus.COMPLETED if already_terminal else TaskStatus.FAILED
        )
        assert task.runner_id is None
    assert settled == [settled_value]
    assert raised is asyncio.CancelledError
    assert [(s.status, s.error_code) for s in received] == reported


@pytest.mark.usefixtures("_clear_bg_manager")
async def test_scheduled_turn_whose_settlement_raises_while_cancelled_is_not_reported(
    canonical,
) -> None:
    """The run failed, one cancel lands while the scheduler writes its FAILED
    settlement, and that real settlement then fails to commit. Nothing was
    committed for the run, so nothing is reported, the cancellation still
    propagates, and the lease is left for the timeout reclaim."""
    received = _capture()
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    settle = orchestrator.settle_task_lease_isolated
    settlement_errors: list[BaseException] = []
    refused_threads: set[int] = set()
    factory, tid = canonical

    def refuse_commit(_session: Session) -> None:
        if threading.get_ident() in refused_threads:
            raise RuntimeError("commit refused")

    def held(*args: Any, **kwargs: Any) -> bool:
        loop.call_soon_threadsafe(entered.set)
        if not release.wait(5):
            raise AssertionError("the held settlement was never released")
        refused_threads.add(threading.get_ident())
        try:
            return settle(*args, **kwargs)
        except BaseException as error:
            settlement_errors.append(error)
            raise
        finally:
            refused_threads.discard(threading.get_ident())

    async def cancel_once(runner: asyncio.Task[None]) -> None:
        try:
            await asyncio.wait_for(entered.wait(), 5)
            runner.cancel()
        finally:
            release.set()

    event.listen(Session, "before_commit", refuse_commit)
    try:
        _lease, _payload, raised = await _schedule_turn(
            canonical,
            _result_manager(RuntimeError("agent failed")),
            extra_patches=(
                patch.object(orchestrator, "settle_task_lease_isolated", new=held),
            ),
            drive=cancel_once,
        )
    finally:
        event.remove(Session, "before_commit", refuse_commit)

    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == TaskStatus.RUNNING
        assert task.runner_id is not None
    assert [type(error) for error in settlement_errors] == [RuntimeError]
    assert raised is asyncio.CancelledError
    assert received == []


@pytest.mark.usefixtures("_clear_bg_manager")
@pytest.mark.parametrize(
    "competing_write", [False, True], ids=["alone", "row_rewritten_after_commit"]
)
async def test_resume_finalizer_keeps_a_terminal_row_and_says_so(
    canonical, competing_write
) -> None:
    """The resumed finalizer finds the task already FAILED, releases its lease
    and reports the status it kept. With ``competing_write`` another session
    rewrites the row to RUNNING right after the finalizer's commit (a later
    run taking the task over); the status reported is still the one read
    under the row lock."""
    from xagent.web.services.task_execution import (
        _finalize_resumed_task,
        _PreparedTaskFileOutputs,
    )

    factory, tid = canonical
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
        task = db.get(Task, tid)
        task.status = TaskStatus.FAILED
        db.commit()

    fired: list[bool] = []
    competitors: set[int] = set()

    def rewrite_after_commit(session: Session) -> None:
        if not competing_write or fired or id(session) in competitors:
            return
        fired.append(True)
        other = factory()
        competitors.add(id(other))
        try:
            other.execute(
                update(Task).where(Task.id == tid).values(status=TaskStatus.RUNNING)
            )
            other.commit()
        finally:
            other.close()

    event.listen(Session, "after_commit", rewrite_after_commit)
    try:
        finalized = await asyncio.to_thread(
            _finalize_resumed_task,
            tid,
            status="completed",
            success=True,
            output="ok",
            task_owner_user_id=None,
            result={"success": True, "output": "ok", "status": "completed"},
            task_lease=lease,
            prepared_outputs=_PreparedTaskFileOutputs((), (), ()),
        )
    finally:
        event.remove(Session, "after_commit", rewrite_after_commit)

    assert fired == ([True] if competing_write else [])
    assert finalized["late_result"] is True
    assert finalized["kept_terminal_status"] == "failed"
    with factory() as db:
        row = db.get(Task, tid)
        assert row.status == (
            TaskStatus.RUNNING if competing_write else TaskStatus.FAILED
        )
        assert row.runner_id is None


async def test_resume_restored_to_its_prior_status_reports_nothing(
    monkeypatch,
) -> None:
    """An unreadable checkpoint returns the task to the status it rested in
    before this resume; no run settled, so nothing is reported."""
    from xagent.core.agent.checkpoint import CheckpointUnavailableError

    received = _capture()
    restore = MagicMock(return_value=True)

    await _drive_resume(
        monkeypatch,
        CheckpointUnavailableError("checkpoint store unavailable"),
        preacquired_prior_status=TaskStatus.WAITING_FOR_USER,
        extra_patches=(
            patch(
                "xagent.web.services.task_execution."
                "_restore_resumed_task_lease_to_prior_status",
                new=restore,
            ),
        ),
    )

    restore.assert_called_once()
    assert received == []


@pytest.mark.usefixtures("_clear_bg_manager")
async def test_channel_report_failing_before_its_status_is_reported_nothing(
    canonical,
) -> None:
    """In coordinator context the channel finalizer committed, but the
    entry's report failed before recording any status and the entry returned
    normally. The scheduler's settlement returns True; with no committed
    status and no failure to report, nothing is reported rather than a
    completed run reported as failed."""
    received = _capture()

    async def channel_run(**kwargs: Any) -> None:
        from xagent.web.services.managed_task_lease import (
            finalize_managed_task_lease_result_sync,
        )

        assert await asyncio.to_thread(
            finalize_managed_task_lease_result_sync,
            kwargs["lease"],
            status=TaskStatus.COMPLETED,
            assistant_content="ok",
            turn_id="cmd",
        )

    settle = orchestrator.settle_task_lease_isolated

    def settle_as_coordinator(*args: Any, **kwargs: Any) -> bool:
        settle(*args, **kwargs)
        return True

    await _schedule_turn(
        canonical,
        _result_manager({"success": True}),
        channel_command=SimpleNamespace(task_id=canonical[1], command_id="cmd"),
        extra_patches=(
            patch(
                "xagent.web.services.shared_channel_execution."
                "execute_channel_background",
                new=channel_run,
            ),
            patch.object(
                orchestrator, "settle_task_lease_isolated", new=settle_as_coordinator
            ),
        ),
    )

    factory, tid = canonical
    with factory() as db:
        assert db.get(Task, tid).status == TaskStatus.COMPLETED
    assert received == []


@pytest.mark.usefixtures("_clear_bg_manager")
async def test_committed_run_whose_settlement_is_deferred_is_reported(
    canonical,
) -> None:
    """The result committed COMPLETED; the heartbeat then reports the lease
    needs recovery, so the scheduler leaves the lease for lease recovery,
    which only handles RUNNING rows. The committed status is reported."""
    from xagent.web.services.task_lease_service import TaskLeaseHeartbeatOutcome

    received = _capture()
    settle = MagicMock()

    _lease, payload, _raised = await _schedule_turn(
        canonical,
        _result_manager({"success": True, "output": "4"}),
        extra_patches=(
            patch.object(
                orchestrator,
                "stop_task_lease_heartbeat",
                new=AsyncMock(return_value=TaskLeaseHeartbeatOutcome(lease_lost=True)),
            ),
            patch.object(orchestrator, "settle_task_lease_isolated", new=settle),
        ),
    )

    factory, tid = canonical
    with factory() as db:
        assert db.get(Task, tid).status == TaskStatus.COMPLETED
    settle.assert_not_called()
    assert [(s.status, s.turn_id) for s in received] == [("completed", payload.turn_id)]
