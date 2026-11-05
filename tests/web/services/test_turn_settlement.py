"""The per-run settlement notification handed to the hosting application.

Delivery is detached from the entry point that reports the run: a hook that
fails or never returns is logged and counted, and never changes how the run
ended.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import ExitStack
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_execution_event_writer import engine as engine_fixture
from tests.web.services.test_task_execution_event_writer import (
    task_id as task_id_fixture,
)
from xagent.web.models.task import Task, TaskStatus
from xagent.web.services import (
    agent_service_manager,
    shared_channel_execution,
)
from xagent.web.services import task_orchestrator as orchestrator
from xagent.web.services import (
    turn_settlement,
)
from xagent.web.services.client_error_messages import ClientErrorCode
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
    set_turn_settled_hook,
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
        "status": "completed",
        "error_code": None,
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


def test_completion_outcome_is_only_reported_for_completed_runs() -> None:
    facts = TurnFacts(
        started_at=datetime.now(timezone.utc), completion_outcome="answered"
    )
    kwargs: dict[str, Any] = {
        "task_id": 1,
        "run_id": None,
        "turn_id": None,
        "task_source": None,
        "error_code": None,
        "ended_at": datetime.now(timezone.utc),
    }

    completed = build_turn_settlement(facts, status="completed", **kwargs)
    failed = build_turn_settlement(facts, status="failed", **kwargs)

    assert completed.completion_outcome == "answered"
    assert failed.completion_outcome is None


def test_absorb_result_keeps_only_allowlisted_fields() -> None:
    facts = TurnFacts(started_at=datetime.now(timezone.utc))
    first = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    facts.absorb_result(
        {
            "metadata": {
                "finish_reason": "length",
                "empty_final_answer": True,
                "failure_code": "not-an-allowlisted-code",
                "first_answer_delta_at": first.isoformat(),
            },
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
            "metadata": {"first_answer_delta_at": "not a time"},
            "model_error": {"kind": "invented"},
        }
    )
    assert facts.first_answer_delta_at is None
    assert facts.model_error_kind is None


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


async def test_drain_cancels_deliveries_past_its_timeout() -> None:
    cancelled: list[bool] = []

    async def hook(_settlement: TurnSettlement) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    set_turn_settled_hook(hook)
    notify_turn_settled(_settlement())

    await drain_turn_settlement_hooks(timeout=0.01)

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
) -> dict[str, Any]:
    """Run ``execute_channel_background`` with its collaborators faked.

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
    monkeypatch.setattr(
        shared_channel_execution,
        "run_db_io_cancellation_safe",
        AsyncMock(return_value=committed),
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
) -> tuple[TaskLease, TaskTurnPayload, type[BaseException] | None]:
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
        try:
            await _schedule_bg(
                task_id=tid,
                task_owner_user_id=1,
                task_source=task_source,
                task_lease=lease,
                payload=payload,
                force_fresh=False,
                context=None,
                channel_command=channel_command,
            )
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
        notify_turn_settled(
            _settlement(task_id=kwargs["command"].task_id, turn_id="cmd")
        )

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
    assert settlement.status == status
    assert settlement.error_code == error_code
    assert settlement.completion_outcome == (
        "answered" if status == "completed" else None
    )


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
