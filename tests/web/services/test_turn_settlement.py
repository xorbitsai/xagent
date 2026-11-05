"""The per-run settlement notification handed to the hosting application.

Delivery is detached from the entry point that reports the run: a hook that
fails or never returns is logged and counted, and never changes how the run
ended.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from xagent.web.models.task import TaskStatus
from xagent.web.services import (
    agent_service_manager,
    shared_channel_execution,
    turn_settlement,
)
from xagent.web.services.client_error_messages import ClientErrorCode
from xagent.web.services.task_execution_context_service import (
    TaskExecutionRecoverySnapshot,
)
from xagent.web.services.task_lease_service import TaskLease, TaskLeaseLostError
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
