"""Each turn entry point tells the runtime cache when its turn has settled.

``note_runtime_settled`` is the single call the three execution entry points
make from their final cleanup. It must run exactly once per turn on every exit
(normal, failing, cancelled, or failing before the agent ran), and a failure of
the call itself must never change how the turn ended.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from tests.web.api.test_execute_task_background_snapshot_passthrough import (
    _build_fake_agent_service,
    _common_patches,
    _make_snapshot,
    _Patches,
)
from xagent.web.models.task import TaskStatus
from xagent.web.services import agent_service_manager, shared_channel_execution
from xagent.web.services.agent_service_manager import AgentServiceManager
from xagent.web.services.task_execution import (
    execute_resume_background,
    execute_task_background,
)
from xagent.web.services.task_execution_context_service import (
    TaskExecutionRecoverySnapshot,
)
from xagent.web.services.task_lease_service import TaskLease
from xagent.web.services.task_settlement_report import SettlementReport

HOOK_WARNING = "Settling the cached runtime for task"
OK_RESULT = {"success": True, "output": "ok", "status": "completed"}

# A turn either finishes, fails inside the agent, or is cancelled.
PATHS = ["normal", "exception", "cancellation"]
ENTRIES = [
    "execute_task_background",
    "execute_channel_background",
    "execute_resume_background",
]


def agent_outcome(path: str) -> dict[str, Any] | BaseException:
    return {
        "normal": OK_RESULT,
        "exception": RuntimeError("agent failed"),
        "cancellation": asyncio.CancelledError(),
    }[path]


def expected_outcome(entry: str, path: str) -> type[BaseException] | None:
    """How each entry point ends for the agent outcome (hook working)."""
    if path == "cancellation":
        return asyncio.CancelledError
    if path == "exception" and entry != "execute_resume_background":
        # The resume entry records the failure on the task instead of raising.
        return RuntimeError
    return None


def lease() -> TaskLease:
    return TaskLease(
        task_id=42, runner_id="runner-a", run_id="run-a", attempt_id="attempt-a"
    )


async def drive_task_background(manager: Any, path: str) -> None:
    outcome = agent_outcome(path)
    service = _build_fake_agent_service()
    manager.get_agent_for_task = AsyncMock(return_value=service)
    manager.execute_task = AsyncMock(
        side_effect=outcome if isinstance(outcome, BaseException) else None,
        return_value=outcome if isinstance(outcome, dict) else None,
    )
    with _Patches(_common_patches(MagicMock(), service)):
        await execute_task_background(
            task_id=42,
            user_message="hi",
            context={},
            agent_manager=manager,
            task_owner_user_id=1,
            task_setup_snapshot=_make_snapshot(),
            task_lease=lease(),
            resolved_execution_scope=None,
        )


async def drive_channel_background(
    manager: Any,
    path: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_before_execute: str | None = None,
) -> None:
    outcome = agent_outcome(path)
    tracer = SimpleNamespace(add_handler=Mock(), remove_handler=Mock())

    def stage(name: str) -> Mock:
        return Mock(
            side_effect=RuntimeError(f"{name} failed")
            if fail_before_execute == name
            else None
        )

    service = SimpleNamespace(
        tracer=tracer,
        set_conversation_history=stage("set_conversation_history"),
        set_execution_context_messages=stage("set_execution_context_messages"),
        set_recovered_skill_context=stage("set_recovered_skill_context"),
    )
    manager.get_agent_for_task = AsyncMock(return_value=service)
    manager.execute_task = AsyncMock(
        side_effect=(
            RuntimeError("execute_task failed")
            if fail_before_execute == "execute_task"
            else outcome
            if isinstance(outcome, BaseException)
            else None
        ),
        return_value=outcome if isinstance(outcome, dict) else None,
    )

    async def materialize(_recovery: Any) -> dict[str, Any]:
        if fail_before_execute == "materialize":
            raise RuntimeError("materialize failed")
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
            task_status=TaskStatus.COMPLETED,
            transcript_content="ok",
            interactions=[],
            message_type="text",
            diagnostic_error=None,
        ),
    )
    monkeypatch.setattr(
        shared_channel_execution, "channel_durable_result", lambda *_a: {}
    )
    monkeypatch.setattr(
        shared_channel_execution,
        "run_db_io_cancellation_safe",
        AsyncMock(return_value=True),
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
        lease=lease(),
        heartbeat_task=None,
        snapshot=snapshot,
        payload=SimpleNamespace(
            for_agent="hello", transcript_message="hello", attachments=[], file_ids=()
        ),
    )


async def drive_resume_background(
    manager: Any, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome = agent_outcome(path)
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
    monkeypatch.setattr(agent_service_manager, "get_agent_manager", lambda: manager)
    finalized = {
        "task_title": "t",
        "task_description": "x",
        "task_execution_mode": "flash",
        "task_agent_id": None,
        "agent_name": None,
        "agent_logo_url": None,
        "final_status": TaskStatus.COMPLETED.value,
        "lease_released": True,
        "report": SettlementReport(),
        "normalized_outputs": [],
        "output": "ok",
        "late_result": False,
    }
    with _Patches(
        [
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
                return_value=True,
            ),
            patch(
                "xagent.web.services.task_execution.load_routine_history_turn_id",
                return_value=None,
            ),
            patch(
                "xagent.web.api.websocket.manager",
                MagicMock(broadcast_to_task=AsyncMock()),
            ),
            patch(
                "xagent.web.services.task_execution.background_task_manager"
                ".promote_resume_task"
            ),
        ]
    ):
        try:
            await execute_resume_background(
                task_id=42,
                agent_service=service,
                task_owner_user_id=1,
                expected_run_id="run-a",
                resolved_execution_scope=None,
                preacquired_lease=lease(),
                preacquired_heartbeat_stop=heartbeat_stop,
                preacquired_heartbeat_task=heartbeat_task,
            )
        finally:
            heartbeat_stop.set()
            await asyncio.gather(heartbeat_task, return_exceptions=True)


def make_driver(
    entry: str, monkeypatch: pytest.MonkeyPatch
) -> Callable[[Any, str], Awaitable[None]]:
    if entry == "execute_task_background":
        return drive_task_background
    if entry == "execute_channel_background":
        return lambda manager, path: drive_channel_background(
            manager, path, monkeypatch
        )
    return lambda manager, path: drive_resume_background(manager, path, monkeypatch)


async def run_and_capture(
    driver: Callable[[Any, str], Awaitable[None]], manager: Any, path: str
) -> type[BaseException] | None:
    try:
        await driver(manager, path)
    except BaseException as error:  # noqa: BLE001 -- the outcome is the datum
        return type(error)
    return None


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("path", PATHS)
async def test_settled_hook_is_best_effort(entry, path, monkeypatch, caplog) -> None:
    driver = make_driver(entry, monkeypatch)

    working = Mock()
    baseline = await run_and_capture(driver, working, path)

    failing = Mock(note_runtime_settled=Mock(side_effect=RuntimeError("hook broke")))
    with caplog.at_level(logging.WARNING):
        caplog.clear()
        outcome = await run_and_capture(driver, failing, path)

    # The hook failing changes nothing about how the turn ended.
    assert outcome is baseline
    assert baseline is expected_outcome(entry, path)
    failing.note_runtime_settled.assert_called_once_with(42)
    warnings = [r for r in caplog.records if HOOK_WARNING in r.getMessage()]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("path", PATHS)
async def test_settled_hook_called_once_from_each_entrypoint(
    entry, path, monkeypatch
) -> None:
    manager = Mock()

    await run_and_capture(make_driver(entry, monkeypatch), manager, path)

    manager.note_runtime_settled.assert_called_once_with(42)


@pytest.mark.parametrize(
    "failing_stage",
    [
        "set_conversation_history",
        "materialize",
        "set_execution_context_messages",
        "set_recovered_skill_context",
        "execute_task",
    ],
)
async def test_channel_hook_runs_when_a_stage_before_the_result_fails(
    failing_stage, monkeypatch
) -> None:
    manager = Mock()

    with pytest.raises(RuntimeError, match=failing_stage):
        await drive_channel_background(
            manager, "normal", monkeypatch, fail_before_execute=failing_stage
        )

    manager.note_runtime_settled.assert_called_once_with(42)


async def test_many_settled_turns_stay_bounded() -> None:
    now = [1000.0]
    manager = AgentServiceManager()
    manager._monotonic = lambda: now[0]
    manager.execute_task = AsyncMock(return_value=dict(OK_RESULT))

    async def get_agent_for_task(task_id: int, *_args: Any, **_kwargs: Any) -> Any:
        service = _build_fake_agent_service()
        service.get_execution_status = MagicMock(return_value=None)
        manager._agents[task_id] = service
        manager._agent_build_locks[task_id] = asyncio.Lock()
        manager._agent_run_generations[task_id] = task_id
        manager._agent_last_active_at[task_id] = manager._monotonic()
        return service

    manager.get_agent_for_task = get_agent_for_task  # type: ignore[method-assign]
    floor = agent_service_manager._IDLE_RUNTIME_SOFT_CAP_MIN_AGE_SECONDS
    max_age = agent_service_manager._IDLE_RUNTIME_MAX_AGE_SECONDS
    cap = agent_service_manager._IDLE_RUNTIME_SOFT_CAP
    peak_unexpired = 0

    with _Patches(_common_patches(MagicMock(), None)):
        for task_id in range(1, 501):
            await execute_task_background(
                task_id=task_id,
                user_message="hi",
                context={},
                agent_manager=manager,
                task_owner_user_id=1,
                task_setup_snapshot=_make_snapshot(),
                task_lease=lease(),
                resolved_execution_scope=None,
            )
            idle = [now[0] - manager._agent_last_active_at[t] for t in manager._agents]
            # Only runtimes past the soft cap's age floor are bounded by it.
            assert sum(1 for i in idle if floor <= i < max_age) <= cap
            peak_unexpired = max(peak_unexpired, sum(1 for i in idle if i < max_age))
            now[0] += 0.5

    # The age rule alone would keep every runtime younger than ``max_age``;
    # holding more than the cap proves the cap is what bounds the older ones.
    assert peak_unexpired > cap
    now[0] += 120
    manager._sweep_idle_runtimes(reason="test")
    assert manager._agents == {}
    assert manager._agent_build_locks == {}
