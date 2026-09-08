"""Regression test: ``get_agent_for_task`` skips reconstruct for fresh
RUNNING tasks but still runs it when prior state actually exists.

Background:
    ``begin_turn`` atomically flips a newly created SDK task's status
    to ``RUNNING`` before ``get_agent_for_task`` is called. A naive
    ``should_reconstruct = status in {RUNNING, PAUSED,
    WAITING_FOR_USER}`` test would route every brand-new SDK task
    into ``_reconstruct_agent_from_history``. The worker-owned task
    snapshot finds no ``TraceEvent`` or ``DAGExecution`` state, so
    reconstruction would only log a misleading warning and fall
    through to normal creation.

    ``TaskSetupSnapshot.has_reconstructable_history`` gates the
    reconstruct branch for ``RUNNING`` tasks. The same detached
    snapshot is then reused for reconstruction or normal creation;
    no history probe is allowed on the asyncio event loop.

    ``PAUSED`` / ``WAITING_FOR_USER`` tasks are NOT gated on the
    pre-check -- those states by definition have prior runtime state
    that must be recovered.

What this test pins:

    * RUNNING + empty history => reconstruct NOT called.
    * RUNNING + trace events present => reconstruct called (regression
      guard against accidentally widening the skip condition).
    * PAUSED => reconstruct called even with empty history (the pre-
      check intentionally doesn't gate non-RUNNING active statuses).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.core.tools.adapters.vibe.config import (
    MCPUnavailableSummary,
    RequiredMCPUnavailableError,
)
from xagent.web.api.chat import AgentServiceManager
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.task import DAGExecution, Task, TaskStatus, TraceEvent
from xagent.web.models.user import User
from xagent.web.services.llm_utils import AutoModelUnavailableError
from xagent.web.services.task_setup_snapshot import (
    RuntimeUserFields,
    TaskOwnerMismatchError,
    TaskReconstructionSnapshot,
    TaskSetupSnapshot,
    _TaskFields,
)


def _make_user() -> User:
    return User(
        id=1,
        username="reconstruct_test_user",
        password_hash="hash",
        is_admin=False,
    )


def _make_task(status: TaskStatus, agent_id: int | None = None) -> Task:
    return Task(
        id=42,
        user_id=1,
        title="reconstruct test",
        description="reconstruct",
        status=status,
        agent_id=agent_id,
        agent_type="standard",
    )


def _make_agent() -> Agent:
    return Agent(
        id=7,
        user_id=1,
        name="reconstruct test agent",
        instructions="be terse",
        status=AgentStatus.PUBLISHED,
        tool_categories=["basic"],
        knowledge_bases=[],
        skills=[],
        execution_mode="flash",
    )


def _build_snapshot(
    task: Task,
    user: User,
    *,
    trace_event: Any | None = None,
    dag_execution: Any | None = None,
) -> TaskSetupSnapshot:
    """Build the detached state produced by the worker-owned snapshot loader."""
    has_history = trace_event is not None or dag_execution is not None
    reconstruction = TaskReconstructionSnapshot(
        tracer_events=(
            (
                {
                    "id": "event-1",
                    "event_type": "agent_step",
                    "task_id": str(task.id),
                    "step_id": None,
                    "timestamp": None,
                    "data": {},
                    "parent_id": None,
                },
            )
            if trace_event is not None
            else ()
        ),
        plan_state={"steps": []} if dag_execution is not None else None,
        has_history=has_history,
    )
    return TaskSetupSnapshot(
        task=_TaskFields(
            id=int(task.id),
            user_id=int(task.user_id),
            status=task.status,
            source=task.source,
            agent_id=task.agent_id,
            agent_config=task.agent_config,
            model_name=task.model_name,
            compact_model_name=task.compact_model_name,
            execution_mode=task.execution_mode,
            agent_type=task.agent_type,
        ),
        runtime_user=RuntimeUserFields(
            id=int(user.id),
            is_admin=bool(user.is_admin),
        ),
        has_reconstructable_history=has_history,
        task_pattern="single_call",
        task_llm=MagicMock(),
        task_fast_llm=None,
        task_vision_llm=None,
        task_compact_llm=None,
        agent=None,
        agent_config=None,
        excluded_agent_id=None,
        reconstruction=reconstruction,
    )


class _Fake:
    """Sentinel used to select trace- or DAG-backed snapshot history."""


def _build_db(
    task: Task,
    *,
    trace_event: Any | None = None,
    dag_execution: Any | None = None,
    agent: Agent | None = None,
    user: User | None = None,
) -> MagicMock:
    """Wire a MagicMock ``db`` whose ``.query(model).filter(...).first()``
    returns whichever fixture row the test supplied.

    All other models default to ``None`` from ``.first()`` and ``[]``
    from ``.all()``, which is what an in-memory SQLAlchemy session
    would do for an empty table.
    """
    by_model: dict[type, Any] = {
        Task: task,
        TraceEvent: trace_event,
        DAGExecution: dag_execution,
        Agent: agent,
        User: user,
    }

    def _query(model: type) -> Any:
        result = MagicMock()
        result.filter = MagicMock(return_value=result)
        result.first = MagicMock(return_value=by_model.get(model))
        result.all = MagicMock(
            return_value=[by_model.get(model)] if by_model.get(model) else []
        )
        result.order_by = MagicMock(return_value=result)
        return result

    db = MagicMock()
    db.query = _query
    return db


def _stub_downstream(manager: AgentServiceManager):
    """Patch the heavy work past the reconstruct decision so the test
    asserts only on whether reconstruct was called.

    LLM resolution + agent-builder config loading moved to module-
    level helpers in ``llm_utils``; patches target those source
    locations so the lazy imports inside the snapshot loader and
    ``_resolve_task_runtime_config`` pick them up.
    """
    return [
        patch(
            "xagent.web.services.llm_utils.UserAwareModelStorage."
            "resolve_llms_from_names",
            return_value=(None, None, None, None),
        ),
        patch(
            "xagent.web.services.llm_utils.make_normalize_model_id",
            return_value=lambda mid, mname: mname,
        ),
        patch(
            "xagent.web.services.llm_utils.load_agent_builder_config",
            return_value={
                "llms": (None, None, None, None),
                "saved_model_ids": {},
                "saved_model_descriptors": {},
                "execution_mode": "flash",
                "instructions": "",
                "knowledge_bases": [],
                "skills": [],
                "tool_categories": ["basic"],
            },
        ),
        patch.object(manager, "_load_persisted_conversation_history"),
        patch(
            "xagent.web.api.chat.create_task_tracer",
            return_value=MagicMock(),
        ),
        patch(
            "xagent.web.api.chat.create_default_tools",
            new=AsyncMock(return_value=([], MagicMock())),
        ),
        patch(
            "xagent.web.sandbox_manager.get_sandbox_manager",
            return_value=None,
        ),
        patch("xagent.web.api.chat.AgentService"),
    ]


@pytest.mark.asyncio
async def test_running_with_no_history_skips_reconstruct() -> None:
    """The pre-check hot case: brand-new SDK task is RUNNING but has zero
    prior state, so reconstruct must be skipped.
    """
    manager = AgentServiceManager()
    user = _make_user()
    task = _make_task(TaskStatus.RUNNING, agent_id=7)
    agent_row = _make_agent()

    db = _build_db(
        task,
        trace_event=None,  # no prior trace events
        dag_execution=None,  # no DAG plan
        agent=agent_row,
        user=user,
    )
    snapshot = _build_snapshot(task, user)

    reconstruct = AsyncMock()
    with (
        patch.object(manager, "_reconstruct_agent_from_history", reconstruct),
        patch(
            "xagent.web.api.chat.load_task_setup_snapshot_sync",
            return_value=snapshot,
        ) as snapshot_loader,
        patch.object(
            manager,
            "_has_reconstructable_history",
            side_effect=AssertionError("history pre-check ran on the event loop"),
            create=True,
        ),
    ):
        with _Patches(_stub_downstream(manager)):
            try:
                await manager.get_agent_for_task(
                    task_id=42,
                    db=db,
                    user=user,
                )
            except Exception:
                # Downstream stubs may raise during agent assembly; the
                # reconstruct-call assertion below records its state
                # before the failure point.
                pass

    reconstruct.assert_not_awaited()
    snapshot_loader.assert_called_once_with(42, None)


@pytest.mark.asyncio
async def test_running_with_prior_trace_event_runs_reconstruct() -> None:
    """Regression guard: if a RUNNING task has prior trace events,
    reconstruct must still run (the pre-check must not widen the skip
    too aggressively).
    """
    manager = AgentServiceManager()
    user = _make_user()
    task = _make_task(TaskStatus.RUNNING, agent_id=7)
    agent_row = _make_agent()

    db = _build_db(
        task,
        trace_event=_Fake(),  # prior trace event exists
        dag_execution=None,
        agent=agent_row,
        user=user,
    )
    snapshot = _build_snapshot(task, user, trace_event=_Fake())

    reconstruct = AsyncMock()
    with (
        patch.object(manager, "_reconstruct_agent_from_history", reconstruct),
        patch(
            "xagent.web.api.chat.load_task_setup_snapshot_sync",
            return_value=snapshot,
        ) as snapshot_loader,
        patch.object(
            manager,
            "_has_reconstructable_history",
            side_effect=AssertionError("history pre-check ran on the event loop"),
            create=True,
        ),
    ):
        with _Patches(_stub_downstream(manager)):
            try:
                await manager.get_agent_for_task(
                    task_id=42,
                    db=db,
                    user=user,
                )
            except Exception:
                pass
    reconstruct.assert_awaited_once_with(
        42,
        db,
        scope=None,
        task_setup_snapshot=snapshot,
        connector_runtime_turn_id=None,
        mcp_runtime_authorization_policy=None,
    )
    snapshot_loader.assert_called_once_with(42, None)


@pytest.mark.asyncio
async def test_required_mcp_failure_does_not_fall_back_after_reconstruct() -> None:
    manager = AgentServiceManager()
    user = _make_user()
    task = _make_task(TaskStatus.RUNNING, agent_id=7)
    db = _build_db(
        task,
        trace_event=_Fake(),
        agent=_make_agent(),
        user=user,
    )
    error = RequiredMCPUnavailableError(
        [MCPUnavailableSummary.from_values("Gmail", "oauth_token_required")]
    )
    snapshot = _build_snapshot(task, user, trace_event=_Fake())

    async def fail_reconstruct(*args, **kwargs):
        manager._agents[42] = MagicMock()
        manager._agent_owner_ids[42] = 1
        raise error

    with (
        patch.object(
            manager, "_reconstruct_agent_from_history", side_effect=fail_reconstruct
        ),
        patch(
            "xagent.web.api.chat.load_task_setup_snapshot_sync",
            return_value=snapshot,
        ) as snapshot_loader,
        patch.object(
            manager,
            "_has_reconstructable_history",
            side_effect=AssertionError("history pre-check ran on the event loop"),
            create=True,
        ),
    ):
        with pytest.raises(RequiredMCPUnavailableError) as exc_info:
            await manager.get_agent_for_task(
                task_id=42,
                db=db,
                user=user,
            )

    assert exc_info.value is error
    snapshot_loader.assert_called_once_with(42, None)
    assert 42 not in manager._agents
    assert 42 not in manager._agent_owner_ids


@pytest.mark.asyncio
async def test_active_snapshot_owner_mismatch_does_not_fall_back() -> None:
    manager = AgentServiceManager()
    user = _make_user()
    task = _make_task(TaskStatus.RUNNING, agent_id=7)
    db = _build_db(task, agent=_make_agent(), user=user)
    mismatch = TaskOwnerMismatchError(42, expected=999, actual=1)

    with patch(
        "xagent.web.api.chat.load_task_setup_snapshot_sync",
        side_effect=mismatch,
    ) as snapshot_loader:
        with pytest.raises(TaskOwnerMismatchError) as exc_info:
            await manager.get_agent_for_task(
                task_id=42,
                db=db,
                user=user,
                task_owner_user_id=999,
            )

    assert exc_info.value is mismatch
    snapshot_loader.assert_called_once_with(42, 999)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [TaskStatus.RUNNING, TaskStatus.PENDING])
async def test_auto_model_unavailable_does_not_fall_back_during_task_setup(
    status: TaskStatus,
) -> None:
    """Auto exhaustion must escape both reconstruction and normal setup."""

    manager = AgentServiceManager()
    fallback = MagicMock()
    manager._default_llm = fallback
    user = _make_user()
    task = _make_task(status)
    db = _build_db(task, user=user)
    error = AutoModelUnavailableError("Auto model has no active configured candidates")

    with patch(
        "xagent.web.api.chat.load_task_setup_snapshot_sync",
        side_effect=error,
    ) as snapshot_loader:
        with pytest.raises(AutoModelUnavailableError) as exc_info:
            await manager.get_agent_for_task(task_id=42, db=db, user=user)

    assert exc_info.value is error
    assert manager._default_llm is fallback
    snapshot_loader.assert_called_once_with(42, None)


@pytest.mark.asyncio
async def test_running_with_dag_plan_runs_reconstruct() -> None:
    """Either signal is sufficient: a DAG plan (no trace event) also
    keeps reconstruct enabled."""
    manager = AgentServiceManager()
    user = _make_user()
    task = _make_task(TaskStatus.RUNNING, agent_id=7)
    agent_row = _make_agent()

    db = _build_db(
        task,
        trace_event=None,
        dag_execution=_Fake(),  # DAG plan exists
        agent=agent_row,
        user=user,
    )
    snapshot = _build_snapshot(task, user, dag_execution=_Fake())

    reconstruct = AsyncMock()
    with (
        patch.object(manager, "_reconstruct_agent_from_history", reconstruct),
        patch(
            "xagent.web.api.chat.load_task_setup_snapshot_sync",
            return_value=snapshot,
        ) as snapshot_loader,
        patch.object(
            manager,
            "_has_reconstructable_history",
            side_effect=AssertionError("history pre-check ran on the event loop"),
            create=True,
        ),
    ):
        with _Patches(_stub_downstream(manager)):
            try:
                await manager.get_agent_for_task(
                    task_id=42,
                    db=db,
                    user=user,
                )
            except Exception:
                pass
    reconstruct.assert_awaited_once_with(
        42,
        db,
        scope=None,
        task_setup_snapshot=snapshot,
        connector_runtime_turn_id=None,
        mcp_runtime_authorization_policy=None,
    )
    snapshot_loader.assert_called_once_with(42, None)


@pytest.mark.asyncio
async def test_paused_with_no_history_still_runs_reconstruct() -> None:
    """The pre-check intentionally only gates ``RUNNING``. A task in
    ``PAUSED`` state has prior runtime state by definition (something
    paused it); even if the DB queries inside reconstruct return empty
    (e.g. due to test fixtures), we must not short-circuit here.
    """
    manager = AgentServiceManager()
    user = _make_user()
    task = _make_task(TaskStatus.PAUSED, agent_id=7)
    agent_row = _make_agent()

    db = _build_db(
        task,
        trace_event=None,
        dag_execution=None,
        agent=agent_row,
        user=user,
    )
    snapshot = _build_snapshot(task, user)

    reconstruct = AsyncMock()
    with (
        patch.object(manager, "_reconstruct_agent_from_history", reconstruct),
        patch(
            "xagent.web.api.chat.load_task_setup_snapshot_sync",
            return_value=snapshot,
        ) as snapshot_loader,
    ):
        with _Patches(_stub_downstream(manager)):
            try:
                await manager.get_agent_for_task(
                    task_id=42,
                    db=db,
                    user=user,
                )
            except Exception:
                pass
    reconstruct.assert_awaited_once_with(
        42,
        db,
        scope=None,
        task_setup_snapshot=snapshot,
        connector_runtime_turn_id=None,
        mcp_runtime_authorization_policy=None,
    )
    snapshot_loader.assert_called_once_with(42, None)


@pytest.mark.asyncio
async def test_waiting_for_user_with_no_history_still_runs_reconstruct() -> None:
    """Same invariant as the PAUSED case above for ``WAITING_FOR_USER``."""
    manager = AgentServiceManager()
    user = _make_user()
    task = _make_task(TaskStatus.WAITING_FOR_USER, agent_id=7)
    agent_row = _make_agent()

    db = _build_db(
        task,
        trace_event=None,
        dag_execution=None,
        agent=agent_row,
        user=user,
    )
    snapshot = _build_snapshot(task, user)

    reconstruct = AsyncMock()
    with (
        patch.object(manager, "_reconstruct_agent_from_history", reconstruct),
        patch(
            "xagent.web.api.chat.load_task_setup_snapshot_sync",
            return_value=snapshot,
        ) as snapshot_loader,
    ):
        with _Patches(_stub_downstream(manager)):
            try:
                await manager.get_agent_for_task(
                    task_id=42,
                    db=db,
                    user=user,
                )
            except Exception:
                pass
    reconstruct.assert_awaited_once_with(
        42,
        db,
        scope=None,
        task_setup_snapshot=snapshot,
        connector_runtime_turn_id=None,
        mcp_runtime_authorization_policy=None,
    )
    snapshot_loader.assert_called_once_with(42, None)


@pytest.mark.asyncio
async def test_reconstruct_return_path_syncs_connector_runtime_turn() -> None:
    manager = AgentServiceManager()
    user = _make_user()
    task = _make_task(TaskStatus.PAUSED, agent_id=7)
    agent_row = _make_agent()
    db = _build_db(
        task, trace_event=None, dag_execution=None, agent=agent_row, user=user
    )
    snapshot = _build_snapshot(task, user)

    class _ToolConfig:
        def __init__(self) -> None:
            self.turn_ids: list[str] = []

        def set_connector_runtime_turn_id(self, turn_id: str) -> bool:
            self.turn_ids.append(turn_id)
            return True

    class _Agent:
        def __init__(self) -> None:
            self.tool_config = _ToolConfig()
            self.invalidated = False

        def invalidate_tools(self) -> None:
            self.invalidated = True

    reconstructed_agent = _Agent()

    async def reconstruct(
        task_id: int,
        _db: Any,
        scope: Any = None,
        task_setup_snapshot: TaskSetupSnapshot | None = None,
        connector_runtime_turn_id: str | None = None,
        mcp_runtime_authorization_policy: Any = None,
    ) -> None:
        assert task_setup_snapshot is snapshot
        assert connector_runtime_turn_id == "turn-reconstructed"
        assert mcp_runtime_authorization_policy is None
        manager._agents[task_id] = reconstructed_agent

    with (
        patch.object(manager, "_reconstruct_agent_from_history", reconstruct),
        patch(
            "xagent.web.api.chat.load_task_setup_snapshot_sync",
            return_value=snapshot,
        ) as snapshot_loader,
        patch.object(manager, "_load_persisted_conversation_history"),
        patch.object(manager, "_load_persisted_execution_context", new=AsyncMock()),
    ):
        agent = await manager.get_agent_for_task(
            task_id=42,
            db=db,
            user=user,
            connector_runtime_turn_id="turn-reconstructed",
        )

    assert agent is reconstructed_agent
    snapshot_loader.assert_called_once_with(42, None)
    assert reconstructed_agent.tool_config.turn_ids == ["turn-reconstructed"]
    assert reconstructed_agent.invalidated is True


class _Patches:
    """Compose a list of ``patch`` objects into a single context
    manager. Equivalent to ``with patch(a), patch(b), ...`` but
    accepts the patch list as a variable.
    """

    def __init__(self, patches: list[Any]) -> None:
        self._patches = patches

    def __enter__(self) -> None:
        for p in self._patches:
            p.start()

    def __exit__(self, *exc_info: Any) -> None:
        for p in reversed(self._patches):
            p.stop()
