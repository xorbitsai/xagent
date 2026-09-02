"""Pin that the orchestrator activates the execution scope on every turn.

The orchestrator resolves the scope and activates it via
``ExecutionScopeContext`` at the same places the acting user is resolved
(``UserContext``): ``execute_task_background`` for normal turns and
``execute_resume_background`` for resumed turns. These tests use a fake
resolver to pin that:

* the resolver is called with the turn's ``task_id`` (as str),
* the resolved scope is active inside the turn's execution context (visible
  to the agent build and the agent run),
* the resumed turn re-resolves the scope (restart/resume correctness), and
* with no resolver registered the turn runs unscoped, exactly as today.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError

from tests.shared.execution_scope import register_scope_resolver
from xagent.core.agent.checkpoint import (
    CheckpointAccessRefusedError,
    CheckpointCorruptError,
    CheckpointUnavailableError,
)
from xagent.core.execution_scope import (
    EXECUTION_SCOPE_NOT_PROVIDED,
    ExecutionScope,
    ExecutionScopeAuthorityError,
    get_execution_scope,
    set_execution_scope_snapshot_loader,
)
from xagent.web.api.websocket import (
    ResumeReservationOutcome,
    _acquire_resume_task_lease,
    _handle_resume_task_unserialized,
    execute_resume_background,
    execute_task_background,
)
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.client_error_messages import CLIENT_SAFE_TASK_FAILURE
from xagent.web.services.task_lease_service import (
    TaskLease,
    TaskLeaseHeartbeatOutcome,
)


def _make_task_orm() -> Task:
    return Task(
        id=42,
        user_id=1,
        title="scope wiring test",
        description="x",
        status=TaskStatus.RUNNING,
        agent_id=None,
        agent_type="standard",
    )


def _make_user_orm() -> User:
    return User(id=1, username="scope-user", password_hash="hash", is_admin=False)


def _build_db_mock() -> MagicMock:
    """A permissive Session double: any ``query(Model)`` chain resolves to
    the fake Task/User rows above (or None for other models)."""
    rows = {Task: _make_task_orm(), User: _make_user_orm()}

    def _query(model: type) -> Any:
        result = MagicMock()
        result.filter = MagicMock(return_value=result)
        result.first = MagicMock(return_value=rows.get(model))
        result.all = MagicMock(return_value=[])
        result.order_by = MagicMock(return_value=result)
        return result

    db = MagicMock()
    db.query = _query
    return db


def _bg_patches(db: Any) -> list[Any]:
    """Stub persistence so tests observe only scope activation."""

    snapshot = SimpleNamespace(
        task=SimpleNamespace(
            id=42,
            user_id=1,
            status=TaskStatus.RUNNING,
            agent_id=None,
        ),
        runtime_user=SimpleNamespace(id=1, is_admin=False),
        agent=None,
        conversation_history=(),
        execution_recovery=SimpleNamespace(messages=(), selected_skill_name=None),
    )

    return [
        patch(
            "xagent.web.services.task_setup_snapshot.load_task_setup_snapshot_sync",
            return_value=snapshot,
        ),
        patch(
            "xagent.web.api.websocket.background_task_manager.wait_for_previous",
            new=AsyncMock(),
        ),
        patch("xagent.web.api.websocket._register_uploaded_files_for_agent"),
        patch(
            "xagent.web.api.websocket._finalize_task_execution_result_isolated",
            return_value=SimpleNamespace(
                normalized_outputs=[],
                ai_response="ok",
                chat_response=None,
                waiting_for_control=False,
                terminal_state_committed=True,
                final_control_snapshot=None,
                final_task_status=TaskStatus.COMPLETED.value,
                broadcast_meta={
                    "id": 42,
                    "title": "scope wiring test",
                    "description": "x",
                    "execution_mode": None,
                    "updated_at": None,
                },
                late_result=False,
            ),
        ),
        patch(
            "xagent.web.api.websocket.manager", MagicMock(broadcast_to_task=AsyncMock())
        ),
    ]


class _Patches:
    def __init__(self, patches: list[Any]) -> None:
        self._patches = patches

    def __enter__(self) -> None:
        for p in self._patches:
            p.start()

    def __exit__(self, *exc_info: Any) -> None:
        for p in reversed(self._patches):
            p.stop()


@pytest.mark.asyncio
async def test_bg_turn_resolves_and_activates_scope() -> None:
    """The resolver runs at turn start and its scope is active during both
    the agent build (``get_agent_for_task``) and the agent run."""
    scope = ExecutionScope(sandbox_key_suffix="tenant-a")
    resolver_calls: list[str] = []

    def resolver(task_id: str) -> ExecutionScope:
        resolver_calls.append(task_id)
        return scope

    register_scope_resolver(resolver)

    seen: dict[str, Any] = {}
    agent_service = MagicMock()
    agent_service.set_outbound_message_handler = MagicMock()
    agent_service.set_execution_context_messages = MagicMock()
    agent_service.set_recovered_skill_context = MagicMock()

    async def _get_agent_for_task(*args: Any, **kwargs: Any) -> Any:
        seen["scope_at_build"] = get_execution_scope()
        return agent_service

    async def _execute_task(**kwargs: Any) -> dict:
        seen["scope_at_run"] = get_execution_scope()
        return {"success": True, "output": "ok", "status": "completed"}

    agent_manager = MagicMock(
        get_agent_for_task=AsyncMock(side_effect=_get_agent_for_task),
        execute_task=AsyncMock(side_effect=_execute_task),
    )

    with _Patches(_bg_patches(_build_db_mock())):
        await execute_task_background(
            task_id=42,
            user_message="hi",
            context={},
            agent_manager=agent_manager,
            task_owner_user_id=1,
        )

    assert resolver_calls == ["42"]
    assert seen["scope_at_build"] is scope
    assert seen["scope_at_run"] is scope
    # The scope is turn-local: nothing leaks past the turn.
    assert get_execution_scope() is None


@pytest.mark.asyncio
async def test_bg_turn_without_resolver_runs_unscoped() -> None:
    """No resolver registered -> the turn executes unscoped (today's
    behavior, byte-for-byte)."""
    seen: dict[str, Any] = {}
    agent_service = MagicMock()

    async def _execute_task(**kwargs: Any) -> dict:
        seen["scope_at_run"] = get_execution_scope()
        return {"success": True, "output": "ok", "status": "completed"}

    agent_manager = MagicMock(
        get_agent_for_task=AsyncMock(return_value=agent_service),
        execute_task=AsyncMock(side_effect=_execute_task),
    )

    with _Patches(_bg_patches(_build_db_mock())):
        await execute_task_background(
            task_id=42,
            user_message="hi",
            context={},
            agent_manager=agent_manager,
            task_owner_user_id=1,
        )

    assert seen["scope_at_run"] is None


@pytest.mark.asyncio
async def test_bg_turn_explicit_unscoped_does_not_resolve_again() -> None:
    """A caller that already resolved the turn to ``None`` must be able to
    pass that result through without making ``None`` look like "not resolved".
    """

    def unexpected_resolver(task_id: str) -> ExecutionScope:
        raise AssertionError(f"scope for task {task_id} was resolved twice")

    register_scope_resolver(unexpected_resolver)
    agent_service = MagicMock()
    agent_manager = MagicMock(
        get_agent_for_task=AsyncMock(return_value=agent_service),
        execute_task=AsyncMock(
            return_value={"success": True, "output": "ok", "status": "completed"}
        ),
    )

    with _Patches(_bg_patches(_build_db_mock())):
        await execute_task_background(
            task_id=42,
            user_message="hi",
            context={},
            agent_manager=agent_manager,
            task_owner_user_id=1,
            resolved_execution_scope=None,
        )

    assert (
        agent_manager.get_agent_for_task.await_args.kwargs["resolved_execution_scope"]
        is None
    )


@pytest.mark.asyncio
async def test_owned_bg_failure_waits_for_exact_settlement_before_broadcast() -> None:
    """The lease owner must durably settle before any terminal event is sent."""
    agent_service = MagicMock()
    agent_manager = MagicMock(
        get_agent_for_task=AsyncMock(return_value=agent_service),
        execute_task=AsyncMock(side_effect=RuntimeError("owned run failed")),
    )
    broadcast = AsyncMock()
    patches = _bg_patches(_build_db_mock())
    patches[-1] = patch(
        "xagent.web.api.websocket.manager",
        MagicMock(broadcast_to_task=broadcast),
    )

    with _Patches(patches), pytest.raises(RuntimeError, match="owned run failed"):
        await execute_task_background(
            task_id=42,
            user_message="hi",
            context={},
            agent_manager=agent_manager,
            task_owner_user_id=1,
            task_lease=TaskLease(
                task_id=42,
                runner_id="runner-a",
                run_id="run-a",
            ),
            resolved_execution_scope=None,
        )

    broadcast.assert_not_awaited()


@pytest.mark.asyncio
async def test_bg_turn_authority_mismatch_fails_the_turn() -> None:
    """Turn-face consumer: a namespace-affecting disagreement between the
    registered resolver and a persisted snapshot must fail the turn --
    unlike the off-turn consumers (``ManagedFileRef``'s construction, the
    pause/resume handlers), which downgrade the same mismatch to the
    resolver's authoritative answer plus a warning. Being reached off a turn
    is not by itself what decides this: ``websocket._scope_segments_for_task``
    also runs off-turn yet resolves fail-closed, because its caller composes
    a storage key for new bytes."""
    register_scope_resolver(
        lambda task_id: ExecutionScope(sandbox_key_suffix="from-resolver"),
    )
    set_execution_scope_snapshot_loader(
        lambda task_id: ExecutionScope(sandbox_key_suffix="from-snapshot")
    )
    agent_manager = MagicMock()

    with (
        _Patches(_bg_patches(_build_db_mock())),
        pytest.raises(ExecutionScopeAuthorityError),
    ):
        await execute_task_background(
            task_id=42,
            user_message="hi",
            context={},
            agent_manager=agent_manager,
            task_owner_user_id=1,
            task_lease=TaskLease(
                task_id=42,
                runner_id="runner-a",
                run_id="run-a",
            ),
        )

    agent_manager.get_agent_for_task.assert_not_called()


@pytest.mark.asyncio
async def test_bg_turn_resolves_scope_before_loading_snapshot_off_loop() -> None:
    """Scope and setup snapshot are resolved sequentially in workers."""
    main_thread_id = threading.get_ident()
    events: list[tuple[str, int]] = []
    scope = ExecutionScope(sandbox_key_suffix="tenant-a")

    def resolver(task_id: str) -> ExecutionScope:
        events.append(("resolve", threading.get_ident()))
        return scope

    register_scope_resolver(resolver)

    def load_snapshot(*_args: Any, **_kwargs: Any) -> Any:
        events.append(("snapshot", threading.get_ident()))
        return SimpleNamespace(
            task=SimpleNamespace(
                id=42,
                user_id=1,
                status=TaskStatus.RUNNING,
                agent_id=None,
            ),
            runtime_user=SimpleNamespace(id=1, is_admin=False),
            agent=None,
            conversation_history=(),
            execution_recovery=SimpleNamespace(messages=(), selected_skill_name=None),
        )

    agent_service = MagicMock()
    agent_manager = MagicMock(
        get_agent_for_task=AsyncMock(return_value=agent_service),
        execute_task=AsyncMock(
            return_value={"success": True, "output": "ok", "status": "completed"}
        ),
    )
    patches = _bg_patches(_build_db_mock())
    patches[0] = patch(
        "xagent.web.services.task_setup_snapshot.load_task_setup_snapshot_sync",
        side_effect=load_snapshot,
    )

    with _Patches(patches):
        await execute_task_background(
            task_id=42,
            user_message="hi",
            context={},
            agent_manager=agent_manager,
            task_owner_user_id=1,
        )

    assert events[0][0] == "resolve"
    assert events[0][1] != main_thread_id
    assert events[1][0] == "snapshot"
    assert events[1][1] != main_thread_id


@pytest.mark.asyncio
async def test_resumed_turn_re_resolves_scope() -> None:
    """A resumed execution re-resolves through the hook and runs with the
    identical scope — this is what makes scope survive a process restart:
    nothing is carried in memory, the resolver re-derives it per turn."""
    scope = ExecutionScope(
        sandbox_key_suffix="tenant-a", workspace_segments=("tenant-a",)
    )
    resolver_calls: list[str] = []

    def resolver(task_id: str) -> ExecutionScope:
        resolver_calls.append(task_id)
        return scope

    register_scope_resolver(resolver)

    seen: dict[str, Any] = {}
    agent_service = MagicMock()

    async def _resume(task_id: str) -> dict:
        seen["scope_at_resume"] = get_execution_scope()
        return {"status": "completed", "success": True, "output": "ok"}

    agent_service.resume_execution_by_id = AsyncMock(side_effect=_resume)

    async def _heartbeat(lease: Any, stop_event: Any) -> None:
        return None

    db = _build_db_mock()

    def _fresh_db_gen():
        yield db

    with _Patches(
        [
            patch(
                "xagent.web.models.database.get_db",
                side_effect=lambda: _fresh_db_gen(),
            ),
            patch(
                "xagent.web.api.websocket._acquire_resume_task_lease",
                return_value=TaskLease(
                    task_id=42,
                    runner_id="scope-runner",
                    run_id="scope-run",
                ),
            ),
            patch(
                "xagent.web.api.websocket.run_task_lease_heartbeat",
                side_effect=_heartbeat,
            ),
            patch(
                "xagent.web.api.websocket.stop_task_lease_heartbeat", new=AsyncMock()
            ),
            patch(
                "xagent.web.api.websocket._finalize_resumed_task",
                return_value={
                    "task_title": "scope wiring test",
                    "task_description": "x",
                    "task_execution_mode": "flash",
                    "task_agent_id": None,
                    "agent_name": None,
                    "agent_logo_url": None,
                    "final_status": TaskStatus.COMPLETED.value,
                    "lease_released": True,
                    "control_event_state": {},
                    "normalized_outputs": [],
                    "output": "ok",
                    "late_result": False,
                },
            ),
            patch(
                "xagent.web.api.websocket.manager",
                MagicMock(broadcast_to_task=AsyncMock()),
            ),
            patch(
                "xagent.web.api.websocket.background_task_manager.promote_resume_task"
            ),
        ]
    ):
        await execute_resume_background(
            task_id=42,
            agent_service=agent_service,
            task_owner_user_id=1,
        )

    agent_service.resume_execution_by_id.assert_awaited_once()
    assert resolver_calls == ["42"]
    assert seen["scope_at_resume"] is scope
    assert get_execution_scope() is None


@pytest.mark.asyncio
async def test_resume_background_adopts_preacquired_lease_without_reacquiring() -> None:
    lease = TaskLease(task_id=42, runner_id="runner-a", run_id="run-a")
    heartbeat_stop = asyncio.Event()

    async def transferred_heartbeat() -> TaskLeaseHeartbeatOutcome:
        await heartbeat_stop.wait()
        return TaskLeaseHeartbeatOutcome()

    heartbeat_task = asyncio.create_task(transferred_heartbeat())
    agent_service = MagicMock()
    agent_service.resume_execution_by_id = AsyncMock(
        return_value={"status": "completed", "success": True, "output": "ok"}
    )
    acquire = MagicMock()
    start_heartbeat = MagicMock()

    with _Patches(
        [
            patch(
                "xagent.web.api.websocket._acquire_resume_task_lease",
                new=acquire,
            ),
            patch(
                "xagent.web.api.websocket.run_task_lease_heartbeat",
                new=start_heartbeat,
            ),
            patch(
                "xagent.web.api.websocket._finalize_resumed_task",
                return_value={
                    "task_title": "prelease",
                    "task_description": "x",
                    "task_execution_mode": "flash",
                    "task_agent_id": None,
                    "agent_name": None,
                    "agent_logo_url": None,
                    "final_status": TaskStatus.COMPLETED.value,
                    "lease_released": True,
                    "control_event_state": {},
                    "normalized_outputs": [],
                    "output": "ok",
                    "late_result": False,
                },
            ),
            patch(
                "xagent.web.api.websocket.manager",
                MagicMock(broadcast_to_task=AsyncMock()),
            ),
            patch(
                "xagent.web.api.websocket.background_task_manager.promote_resume_task"
            ),
        ]
    ):
        await execute_resume_background(
            task_id=42,
            agent_service=agent_service,
            task_owner_user_id=1,
            expected_run_id="run-a",
            resolved_execution_scope=None,
            preacquired_lease=lease,
            preacquired_heartbeat_stop=heartbeat_stop,
            preacquired_heartbeat_task=heartbeat_task,
        )

    acquire.assert_not_called()
    start_heartbeat.assert_not_called()
    agent_service.resume_execution_by_id.assert_awaited_once_with("42")
    assert heartbeat_task.done()


@pytest.mark.asyncio
async def test_resume_handler_resolves_scope_once_off_loop_for_agent_lookup() -> None:
    """The handler's single off-loop resolution feeds only the agent lookup.

    ``get_agent_for_task`` receives the resolved scope to locate the paused
    task's existing agent. ``execute_resume_background`` is a different
    consumer -- the turn that selects a namespace for new bytes -- so it is
    not handed this same value; it gets ``EXECUTION_SCOPE_NOT_PROVIDED`` and
    resolves for itself.
    """
    main_thread_id = threading.get_ident()
    scope = ExecutionScope(
        sandbox_key_suffix="tenant-a", workspace_segments=("tenant-a",)
    )
    resolver_threads: list[int] = []

    def resolver(task_id: str) -> ExecutionScope:
        assert task_id == "42"
        resolver_threads.append(threading.get_ident())
        return scope

    register_scope_resolver(resolver)
    snapshot = SimpleNamespace(
        task=SimpleNamespace(
            id=42,
            user_id=1,
            status=TaskStatus.PAUSED,
            control_state="paused",
            run_id="run-a",
            state_version=3,
        ),
        runtime_user=SimpleNamespace(id=1, is_admin=False),
    )
    agent_service = MagicMock()
    agent_service.supports_live_control.return_value = True
    agent_manager = MagicMock(get_agent_for_task=AsyncMock(return_value=agent_service))
    resume_started = asyncio.Event()
    resume_kwargs: dict[str, Any] = {}

    async def execute_resume(**kwargs: Any) -> None:
        resume_kwargs.update(kwargs)
        resume_started.set()

    background_manager = MagicMock()
    background_manager.running_tasks = {}
    background_manager.resume_admission_state.return_value = None
    background_manager.try_reserve_resume.return_value = (
        ResumeReservationOutcome.RESERVED
    )
    transition = AsyncMock(
        return_value=SimpleNamespace(run_id="run-a", status=TaskStatus.PAUSED)
    )

    with _Patches(
        [
            patch(
                "xagent.web.services.task_setup_snapshot.load_task_setup_snapshot_sync",
                return_value=snapshot,
            ),
            patch("xagent.web.api.chat.get_agent_manager", return_value=agent_manager),
            patch(
                "xagent.web.api.websocket.task_execution_controller.transition",
                new=transition,
            ),
            patch(
                "xagent.web.api.websocket.background_task_manager",
                background_manager,
            ),
            # The handler asks the DB whether another process still holds a
            # live lease before scheduling; this suite drives it without a
            # task row, so answer "no foreign owner" explicitly.
            patch(
                "xagent.web.api.websocket.task_has_live_foreign_runner",
                return_value=False,
            ),
            patch(
                "xagent.web.api.websocket.execute_resume_background",
                side_effect=execute_resume,
            ),
        ]
    ):
        await _handle_resume_task_unserialized(
            MagicMock(),
            42,
            {"user": SimpleNamespace(id=1, is_admin=False)},
        )
        await asyncio.wait_for(resume_started.wait(), timeout=1)

    assert len(resolver_threads) == 1
    assert resolver_threads[0] != main_thread_id
    assert (
        agent_manager.get_agent_for_task.await_args.kwargs["resolved_execution_scope"]
        is scope
    )
    assert resume_kwargs["resolved_execution_scope"] is EXECUTION_SCOPE_NOT_PROVIDED


@pytest.mark.asyncio
async def test_resume_survives_a_scope_authority_mismatch() -> None:
    """A control operation must stay available while the scope is in dispute.

    Resume locates an already-established workspace and sandbox for a task
    that already exists using the downgraded (never-raising) off-turn
    resolution -- that lookup selects no namespace for new bytes, so it must
    not be blocked by the dispute. The turn it hands off to is a different
    consumer that does select a namespace for new bytes, so the handler must
    not pass its own downgraded answer through: it leaves that argument at
    ``EXECUTION_SCOPE_NOT_PROVIDED`` so the scheduled turn performs its own
    fail-closed resolution (see
    ``test_resume_background_fails_closed_on_scope_authority_mismatch`` for
    that resolution actually raising). Failing the handler itself instead
    would leave the user unable to resume, pause or stop the task at all,
    with the authority error escaping the socket loop.
    """
    resolver_scope = ExecutionScope(
        sandbox_key_suffix="from-resolver", workspace_segments=("from-resolver",)
    )
    register_scope_resolver(lambda task_id: resolver_scope)
    set_execution_scope_snapshot_loader(
        lambda task_id: ExecutionScope(
            sandbox_key_suffix="from-snapshot", workspace_segments=("from-snapshot",)
        )
    )
    snapshot = SimpleNamespace(
        task=SimpleNamespace(
            id=42,
            user_id=1,
            status=TaskStatus.PAUSED,
            control_state="paused",
            run_id="run-a",
            state_version=3,
        ),
        runtime_user=SimpleNamespace(id=1, is_admin=False),
    )
    agent_service = MagicMock()
    agent_service.supports_live_control.return_value = True
    agent_manager = MagicMock(get_agent_for_task=AsyncMock(return_value=agent_service))
    resume_started = asyncio.Event()
    resume_kwargs: dict[str, Any] = {}

    async def execute_resume(**kwargs: Any) -> None:
        resume_kwargs.update(kwargs)
        resume_started.set()

    background_manager = MagicMock()
    background_manager.running_tasks = {}
    background_manager.resume_admission_state.return_value = None
    background_manager.try_reserve_resume.return_value = (
        ResumeReservationOutcome.RESERVED
    )

    with _Patches(
        [
            patch(
                "xagent.web.services.task_setup_snapshot.load_task_setup_snapshot_sync",
                return_value=snapshot,
            ),
            patch("xagent.web.api.chat.get_agent_manager", return_value=agent_manager),
            patch(
                "xagent.web.api.websocket.task_execution_controller.transition",
                new=AsyncMock(
                    return_value=SimpleNamespace(
                        run_id="run-a", status=TaskStatus.PAUSED
                    )
                ),
            ),
            patch(
                "xagent.web.api.websocket.background_task_manager",
                background_manager,
            ),
            # The handler asks the DB whether another process still holds a
            # live lease before scheduling; this suite drives it without a
            # task row, so answer "no foreign owner" explicitly.
            patch(
                "xagent.web.api.websocket.task_has_live_foreign_runner",
                return_value=False,
            ),
            patch(
                "xagent.web.api.websocket.execute_resume_background",
                side_effect=execute_resume,
            ),
        ]
    ):
        await _handle_resume_task_unserialized(
            MagicMock(),
            42,
            {"user": SimpleNamespace(id=1, is_admin=False)},
        )
        await asyncio.wait_for(resume_started.wait(), timeout=1)

    # The off-turn lookup that locates the existing agent still uses the
    # downgraded resolver answer -- a control operation on an existing task
    # must not be blocked by the dispute.
    assert (
        agent_manager.get_agent_for_task.await_args.kwargs["resolved_execution_scope"]
        is resolver_scope
    )
    # The scheduled turn is a different consumer: the handler must not pass
    # its own downgraded answer through, so the turn can run its own
    # fail-closed resolution instead of silently selecting the resolver's
    # disputed answer for new bytes.
    assert resume_kwargs["resolved_execution_scope"] is EXECUTION_SCOPE_NOT_PROVIDED


@pytest.mark.asyncio
async def test_resume_background_fails_closed_on_scope_authority_mismatch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The scheduled turn's own re-resolution is fail-closed, not downgraded.

    ``execute_resume_background`` is the consumer that selects a namespace
    for new bytes for the resumed turn. Unlike the handler's off-turn lookup
    pinned in ``test_resume_survives_a_scope_authority_mismatch``, a
    resolver/snapshot disagreement reaching this function's own resolution
    (``resolved_execution_scope`` left at ``EXECUTION_SCOPE_NOT_PROVIDED``)
    must fail the turn instead of silently proceeding under either answer.
    The ``ExecutionScopeAuthorityError`` raised inside this background task
    is caught by its own settlement handler (a background task has no
    caller to propagate to) and converted into a broadcast task-failure
    event rather than executing the resume -- that conversion, not an
    escaping exception, is what "fails closed" means for this consumer.
    """
    register_scope_resolver(
        lambda task_id: ExecutionScope(sandbox_key_suffix="from-resolver")
    )
    set_execution_scope_snapshot_loader(
        lambda task_id: ExecutionScope(sandbox_key_suffix="from-snapshot")
    )
    agent_service = MagicMock()
    agent_service.resume_execution_by_id = AsyncMock()
    broadcast_manager = MagicMock(broadcast_to_task=AsyncMock())

    with _Patches(
        [
            patch(
                "xagent.web.api.websocket.background_task_manager.promote_resume_task"
            ),
            patch("xagent.web.api.websocket.manager", broadcast_manager),
        ]
    ):
        await execute_resume_background(
            task_id=42,
            agent_service=agent_service,
            task_owner_user_id=1,
        )

    agent_service.resume_execution_by_id.assert_not_awaited()
    broadcast_manager.broadcast_to_task.assert_awaited_once()
    broadcast_args, _ = broadcast_manager.broadcast_to_task.await_args
    event, broadcast_task_id = broadcast_args
    assert broadcast_task_id == 42
    assert event["type"] == "task_error"
    assert event["task"]["status"] == TaskStatus.FAILED.value
    assert event["message"] == CLIENT_SAFE_TASK_FAILURE
    assert event["error"] == CLIENT_SAFE_TASK_FAILURE
    assert "execution scope authority mismatch" not in repr(event).lower()
    assert "execution scope authority mismatch" in caplog.text.lower()


def test_resume_acquire_checkout_timeout_before_claim_does_not_try_cleanup() -> None:
    query = MagicMock()
    query.filter.return_value = query
    query.first.side_effect = SQLAlchemyTimeoutError("pool exhausted")
    db = MagicMock()
    db.query.return_value = query
    session_context = MagicMock()
    session_context.__enter__.return_value = db
    session_context.__exit__.return_value = False
    session_factory = MagicMock(return_value=session_context)
    transaction_claim = MagicMock()

    with (
        patch(
            "xagent.web.api.websocket.get_session_local",
            return_value=session_factory,
        ),
        patch(
            "xagent.web.api.websocket.acquire_task_lease_no_commit",
            transaction_claim,
        ),
        pytest.raises(SQLAlchemyTimeoutError, match="pool exhausted"),
    ):
        _acquire_resume_task_lease(42, 1, "run-a")

    transaction_claim.assert_not_called()


@pytest.mark.asyncio
async def test_resume_db_lifecycle_runs_in_short_session_workers() -> None:
    """Resume lease acquisition, terminal persistence, and fallback release
    must all run off-loop, while the tracker is constructed without a caller
    Session and the resolved unscoped value stays explicit.
    """
    main_thread_id = threading.get_ident()
    worker_events: list[tuple[str, int]] = []
    lease = TaskLease(task_id=42, runner_id="runner-a", run_id="run-a")

    def acquire_resume_lease(*args: Any, **kwargs: Any) -> object:
        worker_events.append(("acquire", threading.get_ident()))
        return lease

    def finalize_resume(*args: Any, **kwargs: Any) -> dict[str, Any]:
        worker_events.append(("finalize", threading.get_ident()))
        return {
            "task_title": "scope wiring test",
            "task_description": "x",
            "task_execution_mode": "flash",
            "task_agent_id": None,
            "agent_name": None,
            "agent_logo_url": None,
            "final_status": TaskStatus.COMPLETED.value,
            "lease_released": False,
            "control_event_state": {},
            "normalized_outputs": [],
            "output": "ok",
            "late_result": False,
        }

    def release_resume_lease(
        acquired_lease: object,
        *,
        error_message: str | None,
        turn_id: str | None = None,
    ) -> None:
        assert acquired_lease is lease
        assert error_message is None
        worker_events.append(("release", threading.get_ident()))

    class FakeTracker:
        quota_interrupt_reason = None

        def __init__(self, *, task_id: int, **kwargs: Any) -> None:
            assert task_id == 42
            assert kwargs == {
                "expected_run_id": "run-a",
                "expected_runner_id": "runner-a",
            }

        async def start_tracking(self) -> None:
            return None

        async def complete_tracking(self) -> None:
            return None

        async def interrupt_reason_for_quota(self) -> None:
            return None

    agent_service = MagicMock()
    agent_service.resume_execution_by_id = AsyncMock(
        return_value={"status": "completed", "success": True, "output": "ok"}
    )

    with _Patches(
        [
            patch(
                "xagent.web.api.websocket._acquire_resume_task_lease",
                side_effect=acquire_resume_lease,
                create=True,
            ),
            patch(
                "xagent.web.api.websocket._finalize_resumed_task",
                side_effect=finalize_resume,
                create=True,
            ),
            patch(
                "xagent.web.api.websocket._settle_resumed_task_lease",
                side_effect=release_resume_lease,
                create=True,
            ),
            patch(
                "xagent.web.api.websocket.run_task_lease_heartbeat",
                new=AsyncMock(),
            ),
            patch(
                "xagent.web.api.websocket.stop_task_lease_heartbeat",
                new=AsyncMock(),
            ),
            patch(
                "xagent.web.api.websocket.manager",
                MagicMock(broadcast_to_task=AsyncMock()),
            ),
            patch(
                "xagent.web.api.websocket.background_task_manager.promote_resume_task"
            ),
            patch("xagent.web.tracking.task_tracker.TaskTracker", FakeTracker),
        ]
    ):
        await execute_resume_background(
            task_id=42,
            agent_service=agent_service,
            task_owner_user_id=1,
            resolved_execution_scope=None,
        )

    assert [name for name, _thread_id in worker_events] == [
        "acquire",
        "finalize",
        "release",
    ]
    assert all(thread_id != main_thread_id for _, thread_id in worker_events)


@pytest.mark.asyncio
async def test_resume_final_usage_and_heartbeat_finish_before_lease_release() -> None:
    lease = TaskLease(task_id=42, runner_id="runner-a", run_id="run-a")
    events: list[str] = []

    class FakeTracker:
        quota_interrupt_reason = None

        def __init__(self, *, task_id: int, **kwargs: Any) -> None:
            assert task_id == 42
            assert kwargs == {
                "expected_run_id": "run-a",
                "expected_runner_id": "runner-a",
            }

        async def start_tracking(self) -> None:
            return None

        async def complete_tracking(self) -> None:
            events.append("usage")

        async def interrupt_reason_for_quota(self) -> None:
            return None

    async def stop_heartbeat(*_args: Any) -> None:
        events.append("heartbeat")

    def finalize(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        events.append("lease-finalizer")
        return {
            "task_title": "scope wiring test",
            "task_description": "x",
            "task_execution_mode": "flash",
            "task_agent_id": None,
            "agent_name": None,
            "agent_logo_url": None,
            "final_status": TaskStatus.COMPLETED.value,
            "lease_released": True,
            "control_event_state": {},
            "normalized_outputs": [],
            "output": "ok",
            "late_result": False,
        }

    agent_service = MagicMock()
    agent_service.resume_execution_by_id = AsyncMock(
        return_value={"status": "completed", "success": True, "output": "ok"}
    )

    with _Patches(
        [
            patch(
                "xagent.web.api.websocket._acquire_resume_task_lease",
                return_value=lease,
            ),
            patch(
                "xagent.web.api.websocket.run_task_lease_heartbeat",
                new=AsyncMock(),
            ),
            patch(
                "xagent.web.api.websocket.stop_task_lease_heartbeat",
                side_effect=stop_heartbeat,
            ),
            patch(
                "xagent.web.api.websocket._finalize_resumed_task",
                side_effect=finalize,
            ),
            patch(
                "xagent.web.api.websocket.manager",
                MagicMock(broadcast_to_task=AsyncMock()),
            ),
            patch(
                "xagent.web.api.websocket.background_task_manager.promote_resume_task"
            ),
            patch("xagent.web.tracking.task_tracker.TaskTracker", FakeTracker),
        ]
    ):
        await execute_resume_background(
            task_id=42,
            agent_service=agent_service,
            task_owner_user_id=1,
            resolved_execution_scope=None,
        )

    assert events == ["usage", "heartbeat", "lease-finalizer"]


@pytest.mark.asyncio
async def test_resume_pool_timeout_does_not_start_secondary_db_cleanup(caplog) -> None:
    """A resume checkout timeout retains its exact lease for TTL recovery."""
    lease = TaskLease(task_id=42, runner_id="runner-a", run_id="run-a")
    mark_delivery = MagicMock()
    settle = MagicMock()
    snapshot = AsyncMock()
    tracker_instances: list[Any] = []

    class FakeTracker:
        quota_interrupt_reason = None

        def __init__(self, *, task_id: int, **kwargs: Any) -> None:
            assert task_id == 42
            assert kwargs == {
                "expected_run_id": "run-a",
                "expected_runner_id": "runner-a",
            }
            self.complete_tracking = AsyncMock()
            self.stop_periodic_updates = AsyncMock()
            tracker_instances.append(self)

        async def start_tracking(self) -> None:
            return None

        async def interrupt_reason_for_quota(self) -> None:
            return None

    agent_service = MagicMock()
    agent_service.resume_execution_by_id = AsyncMock(
        return_value={"status": "completed", "success": True, "output": "ok"}
    )
    ws_manager = MagicMock(broadcast_to_task=AsyncMock())
    caplog.set_level(logging.ERROR, logger="xagent.web.api.websocket")

    with _Patches(
        [
            patch(
                "xagent.web.api.websocket._acquire_resume_task_lease",
                return_value=lease,
            ),
            patch(
                "xagent.web.api.websocket._finalize_resumed_task",
                side_effect=SQLAlchemyTimeoutError("pool exhausted"),
            ),
            patch(
                "xagent.web.api.websocket._settle_resumed_task_lease",
                settle,
            ),
            patch(
                "xagent.web.api.websocket.mark_user_message_delivery_sync",
                mark_delivery,
            ),
            patch(
                "xagent.web.api.websocket.task_execution_controller.snapshot",
                new=snapshot,
            ),
            patch(
                "xagent.web.api.websocket.run_task_lease_heartbeat",
                new=AsyncMock(),
            ),
            patch(
                "xagent.web.api.websocket.stop_task_lease_heartbeat",
                new=AsyncMock(),
            ),
            patch("xagent.web.api.websocket.manager", ws_manager),
            patch(
                "xagent.web.api.websocket.background_task_manager.promote_resume_task"
            ),
            patch("xagent.web.tracking.task_tracker.TaskTracker", FakeTracker),
        ]
    ):
        await execute_resume_background(
            task_id=42,
            agent_service=agent_service,
            task_owner_user_id=1,
            delivery_turn_id="resume-turn",
            expected_run_id="run-a",
            resolved_execution_scope=None,
        )

    assert tracker_instances
    tracker_instances[0].complete_tracking.assert_awaited_once()
    tracker_instances[0].stop_periodic_updates.assert_not_awaited()
    mark_delivery.assert_not_called()
    snapshot.assert_not_awaited()
    settle.assert_not_called()
    assert "task_id=42" in caplog.text
    assert "component=resume" in caplog.text
    assert "retaining lease for TTL recovery" in caplog.text
    assert not any(
        call.args[0].get("type") == "task_error"
        for call in ws_manager.broadcast_to_task.call_args_list
    )


def test_acquire_resume_lease_reports_prior_status_before_claiming() -> None:
    """The real acquire must fill the prior-status box on the claim path.

    Every restore-to-prior test below patches this function with a fake
    that mirrors the out parameter, so nothing else would notice if the
    production function stopped populating it -- the resume path would
    just silently fall back to the terminal FAILED branch. Pin the real
    function against its own contract, and pin that the status recorded
    is the pre-acquisition one rather than the RUNNING the claim writes.
    """
    lease = TaskLease(task_id=42, runner_id="runner-a", run_id="run-a")
    task = SimpleNamespace(id=42, user_id=1, status=TaskStatus.WAITING_FOR_USER)
    query = MagicMock()
    query.filter.return_value = query
    query.first.return_value = task
    db = MagicMock()
    db.query.return_value = query
    session_context = MagicMock()
    session_context.__enter__.return_value = db
    session_context.__exit__.return_value = False
    session_factory = MagicMock(return_value=session_context)

    prior_status_box: list[TaskStatus] = []
    with (
        patch(
            "xagent.web.api.websocket.get_session_local",
            return_value=session_factory,
        ),
        patch(
            "xagent.web.api.websocket.acquire_task_lease_no_commit",
            return_value=lease,
        ),
        patch("xagent.web.api.websocket.sync_workforce_run_status"),
    ):
        acquired = _acquire_resume_task_lease(
            42,
            1,
            "run-a",
            prior_status_out=prior_status_box,
        )

    assert acquired is lease
    assert prior_status_box == [TaskStatus.WAITING_FOR_USER]


def test_acquire_resume_lease_reports_prior_status_when_claim_is_lost() -> None:
    """A lost claim must not leave the box holding a fabricated status."""
    task = SimpleNamespace(id=42, user_id=1, status=TaskStatus.PAUSED)
    query = MagicMock()
    query.filter.return_value = query
    query.first.return_value = task
    db = MagicMock()
    db.query.return_value = query
    session_context = MagicMock()
    session_context.__enter__.return_value = db
    session_context.__exit__.return_value = False
    session_factory = MagicMock(return_value=session_context)

    prior_status_box: list[TaskStatus] = []
    with (
        patch(
            "xagent.web.api.websocket.get_session_local",
            return_value=session_factory,
        ),
        patch(
            "xagent.web.api.websocket.acquire_task_lease_no_commit",
            return_value=None,
        ),
    ):
        acquired = _acquire_resume_task_lease(
            42,
            1,
            "run-a",
            prior_status_out=prior_status_box,
        )

    assert acquired is None
    # Recorded, but the caller only consults it when a lease was claimed,
    # so a lost claim can never restore a status it does not own.
    assert prior_status_box == [TaskStatus.PAUSED]


@pytest.mark.asyncio
async def test_resume_background_restores_prior_status_for_preacquired_lease() -> None:
    """A resume that adopts someone else's lease owes the same restore.

    The A2A input-required flow claims the lease itself and hands it over,
    so this entry never runs the acquisition that captures the status. If
    the handover does not carry it, a checkpoint the resume cannot read
    downgrades a still-resumable task to a terminal FAILED on that path
    while the self-acquiring path recovers it.
    """
    lease = TaskLease(task_id=42, runner_id="runner-a", run_id="run-a")
    heartbeat_stop = asyncio.Event()
    heartbeat_task: asyncio.Task[Any] = asyncio.create_task(asyncio.sleep(0))
    settle = MagicMock()
    restore = MagicMock(return_value=True)

    class FakeTracker:
        quota_interrupt_reason = None

        def __init__(self, *, task_id: int, **kwargs: Any) -> None:
            self.complete_tracking = AsyncMock()
            self.stop_periodic_updates = AsyncMock()

        async def start_tracking(self) -> None:
            return None

        async def interrupt_reason_for_quota(self) -> None:
            return None

    agent_service = MagicMock()
    agent_service.resume_execution_by_id = AsyncMock(
        side_effect=CheckpointUnavailableError("checkpoint query failed")
    )
    ws_manager = MagicMock(broadcast_to_task=AsyncMock())

    with _Patches(
        [
            patch("xagent.web.api.websocket._settle_resumed_task_lease", settle),
            patch(
                "xagent.web.api.websocket._restore_resumed_task_lease_to_prior_status",
                restore,
            ),
            patch(
                "xagent.web.api.websocket.stop_task_lease_heartbeat",
                new=AsyncMock(),
            ),
            patch("xagent.web.api.websocket.manager", ws_manager),
            patch(
                "xagent.web.api.websocket.background_task_manager.promote_resume_task"
            ),
            patch("xagent.web.tracking.task_tracker.TaskTracker", FakeTracker),
        ]
    ):
        await execute_resume_background(
            task_id=42,
            agent_service=agent_service,
            task_owner_user_id=1,
            expected_run_id="run-a",
            resolved_execution_scope=None,
            preacquired_lease=lease,
            preacquired_heartbeat_stop=heartbeat_stop,
            preacquired_heartbeat_task=heartbeat_task,
            preacquired_prior_status=TaskStatus.WAITING_FOR_USER,
        )

    settle.assert_not_called()
    restore.assert_called_once()
    assert restore.call_args.kwargs["status"] == TaskStatus.WAITING_FOR_USER
    assert not any(
        call.args[0].get("type") == "task_error"
        for call in ws_manager.broadcast_to_task.call_args_list
    )


def _fake_acquire_with_prior_status(lease: TaskLease, prior_status: TaskStatus):
    """Mirror ``_acquire_resume_task_lease``'s prior-status side channel."""

    def _acquire(
        task_id_arg: int,
        task_owner_user_id_arg: int | None,
        expected_run_id_arg: str | None,
        *,
        prior_status_out: list[Any] | None = None,
    ) -> TaskLease:
        if prior_status_out is not None:
            prior_status_out.append(prior_status)
        return lease

    return _acquire


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "read_error",
    [
        CheckpointUnavailableError("checkpoint store unavailable"),
        CheckpointAccessRefusedError("partition refused"),
    ],
)
async def test_resume_background_restores_prior_status_on_checkpoint_read_failure(
    read_error: Exception,
) -> None:
    """A retryable checkpoint read failure during resume (not a pool
    timeout) must restore the task to whatever it was before this attempt
    claimed the lease, not fall through to the generic terminal FAILED
    path. Unavailable and refused reads share the restore branch."""
    lease = TaskLease(task_id=42, runner_id="runner-a", run_id="run-a")
    settle = MagicMock()
    restore = MagicMock(return_value=True)
    mark_delivery = MagicMock()

    class FakeTracker:
        quota_interrupt_reason = None

        def __init__(self, *, task_id: int, **kwargs: Any) -> None:
            self.complete_tracking = AsyncMock()
            self.stop_periodic_updates = AsyncMock()

        async def start_tracking(self) -> None:
            return None

        async def interrupt_reason_for_quota(self) -> None:
            return None

    agent_service = MagicMock()
    agent_service.resume_execution_by_id = AsyncMock(side_effect=read_error)
    ws_manager = MagicMock(broadcast_to_task=AsyncMock())

    with _Patches(
        [
            patch(
                "xagent.web.api.websocket._acquire_resume_task_lease",
                side_effect=_fake_acquire_with_prior_status(
                    lease, TaskStatus.WAITING_FOR_USER
                ),
            ),
            patch("xagent.web.api.websocket._settle_resumed_task_lease", settle),
            patch(
                "xagent.web.api.websocket._restore_resumed_task_lease_to_prior_status",
                restore,
            ),
            patch(
                "xagent.web.api.websocket.mark_user_message_delivery_sync",
                mark_delivery,
            ),
            patch(
                "xagent.web.api.websocket.run_task_lease_heartbeat",
                new=AsyncMock(),
            ),
            patch(
                "xagent.web.api.websocket.stop_task_lease_heartbeat",
                new=AsyncMock(),
            ),
            patch("xagent.web.api.websocket.manager", ws_manager),
            patch(
                "xagent.web.api.websocket.background_task_manager.promote_resume_task"
            ),
            patch("xagent.web.tracking.task_tracker.TaskTracker", FakeTracker),
        ]
    ):
        await execute_resume_background(
            task_id=42,
            agent_service=agent_service,
            task_owner_user_id=1,
            delivery_turn_id="resume-turn",
            expected_run_id="run-a",
            resolved_execution_scope=None,
        )

    settle.assert_not_called()
    restore.assert_called_once()
    _, restore_kwargs = restore.call_args
    assert restore.call_args.args[0] == lease
    assert restore_kwargs["status"] == TaskStatus.WAITING_FOR_USER
    # Retryable, not a rejected/failed delivery: the client should retry
    # with a fresh id once the task is back to waiting.
    mark_delivery.assert_called_once()
    assert not any(
        call.args[0].get("type") == "task_error"
        for call in ws_manager.broadcast_to_task.call_args_list
    )


@pytest.mark.asyncio
async def test_resume_background_restore_broadcast_failure_does_not_affect_restore() -> (
    None
):
    """The corrective post-restore broadcast is best-effort: a failure there
    must not undo the already-committed restore or escape the call."""
    lease = TaskLease(task_id=42, runner_id="runner-a", run_id="run-a")
    settle = MagicMock()
    restore = MagicMock(return_value=True)
    mark_delivery = MagicMock()

    class FakeTracker:
        quota_interrupt_reason = None

        def __init__(self, *, task_id: int, **kwargs: Any) -> None:
            self.complete_tracking = AsyncMock()
            self.stop_periodic_updates = AsyncMock()

        async def start_tracking(self) -> None:
            return None

        async def interrupt_reason_for_quota(self) -> None:
            return None

    agent_service = MagicMock()
    agent_service.resume_execution_by_id = AsyncMock(
        side_effect=CheckpointUnavailableError("checkpoint query failed")
    )

    async def _broadcast(payload: dict, *_args: Any, **_kwargs: Any) -> None:
        if payload.get("type") in {"task_waiting_for_user", "task_paused"}:
            raise RuntimeError("broadcast down")

    ws_manager = MagicMock(broadcast_to_task=AsyncMock(side_effect=_broadcast))

    with _Patches(
        [
            patch(
                "xagent.web.api.websocket._acquire_resume_task_lease",
                side_effect=_fake_acquire_with_prior_status(
                    lease, TaskStatus.WAITING_FOR_USER
                ),
            ),
            patch("xagent.web.api.websocket._settle_resumed_task_lease", settle),
            patch(
                "xagent.web.api.websocket._restore_resumed_task_lease_to_prior_status",
                restore,
            ),
            patch(
                "xagent.web.api.websocket.mark_user_message_delivery_sync",
                mark_delivery,
            ),
            patch(
                "xagent.web.api.websocket.run_task_lease_heartbeat",
                new=AsyncMock(),
            ),
            patch(
                "xagent.web.api.websocket.stop_task_lease_heartbeat",
                new=AsyncMock(),
            ),
            patch("xagent.web.api.websocket.manager", ws_manager),
            patch(
                "xagent.web.api.websocket.background_task_manager.promote_resume_task"
            ),
            patch("xagent.web.tracking.task_tracker.TaskTracker", FakeTracker),
        ]
    ):
        # Must not raise even though the corrective broadcast fails.
        await execute_resume_background(
            task_id=42,
            agent_service=agent_service,
            task_owner_user_id=1,
            delivery_turn_id="resume-turn",
            expected_run_id="run-a",
            resolved_execution_scope=None,
        )

    settle.assert_not_called()
    restore.assert_called_once()


@pytest.mark.asyncio
async def test_resume_background_checkpoint_unavailable_from_pool_timeout_keeps_lease() -> (
    None
):
    """When the unavailable read is itself pool exhaustion, the existing
    pool-timeout recovery (retain the lease for TTL reaping) wins over the
    new restore-to-prior path -- ``is_database_pool_timeout`` must see
    through the ``CheckpointUnavailableError`` wrapper via its cause chain."""
    lease = TaskLease(task_id=42, runner_id="runner-a", run_id="run-a")
    settle = MagicMock()
    restore = MagicMock(return_value=True)

    def _raise_wrapped_pool_timeout(*args: Any, **kwargs: Any) -> Any:
        try:
            raise SQLAlchemyTimeoutError("pool exhausted")
        except SQLAlchemyTimeoutError as exc:
            raise CheckpointUnavailableError("checkpoint query failed") from exc

    class FakeTracker:
        quota_interrupt_reason = None

        def __init__(self, *, task_id: int, **kwargs: Any) -> None:
            self.complete_tracking = AsyncMock()
            self.stop_periodic_updates = AsyncMock()

        async def start_tracking(self) -> None:
            return None

        async def interrupt_reason_for_quota(self) -> None:
            return None

    agent_service = MagicMock()
    agent_service.resume_execution_by_id = AsyncMock(
        side_effect=_raise_wrapped_pool_timeout
    )
    ws_manager = MagicMock(broadcast_to_task=AsyncMock())

    with _Patches(
        [
            patch(
                "xagent.web.api.websocket._acquire_resume_task_lease",
                side_effect=_fake_acquire_with_prior_status(
                    lease, TaskStatus.WAITING_FOR_USER
                ),
            ),
            patch("xagent.web.api.websocket._settle_resumed_task_lease", settle),
            patch(
                "xagent.web.api.websocket._restore_resumed_task_lease_to_prior_status",
                restore,
            ),
            patch(
                "xagent.web.api.websocket.run_task_lease_heartbeat",
                new=AsyncMock(),
            ),
            patch(
                "xagent.web.api.websocket.stop_task_lease_heartbeat",
                new=AsyncMock(),
            ),
            patch("xagent.web.api.websocket.manager", ws_manager),
            patch(
                "xagent.web.api.websocket.background_task_manager.promote_resume_task"
            ),
            patch("xagent.web.tracking.task_tracker.TaskTracker", FakeTracker),
        ]
    ):
        await execute_resume_background(
            task_id=42,
            agent_service=agent_service,
            task_owner_user_id=1,
            expected_run_id="run-a",
            resolved_execution_scope=None,
        )

    settle.assert_not_called()
    restore.assert_not_called()
    assert not any(
        call.args[0].get("type") == "task_error"
        for call in ws_manager.broadcast_to_task.call_args_list
    )


@pytest.mark.asyncio
async def test_resume_background_marks_failed_on_checkpoint_corrupt() -> None:
    """Corrupt is a terminal state, not a retryable one -- it must fall
    through to the ordinary FAILED settlement, not the restore path."""
    lease = TaskLease(task_id=42, runner_id="runner-a", run_id="run-a")
    settle = MagicMock(return_value=True)
    restore = MagicMock()

    class FakeTracker:
        quota_interrupt_reason = None

        def __init__(self, *, task_id: int, **kwargs: Any) -> None:
            self.complete_tracking = AsyncMock()
            self.stop_periodic_updates = AsyncMock()

        async def start_tracking(self) -> None:
            return None

        async def interrupt_reason_for_quota(self) -> None:
            return None

    agent_service = MagicMock()
    agent_service.resume_execution_by_id = AsyncMock(
        side_effect=CheckpointCorruptError("all matching rows undecodable")
    )
    ws_manager = MagicMock(broadcast_to_task=AsyncMock())

    with _Patches(
        [
            patch(
                "xagent.web.api.websocket._acquire_resume_task_lease",
                side_effect=_fake_acquire_with_prior_status(
                    lease, TaskStatus.WAITING_FOR_USER
                ),
            ),
            patch("xagent.web.api.websocket._settle_resumed_task_lease", settle),
            patch(
                "xagent.web.api.websocket._restore_resumed_task_lease_to_prior_status",
                restore,
            ),
            patch(
                "xagent.web.api.websocket.run_task_lease_heartbeat",
                new=AsyncMock(),
            ),
            patch(
                "xagent.web.api.websocket.stop_task_lease_heartbeat",
                new=AsyncMock(),
            ),
            patch("xagent.web.api.websocket.manager", ws_manager),
            patch(
                "xagent.web.api.websocket.background_task_manager.promote_resume_task"
            ),
            patch("xagent.web.tracking.task_tracker.TaskTracker", FakeTracker),
        ]
    ):
        await execute_resume_background(
            task_id=42,
            agent_service=agent_service,
            task_owner_user_id=1,
            expected_run_id="run-a",
            resolved_execution_scope=None,
        )

    restore.assert_not_called()
    settle.assert_called_once()
    assert any(
        call.args[0].get("type") == "task_error"
        for call in ws_manager.broadcast_to_task.call_args_list
    )


@pytest.mark.asyncio
async def test_resume_tracker_pool_timeout_skips_result_and_lease_checkouts() -> None:
    """A final-usage timeout is the only checkout attempted during cleanup."""
    lease = TaskLease(task_id=42, runner_id="runner-a", run_id="run-a")
    finalize = MagicMock()
    settle = MagicMock()

    class FakeTracker:
        quota_interrupt_reason = None

        def __init__(self, *, task_id: int, **kwargs: Any) -> None:
            assert task_id == 42
            assert kwargs == {
                "expected_run_id": "run-a",
                "expected_runner_id": "runner-a",
            }

        async def start_tracking(self) -> None:
            return None

        async def complete_tracking(self) -> None:
            raise SQLAlchemyTimeoutError("pool exhausted")

        async def interrupt_reason_for_quota(self) -> None:
            return None

    agent_service = MagicMock()
    agent_service.resume_execution_by_id = AsyncMock(
        return_value={"status": "completed", "success": True, "output": "ok"}
    )

    with _Patches(
        [
            patch(
                "xagent.web.api.websocket._acquire_resume_task_lease",
                return_value=lease,
            ),
            patch("xagent.web.api.websocket._finalize_resumed_task", finalize),
            patch("xagent.web.api.websocket._settle_resumed_task_lease", settle),
            patch(
                "xagent.web.api.websocket.run_task_lease_heartbeat",
                new=AsyncMock(),
            ),
            patch(
                "xagent.web.api.websocket.stop_task_lease_heartbeat",
                new=AsyncMock(),
            ),
            patch(
                "xagent.web.api.websocket.manager",
                MagicMock(broadcast_to_task=AsyncMock()),
            ),
            patch(
                "xagent.web.api.websocket.background_task_manager.promote_resume_task"
            ),
            patch("xagent.web.tracking.task_tracker.TaskTracker", FakeTracker),
        ]
    ):
        await execute_resume_background(
            task_id=42,
            agent_service=agent_service,
            task_owner_user_id=1,
            resolved_execution_scope=None,
        )

    finalize.assert_not_called()
    settle.assert_not_called()


@pytest.mark.asyncio
async def test_resume_heartbeat_pool_timeout_skips_result_and_lease_checkouts() -> None:
    """Heartbeat exhaustion retains the lease instead of checking out again."""
    lease = TaskLease(task_id=42, runner_id="runner-a", run_id="run-a")
    finalize = MagicMock()
    settle = MagicMock()
    heartbeat_timeout = SQLAlchemyTimeoutError("heartbeat pool exhausted")

    class FakeTracker:
        quota_interrupt_reason = None

        def __init__(self, *, task_id: int, **kwargs: Any) -> None:
            assert task_id == 42
            assert kwargs == {
                "expected_run_id": "run-a",
                "expected_runner_id": "runner-a",
            }

        async def start_tracking(self) -> None:
            return None

        async def complete_tracking(self) -> None:
            return None

        async def interrupt_reason_for_quota(self) -> None:
            return None

    agent_service = MagicMock()
    agent_service.resume_execution_by_id = AsyncMock(
        return_value={"status": "completed", "success": True, "output": "ok"}
    )

    with _Patches(
        [
            patch(
                "xagent.web.api.websocket._acquire_resume_task_lease",
                return_value=lease,
            ),
            patch("xagent.web.api.websocket._finalize_resumed_task", finalize),
            patch("xagent.web.api.websocket._settle_resumed_task_lease", settle),
            patch(
                "xagent.web.api.websocket.run_task_lease_heartbeat",
                new=AsyncMock(),
            ),
            patch(
                "xagent.web.api.websocket.stop_task_lease_heartbeat",
                new=AsyncMock(
                    return_value=TaskLeaseHeartbeatOutcome(
                        pool_timeout=heartbeat_timeout
                    )
                ),
            ),
            patch(
                "xagent.web.api.websocket.manager",
                MagicMock(broadcast_to_task=AsyncMock()),
            ),
            patch(
                "xagent.web.api.websocket.background_task_manager.promote_resume_task"
            ),
            patch("xagent.web.tracking.task_tracker.TaskTracker", FakeTracker),
        ]
    ):
        await execute_resume_background(
            task_id=42,
            agent_service=agent_service,
            task_owner_user_id=1,
            resolved_execution_scope=None,
        )

    finalize.assert_not_called()
    settle.assert_not_called()


@pytest.mark.asyncio
async def test_resume_lease_loss_cancels_execution_without_stale_side_effects() -> None:
    lease = TaskLease(task_id=42, runner_id="runner-a", run_id="run-a")
    resume_started = asyncio.Event()
    resume_cancelled = asyncio.Event()
    finalize = MagicMock()
    settle = MagicMock()
    tracker_instances: list[Any] = []

    class FakeTracker:
        quota_interrupt_reason = None

        def __init__(self, *, task_id: int, **kwargs: Any) -> None:
            assert task_id == 42
            assert kwargs == {
                "expected_run_id": "run-a",
                "expected_runner_id": "runner-a",
            }
            self.complete_tracking = AsyncMock()
            self.stop_periodic_updates = AsyncMock()
            tracker_instances.append(self)

        async def start_tracking(self) -> None:
            return None

        async def interrupt_reason_for_quota(self) -> None:
            return None

    async def resume(_task_id: str) -> dict[str, Any]:
        resume_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            resume_cancelled.set()

    async def lose_lease(
        _lease: TaskLease,
        _stop_event: asyncio.Event,
    ) -> TaskLeaseHeartbeatOutcome:
        await resume_started.wait()
        return TaskLeaseHeartbeatOutcome(lease_lost=True)

    agent_service = MagicMock()
    agent_service.resume_execution_by_id = AsyncMock(side_effect=resume)
    ws_manager = MagicMock(broadcast_to_task=AsyncMock())

    with _Patches(
        [
            patch(
                "xagent.web.api.websocket._acquire_resume_task_lease",
                return_value=lease,
            ),
            patch(
                "xagent.web.api.websocket.run_task_lease_heartbeat",
                new=lose_lease,
            ),
            patch("xagent.web.api.websocket._finalize_resumed_task", finalize),
            patch(
                "xagent.web.api.websocket._settle_resumed_task_lease",
                settle,
            ),
            patch("xagent.web.api.websocket.manager", ws_manager),
            patch(
                "xagent.web.api.websocket.background_task_manager.promote_resume_task"
            ),
            patch("xagent.web.api.websocket.background_task_manager.cleanup_task"),
            patch("xagent.web.tracking.task_tracker.TaskTracker", FakeTracker),
        ]
    ):
        await asyncio.wait_for(
            execute_resume_background(
                task_id=42,
                agent_service=agent_service,
                task_owner_user_id=1,
                resolved_execution_scope=None,
            ),
            timeout=1,
        )

    assert resume_cancelled.is_set()
    agent_service.resume_execution_by_id.assert_awaited_once_with("42")
    finalize.assert_not_called()
    settle.assert_not_called()
    assert [
        call.args[0]["type"] for call in ws_manager.broadcast_to_task.call_args_list
    ] == ["task_resumed"]
    assert tracker_instances
    tracker_instances[0].complete_tracking.assert_not_awaited()
    tracker_instances[0].stop_periodic_updates.assert_awaited_once()


@pytest.mark.asyncio
async def test_resume_error_delivery_pool_timeout_skips_lease_checkout() -> None:
    """A timeout while recording failure must not trigger lease settlement."""
    lease = TaskLease(task_id=42, runner_id="runner-a", run_id="run-a")
    settle = MagicMock()
    snapshot = AsyncMock()
    mark_delivery = MagicMock(
        side_effect=SQLAlchemyTimeoutError("delivery pool exhausted")
    )

    class FakeTracker:
        quota_interrupt_reason = None

        def __init__(self, *, task_id: int, **kwargs: Any) -> None:
            assert task_id == 42
            assert kwargs == {
                "expected_run_id": "run-a",
                "expected_runner_id": "runner-a",
            }

        async def start_tracking(self) -> None:
            return None

        async def complete_tracking(self) -> None:
            return None

        async def interrupt_reason_for_quota(self) -> None:
            return None

    agent_service = MagicMock()
    agent_service.resume_execution_by_id = AsyncMock(
        side_effect=RuntimeError("resume failed")
    )

    with _Patches(
        [
            patch(
                "xagent.web.api.websocket._acquire_resume_task_lease",
                return_value=lease,
            ),
            patch(
                "xagent.web.api.websocket._settle_resumed_task_lease",
                settle,
            ),
            patch(
                "xagent.web.api.websocket.mark_user_message_delivery_sync",
                mark_delivery,
            ),
            patch(
                "xagent.web.api.websocket.task_execution_controller.snapshot",
                new=snapshot,
            ),
            patch(
                "xagent.web.api.websocket.run_task_lease_heartbeat",
                new=AsyncMock(),
            ),
            patch(
                "xagent.web.api.websocket.stop_task_lease_heartbeat",
                new=AsyncMock(),
            ),
            patch(
                "xagent.web.api.websocket.manager",
                MagicMock(broadcast_to_task=AsyncMock()),
            ),
            patch(
                "xagent.web.api.websocket.background_task_manager.promote_resume_task"
            ),
            patch("xagent.web.tracking.task_tracker.TaskTracker", FakeTracker),
        ]
    ):
        await execute_resume_background(
            task_id=42,
            agent_service=agent_service,
            task_owner_user_id=1,
            delivery_turn_id="resume-turn",
            expected_run_id="run-a",
            resolved_execution_scope=None,
        )

    mark_delivery.assert_called_once()
    snapshot.assert_not_awaited()
    settle.assert_not_called()


@pytest.mark.asyncio
async def test_resume_cancellation_drains_all_final_cleanup() -> None:
    """Caller cancellation must not skip heartbeat stop or exact settlement."""
    lease = TaskLease(task_id=42, runner_id="runner-a", run_id="run-a")
    completion_started = asyncio.Event()
    allow_completion = asyncio.Event()
    events: list[str] = []

    class FakeTracker:
        quota_interrupt_reason = None

        def __init__(self, *, task_id: int, **kwargs: Any) -> None:
            assert task_id == 42
            assert kwargs == {
                "expected_run_id": "run-a",
                "expected_runner_id": "runner-a",
            }

        async def start_tracking(self) -> None:
            return None

        async def complete_tracking(self) -> None:
            completion_started.set()
            try:
                await allow_completion.wait()
            except asyncio.CancelledError:
                # Match TaskTracker's contract: drain its inner worker before
                # returning the caller's cancellation.
                await allow_completion.wait()
                events.append("usage")
                raise
            events.append("usage")

        async def interrupt_reason_for_quota(self) -> None:
            return None

    async def stop_heartbeat(*_args: Any) -> None:
        events.append("heartbeat")

    def settle(*_args: Any, **_kwargs: Any) -> bool:
        events.append("lease")
        return True

    agent_service = MagicMock()
    agent_service.resume_execution_by_id = AsyncMock(
        side_effect=RuntimeError("run failed")
    )

    with _Patches(
        [
            patch(
                "xagent.web.api.websocket._acquire_resume_task_lease",
                return_value=lease,
            ),
            patch(
                "xagent.web.api.websocket._settle_resumed_task_lease",
                side_effect=settle,
            ),
            patch(
                "xagent.web.api.websocket.run_task_lease_heartbeat",
                new=AsyncMock(),
            ),
            patch(
                "xagent.web.api.websocket.stop_task_lease_heartbeat",
                side_effect=stop_heartbeat,
            ),
            patch(
                "xagent.web.api.websocket.manager",
                MagicMock(broadcast_to_task=AsyncMock()),
            ),
            patch(
                "xagent.web.api.websocket.background_task_manager.promote_resume_task"
            ),
            patch("xagent.web.tracking.task_tracker.TaskTracker", FakeTracker),
        ]
    ):
        execution = asyncio.create_task(
            execute_resume_background(
                task_id=42,
                agent_service=agent_service,
                task_owner_user_id=1,
                resolved_execution_scope=None,
            )
        )
        await asyncio.wait_for(completion_started.wait(), timeout=1)
        execution.cancel()
        await asyncio.sleep(0)
        allow_completion.set()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(execution, timeout=1)

    assert events == ["usage", "heartbeat", "lease"]
