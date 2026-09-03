"""Pin where a pause/resume agent-cache-miss build materializes real bytes.

Pause and resume are off-turn control operations: both resolve the task's
execution scope through ``resolve_execution_scope_off_turn``
(``xagent.core.execution_scope``), which downgrades a resolver-vs-snapshot
namespace disagreement to the resolver's own authoritative answer instead of
raising, so a control operation on a running task stays available while the
scope is in dispute. The handlers then pass that scope to
``get_agent_manager().get_agent_for_task(..., resolved_execution_scope=...)``.

On an agent-cache HIT this only locates an already-running agent -- nothing
new is built. On a MISS, ``AgentServiceManager._get_agent_for_task_unlocked``
builds a fresh ``AgentService``, which reads ``scope.workspace_segments`` and
constructs a real ``TaskWorkspace`` whose constructor creates the workspace
directory tree on disk. Every other test in this area mocks
``get_agent_for_task`` wholesale, so that build branch never runs and nothing
pins where those bytes land.

These tests drive the real ``AgentServiceManager.get_agent_for_task`` build
branch (a fresh manager instance, empty cache, real ``AgentService`` and real
``TaskWorkspace``) through the actual ``_handle_pause_task_unserialized`` /
``_handle_resume_task_unserialized`` handlers, and assert on the real
filesystem path the workspace lands under. Only what is genuinely external is
mocked: the sandbox/container runtime (``get_sandbox_manager``) and tool
construction (``create_default_tools``, which would otherwise reach out to
configured LLM/KB/MCP providers).
"""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.shared.execution_scope import register_scope_resolver
from xagent.config import get_uploads_dir
from xagent.core.execution_scope import (
    DeferToSnapshot,
    ExecutionScope,
    ExecutionScopeAbstentionMismatchError,
    scope_fingerprint,
    set_execution_scope_snapshot_loader,
)
from xagent.core.workspace import scoped_user_root
from xagent.web.api import chat as chat_api
from xagent.web.api import websocket as websocket_api
from xagent.web.api.chat import AgentServiceManager
from xagent.web.models.task import TaskStatus
from xagent.web.services import task_setup_snapshot as snapshot_module
from xagent.web.services.task_setup_snapshot import (
    RuntimeUserFields,
    TaskSetupSnapshot,
    _TaskFields,
)

TASK_ID = 501
OWNER_ID = 7


def _snapshot(*, status: TaskStatus, run_id: str) -> TaskSetupSnapshot:
    control_state = "paused" if status == TaskStatus.PAUSED else "running"
    return TaskSetupSnapshot(
        task=_TaskFields(
            id=TASK_ID,
            user_id=OWNER_ID,
            status=status,
            agent_id=None,
            agent_config=None,
            model_name=None,
            compact_model_name=None,
            execution_mode="flash",
            agent_type="standard",
            run_id=run_id,
            state_version=1,
            control_state=control_state,
        ),
        runtime_user=RuntimeUserFields(id=OWNER_ID, is_admin=False),
        has_reconstructable_history=False,
        task_pattern="single_call",
        task_llm=None,
        task_fast_llm=None,
        task_vision_llm=None,
        task_compact_llm=None,
        agent=None,
        agent_config=None,
        excluded_agent_id=None,
    )


def _enter_build_only_patches(stack: ExitStack, manager: AgentServiceManager) -> None:
    """Mock only the genuinely external boundaries of an agent build.

    ``create_default_tools`` returns ``(tools=[], tool_config=None)``: an
    empty tool list plus a ``None`` tool config keeps ``AgentService.__init__``
    on its real ``enable_workspace`` branch (``self._setup_workspace()``)
    instead of the ``tool_config._workspace_config`` branch, so the
    constructor builds a genuine ``TaskWorkspace`` from ``workspace_base_dir``
    -- the exact object whose constructor creates the directory tree on disk.
    """
    manager._default_llm = MagicMock()
    stack.enter_context(
        patch("xagent.web.api.chat.create_task_tracer", return_value=MagicMock())
    )
    stack.enter_context(
        patch(
            "xagent.web.api.chat.create_default_tools",
            new=AsyncMock(return_value=([], None)),
        )
    )
    stack.enter_context(
        patch("xagent.web.sandbox_manager.get_sandbox_manager", return_value=None)
    )


def _connection_manager() -> MagicMock:
    manager = MagicMock()
    manager.send_personal_message = AsyncMock()
    manager.broadcast_to_task = AsyncMock()
    return manager


def _workspace_dir(owner_id: int, segments: tuple[str, ...]):
    return (
        scoped_user_root(get_uploads_dir(), owner_id, segments) / f"web_task_{TASK_ID}"
    )


@pytest.mark.asyncio
async def test_pause_cache_miss_builds_under_resolver_namespace_not_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Resolver/snapshot mismatch, agent-cache MISS: pause builds the agent
    and materializes the workspace tree under the RESOLVER's namespace, never
    the snapshot's -- ``resolve_execution_scope_off_turn`` downgrades the
    mismatch to the resolver's own answer instead of raising, and that
    downgraded scope is what reaches ``AgentService``'s workspace
    construction on the cache-miss build path."""
    uploads_root = tmp_path / "uploads"
    uploads_root.mkdir()
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads_root))

    register_scope_resolver(
        lambda task_id: ExecutionScope(
            sandbox_key_suffix="from-resolver", workspace_segments=("from-resolver",)
        )
    )
    set_execution_scope_snapshot_loader(
        lambda task_id: ExecutionScope(
            sandbox_key_suffix="from-snapshot", workspace_segments=("from-snapshot",)
        )
    )

    snapshot = _snapshot(status=TaskStatus.RUNNING, run_id="run-1")
    manager = AgentServiceManager()
    connection_manager = _connection_manager()

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                snapshot_module, "load_task_setup_snapshot_sync", return_value=snapshot
            )
        )
        stack.enter_context(
            patch.object(chat_api, "get_agent_manager", lambda: manager)
        )
        stack.enter_context(patch.object(websocket_api, "manager", connection_manager))
        stack.enter_context(
            patch(
                "xagent.core.agent.service.AgentService.pause_execution",
                new=AsyncMock(return_value=True),
            )
        )
        stack.enter_context(
            patch.object(
                websocket_api, "_apply_pause_requested_isolated", lambda *a, **k: True
            )
        )
        _enter_build_only_patches(stack, manager)
        try:
            await websocket_api._handle_pause_task_unserialized(
                MagicMock(),
                TASK_ID,
                {"user": SimpleNamespace(id=OWNER_ID, is_admin=False)},
            )
        finally:
            websocket_api._clear_task_pause_accepted(TASK_ID)

    resolver_workspace = _workspace_dir(OWNER_ID, ("from-resolver",))
    snapshot_workspace = _workspace_dir(OWNER_ID, ("from-snapshot",))
    assert resolver_workspace.is_dir()
    assert (resolver_workspace / "input").is_dir()
    assert (resolver_workspace / "output").is_dir()
    assert (resolver_workspace / "temp").is_dir()
    assert not snapshot_workspace.exists()
    assert manager._agent_scope_fingerprints.get(TASK_ID) == scope_fingerprint(
        ExecutionScope(
            sandbox_key_suffix="from-resolver", workspace_segments=("from-resolver",)
        )
    )


@pytest.mark.asyncio
async def test_resume_cache_miss_builds_under_resolver_namespace_not_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Same mismatch, through resume: the off-turn build lands under the
    resolver's namespace, and the scheduled resumed turn (mocked here, since
    it is a different consumer -- see
    ``test_execution_scope_turn_wiring.test_resume_survives_a_scope_authority_mismatch``)
    is never reached before the assertion."""
    uploads_root = tmp_path / "uploads"
    uploads_root.mkdir()
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads_root))

    register_scope_resolver(
        lambda task_id: ExecutionScope(
            sandbox_key_suffix="from-resolver", workspace_segments=("from-resolver",)
        )
    )
    set_execution_scope_snapshot_loader(
        lambda task_id: ExecutionScope(
            sandbox_key_suffix="from-snapshot", workspace_segments=("from-snapshot",)
        )
    )

    snapshot = _snapshot(status=TaskStatus.PAUSED, run_id="run-1")
    manager = AgentServiceManager()
    connection_manager = _connection_manager()
    background_manager = MagicMock()
    background_manager.running_tasks = {}
    background_manager.resume_admission_state.return_value = None
    background_manager.try_reserve_resume.return_value = (
        websocket_api.ResumeReservationOutcome.RESERVED
    )
    transition = AsyncMock(
        return_value=SimpleNamespace(run_id="run-1", status=TaskStatus.PAUSED)
    )
    resume_scheduled = asyncio.Event()

    async def _stub_execute_resume_background(**kwargs: object) -> None:
        resume_scheduled.set()

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                snapshot_module, "load_task_setup_snapshot_sync", return_value=snapshot
            )
        )
        stack.enter_context(
            patch.object(chat_api, "get_agent_manager", lambda: manager)
        )
        stack.enter_context(patch.object(websocket_api, "manager", connection_manager))
        stack.enter_context(
            patch.object(
                websocket_api.task_execution_controller, "transition", new=transition
            )
        )
        stack.enter_context(
            patch.object(websocket_api, "background_task_manager", background_manager)
        )
        # The handler asks the DB whether another process still holds a live
        # lease before it schedules; these suites drive the handler without a
        # task row, so answer "no foreign owner" explicitly.
        stack.enter_context(
            patch.object(
                websocket_api, "task_has_live_foreign_runner", return_value=False
            )
        )
        stack.enter_context(
            patch.object(
                websocket_api,
                "execute_resume_background",
                side_effect=_stub_execute_resume_background,
            )
        )
        _enter_build_only_patches(stack, manager)
        await websocket_api._handle_resume_task_unserialized(
            MagicMock(),
            TASK_ID,
            {"user": SimpleNamespace(id=OWNER_ID, is_admin=False)},
        )
        # The handler fires the scheduled turn as a background asyncio task
        # and returns without awaiting it; give the loop one chance to run it
        # before asserting the stub observed the call.
        await asyncio.wait_for(resume_scheduled.wait(), timeout=1)

    resolver_workspace = _workspace_dir(OWNER_ID, ("from-resolver",))
    snapshot_workspace = _workspace_dir(OWNER_ID, ("from-snapshot",))
    assert resolver_workspace.is_dir()
    assert (resolver_workspace / "input").is_dir()
    assert not snapshot_workspace.exists()
    assert resume_scheduled.is_set()
    assert manager._agent_scope_fingerprints.get(TASK_ID) == scope_fingerprint(
        ExecutionScope(
            sandbox_key_suffix="from-resolver", workspace_segments=("from-resolver",)
        )
    )


@pytest.mark.asyncio
async def test_pause_abstention_mismatch_fails_closed_with_no_workspace_residue(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Resolver ABSTAINS (``DeferToSnapshot``) and the snapshot WIDENS the
    abstention's fallback: this is the
    ``ExecutionScopeAbstentionMismatchError`` path, which
    ``resolve_execution_scope_off_turn`` re-raises instead of downgrading --
    an abstention never produced an authoritative answer to downgrade to.
    The pause handler must fail closed: no agent is built, and neither
    candidate namespace's directory is created."""
    uploads_root = tmp_path / "uploads"
    uploads_root.mkdir()
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads_root))

    register_scope_resolver(lambda task_id: DeferToSnapshot(fallback=ExecutionScope()))
    set_execution_scope_snapshot_loader(
        lambda task_id: ExecutionScope(workspace_segments=("wider",))
    )

    snapshot = _snapshot(status=TaskStatus.RUNNING, run_id="run-1")
    manager = AgentServiceManager()
    connection_manager = _connection_manager()
    agent_service_spy = MagicMock()

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                snapshot_module, "load_task_setup_snapshot_sync", return_value=snapshot
            )
        )
        stack.enter_context(
            patch.object(chat_api, "get_agent_manager", lambda: manager)
        )
        stack.enter_context(patch.object(websocket_api, "manager", connection_manager))
        stack.enter_context(
            patch("xagent.web.api.chat.AgentService", agent_service_spy)
        )
        _enter_build_only_patches(stack, manager)
        with pytest.raises(ExecutionScopeAbstentionMismatchError):
            await websocket_api._handle_pause_task_unserialized(
                MagicMock(),
                TASK_ID,
                {"user": SimpleNamespace(id=OWNER_ID, is_admin=False)},
            )

    agent_service_spy.assert_not_called()
    assert manager._agents.get(TASK_ID) is None
    fallback_workspace = _workspace_dir(OWNER_ID, ())
    widened_workspace = _workspace_dir(OWNER_ID, ("wider",))
    assert not fallback_workspace.exists()
    assert not widened_workspace.exists()
    # Nothing under the uploads root at all: the refusal leaves no residue.
    assert list(uploads_root.iterdir()) == []


@pytest.mark.asyncio
async def test_pause_cache_hit_returns_running_agent_without_new_namespace(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Agent-cache HIT: the already-running agent is returned untouched, and
    no build (hence no new namespace / directory) happens at all -- the
    off-turn scope resolution here only ever locates it."""
    uploads_root = tmp_path / "uploads"
    uploads_root.mkdir()
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads_root))

    scope = ExecutionScope(
        sandbox_key_suffix="tenant-a", workspace_segments=("tenant-a",)
    )
    register_scope_resolver(lambda task_id: scope)

    snapshot = _snapshot(status=TaskStatus.RUNNING, run_id="run-1")
    manager = AgentServiceManager()
    cached_agent = MagicMock()
    cached_agent.pause_execution = AsyncMock(return_value=True)
    manager._agents[TASK_ID] = cached_agent
    manager._agent_owner_ids[TASK_ID] = OWNER_ID
    manager._agent_scope_fingerprints[TASK_ID] = scope_fingerprint(scope)
    connection_manager = _connection_manager()
    agent_service_spy = MagicMock()
    create_default_tools_spy = AsyncMock(return_value=([], None))

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                snapshot_module, "load_task_setup_snapshot_sync", return_value=snapshot
            )
        )
        stack.enter_context(
            patch.object(chat_api, "get_agent_manager", lambda: manager)
        )
        stack.enter_context(patch.object(websocket_api, "manager", connection_manager))
        stack.enter_context(
            patch.object(
                websocket_api, "_apply_pause_requested_isolated", lambda *a, **k: True
            )
        )
        stack.enter_context(
            patch("xagent.web.api.chat.AgentService", agent_service_spy)
        )
        stack.enter_context(
            patch(
                "xagent.web.api.chat.create_default_tools", new=create_default_tools_spy
            )
        )
        stack.enter_context(
            patch("xagent.web.sandbox_manager.get_sandbox_manager", return_value=None)
        )
        try:
            await websocket_api._handle_pause_task_unserialized(
                MagicMock(),
                TASK_ID,
                {"user": SimpleNamespace(id=OWNER_ID, is_admin=False)},
            )
        finally:
            websocket_api._clear_task_pause_accepted(TASK_ID)

    cached_agent.pause_execution.assert_awaited_once_with()
    agent_service_spy.assert_not_called()
    create_default_tools_spy.assert_not_called()
    assert manager._agents[TASK_ID] is cached_agent
    # No workspace tree materialized anywhere under the uploads root.
    assert list(uploads_root.iterdir()) == []
