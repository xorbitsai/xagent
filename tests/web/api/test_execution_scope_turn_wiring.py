"""Pin that the orchestrator activates the execution scope on every turn.

Slice 1 of #757 wires ``turn_execution_scope`` at the same places the acting
user is resolved (``UserContext``): ``execute_task_background`` for normal
turns and ``execute_resume_background`` for resumed turns. These tests use a
fake resolver to pin that:

* the resolver is called with the turn's ``task_id`` (as str),
* the resolved scope is active inside the turn's execution context (visible
  to the agent build and the agent run),
* the resumed turn re-resolves the scope (restart/resume correctness), and
* with no resolver registered the turn runs unscoped, exactly as today.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.core.execution_scope import (
    ExecutionScope,
    get_execution_scope,
    set_execution_scope_resolver,
)
from xagent.web.api.websocket import (
    execute_resume_background,
    execute_task_background,
)
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User


@pytest.fixture(autouse=True)
def _clear_resolver():
    set_execution_scope_resolver(None)
    yield
    set_execution_scope_resolver(None)


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
    """Stub ``execute_task_background``'s surroundings (DB sessions, file
    normalization, transcript persistence) so the tests observe only the
    scope activation around the agent build/run."""

    def _fresh_db_gen():
        yield db

    return [
        patch(
            "xagent.web.models.database.get_db",
            side_effect=lambda: _fresh_db_gen(),
        ),
        patch(
            "xagent.web.api.websocket.background_task_manager.wait_for_previous",
            new=AsyncMock(),
        ),
        patch("xagent.web.api.websocket._register_uploaded_files_for_agent"),
        patch(
            "xagent.web.api.websocket._normalize_task_file_outputs",
            return_value=([], {}),
        ),
        patch(
            "xagent.web.api.websocket._rewrite_file_links_to_file_id",
            side_effect=lambda s, _m: s,
        ),
        patch(
            "xagent.web.services.task_execution_context_service.load_task_execution_recovery_state",
            new=AsyncMock(return_value={"messages": [], "skill_context": None}),
        ),
        patch("xagent.web.services.chat_history_service.persist_assistant_message"),
        patch(
            "xagent.web.services.chat_history_service.load_task_transcript",
            return_value=[],
        ),
        patch("xagent.web.api.websocket.sync_workforce_run_status", return_value=False),
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

    set_execution_scope_resolver(resolver)

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

    set_execution_scope_resolver(resolver)

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
            patch("xagent.web.api.websocket.acquire_task_lease", return_value=object()),
            patch(
                "xagent.web.api.websocket.run_task_lease_heartbeat",
                side_effect=_heartbeat,
            ),
            patch(
                "xagent.web.api.websocket.stop_task_lease_heartbeat", new=AsyncMock()
            ),
            patch(
                "xagent.web.api.websocket.release_current_runner_task_lease_with_workforce_sync",
                return_value=True,
            ),
            patch(
                "xagent.web.api.websocket.sync_workforce_run_status",
                return_value=False,
            ),
            patch(
                "xagent.web.api.websocket._normalize_task_file_outputs",
                return_value=([], {}),
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
