"""Test that ``get_agent_for_task`` reuses a caller-supplied snapshot
instead of spinning up its own worker thread.

Background:
    ``get_agent_for_task`` runs ``await asyncio.to_thread(
    load_task_setup_snapshot_sync, ...)`` to push the Task / Agent /
    LLM DB block off the main event loop. ``_schedule_bg._runner``
    also runs the snapshot loader off-loop before calling
    ``execute_task_background`` -- so ``get_agent_for_task`` must
    accept a caller-supplied snapshot and skip its own in-method
    thread call, otherwise both layers would spawn redundant worker
    threads and re-read the same rows.

What this test pins:

    * When the caller supplies ``task_setup_snapshot``,
      ``load_task_setup_snapshot_sync`` is **not** invoked from
      ``get_agent_for_task`` -- the passthrough is honored. This is
      the load-bearing test against the "two layers each spin
      their own worker" regression.
    * When the caller passes ``task_setup_snapshot=None`` (WS / chat
      single-task / tests that don't have the upstream snapshot),
      the in-method thread call still runs exactly once.
"""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from time import monotonic
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from xagent.core.execution_scope import scope_fingerprint
from xagent.web.api.chat import AgentServiceManager
from xagent.web.models.agent import AgentStatus
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.llm_utils import AgentRuntimeFields
from xagent.web.services.task_setup_snapshot import (
    RuntimeUserFields,
    TaskOwnerMismatchError,
    TaskSetupSnapshot,
    _TaskFields,
)


def _make_user() -> User:
    return User(id=1, username="snap-pt-user", password_hash="hash", is_admin=False)


def _build_snapshot() -> TaskSetupSnapshot:
    return TaskSetupSnapshot(
        task=_TaskFields(
            id=42,
            user_id=1,
            status=TaskStatus.PENDING,
            agent_id=7,
            agent_config=None,
            model_name=None,
            compact_model_name=None,
            execution_mode="flash",
            agent_type="standard",
        ),
        runtime_user=RuntimeUserFields(id=1, is_admin=False),
        has_reconstructable_history=False,
        task_pattern="single_call",
        task_llm=None,
        task_fast_llm=None,
        task_vision_llm=None,
        task_compact_llm=None,
        agent=AgentRuntimeFields(
            id=7,
            name="snap-agent",
            status=AgentStatus.PUBLISHED,
            instructions="be terse",
        ),
        agent_config={
            "llms": (None, None, None, None),
            "execution_mode": "flash",
            "instructions": "be terse",
            "skills": [],
            "knowledge_bases": [],
            "tool_categories": ["basic"],
        },
        excluded_agent_id=7,
    )


def _build_db_mock(task_row: Task) -> MagicMock:
    """Mock request session used by legacy-path tests."""
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = task_row
    return db


def _common_patches(manager: AgentServiceManager) -> list[Any]:
    return [
        patch.object(manager, "_load_persisted_conversation_history"),
        patch.object(manager, "_load_persisted_execution_context", new=AsyncMock()),
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
async def test_caller_supplied_snapshot_skips_internal_to_thread() -> None:
    """The passthrough contract: if the caller already loaded a
    snapshot, ``get_agent_for_task`` must NOT call
    ``load_task_setup_snapshot_sync`` again. A regression that
    re-spins the worker thread would silently double the snapshot
    load cost per turn.
    """
    manager = AgentServiceManager()
    user = _make_user()
    snapshot = _build_snapshot()

    task_row = Task(
        id=42,
        user_id=1,
        title="snap-pt task",
        description="snap-pt",
        status=TaskStatus.PENDING,
        agent_id=7,
        agent_type="standard",
    )
    db = _build_db_mock(task_row)

    with ExitStack() as stack:
        loader_mock = stack.enter_context(
            patch("xagent.web.api.chat.load_task_setup_snapshot_sync")
        )
        for p in _common_patches(manager):
            stack.enter_context(p)
        try:
            await manager.get_agent_for_task(
                task_id=42, db=db, user=user, task_setup_snapshot=snapshot
            )
        except Exception:
            # Downstream stubs (AgentService) may raise after the
            # snapshot consumption -- the call-count assertion below
            # is recorded before that point.
            pass

    loader_mock.assert_not_called()
    db.query.assert_not_called()


@pytest.mark.asyncio
async def test_snapshot_startup_stays_responsive_with_exhausted_pool(tmp_path) -> None:
    """A ready snapshot must reach non-DB setup without waiting on QueuePool."""

    engine = create_engine(
        f"sqlite:///{tmp_path / 'startup-pool.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.25,
    )
    SessionLocal = sessionmaker(bind=engine)
    held_connection = engine.connect()
    db = SessionLocal()
    manager = AgentServiceManager()
    manager._default_llm = MagicMock()
    snapshot = _build_snapshot()
    sandbox_entered = asyncio.Event()
    release_sandbox = asyncio.Event()
    ticker_stop = asyncio.Event()
    ticks = 0

    async def blocked_sandbox(**_kwargs: Any) -> None:
        sandbox_entered.set()
        await release_sandbox.wait()
        raise RuntimeError("stop after the startup boundary")

    async def ticker() -> None:
        nonlocal ticks
        while not ticker_stop.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    with ExitStack() as stack:
        for p in _common_patches(manager):
            stack.enter_context(p)
        stack.enter_context(
            patch.object(manager, "_get_or_create_task_sandbox", blocked_sandbox)
        )
        started = monotonic()
        build_task = asyncio.create_task(
            manager.get_agent_for_task(
                task_id=42,
                db=db,
                user=None,
                task_setup_snapshot=snapshot,
                task_owner_user_id=1,
                resolved_execution_scope=None,
            )
        )
        ticker_task = asyncio.create_task(ticker())
        try:
            await asyncio.wait_for(sandbox_entered.wait(), timeout=0.4)
            elapsed = monotonic() - started
            await asyncio.sleep(0.04)
            assert elapsed < 0.15, (
                "Snapshot startup waited for the exhausted request pool; "
                f"sandbox boundary reached after {elapsed:.3f}s."
            )
            assert ticks >= 3, f"Event-loop ticker advanced only {ticks} time(s)."
        finally:
            release_sandbox.set()
            ticker_stop.set()
            results = await asyncio.gather(
                build_task,
                ticker_task,
                return_exceptions=True,
            )
            db.close()
            held_connection.close()
            engine.dispose()

    assert isinstance(results[0], RuntimeError)


@pytest.mark.asyncio
async def test_no_snapshot_falls_back_to_internal_to_thread() -> None:
    """The WS fallback contract: when no snapshot is supplied, the
    Step-3 in-method ``to_thread`` call still fires exactly once."""
    manager = AgentServiceManager()
    user = _make_user()
    snapshot = _build_snapshot()

    task_row = Task(
        id=42,
        user_id=1,
        title="snap-pt task",
        description="snap-pt",
        status=TaskStatus.PENDING,
        agent_id=7,
        agent_type="standard",
    )
    db = _build_db_mock(task_row)

    with ExitStack() as stack:
        loader_mock = stack.enter_context(
            patch(
                "xagent.web.api.chat.load_task_setup_snapshot_sync",
                return_value=snapshot,
            )
        )
        for p in _common_patches(manager):
            stack.enter_context(p)
        try:
            await manager.get_agent_for_task(
                task_id=42, db=db, user=user, task_setup_snapshot=None
            )
        except Exception:
            pass

    assert loader_mock.call_count == 1, (
        f"Expected exactly 1 call to load_task_setup_snapshot_sync on the "
        f"fallback path, got {loader_mock.call_count}. A regression here "
        "means either the in-method fallback was removed (breaking WS / "
        "non-passthrough callers) or the snapshot is being loaded twice."
    )


@pytest.mark.asyncio
async def test_explicit_unscoped_result_skips_internal_scope_resolver() -> None:
    """Explicit ``None`` is a resolved unscoped turn, not an instruction to
    resolve again inside the cache owner.
    """
    manager = AgentServiceManager()
    cached_agent = MagicMock()
    manager._agents[42] = cached_agent
    manager._agent_owner_ids[42] = 1
    manager._agent_scope_fingerprints[42] = scope_fingerprint(None)

    with patch("xagent.web.api.chat.resolve_execution_scope") as resolver_mock:
        resolved = await manager.get_agent_for_task(
            task_id=42,
            user=_make_user(),
            task_owner_user_id=1,
            resolved_execution_scope=None,
        )

    assert resolved is cached_agent
    resolver_mock.assert_not_called()


@pytest.mark.asyncio
async def test_wrong_owner_eviction_cleans_up_workspace() -> None:
    """Evicting a cached AgentService built for a different owner must clean
    up that wrong-owner workspace before dropping the instance -- otherwise a
    workspace created under the wrong owner is left on disk. The correct
    owner's workspace lives at a different user-scoped path and is untouched.
    """
    manager = AgentServiceManager()
    snapshot = _build_snapshot()  # owner == 1

    task_row = Task(
        id=42,
        user_id=1,
        title="snap-pt task",
        description="snap-pt",
        status=TaskStatus.PENDING,
        agent_id=7,
        agent_type="standard",
    )
    db = _build_db_mock(task_row)

    # Seed a cached instance owned by a DIFFERENT user (2).
    stale_agent = MagicMock()
    manager._agents[42] = stale_agent
    manager._agent_owner_ids[42] = 2

    with ExitStack() as stack:
        for p in _common_patches(manager):
            stack.enter_context(p)
        try:
            # Requesting owner 1 != cached owner 2 -> eviction path.
            await manager.get_agent_for_task(
                task_id=42, db=db, task_setup_snapshot=snapshot, task_owner_user_id=1
            )
        except Exception:
            # Downstream stubs may raise after eviction; the eviction
            # cleanup assertion below is what this test pins.
            pass

    stale_agent.cleanup_workspace.assert_called_once()
    # The wrong-owner instance is gone (rebuilt or dropped), not silently kept.
    assert manager._agents.get(42) is not stale_agent


@pytest.mark.asyncio
async def test_passthrough_snapshot_owner_mismatch_raises() -> None:
    """A caller-supplied snapshot whose owner disagrees with the
    requested ``task_owner_user_id`` is an identity fault. The loader's
    owner guard is bypassed on the passthrough branch, so
    ``get_agent_for_task`` must re-assert it and raise
    ``TaskOwnerMismatchError`` -- never swallow it into the default-LLM
    fallback and build the runtime as the wrong user.
    """
    manager = AgentServiceManager()
    user = _make_user()  # id=1
    snapshot = _build_snapshot()  # snapshot.task.user_id == 1

    task_row = Task(
        id=42,
        user_id=1,
        title="snap-pt task",
        description="snap-pt",
        status=TaskStatus.PENDING,
        agent_id=7,
        agent_type="standard",
    )
    db = _build_db_mock(task_row)

    with ExitStack() as stack:
        for p in _common_patches(manager):
            stack.enter_context(p)
        with pytest.raises(TaskOwnerMismatchError):
            await manager.get_agent_for_task(
                task_id=42,
                db=db,
                user=user,
                task_setup_snapshot=snapshot,
                task_owner_user_id=999,  # != snapshot owner (1)
            )
