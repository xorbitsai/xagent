"""Regression test: ``get_agent_for_task`` must not double-query Task or Agent
on the request session.

Invariants pinned here:

    * ``db.query(Task)``, ``db.query(Agent)``, and ``db.query(User)`` fire
      zero times on the request session for an existing task.
    * Existence, owner, and published-agent lookups live inside the off-loop
      snapshot loader (``task_setup_snapshot.load_task_setup_snapshot_sync``),
      which uses its own ``SessionLocal``.

Snapshot-internal query counts are not the responsibility of this
file; those are covered by ``test_task_setup_snapshot.py``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.web.api.chat import AgentServiceManager
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.llm_utils import AgentRuntimeFields
from xagent.web.services.task_setup_snapshot import (
    RuntimeUserFields,
    TaskSetupSnapshot,
    _TaskFields,
)


def _make_user() -> User:
    return User(
        id=1,
        username="dedup_test_user",
        password_hash="hash",
        is_admin=False,
    )


def _make_task(
    agent_id: int | None = None, status: TaskStatus = TaskStatus.PENDING
) -> Task:
    return Task(
        id=42,
        user_id=1,
        title="dedup test",
        description="dedup",
        status=status,
        agent_id=agent_id,
        agent_type="standard",
    )


def _make_agent() -> Agent:
    return Agent(
        id=7,
        user_id=1,
        name="dedup agent",
        instructions="be terse",
        status=AgentStatus.PUBLISHED,
        tool_categories=["basic"],
        knowledge_bases=[],
        skills=[],
        execution_mode="flash",
    )


class _QueryCounter:
    """Wraps ``db.query`` so we can count invocations per model class.

    SQLAlchemy ``Session.query`` returns a Query object; subsequent
    ``.filter().first()`` calls don't re-enter ``Session.query``, so a
    simple counter on the entry point is sufficient to detect double
    SELECTs against the same table.
    """

    def __init__(self) -> None:
        self.calls_by_model: Counter[type] = Counter()
        self._returns: dict[type, Any] = {}

    def set_first(self, model: type, value: Any) -> None:
        """Configure what ``.filter(...).first()`` will return for queries
        against the given model.
        """
        self._returns[model] = value

    def __call__(self, model: type) -> Any:
        self.calls_by_model[model] += 1
        result = MagicMock()
        result.filter = MagicMock(return_value=result)
        result.first = MagicMock(return_value=self._returns.get(model))
        result.all = MagicMock(return_value=[])
        result.order_by = MagicMock(return_value=result)
        return result


@pytest.mark.asyncio
async def test_existing_task_with_agent_dedups_task_and_agent_queries() -> None:
    """Existing task + agent path does not query the request session.

    Current invariant:
      - ``db.query(Task)`` happens zero times. The detached snapshot is the
        SSOT for existence, owner, and runtime configuration.
      - ``db.query(Agent)`` happens zero times. The snapshot loader
        owns the Agent lookup; there is no published-agent
        re-read on the request session.
    """
    manager = AgentServiceManager()
    user = _make_user()
    task = _make_task(agent_id=7, status=TaskStatus.PENDING)
    agent_row = _make_agent()

    counter = _QueryCounter()
    counter.set_first(Task, task)
    counter.set_first(Agent, agent_row)
    counter.set_first(User, user)

    db = MagicMock()
    db.query = counter

    # The snapshot loader uses its own SessionLocal (not the request
    # ``db``), so we replace it with a stubbed return that mirrors a
    # successful real load. The point of this test is the request
    # session's query counts; the snapshot's internal counts are
    # tested separately in ``test_task_setup_snapshot.py``.
    snapshot_stub = TaskSetupSnapshot(
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
            name="dedup agent",
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

    # Make ``task.status not in [RUNNING, PAUSED, WAITING_FOR_USER]`` so the
    # reconstruct branch is skipped entirely; we want to count the
    # ``normal creation`` path's queries, not reconstruct internal ones.
    # ``PENDING`` already satisfies that (see ``should_reconstruct`` check
    # in chat.py:720).

    # Stub the heavy work that ``get_agent_for_task`` performs after the
    # DB queries we care about, so the test focuses on query counts.
    # ``load_task_setup_snapshot_sync`` is patched at the import site
    # inside chat.py: get_agent_for_task imports it at module load, so
    # patching the chat module's symbol intercepts the call. Returning
    # the stub above lets the rest of the function consume snapshot
    # fields without ever opening a real SessionLocal.
    with (
        patch(
            "xagent.web.api.chat.load_task_setup_snapshot_sync",
            return_value=snapshot_stub,
        ) as snapshot_loader,
        patch.object(
            manager,
            "_load_persisted_conversation_history",
        ),
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
        patch(
            "xagent.web.api.chat.AgentService",
        ),
    ):
        try:
            await manager.get_agent_for_task(task_id=42, db=db, user=user)
        except Exception:
            # Some downstream stubs may raise during agent assembly --
            # query-count assertions below are what we're verifying, and
            # they're recorded before the failure point.
            pass

    assert counter.calls_by_model[Task] == 0, (
        f"Task queried {counter.calls_by_model[Task]} times on the request "
        "session -- expected 0. A non-zero count means task existence or "
        "runtime configuration bypassed the worker-owned snapshot."
    )
    assert counter.calls_by_model[Agent] == 0, (
        f"Agent queried {counter.calls_by_model[Agent]} times on the "
        "request session -- expected 0. The agent lookup belongs to "
        "``task_setup_snapshot`` (its own SessionLocal). A non-zero "
        "count here means something is bypassing the snapshot and "
        "re-doing the agent lookup on the loop thread."
    )
    assert counter.calls_by_model[User] == 0
    snapshot_loader.assert_called_once_with(42, None)


@pytest.mark.asyncio
async def test_owner_mismatch_reload_failure_detaches_the_live_owner_row() -> None:
    """Regression: a cached task whose owner mismatches gets evicted and
    rebuilt. Runtime identity is first resolved onto a live ORM `User`
    (the owner row, queried directly on the request session) as a
    fallback ahead of the richer snapshot load that would normally
    replace it with a detached RuntimeUserFields. If that richer load
    fails - anything other than HTTPException/TaskOwnerMismatchError/
    _AgentRuntimeSessionBoundaryError - the live `User` must not survive
    past this point: release_db_connection_if_clean's rollback expires
    every object the request session loaded, and any later attribute
    read (e.g. voice_from_runtime_user's `.preferences` access in
    create_default_tools) would otherwise force an implicit reload on
    the event loop instead of using the detached snapshot this whole
    mechanism exists to provide."""
    manager = AgentServiceManager()
    acting_user = _make_user()  # id=1
    owner_row = User(id=2, username="owner", password_hash="hash", is_admin=False)
    task = _make_task(status=TaskStatus.PENDING)
    task.user_id = 2

    # A cached entry for a DIFFERENT owner (id=1) is what triggers the
    # owner-mismatch eviction once the real owner (id=2) resolves below -
    # the eviction is what skips the *first* snapshot load, which is
    # only attempted up front for a task not already cached.
    manager._agents[42] = MagicMock()
    manager._agent_owner_ids[42] = 1

    def query_side_effect(model):
        result = MagicMock()
        result.filter = MagicMock(return_value=result)
        if model is Task.user_id:
            result.first = MagicMock(return_value=(2,))
        elif model is User:
            result.first = MagicMock(return_value=owner_row)
        else:
            result.first = MagicMock(return_value=None)
        return result

    db = MagicMock()
    db.query = MagicMock(side_effect=query_side_effect)

    captured_tool_kwargs: dict[str, Any] = {}

    async def fake_create_default_tools(*_args, **kwargs):
        captured_tool_kwargs.update(kwargs)
        return ([], MagicMock())

    with (
        patch(
            "xagent.web.api.chat.load_task_setup_snapshot_sync",
            side_effect=RuntimeError("snapshot load failed"),
        ),
        patch.object(manager, "_load_persisted_conversation_history"),
        patch("xagent.web.api.chat.create_task_tracer", return_value=MagicMock()),
        patch(
            "xagent.web.api.chat.create_default_tools",
            new=fake_create_default_tools,
        ),
        patch("xagent.web.sandbox_manager.get_sandbox_manager", return_value=None),
        patch("xagent.web.api.chat.AgentService"),
        patch(
            "xagent.web.api.chat.resolve_execution_scope",
            return_value=None,
        ),
    ):
        try:
            await manager.get_agent_for_task(task_id=42, db=db, user=acting_user)
        except Exception:
            # The assertion below only needs create_default_tools to have
            # already been called with the runtime identity this test
            # verifies; a later stub raising doesn't invalidate that.
            pass

    assert "user" in captured_tool_kwargs, (
        "create_default_tools was never called - the owner-mismatch "
        "eviction and reload-failure path this test targets wasn't "
        "reached; fix the test setup, not the assertion below."
    )
    assert not isinstance(captured_tool_kwargs["user"], User), (
        "a live ORM User reached create_default_tools after the richer "
        "snapshot load failed - it must be detached to RuntimeUserFields "
        "first, or a later attribute read forces an implicit reload once "
        "the request session's connection is released"
    )
    assert isinstance(captured_tool_kwargs["user"], RuntimeUserFields)
    assert captured_tool_kwargs["user"].id == 2
