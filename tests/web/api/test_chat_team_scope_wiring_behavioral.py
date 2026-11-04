"""Behavioral pins for the team-scope values ``chat.py`` forwards into
``create_default_tools``, covering what the source-level guard
(``test_chat_team_scope_wiring_static.py``) cannot reach: that guard only
proves no derivation point passes a literal ``None``, so it cannot catch
one that derives the *wrong* non-``None`` value (e.g. the running user's
id instead of the agent creator's), or one that hardcodes an empty list
where the agent's own declaration belongs.

These tests drive the real methods (``AgentServiceManager._build_tools_for_task``
and ``AgentServiceManager.get_agent_for_task``) and assert the values
forwarded to ``create_default_tools`` equal the agent's own fields -- not
merely "not None" -- so a swap to the runner's id, or a hardcoded ``[]``,
fails the assertion. All three derivation points are covered: the two
snapshot branches, driven with a fake ``TaskSetupSnapshot``, and
``_build_tools_for_task``'s snapshot-less branch, driven with a fake
session that returns a live ``Agent`` row.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.agent_service_manager import AgentServiceManager
from xagent.web.services.llm_utils import AgentRuntimeFields
from xagent.web.services.task_setup_snapshot import (
    RuntimeUserFields,
    TaskSetupSnapshot,
    _TaskFields,
)

# Deliberately distinct from the runner so a mutation that reads the
# runner's own id instead of the agent's creator id is observable.
_CREATOR_USER_ID = 4242
_RUNNER_USER_ID = 1
_TEAM_ID = 99
_DECLARED_KBS = ["gov-team-handbook", "gov-team-runbook"]


def _make_snapshot() -> TaskSetupSnapshot:
    return TaskSetupSnapshot(
        task=_TaskFields(
            id=42,
            user_id=_RUNNER_USER_ID,
            status=TaskStatus.PENDING,
            agent_id=7,
            agent_config=None,
            model_name=None,
            compact_model_name=None,
            execution_mode="flash",
            agent_type="standard",
        ),
        runtime_user=RuntimeUserFields(id=_RUNNER_USER_ID, is_admin=False),
        has_reconstructable_history=False,
        task_pattern="single_call",
        task_llm=None,
        task_fast_llm=None,
        task_vision_llm=None,
        task_compact_llm=None,
        agent=AgentRuntimeFields(
            id=7,
            name="gov-agent",
            status=AgentStatus.PUBLISHED,
            instructions="be terse",
            team_id=_TEAM_ID,
            agent_creator_user_id=_CREATOR_USER_ID,
        ),
        agent_config={
            "llms": (None, None, None, None),
            "execution_mode": "flash",
            "instructions": "be terse",
            "skills": [],
            "knowledge_bases": list(_DECLARED_KBS),
            "tool_categories": ["basic"],
        },
        excluded_agent_id=None,
    )


@pytest.mark.asyncio
async def test_build_tools_for_task_forwards_snapshot_creator_not_runner() -> None:
    """Pins the snapshot branch inside ``_build_tools_for_task`` (the first
    of the two ``agent_creator_user_id`` derivation points): the value
    forwarded to ``create_default_tools`` must be the agent's own creator,
    never the id of whoever is running the task.
    """
    manager = AgentServiceManager()
    snapshot = _make_snapshot()

    with (
        patch(
            "xagent.web.services.agent_service_manager.create_default_tools",
            new=AsyncMock(return_value=([], MagicMock())),
        ) as create_tools_mock,
        patch.object(
            manager, "_get_or_create_task_sandbox", AsyncMock(return_value=None)
        ),
    ):
        await manager._build_tools_for_task(
            task_id=42,
            task=snapshot.task,
            db=None,
            user=snapshot.runtime_user,
            agent_config=snapshot.agent_config,
            task_llm=None,
            task_vision_llm=None,
            task_setup_snapshot=snapshot,
        )

    kwargs = create_tools_mock.call_args.kwargs
    assert kwargs.get("connector_team_id") == _TEAM_ID
    assert kwargs.get("agent_creator_user_id") == _CREATOR_USER_ID
    assert kwargs.get("agent_creator_user_id") != _RUNNER_USER_ID
    assert kwargs.get("declared_knowledge_bases") == _DECLARED_KBS


def _make_runner_user() -> User:
    return User(
        id=_RUNNER_USER_ID,
        username="wiring-runner",
        password_hash="hash",
        is_admin=False,
    )


@pytest.mark.asyncio
async def test_get_agent_for_task_forwards_snapshot_declaration_not_empty() -> None:
    """Pins the second call site, inside ``_get_agent_for_task_unlocked``:
    the declaration forwarded to ``create_default_tools`` must be the
    agent's own stored ``knowledge_bases`` list, never a hardcoded empty
    list standing in for "no declaration".
    """
    manager = AgentServiceManager()
    manager._default_llm = MagicMock()
    snapshot = _make_snapshot()

    task_row = Task(
        id=42,
        user_id=_RUNNER_USER_ID,
        title="wiring-behavioral",
        description="wiring-behavioral",
        status=TaskStatus.PENDING,
        agent_id=7,
        agent_type="standard",
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = task_row

    with (
        patch(
            "xagent.web.services.agent_service_manager.create_default_tools",
            new=AsyncMock(return_value=([], MagicMock())),
        ) as create_tools_mock,
        patch.object(manager, "_load_persisted_conversation_history"),
        patch.object(manager, "_load_persisted_execution_context", new=AsyncMock()),
        patch(
            "xagent.web.services.agent_service_manager.create_task_tracer",
            return_value=MagicMock(),
        ),
        patch("xagent.web.sandbox_manager.get_sandbox_manager", return_value=None),
        patch("xagent.web.services.agent_service_manager.AgentService"),
    ):
        try:
            await manager.get_agent_for_task(
                task_id=42,
                db=db,
                user=_make_runner_user(),
                task_setup_snapshot=snapshot,
            )
        except Exception:
            # Downstream construction may raise against the mocked
            # AgentService; the forwarding assertion below is recorded
            # before that point (same pattern as the snapshot-passthrough
            # suite this file sits alongside).
            pass

    kwargs = create_tools_mock.call_args.kwargs
    assert kwargs.get("connector_team_id") == _TEAM_ID
    assert kwargs.get("agent_creator_user_id") == _CREATOR_USER_ID
    assert kwargs.get("declared_knowledge_bases") == _DECLARED_KBS
    assert kwargs.get("declared_knowledge_bases") != []


@pytest.mark.asyncio
async def test_build_tools_for_task_live_orm_branch_reads_the_agent_row() -> None:
    """Pins the third derivation point: the snapshot-less branch of
    ``_build_tools_for_task``, which reads the creator off a live ``Agent``
    row instead of a frozen snapshot.

    The static guard only proves this branch does not hardcode ``None``,
    which leaves the same swap the other two branches are pinned against
    wide open: reading the *task's* ``user_id`` (whoever is running the
    task) where the *agent row's* ``user_id`` (whoever created the agent)
    belongs. The two ids differ in this fixture, so that swap is visible.

    The branch is not reachable from production today -- the only caller
    always supplies a snapshot -- but the team-scope rule now consumes
    ``agent_creator_user_id`` to decide which declared knowledge-base names
    a runner may resolve, so a snapshot-less path added later would widen
    or narrow that resolution silently if it derived the value wrongly.
    """
    manager = AgentServiceManager()
    snapshot = _make_snapshot()

    # Deliberately owned by the runner, while the agent row below is owned
    # by the creator: the forwarded value must follow the agent, not the task.
    task_row = Task(
        id=42,
        user_id=_RUNNER_USER_ID,
        title="live-orm-branch",
        description="live-orm-branch",
        status=TaskStatus.PENDING,
        agent_id=7,
        agent_type="standard",
    )
    agent_row = Agent(
        id=7,
        user_id=_CREATOR_USER_ID,
        team_id=_TEAM_ID,
        name="gov-agent",
        status=AgentStatus.PUBLISHED,
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = agent_row

    with (
        patch(
            "xagent.web.services.agent_service_manager.create_default_tools",
            new=AsyncMock(return_value=([], MagicMock())),
        ) as create_tools_mock,
        patch(
            "xagent.web.services.agent_service_manager.resolve_workforce_task_runtime",
            return_value=None,
        ),
        patch.object(
            manager, "_get_or_create_task_sandbox", AsyncMock(return_value=None)
        ),
    ):
        await manager._build_tools_for_task(
            task_id=42,
            task=task_row,
            db=db,
            user=snapshot.runtime_user,
            agent_config=snapshot.agent_config,
            task_llm=None,
            task_vision_llm=None,
            task_setup_snapshot=None,
        )

    kwargs = create_tools_mock.call_args.kwargs
    assert kwargs.get("agent_creator_user_id") == _CREATOR_USER_ID
    assert kwargs.get("agent_creator_user_id") != _RUNNER_USER_ID
    assert kwargs.get("connector_team_id") == _TEAM_ID
    assert kwargs.get("declared_knowledge_bases") == _DECLARED_KBS
