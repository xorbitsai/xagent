"""The ``chat.py`` derivation actually threads the governing agent's team id
down to ``WebToolConfig``, on the snapshot branch.

The snapshot branch in ``_build_tools_for_task`` is the only branch
production reaches (``_build_tools_for_task`` has exactly one production
caller, and it cannot reach it with a ``None`` snapshot). This test is the
only one that exercises the
``snapshot -> chat.py derivation -> WebToolConfig._connector_team_id`` link
end to end; the team-keyed resolution itself and the personal-agent negative
are pinned directly against the lower-level functions in
``tests/web/tools/test_mcp_team_visibility.py``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from xagent.web.models.database import Base, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.agent_service_manager import AgentServiceManager
from xagent.web.services.llm_utils import AgentRuntimeFields
from xagent.web.services.task_setup_snapshot import (
    RuntimeUserFields,
    TaskSetupSnapshot,
    _TaskFields,
)


@pytest.fixture(autouse=True)
def _init_db(tmp_path):
    # ``db=None`` on the snapshot branch still routes through
    # ``create_default_tools``'s worker session factory
    # (``get_session_local()``), so a real (if unused-by-the-assertion)
    # SessionLocal must exist.
    init_db(db_url=f"sqlite:///{tmp_path / 'e2e.db'}")
    yield
    Base.metadata.drop_all(bind=get_engine())


def _task_fields(task: Task) -> _TaskFields:
    return _TaskFields(
        id=int(task.id),
        user_id=int(task.user_id),
        status=task.status,
        agent_id=task.agent_id,
        agent_config=task.agent_config,
        model_name=task.model_name,
        compact_model_name=task.compact_model_name,
        execution_mode=task.execution_mode,
        agent_type=task.agent_type,
        source=task.source,
    )


async def _noop_create_all_tools(
    config, apply_user_override_filter=True, additional_tools=()
):
    return list(additional_tools)


@pytest.mark.asyncio
async def test_snapshot_branch_threads_agent_team_to_tool_config(monkeypatch):
    manager = AgentServiceManager()
    user = User(id=1, username="e2e-team-user", password_hash="hash", is_admin=False)
    task = Task(
        id=1,
        user_id=1,
        title="team-derivation task",
        description="e2e",
        status=TaskStatus.PENDING,
        model_name="gpt-4",
        small_fast_model_name="gpt-3.5-turbo",
        agent_type="standard",
    )
    agent_config = {"tool_categories": ["basic"], "knowledge_bases": [], "skills": []}
    snapshot = TaskSetupSnapshot(
        task=_task_fields(task),
        runtime_user=RuntimeUserFields(id=int(user.id), is_admin=False),
        has_reconstructable_history=False,
        task_pattern="dag_plan_execute",
        task_llm=None,
        task_fast_llm=None,
        task_vision_llm=None,
        task_compact_llm=None,
        agent=AgentRuntimeFields(
            id=99,
            name="team-agent",
            status="published",
            instructions=None,
            team_id=101,
        ),
        agent_config=agent_config,
        excluded_agent_id=99,
    )

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.factory.ToolFactory.create_all_tools",
        _noop_create_all_tools,
    )

    with patch("xagent.web.sandbox_manager.get_sandbox_manager", return_value=None):
        _tools, tool_config = await manager._build_tools_for_task(
            task_id=int(task.id),
            task=task,
            db=None,
            user=user,
            agent_config=agent_config,
            task_llm=None,
            task_vision_llm=None,
            task_setup_snapshot=snapshot,
        )

    assert tool_config._connector_team_id == 101


@pytest.mark.asyncio
async def test_snapshot_branch_with_no_governing_agent_stays_personal(monkeypatch):
    """The negative case: no ``snapshot.agent`` resolves ``None``, not some
    inherited or stale team id.

    A bare "no agent resolves None" assertion would be near-tautological on
    its own, since ``None`` is also the parameter's closed default -- it
    cannot tell "correctly resolved no team" apart from "the derivation was
    never reached". This test first drives ``_build_tools_for_task`` with a
    *team* snapshot (team_id=101, same manager instance) and confirms 101
    comes through, then immediately drives it again with no governing agent
    and confirms ``None`` -- so a bug where the team id leaks across builds
    (e.g. a shared-default-argument or manager-level caching mistake) would
    turn this red, which a standalone assertion could not catch.
    """
    manager = AgentServiceManager()
    user = User(
        id=1, username="e2e-personal-user", password_hash="hash", is_admin=False
    )

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.factory.ToolFactory.create_all_tools",
        _noop_create_all_tools,
    )

    team_task = Task(
        id=3,
        user_id=1,
        title="personal-negative team baseline",
        description="e2e",
        status=TaskStatus.PENDING,
        model_name="gpt-4",
        small_fast_model_name="gpt-3.5-turbo",
        agent_type="standard",
    )
    team_agent_config = {
        "tool_categories": ["basic"],
        "knowledge_bases": [],
        "skills": [],
    }
    team_snapshot = TaskSetupSnapshot(
        task=_task_fields(team_task),
        runtime_user=RuntimeUserFields(id=int(user.id), is_admin=False),
        has_reconstructable_history=False,
        task_pattern="dag_plan_execute",
        task_llm=None,
        task_fast_llm=None,
        task_vision_llm=None,
        task_compact_llm=None,
        agent=AgentRuntimeFields(
            id=98,
            name="team-agent-baseline",
            status="published",
            instructions=None,
            team_id=101,
        ),
        agent_config=team_agent_config,
        excluded_agent_id=98,
    )
    with patch("xagent.web.sandbox_manager.get_sandbox_manager", return_value=None):
        _tools, team_tool_config = await manager._build_tools_for_task(
            task_id=int(team_task.id),
            task=team_task,
            db=None,
            user=user,
            agent_config=team_agent_config,
            task_llm=None,
            task_vision_llm=None,
            task_setup_snapshot=team_snapshot,
        )
    assert team_tool_config._connector_team_id == 101

    task = Task(
        id=2,
        user_id=1,
        title="personal task",
        description="e2e",
        status=TaskStatus.PENDING,
        model_name="gpt-4",
        small_fast_model_name="gpt-3.5-turbo",
        agent_type="standard",
    )
    snapshot = TaskSetupSnapshot(
        task=_task_fields(task),
        runtime_user=RuntimeUserFields(id=int(user.id), is_admin=False),
        has_reconstructable_history=False,
        task_pattern="dag_plan_execute",
        task_llm=None,
        task_fast_llm=None,
        task_vision_llm=None,
        task_compact_llm=None,
        agent=None,
        agent_config=None,
        excluded_agent_id=None,
    )

    with patch("xagent.web.sandbox_manager.get_sandbox_manager", return_value=None):
        _tools, tool_config = await manager._build_tools_for_task(
            task_id=int(task.id),
            task=task,
            db=None,
            user=user,
            agent_config=None,
            task_llm=None,
            task_vision_llm=None,
            task_setup_snapshot=snapshot,
        )

    assert tool_config._connector_team_id is None
