"""The task-setup snapshot actually carries the governing agent's team id,
which is what makes team-keyed connector visibility representable on the
snapshot branch (``chat.py``'s only production-reachable tool-build path).

``AgentRuntimeFields.team_id`` has a closed (``None``) default, so a
forgotten production construction site is otherwise silent across every
existing ``AgentRuntimeFields(`` test file -- this is the one pin that would
catch it.
"""

from __future__ import annotations

import pytest

from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.llm_utils import AgentRuntimeFields
from xagent.web.services.task_setup_snapshot import load_task_setup_snapshot_sync


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'snapshot_agent_team.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


def _create_user(db) -> User:
    user = User(username="snap-team-user", password_hash="hash", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _create_agent(db, user_id: int, **overrides) -> Agent:
    defaults = dict(
        user_id=user_id,
        name="snap-team-agent",
        instructions="be terse",
        status=AgentStatus.PUBLISHED,
        execution_mode="balanced",
        models={},
        knowledge_bases=[],
        skills=[],
        tool_categories=["basic"],
    )
    defaults.update(overrides)
    agent = Agent(**defaults)
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return agent


def _create_task(db, user_id: int, **overrides) -> Task:
    defaults = dict(
        user_id=user_id,
        title="Snapshot agent-team test",
        description="snapshot",
        status=TaskStatus.PENDING,
        execution_mode="flash",
        source="sdk",
    )
    defaults.update(overrides)
    task = Task(**defaults)
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def test_snapshot_carries_agent_team_id(db_session) -> None:
    user = _create_user(db_session)
    agent = _create_agent(db_session, user_id=int(user.id), team_id=101)
    task = _create_task(db_session, user_id=int(user.id), agent_id=int(agent.id))

    snapshot = load_task_setup_snapshot_sync(
        task_id=int(task.id), task_owner_user_id=int(user.id)
    )

    assert snapshot is not None
    assert snapshot.agent is not None
    assert isinstance(snapshot.agent, AgentRuntimeFields)
    assert snapshot.agent.team_id == 101


def test_snapshot_agent_team_id_defaults_to_none_for_personal_agent(db_session) -> None:
    """Negative sibling of ``test_snapshot_carries_agent_team_id``. A bare
    "personal agent resolves None" assertion would be near-tautological on
    its own -- ``None`` is also the dataclass's closed default, so the
    assertion cannot distinguish "correctly resolved no team" from "the
    field was never wired and just fell through". This test first
    resolves a *team* agent's snapshot in the same session and confirms it
    reads back 101, then resolves the personal agent and confirms it reads
    back ``None`` -- so a bug where the resolution leaks or caches the prior
    team id across agents (rather than reading each row's own column) would
    turn this red, which the original standalone assertion could not catch.
    """
    user = _create_user(db_session)
    team_agent = _create_agent(db_session, user_id=int(user.id), team_id=101)
    team_task = _create_task(
        db_session, user_id=int(user.id), agent_id=int(team_agent.id)
    )
    team_snapshot = load_task_setup_snapshot_sync(
        task_id=int(team_task.id), task_owner_user_id=int(user.id)
    )
    assert team_snapshot is not None and team_snapshot.agent is not None
    assert team_snapshot.agent.team_id == 101

    agent = _create_agent(
        db_session, user_id=int(user.id), name="snap-team-agent-personal"
    )  # team_id left unset
    task = _create_task(db_session, user_id=int(user.id), agent_id=int(agent.id))

    snapshot = load_task_setup_snapshot_sync(
        task_id=int(task.id), task_owner_user_id=int(user.id)
    )

    assert snapshot is not None
    assert snapshot.agent is not None
    assert snapshot.agent.team_id is None
