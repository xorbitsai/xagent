"""``AgentRuntimeFields.agent_creator_user_id`` always names the same agent
as ``team_id``, at the single construction site in
``llm_utils.resolve_task_runtime_config_core`` that both the snapshot
branch and the stored-snapshot branch of ``chat.py`` read off.

Coverage note, stated plainly rather than left implicit: this file proves
the shared construction site is correct, which is what makes those two
``chat.py`` read sites (``task_setup_snapshot.agent.agent_creator_user_id``
and ``snapshot.agent.agent_creator_user_id``) trustworthy -- but it does
not call ``chat.py`` itself, so it cannot catch a ``chat.py`` change that
stops reading this field, or reads it off the wrong object. The third
``chat.py`` site -- the snapshot-less branch that reads a live ``Agent``
ORM row directly (``int(current_agent.team_id)`` /
``int(current_agent.user_id)``), which the design itself documents as not
reachable from production today -- has no test in this file at all: an
earlier draft of this file asserted a copy of that expression against
itself rather than against ``chat.py``'s actual code, which cannot fail no
matter what ``chat.py`` does, so it was removed rather than left as a
false-confidence pin. Closing this gap for real requires driving
``AgentServiceManager._build_tools_for_task`` through its snapshot-less
branch, which means constructing a real session, task, and workforce
runtime; that has not been done.
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
    init_db(db_url=f"sqlite:///{tmp_path / 'agent_creator.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


def _create_user(db, username: str, *, is_admin: bool = False) -> User:
    user = User(username=username, password_hash="hash", is_admin=is_admin)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _create_agent(db, user_id: int, **overrides) -> Agent:
    defaults = dict(
        user_id=user_id,
        name="creator-pin-agent",
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
        title="creator pin test",
        description="d",
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


def test_snapshot_carries_agent_creator_alongside_team_id(db_session) -> None:
    """Derivation points 1 and 3 (both read ``snapshot.agent``): the
    snapshot's ``team_id`` and ``agent_creator_user_id`` describe the same
    agent.

    The agent creator and the task owner are two different users here --
    a binding that read ``agent_creator_user_id`` off the task owner
    instead of ``agent_row.user_id`` would still pass against an
    equal-user fixture, so this fixture keeps the two apart and asserts
    the field names the creator, not the owner. The owner is an admin
    only so that ``_load_agent_for_task_runtime`` can load a PUBLISHED
    agent it does not own and is not team-scoped into -- standalone
    xagent has no other cross-user visibility path for a published
    agent, and that access check is incidental to what this test pins.
    A creator-vs-runner *resolution* scenario, where the two differ, is
    exercised by the resolution-outcome tests against
    ``_search_knowledge_base_impl`` directly.
    """
    creator = _create_user(db_session, "creator-pin-creator")
    owner = _create_user(db_session, "creator-pin-owner", is_admin=True)
    agent = _create_agent(db_session, user_id=int(creator.id), team_id=101)
    task = _create_task(db_session, user_id=int(owner.id), agent_id=int(agent.id))

    snapshot = load_task_setup_snapshot_sync(
        task_id=int(task.id), task_owner_user_id=int(owner.id)
    )

    assert snapshot is not None and snapshot.agent is not None
    assert isinstance(snapshot.agent, AgentRuntimeFields)
    assert snapshot.agent.team_id == 101
    assert snapshot.agent.agent_creator_user_id == int(creator.id)
    assert snapshot.agent.agent_creator_user_id != int(owner.id)


# No test below this point for chat.py's snapshot-less derivation branch
# (AgentServiceManager._build_tools_for_task's `else` arm, which reads
# `current_agent.team_id` / `current_agent.user_id` directly off a live
# ORM row instead of an AgentRuntimeFields snapshot). See the module
# docstring above for why and what closing this gap would require.
