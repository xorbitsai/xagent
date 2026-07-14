import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import operators
from sqlalchemy.sql.elements import BinaryExpression, BooleanClauseList

from xagent.web.models.agent import Agent
from xagent.web.models.database import Base
from xagent.web.models.user import User
from xagent.web.services import agent_team_scope as ats
from xagent.web.services.agent_store import AgentStore
from xagent.web.services.agent_team_scope import AgentTeamScope


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)


def _make_user(db, email):
    # username/password_hash are NOT NULL in the User model.
    user = User(username=email, email=email, password_hash="x", is_admin=False)
    db.add(user)
    db.flush()
    return user


def test_agent_model_has_team_id_column():
    # team_id is a nullable scoping column owned by the SaaS overlay.
    col = Agent.__table__.columns["team_id"]
    assert col.nullable is True


def test_scope_is_none_without_hook():
    ats.set_agent_team_scope_hook(None)
    assert ats.get_agent_team_scope(None, 7) is None


def test_scope_uses_hook_when_installed():
    ats.set_agent_team_scope_hook(
        lambda db, uid: (
            AgentTeamScope(team_id=42, is_team_admin=False) if uid == 7 else None
        )
    )
    try:
        assert ats.get_agent_team_scope(None, 7).team_id == 42
        assert ats.get_agent_team_scope(None, 9) is None
        assert ats.get_agent_team_scope(None, None) is None  # None user -> no scope
    finally:
        ats.set_agent_team_scope_hook(None)


def test_owned_clause_shape():
    # No scope: a single equality on exactly the agents.user_id column.
    user_only = ats.owned_agent_clause(7, None)
    assert isinstance(user_only, BinaryExpression)
    assert user_only.operator is operators.eq
    assert user_only.left.table is Agent.__table__
    assert user_only.left.name == "user_id"

    # Scoped: an OR combining the legacy user-only row and the team clause.
    team = ats.owned_agent_clause(7, AgentTeamScope(team_id=42, is_team_admin=True))
    assert isinstance(team, BooleanClauseList)
    assert team.operator is operators.or_
    assert "team_id" in str(team) and "user_id" in str(team)


def _agent(user_id, team_id=None, visibility="team"):
    return Agent(user_id=user_id, team_id=team_id, visibility=visibility)


def test_owns_agent_mirrors_owned_clause_branches():
    admin = AgentTeamScope(team_id=100, is_team_admin=True)
    member = AgentTeamScope(team_id=100, is_team_admin=False)

    # No scope: pure user_id equality.
    assert ats.owns_agent(_agent(7), 7, None) is True
    assert ats.owns_agent(_agent(9), 7, None) is False

    # Legacy row (no team) resolves via its own user_id even under a scope.
    assert ats.owns_agent(_agent(7, team_id=None), 7, member) is True
    assert ats.owns_agent(_agent(9, team_id=None), 7, member) is False

    # Team admin owns every team agent regardless of visibility.
    assert ats.owns_agent(_agent(9, team_id=100, visibility="admins"), 7, admin) is True

    # Non-admin teammate owns team-visible agents only...
    assert ats.owns_agent(_agent(9, team_id=100, visibility="team"), 7, member) is True
    # ...and loses access once an admin flips it to admins-only (the workforce
    # bypass the raw user_id check used to leave open for the creator).
    assert (
        ats.owns_agent(_agent(7, team_id=100, visibility="admins"), 7, member) is False
    )

    # Foreign team never owned.
    assert ats.owns_agent(_agent(9, team_id=200, visibility="team"), 7, member) is False


def test_cache_keys_are_team_scoped_when_scope_present():
    from xagent.web.services import hot_path_cache as hpc

    assert hpc.agent_list_key(7) == "agent:list:7"
    assert (
        hpc.agent_list_key(7, team_id=42, is_team_admin=True)
        == "agent:list:team:42:7:a"
    )
    assert hpc.agent_detail_key(7, 5) == "agent:detail:7:5"
    assert hpc.agent_detail_key(7, 5, team_id=42) == "agent:detail:team:42:7:m:5"
    assert (
        hpc.agent_detail_key(7, 5, team_id=42, is_team_admin=True)
        == "agent:detail:team:42:7:a:5"
    )


@pytest.fixture
def two_users_one_team(db):
    a = _make_user(db, "a@t.co")
    b = _make_user(db, "b@t.co")
    db.flush()
    # Both users resolve to the same team scope (id 100).
    ats.set_agent_team_scope_hook(
        lambda _db, uid: (
            AgentTeamScope(team_id=100, is_team_admin=False)
            if uid in {a.id, b.id}
            else None
        )
    )
    yield a, b
    ats.set_agent_team_scope_hook(None)


def test_member_can_read_and_manage_teammates_agent(db, two_users_one_team):
    a, b = two_users_one_team
    store = AgentStore(db)
    created = store.create_agent(
        user_id=int(a.id), name="Shared", description=None, instructions=None
    )
    # B sees it in the list...
    b_list = store.list_agent_items(int(b.id))
    assert any(item["id"] == int(created.id) for item in b_list)
    # ...can fetch it as an owned agent...
    assert store.get_owned_agent(int(b.id), int(created.id)) is not None
    # ...and it was stamped with the team scope on create.
    assert db.query(Agent).get(int(created.id)).team_id == 100


def test_name_uniqueness_is_per_team(db, two_users_one_team):
    a, b = two_users_one_team
    store = AgentStore(db)
    store.create_agent(
        user_id=int(a.id), name="Dup", description=None, instructions=None
    )
    assert store.agent_name_exists(int(b.id), "Dup") is True


def test_teammate_agent_is_owner_access(db, two_users_one_team):
    from xagent.web.services.agent_access import list_accessible_agents

    a, b = two_users_one_team
    AgentStore(db).create_agent(
        user_id=int(a.id), name="Shared2", description=None, instructions=None
    )
    items = list_accessible_agents(db, b)
    match = [i for i in items if i.agent.name == "Shared2"]
    assert match and match[0].access == "owner"
    assert match[0].can_edit and match[0].can_delete


# ===== Agent sub-resources: API keys and triggers are team-co-managed =====


def test_member_can_manage_teammates_agent_api_key(db, two_users_one_team):
    from xagent.web.services.api_keys import AgentApiKeyService

    a, b = two_users_one_team
    agent = AgentStore(db).create_agent(
        user_id=int(a.id), name="KeyShared", description=None, instructions=None
    )
    svc = AgentApiKeyService(db)
    # B (teammate) can create and then list a key on A's agent.
    created = svc.create_key(int(b.id), int(agent.id), label="from-b")
    assert created is not None
    keys = svc.list_keys_for_user(int(b.id), agent_id=int(agent.id))
    assert [k.label for k in keys] == ["from-b"]


def test_member_can_manage_teammates_agent_trigger(db, two_users_one_team):
    from xagent.web.services.triggers import get_owned_agent

    a, b = two_users_one_team
    agent = AgentStore(db).create_agent(
        user_id=int(a.id), name="TrigShared", description=None, instructions=None
    )
    # B (teammate) resolves A's agent as owned -- the gate every trigger
    # endpoint routes through.
    assert get_owned_agent(db, user_id=int(b.id), agent_id=int(agent.id)) is not None


def test_member_can_read_and_edit_teammates_trigger(db, two_users_one_team):
    # A trigger created by A on a co-owned agent must be visible/editable/
    # deletable by teammate B, whose user_id differs from the trigger creator's.
    from xagent.web.services.triggers import (
        create_agent_trigger,
        get_owned_trigger,
        update_agent_trigger,
    )

    a, b = two_users_one_team
    agent = AgentStore(db).create_agent(
        user_id=int(a.id), name="TrigCoManage", description=None, instructions=None
    )
    trigger, _ = create_agent_trigger(
        db,
        user_id=int(a.id),
        agent_id=int(agent.id),
        trigger_type="scheduled",
        config={"interval_seconds": 3600},
    )
    assert int(trigger.user_id) == int(a.id)

    # B resolves the trigger even though B did not create it.
    resolved = get_owned_trigger(
        db, user_id=int(b.id), agent_id=int(agent.id), trigger_id=int(trigger.id)
    )
    assert resolved is not None and int(resolved.id) == int(trigger.id)

    # B can update it; the creator's user_id is preserved.
    updated, _ = update_agent_trigger(
        db,
        user_id=int(b.id),
        agent_id=int(agent.id),
        trigger_id=int(trigger.id),
        updates={"name": "renamed-by-b"},
    )
    assert str(updated.name) == "renamed-by-b"
    assert int(updated.user_id) == int(a.id)


def test_workforce_names_agent_name_exists_is_per_team(db, two_users_one_team):
    from xagent.web.services.workforce_names import agent_name_exists

    a, b = two_users_one_team
    AgentStore(db).create_agent(
        user_id=int(a.id), name="WFDup", description=None, instructions=None
    )
    # B's name-existence check sees A's team-visible agent.
    assert agent_name_exists(db, user_id=int(b.id), name="WFDup") is True


def test_non_team_user_still_gets_404_on_subresources(db):
    # No hook installed -> user-only ownership; an outsider sees nothing.
    from xagent.web.services.api_keys import AgentApiKeyService
    from xagent.web.services.triggers import get_owned_agent

    ats.set_agent_team_scope_hook(None)
    a = _make_user(db, "owner@t.co")
    outsider = _make_user(db, "out@t.co")
    db.flush()
    agent = AgentStore(db).create_agent(
        user_id=int(a.id), name="Solo", description=None, instructions=None
    )
    assert get_owned_agent(db, user_id=int(outsider.id), agent_id=int(agent.id)) is None
    svc = AgentApiKeyService(db)
    assert svc.create_key(int(outsider.id), int(agent.id), label="x") is None
