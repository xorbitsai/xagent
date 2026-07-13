import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.models.agent import Agent
from xagent.web.models.database import Base


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


def test_agent_has_visibility_column_defaulting_to_team():
    col = Agent.__table__.columns["visibility"]
    assert col.nullable is False
    assert col.server_default is not None


def test_serializers_include_visibility(db):
    # db: in-memory sqlite fixture matching tests/web/test_agent_team_scope.py
    from xagent.web.models.user import User
    from xagent.web.services.agent_store import AgentStore

    u = User(username="ser", password_hash="h", is_admin=False)
    db.add(u)
    db.flush()
    store = AgentStore(db)
    agent = store.create_agent(
        user_id=int(u.id), name="Ser", description=None, instructions=None
    )
    assert store.agent_to_response_dict(agent)["visibility"] == "team"
    assert store.agent_to_list_item_dict(agent)["visibility"] == "team"


def test_team_id_of():
    from xagent.web.services.agent_team_scope import AgentTeamScope, team_id_of

    assert team_id_of(None) is None
    assert team_id_of(AgentTeamScope(team_id=5, is_team_admin=True)) == 5


def test_clause_admin_sees_all_team_agents():
    from xagent.web.services.agent_team_scope import (
        AgentTeamScope,
        owned_agent_clause,
    )

    clause = owned_agent_clause(7, AgentTeamScope(team_id=42, is_team_admin=True))
    # admin branch scopes by team_id only, no visibility filter
    text = str(clause)
    assert "visibility" not in text
    assert "team_id" in text


def test_clause_member_filtered_by_visibility():
    from xagent.web.services.agent_team_scope import (
        AgentTeamScope,
        owned_agent_clause,
    )

    clause = owned_agent_clause(7, AgentTeamScope(team_id=42, is_team_admin=False))
    text = str(clause)
    assert "visibility" in text
    assert "team_id" in text


def test_clause_no_scope_is_user_only():
    from xagent.web.models.agent import Agent
    from xagent.web.services.agent_team_scope import owned_agent_clause

    clause = owned_agent_clause(7, None)
    assert str(clause) == str(Agent.user_id == 7)


def test_team_list_key_splits_by_role():
    from xagent.web.services import hot_path_cache as hpc

    assert hpc.agent_list_key(7) == "agent:list:7"
    assert (
        hpc.agent_list_key(7, team_id=42, is_team_admin=True)
        == "agent:list:team:42:7:a"
    )
    assert (
        hpc.agent_list_key(7, team_id=42, is_team_admin=False)
        == "agent:list:team:42:7:m"
    )


def test_team_cache_keys_are_per_user():
    # owned_agent_clause still matches a user's own legacy (team_id IS NULL)
    # agents, so two users in the same team + role must get DIFFERENT cache
    # keys -- otherwise one member's private agents leak to another on a hit.
    from xagent.web.services import hot_path_cache as hpc

    assert hpc.agent_list_key(1, team_id=9, is_team_admin=False) != hpc.agent_list_key(
        2, team_id=9, is_team_admin=False
    )
    assert hpc.agent_detail_key(1, 5, team_id=9) != hpc.agent_detail_key(
        2, 5, team_id=9
    )
    assert ":1:" in hpc.agent_list_key(1, team_id=9)


@pytest.fixture
def team_of_admin_and_member(db):
    from xagent.web.models.user import User
    from xagent.web.services import agent_team_scope as ats
    from xagent.web.services.agent_team_scope import AgentTeamScope

    admin = User(username="adm", password_hash="h", is_admin=False)
    member = User(username="mem", password_hash="h", is_admin=False)
    db.add_all([admin, member])
    db.flush()
    admins = {int(admin.id)}
    ats.set_agent_team_scope_hook(
        lambda _db, uid: (
            AgentTeamScope(team_id=200, is_team_admin=uid in admins)
            if uid in {int(admin.id), int(member.id)}
            else None
        )
    )
    yield admin, member
    ats.set_agent_team_scope_hook(None)


def test_member_cannot_see_admins_only_agent(db, team_of_admin_and_member):
    from xagent.web.services.agent_store import AgentStore

    admin, member = team_of_admin_and_member
    store = AgentStore(db)
    agent = store.create_agent(
        user_id=int(admin.id), name="Secret", description=None, instructions=None
    )
    store.update_agent_fields(int(admin.id), int(agent.id), {"visibility": "admins"})

    assert store.get_owned_agent(int(member.id), int(agent.id)) is None
    assert all(i["id"] != int(agent.id) for i in store.list_agent_items(int(member.id)))
    # admin still sees + lists it
    assert store.get_owned_agent(int(admin.id), int(agent.id)) is not None
    assert any(i["id"] == int(agent.id) for i in store.list_agent_items(int(admin.id)))


def test_member_cannot_set_visibility_admins(db, team_of_admin_and_member):
    from xagent.web.services.agent_store import AgentStore

    admin, member = team_of_admin_and_member
    store = AgentStore(db)
    # member-created, team-visible agent
    agent = store.create_agent(
        user_id=int(member.id), name="Mine", description=None, instructions=None
    )
    with pytest.raises(PermissionError):
        store.update_agent_fields(
            int(member.id), int(agent.id), {"visibility": "admins"}
        )


def test_admin_can_create_admins_only_agent(db, team_of_admin_and_member):
    from xagent.web.services.agent_store import AgentStore

    admin, _member = team_of_admin_and_member
    store = AgentStore(db)
    agent = store.create_agent(
        user_id=int(admin.id),
        name="Secret",
        description=None,
        instructions=None,
        visibility="admins",
    )
    assert agent.visibility == "admins"


def test_member_cannot_create_admins_only_agent(db, team_of_admin_and_member):
    from xagent.web.services.agent_store import AgentStore

    _admin, member = team_of_admin_and_member
    store = AgentStore(db)
    with pytest.raises(PermissionError):
        store.create_agent(
            user_id=int(member.id),
            name="Nope",
            description=None,
            instructions=None,
            visibility="admins",
        )


def test_create_defaults_visibility_to_team(db, team_of_admin_and_member):
    from xagent.web.services.agent_store import AgentStore

    admin, _member = team_of_admin_and_member
    store = AgentStore(db)
    agent = store.create_agent(
        user_id=int(admin.id), name="Plain", description=None, instructions=None
    )
    assert agent.visibility == "team"


def test_create_request_carries_visibility():
    from xagent.web.api.agents import AgentCreateRequest

    assert "visibility" in AgentCreateRequest.model_fields
    assert AgentCreateRequest(name="x", visibility="admins").visibility == "admins"
    assert AgentCreateRequest(name="x").visibility is None


def test_update_request_and_response_carry_visibility():
    from xagent.web.api.agents import (
        AgentListItem,
        AgentResponse,
        AgentUpdateRequest,
    )

    assert "visibility" in AgentUpdateRequest.model_fields
    assert "visibility" in AgentResponse.model_fields
    assert "visibility" in AgentListItem.model_fields
    # update request accepts the value
    assert AgentUpdateRequest(visibility="admins").visibility == "admins"


def test_update_request_rejects_bad_visibility():
    import pytest
    from pydantic import ValidationError

    from xagent.web.api.agents import AgentUpdateRequest

    with pytest.raises(ValidationError):
        AgentUpdateRequest(visibility="public")
    # 合法值仍通过
    assert AgentUpdateRequest(visibility="admins").visibility == "admins"
    assert AgentUpdateRequest(visibility="team").visibility == "team"


# ===== Regression coverage for the review findings =====


def test_detail_cache_does_not_leak_admins_only_to_member(db, team_of_admin_and_member):
    """#1: admin reading an admins-only detail must not seed a cache entry a
    member's read then hits (member and admin key different role variants)."""
    from xagent.web.services.agent_store import AgentStore
    from xagent.web.services.hot_path_cache import (
        InMemoryTTLCache,
        set_cache_backend_for_testing,
    )

    set_cache_backend_for_testing(InMemoryTTLCache())
    try:
        admin, member = team_of_admin_and_member
        store = AgentStore(db)
        agent = store.create_agent(
            user_id=int(admin.id),
            name="Secret",
            description=None,
            instructions=None,
            visibility="admins",
        )
        # Admin reads (and caches) the detail first.
        assert store.get_agent_response(int(admin.id), int(agent.id)) is not None
        # Member must not hit the admin's cache entry -> no detail.
        assert store.get_agent_response(int(member.id), int(agent.id)) is None
    finally:
        set_cache_backend_for_testing(None)


def test_list_accessible_agents_hides_published_admins_only_from_member(
    db, team_of_admin_and_member
):
    """#4: a published admins-only agent must not surface via the policy path."""
    from xagent.web.models.agent import AgentStatus
    from xagent.web.services import workforce_access
    from xagent.web.services.agent_access import list_accessible_agents
    from xagent.web.services.agent_store import AgentStore

    admin, member = team_of_admin_and_member
    store = AgentStore(db)
    agent = store.create_agent(
        user_id=int(admin.id),
        name="SecretPub",
        description=None,
        instructions=None,
        visibility="admins",
        status=AgentStatus.PUBLISHED,
    )
    # Force the policy branch to "see" this id, as if a workforce policy did.
    orig = workforce_access.get_visible_agent_ids
    workforce_access.get_visible_agent_ids = lambda _db, _u, _p: {int(agent.id)}
    # agent_access imported the symbol directly; patch there too.
    import xagent.web.services.agent_access as aa

    orig_aa = aa.get_visible_agent_ids
    aa.get_visible_agent_ids = lambda _db, _u, _p: {int(agent.id)}
    try:
        items = list_accessible_agents(db, member)
        assert all(int(i.agent.id) != int(agent.id) for i in items)
        # admin still gets it (owner branch).
        admin_items = list_accessible_agents(db, admin)
        assert any(int(i.agent.id) == int(agent.id) for i in admin_items)
    finally:
        workforce_access.get_visible_agent_ids = orig
        aa.get_visible_agent_ids = orig_aa


def test_set_admins_visibility_fail_closed_without_scope(db):
    """#8: with no team scope, the store must refuse to set admins-only."""
    from xagent.web.models.user import User
    from xagent.web.services import agent_team_scope as ats
    from xagent.web.services.agent_store import AgentStore

    ats.set_agent_team_scope_hook(None)
    u = User(username="solo", password_hash="h", is_admin=False)
    db.add(u)
    db.flush()
    store = AgentStore(db)
    with pytest.raises(PermissionError):
        store.create_agent(
            user_id=int(u.id),
            name="Nope",
            description=None,
            instructions=None,
            visibility="admins",
        )


def test_store_rejects_unknown_visibility_value(db, team_of_admin_and_member):
    """#14: store-layer domain check rejects values outside {team, admins}."""
    from xagent.web.services.agent_store import AgentStore

    admin, _member = team_of_admin_and_member
    store = AgentStore(db)
    with pytest.raises(ValueError):
        store.create_agent(
            user_id=int(admin.id),
            name="Bad",
            description=None,
            instructions=None,
            visibility="public",
        )


def test_non_admin_cannot_downgrade_admins_only_agent(db, team_of_admin_and_member):
    """#9: the visibility guard rejects a non-admin changing (downgrading) an
    agent that is currently admins-only, even when the agent is handed in
    pre-fetched (the endpoint's reuse path). Members normally 404 at the
    ownership layer; this asserts the guard itself is the second line."""
    from xagent.web.services.agent_store import (
        AgentStore,
        _assert_can_set_visibility,
    )
    from xagent.web.services.agent_team_scope import AgentTeamScope

    admin, member = team_of_admin_and_member
    store = AgentStore(db)
    agent = store.create_agent(
        user_id=int(admin.id),
        name="SecretDown",
        description=None,
        instructions=None,
        visibility="admins",
    )
    non_admin_scope = AgentTeamScope(team_id=200, is_team_admin=False)
    # Guard rejects a downgrade attempt with a non-admin scope.
    with pytest.raises(PermissionError):
        _assert_can_set_visibility(non_admin_scope, "team", agent.visibility)
    # And through the store when the owned agent is passed in explicitly.
    with pytest.raises(PermissionError):
        store.update_agent_fields(
            int(member.id),
            int(agent.id),
            {"visibility": "team"},
            team_scope=non_admin_scope,
            agent=agent,
        )


def test_run_path_team_member_can_load_team_agent(db, two_users_one_team_via_scope):
    """#2: a teammate can load a team-visible agent for a task; admins-only
    stays hidden from a non-admin runner."""
    from xagent.web.api.chat import _load_agent_for_task_create
    from xagent.web.models.agent import AgentStatus
    from xagent.web.services.agent_store import AgentStore

    admin, member = two_users_one_team_via_scope
    store = AgentStore(db)
    team_agent = store.create_agent(
        user_id=int(admin.id),
        name="TeamRun",
        description=None,
        instructions=None,
        status=AgentStatus.PUBLISHED,
    )
    secret = store.create_agent(
        user_id=int(admin.id),
        name="AdminRun",
        description=None,
        instructions=None,
        visibility="admins",
        status=AgentStatus.PUBLISHED,
    )
    assert _load_agent_for_task_create(db, member, int(team_agent.id)) is not None
    assert _load_agent_for_task_create(db, member, int(secret.id)) is None


@pytest.fixture
def two_users_one_team_via_scope(db):
    """Admin + member on one team, member is non-admin (like the visibility
    fixture but exposing both User rows for the run-path test)."""
    from xagent.web.models.user import User
    from xagent.web.services import agent_team_scope as ats
    from xagent.web.services.agent_team_scope import AgentTeamScope

    admin = User(username="radm", password_hash="h", is_admin=False)
    member = User(username="rmem", password_hash="h", is_admin=False)
    db.add_all([admin, member])
    db.flush()
    admins = {int(admin.id)}
    ats.set_agent_team_scope_hook(
        lambda _db, uid: (
            AgentTeamScope(team_id=300, is_team_admin=uid in admins)
            if uid in {int(admin.id), int(member.id)}
            else None
        )
    )
    yield admin, member
    ats.set_agent_team_scope_hook(None)
