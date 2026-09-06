"""The runtime-context view and the tool loaders agree on visibility, for
both connector kinds.

One fixture seeded with all four connector categories (owner-active,
owner-inactive, team-only, stranger) simultaneously per kind, plus one
OAuth-without-grant MCP row so the ``config["server_id"]`` fallback below is
actually exercised (the MCP tool loader's "unavailable" shape has no
top-level ``id``; the custom-API loader always emits one, so no such
fallback is needed on that half). The matrix varies the team parameter over
``{T1, None}`` on the run owner, plus one ``T2`` cell (so the agent's team
differs from the run owner's own team, per the run owner's
agent-team-scope mapping below) -- trimmed to the cells that each
discriminate a distinct case rather than repeating one already covered.

Scope note: this asserts parity on the **live** path only, for both
connector kinds. ``_load_visible_runtime_connectors`` and the custom-API
tool loaders in ``config.py`` are now team-keyed together (see
``test_new_hook_branch_unions_team_custom_api_too`` in
``test_connector_team_scope.py``), so the custom-API half of this parity
check is live, not trivial, the same way the MCP half always was. The
custom-API snapshot/prefetch path is covered separately, by its own
``test_snapshot_and_live_custom_api_paths_agree``.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from xagent.web.models import Base, MCPServer, User, UserMCPServer
from xagent.web.models.custom_api import CustomApi, UserCustomApi
from xagent.web.services import agent_team_scope, connector_team_scope
from xagent.web.services.connector_runtime import _load_visible_runtime_connectors
from xagent.web.tools.config import WebToolConfig

T1 = 101
T2 = 102


@pytest.fixture()
def db_session() -> Iterator[Session]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture(autouse=True)
def _reset_hooks() -> Iterator[None]:
    yield
    connector_team_scope.set_connector_team_hooks()
    agent_team_scope.set_agent_team_scope_hook(None)


def _create_user(db: Session, username: str) -> User:
    user = User(username=username, password_hash="hash")
    db.add(user)
    db.flush()
    return user


def _create_mcp(
    db: Session,
    name: str,
    *,
    owner: User | None = None,
    active: bool = True,
    transport: str = "streamable_http",
) -> MCPServer:
    server = MCPServer(
        name=name,
        description=f"{name} description",
        managed="external",
        transport=transport,
        url="https://example.com/mcp" if transport != "oauth" else None,
    )
    db.add(server)
    db.flush()
    if owner is not None:
        db.add(
            UserMCPServer(
                user_id=owner.id,
                mcpserver_id=server.id,
                is_owner=True,
                can_edit=True,
                can_delete=True,
                is_active=active,
            )
        )
        db.flush()
    return server


def _create_custom_api(
    db: Session, name: str, *, owner: User | None = None, active: bool = True
) -> CustomApi:
    api = CustomApi(
        name=name,
        description=f"{name} description",
        url="https://api.example.test",
        method="GET",
        headers={},
        body=None,
        env={},
    )
    db.add(api)
    db.flush()
    if owner is not None:
        db.add(
            UserCustomApi(
                user_id=owner.id,
                custom_api_id=api.id,
                is_owner=True,
                can_edit=True,
                can_delete=True,
                is_active=active,
            )
        )
        db.flush()
    return api


@pytest.fixture()
def seed(db_session: Session):
    c = _create_user(db_session, "run-owner")
    stranger_owner = _create_user(db_session, "stranger-owner")
    team_a_owner = _create_user(db_session, "team-a-owner")
    team_b_owner = _create_user(db_session, "team-b-owner")

    active_own = _create_mcp(db_session, "active-own", owner=c, active=True)
    inactive_own = _create_mcp(db_session, "inactive-own", owner=c, active=False)
    stranger = _create_mcp(db_session, "stranger", owner=stranger_owner)
    team_s = _create_mcp(db_session, "team-s")
    team_x = _create_mcp(db_session, "team-x")
    # OAuth server with no matching catalog app: exercises the tool loader's
    # "unavailable" shape, whose id lives at config["server_id"], not "id".
    oauth_no_grant = _create_mcp(
        db_session, "oauth-no-grant", owner=c, transport="oauth"
    )

    active_own_api = _create_custom_api(
        db_session, "active-own-api", owner=c, active=True
    )
    inactive_own_api = _create_custom_api(
        db_session, "inactive-own-api", owner=c, active=False
    )
    stranger_api = _create_custom_api(db_session, "stranger-api", owner=stranger_owner)
    team_a_api = _create_custom_api(db_session, "team-a-api", owner=team_a_owner)
    team_b_api = _create_custom_api(db_session, "team-b-api", owner=team_b_owner)
    # StaticPool shares one DBAPI connection across caller and worker Sessions.
    # Commit first because closing the worker can roll back shared pending rows.
    db_session.commit()

    connector_team_scope.set_connector_team_hooks(
        # inactive_own/inactive_own_api also carry a live T1 team grant here,
        # pinning the deliberate "no is_active veto" decision: a member's
        # deactivated personal link must not silently hide a connector the
        # team still shares. Without this, the inactive rows would only
        # ever pin "inactive personal link alone -> not visible", never the
        # combination that actually matters.
        team_visibility=lambda db, *, team_id: (
            {
                "mcp": {int(team_s.id), int(inactive_own.id)},
                "custom_api": {int(team_a_api.id), int(inactive_own_api.id)},
            }
            if team_id == T1
            else {
                "mcp": {int(team_x.id)},
                "custom_api": {int(team_b_api.id)},
            }
            if team_id == T2
            else {"mcp": set(), "custom_api": set()}
        )
    )
    # Negative-control fixture (see test_mcp_team_visibility.py's
    # ``_install_env_t`` docstring): maps the run owner to T1 so a
    # runner-keyed misimplementation is distinguishable from the correct
    # agent-keyed one on the ``team=None`` cell.
    agent_team_scope.set_agent_team_scope_hook(
        lambda db, user_id: (
            agent_team_scope.AgentTeamScope(team_id=T1, is_team_admin=False)
            if user_id == int(c.id)
            else None
        )
    )
    return SimpleNamespace(
        c=c,
        active_own=active_own,
        inactive_own=inactive_own,
        stranger=stranger,
        team_s=team_s,
        team_x=team_x,
        oauth_no_grant=oauth_no_grant,
        active_own_api=active_own_api,
        inactive_own_api=inactive_own_api,
        stranger_api=stranger_api,
        team_a_api=team_a_api,
        team_b_api=team_b_api,
    )


async def _parity_ids(
    db_session: Session, seed, *, team: int | None, connector_type: str
) -> tuple[set[int], set[int]]:
    visible = _load_visible_runtime_connectors(
        db_session, user_id=int(seed.c.id), agent_team_id=team
    )
    runtime_ids = {
        ref.connector_id for ref in visible if ref.connector_type == connector_type
    }

    cfg = WebToolConfig(
        db=db_session,
        request=None,
        user_id=int(seed.c.id),
        connector_team_id=team,
        include_mcp_tools=True,
    )
    if connector_type == "mcp":
        configs = await cfg._load_mcp_server_configs()
        loader_ids = {
            c.get("id") or c.get("config", {}).get("server_id") for c in configs
        }
    else:
        # The custom-API loader always emits a top-level "id"
        # (_custom_api_config_from_model, config.py:781), so no
        # _build_unavailable_mcp_config-style fallback is needed here --
        # that fallback stays on the MCP half above.
        configs = cfg.get_custom_api_configs()
        loader_ids = {c["id"] for c in configs}
    return runtime_ids, loader_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("team", [T1, None, T2])
async def test_runtime_view_and_tool_config_agree_mcp(db_session, seed, team):
    runtime_ids, loader_ids = await _parity_ids(
        db_session, seed, team=team, connector_type="mcp"
    )
    assert runtime_ids == loader_ids

    # Every seeded category is represented in the agreed-on set so the
    # parity assertion isn't vacuous for any row of the trimmed matrix.
    assert int(seed.active_own.id) in loader_ids
    assert int(seed.oauth_no_grant.id) in loader_ids
    assert int(seed.stranger.id) not in loader_ids
    if team == T1:
        assert int(seed.team_s.id) in loader_ids
        assert int(seed.team_x.id) not in loader_ids
        # No is_active veto: a deactivated personal link plus a live T1
        # team grant still resolves visible, in parity on both read points
        # -- the team grant is not blocked by the member's own inactive
        # association.
        assert int(seed.inactive_own.id) in loader_ids
    elif team == T2:
        assert int(seed.team_x.id) in loader_ids
        assert int(seed.team_s.id) not in loader_ids
        # inactive_own has no T2 grant -- the deactivated personal link
        # alone stays not-visible.
        assert int(seed.inactive_own.id) not in loader_ids
    else:
        assert int(seed.team_s.id) not in loader_ids
        assert int(seed.team_x.id) not in loader_ids
        assert int(seed.inactive_own.id) not in loader_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("team", [T1, None, T2])
async def test_runtime_view_and_tool_config_agree_custom_api(db_session, seed, team):
    runtime_ids, loader_ids = await _parity_ids(
        db_session, seed, team=team, connector_type="custom_api"
    )
    assert runtime_ids == loader_ids

    assert int(seed.active_own_api.id) in loader_ids
    assert int(seed.stranger_api.id) not in loader_ids
    if team == T1:
        assert int(seed.team_a_api.id) in loader_ids
        assert int(seed.team_b_api.id) not in loader_ids
        # No is_active veto, on the custom-API half too.
        assert int(seed.inactive_own_api.id) in loader_ids
    elif team == T2:
        assert int(seed.team_b_api.id) in loader_ids
        assert int(seed.team_a_api.id) not in loader_ids
        assert int(seed.inactive_own_api.id) not in loader_ids
    else:
        assert int(seed.team_a_api.id) not in loader_ids
        assert int(seed.team_b_api.id) not in loader_ids
        assert int(seed.inactive_own_api.id) not in loader_ids
