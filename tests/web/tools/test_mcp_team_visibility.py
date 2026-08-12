"""Team-scope connector visibility at the MCP tool-load boundary.

``WebToolConfig._load_mcp_server_configs`` resolves the servers visible to a
run from ``personal ∪ the governing agent's team`` instead of ``personal
only``, through the optional ``team_visibility`` hook on
``connector_team_scope``. These tests pin the closed-by-default shape (no
hook installed, or a hook installed but no team supplied) as well as the
positive team-keyed case and the two failure-contract invariants the seam
now carries (a broken hook fails the turn instead of dropping the MCP tool
set silently).

Fixture seed (five MCP rows, run owner ``C`` throughout unless noted):

    active_own    -- C holds an active personal link
    inactive_own  -- C holds an inactive personal link
    stranger      -- owned by a third user, no link to anything
    team_s        -- reachable only through a team hook keyed on T1
    team_x        -- reachable only through a team hook keyed on T2
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from xagent.core.tools.adapters.vibe.mcp_tools import create_mcp_tools
from xagent.web.models import Base, MCPServer, User, UserMCPServer
from xagent.web.services import agent_team_scope, connector_team_scope
from xagent.web.tools.config import WebToolConfig

T1 = 101
T2 = 102


class _ProbeError(RuntimeError):
    """Distinguishable failure raised by a broken team-visibility hook."""


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
    db: Session, name: str, *, owner: User | None = None, active: bool = True
) -> MCPServer:
    server = MCPServer(
        name=name,
        description=f"{name} description",
        managed="external",
        transport="streamable_http",
        url="https://example.com/mcp",
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


@pytest.fixture()
def seed(db_session: Session):
    c = _create_user(db_session, "run-owner")
    z = _create_user(db_session, "stranger-owner")
    active_own = _create_mcp(db_session, "active-own", owner=c, active=True)
    inactive_own = _create_mcp(db_session, "inactive-own", owner=c, active=False)
    stranger = _create_mcp(db_session, "stranger", owner=z)
    team_s = _create_mcp(db_session, "team-s")
    team_x = _create_mcp(db_session, "team-x")
    return SimpleNamespace(
        c=c,
        z=z,
        active_own=active_own,
        inactive_own=inactive_own,
        stranger=stranger,
        team_s=team_s,
        team_x=team_x,
    )


def install_team_hooks(*, team_visibility, agent_owner_id: int) -> None:
    """Install a ``team_visibility`` hook *and* the agent-team-scope hook
    mapping ``agent_owner_id`` -> T1.

    Shared by the MCP and custom-API team-visibility suites so the
    hook-*install* pattern is written once; each suite's own autouse
    ``_reset_hooks`` undoes both hooks after every test -- this module's
    copy and the custom-API suite's separate copy are two independent
    fixtures, not one shared teardown.

    The scope-hook mapping is a negative-control fixture, not something
    production code consults for this decision (both tool loaders key on
    ``self._connector_team_id``, never on ``get_agent_team_scope(db, owner)``)
    -- but without it installed, a runner-keyed misimplementation would be
    indistinguishable from the correct one on these fixtures, since
    ``get_agent_team_scope`` would resolve ``None`` for everybody either way.
    """
    connector_team_scope.set_connector_team_hooks(team_visibility=team_visibility)
    agent_team_scope.set_agent_team_scope_hook(
        lambda db, user_id: (
            agent_team_scope.AgentTeamScope(team_id=T1, is_team_admin=False)
            if user_id == agent_owner_id
            else None
        )
    )


def _team_visibility_hook(seed):
    """A team_visibility hook whose answer is disjoint per team, empty otherwise."""

    def _hook(db, *, team_id):
        if team_id == T1:
            return {"mcp": {int(seed.team_s.id)}, "custom_api": set()}
        if team_id == T2:
            return {"mcp": {int(seed.team_x.id)}, "custom_api": set()}
        return {"mcp": set(), "custom_api": set()}

    return _hook


def _install_env_t(seed) -> None:
    """Install the team_visibility hook *and* the agent-team-scope hook
    mapping the run owner C -> T1 (the team that owns ``team_s``)."""
    install_team_hooks(
        team_visibility=_team_visibility_hook(seed), agent_owner_id=int(seed.c.id)
    )


def _cfg(db_session: Session, seed, *, connector_team_id: int | None) -> WebToolConfig:
    return WebToolConfig(
        db=db_session,
        request=None,
        user_id=int(seed.c.id),
        connector_team_id=connector_team_id,
        include_mcp_tools=True,
    )


# ---------------------------------------------------------------------------
# Standalone (no hooks installed) is result-identical to today, regardless
# of what connector_team_id is passed -- there is nothing for it to resolve
# against without a hook.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("connector_team_id", [None, 7])
async def test_no_hooks_matches_legacy_result_set(db_session, seed, connector_team_id):
    from xagent.web.services.connector_runtime import _load_visible_runtime_connectors

    cfg = _cfg(db_session, seed, connector_team_id=connector_team_id)
    configs = await cfg._load_mcp_server_configs()
    assert {c["name"] for c in configs} == {seed.active_own.name}

    visible = _load_visible_runtime_connectors(
        db_session, user_id=int(seed.c.id), agent_team_id=None
    )
    mcp_ids = {ref.connector_id for ref in visible if ref.connector_type == "mcp"}
    assert mcp_ids == {int(seed.active_own.id)}


# ---------------------------------------------------------------------------
# A team agent resolves its team's connector for a run owner with no
# personal row on that server.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_team_agent_loads_team_connector(db_session, seed):
    _install_env_t(seed)
    cfg = _cfg(db_session, seed, connector_team_id=T1)
    configs = await cfg._load_mcp_server_configs()
    assert {c["name"] for c in configs} == {seed.active_own.name, seed.team_s.name}


# ---------------------------------------------------------------------------
# A personal agent (no governing team) gets nothing from any team, and the
# authorization path is read-only: no personal link is materialized.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_personal_agent_gets_no_team_connector(db_session, seed):
    from xagent.web.services.connector_runtime import _load_visible_runtime_connectors

    _install_env_t(seed)
    cfg = _cfg(db_session, seed, connector_team_id=None)
    configs = await cfg._load_mcp_server_configs()
    assert {c["name"] for c in configs} == {seed.active_own.name}

    visible = _load_visible_runtime_connectors(
        db_session, user_id=int(seed.c.id), agent_team_id=None
    )
    mcp_ids = {ref.connector_id for ref in visible if ref.connector_type == "mcp"}
    assert mcp_ids == {int(seed.active_own.id)}

    before = db_session.query(UserMCPServer).filter_by(user_id=int(seed.c.id)).count()
    assert before == 2  # active_own + inactive_own, seeded, unchanged by resolution


# ---------------------------------------------------------------------------
# The production MCP query is a semi-join with a deterministic order.
# Compiled on the production query object, not a parallel reconstruction.
# ---------------------------------------------------------------------------


def test_production_mcp_query_shape(db_session, seed):
    cfg = _cfg(db_session, seed, connector_team_id=T1)
    sql = str(
        cfg._visible_mcp_server_query(
            frozenset({int(seed.team_s.id)})
        ).statement.compile(dialect=postgresql.dialect())
    )
    norm = " ".join(sql.split()).upper()
    assert norm.startswith("SELECT")
    assert "MCP_SERVERS" in norm
    assert "JOIN USER_MCPSERVERS" not in norm
    assert norm.endswith("ORDER BY MCP_SERVERS.ID")  # exact ordering column


# ---------------------------------------------------------------------------
# A team-hook failure is not folded into MCPConfigLoadError -- it is raised
# from outside that error's guarded region -- and it survives the
# tool-creator boundary as ConnectorRuntimeError instead of being dropped
# into an empty MCP tool set (checked at two frames: the direct creator
# call, and through the tool registry).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_team_hook_failure_is_not_mcp_config_load_error(db_session, seed):
    def _boom(db, *, team_id):
        raise _ProbeError("boom")

    connector_team_scope.set_connector_team_hooks(team_visibility=_boom)
    cfg = _cfg(db_session, seed, connector_team_id=T1)

    with pytest.raises(ConnectorRuntimeError) as excinfo:
        await cfg._load_mcp_server_configs()

    assert excinfo.value.details["reason"] == "team_scope_resolution_failed"
    assert isinstance(excinfo.value.__cause__, _ProbeError)


@pytest.mark.asyncio
async def test_team_hook_failure_surfaces_as_connector_runtime_error(db_session, seed):
    def _boom(db, *, team_id):
        raise _ProbeError("boom")

    connector_team_scope.set_connector_team_hooks(team_visibility=_boom)
    cfg = _cfg(db_session, seed, connector_team_id=T1)

    # Positive control: prove the creator is not short-circuiting before the
    # seam because no selection spec excludes MCP.
    assert cfg.get_tool_selection_spec() is None

    with pytest.raises(ConnectorRuntimeError) as excinfo:
        await create_mcp_tools(cfg)
    assert excinfo.value.details["reason"] == "team_scope_resolution_failed"


@pytest.mark.asyncio
async def test_team_hook_failure_survives_tool_registry_boundary(db_session, seed):
    """Companion to the direct-creator assertion above: without this frame the
    test proves only that the seam raises, not that anybody downstream is
    left to hear it. ``factory.py``'s registry loop re-raises
    ``ConnectorRuntimeError`` specifically and would otherwise swallow an
    untyped exception into a WARNING and an empty tool list.
    """
    from xagent.core.tools.adapters.vibe.factory import ToolRegistry
    from xagent.core.tools.adapters.vibe.mcp_tools import (
        create_mcp_tools as real_create_mcp_tools,
    )

    def _boom(db, *, team_id):
        raise _ProbeError("boom")

    connector_team_scope.set_connector_team_hooks(team_visibility=_boom)
    cfg = _cfg(db_session, seed, connector_team_id=T1)

    saved_creators = list(ToolRegistry._tool_creators)
    saved_imported = ToolRegistry._modules_imported
    ToolRegistry._tool_creators = []
    ToolRegistry._modules_imported = True
    try:
        ToolRegistry.register(
            real_create_mcp_tools, categories={"mcp"}, selection_gate="mcp"
        )
        with pytest.raises(ConnectorRuntimeError) as excinfo:
            await ToolRegistry.create_registered_tools(cfg)
    finally:
        ToolRegistry._tool_creators = saved_creators
        ToolRegistry._modules_imported = saved_imported

    assert excinfo.value.details["reason"] == "team_scope_resolution_failed"


# ---------------------------------------------------------------------------
# The omitted connector_team_id parameter's default is the closed one.
#
# Upstream commit 9891cad8 ("fix(auth): preserve infrastructure failures at
# authentication boundaries") removed WebToolConfig's id-1 authentication
# fallback. Before that commit, an unidentified construction (``user_id``
# omitted, no request-derived identity) silently became user 1. After that
# commit, an unidentified construction keeps ``_user_id`` as ``None`` --
# there is no fallback identity to silently become. This is a stronger
# closed-default contract, not a weaker one: the production entry point,
# ``get_mcp_server_configs()``, now short-circuits to ``[]`` whenever
# ``self._user_id is None``, before this seam's team-scope resolution is
# ever reached in production -- so an unauthenticated construction gets
# nothing at all, not merely "personal-only". Split into three tests below:
# the closed default itself, the production-reachable guarantee (via the
# public entry point), and one test documenting -- not fixing -- the
# private loader's own lack of an independent identity guard, since this
# suite calls that private method directly throughout for granularity.
# ---------------------------------------------------------------------------


def test_omitted_connector_team_id_defaults_to_none(db_session, seed):
    # ``connector_team_id`` omitted entirely -- not even ``None`` passed --
    # so this actually exercises the constructor's own default, not merely
    # an explicit None round-tripping.
    cfg = WebToolConfig(
        db=db_session,
        request=None,
        user_id=int(seed.c.id),
        include_mcp_tools=True,
    )
    assert cfg._connector_team_id is None


@pytest.mark.asyncio
async def test_unidentified_construction_resolves_no_mcp_configs_regardless_of_team(
    db_session,
):
    """The production-reachable contract: no identity means no MCP tools,
    even when a team grant exists and ``connector_team_id`` names it -- the
    identity guard in ``get_mcp_server_configs()`` sits before the
    team-scope resolution entirely, so a team grant cannot leak through for
    an unauthenticated build."""
    team_server = _create_mcp(db_session, "team-only-unidentified-probe")
    connector_team_scope.set_connector_team_hooks(
        team_visibility=lambda db, *, team_id: (
            {"mcp": {int(team_server.id)}, "custom_api": set()}
            if team_id == T1
            else {"mcp": set(), "custom_api": set()}
        )
    )
    try:
        cfg = WebToolConfig(
            db=db_session,
            request=None,
            user_id=None,
            connector_team_id=T1,
            include_mcp_tools=True,
        )
        assert cfg._user_id is None

        configs = await cfg.get_mcp_server_configs()
        assert configs == []
    finally:
        connector_team_scope.set_connector_team_hooks()


@pytest.mark.asyncio
async def test_direct_private_loader_call_resolves_nothing_without_identity(db_session):
    """``_load_mcp_server_configs()`` has no identity guard of its own;
    ``get_mcp_server_configs()`` is its only production caller (``grep -n
    "_load_mcp_server_configs\\b" src/xagent/web/tools/config.py`` shows
    exactly one call site, inside that wrapper), and that wrapper's identity
    guard is what keeps an unauthenticated build from ever reaching this
    seam's team-scope resolution in production.

    Calling the private method directly -- as this test suite already does
    throughout, for granularity -- bypasses that guard, but resolves
    nothing rather than leaking the team connector: the visibility
    predicate itself short-circuits to the personal arm whenever the owner
    id is ``None``, regardless of any team ids supplied, so the query never
    matches the team-only server in the first place and no per-server
    config is ever built for it. This test exists so a future reader of
    this file's many other ``_load_mcp_server_configs()`` calls does not
    mistake "reachable in a unit test" for "reachable in production", and
    knows the measured behaviour without identity is silence, not a leak.
    """
    team_server = _create_mcp(db_session, "team-only-private-loader-probe")
    connector_team_scope.set_connector_team_hooks(
        team_visibility=lambda db, *, team_id: (
            {"mcp": {int(team_server.id)}, "custom_api": set()}
            if team_id == T1
            else {"mcp": set(), "custom_api": set()}
        )
    )
    try:
        cfg = WebToolConfig(
            db=db_session,
            request=None,
            user_id=None,
            connector_team_id=T1,
            include_mcp_tools=True,
        )
        configs = await cfg._load_mcp_server_configs()
        assert configs == []
    finally:
        connector_team_scope.set_connector_team_hooks()
