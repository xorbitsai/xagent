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

import asyncio
import threading
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool

from xagent.core.tools.adapters.vibe.config import MCPConfigLoadError
from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from xagent.core.tools.adapters.vibe.mcp_tools import create_mcp_tools
from xagent.web.models import Base, MCPServer, User, UserMCPServer
from xagent.web.services import agent_team_scope, connector_team_scope
from xagent.web.tools import config as config_module
from xagent.web.tools.config import WebToolConfig

T1 = 101
T2 = 102


class _ProbeError(RuntimeError):
    """Distinguishable failure raised by a broken team-visibility hook."""


class _QueryDb:
    """Non-Session database double with the loader's supported query surface."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def query(self, *entities):
        return self._session.query(*entities)


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
    # StaticPool shares one DBAPI connection across caller and worker Sessions.
    # Commit first because closing the worker can roll back shared pending rows.
    db_session.commit()
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
    team_s_id = int(seed.team_s.id)
    team_x_id = int(seed.team_x.id)

    def _hook(db, *, team_id):
        if team_id == T1:
            return {"mcp": {team_s_id}, "custom_api": set()}
        if team_id == T2:
            return {"mcp": {team_x_id}, "custom_api": set()}
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
async def test_mcp_team_snapshot_uses_query_shaped_db_inline(db_session, seed):
    team_server_id = int(seed.team_s.id)
    hook_databases: list[object] = []
    query_db = _QueryDb(db_session)

    def team_visibility(hook_db, *, team_id):
        hook_databases.append(hook_db)
        return {
            "mcp": {team_server_id} if team_id == T1 else set(),
            "custom_api": set(),
        }

    install_team_hooks(
        team_visibility=team_visibility,
        agent_owner_id=int(seed.c.id),
    )
    cfg = WebToolConfig(
        db=query_db,
        request=None,
        user_id=int(seed.c.id),
        connector_team_id=T1,
        include_mcp_tools=True,
    )

    configs = await cfg._load_mcp_server_configs()

    assert hook_databases == [query_db]
    assert {config["name"] for config in configs} == {
        seed.active_own.name,
        seed.team_s.name,
    }


@pytest.mark.asyncio
async def test_mcp_team_snapshot_resolves_factory_before_worker(
    db_session, seed, monkeypatch
):
    team_server_id = int(seed.team_s.id)
    connector_team_scope.set_connector_team_hooks(
        team_visibility=lambda db, *, team_id: {
            "mcp": {team_server_id} if team_id == T1 else set(),
            "custom_api": set(),
        }
    )
    cfg = _cfg(db_session, seed, connector_team_id=T1)
    main_thread_id = threading.get_ident()
    factory_thread_ids: list[int] = []
    worker_thread_ids: list[int] = []
    factory_resolved = False
    caller_released = False
    original_get_session_factory = cfg.get_session_factory
    original_release = cfg.release_db_connection

    def record_session_factory():
        nonlocal factory_resolved
        assert not caller_released
        factory_thread_ids.append(threading.get_ident())
        factory_resolved = True
        session_factory = original_get_session_factory()

        def worker_session_factory():
            worker_thread_ids.append(threading.get_ident())
            return session_factory()

        return worker_session_factory

    def record_release():
        nonlocal caller_released
        assert factory_resolved
        caller_released = True
        return original_release()

    monkeypatch.setattr(cfg, "get_session_factory", record_session_factory)
    monkeypatch.setattr(cfg, "release_db_connection", record_release)

    configs = await cfg._load_mcp_server_configs()

    assert factory_thread_ids == [main_thread_id]
    assert worker_thread_ids and worker_thread_ids[0] != main_thread_id
    assert {config["name"] for config in configs} == {
        seed.active_own.name,
        seed.team_s.name,
    }


@pytest.mark.asyncio
async def test_team_hook_wait_does_not_block_event_loop(db_session, seed):
    release = threading.Event()
    wait_results: list[bool] = []
    ticks_during_hook: list[int] = []
    # This diagnostic targets CPython. Its GIL serializes these integer reads
    # and writes, and the threshold allows substantial scheduling variance.
    ticks = 0
    stop = False
    team_server_id = int(seed.team_s.id)

    def blocking_visibility(db, *, team_id):
        ticks_before_wait = ticks
        timer = threading.Timer(0.1, release.set)
        timer.daemon = True
        timer.start()
        wait_results.append(release.wait(timeout=1))
        ticks_during_hook.append(ticks - ticks_before_wait)
        return {
            "mcp": {team_server_id} if team_id == T1 else set(),
            "custom_api": set(),
        }

    async def ticker() -> None:
        nonlocal ticks
        while not stop:
            ticks += 1
            await asyncio.sleep(0.01)

    install_team_hooks(
        team_visibility=blocking_visibility,
        agent_owner_id=int(seed.c.id),
    )
    ticker_task = asyncio.create_task(ticker())
    try:
        configs = await _cfg(
            db_session, seed, connector_team_id=T1
        )._load_mcp_server_configs()
    finally:
        stop = True
        await ticker_task

    assert wait_results == [True]
    assert ticks_during_hook[0] >= 3
    assert {config["name"] for config in configs} == {
        seed.active_own.name,
        seed.team_s.name,
    }


@pytest.mark.asyncio
async def test_mcp_team_snapshot_drains_worker_before_cancellation(db_session, seed):
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    wait_results: list[bool] = []

    def blocking_visibility(db, *, team_id):
        started.set()
        wait_results.append(release.wait(timeout=1))
        finished.set()
        return {"mcp": set(), "custom_api": set()}

    install_team_hooks(
        team_visibility=blocking_visibility,
        agent_owner_id=int(seed.c.id),
    )
    caller = asyncio.create_task(
        _cfg(db_session, seed, connector_team_id=T1)._load_mcp_server_configs()
    )
    assert await asyncio.to_thread(started.wait, 1)

    caller.cancel()
    await asyncio.sleep(0.02)
    assert not caller.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(caller, timeout=1)

    assert wait_results == [True]
    assert finished.is_set()


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
    team_server_id = int(team_server.id)
    db_session.commit()
    connector_team_scope.set_connector_team_hooks(
        team_visibility=lambda db, *, team_id: (
            {"mcp": {team_server_id}, "custom_api": set()}
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


@pytest.mark.asyncio
async def test_mcp_team_snapshot_maps_unexpected_error_to_config_error(
    db_session, seed, monkeypatch
):
    failure = _ProbeError("unexpected worker failure")

    def fail_snapshot(*args, **kwargs):
        raise failure

    connector_team_scope.set_connector_team_hooks(
        team_visibility=lambda db, *, team_id: {
            "mcp": set(),
            "custom_api": set(),
        }
    )
    monkeypatch.setattr(config_module, "_run_with_checked_out_session", fail_snapshot)
    cfg = _cfg(db_session, seed, connector_team_id=T1)

    with pytest.raises(MCPConfigLoadError) as excinfo:
        await cfg._load_mcp_server_configs()

    assert excinfo.value.__cause__ is failure


@pytest.mark.asyncio
async def test_mcp_team_snapshot_maps_pool_timeout_to_config_error(tmp_path):
    """A saturated worker checkout must keep MCP failure handling typed."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'mcp-team-timeout.db'}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.05,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine)
    db = factory()
    try:
        user = User(username="pending-team-user", password_hash="hash")
        db.add(user)
        db.flush()
        connector_team_scope.set_connector_team_hooks(
            team_visibility=lambda db, *, team_id: {
                "mcp": set(),
                "custom_api": set(),
            }
        )
        cfg = WebToolConfig(
            db=db,
            request=None,
            user_id=int(user.id),
            connector_team_id=T1,
            include_mcp_tools=True,
        )

        with pytest.raises(MCPConfigLoadError) as exc_info:
            await cfg._load_mcp_server_configs()

        assert isinstance(exc_info.value.__cause__, SQLAlchemyTimeoutError)
    finally:
        db.rollback()
        db.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.mark.asyncio
async def test_mcp_team_snapshot_keeps_dirty_caller_session(tmp_path):
    """A dirty caller can keep its connection when the pool has spare capacity."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'mcp-team-dirty-caller.db'}",
        poolclass=QueuePool,
        pool_size=2,
        max_overflow=0,
        pool_timeout=0.05,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine)
    db = factory()
    try:
        user = _create_user(db, "dirty-caller-user")
        server = _create_mcp(db, "dirty-caller-server", owner=user)
        user_id = int(user.id)
        server_id = int(server.id)
        db.commit()

        pending = User(username="pending-dirty-user", password_hash="hash")
        db.add(pending)
        db.flush()

        def team_visibility(hook_db, *, team_id):
            hook_db.connection()
            return {
                "mcp": {server_id} if team_id == T1 else set(),
                "custom_api": set(),
            }

        connector_team_scope.set_connector_team_hooks(team_visibility=team_visibility)
        cfg = WebToolConfig(
            db=db,
            request=None,
            user_id=user_id,
            connector_team_id=T1,
            include_mcp_tools=True,
        )

        assert engine.pool.checkedout() == 1
        configs = await cfg._load_mcp_server_configs()

        assert {config["id"] for config in configs} == {server_id}
        assert engine.pool.checkedout() == 1
        assert pending in db
    finally:
        db.rollback()
        db.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.mark.parametrize(
    ("connector_team_id", "install_hook"),
    [(None, True), (T1, False)],
    ids=["no-team", "no-hook"],
)
@pytest.mark.asyncio
async def test_mcp_team_snapshot_skips_worker_without_team_scope(
    tmp_path, connector_team_id, install_hook
):
    """No team-hook work must not require a second pool connection."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'mcp-team-skip.db'}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.05,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine)
    db = factory()
    try:
        user = User(username="pending-user", password_hash="hash")
        db.add(user)
        db.flush()
        if install_hook:
            connector_team_scope.set_connector_team_hooks(
                team_visibility=lambda db, *, team_id: {
                    "mcp": set(),
                    "custom_api": set(),
                }
            )
        cfg = WebToolConfig(
            db=db,
            request=None,
            user_id=int(user.id),
            connector_team_id=connector_team_id,
            include_mcp_tools=True,
        )

        assert engine.pool.checkedout() == 1
        assert await cfg._load_mcp_server_configs() == []
        assert engine.pool.checkedout() == 1
    finally:
        db.rollback()
        db.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.mark.asyncio
async def test_mcp_team_snapshot_releases_clean_caller_session(tmp_path):
    """The worker checks out its own connection after the caller releases one."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'mcp-team-handoff.db'}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.05,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine)
    db = factory()
    main_thread_id = threading.get_ident()
    hook_session_ids: list[int] = []
    hook_thread_ids: list[int] = []
    try:
        user = _create_user(db, "handoff-user")
        server = _create_mcp(db, "handoff-server", owner=user)
        user_id = int(user.id)
        server_id = int(server.id)
        db.commit()

        def team_visibility(hook_db, *, team_id):
            hook_session_ids.append(id(hook_db))
            hook_thread_ids.append(threading.get_ident())
            return {
                "mcp": {server_id} if team_id == T1 else set(),
                "custom_api": set(),
            }

        connector_team_scope.set_connector_team_hooks(team_visibility=team_visibility)
        cfg = WebToolConfig(
            db=db,
            request=None,
            user_id=user_id,
            connector_team_id=T1,
            include_mcp_tools=True,
        )

        assert db.query(MCPServer).all() == [server]
        assert engine.pool.checkedout() == 1

        configs = await cfg._load_mcp_server_configs()

        assert {config["id"] for config in configs} == {server_id}
        assert hook_session_ids and hook_session_ids[0] != id(db)
        assert hook_thread_ids and hook_thread_ids[0] != main_thread_id
        assert engine.pool.checkedout() == 1

        db.rollback()
        assert engine.pool.checkedout() == 0
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()
