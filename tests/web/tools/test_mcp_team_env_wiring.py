"""Wiring pins for the team-keyed shared env layer inside
``WebToolConfig._load_mcp_server_configs``.

Companion to tests/web/tools/test_mcp_team_visibility.py, which pins *which*
servers a team-governed run can see. This file pins what env each
team-owned server resolves to: for a team-owned id, the shared env layer
comes from the *governing* team's own row (``mcp_runtime.set_mcp_team_env_hook``)
and from nothing else -- never the running user's own team, never any
other team, never a degraded/partial mix -- while ids outside the team's
visible set and runs with no governing team keep the pre-existing
personal/legacy behaviour unchanged.

Loader-level unit pins for the hook slot itself (predicate, validator,
snapshot primitive) live in
tests/web/services/test_mcp_team_env_hook.py; this file only pins how
config.py's wiring block consumes that loader.

Fixture seed (one stdio MCP row reachable only through the team hook, one
stdio MCP row owned personally by the run owner, run owner ``C``
throughout):

    team_server      -- reachable only through a team-visibility hook keyed
                         on T1 (the governing team in every test below)
    personal_server   -- C holds an active personal link, never surfaced by
                         any team-visibility hook
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Iterator
from types import SimpleNamespace
from typing import cast

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from xagent.core.tools.adapters.vibe.mcp_tools import create_mcp_tools
from xagent.web.models import Base, MCPServer, User, UserMCPServer
from xagent.web.services import agent_team_scope, connector_team_scope, mcp_runtime
from xagent.web.tools.config import (
    WebToolConfig,
    _load_mcp_team_hook_snapshot_from_db,
)

T1 = 301  # the governing team throughout this file
T2 = 302  # a team the runner might also belong to, never the governing one


class _ProbeError(RuntimeError):
    """Distinguishable failure raised by a broken team-env hook."""


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
    # The env hook slots restore through the module's own snapshot
    # primitive; the two team-scope hooks below live in other modules and
    # have no such primitive, so they are still reset by hand.
    with mcp_runtime.mcp_env_hooks_snapshot():
        yield
        connector_team_scope.set_connector_team_hooks()
        agent_team_scope.set_agent_team_scope_hook(None)


def _create_user(db: Session, username: str) -> User:
    user = User(username=username, password_hash="hash")
    db.add(user)
    db.flush()
    return user


def _create_stdio_mcp(
    db: Session,
    name: str,
    *,
    owner: User | None = None,
    env: dict | None = None,
) -> MCPServer:
    server = MCPServer(
        name=name,
        description=f"{name} description",
        managed="external",
        transport="stdio",
        command="python",
        args=["-m", "probe"],
        env=env,
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
                is_active=True,
            )
        )
        db.flush()
    return server


def _link_own_env(
    db: Session,
    *,
    user: User,
    server: MCPServer,
    env: dict | None,
    env_source: str | None,
) -> None:
    """Give ``user`` an ACTIVE personal env link to ``server``. The
    cross-team-borrowing test below needs this, otherwise it would pass
    under the pre-PR rule for reasons unrelated to the invariant under
    test."""
    db.add(
        UserMCPServer(
            user_id=user.id,
            mcpserver_id=server.id,
            is_owner=False,
            can_edit=False,
            can_delete=False,
            is_active=True,
            env=env,
            env_source=env_source,
        )
    )
    db.flush()


@pytest.fixture()
def seed(db_session: Session):
    c = _create_user(db_session, "run-owner")
    team_server = _create_stdio_mcp(
        db_session, "team-server", env={"GLOBAL": "global-value"}
    )
    personal_server = _create_stdio_mcp(
        db_session, "personal-server", owner=c, env={"GLOBAL": "global-value"}
    )
    # StaticPool shares one DBAPI connection across caller and worker Sessions.
    # Commit first because closing the worker can roll back shared pending rows.
    db_session.commit()
    return SimpleNamespace(
        c=c, team_server=team_server, personal_server=personal_server
    )


def _install_visibility(*, ids_by_team: dict[int, set[int]]) -> None:
    connector_team_scope.set_connector_team_hooks(
        team_visibility=lambda db, *, team_id: {
            "mcp": ids_by_team.get(team_id, set()),
            "custom_api": set(),
        }
    )


def _cfg(db_session: Session, seed, *, connector_team_id: int | None) -> WebToolConfig:
    return WebToolConfig(
        db=db_session,
        request=None,
        user_id=int(seed.c.id),
        connector_team_id=connector_team_id,
        include_mcp_tools=True,
    )


async def _env_for(db_session, seed, server, *, connector_team_id) -> dict:
    # Tests add links after the seed fixture commits. Commit those per-test rows
    # here before the worker snapshot; the seed fixture cannot include them.
    db_session.commit()
    cfg = _cfg(db_session, seed, connector_team_id=connector_team_id)
    configs = await cfg._load_mcp_server_configs()
    (config,) = [c for c in configs if c["id"] == int(server.id)]
    env = dict(config["config"].get("env") or {})
    env.pop("XAGENT_MCP_CALLER_ID", None)
    return env


def _create_http_mcp(db: Session, name: str) -> MCPServer:
    """A team-owned server on a non-stdio transport. The env layers are a
    stdio-only concept, so this exists to pin that they stay there."""
    server = MCPServer(
        name=name,
        description=f"{name} description",
        managed="external",
        transport="streamable_http",
        url="https://example.com/mcp",
    )
    db.add(server)
    db.flush()
    return server


async def _config_for(db_session, seed, server, *, connector_team_id) -> dict:
    # Tests add links after the seed fixture commits. Commit those per-test rows
    # here before the worker snapshot; the seed fixture cannot include them.
    db_session.commit()
    cfg = _cfg(db_session, seed, connector_team_id=connector_team_id)
    configs = await cfg._load_mcp_server_configs()
    (config,) = [c for c in configs if c["id"] == int(server.id)]
    return config


# ---------------------------------------------------------------------------
# Team-owned id + governing-team row => that row is the shared layer,
# for a runner the agent-team-scope hook calls a member and one it doesn't
# -- the resolution never consults that hook at all, so the outcome
# must be identical either way.
# ---------------------------------------------------------------------------


def test_team_env_snapshot_mapping_is_read_only(db_session, seed):
    server_id = int(seed.team_server.id)
    _install_visibility(ids_by_team={T1: {server_id}})
    mcp_runtime.set_mcp_team_env_hook(
        lambda db, *, team_id: {server_id: {"TEAM_KEY": "team-value"}}
    )

    snapshot = _load_mcp_team_hook_snapshot_from_db(
        db_session,
        user_id=int(seed.c.id),
        connector_team_id=T1,
    )

    assert snapshot.team_env_by_id[server_id]["TEAM_KEY"] == "team-value"
    with pytest.raises(TypeError):
        cast(dict[int, dict[str, str]], snapshot.team_env_by_id)[server_id] = {}


def test_team_env_snapshot_detaches_nested_hook_values(db_session, seed):
    server_id = int(seed.team_server.id)
    hook_owned = {server_id: {"TEAM_KEY": "initial"}}
    _install_visibility(ids_by_team={T1: {server_id}})
    mcp_runtime.set_mcp_team_env_hook(lambda db, *, team_id: hook_owned)

    snapshot = _load_mcp_team_hook_snapshot_from_db(
        db_session,
        user_id=int(seed.c.id),
        connector_team_id=T1,
    )
    hook_owned[server_id]["TEAM_KEY"] = "mutated"

    assert snapshot.team_env_by_id[server_id]["TEAM_KEY"] == "initial"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner_is_member", [True, False], ids=["member", "non-member"]
)
async def test_team_env_layer_keyed_on_governing_team(
    db_session, seed, runner_is_member
):
    team_server_id = int(seed.team_server.id)
    run_user_id = int(seed.c.id)
    _install_visibility(ids_by_team={T1: {team_server_id}})
    mcp_runtime.set_mcp_team_env_hook(
        lambda db, *, team_id: (
            {team_server_id: {"TEAM_KEY": "team-value"}} if team_id == T1 else {}
        )
    )
    agent_team_scope.set_agent_team_scope_hook(
        lambda db, user_id: (
            agent_team_scope.AgentTeamScope(team_id=T1, is_team_admin=False)
            if runner_is_member and user_id == run_user_id
            else None
        )
    )

    env = await _env_for(db_session, seed, seed.team_server, connector_team_id=T1)
    assert env.get("TEAM_KEY") == "team-value"


@pytest.mark.asyncio
async def test_team_env_hook_wait_does_not_block_event_loop(db_session, seed):
    _install_visibility(ids_by_team={T1: {int(seed.team_server.id)}})
    release = threading.Event()
    wait_results: list[bool] = []
    ticks_during_hook: list[int] = []
    # This diagnostic targets CPython. Its GIL serializes these integer reads
    # and writes, and the threshold allows substantial scheduling variance.
    ticks = 0
    stop = False
    team_server_id = int(seed.team_server.id)

    def blocking_team_env(db, *, team_id):
        ticks_before_wait = ticks
        timer = threading.Timer(0.1, release.set)
        timer.daemon = True
        timer.start()
        wait_results.append(release.wait(timeout=1))
        ticks_during_hook.append(ticks - ticks_before_wait)
        return {team_server_id: {"TEAM_KEY": "team-value"}}

    async def ticker() -> None:
        nonlocal ticks
        while not stop:
            ticks += 1
            await asyncio.sleep(0.01)

    mcp_runtime.set_mcp_team_env_hook(blocking_team_env)
    ticker_task = asyncio.create_task(ticker())
    try:
        config = await _config_for(
            db_session,
            seed,
            seed.team_server,
            connector_team_id=T1,
        )
    finally:
        stop = True
        await ticker_task

    assert wait_results == [True]
    assert ticks_during_hook[0] >= 3
    assert config["config"]["env"]["TEAM_KEY"] == "team-value"


# ---------------------------------------------------------------------------
# The hook is never invoked -- not merely "invoked and answers
# empty" -- when connector_team_id is None, or when team_mcp_ids is empty.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connector_team_id_none_never_invokes_team_env_hook(
    db_session, seed, monkeypatch
):
    """In practice team_mcp_ids is already empty whenever connector_team_id
    is None (resolve_team_connector_ids_or_raise short-circuits on
    ``team_id is None`` before ever calling the visibility hook), and
    load_team_env_overrides has its own ``team_id is None`` guard too, so
    this forces team_mcp_ids non-empty AND spies on the loader call itself
    (rather than the hook) to isolate config.py's own
    ``connector_team_id is not None`` condition from both of those other
    guards: it must gate the wiring block on its own terms, not only rely
    on either guard beneath it."""
    from xagent.web.services import connector_team_scope
    from xagent.web.services import mcp_runtime as mcp_runtime_module

    monkeypatch.setattr(
        connector_team_scope,
        "resolve_team_connector_ids_or_raise",
        lambda db, *, team_id, log_subject: {
            "mcp": {int(seed.team_server.id)},
            "custom_api": set(),
        },
    )
    mcp_runtime.set_mcp_team_env_hook(lambda db, *, team_id: {})
    calls: list[object] = []
    real_loader = mcp_runtime_module.load_team_env_overrides

    def spy_loader(db, team_id):
        calls.append(team_id)
        return real_loader(db, team_id)

    monkeypatch.setattr(mcp_runtime_module, "load_team_env_overrides", spy_loader)

    cfg = _cfg(db_session, seed, connector_team_id=None)
    await cfg._load_mcp_server_configs()

    assert calls == []


@pytest.mark.asyncio
async def test_empty_team_mcp_ids_never_invokes_team_env_hook(db_session, seed):
    _install_visibility(ids_by_team={})  # T1 resolves to no ids at all
    calls: list[int] = []
    mcp_runtime.set_mcp_team_env_hook(
        lambda db, *, team_id: (calls.append(team_id), {})[1]
    )

    cfg = _cfg(db_session, seed, connector_team_id=T1)
    await cfg._load_mcp_server_configs()

    assert calls == []


# ---------------------------------------------------------------------------
# Governing team has NO row => no shared layer at all for that id --
# in particular never the row of a DIFFERENT team (T2) that also holds one
# for the same server, even though the code never even asks T2 for it.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_cross_team_env_borrowing(db_session, seed):
    team_server_id = int(seed.team_server.id)
    _install_visibility(ids_by_team={T1: {team_server_id}})
    # T1 (governing) has no row; T2 (a different team) does -- this must
    # never surface, and production code never even calls the hook with
    # team_id=T2 for this run.
    calls: list[int] = []

    def hook(db, *, team_id):
        calls.append(team_id)
        if team_id == T2:
            return {team_server_id: {"STOLEN": "stolen-value"}}
        return {}

    mcp_runtime.set_mcp_team_env_hook(hook)
    _link_own_env(
        db_session,
        user=seed.c,
        server=seed.team_server,
        env={"OWN_KEY": "own-value"},
        env_source=None,
    )

    env = await _env_for(db_session, seed, seed.team_server, connector_team_id=T1)

    assert "STOLEN" not in env
    assert env.get("OWN_KEY") == "own-value"
    assert calls == [T1]  # only ever asked for the governing team


@pytest.mark.asyncio
async def test_governing_team_no_row_scrubs_legacy_shared_value(db_session, seed):
    """The mechanism behind test_no_cross_team_env_borrowing: even a value
    that already made it into shared_env_by_id through the pre-existing
    user-keyed shared-env hook (the only way a "wrong team's row" could
    reach this map before this PR, since that hook's application-side
    wiring is what used to resolve "shared" for a team-owned connector) is
    scrubbed once a governing team is known and has no row of its own --
    not merely never fetched from elsewhere. This is the direct pin for
    the ``shared_env_by_id.pop(server_id, None)`` line."""
    _install_visibility(ids_by_team={T1: {int(seed.team_server.id)}})
    # Simulates what the legacy, user-keyed-only resolution could already
    # have put in shared_env_by_id for this server, before the team-env
    # block ever runs.
    mcp_runtime.set_mcp_shared_env_hook(
        lambda db, user_id: {seed.team_server.id: {"STOLEN": "legacy-shared-value"}}
    )
    mcp_runtime.set_mcp_team_env_hook(lambda db, *, team_id: {})  # T1 has no row
    _link_own_env(
        db_session,
        user=seed.c,
        server=seed.team_server,
        env={"OWN_KEY": "own-value"},
        env_source=None,
    )

    env = await _env_for(db_session, seed, seed.team_server, connector_team_id=T1)

    assert "STOLEN" not in env
    assert env.get("OWN_KEY") == "own-value"


# ---------------------------------------------------------------------------
# An explicit "shared" pick degrades to the legacy default
# (global < user) when the governing team has no row -- own key merges if
# present, global-only if it does not -- and never a foreign row.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "has_own_key", [True, False], ids=["own-key-present", "own-key-absent"]
)
async def test_shared_pick_degrades_when_owning_team_has_no_row(
    db_session, seed, has_own_key
):
    team_server_id = int(seed.team_server.id)
    _install_visibility(ids_by_team={T1: {team_server_id}})
    mcp_runtime.set_mcp_team_env_hook(
        lambda db, *, team_id: (
            {team_server_id: {"STOLEN": "stolen-value"}} if team_id == T2 else {}
        )
    )
    _link_own_env(
        db_session,
        user=seed.c,
        server=seed.team_server,
        env={"OWN_KEY": "own-value"} if has_own_key else None,
        env_source="shared",
    )

    env = await _env_for(db_session, seed, seed.team_server, connector_team_id=T1)

    assert "STOLEN" not in env
    if has_own_key:
        assert env.get("OWN_KEY") == "own-value"
    else:
        assert "OWN_KEY" not in env
        assert env.get("GLOBAL") == "global-value"


# ---------------------------------------------------------------------------
# The substituted team layer occupies the *shared* slot of
# resolve_stdio_env -- the user layer still wins on an overlapping key, and
# a team-only key still passes through.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_team_env_occupies_shared_slot_user_still_wins(db_session, seed):
    team_server_id = int(seed.team_server.id)
    _install_visibility(ids_by_team={T1: {team_server_id}})
    mcp_runtime.set_mcp_team_env_hook(
        lambda db, *, team_id: (
            {team_server_id: {"SHARED_KEY": "from-team", "ONLY_TEAM": "team-only"}}
            if team_id == T1
            else {}
        )
    )
    _link_own_env(
        db_session,
        user=seed.c,
        server=seed.team_server,
        env={"SHARED_KEY": "own-wins", "ONLY_OWN": "own-only"},
        env_source=None,  # legacy fallback: global < shared < user
    )

    env = await _env_for(db_session, seed, seed.team_server, connector_team_id=T1)

    assert env["SHARED_KEY"] == "own-wins"
    assert env["ONLY_TEAM"] == "team-only"
    assert env["ONLY_OWN"] == "own-only"
    assert env["GLOBAL"] == "global-value"


# ---------------------------------------------------------------------------
# Team hook absent (not installed at all) => legacy behaviour, byte for
# byte -- including that a pre-existing user-keyed shared-env hook, if
# installed, is left completely untouched.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_team_env_hook_absent_falls_back_to_user_keyed_layer(db_session, seed):
    _install_visibility(ids_by_team={T1: {int(seed.team_server.id)}})
    # No mcp_runtime.set_mcp_team_env_hook call at all in this test.
    mcp_runtime.set_mcp_shared_env_hook(
        lambda db, user_id: (
            {seed.team_server.id: {"LEGACY_SHARED": "legacy-value"}}
            if user_id == int(seed.c.id)
            else {}
        )
    )
    _link_own_env(
        db_session,
        user=seed.c,
        server=seed.team_server,
        env={"OWN_KEY": "own-value"},
        env_source=None,
    )

    assert mcp_runtime.team_env_hook_installed() is False
    env = await _env_for(db_session, seed, seed.team_server, connector_team_id=T1)

    assert env.get("LEGACY_SHARED") == "legacy-value"
    assert env.get("OWN_KEY") == "own-value"


@pytest.mark.asyncio
async def test_installed_hook_answering_empty_still_applies_degrade(db_session, seed):
    """The wiring selects on team_env_hook_installed(), never on whether
    load_team_env_overrides(...) happens to return {} for this team: an
    installed hook legitimately answers empty for a team that owns
    nothing (mirrors team_connector_hook_installed's own rule), and the
    degrade below must still fire in that case -- selecting on an empty
    *answer* instead would wrongly skip the whole block and leave a
    "shared" pick undegraded, silently pinning the tool to global-only
    even though the runner has an own key on file."""
    _install_visibility(ids_by_team={T1: {int(seed.team_server.id)}})
    mcp_runtime.set_mcp_team_env_hook(lambda db, *, team_id: {})  # installed, empty
    _link_own_env(
        db_session,
        user=seed.c,
        server=seed.team_server,
        env={"OWN_KEY": "own-value"},
        env_source="shared",
    )

    env = await _env_for(db_session, seed, seed.team_server, connector_team_id=T1)

    assert env.get("OWN_KEY") == "own-value"


# ---------------------------------------------------------------------------
# Ids not in team_mcp_ids are untouched -- even if a hook installed for
# this run would (incorrectly, if consulted) answer for them too.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ids_outside_team_mcp_ids_untouched(db_session, seed):
    team_server_id = int(seed.team_server.id)
    personal_server_id = int(seed.personal_server.id)
    _install_visibility(ids_by_team={T1: {team_server_id}})
    mcp_runtime.set_mcp_team_env_hook(
        lambda db, *, team_id: (
            {
                team_server_id: {"TEAM_KEY": "team-value"},
                personal_server_id: {"SHOULD_NOT_APPEAR": "leak"},
            }
            if team_id == T1
            else {}
        )
    )

    env = await _env_for(db_session, seed, seed.personal_server, connector_team_id=T1)

    assert "SHOULD_NOT_APPEAR" not in env
    assert env == {"GLOBAL": "global-value"}


# ---------------------------------------------------------------------------
# Wiring level: a broken team-env hook raises the typed error from
# _load_mcp_server_configs, and that error survives the create_mcp_tools
# boundary instead of being dropped into a silently env-less/empty tool set.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_team_env_hook_failure_raises_connector_runtime_error(db_session, seed):
    _install_visibility(ids_by_team={T1: {int(seed.team_server.id)}})

    def boom(db, *, team_id):
        raise _ProbeError("boom")

    mcp_runtime.set_mcp_team_env_hook(boom)
    cfg = _cfg(db_session, seed, connector_team_id=T1)

    with pytest.raises(ConnectorRuntimeError) as excinfo:
        await cfg._load_mcp_server_configs()

    assert excinfo.value.status_code == 503
    assert excinfo.value.details["reason"] == "team_env_resolution_failed"
    assert isinstance(excinfo.value.__cause__, _ProbeError)


@pytest.mark.asyncio
async def test_team_env_hook_failure_survives_create_mcp_tools_boundary(
    db_session, seed
):
    _install_visibility(ids_by_team={T1: {int(seed.team_server.id)}})

    def boom(db, *, team_id):
        raise _ProbeError("boom")

    mcp_runtime.set_mcp_team_env_hook(boom)
    cfg = _cfg(db_session, seed, connector_team_id=T1)

    assert cfg.get_tool_selection_spec() is None
    with pytest.raises(ConnectorRuntimeError) as excinfo:
        await create_mcp_tools(cfg)
    assert excinfo.value.details["reason"] == "team_env_resolution_failed"


# ---------------------------------------------------------------------------
# Wiring level: a malformed hook answer raises the same typed error.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_team_env_hook_shape_violation_raises(db_session, seed):
    _install_visibility(ids_by_team={T1: {int(seed.team_server.id)}})
    mcp_runtime.set_mcp_team_env_hook(lambda db, *, team_id: "not-a-dict")
    cfg = _cfg(db_session, seed, connector_team_id=T1)

    with pytest.raises(ConnectorRuntimeError) as excinfo:
        await cfg._load_mcp_server_configs()

    assert excinfo.value.status_code == 503
    assert isinstance(excinfo.value.__cause__, ValueError)


# ---------------------------------------------------------------------------
# Wiring level: neither loader-owned mapping is mutated in place by
# the re-key block -- a hook that keeps a reference to its own returned
# dict observes it unchanged after a build. This is the direct pin for the
# `shared_env_by_id = dict(shared_env_by_id)` copy: load_shared_env_overrides
# hands back the hook's own object unwrapped, so removing that copy would
# mutate it here.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hook_owned_mappings_not_mutated_by_team_rekey(db_session, seed):
    _install_visibility(ids_by_team={T1: {int(seed.team_server.id)}})
    shared_hook_owned = {seed.team_server.id: {"OLD_SHARED": "old-value"}}
    team_hook_owned = {seed.team_server.id: {"TEAM_KEY": "team-value"}}
    mcp_runtime.set_mcp_shared_env_hook(lambda db, user_id: shared_hook_owned)
    mcp_runtime.set_mcp_team_env_hook(
        lambda db, *, team_id: team_hook_owned if team_id == T1 else {}
    )

    env = await _env_for(db_session, seed, seed.team_server, connector_team_id=T1)

    assert env.get("TEAM_KEY") == "team-value"  # the re-key did take effect
    assert shared_hook_owned == {seed.team_server.id: {"OLD_SHARED": "old-value"}}
    assert team_hook_owned == {seed.team_server.id: {"TEAM_KEY": "team-value"}}


# ---------------------------------------------------------------------------
# The four picks/transport cells the suite was missing: a "shared" pick
# with the team row present, "own" and "platform" picks on a team-owned
# server, and a team-owned server on a non-stdio transport.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shared_pick_with_team_row_uses_the_team_row(db_session, seed):
    """The pick="shared" + row-PRESENT combination: the team row is the
    shared layer and the runner's own key is not merged at all. The
    row-ABSENT half of this pick is pinned separately above."""
    team_server_id = int(seed.team_server.id)
    _install_visibility(ids_by_team={T1: {team_server_id}})
    mcp_runtime.set_mcp_team_env_hook(
        lambda db, *, team_id: (
            {team_server_id: {"TEAM_KEY": "team-value"}} if team_id == T1 else {}
        )
    )
    _link_own_env(
        db_session,
        user=seed.c,
        server=seed.team_server,
        env={"OWN_KEY": "own-value"},
        env_source="shared",
    )

    env = await _env_for(db_session, seed, seed.team_server, connector_team_id=T1)

    assert env.get("TEAM_KEY") == "team-value"
    assert "OWN_KEY" not in env
    assert env.get("GLOBAL") == "global-value"


@pytest.mark.asyncio
async def test_own_pick_on_a_team_server_ignores_the_team_row(db_session, seed):
    """Removing the `== "shared"` condition from the degrade in the wiring
    block would flip this pick to the legacy fallback and leak the team row
    into an "own" run; nothing else in this file would notice."""
    team_server_id = int(seed.team_server.id)
    _install_visibility(ids_by_team={T1: {team_server_id}})
    mcp_runtime.set_mcp_team_env_hook(
        lambda db, *, team_id: (
            {team_server_id: {"TEAM_KEY": "team-value"}} if team_id == T1 else {}
        )
    )
    _link_own_env(
        db_session,
        user=seed.c,
        server=seed.team_server,
        env={"OWN_KEY": "own-value"},
        env_source="own",
    )

    env = await _env_for(db_session, seed, seed.team_server, connector_team_id=T1)

    assert env.get("OWN_KEY") == "own-value"
    assert "TEAM_KEY" not in env
    assert env.get("GLOBAL") == "global-value"


@pytest.mark.asyncio
async def test_platform_pick_on_a_team_server_stays_global_only(db_session, seed):
    team_server_id = int(seed.team_server.id)
    _install_visibility(ids_by_team={T1: {team_server_id}})
    mcp_runtime.set_mcp_team_env_hook(
        lambda db, *, team_id: (
            {team_server_id: {"TEAM_KEY": "team-value"}} if team_id == T1 else {}
        )
    )
    _link_own_env(
        db_session,
        user=seed.c,
        server=seed.team_server,
        env={"OWN_KEY": "own-value"},
        env_source="platform",
    )

    env = await _env_for(db_session, seed, seed.team_server, connector_team_id=T1)

    assert env == {"GLOBAL": "global-value"}


@pytest.mark.asyncio
async def test_non_stdio_team_server_never_receives_an_env_layer(db_session):
    """The env layers are stdio-only. The re-key block itself is transport
    agnostic -- it rewrites the map for every team-owned id -- so this pins
    the other end: for an HTTP-transport server the map is never read, and
    the produced config carries no env key at all."""
    c = _create_user(db_session, "run-owner")
    http_server = _create_http_mcp(db_session, "team-http-server")
    seed = SimpleNamespace(c=c, team_server=http_server, personal_server=http_server)
    http_server_id = int(http_server.id)

    _install_visibility(ids_by_team={T1: {http_server_id}})
    mcp_runtime.set_mcp_team_env_hook(
        lambda db, *, team_id: (
            {http_server_id: {"TEAM_KEY": "team-value"}} if team_id == T1 else {}
        )
    )

    config = await _config_for(db_session, seed, http_server, connector_team_id=T1)

    assert config["transport"] == "streamable_http"
    assert "env" not in config["config"]


# ---------------------------------------------------------------------------
# A stored row carrying an empty env mapping must be indistinguishable
# from no row at all.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_team_row_behaves_exactly_like_no_row(db_session, seed):
    """A stored row carrying an empty env mapping must be indistinguishable
    from no row at all. Both are run here and their outcomes compared
    directly, so the two states cannot drift apart later."""
    team_server_id = int(seed.team_server.id)
    _install_visibility(ids_by_team={T1: {team_server_id}})
    _link_own_env(
        db_session,
        user=seed.c,
        server=seed.team_server,
        env={"OWN_KEY": "own-value"},
        env_source="shared",
    )

    mcp_runtime.set_mcp_team_env_hook(lambda db, *, team_id: {team_server_id: {}})
    env_empty_row = await _env_for(
        db_session, seed, seed.team_server, connector_team_id=T1
    )

    mcp_runtime.set_mcp_team_env_hook(lambda db, *, team_id: {})
    env_no_row = await _env_for(
        db_session, seed, seed.team_server, connector_team_id=T1
    )

    assert env_empty_row == env_no_row
    # And the shared outcome is the degrade, not a global-only pin.
    assert env_empty_row == {"GLOBAL": "global-value", "OWN_KEY": "own-value"}


# ---------------------------------------------------------------------------
# The half-installed state: connectors are shared to a team, but no
# credential-side hook was installed, so the shared layer silently stays
# keyed on the running user. Warns once per process.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_warns_once_when_team_connectors_exist_without_the_env_hook(
    db_session, seed, monkeypatch, caplog
):
    """The half-installed state: connectors are shared to a team, but no
    credential-side hook was installed, so the shared layer silently stays
    keyed on the running user. The flag is reset here because it is
    process-wide and another test in this file reaches the same state."""
    monkeypatch.setattr(mcp_runtime, "_team_env_hook_missing_warned", False)
    _install_visibility(ids_by_team={T1: {int(seed.team_server.id)}})

    with caplog.at_level(logging.WARNING, logger=mcp_runtime.__name__):
        await _env_for(db_session, seed, seed.team_server, connector_team_id=T1)
        first = [r for r in caplog.records if "team-keyed env hook" in r.message]
        await _env_for(db_session, seed, seed.team_server, connector_team_id=T1)
        second = [r for r in caplog.records if "team-keyed env hook" in r.message]

    assert len(first) == 1
    assert len(second) == 1  # once per process, not once per run


@pytest.mark.asyncio
async def test_no_warning_when_the_env_hook_is_installed(
    db_session, seed, monkeypatch, caplog
):
    monkeypatch.setattr(mcp_runtime, "_team_env_hook_missing_warned", False)
    _install_visibility(ids_by_team={T1: {int(seed.team_server.id)}})
    mcp_runtime.set_mcp_team_env_hook(lambda db, *, team_id: {})

    with caplog.at_level(logging.WARNING, logger=mcp_runtime.__name__):
        await _env_for(db_session, seed, seed.team_server, connector_team_id=T1)

    assert not [r for r in caplog.records if "team-keyed env hook" in r.message]


@pytest.mark.asyncio
async def test_no_warning_when_the_team_owns_no_visible_connector(
    db_session, seed, monkeypatch, caplog
):
    """The condition a deployment that uses none of the team features can
    never reach: with no visibility hook there are no team-owned ids."""
    monkeypatch.setattr(mcp_runtime, "_team_env_hook_missing_warned", False)

    with caplog.at_level(logging.WARNING, logger=mcp_runtime.__name__):
        await _env_for(db_session, seed, seed.personal_server, connector_team_id=T1)

    assert not [r for r in caplog.records if "team-keyed env hook" in r.message]
