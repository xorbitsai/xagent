"""Unit tests for the ``connector_team_scope`` seam itself, plus the checks
that need a real database: agent-team-keyed visibility (never the runner's
own membership), the legacy-hook-only fallback contract, and the
custom-API twins of the MCP-focused checks -- team-keyed connector
visibility covers both connector kinds, even though most of the
surrounding tests in this file only exercise MCP.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from xagent.web.models import Base, MCPServer, Task, User, UserMCPServer
from xagent.web.models.custom_api import CustomApi, UserCustomApi
from xagent.web.models.task import TaskStatus
from xagent.web.services import agent_team_scope, connector_team_scope
from xagent.web.services.connector_runtime import _load_visible_runtime_connectors
from xagent.web.tools.config import WebToolConfig, _load_custom_api_runtime_view_sync

T1 = 101
T2 = 102


# ---------------------------------------------------------------------------
# Seam unit tests -- no DB required.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_hooks() -> Iterator[None]:
    yield
    connector_team_scope.set_connector_team_hooks()
    agent_team_scope.set_agent_team_scope_hook(None)


def test_team_connector_ids_empty_without_hook_installed():
    assert connector_team_scope.team_connector_hook_installed() is False
    assert connector_team_scope.team_connector_ids(None, team_id=5) == {
        "mcp": set(),
        "custom_api": set(),
    }


def test_team_connector_hook_installed_reflects_presence():
    connector_team_scope.set_connector_team_hooks(
        team_visibility=lambda db, *, team_id: {"mcp": set(), "custom_api": set()}
    )
    try:
        assert connector_team_scope.team_connector_hook_installed() is True
    finally:
        connector_team_scope.set_connector_team_hooks()
    assert connector_team_scope.team_connector_hook_installed() is False


def test_team_connector_ids_resolves_none_team_without_calling_hook():
    calls = []

    def _hook(db, *, team_id):
        calls.append(team_id)
        return {"mcp": {1}, "custom_api": set()}

    connector_team_scope.set_connector_team_hooks(team_visibility=_hook)
    try:
        assert connector_team_scope.team_connector_ids(None, team_id=None) == {
            "mcp": set(),
            "custom_api": set(),
        }
        assert calls == []
    finally:
        connector_team_scope.set_connector_team_hooks()


def test_team_hook_invocation_contract():
    """The hook is called exactly once, by keyword, and never for a None team."""
    calls: list[tuple[str, object]] = []

    def _record(db, *, team_id):
        calls.append(("kw", team_id))
        return {"mcp": set(), "custom_api": set()}

    connector_team_scope.set_connector_team_hooks(team_visibility=_record)
    try:
        assert connector_team_scope.team_connector_ids(None, team_id=None) == {
            "mcp": set(),
            "custom_api": set(),
        }
        assert calls == []
        connector_team_scope.team_connector_ids(None, team_id=T1)
        assert calls == [("kw", T1)]
    finally:
        connector_team_scope.set_connector_team_hooks()


def test_team_hook_positional_only_callable_raises():
    """Positive control: a positional-only hook must not type-check
    silently -- the keyword call is what stands between a swapped install and
    an unrelated team's connectors on every run."""

    def _positional_only(db, team_id, /):
        return {"mcp": set(), "custom_api": set()}

    connector_team_scope.set_connector_team_hooks(team_visibility=_positional_only)
    try:
        with pytest.raises(TypeError):
            connector_team_scope.team_connector_ids(None, team_id=T1)
    finally:
        connector_team_scope.set_connector_team_hooks()


# ---------------------------------------------------------------------------
# DB-backed fixtures for the checks below.
# ---------------------------------------------------------------------------


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


def _create_user(db: Session, username: str) -> User:
    user = User(username=username, password_hash="hash")
    db.add(user)
    db.flush()
    return user


def _create_mcp(db: Session, name: str, *, owner: User | None = None) -> MCPServer:
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
                is_active=True,
            )
        )
        db.flush()
    return server


def _create_custom_api(
    db: Session, name: str, *, owner: User | None = None
) -> CustomApi:
    api = CustomApi(
        name=name,
        description=f"{name} description",
        url="https://example.com/api",
        method="GET",
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
                is_active=True,
            )
        )
        db.flush()
    return api


@pytest.fixture()
def seed(db_session: Session):
    c = _create_user(db_session, "run-owner")
    active_own = _create_mcp(db_session, "active-own", owner=c)
    team_s = _create_mcp(db_session, "team-s")
    team_x = _create_mcp(db_session, "team-x")
    capi_own = _create_custom_api(db_session, "capi-own", owner=c)
    a_capi = _create_custom_api(db_session, "a-capi")
    return SimpleNamespace(
        c=c,
        active_own=active_own,
        team_s=team_s,
        team_x=team_x,
        capi_own=capi_own,
        a_capi=a_capi,
    )


def _team_hook(seed):
    def _hook(db, *, team_id):
        if team_id == T1:
            return {"mcp": {int(seed.team_s.id)}, "custom_api": {int(seed.a_capi.id)}}
        if team_id == T2:
            return {"mcp": {int(seed.team_x.id)}, "custom_api": set()}
        return {"mcp": set(), "custom_api": set()}

    return _hook


# ---------------------------------------------------------------------------
# visible_mcp_server_clause is fail-closed on its own terms, not only
# because of how its one production caller happens to be gated today.
# ---------------------------------------------------------------------------


def test_visible_mcp_server_clause_matches_nothing_for_none_owner_even_with_team_ids(
    db_session, seed
):
    """With ``owner_user_id=None``, the clause reduces to the personal arm
    regardless of ``team_mcp_ids`` -- it never matches a team-owned row even
    when one is named, rather than relying on a caller to have already
    checked identity first."""
    from xagent.web.services.connector_team_scope import visible_mcp_server_clause

    clause = visible_mcp_server_clause(None, {int(seed.team_s.id)})
    matches = db_session.query(MCPServer).filter(clause).all()
    assert matches == []


# ---------------------------------------------------------------------------
# Keyed on the agent's team; the run owner's own membership is irrelevant.
# Parameterised over three run-owner states, each encoded by the
# agent-team-scope hook -- a negative control for a runner-keyed
# implementation: if visibility were keyed on the runner instead of the
# governing agent, this would flip for the T2 and no-team parameters.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_team", [T2, T1, None])
async def test_scope_keys_on_agent_team_not_runner(db_session, seed, owner_team):
    connector_team_scope.set_connector_team_hooks(team_visibility=_team_hook(seed))
    if owner_team is not None:
        agent_team_scope.set_agent_team_scope_hook(
            lambda db, user_id, _team=owner_team: agent_team_scope.AgentTeamScope(
                team_id=_team, is_team_admin=False
            )
        )
    try:
        cfg = WebToolConfig(
            db=db_session,
            request=None,
            user_id=int(seed.c.id),
            connector_team_id=T1,
            include_mcp_tools=True,
        )
        configs = await cfg._load_mcp_server_configs()
        assert {c["name"] for c in configs} == {
            seed.active_own.name,
            seed.team_s.name,
        }
    finally:
        connector_team_scope.set_connector_team_hooks()
        agent_team_scope.set_agent_team_scope_hook(None)


# ---------------------------------------------------------------------------
# A checkout that adopts this revision without installing the new team hook
# is unchanged on both read points, for both connector kinds.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_visibility_hook_alone_is_unchanged(db_session, seed):
    connector_team_scope.set_connector_team_hooks(
        # Also matches ``T1``: a hypothetical implementation that put the
        # fallback inside the shared helper instead of at this one read
        # point would call this hook with the *team* id misread as a user
        # id. Real user ids and team ids are unrelated dense integers, so
        # that collision isn't guaranteed by construction -- matching T1
        # here makes the assertion below deterministic rather than leaving
        # it to an incidental id coincidence.
        visibility=lambda db, user_id: (
            {"mcp": {int(seed.team_s.id)}, "custom_api": {int(seed.a_capi.id)}}
            if user_id in (int(seed.c.id), T1)
            else {"mcp": set(), "custom_api": set()}
        )
    )
    try:
        assert connector_team_scope.team_connector_hook_installed() is False

        # The tool loader consults no hook today and must not widen.
        cfg = WebToolConfig(
            db=db_session,
            request=None,
            user_id=int(seed.c.id),
            connector_team_id=T1,
            include_mcp_tools=True,
        )
        configs = await cfg._load_mcp_server_configs()
        assert {c["name"] for c in configs} == {seed.active_own.name}

        # The runtime-connector loader keeps exactly today's answer via the
        # fallback, for both connector kinds.
        visible = _load_visible_runtime_connectors(
            db_session, user_id=int(seed.c.id), agent_team_id=T1
        )
        mcp_ids = {r.connector_id for r in visible if r.connector_type == "mcp"}
        capi_ids = {r.connector_id for r in visible if r.connector_type == "custom_api"}
        assert mcp_ids == {int(seed.active_own.id), int(seed.team_s.id)}
        assert capi_ids == {int(seed.capi_own.id), int(seed.a_capi.id)}
    finally:
        connector_team_scope.set_connector_team_hooks()


# ---------------------------------------------------------------------------
# A personal agent (no governing team) resolves no team custom API -- the
# custom-API twin of the MCP-side check with the same shape.
# ---------------------------------------------------------------------------


def test_personal_agent_gets_no_team_custom_api(db_session, seed):
    connector_team_scope.set_connector_team_hooks(team_visibility=_team_hook(seed))
    try:
        visible = _load_visible_runtime_connectors(
            db_session, user_id=int(seed.c.id), agent_team_id=None
        )
        capi_ids = {r.connector_id for r in visible if r.connector_type == "custom_api"}
        assert capi_ids == {int(seed.capi_own.id)}
    finally:
        connector_team_scope.set_connector_team_hooks()


# ---------------------------------------------------------------------------
# The fallback selects on hook presence, never on an empty answer: an
# installed hook legitimately answering "this team owns nothing" must not
# be silently overridden by the legacy runner-keyed hook.
# ---------------------------------------------------------------------------


def test_installed_hook_returning_empty_does_not_fall_back(db_session, seed):
    connector_team_scope.set_connector_team_hooks(
        visibility=lambda db, user_id: (
            {"mcp": {int(seed.team_s.id)}, "custom_api": {int(seed.a_capi.id)}}
            if user_id == int(seed.c.id)
            else {"mcp": set(), "custom_api": set()}
        ),
        team_visibility=lambda db, *, team_id: {"mcp": set(), "custom_api": set()},
    )
    try:
        visible = _load_visible_runtime_connectors(
            db_session, user_id=int(seed.c.id), agent_team_id=T1
        )
        mcp_ids = {r.connector_id for r in visible if r.connector_type == "mcp"}
        capi_ids = {r.connector_id for r in visible if r.connector_type == "custom_api"}
        assert mcp_ids == {int(seed.active_own.id)}
        assert capi_ids == {int(seed.capi_own.id)}
    finally:
        connector_team_scope.set_connector_team_hooks()


# ---------------------------------------------------------------------------
# Installing a team-keyed hook fully supersedes the legacy user-keyed
# overlay for every resolution, including a run with no governing agent.
# This is deliberate, not an oversight: falling back to the legacy overlay
# whenever there is no governing agent would re-introduce runner-keyed
# visibility for exactly the population this design excludes -- most
# visibly, a personal agent would inherit its own owner's team connectors,
# which the personal-agent checks above exist to forbid. It is also the
# actual configuration a deployment that installs both the legacy and the
# new hook together runs with on every agent-less resolution, not a
# hypothetical corner case.
# ---------------------------------------------------------------------------


def test_installed_hook_with_no_governing_agent_supersedes_legacy_overlay(
    db_session, seed
):
    connector_team_scope.set_connector_team_hooks(
        visibility=lambda db, user_id: (
            {"mcp": {int(seed.team_s.id)}, "custom_api": {int(seed.a_capi.id)}}
            if user_id == int(seed.c.id)
            else {"mcp": set(), "custom_api": set()}
        ),
        team_visibility=_team_hook(seed),
    )
    try:
        visible = _load_visible_runtime_connectors(
            db_session, user_id=int(seed.c.id), agent_team_id=None
        )
        mcp_ids = {r.connector_id for r in visible if r.connector_type == "mcp"}
        capi_ids = {r.connector_id for r in visible if r.connector_type == "custom_api"}
        # Personal-only on both connector kinds: seed.team_s / seed.a_capi
        # (the legacy hook's answer) do NOT appear, even though the legacy
        # hook alone would have granted them.
        assert mcp_ids == {int(seed.active_own.id)}
        assert capi_ids == {int(seed.capi_own.id)}
    finally:
        connector_team_scope.set_connector_team_hooks()


# ---------------------------------------------------------------------------
# The new-hook branch unions both connector kinds: a team hook's
# "custom_api" grant is consumed at this seam exactly like its "mcp" grant.
# Custom API is now team-keyed on both sides of this seam -- the tool-build
# loaders (WebToolConfig's custom-API paths) are team-keyed too, so a
# team-owned custom API entering a task's runtime selection snapshot always
# has a personal-or-team-satisfied runtime-view resolution and a tool
# loader able to build it. This is the same shape as the legacy branch
# above (test_legacy_visibility_hook_alone_is_unchanged), which already
# grants team custom APIs through the legacy user-keyed hook -- the two
# branches now agree on custom API instead of diverging.
# ---------------------------------------------------------------------------


def test_new_hook_branch_unions_team_custom_api_too(db_session, seed):
    connector_team_scope.set_connector_team_hooks(team_visibility=_team_hook(seed))
    try:
        visible = _load_visible_runtime_connectors(
            db_session, user_id=int(seed.c.id), agent_team_id=T1
        )
        mcp_ids = {r.connector_id for r in visible if r.connector_type == "mcp"}
        capi_ids = {r.connector_id for r in visible if r.connector_type == "custom_api"}
        # T1's hook (see _team_hook above) grants both seed.team_s (mcp) and
        # seed.a_capi (custom_api). Both grants union in now.
        assert mcp_ids == {int(seed.active_own.id), int(seed.team_s.id)}
        assert capi_ids == {int(seed.capi_own.id), int(seed.a_capi.id)}
    finally:
        connector_team_scope.set_connector_team_hooks()


# ---------------------------------------------------------------------------
# The factory-runtime prefetch plan actually carries the agent's team
# id. Without this pin, the custom-API prefetch path silently resolves
# personal-only for every team agent while every other custom-API invariant
# stays green because none of them build the plan.
# ---------------------------------------------------------------------------


def test_factory_runtime_plan_carries_agent_team_id(db_session, seed):
    cfg = WebToolConfig(
        db=db_session,
        request=None,
        user_id=int(seed.c.id),
        connector_team_id=T1,
    )
    plan = cfg._build_factory_runtime_load_plan()
    assert plan.connector_team_id == T1 == cfg._connector_team_id


# ---------------------------------------------------------------------------
# Shape validation on the team-visibility hook's answer. This is an
# authorization input -- a malformed answer must fail loudly (raise), never
# be normalized, coerced, or defaulted to empty. Extra keys are accepted and
# ignored: only "mcp" and "custom_api" are ever read.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "malformed_answer",
    [
        None,
        "not-a-dict",
        {"mcp": set()},  # missing "custom_api"
        {"custom_api": set()},  # missing "mcp"
        {"mcp": [1, 2], "custom_api": set()},  # list, not a set
        {"mcp": {"1", "2"}, "custom_api": set()},  # set of strings, not ints
        {"mcp": {True}, "custom_api": set()},  # bool, not accepted as int
    ],
    ids=[
        "none",
        "non-dict",
        "missing-custom_api",
        "missing-mcp",
        "list-not-set",
        "set-of-strings",
        "set-of-bools",
    ],
)
def test_team_connector_ids_raises_on_malformed_hook_answer(malformed_answer):
    connector_team_scope.set_connector_team_hooks(
        team_visibility=lambda db, *, team_id: malformed_answer
    )
    try:
        with pytest.raises(ValueError):
            connector_team_scope.team_connector_ids(None, team_id=T1)
    finally:
        connector_team_scope.set_connector_team_hooks()


def test_team_connector_ids_accepts_and_ignores_extra_keys():
    connector_team_scope.set_connector_team_hooks(
        team_visibility=lambda db, *, team_id: {
            "mcp": {1, 2},
            "custom_api": {3},
            # An unknown extra key with a value that would itself be
            # malformed if it were ever inspected -- proves the validator
            # only probes "mcp"/"custom_api" and does not iterate every key.
            "unexpected_extra_key": object(),
        }
    )
    try:
        result = connector_team_scope.team_connector_ids(None, team_id=T1)
        assert result["mcp"] == {1, 2}
        assert result["custom_api"] == {3}
    finally:
        connector_team_scope.set_connector_team_hooks()


@pytest.mark.asyncio
async def test_mcp_loader_seam_retypes_malformed_hook_answer(db_session, seed):
    # A hook that (through a type coercion bug on the application side)
    # returns the "mcp" id set as a string instead of a set. Without shape
    # validation this is a SQLite type-affinity fail-open case: the string
    # is silently iterated into single-character values rather than raising.
    connector_team_scope.set_connector_team_hooks(
        team_visibility=lambda db, *, team_id: {"mcp": "12", "custom_api": set()}
    )
    try:
        cfg = WebToolConfig(
            db=db_session,
            request=None,
            user_id=int(seed.c.id),
            connector_team_id=T1,
            include_mcp_tools=True,
        )
        with pytest.raises(ConnectorRuntimeError) as excinfo:
            await cfg._load_mcp_server_configs()
        assert excinfo.value.status_code == 503
        assert excinfo.value.details["reason"] == "team_scope_resolution_failed"
        assert isinstance(excinfo.value.__cause__, ValueError)
    finally:
        connector_team_scope.set_connector_team_hooks()


def test_runtime_view_seam_retypes_malformed_hook_answer(db_session, seed):
    task = Task(
        user_id=seed.c.id,
        title="malformed hook runtime task",
        status=TaskStatus.PENDING,
        source="sdk",
        connector_runtime_selected_refs=[
            {"connector_type": "mcp", "connector_id": int(seed.active_own.id)}
        ],
    )
    db_session.add(task)
    db_session.flush()

    connector_team_scope.set_connector_team_hooks(
        team_visibility=lambda db, *, team_id: {"mcp": "12", "custom_api": set()}
    )
    try:
        with pytest.raises(ConnectorRuntimeError) as excinfo:
            _load_custom_api_runtime_view_sync(
                db_session,
                task_id=str(task.id),
                connector_runtime_turn_id=None,
                user_id=int(seed.c.id),
                agent_team_id=T1,
            )
        assert excinfo.value.status_code == 503
        assert isinstance(excinfo.value.__cause__, ValueError)
    finally:
        connector_team_scope.set_connector_team_hooks()


def test_resolve_or_raise_passes_a_typed_error_through_unchanged():
    """The shared wrap's ``except ConnectorRuntimeError: raise`` arm: a hook
    that already raises the typed error must reach the caller as that exact
    object -- not re-wrapped, not given a new cause -- so an inner seam's
    more specific reason survives to whatever renders the failure."""
    planted = ConnectorRuntimeError(
        "planted_code",
        "planted typed failure",
        details={"reason": "planted_inner_reason"},
        status_code=503,
    )

    def _raising_hook(db, *, team_id):
        raise planted

    connector_team_scope.set_connector_team_hooks(team_visibility=_raising_hook)
    try:
        with pytest.raises(ConnectorRuntimeError) as excinfo:
            connector_team_scope.resolve_team_connector_ids_or_raise(
                None, team_id=T1, log_subject="passthrough-probe"
            )
        assert excinfo.value is planted
        assert excinfo.value.details["reason"] == "planted_inner_reason"
    finally:
        connector_team_scope.set_connector_team_hooks()
