"""Team-scope connector visibility at the custom-API tool-load boundary.

The mirror of ``test_mcp_team_visibility.py``, extended for the two things
that only exist on this side: two read points instead of one (a live path,
``WebToolConfig.get_custom_api_configs``, and an off-loop snapshot prefetch,
``_load_custom_api_factory_inputs``, selected by
``snapshot is not None and snapshot.plan.load_custom_api``), and no
per-member credential layer (``CustomApi.env`` lives on the shared
definition row; ``UserCustomApi`` carries no ``env`` column at all).

Fixture seed (six custom-API rows, run owner ``C`` throughout unless noted):

    active_own            -- C holds an active personal link
    inactive_own           -- C holds an inactive personal link
    stranger                -- owned by a third user, no link to anything
    a                        -- another user's, reachable only through a
                                team hook keyed on T1
    b                        -- another user's, reachable only through a
                                team hook keyed on T2
    inactive_own_shared     -- C holds an inactive personal link AND it is
                                in T1's team-owned set -- the one cell where
                                "personal ∪ team" is not a strict widening
                                of C's own list (a member's personal
                                deactivate does not hide a team-shared API)

The deterministic-ordering contract (``ORDER BY custom_apis.id``) is pinned
by the compiled-SQL assertion in ``test_production_custom_api_query_shape``,
not by any list-order assertion in this file -- no test here asserts the
order of a multi-row-visible result, so the seed's insertion order carries
no test-discrimination weight and is left natural.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

# Reused verbatim from the MCP sibling: same probe exception, same team ids,
# same hook install/teardown factory. ``db_session`` and the autouse hook
# reset are declared locally below rather than imported -- both are cheap
# and self-contained, and importing a ``@pytest.fixture`` for use as a test
# parameter name collides with the parameter of the same name in every
# consuming test.
from tests.web.tools.test_mcp_team_visibility import (
    T1,
    T2,
    _create_user,
    _ProbeError,
    install_team_hooks,
)
from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from xagent.core.tools.adapters.vibe.custom_api_factory import (
    create_db_custom_api_tools,
)
from xagent.web.models import Base
from xagent.web.models.custom_api import CustomApi, UserCustomApi
from xagent.web.services import agent_team_scope, connector_team_scope
from xagent.web.services.connector_runtime import _load_visible_runtime_connectors
from xagent.web.tools.config import (
    WebToolConfig,
    _load_custom_api_factory_inputs,
    _load_tool_factory_runtime_snapshot,
    _visible_custom_api_query,
)


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


def _create_custom_api(
    db: Session,
    name: str,
    *,
    owner=None,
    active: bool = True,
    env: dict | None = None,
) -> CustomApi:
    api = CustomApi(
        name=name,
        description=f"{name} description",
        url="https://api.example.test",
        method="GET",
        headers={},
        body=None,
        env=env if env is not None else {},
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


def _build_seed(db: Session) -> SimpleNamespace:
    c = _create_user(db, "run-owner")
    z = _create_user(db, "stranger-owner")
    w = _create_user(db, "team-a-owner")
    v = _create_user(db, "team-b-owner")

    # Creation order fixes custom_apis.id ascending in this sequence.
    stranger = _create_custom_api(db, "stranger-api", owner=z)
    a = _create_custom_api(db, "team-a-api", owner=w, env={"API_KEY": "team-a-secret"})
    b = _create_custom_api(db, "team-b-api", owner=v)
    active_own = _create_custom_api(db, "active-own-api")
    inactive_own_shared = _create_custom_api(db, "inactive-own-shared-api")
    inactive_own = _create_custom_api(db, "inactive-own-api")

    db.add(
        UserCustomApi(
            user_id=c.id,
            custom_api_id=active_own.id,
            is_owner=True,
            can_edit=True,
            can_delete=True,
            is_active=True,
        )
    )
    db.add(
        UserCustomApi(
            user_id=c.id,
            custom_api_id=inactive_own_shared.id,
            is_owner=True,
            can_edit=True,
            can_delete=True,
            is_active=False,
        )
    )
    db.add(
        UserCustomApi(
            user_id=c.id,
            custom_api_id=inactive_own.id,
            is_owner=True,
            can_edit=True,
            can_delete=True,
            is_active=False,
        )
    )
    db.flush()

    return SimpleNamespace(
        c=c,
        z=z,
        w=w,
        v=v,
        active_own=active_own,
        inactive_own=inactive_own,
        stranger=stranger,
        a=a,
        b=b,
        inactive_own_shared=inactive_own_shared,
    )


@pytest.fixture()
def seed(db_session: Session) -> SimpleNamespace:
    return _build_seed(db_session)


def _team_visibility_hook(seed):
    """ENV-T's team_visibility hook: disjoint per-team custom_api sets."""

    def _hook(db, *, team_id):
        if team_id == T1:
            return {
                "mcp": set(),
                "custom_api": {int(seed.a.id), int(seed.inactive_own_shared.id)},
            }
        if team_id == T2:
            return {"mcp": set(), "custom_api": {int(seed.b.id)}}
        return {"mcp": set(), "custom_api": set()}

    return _hook


def _install_env_t(seed) -> None:
    install_team_hooks(
        team_visibility=_team_visibility_hook(seed), agent_owner_id=int(seed.c.id)
    )


def _install_env_legacy(seed) -> None:
    """ENV-LEGACY's user-keyed ``visibility=`` hook, custom_api-only for C."""

    connector_team_scope.set_connector_team_hooks(
        visibility=lambda db, user_id: (
            {"mcp": set(), "custom_api": {int(seed.a.id)}}
            if user_id == int(seed.c.id)
            else {"mcp": set(), "custom_api": set()}
        )
    )


def _cfg(db_session: Session, seed, *, connector_team_id: int | None) -> WebToolConfig:
    return WebToolConfig(
        db=db_session,
        request=None,
        user_id=int(seed.c.id),
        connector_team_id=connector_team_id,
    )


def _file_engine(tmp_path, name: str):
    engine = create_engine(
        f"sqlite:///{tmp_path / name}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    return engine


# ---------------------------------------------------------------------------
# A team agent loads its group's custom API for a run owner with no
# junction row.
# ---------------------------------------------------------------------------


def test_team_agent_loads_team_custom_api(db_session, seed):
    _install_env_t(seed)
    cfg = _cfg(db_session, seed, connector_team_id=T1)
    configs = cfg.get_custom_api_configs()

    assert {c["name"] for c in configs} == {
        seed.active_own.name,
        seed.a.name,
        seed.inactive_own_shared.name,
    }
    # Correction A: there is no per-member env layer to be missing -- a
    # team-only API's env comes straight off the definition row.
    a_config = next(c for c in configs if c["name"] == seed.a.name)
    assert a_config["env"] == {"API_KEY": "team-a-secret"}


# ---------------------------------------------------------------------------
# A personal agent gets nothing from any team, and writes nothing.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env", ["env_t", "env_legacy"])
def test_personal_agent_gets_no_team_custom_api(db_session, seed, env):
    if env == "env_t":
        _install_env_t(seed)
    else:
        _install_env_legacy(seed)

    cfg = _cfg(db_session, seed, connector_team_id=None)

    before = db_session.query(UserCustomApi).filter_by(user_id=int(seed.c.id)).count()
    assert before == 3  # active_own + inactive_own + inactive_own_shared, seeded

    configs = cfg.get_custom_api_configs()
    names = {c["name"] for c in configs}
    # Positive control: the probe actually ran and returned the caller's own
    # row, so the two negatives below aren't vacuously true of an
    # unconditional [].
    assert seed.active_own.name in names
    assert seed.a.name not in names
    assert seed.b.name not in names

    after = db_session.query(UserCustomApi).filter_by(user_id=int(seed.c.id)).count()
    assert after == before


# ---------------------------------------------------------------------------
# A known, deliberately-pinned divergence in a legacy-hook-only deployment
# that also names a governing team: the runtime-context view is user-keyed
# and unions a legacy-granted team-shared custom API, while the tool loader
# resolves personal-only -- it consults the team-keyed hook exclusively and
# never the legacy one. This asserts the actual divergent behavior, not an
# endorsement of it; connector_runtime.py's comment documents the same gap.
# ---------------------------------------------------------------------------


def test_legacy_hook_with_governing_team_diverges_from_runtime_view(db_session, seed):
    _install_env_legacy(seed)

    visible = _load_visible_runtime_connectors(
        db_session, user_id=int(seed.c.id), agent_team_id=T1
    )
    capi_ids = {r.connector_id for r in visible if r.connector_type == "custom_api"}
    # The legacy hook grants seed.a to C regardless of connector_team_id --
    # it is user-keyed, not team-keyed -- so the runtime view sees it here.
    assert int(seed.a.id) in capi_ids

    cfg = _cfg(db_session, seed, connector_team_id=T1)
    configs = cfg.get_custom_api_configs()
    names = {c["name"] for c in configs}
    # But no team_visibility hook is installed, so the loader's team
    # resolution stays empty regardless of connector_team_id, and it never
    # builds a runtime tool for seed.a -- selectable and persistable via the
    # runtime view above, but toolless.
    assert seed.a.name not in names


# ---------------------------------------------------------------------------
# The snapshot prefetch and the live path return the same ordered list,
# for every governing-team parameter.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("connector_team_id", [T1, None, T2])
def test_snapshot_and_live_custom_api_paths_agree(tmp_path, connector_team_id):
    engine = _file_engine(tmp_path, "j3.db")
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    db = factory()
    try:
        seed = _build_seed(db)
        _install_env_t(seed)
        db.commit()

        live_cfg = _cfg(db, seed, connector_team_id=connector_team_id)
        live_ids = [c["id"] for c in live_cfg.get_custom_api_configs()]
        # Positive control: the comparison below isn't vacuously satisfied
        # by a defect that empties both sides identically.
        assert live_ids

        snapshot_cfg = WebToolConfig(
            db=None,
            request=None,
            db_factory=factory,
            user_id=int(seed.c.id),
            connector_team_id=connector_team_id,
        )
        # Plan-field wiring pin: the plan must actually be built through
        # production, not hand-constructed, or a dropped
        # `connector_team_id=` at the plan-construction call site would
        # silently resolve custom APIs personal-only while this test still
        # passed.
        plan = snapshot_cfg._build_factory_runtime_load_plan()
        assert plan.connector_team_id == snapshot_cfg._connector_team_id
        assert plan.load_custom_api is True

        snapshot = _load_tool_factory_runtime_snapshot(factory, plan)
        assert snapshot.plan.load_custom_api is True
        snapshot_cfg._factory_runtime_snapshot = snapshot
        snapshot_ids = [c["id"] for c in snapshot_cfg.get_custom_api_configs()]

        assert snapshot_ids == live_ids
    finally:
        db.close()
        connector_team_scope.set_connector_team_hooks()
        agent_team_scope.set_agent_team_scope_hook(None)
        engine.dispose()


# ---------------------------------------------------------------------------
# The production custom-API *builder* drives off custom_apis, with a
# deterministic order. Compiled on the production module-level builder.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("team_ids_present", [False, True])
def test_production_custom_api_query_shape(db_session, seed, team_ids_present):
    team_api_ids = frozenset({int(seed.a.id)}) if team_ids_present else frozenset()
    sql = str(
        _visible_custom_api_query(
            db_session, owner_user_id=int(seed.c.id), team_api_ids=team_api_ids
        ).statement.compile(dialect=postgresql.dialect())
    )
    norm = " ".join(sql.split()).upper()
    assert norm.startswith("SELECT")  # positive control
    assert re.search(r"\bFROM CUSTOM_APIS\b", norm)  # driving table
    assert "JOIN USER_CUSTOM_APIS" not in norm  # misuse guard
    assert norm.endswith("ORDER BY CUSTOM_APIS.ID")  # exact ordering column


# ---------------------------------------------------------------------------
# visible_custom_api_clause is fail-closed on its own terms, not only
# because of how its two production callers happen to be gated today (both
# already refuse to call it with a None owner). Mirrors the same guard the
# MCP predicate carries.
# ---------------------------------------------------------------------------


def test_visible_custom_api_clause_matches_nothing_for_none_owner_even_with_team_ids(
    db_session, seed
):
    """With ``owner_user_id=None``, the clause reduces to the personal arm
    regardless of ``team_api_ids`` -- it never matches a team-owned row even
    when one is named, rather than relying on a caller to have already
    checked identity first."""
    from xagent.web.services.connector_team_scope import visible_custom_api_clause

    clause = visible_custom_api_clause(None, {int(seed.a.id)})
    matches = db_session.query(CustomApi).filter(clause).all()
    assert matches == []


# ---------------------------------------------------------------------------
# No identity means no custom-API configs, even when a team grant exists
# and connector_team_id names it -- both read points' identity guards sit
# before the team-scope resolution entirely, so a team grant cannot leak
# through, and cannot turn into a 503 either, for an unauthenticated build.
# Mirrors the MCP suite's equivalent pair.
# ---------------------------------------------------------------------------


def test_unidentified_live_call_resolves_no_custom_api_configs_regardless_of_team(
    db_session,
):
    team_api = _create_custom_api(db_session, "team-only-unidentified-probe")
    connector_team_scope.set_connector_team_hooks(
        team_visibility=lambda db, *, team_id: (
            {"mcp": set(), "custom_api": {int(team_api.id)}}
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
        )
        assert cfg._user_id is None

        configs = cfg.get_custom_api_configs()
        assert configs == []
    finally:
        connector_team_scope.set_connector_team_hooks()


def test_unidentified_prefetch_call_resolves_no_custom_api_configs_regardless_of_team(
    db_session,
):
    team_api = _create_custom_api(db_session, "team-only-prefetch-probe")
    connector_team_scope.set_connector_team_hooks(
        team_visibility=lambda db, *, team_id: (
            {"mcp": set(), "custom_api": {int(team_api.id)}}
            if team_id == T1
            else {"mcp": set(), "custom_api": set()}
        )
    )
    try:
        configs = _load_custom_api_factory_inputs(
            db_session,
            user_id=None,
            task_id=None,
            connector_runtime_turn_id=None,
            connector_team_id=T1,
        )
        assert configs == []
    finally:
        connector_team_scope.set_connector_team_hooks()


# ---------------------------------------------------------------------------
# A team-hook failure surfaces as a typed ConnectorRuntimeError at the
# tool-creator boundary, not as an empty tool set. Both read points.
# ---------------------------------------------------------------------------


def test_team_hook_failure_surfaces_as_connector_runtime_error(db_session, seed):
    def _boom(db, *, team_id):
        raise _ProbeError("boom")

    connector_team_scope.set_connector_team_hooks(team_visibility=_boom)
    cfg = _cfg(db_session, seed, connector_team_id=T1)

    with pytest.raises(ConnectorRuntimeError) as exc_info:
        cfg.get_custom_api_configs()
    assert exc_info.value.details["reason"] == "team_scope_resolution_failed"

    # One frame up, where the user-visible outcome is decided.
    with pytest.raises(ConnectorRuntimeError):
        asyncio.run(create_db_custom_api_tools(cfg))


# ---------------------------------------------------------------------------
# A genuine DB/query failure while loading custom APIs fails the turn,
# matching the MCP twin (_load_mcp_server_configs, which raises
# MCPConfigLoadError on the same class of failure), instead of silently
# degrading to an empty tool set.
# ---------------------------------------------------------------------------


class _FailingCustomApiQuery:
    def filter(self, *a, **k):
        return self

    def order_by(self, *a, **k):
        return self

    def all(self):
        raise RuntimeError("database unavailable")


class _FailingCustomApiQuerySession:
    def query(self, *a, **k):
        return _FailingCustomApiQuery()


def test_query_failure_fails_closed_instead_of_returning_empty():
    cfg = WebToolConfig(
        db=_FailingCustomApiQuerySession(),
        request=None,
        user_id=1,
    )

    with pytest.raises(ConnectorRuntimeError) as exc_info:
        cfg.get_custom_api_configs()
    assert exc_info.value.details["reason"] == "custom_api_config_load_failed"

    # One frame up, where the user-visible outcome is decided.
    with pytest.raises(ConnectorRuntimeError):
        asyncio.run(create_db_custom_api_tools(cfg))


def test_team_hook_failure_surfaces_in_prefetch_snapshot(tmp_path):
    """Prefetch twin: the same failure escapes ``_load_tool_factory_runtime_snapshot``
    because ``propagated_exceptions=(ConnectorRuntimeError,)`` names it."""

    engine = _file_engine(tmp_path, "j5.db")
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    db = factory()
    try:
        seed = _build_seed(db)
        db.commit()

        def _boom(db, *, team_id):
            raise _ProbeError("boom")

        connector_team_scope.set_connector_team_hooks(team_visibility=_boom)

        cfg = WebToolConfig(
            db=None,
            request=None,
            db_factory=factory,
            user_id=int(seed.c.id),
            connector_team_id=T1,
        )
        plan = cfg._build_factory_runtime_load_plan()
        assert plan.load_custom_api is True

        with pytest.raises(ConnectorRuntimeError) as exc_info:
            _load_tool_factory_runtime_snapshot(factory, plan)
        assert exc_info.value.details["reason"] == "team_scope_resolution_failed"
    finally:
        db.close()
        connector_team_scope.set_connector_team_hooks()
        agent_team_scope.set_agent_team_scope_hook(None)
        engine.dispose()


# ---------------------------------------------------------------------------
# Standalone (no hooks installed) is result-identical to today as a set,
# and the new order is custom_apis.id. Control: every revert in the suite
# must leave the set-equality half of this test green.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["live", "snapshot"])
@pytest.mark.parametrize("connector_team_id", [None, 7])
def test_no_hooks_matches_legacy_custom_api_result_set(
    tmp_path, connector_team_id, path
):
    engine = _file_engine(tmp_path, f"j6-{connector_team_id}-{path}.db")
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    db = factory()
    try:
        seed = _build_seed(db)
        db.commit()

        if path == "live":
            cfg = _cfg(db, seed, connector_team_id=connector_team_id)
            configs = cfg.get_custom_api_configs()
        else:
            cfg = WebToolConfig(
                db=None,
                request=None,
                db_factory=factory,
                user_id=int(seed.c.id),
                connector_team_id=connector_team_id,
            )
            plan = cfg._build_factory_runtime_load_plan()
            snapshot = _load_tool_factory_runtime_snapshot(factory, plan)
            cfg._factory_runtime_snapshot = snapshot
            configs = cfg.get_custom_api_configs()

        loaded_names = [c["name"] for c in configs]
        # Inertness, as a set: with no hooks installed, only the one active
        # personal row is visible regardless of the connector_team_id param.
        assert set(loaded_names) == {seed.active_own.name}

        # No ordering assertion here: ENV-0 has exactly one visible row for
        # the run owner, so a list comparison against that single row is a
        # tautology (measured -- SQLite resolves the legacy junction loop
        # through sqlite_autoindex_user_custom_apis_1 on
        # (user_id, custom_api_id), so its order coincides with
        # custom_apis.id order regardless of insertion order; seeding a
        # second visible row would not change that). The ordering contract
        # (`ORDER BY custom_apis.id`) is pinned at the SQL level by
        # test_production_custom_api_query_shape.
    finally:
        db.close()
        engine.dispose()


def test_no_hooks_zero_row_owner_never_reaches_runtime_view(tmp_path, monkeypatch):
    """The zero-row guard (config.py §4.5) runs before
    ``_load_custom_api_runtime_view_sync`` on the snapshot path: an owner
    with no visible custom APIs gets ``[]`` and never resolves the runtime
    view, so a broken runtime-view hook does not turn a "nothing to load"
    run into a 503.
    """

    engine = _file_engine(tmp_path, "j6-zero-row.db")
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    db = factory()
    try:
        owner = _create_user(db, "no-apis-owner")
        db.commit()

        def _explode(**_kwargs):
            raise AssertionError(
                "runtime view must not be resolved when there are zero rows"
            )

        monkeypatch.setattr(
            "xagent.web.services.connector_runtime.load_connector_runtime_view",
            _explode,
        )

        cfg = WebToolConfig(
            db=None,
            request=None,
            db_factory=factory,
            user_id=int(owner.id),
            task_id="web_task_1",
            connector_runtime_turn_id="turn-1",
        )
        plan = cfg._build_factory_runtime_load_plan()
        snapshot = _load_tool_factory_runtime_snapshot(factory, plan)
        assert snapshot.custom_api_configs == []
    finally:
        db.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_query_failure_survives_tool_registry_boundary():
    """Companion to the direct-creator assertion above, mirroring the MCP
    suite's registry-boundary test: without this frame the test proves only
    that the loader raises, not that anybody downstream is left to hear it.
    ``factory.py``'s registry loop re-raises ``ConnectorRuntimeError``
    specifically and would swallow an untyped exception into a WARNING and
    an empty tool list."""
    from xagent.core.tools.adapters.vibe.factory import ToolRegistry

    cfg = WebToolConfig(
        db=_FailingCustomApiQuerySession(),
        request=None,
        user_id=1,
    )

    saved_creators = list(ToolRegistry._tool_creators)
    saved_imported = ToolRegistry._modules_imported
    ToolRegistry._tool_creators = []
    ToolRegistry._modules_imported = True
    try:
        ToolRegistry.register(create_db_custom_api_tools)
        with pytest.raises(ConnectorRuntimeError) as excinfo:
            await ToolRegistry.create_registered_tools(cfg)
    finally:
        ToolRegistry._tool_creators = saved_creators
        ToolRegistry._modules_imported = saved_imported

    assert excinfo.value.details["reason"] == "custom_api_config_load_failed"
