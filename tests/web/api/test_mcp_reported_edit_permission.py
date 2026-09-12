"""The single-server ``GET``/``PUT`` gate surfaces a raising hook's declared
status rather than a 500, and restores the session afterward. A ``GET``
whose verdict resolution fails degrades ``can_edit_global`` to False rather
than failing the read; a ``PUT`` fails closed with a typed 503 only for the
caller the verdict is the gate for -- one with no personal association row;
a caller who already holds one degrades the same way ``GET`` does.

The four MCP OAuth routes, the rename call's scope, and every route's
no-hook-installed shape are all unchanged by threading that verdict through.

Every test installs hooks (or explicitly installs none) through
``snapshot_connector_team_hooks`` so no hook state leaks between tests or
into suites that run after this one.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from xagent.web.api.mcp import (
    MCPOAuthConnectRequest,
    MCPOAuthDiscoverRequest,
    MCPServerUpdate,
    connect_mcp_oauth,
    delete_mcp_oauth_grant,
    discover_mcp_oauth,
    get_mcp_oauth_status,
    get_mcp_server,
    update_mcp_server,
)
from xagent.web.models.agent import Agent
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.user import User
from xagent.web.services.connector_team_scope import (
    ConnectorAccess,
    set_connector_team_hooks,
    snapshot_connector_team_hooks,
)


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def _make_user(db, user_id: int, *, is_admin: bool = False) -> User:
    user = User(
        id=user_id, username=f"user-{user_id}", password_hash="x", is_admin=is_admin
    )
    db.add(user)
    db.commit()
    return user


def _make_owned_server(db, owner_id: int, *, name: str = "shared-server") -> MCPServer:
    server = MCPServer(name=name, transport="stdio", managed="external", command="true")
    db.add(server)
    db.flush()
    db.add(
        UserMCPServer(
            user_id=owner_id,
            mcpserver_id=server.id,
            is_owner=True,
            is_active=True,
        )
    )
    db.commit()
    return server


def _fixed_answer_hook(access_answer):
    """Build a batch access hook that answers every requested ref with the
    same fixed verdict -- or, when ``access_answer`` is ``None``, answers
    with an empty map, which is how "the caller's team does not link this"
    is expressed under the batch contract."""

    def _hook(db, user_id, refs):
        if access_answer is None:
            return {}
        return {ref: access_answer for ref in refs}

    return _hook


class TestOAuthRoutesKeepTheirOwnGate:
    """The four MCP OAuth routes keep the old personal-row-only helper and
    still 404 a team member with no personal row, verdict or not."""

    async def test_all_four_oauth_routes_404_a_team_member_with_no_personal_row(
        self, db
    ):
        owner = _make_user(db, 40)
        member = _make_user(db, 41)
        server = _make_owned_server(db, owner.id, name="oauth-gate-untouched")
        server_id = server.id

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=_fixed_answer_hook(
                    ConnectorAccess(team_owned=True, can_edit=True)
                )
            )

            with pytest.raises(HTTPException) as exc:
                await discover_mcp_oauth(
                    server_id, MCPOAuthDiscoverRequest(), current_user=member, db=db
                )
            assert exc.value.status_code == 404

            with pytest.raises(HTTPException) as exc:
                await connect_mcp_oauth(
                    server_id,
                    MCPOAuthConnectRequest(),
                    current_user=member,
                    db=db,
                    accept=None,
                )
            assert exc.value.status_code == 404

            with pytest.raises(HTTPException) as exc:
                await get_mcp_oauth_status(server_id, current_user=member, db=db)
            assert exc.value.status_code == 404

            with pytest.raises(HTTPException) as exc:
                await delete_mcp_oauth_grant(server_id, 1, current_user=member, db=db)
            assert exc.value.status_code == 404


class TestRenameStaysScopedToItsOwnConnector:
    """Renaming one connector must not reach outside the connector actually
    being renamed."""

    def test_renaming_one_connector_does_not_touch_an_outsiders_own_connector(self, db):
        """A narrower, database-level regression guard, kept alongside the
        selector oracle below because it pins a different failure mode: a
        stray write to the wrong MCPServer row entirely. Passing this
        alone does not prove the rename call is scoped correctly against
        an outsider who links the *same* connector being renamed -- that
        is what the second test in this class checks."""
        owner_a = _make_user(db, 60)
        editor = _make_user(db, 61)
        outsider = _make_user(db, 62)

        server_a = _make_owned_server(db, owner_a.id, name="rename-target")
        server_b = _make_owned_server(db, outsider.id, name="outsiders-own-connector")
        server_a_id, server_b_id = server_a.id, server_b.id

        renamed_calls: list[tuple[int, str, str]] = []

        def spy_renamed_hook(_db, _user_id, _connector_type, connector_id, old, new):
            renamed_calls.append((connector_id, old, new))

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=_fixed_answer_hook(
                    ConnectorAccess(team_owned=True, can_edit=True)
                ),
                renamed=spy_renamed_hook,
            )
            update_mcp_server(
                server_a_id,
                MCPServerUpdate(name="renamed-target"),
                current_user=editor,
                db=db,
            )

        assert renamed_calls == [(server_a_id, "rename-target", "renamed-target")]

        db.rollback()
        outsiders_server = db.query(MCPServer).filter(MCPServer.id == server_b_id).one()
        assert outsiders_server.name == "outsiders-own-connector"

    def test_renaming_a_connector_does_not_rewrite_an_outsiders_own_agent_selectors(
        self, db
    ):
        """The rename call itself installs no selector fan-out of its own:
        rewriting a stored name-based selector is entirely the installed
        renamed-hook's job (not exercised here at all -- no ``renamed``
        hook is installed), never something the core rename call does on
        its own reach. An outsider who also links the exact connector
        being renamed, and whose own agent selects it by name in
        ``tool_categories``, must see that selector completely untouched
        by the call. Constructing that second association and reading
        back ``tool_categories`` is the point: a test that only checks an
        unrelated connector's own row (the test above) would stay green
        even if this call directly rewrote every agent's selectors on its
        own, because it never looks at an agent at all."""
        owner = _make_user(db, 63)
        editor = _make_user(db, 64)
        outsider = _make_user(db, 65)

        server = _make_owned_server(db, owner.id, name="rename-target-selected")
        server_id = server.id

        # The second association: the outsider also personally links this
        # exact connector, on a verdict that passes -- not the separate,
        # unrelated connector the test above uses.
        db.add(
            UserMCPServer(
                user_id=outsider.id,
                mcpserver_id=server_id,
                is_owner=False,
                is_active=True,
            )
        )
        outsiders_agent = Agent(
            user_id=outsider.id,
            name="outsiders-agent",
            tool_categories=["rename-target-selected"],
        )
        db.add(outsiders_agent)
        db.commit()
        agent_id = outsiders_agent.id

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=_fixed_answer_hook(
                    ConnectorAccess(team_owned=True, can_edit=True)
                ),
            )
            update_mcp_server(
                server_id,
                MCPServerUpdate(name="renamed-target-selected"),
                current_user=editor,
                db=db,
            )

        db.rollback()
        refreshed_agent = db.query(Agent).filter(Agent.id == agent_id).one()
        assert refreshed_agent.tool_categories == ["rename-target-selected"]


class TestStandaloneParityWithNoHookInstalled:
    """With no hook installed at all, the routes this work touches behave
    exactly as they did before any of it started, for both populations
    standalone xagent can actually construct: A (the connector's owner)
    and B (a caller with a personal, non-owner link row -- legacy
    per-connector sharing that predates team editing). A third
    population, a complete stranger with neither row nor link, is covered
    separately below, against the single-server routes.
    """

    def test_a_complete_stranger_still_gets_404_with_no_hook(self, db):
        """A caller with neither a personal row nor any team link cannot be
        constructed in the matrix above, which only builds callers that do
        have a personal row. The pre-change 404 for that caller is worth
        pinning on its own."""
        owner = _make_user(db, 702)
        stranger = _make_user(db, 703)
        server = _make_owned_server(db, owner.id, name="parity-stranger-mcp")
        server_id = server.id

        with snapshot_connector_team_hooks():
            set_connector_team_hooks()

            with pytest.raises(HTTPException) as exc:
                get_mcp_server(server_id, current_user=stranger, db=db)
            assert exc.value.status_code == 404

            with pytest.raises(HTTPException) as exc:
                update_mcp_server(
                    server_id,
                    MCPServerUpdate(description="x"),
                    current_user=stranger,
                    db=db,
                )
            assert exc.value.status_code == 404


def poison_by_orm_flush(db, *, colliding_user_id):
    """Poison the session by flushing a row that violates a real unique
    constraint -- this poisons the ORM ``Session`` itself (not only the
    underlying DB transaction) on every backend: SQLAlchemy marks the
    session's transaction inactive after a failed flush, and any later
    operation on it raises ``PendingRollbackError`` until a rollback runs.
    """
    db.add(User(id=colliding_user_id, username="flush-poison-dup", password_hash="x"))
    db.flush()


POISON_SHAPES = [poison_by_orm_flush]
POISON_SHAPE_IDS = ["orm-flush"]


class TestSessionRecoveryAfterHookFailure:
    """A hook that leaves a failed statement on the shared session must not
    turn a route that would otherwise succeed into a 500 -- the seam's
    single hook-invocation door restores the session before the failure
    ever reaches a caller to convert into a typed error (see
    ``_call_connector_hook_gate`` in connector_team_scope.py).
    """

    def test_a_typed_error_raised_by_the_hook_itself_also_restores_the_session(
        self, db
    ):
        """The ``except ConnectorRuntimeError: raise`` arm must restore the
        session too -- a hook can poison the session and *then* raise its
        own typed error, not only a bare exception."""
        owner = _make_user(db, 96)
        member = _make_user(db, 97)
        member_id = member.id
        server = _make_owned_server(db, owner.id, name="typed-error-poison-target")
        server_id = server.id

        def poisoning_typed_hook(_db, _user_id, _refs):
            try:
                poison_by_orm_flush(_db, colliding_user_id=member_id)
            except Exception:
                pass
            raise ConnectorRuntimeError("planted", "planted failure", status_code=409)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=poisoning_typed_hook,
                visibility=lambda _db, _uid: {"mcp": {server_id}, "custom_api": set()},
            )
            with pytest.raises(HTTPException) as exc:
                get_mcp_server(server_id, current_user=member, db=db)
            assert exc.value.status_code == 409

            # The session must be usable again immediately afterward --
            # not just after an explicit external rollback.
            still_works = db.query(MCPServer).filter(MCPServer.id == server_id).first()
        assert still_works is not None


class TestSingleServerAccessResolutionFailure:
    """A single MCP server's verdict plays two different roles depending on
    the caller's population, on both ``GET`` and ``PUT``: for a caller who
    already holds a personal row -- an owner, a platform administrator, or a
    non-owner member writing only their own association fields -- the
    verdict is decoration on something the route can already answer without
    it, so a resolution failure there degrades ``can_edit_global`` to False
    and the request still succeeds. For a caller with no personal row the
    verdict *is* the gate, so a resolution failure there must still fail
    closed with a typed 503 -- never a silent 200 or a 404 that would
    misreport "does not exist" for a connector the caller merely could not
    be asked about."""

    def test_reading_one_server_survives_a_failing_hook_when_a_personal_row_exists(
        self, db
    ):
        owner = _make_user(db, 102)
        member = _make_user(db, 103)
        server = _make_owned_server(db, owner.id, name="read-degrade-target")
        db.add(
            UserMCPServer(
                user_id=member.id,
                mcpserver_id=server.id,
                is_owner=False,
                is_active=True,
            )
        )
        db.commit()
        server_id = server.id

        def raising_access(_db, _user_id, _refs):
            raise ValueError("hook exploded")

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=raising_access)
            response = get_mcp_server(server_id, current_user=member, db=db)

        assert response.can_edit_global is False

    def test_reading_one_server_still_fails_closed_without_a_personal_row(self, db):
        owner = _make_user(db, 104)
        member = _make_user(db, 105)
        server = _make_owned_server(db, owner.id, name="read-gate-target")
        server_id = server.id

        def raising_access(_db, _user_id, _refs):
            raise ValueError("hook exploded")

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=raising_access,
                visibility=lambda _db, _uid: {
                    "mcp": {server_id},
                    "custom_api": set(),
                },
            )
            with pytest.raises(HTTPException) as exc:
                get_mcp_server(server_id, current_user=member, db=db)

        # Must be 503 (typed, fail-closed) -- specifically not 404
        # (which would misreport "does not exist" for a connector the
        # team's own visibility hook just said this caller can see) and
        # not 200 (which would be the door itself failing open).
        assert exc.value.status_code == 503

    def test_updating_one_server_degrades_for_a_caller_who_holds_a_personal_row(
        self, db
    ):
        owner = _make_user(db, 106)
        member = _make_user(db, 107)
        member_id = member.id
        server = _make_owned_server(db, owner.id, name="write-gate-target")
        db.add(
            UserMCPServer(
                user_id=member.id,
                mcpserver_id=server.id,
                is_owner=False,
                is_active=True,
            )
        )
        db.commit()
        server_id = server.id

        def raising_access(_db, _user_id, _refs):
            raise ValueError("hook exploded")

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=raising_access)
            response = update_mcp_server(
                server_id,
                MCPServerUpdate(is_active=False),
                current_user=member,
                db=db,
            )

        # This caller was admitted by their own association row, not by the
        # verdict: they are a non-owner writing only a field that lives on
        # that row, so no verdict governs the write. An outage of the
        # optional team lookup therefore degrades the reported edit right
        # and lets the write land, rather than becoming the answer.
        assert response.can_edit_global is False

        db.commit()
        stored = (
            db.query(UserMCPServer)
            .filter(
                UserMCPServer.user_id == member_id,
                UserMCPServer.mcpserver_id == server_id,
            )
            .one()
        )
        assert stored.is_active is False
