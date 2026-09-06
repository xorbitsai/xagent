"""The reported ``can_edit_global``/``can_configure`` fields agree with what
this module's own gates actually enforce, across every response-builder call
site -- and the four MCP OAuth routes, the rename call's scope, and every
route's no-hook-installed shape are all unchanged by threading that verdict
through.

The aggregate listings project Custom API rows as well as MCP servers, so the
reported field is pinned for both kinds here. What is not pinned here is
whether that reported value agrees with the Custom API routes' own write
outcome: those routes gate on their own association row and this module makes
no claim about them.

Every test installs hooks (or explicitly installs none) through
``snapshot_connector_team_hooks`` so no hook state leaks between tests or
into suites that run after this one.
"""

from __future__ import annotations

import logging

import pytest
import sqlalchemy as sa
from fastapi import HTTPException
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from xagent.web.api.mcp import (
    MCPAppConnectRequest,
    MCPOAuthConnectRequest,
    MCPOAuthDiscoverRequest,
    MCPServerUpdate,
    connect_mcp_app,
    connect_mcp_oauth,
    delete_mcp_oauth_grant,
    delete_mcp_server,
    discover_mcp_oauth,
    get_mcp_oauth_status,
    get_mcp_server,
    get_mcp_servers,
    list_mcp_apps,
    toggle_mcp_server,
    update_mcp_server,
)
from xagent.web.models.agent import Agent
from xagent.web.models.custom_api import CustomApi, UserCustomApi
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
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


def _make_owned_api(db, owner_id: int, *, name: str = "shared-api") -> CustomApi:
    api = CustomApi(name=name, url="https://example.com/api", method="GET")
    db.add(api)
    db.flush()
    db.add(
        UserCustomApi(
            user_id=owner_id,
            custom_api_id=api.id,
            is_owner=True,
            can_edit=True,
            can_delete=True,
            is_active=True,
        )
    )
    db.commit()
    return api


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


class TestListEndpointAccessHookCallBudget:
    """The list endpoint asks the access hook at most once per request, no
    matter how many rows need a verdict -- pinned across two different
    population sizes with a counting test double. Counting hook calls alone
    would hide any SQL the endpoint's own queries issue on top of it, or
    that the hook's own body issues, so a SQLAlchemy
    ``before_cursor_execute`` listener additionally pins the *total* number
    of SQL statements for two different row counts: if either grew with row
    count, that would mean the endpoint reverted to a per-row hook call
    after all."""

    @pytest.mark.parametrize("num_rows", [2, 6], ids=["R=2", "R=6"])
    def test_the_list_asks_the_access_hook_exactly_once_no_matter_how_many_rows(
        self, db, num_rows
    ):
        caller = _make_user(db, 100 + num_rows)
        other_owner = _make_user(db, 200 + num_rows)

        # P = 2 personal rows the caller owns outright -- never worth a
        # hook call.
        owned = [
            _make_owned_server(db, caller.id, name=f"owned-{num_rows}-{i}")
            for i in range(2)
        ]

        # Q = num_rows personal rows the caller holds but does not own (a
        # second link on a connector someone else owns).
        shared_personal = []
        for i in range(num_rows):
            server = _make_owned_server(
                db, other_owner.id, name=f"shared-personal-{num_rows}-{i}"
            )
            db.add(
                UserMCPServer(
                    user_id=caller.id,
                    mcpserver_id=server.id,
                    is_owner=False,
                    is_active=True,
                )
            )
            db.commit()
            shared_personal.append(server)

        # R = num_rows rows the caller has no personal row for at all, made
        # visible through the separate visibility hook (not the access hook
        # under test here).
        stand_in = [
            _make_owned_server(db, other_owner.id, name=f"stand-in-{num_rows}-{i}")
            for i in range(num_rows)
        ]

        # Read every id the hooks below will need before the query listener
        # attaches: the objects above were expired by their own setup
        # commits (session default expire_on_commit=True), so reading .id
        # for the first time inside the measured window would count as a
        # query the *endpoint* issues, when it is really just this test's
        # own setup catching up. caller.id specifically: get_mcp_servers
        # reads current_user.id as its very first act.
        _ = caller.id
        owned_ids = {s.id for s in owned}
        shared_personal_ids = {s.id for s in shared_personal}
        stand_in_ids = {s.id for s in stand_in}

        calls: list[object] = []

        def counting_access_hook(hook_db, user_id, refs):
            calls.append(refs)
            # A realistic hook resolves its own team-membership rows to
            # answer the batch -- simulated here as three throwaway
            # statements run once per call, regardless of how many refs
            # were asked about. If the endpoint ever regressed to one hook
            # call per row, the total statement count below would grow
            # with num_rows; it must not.
            for _ in range(3):
                hook_db.execute(sa.select(sa.literal(1)))
            return {
                ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
            }

        def visibility_hook(_db, _user_id):
            return {"mcp": set(stand_in_ids), "custom_api": set()}

        queries: list[str] = []

        def record_query(conn, cursor, statement, parameters, context, executemany):
            del conn, cursor, parameters, context, executemany
            queries.append(statement)

        engine = db.get_bind()
        event.listen(engine, "before_cursor_execute", record_query)
        try:
            with snapshot_connector_team_hooks():
                set_connector_team_hooks(
                    access=counting_access_hook, visibility=visibility_hook
                )
                get_mcp_servers(current_user=caller, db=db)
        finally:
            event.remove(engine, "before_cursor_execute", record_query)

        assert len(calls) == 1
        requested_refs = calls[0]
        assert set(requested_refs) == {
            ("mcp", sid) for sid in shared_personal_ids | stand_in_ids
        }
        assert {rid for (_kind, rid) in requested_refs}.isdisjoint(owned_ids)

        # The hook-call count above cannot see the SQL the endpoint's own
        # queries issue on top of it, or the hook's own three statements.
        # Observed by running this exact population and reading the
        # recorded statements, not derived from a formula -- but pinned as
        # a constant on purpose: it must come out identical for num_rows=2
        # and num_rows=6, since every row within P, Q or R is served by one
        # batched IN-clause query (or the single hook call), never a query
        # or a hook call per row. Includes one additional catalog-keys
        # SELECT that fires once per request, not once per row: every
        # granting verdict this hook returns has to be checked against the
        # platform catalog before it can be trusted as an edit grant, and
        # that catalog is read once and shared across every row's check.
        assert len(queries) == 8, queries


class TestAppsListEndpointAccessHookCallBudget:
    """The sister endpoint's budget: ``/api/mcp/apps`` (``location=local``)
    also asks the access hook at most once per request, covering both
    connector kinds in the same call, independent of row count."""

    @pytest.mark.parametrize("num_rows", [2, 6], ids=["R=2", "R=6"])
    def test_the_apps_listing_asks_the_access_hook_exactly_once_no_matter_how_many_rows(
        self, db, num_rows
    ):
        owner = _make_user(db, 300 + num_rows)
        member = _make_user(db, 400 + num_rows)

        # Personal rows the member owns outright -- a personal row already
        # answers can_configure on its own, so these are never worth a
        # hook call.
        owned_mcp = [
            _make_owned_server(db, member.id, name=f"apps-owned-mcp-{num_rows}-{i}")
            for i in range(2)
        ]
        owned_api = [
            _make_owned_api(db, member.id, name=f"apps-owned-api-{num_rows}-{i}")
            for i in range(2)
        ]

        # Stand-in rows across both kinds -- every one of these needs a
        # verdict.
        stand_in_mcp = [
            _make_owned_server(db, owner.id, name=f"apps-stand-in-mcp-{num_rows}-{i}")
            for i in range(num_rows)
        ]
        stand_in_api = [
            _make_owned_api(db, owner.id, name=f"apps-stand-in-api-{num_rows}-{i}")
            for i in range(num_rows)
        ]

        _ = member.id
        owned_mcp_ids = {s.id for s in owned_mcp}
        owned_api_ids = {a.id for a in owned_api}
        stand_in_mcp_ids = {s.id for s in stand_in_mcp}
        stand_in_api_ids = {a.id for a in stand_in_api}

        calls: list[object] = []

        def counting_access_hook(hook_db, user_id, refs):
            calls.append(refs)
            for _ in range(3):
                hook_db.execute(sa.select(sa.literal(1)))
            return {
                ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
            }

        def visibility_hook(_db, _user_id):
            return {"mcp": set(stand_in_mcp_ids), "custom_api": set(stand_in_api_ids)}

        queries: list[str] = []

        def record_query(conn, cursor, statement, parameters, context, executemany):
            del conn, cursor, parameters, context, executemany
            queries.append(statement)

        engine = db.get_bind()
        event.listen(engine, "before_cursor_execute", record_query)
        try:
            with snapshot_connector_team_hooks():
                set_connector_team_hooks(
                    access=counting_access_hook, visibility=visibility_hook
                )
                list_mcp_apps(location="local", current_user=member, db=db)
        finally:
            event.remove(engine, "before_cursor_execute", record_query)

        assert len(calls) == 1
        requested_refs = calls[0]
        assert set(requested_refs) == {("mcp", sid) for sid in stand_in_mcp_ids} | {
            ("custom_api", aid) for aid in stand_in_api_ids
        }
        called_mcp_ids = {rid for (kind, rid) in requested_refs if kind == "mcp"}
        called_api_ids = {rid for (kind, rid) in requested_refs if kind == "custom_api"}
        assert called_mcp_ids.isdisjoint(owned_mcp_ids)
        assert called_api_ids.isdisjoint(owned_api_ids)

        # Pinned as a constant for the same reason as the sibling test
        # above: it must be identical for num_rows=2 and num_rows=6.
        assert len(queries) == 10, queries


class TestDegradedListingQueryCostGrowsWithRowCount:
    """The healthy half of this class's own name is already covered above
    (the call budget classes pin a constant statement count for a healthy
    hook). This class covers the other half: when the access hook fails,
    ``_restore_session_after_hook_failure`` (connector_team_scope.py) calls
    ``db.rollback()`` to recover the session the failed hook may have left
    mid-statement. The same rollback, and so the same cost, applies when
    the hook returns normally but its answer is rejected by the seam's
    validator: the door restores the session for both. On SQLAlchemy
    2.0.48, that rollback expires every already-loaded object's every
    mapped field, including primary keys --
    so the two listing loops below, each iterating a stand-in row per
    connector, re-``SELECT`` that row one at a time on next access. Repo
    issue #1711 independently confirmed this rollback behavior. This test
    exists to pin that cost as a number CI will notice moving, not to
    remove it: the recovery itself is required (a failed hook can leave a
    statement failed on the shared session, and the next request on that
    session needs it usable again), and there is no cheaper way to get
    there available to this seam.

    Counts only ``SELECT`` statements (``q.lstrip().upper().startswith
    ("SELECT")``) -- a different count than the two call-budget classes
    above, which count every statement including the hook's own. The two
    numbers are not meant to line up; this class exists to see the
    per-row re-select specifically, and INSERT/UPDATE noise from a
    healthy hook's own bookkeeping would only blur that.

    Population: ``num_rows`` stand-in MCP servers and ``num_rows``
    stand-in Custom APIs (owner-owned, visible to the caller only through
    the visibility hook), with the caller holding zero personal
    association rows of its own -- every row in both listings therefore
    needs a verdict, so the degradation this class measures actually
    fires for the whole listing, not just part of it.
    """

    # Measured directly against this PR's own code (2026-08-26, SQLite,
    # SQLAlchemy 2.0.48): constant while healthy, BASE + 2*num_rows while
    # failing. The "+2*num_rows" is one re-SELECT for the MCPServer/
    # CustomApi row and one for the UserMCPServer/UserCustomApi row per
    # stand-in connector (both listings build one stand-in per row across
    # both kinds; num_rows stand-ins per kind here, so 2*num_rows total
    # re-selects). The extra "+1" on ``servers`` alone reflects that
    # endpoint's own extra per-owner-lookup query the apps endpoint does
    # not have; it does not grow with num_rows.
    #
    # ``HEALTHY["servers"]`` carries one further "+1" that ``BASE["servers"]``
    # does not: a healthy hook here always grants edit, so the servers
    # listing's platform-catalog check reads the catalog once per request.
    # The failing hook's verdicts map is empty, so that check never fires --
    # BASE stays the pre-catalog-check number on purpose.
    HEALTHY = {"apps": 7, "servers": 6}
    BASE = {"apps": 7, "servers": 5}
    EXTRA = {"apps": 0, "servers": 1}

    def _run(self, db, *, endpoint, num_rows, failing):
        owner = _make_user(db, 900 + num_rows * 10 + (1 if failing else 0))
        member = _make_user(db, 950 + num_rows * 10 + (1 if failing else 0))

        stand_in_mcp = [
            _make_owned_server(
                db, owner.id, name=f"cost-mcp-{endpoint}-{num_rows}-{failing}-{i}"
            )
            for i in range(num_rows)
        ]
        stand_in_api = [
            _make_owned_api(
                db, owner.id, name=f"cost-api-{endpoint}-{num_rows}-{failing}-{i}"
            )
            for i in range(num_rows)
        ]
        # Warms member's attributes before the listener below is attached:
        # every _make_user/_make_owned_* call above commits, which expires
        # every already-loaded object under this session's default
        # expire_on_commit. Without this access, the route's own first
        # touch of current_user.id would trigger member's refresh SELECT
        # after the listener is attached, inflating the count by one for a
        # reason that has nothing to do with the degradation this class
        # measures.
        _ = member.id
        mcp_ids = {s.id for s in stand_in_mcp}
        api_ids = {a.id for a in stand_in_api}

        def failing_hook(hook_db, user_id, refs):
            raise ValueError("hook exploded")

        def ok_hook(hook_db, user_id, refs):
            return {
                ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
            }

        def visibility_hook(_db, _user_id):
            return {"mcp": set(mcp_ids), "custom_api": set(api_ids)}

        queries: list[str] = []

        def record_query(conn, cursor, statement, parameters, context, executemany):
            del conn, cursor, parameters, context, executemany
            queries.append(statement)

        engine = db.get_bind()
        event.listen(engine, "before_cursor_execute", record_query)
        try:
            with snapshot_connector_team_hooks():
                set_connector_team_hooks(
                    access=failing_hook if failing else ok_hook,
                    visibility=visibility_hook,
                )
                if endpoint == "apps":
                    rows = list_mcp_apps(location="local", current_user=member, db=db)
                else:
                    rows = get_mcp_servers(current_user=member, db=db)
        finally:
            event.remove(engine, "before_cursor_execute", record_query)

        assert len(rows) == 2 * num_rows
        n_select = sum(1 for q in queries if q.lstrip().upper().startswith("SELECT"))
        return n_select

    @pytest.mark.parametrize("failing", [False, True], ids=["healthy", "failing"])
    @pytest.mark.parametrize("endpoint", ["apps", "servers"])
    @pytest.mark.parametrize("num_rows", [2, 6], ids=["R=2", "R=6"])
    def test_select_count(self, db, endpoint, num_rows, failing):
        n_select = self._run(db, endpoint=endpoint, num_rows=num_rows, failing=failing)
        if failing:
            expected = self.BASE[endpoint] + 2 * num_rows + self.EXTRA[endpoint]
            assert n_select == expected, (
                f"expected {expected} SELECTs for a failing hook with "
                f"num_rows={num_rows} on {endpoint} (base "
                f"{self.BASE[endpoint]} + 2*{num_rows} row re-selects + "
                f"{self.EXTRA[endpoint]} endpoint-specific extra), got "
                f"{n_select}"
            )
        else:
            assert n_select == self.HEALTHY[endpoint], (
                f"expected a constant {self.HEALTHY[endpoint]} SELECTs for "
                f"a healthy hook on {endpoint} regardless of num_rows, got "
                f"{n_select}"
            )


class TestReportedEditPermissionConsistencyMcp:
    """The response's can_edit_global must agree across every surface that
    reports it, for the same (user, connector) -- for MCP connectors, across
    the list, GET, PUT's response and toggle's response.

    One population is the exception: a stand-in whose verdict denies edit
    no longer gets a PUT response to compare at all -- that payload's
    writable field set is empty, so the route refuses it outright (see
    TestADenyingStandInIsRefusedRatherThanReportedSuccessful in
    test_mcp_team_connector_edit.py) rather than reporting a decorative
    can_edit_global on a write that could never have landed."""

    @pytest.mark.parametrize(
        "population,access_answer,has_personal_row",
        [
            ("owner", None, True),
            ("personal_non_owner_no_team_link", None, True),
            (
                # The PR's own central capability: a caller who already has
                # a personal row that does not grant edit, widened by a
                # granting team verdict. Every other population here either
                # has no personal row (the stand-ins) or no verdict.
                "personal_row_and_granting_verdict",
                ConnectorAccess(team_owned=True, can_edit=True),
                True,
            ),
            (
                "stand_in_granting_edit",
                ConnectorAccess(team_owned=True, can_edit=True),
                False,
            ),
            (
                "stand_in_denying_edit",
                ConnectorAccess(team_owned=True, can_edit=False),
                False,
            ),
            (
                # The admin bypass in _check_mcp_permission wins even over a
                # verdict that itself denies edit -- this is the one
                # population where the two connector kinds genuinely
                # diverge (Custom API's own gate has no admin bypass at
                # all), so it is pinned per kind, not by cross-kind equality.
                "platform_admin",
                ConnectorAccess(team_owned=True, can_edit=False),
                False,
            ),
        ],
    )
    def test_can_edit_global_agrees_across_list_get_put_and_toggle(
        self, db, population, access_answer, has_personal_row
    ):
        owner = _make_user(db, 10)
        if population == "owner":
            caller = owner
        elif population == "platform_admin":
            caller = _make_user(db, 12, is_admin=True)
        else:
            caller = _make_user(db, 11)
        server = _make_owned_server(db, owner.id, name=f"consistency-mcp-{population}")
        server_id = server.id

        if population in (
            "personal_non_owner_no_team_link",
            "personal_row_and_granting_verdict",
        ):
            db.add(
                UserMCPServer(
                    user_id=caller.id,
                    mcpserver_id=server_id,
                    is_owner=False,
                    is_active=True,
                )
            )
            db.commit()

        expected = population in ("owner", "platform_admin") or bool(
            access_answer is not None and access_answer.can_edit
        )

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=_fixed_answer_hook(access_answer),
                visibility=lambda _db, _uid: {"mcp": {server_id}, "custom_api": set()},
            )

            list_entries = get_mcp_servers(current_user=caller, db=db)
            list_entry = next(r for r in list_entries if r.id == server_id)

            get_response = get_mcp_server(server_id, current_user=caller, db=db)

            # A denying stand-in's PUT no longer reaches a can_edit_global
            # value to agree with: it is refused outright before this route
            # builds a response at all (empty writable field set -- see
            # TestADenyingStandInIsRefusedRatherThanReportedSuccessful in
            # test_mcp_team_connector_edit.py). The other three surfaces
            # below are unaffected by that guard and still agree on
            # ``expected``.
            put_response = None
            if population == "stand_in_denying_edit":
                with pytest.raises(HTTPException) as exc:
                    update_mcp_server(
                        server_id, MCPServerUpdate(), current_user=caller, db=db
                    )
                assert exc.value.status_code == 403
            else:
                put_response = update_mcp_server(
                    server_id, MCPServerUpdate(), current_user=caller, db=db
                )

            toggle_response = None
            if has_personal_row:
                toggle_response = toggle_mcp_server(
                    server_id, current_user=caller, db=db
                )

        assert list_entry.can_edit_global == expected
        assert get_response.can_edit_global == expected
        if put_response is not None:
            assert put_response.can_edit_global == expected
        if toggle_response is not None:
            assert toggle_response.can_edit_global == expected


class TestLocalCanConfigureWidening:
    """``_local_mcp_can_configure`` answers True for a stand-in whose team
    access verdict links the connector but denies edit -- visible and
    reachable rather than invisible on ``association is None`` alone, for
    both connector kinds."""

    def test_mcp_stand_in_with_a_linked_but_not_editable_verdict_is_configurable(
        self, db
    ):
        owner = _make_user(db, 30)
        member = _make_user(db, 31)
        server = _make_owned_server(db, owner.id, name="visible-not-editable-mcp")
        server_id = server.id

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=_fixed_answer_hook(
                    ConnectorAccess(team_owned=True, can_edit=False)
                ),
                visibility=lambda _db, _uid: {"mcp": {server_id}, "custom_api": set()},
            )
            entries = list_mcp_apps(location="local", current_user=member, db=db)
            entry = next(e for e in entries if e["server_id"] == server_id)
            assert entry["can_configure"] is True

            # The actual route (fixed independently of this UI hint) already
            # resolves for this population -- this proves the hint agrees.
            response = get_mcp_server(server_id, current_user=member, db=db)
        assert response.id == server_id

    def test_custom_api_stand_in_with_a_linked_but_not_editable_verdict_is_configurable(
        self, db
    ):
        owner = _make_user(db, 32)
        member = _make_user(db, 33)
        api = _make_owned_api(db, owner.id, name="visible-not-editable-api")
        api_id = api.id

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=_fixed_answer_hook(
                    ConnectorAccess(team_owned=True, can_edit=False)
                ),
                visibility=lambda _db, _uid: {"mcp": set(), "custom_api": {api_id}},
            )
            entries = list_mcp_apps(location="local", current_user=member, db=db)
            entry = next(
                e
                for e in entries
                if e["server_id"] == api_id and e["transport"] == "custom_api"
            )
            assert entry["can_configure"] is True


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


class TestDenyingVerdictIsFalseEverywhere:
    """A connector whose verdict denies edit reports can_edit_global False
    in the list, in the response from GET, and in the response from PUT
    alike.

    ``member`` holds a personal, non-owner association row here (population
    D: personal row + team link + denying verdict), not a stand-in: a
    stand-in whose verdict denies edit is now refused outright by PUT (see
    TestADenyingStandInIsRefusedRatherThanReportedSuccessful in
    test_mcp_team_connector_edit.py), so it can no longer reach a
    successful PUT response to assert can_edit_global on. Population D
    still can -- can_edit_global is False by the same route (no personal
    can_edit, no granting verdict) on all three surfaces, and its PUT
    succeeds because it is writing its own association row, not the
    verdict-gated shared config.
    """

    def test_a_denying_verdict_yields_false_in_the_list_get_and_put_response(self, db):
        owner = _make_user(db, 50)
        member = _make_user(db, 51)
        server = _make_owned_server(db, owner.id, name="denied-everywhere")
        server_id = server.id
        db.add(
            UserMCPServer(
                user_id=member.id,
                mcpserver_id=server_id,
                is_owner=False,
                is_active=True,
            )
        )
        db.commit()

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=_fixed_answer_hook(
                    ConnectorAccess(team_owned=True, can_edit=False)
                ),
                visibility=lambda _db, _uid: {"mcp": {server_id}, "custom_api": set()},
            )
            list_entries = get_mcp_servers(current_user=member, db=db)
            list_entry = next(r for r in list_entries if r.id == server_id)
            get_response = get_mcp_server(server_id, current_user=member, db=db)
            put_response = update_mcp_server(
                server_id, MCPServerUpdate(), current_user=member, db=db
            )

        assert list_entry.can_edit_global is False
        assert get_response.can_edit_global is False
        assert put_response.can_edit_global is False


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
    """With no hook installed at all, every route touched by this work
    behaves exactly as it did before any of it started -- across every
    route surface this change touches, for both populations standalone
    xagent can actually construct: A (the
    connector's owner) and B (a caller with a personal, non-owner link
    row -- legacy per-connector sharing that predates team editing). A
    third population, a complete stranger with neither row nor link, is
    covered separately below.

    Two additional route legs this same work touched but that fall
    outside the matrix -- ``/api/mcp/apps``'s
    ``can_configure`` and ``connect_mcp_app``'s ``can_edit_global`` -- are
    pinned for both populations at the end of this class, so this
    module's own docstring claim ("every route touched by this work") is
    backed by actual coverage rather than just asserted.

    The rows below are numbered for reference within this test:
    1/2 GET /servers list (presence, can_edit_global), 3 GET /servers/{id},
    4 PUT changing a global field, 5 PUT resubmitting a global field's
    current value unchanged, 6 PUT touching only a personal field, 7 PUT
    touching both at once, 8 DELETE /servers/{id}, 9 POST .../toggle.

    Row 1/2 covers both connector kinds, because the aggregate listing
    projects Custom API rows through ``_custom_api_to_mcp_response`` and
    this work changed that projection. The remaining rows are the MCP
    routes only: this module covers what these routes report and refuse,
    and the Custom API routes' own write contracts are not among them.
    """

    @pytest.mark.parametrize(
        "population", ["owner", "personal_non_owner"], ids=["A=owner", "B=personal"]
    )
    async def test_the_matrix_rows_match_pre_change_behavior_with_no_hook(
        self, db, population
    ):
        owner = _make_user(db, 700)
        member = _make_user(db, 701)
        caller = owner if population == "owner" else member

        server = _make_owned_server(db, owner.id, name=f"parity-mcp-{population}")
        server_id = server.id
        api = _make_owned_api(db, owner.id, name=f"parity-api-{population}")
        api_id = api.id

        if population == "personal_non_owner":
            db.add(
                UserMCPServer(
                    user_id=member.id,
                    mcpserver_id=server_id,
                    is_owner=False,
                    is_active=True,
                )
            )
            db.add(
                UserCustomApi(
                    user_id=member.id,
                    custom_api_id=api_id,
                    is_owner=False,
                    can_edit=False,
                    is_active=True,
                )
            )
            db.commit()

        # Whether an *actual global-config change* (rows 4 and 7) is
        # expected to succeed for this population -- owner always can,
        # a personal-but-non-owner caller never can with no hook and thus
        # no team verdict.
        can_edit_global_config = population == "owner"

        with snapshot_connector_team_hooks():
            set_connector_team_hooks()  # explicit reset: no hooks installed

            # Rows 1-2: GET /servers list -- presence and can_edit_global,
            # for BOTH connector kinds. The aggregate listing projects
            # Custom API rows through _custom_api_to_mcp_response, which
            # this work also changed; asserting only the MCP row would
            # leave that projection unpinned. Both rows are selected by
            # (id, transport): the two kinds live in separate tables and
            # their ids collide freely.
            list_entries = get_mcp_servers(current_user=caller, db=db)
            mcp_entry = next(
                r
                for r in list_entries
                if r.id == server_id and r.transport != "custom_api"
            )
            assert mcp_entry.can_edit_global is can_edit_global_config
            api_list_entry = next(
                r
                for r in list_entries
                if r.id == api_id and r.transport == "custom_api"
            )
            assert api_list_entry.can_edit_global is can_edit_global_config

            # Row 3: GET /servers/{id}.
            get_response = get_mcp_server(server_id, current_user=caller, db=db)
            assert get_response.can_edit_global is can_edit_global_config

            # Row 4: PUT changing a global field (description).
            if can_edit_global_config:
                put_response = update_mcp_server(
                    server_id,
                    MCPServerUpdate(description="row4-changed"),
                    current_user=caller,
                    db=db,
                )
                assert put_response.can_edit_global is True
                current_description = "row4-changed"
            else:
                with pytest.raises(HTTPException) as exc:
                    update_mcp_server(
                        server_id,
                        MCPServerUpdate(description="row4-attempted"),
                        current_user=caller,
                        db=db,
                    )
                assert exc.value.status_code == 403
                current_description = None  # unchanged from creation (None)

            # Row 5: PUT resubmitting a global field's *current* value --
            # not an actual change, so it must succeed regardless of edit
            # rights (the tamper check compares against the stored value).
            row5_response = update_mcp_server(
                server_id,
                MCPServerUpdate(description=current_description),
                current_user=caller,
                db=db,
            )
            assert row5_response.can_edit_global is can_edit_global_config

            # Row 6: PUT touching only a personal field (is_active) --
            # always allowed for a caller with a personal row, independent
            # of global edit rights.
            row6_response = update_mcp_server(
                server_id,
                MCPServerUpdate(is_active=False),
                current_user=caller,
                db=db,
            )
            assert row6_response.can_edit_global is can_edit_global_config
            assert row6_response.is_active is False

            # Row 7: PUT touching a global field and a personal field at
            # the same time -- the global half decides the outcome.
            if can_edit_global_config:
                row7_response = update_mcp_server(
                    server_id,
                    MCPServerUpdate(description="row7-changed", is_active=True),
                    current_user=caller,
                    db=db,
                )
                assert row7_response.can_edit_global is True
                assert row7_response.is_active is True
            else:
                with pytest.raises(HTTPException) as exc:
                    update_mcp_server(
                        server_id,
                        MCPServerUpdate(description="row7-attempted", is_active=True),
                        current_user=caller,
                        db=db,
                    )
                assert exc.value.status_code == 403

            # Row 9: POST .../toggle -- gated on a personal row's mere
            # existence, not on edit rights; both populations have one.
            toggle_response = toggle_mcp_server(server_id, current_user=caller, db=db)
            assert toggle_response.can_edit_global is can_edit_global_config

            # Row 8: DELETE /servers/{id} -- last, since it consumes the
            # row. Gated on is_owner OR can_delete; population B has
            # neither.
            if population == "owner":
                await delete_mcp_server(server_id, current_user=caller, db=db)
            else:
                with pytest.raises(HTTPException) as exc:
                    await delete_mcp_server(server_id, current_user=caller, db=db)
                assert exc.value.status_code == 403

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

    @pytest.mark.parametrize(
        "population", ["owner", "personal_non_owner"], ids=["A=owner", "B=personal"]
    )
    def test_the_apps_listing_can_configure_matches_pre_change_behavior(
        self, db, population
    ):
        """Outside the numbered rows above but touched by this same work:
        ``/api/mcp/apps``'s ``can_configure`` reads only whether a personal
        association row exists (or, absent one, a team verdict) -- both
        constructible populations have a personal row, so both see True,
        with no hook installed, for both connector kinds."""
        owner = _make_user(db, 704)
        member = _make_user(db, 705)
        caller = owner if population == "owner" else member

        server = _make_owned_server(db, owner.id, name=f"parity-apps-mcp-{population}")
        server_id = server.id
        api = _make_owned_api(db, owner.id, name=f"parity-apps-api-{population}")
        api_id = api.id

        if population == "personal_non_owner":
            db.add(
                UserMCPServer(
                    user_id=member.id,
                    mcpserver_id=server_id,
                    is_owner=False,
                    is_active=True,
                )
            )
            db.add(
                UserCustomApi(
                    user_id=member.id,
                    custom_api_id=api_id,
                    is_owner=False,
                    can_edit=False,
                    is_active=True,
                )
            )
            db.commit()

        with snapshot_connector_team_hooks():
            set_connector_team_hooks()
            entries = list_mcp_apps(location="local", current_user=caller, db=db)

        mcp_entry = next(
            e
            for e in entries
            if e["server_id"] == server_id and e["transport"] != "custom_api"
        )
        assert mcp_entry["can_configure"] is True
        api_entry = next(
            e
            for e in entries
            if e["server_id"] == api_id and e["transport"] == "custom_api"
        )
        assert api_entry["can_configure"] is True

    @pytest.mark.parametrize(
        "population", ["owner", "personal_non_owner"], ids=["A=owner", "B=personal"]
    )
    def test_connecting_an_app_can_edit_global_matches_pre_change_behavior(
        self, db, population
    ):
        """Outside the numbered rows above but touched by this same work:
        connecting to a catalog app always creates a fresh, non-owning
        association (``is_owner=False``), so ``can_edit_global`` is False
        regardless of which population is doing the connecting -- pinned
        for both, with no hook installed, so a future change that makes
        this population-dependent would be caught."""
        owner = _make_user(db, 706)
        member = _make_user(db, 707)
        caller = owner if population == "owner" else member
        _seed_catalog_app(db, f"parity-connect-app-{population}")

        with snapshot_connector_team_hooks():
            set_connector_team_hooks()
            response = connect_mcp_app(
                f"parity-connect-app-{population}",
                MCPAppConnectRequest(),
                current_user=caller,
                db=db,
            )

        assert response.can_edit_global is False


class TestListMcpAppsPerRowDegradation:
    """``/api/mcp/apps``'s local-connector loop now resolves every stand-in
    row's verdict, across both connector kinds, with one batched call --
    consolidated from the one-hook-call-per-row shape this route used to
    have. A ref missing from an otherwise-successful answer still degrades
    only that one row's ``can_configure`` to False, the same per-row
    degradation this route has always offered -- now expressed by the
    batch answer omitting a ref rather than a per-row hook call raising. A
    hook that fails for the whole batch call degrades every row that
    needed a verdict, but the response itself stays 200 with every row
    present -- the failure never blanks the list."""

    def test_an_answer_that_omits_one_connector_degrades_only_that_row(self, db):
        owner = _make_user(db, 80)
        member = _make_user(db, 81)
        healthy_mcp = _make_owned_server(db, owner.id, name="healthy-connector")
        omitted_mcp = _make_owned_server(db, owner.id, name="omitted-connector")
        healthy_api = _make_owned_api(db, owner.id, name="healthy-api")
        omitted_api = _make_owned_api(db, owner.id, name="omitted-api")
        healthy_mcp_id, omitted_mcp_id = healthy_mcp.id, omitted_mcp.id
        healthy_api_id, omitted_api_id = healthy_api.id, omitted_api.id

        def partial_access(_db, _user_id, refs):
            # A legitimate "not linked" answer for the two omitted refs,
            # not a failure -- distinct from the whole-batch failure the
            # next test exercises.
            omitted = {("mcp", omitted_mcp_id), ("custom_api", omitted_api_id)}
            return {
                ref: ConnectorAccess(team_owned=True, can_edit=True)
                for ref in refs
                if ref not in omitted
            }

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=partial_access,
                visibility=lambda _db, _uid: {
                    "mcp": {healthy_mcp_id, omitted_mcp_id},
                    "custom_api": {healthy_api_id, omitted_api_id},
                },
            )
            entries = list_mcp_apps(location="local", current_user=member, db=db)

        healthy_mcp_entry = next(e for e in entries if e["server_id"] == healthy_mcp_id)
        omitted_mcp_entry = next(e for e in entries if e["server_id"] == omitted_mcp_id)
        healthy_api_entry = next(
            e
            for e in entries
            if e["server_id"] == healthy_api_id and e["transport"] == "custom_api"
        )
        omitted_api_entry = next(
            e
            for e in entries
            if e["server_id"] == omitted_api_id and e["transport"] == "custom_api"
        )
        assert healthy_mcp_entry["can_configure"] is True
        assert omitted_mcp_entry["can_configure"] is False
        assert healthy_api_entry["can_configure"] is True
        assert omitted_api_entry["can_configure"] is False

    def test_a_failing_hook_does_not_blank_the_whole_apps_list(self, db):
        owner = _make_user(db, 82)
        member = _make_user(db, 83)
        mcp_row = _make_owned_server(db, owner.id, name="stand-in-mcp")
        api_row = _make_owned_api(db, owner.id, name="stand-in-api")
        mcp_id, api_id = mcp_row.id, api_row.id

        def raising_access(_db, _user_id, _refs):
            raise ValueError("hook exploded")

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=raising_access,
                visibility=lambda _db, _uid: {
                    "mcp": {mcp_id},
                    "custom_api": {api_id},
                },
            )
            entries = list_mcp_apps(location="local", current_user=member, db=db)

        mcp_entry = next(e for e in entries if e["server_id"] == mcp_id)
        api_entry = next(
            e
            for e in entries
            if e["server_id"] == api_id and e["transport"] == "custom_api"
        )
        assert mcp_entry["can_configure"] is False
        assert api_entry["can_configure"] is False


def poison_by_raw_statement(db, *, colliding_user_id=None):
    """Poison the session with a raw statement that fails outright.

    On PostgreSQL this aborts the surrounding transaction, so every later
    statement on the same connection is refused until a rollback. On
    SQLite, a failed Core-level statement like this one does not put the
    ORM ``Session`` into a deactivated state the way a failed flush does
    (see ``poison_by_orm_flush``) -- so this shape's recovery proof lives
    in the PostgreSQL-only sibling suite
    (test_connector_hook_session_fault_postgresql.py), not in the tests
    that use this factory here. ``colliding_user_id`` is accepted and
    ignored so both poison factories share one call signature.
    """
    del colliding_user_id
    db.execute(sa.text("select * from no_such_table_at_all"))


def poison_by_orm_flush(db, *, colliding_user_id):
    """Poison the session by flushing a row that violates a real unique
    constraint -- unlike ``poison_by_raw_statement``, this poisons the
    ORM ``Session`` itself (not only the underlying DB transaction) on
    every backend: SQLAlchemy marks the session's transaction inactive
    after a failed flush, and any later operation on it raises
    ``PendingRollbackError`` until a rollback runs.
    """
    db.add(User(id=colliding_user_id, username="flush-poison-dup", password_hash="x"))
    db.flush()


POISON_SHAPES = [poison_by_raw_statement, poison_by_orm_flush]
POISON_SHAPE_IDS = ["raw-statement", "orm-flush"]


def _seed_catalog_app(db, app_id: str = "session-fault-app") -> None:
    db.add(
        PublicMCPApp(
            app_id=app_id,
            name=app_id,
            description="Session fault test app",
            transport="stdio",
            launch_config={"command": "npx", "args": ["-y", app_id]},
        )
    )
    db.commit()


class TestSessionRecoveryAfterHookFailure:
    """A hook that leaves a failed statement on the shared session must not
    turn a route that would otherwise succeed (or gracefully degrade) into
    a 500 -- the seam's single hook-invocation door restores the session
    before the failure ever reaches a caller to convert into a typed error
    (see ``_call_connector_hook_gate`` in connector_team_scope.py).

    ``poison_by_raw_statement`` only actually poisons PostgreSQL (see its
    docstring); it is still parametrized here so the SQLite half of this
    file documents that shape's expected (correct, unaffected) behavior
    too. The PostgreSQL-only proof that this shape needs the fix lives in
    test_connector_hook_session_fault_postgresql.py.
    """

    @pytest.mark.parametrize("poison", POISON_SHAPES, ids=POISON_SHAPE_IDS)
    def test_a_toggle_that_already_committed_still_returns_200_when_the_hook_poisons_the_session(
        self, db, poison
    ):
        owner = _make_user(db, 90)
        server = _make_owned_server(db, owner.id, name="toggle-poison-target")
        server_id = server.id
        owner_id = owner.id

        def poisoning_access(_db, _user_id, _refs):
            poison(_db, colliding_user_id=owner_id)
            return {}

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=poisoning_access)
            response = toggle_mcp_server(server_id, current_user=owner, db=db)

        assert response.can_edit_global is True

        # No rollback here on purpose: the query below is the statement
        # that proves the seam's hook door restored this session, not just
        # an incidental fresh read (see the same note in
        # test_connector_hook_session_fault_postgresql.py, where the
        # orm-flush shape poisons on every backend the same way).
        refreshed = (
            db.query(UserMCPServer)
            .filter(
                UserMCPServer.user_id == owner_id,
                UserMCPServer.mcpserver_id == server_id,
            )
            .one()
        )
        # The connector was created active; toggling it once must have
        # flipped it to inactive, and that flip must have durably
        # committed (it happens before the hook is ever consulted) even
        # though the hook poisoned the session afterward.
        assert refreshed.is_active is False

    @pytest.mark.parametrize("poison", POISON_SHAPES, ids=POISON_SHAPE_IDS)
    def test_connecting_an_app_still_returns_200_when_the_hook_poisons_the_session(
        self, db, poison
    ):
        member = _make_user(db, 91)
        member_id = member.id
        _seed_catalog_app(db, "connect-poison-app")

        def poisoning_access(_db, _user_id, _refs):
            poison(_db, colliding_user_id=member_id)
            return {}

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=poisoning_access)
            response = connect_mcp_app(
                "connect-poison-app",
                MCPAppConnectRequest(),
                current_user=member,
                db=db,
            )

        # Connecting never grants ownership (a fresh association is always
        # is_owner=False), so with the hook degraded to no verdict at all,
        # can_edit_global is False here -- the same value this route
        # always reported before any verdict existed.
        assert response.can_edit_global is False

        # No rollback here on purpose -- see the same note in the toggle
        # test above.
        assoc = (
            db.query(UserMCPServer)
            .join(MCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
            .filter(
                UserMCPServer.user_id == member_id,
                MCPServer.name == "connect-poison-app",
            )
            .one()
        )
        # ``.one()`` already raises when the row is absent, so asserting it is
        # not None asserts nothing. What this test is actually about is that
        # the association survived the poisoned session with the shape connect
        # writes: a non-owning, active personal link.
        assert (assoc.is_owner, assoc.is_active) == (False, True)

    # test_the_servers_listing_still_returns_every_row_when_the_hook_poisons_the_session
    # is not here: it lives in TestListMcpServersPerRowDegradation below,
    # next to /api/mcp/servers's own per-request degradation catch --
    # that catch did not exist yet at the point this class was written,
    # so the poison test could not have asserted a guarantee this route
    # did not yet provide.

    def test_the_apps_listing_still_returns_every_row_when_the_hook_poisons_the_session(
        self, db
    ):
        owner = _make_user(db, 94)
        member = _make_user(db, 95)
        member_id = member.id
        server = _make_owned_server(db, owner.id, name="apps-list-poison-target")
        server_id = server.id

        def poisoning_access(_db, _user_id, _refs):
            poison_by_orm_flush(_db, colliding_user_id=member_id)
            return {}

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=poisoning_access,
                visibility=lambda _db, _uid: {"mcp": {server_id}, "custom_api": set()},
            )
            entries = list_mcp_apps(location="local", current_user=member, db=db)

        entry = next(e for e in entries if e["server_id"] == server_id)
        assert entry["can_configure"] is False

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


class TestListMcpServersPerRowDegradation:
    """``/api/mcp/servers``'s response loops resolve every row that still
    needs a verdict -- a non-owner personal row, or a stand-in row with no
    personal row at all -- with one batched call (see the shape built in
    get_mcp_servers). A ref missing from an otherwise-successful answer
    degrades only that one row's ``can_edit_global`` to False, the same
    per-row degradation this route has always offered. A hook that fails
    for the whole batch call degrades every row that needed a verdict, but
    the response itself stays 200 with every row present -- the failure
    never blanks the list. Mirrors ``TestListMcpAppsPerRowDegradation``
    above for the sister listing endpoint."""

    def test_an_answer_that_omits_one_connector_degrades_only_that_row(self, db):
        owner = _make_user(db, 84)
        member = _make_user(db, 85)
        healthy = _make_owned_server(db, owner.id, name="servers-healthy")
        omitted = _make_owned_server(db, owner.id, name="servers-omitted")
        healthy_id, omitted_id = healthy.id, omitted.id

        def partial_access(_db, _user_id, refs):
            # A legitimate "not linked" answer for the omitted ref, not a
            # failure -- distinct from the whole-batch failure the next
            # test exercises.
            skip = {("mcp", omitted_id)}
            return {
                ref: ConnectorAccess(team_owned=True, can_edit=True)
                for ref in refs
                if ref not in skip
            }

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=partial_access,
                visibility=lambda _db, _uid: {
                    "mcp": {healthy_id, omitted_id},
                    "custom_api": set(),
                },
            )
            entries = get_mcp_servers(current_user=member, db=db)

        healthy_entry = next(e for e in entries if e.id == healthy_id)
        omitted_entry = next(e for e in entries if e.id == omitted_id)
        assert healthy_entry.can_edit_global is True
        assert omitted_entry.can_edit_global is False

    def test_a_failing_hook_does_not_blank_the_whole_servers_list(self, db):
        owner = _make_user(db, 86)
        member = _make_user(db, 87)
        owned_by_member = _make_owned_server(db, member.id, name="servers-member-owned")
        personal_non_owner = _make_owned_server(
            db, owner.id, name="servers-personal-non-owner"
        )
        db.add(
            UserMCPServer(
                user_id=member.id,
                mcpserver_id=personal_non_owner.id,
                is_owner=False,
                is_active=True,
            )
        )
        db.commit()
        stand_in = _make_owned_server(db, owner.id, name="servers-stand-in")

        def raising_access(_db, _user_id, _refs):
            raise ValueError("hook exploded")

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=raising_access,
                visibility=lambda _db, _uid: {
                    "mcp": {stand_in.id},
                    "custom_api": set(),
                },
            )
            entries = get_mcp_servers(current_user=member, db=db)

        assert {e.id for e in entries} == {
            owned_by_member.id,
            personal_non_owner.id,
            stand_in.id,
        }
        owned_entry = next(e for e in entries if e.id == owned_by_member.id)
        personal_entry = next(e for e in entries if e.id == personal_non_owner.id)
        stand_in_entry = next(e for e in entries if e.id == stand_in.id)
        # The owner's own row never needed a verdict at all -- the edit
        # branch returns True on is_owner alone, so a failed batch call
        # cannot touch it.
        assert owned_entry.can_edit_global is True
        assert personal_entry.can_edit_global is False
        assert stand_in_entry.can_edit_global is False

    def test_the_servers_listing_still_returns_every_row_when_the_hook_poisons_the_session(
        self, db
    ):
        """Sibling to the SQLite-side poison tests in
        ``TestSessionRecoveryAfterHookFailure`` above -- deferred to this
        class specifically because ``/api/mcp/servers`` had no per-request
        degradation catch of its own until this same revision added one;
        before that, a poisoned session on this route would have failed
        the whole request regardless of any session-recovery fix."""
        owner = _make_user(db, 88)
        member = _make_user(db, 89)
        member_id = member.id
        server = _make_owned_server(db, owner.id, name="servers-list-poison-target")
        server_id = server.id

        def poisoning_access(_db, _user_id, _refs):
            poison_by_orm_flush(_db, colliding_user_id=member_id)
            return {}

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=poisoning_access,
                visibility=lambda _db, _uid: {
                    "mcp": {server_id},
                    "custom_api": set(),
                },
            )
            entries = get_mcp_servers(current_user=member, db=db)

        entry = next(e for e in entries if e.id == server_id)
        assert entry.can_edit_global is False

    def test_the_degraded_listing_log_names_the_failure_it_degraded_on(
        self, db, caplog
    ):
        """The degrade arm keeps the response at 200, so this warning is the
        only record of why a row lost its reported edit right. A hook that
        raises ``ConnectorRuntimeError`` itself reaches this arm untouched --
        the seam re-raises it without logging a traceback of its own -- so
        the line here has to carry the code, or the failure has no identity
        anywhere."""
        owner = _make_user(db, 90)
        member = _make_user(db, 91)
        server = _make_owned_server(db, owner.id, name="servers-log-identity")
        server_id = server.id

        def typed_failure(_db, _user_id, _refs):
            raise ConnectorRuntimeError(
                "team_directory_unreachable",
                "planted failure",
                status_code=503,
            )

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=typed_failure,
                visibility=lambda _db, _uid: {
                    "mcp": {server_id},
                    "custom_api": set(),
                },
            )
            with caplog.at_level(logging.WARNING, logger="xagent.web.api.mcp"):
                entries = get_mcp_servers(current_user=member, db=db)

        entry = next(e for e in entries if e.id == server_id)
        assert entry.can_edit_global is False
        degraded = [
            record.getMessage()
            for record in caplog.records
            if "Connector access resolution failed" in record.getMessage()
        ]
        assert len(degraded) == 1
        assert "team_directory_unreachable" in degraded[0]


class TestSingleServerAccessResolutionFailure:
    """A single MCP server's verdict plays two different roles depending on
    the route: ``GET`` uses it as decoration on a row the caller can
    already read (a personal row, or a team gate that already passed), so
    a resolution failure there degrades ``can_edit_global`` to False and
    the read still succeeds. ``PUT`` uses the same verdict as the gate
    itself for a non-owner caller, so a resolution failure there must
    still fail closed with a typed 503 -- never a silent 200 or a 404 that
    would misreport "does not exist" for a connector the caller merely
    could not be asked about."""

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

    def test_updating_one_server_still_fails_closed_on_a_personal_only_payload(
        self, db
    ):
        owner = _make_user(db, 106)
        member = _make_user(db, 107)
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
            with pytest.raises(HTTPException) as exc:
                update_mcp_server(
                    server_id,
                    MCPServerUpdate(is_active=False),
                    current_user=member,
                    db=db,
                )

        # PUT never degrades, even for a payload that only touches the
        # caller's own personal fields: the verdict is the gate that
        # decides whether this caller may write at all.
        assert exc.value.status_code == 503


class TestAdminInspectingAnotherUsersListReportsPerKindSubject:
    """``GET /api/mcp/servers?user_id=<target>`` reports ``can_edit_global``
    from a different subject depending on connector kind, today: an MCP
    row blends the *acting admin's own* bypass with the target's team
    verdict (``_check_mcp_permission``'s ``is_admin`` short-circuit runs
    before any verdict is even consulted), while a Custom API row reports
    purely the *target's own* ``can_edit`` and team verdict, since Custom
    API's write gate has no admin bypass at all
    (``_custom_api_to_mcp_response`` never reads ``is_admin``).

    This pins the subject mix as it exists today -- it is not an
    endorsement of it. "Whose capability should this field describe" is
    an undecided product rule, tracked in xorbitsai/xagent#1703. This
    test is a regression guard against either subject silently changing,
    not a statement that the current split is correct.
    """

    def test_admin_inspecting_another_users_list_reports_each_kind_from_its_own_subject(
        self, db
    ):
        admin = _make_user(db, 800, is_admin=True)
        target = _make_user(db, 801)
        other_owner = _make_user(db, 802)

        server = _make_owned_server(db, other_owner.id, name="admin-subject-mcp")
        server_id = server.id
        db.add(
            UserMCPServer(
                user_id=target.id,
                mcpserver_id=server_id,
                is_owner=False,
                is_active=True,
            )
        )
        api = _make_owned_api(db, target.id, name="admin-subject-api")
        api_id = api.id
        # A second Custom API the target can see but genuinely cannot
        # edit -- a non-owning personal row, with the hook denying the
        # target's own verdict on it too. Distinct from `api` above:
        # `api`'s True could in principle come from an admin bypass this
        # module does not have rather than from the target's own
        # can_edit, and the two would be indistinguishable there (True or
        # True is still True). This row is the one that actually proves
        # the subject is the target and not the admin -- if a bypass on
        # is_admin were ever added to Custom API's response builder, the
        # admin's own True would leak into this row and flip it.
        other_owner_api = _make_owned_api(
            db, other_owner.id, name="admin-subject-denied-api"
        )
        denied_api_id = other_owner_api.id
        db.add(
            UserCustomApi(
                user_id=target.id,
                custom_api_id=denied_api_id,
                is_owner=False,
                can_edit=False,
                is_active=True,
            )
        )
        db.commit()

        # Denies the target's own verdict on every ref -- but, deliberately,
        # *grants* anyone else's, including the admin's own id. A correct
        # list implementation always asks about the target being
        # inspected, regardless of who is doing the viewing, so this
        # granting branch should never be reached for this list call. A
        # mutation that asked about the *viewer's* id instead of the
        # target's would reach it and leak a wrong grant into a
        # target-subject row -- this is what makes that class of bug
        # visible rather than merely restating "the target is denied".
        def access_hook_keyed_on_who_is_asked_about(_db, user_id, refs):
            if user_id == target.id:
                return {
                    ref: ConnectorAccess(team_owned=True, can_edit=False)
                    for ref in refs
                }
            return {
                ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
            }

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=access_hook_keyed_on_who_is_asked_about)
            list_entries = get_mcp_servers(user_id=target.id, current_user=admin, db=db)

        mcp_entry = next(r for r in list_entries if r.id == server_id)
        api_entry = next(
            r for r in list_entries if r.id == api_id and r.transport == "custom_api"
        )
        denied_api_entry = next(
            r
            for r in list_entries
            if r.id == denied_api_id and r.transport == "custom_api"
        )

        # The MCP row's subject is the acting admin: True here, even
        # though the target's own verdict (fetched for the target, not
        # the admin) denies edit -- the admin bypass wins before any
        # verdict is consulted.
        assert mcp_entry.can_edit_global is True

        # The Custom API row's subject is the target: True because the
        # target owns this API outright (can_edit=True on their own row),
        # independent of the acting admin's identity or the denying
        # verdict above -- if this test's admin were somehow the subject
        # here too, this would need to be False (the verdict denies it).
        assert api_entry.can_edit_global is True

        # This row is the one that actually distinguishes "target" from
        # "admin" as the subject: the target's own verdict on it is
        # denied and they do not own it, so it must be False despite the
        # acting caller being an admin. An admin bypass leaking into
        # Custom API's response builder would flip this to True.
        assert denied_api_entry.can_edit_global is False
