"""The edit right on a team-linked MCP connector: ``GET``/``PUT
/api/mcp/servers/{server_id}`` resolve a caller with no personal row
through the connector access hook instead of 404ing outright, the edit
branch of ``_check_mcp_permission`` falls back to that verdict, the two
per-user fields reject outright for a caller with no row to hold them, a
raising hook surfaces as its declared status rather than a 500, a
stand-in whose verdict denies edit is refused outright rather than
reported as an empty success, and a personal row that vanishes while this
request waits for the definition row's lock is answered with the gate's
own 404 -- an admin with neither a personal row nor a verdict included,
though an admin who does hold a verdict is a separate case this module
does not cover.

Every test that installs the access hook does so through
``snapshot_connector_team_hooks`` so no hook state leaks between tests or
into suites that run after this one.
"""

from __future__ import annotations

import ast
import importlib
import inspect

import pytest
import sqlalchemy as sa
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from xagent.web.api.mcp import (
    MCPServerUpdate,
    _check_mcp_permission,
    get_mcp_server,
    update_mcp_server,
)
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


def _sequenced_access_hook(*answers):
    """An access hook that answers differently on successive calls, so a
    test can make the second (post-lock) resolution disagree with the
    first. ``None`` in the sequence means an empty answer -- the batch
    contract's way of saying "the caller's team does not link this". An
    entry that is an exception instance is raised instead of returned, so a
    test can make the second resolution fail outright. The last entry
    repeats for any further call. Records every call's ``refs`` on
    ``.calls`` so a test can pin how many round trips the route pays."""
    calls: list[object] = []

    def hook(db, user_id, refs):
        calls.append(refs)
        index = min(len(calls) - 1, len(answers) - 1)
        answer = answers[index]
        if isinstance(answer, BaseException):
            raise answer
        if answer is None:
            return {}
        return {ref: answer for ref in refs}

    hook.calls = calls
    return hook


class TestCheckMcpPermissionTeamAccessFallback:
    """The team access verdict as a fallback on the ``edit`` branch.

    Covers only the verdict-aware behavior; the owner/admin/delete
    behavior this function has always had is covered by
    ``test_check_mcp_permission`` in test_mcp_api.py."""

    def test_owner_wins_the_edit_branch_without_consulting_the_verdict(self):
        from unittest.mock import MagicMock

        owner = MagicMock(is_owner=True, can_delete=False)
        # A verdict that would deny edit rights on its own is still beaten
        # by is_owner -- the verdict is a fallback, never an override.
        denying_access = ConnectorAccess(team_owned=True, can_edit=False)
        assert (
            _check_mcp_permission(
                owner, is_admin=False, require="edit", team_access=denying_access
            )
            is True
        )

    def test_non_owner_falls_back_to_a_granting_verdict(self):
        from unittest.mock import MagicMock

        guest = MagicMock(is_owner=False, can_delete=False)
        granting_access = ConnectorAccess(team_owned=True, can_edit=True)
        assert (
            _check_mcp_permission(
                guest, is_admin=False, require="edit", team_access=granting_access
            )
            is True
        )

    def test_non_owner_stays_denied_by_a_linked_but_not_editable_verdict(self):
        from unittest.mock import MagicMock

        guest = MagicMock(is_owner=False, can_delete=False)
        linked_only = ConnectorAccess(team_owned=True, can_edit=False)
        assert (
            _check_mcp_permission(
                guest, is_admin=False, require="edit", team_access=linked_only
            )
            is False
        )

    def test_missing_team_access_keyword_behaves_exactly_as_before(self):
        from unittest.mock import MagicMock

        owner = MagicMock(is_owner=True, can_delete=False)
        guest = MagicMock(is_owner=False, can_delete=False)
        assert _check_mcp_permission(owner, is_admin=False, require="edit") is True
        assert _check_mcp_permission(guest, is_admin=False, require="edit") is False

    def test_delete_branch_ignores_team_access_entirely(self):
        """Delete stays exactly as it is today: a granting verdict changes
        nothing on the ``delete`` branch, which reads only ``can_delete``."""
        from unittest.mock import MagicMock

        guest = MagicMock(is_owner=False, can_delete=False)
        granting_access = ConnectorAccess(team_owned=True, can_edit=True)
        assert (
            _check_mcp_permission(
                guest, is_admin=False, require="delete", team_access=granting_access
            )
            is False
        )


class TestGateHelperOnGetAndPut:
    def test_get_404s_for_an_unrelated_user_with_no_link_and_no_team_access(self, db):
        owner = _make_user(db, 1)
        stranger = _make_user(db, 2)
        server = _make_owned_server(db, owner.id)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=lambda db, user_id, refs: {})
            with pytest.raises(HTTPException) as exc:
                get_mcp_server(server.id, current_user=stranger, db=db)
        assert exc.value.status_code == 404

    def test_get_returns_the_stand_in_for_a_team_member_with_no_personal_row(self, db):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        server = _make_owned_server(db, owner.id)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
                }
            )
            response = get_mcp_server(server.id, current_user=member, db=db)

        assert response.id == server.id
        assert response.user_id == member.id

    def test_get_owner_behaviour_is_unchanged_with_no_hook_installed(self, db):
        owner = _make_user(db, 1)
        server = _make_owned_server(db, owner.id)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks()
            response = get_mcp_server(server.id, current_user=owner, db=db)

        assert response.id == server.id
        assert response.can_edit_global is True


class TestPutWiringForATeamEditor:
    def test_team_editor_edit_is_durable_and_creates_no_association_row(self, db):
        owner = _make_user(db, 1)
        editor = _make_user(db, 2)
        server = _make_owned_server(db, owner.id)
        server_id = server.id

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
                }
            )
            response = update_mcp_server(
                server_id,
                MCPServerUpdate(description="edited by the team"),
                current_user=editor,
                db=db,
            )

        assert response.description == "edited by the team"

        # Durability, not staging: the rollback below is what makes this a
        # real check -- a same-session query would still see an uncommitted
        # UPDATE even if the route never committed.
        db.rollback()
        refreshed = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert refreshed.description == "edited by the team"

        # The edit did not fabricate a personal association for the team
        # editor -- that would be a get-or-create write on an
        # authorization path.
        assert (
            db.query(UserMCPServer).filter(UserMCPServer.user_id == editor.id).first()
            is None
        )

    def test_a_member_with_a_personal_row_edits_the_shared_config_durably(self, db):
        """Durability and no-fabricated-row, for the population the other
        tests in this class do not cover: a caller who does have a personal
        row, but one that grants no edit, widened by a granting team
        verdict. Every other test here uses the stand-in population, which
        has no personal row at all."""
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        server = _make_owned_server(db, owner.id, name="both-rows-mcp")
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
                access=_sequenced_access_hook(
                    ConnectorAccess(team_owned=True, can_edit=True)
                )
            )
            response = update_mcp_server(
                server_id,
                MCPServerUpdate(description="widened-by-the-team"),
                current_user=member,
                db=db,
            )

        assert response.can_edit_global is True

        # Durability, not staging.
        db.rollback()
        refreshed = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert refreshed.description == "widened-by-the-team"
        # The caller's one personal row, not a second one.
        assert (
            db.query(UserMCPServer)
            .filter(
                UserMCPServer.user_id == member.id,
                UserMCPServer.mcpserver_id == server_id,
            )
            .count()
            == 1
        )

    def test_view_only_team_member_cannot_tamper_the_shared_config(self, db):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        server = _make_owned_server(db, owner.id)
        server_id = server.id
        # A personal, non-owner association row (population D), not a
        # stand-in: a stand-in whose verdict denies edit is now refused
        # outright before this route ever reaches the shared-config tamper
        # check this test is pinning (see
        # TestADenyingStandInIsRefusedRatherThanReportedSuccessful). D still
        # has no can_edit of its own and no granting verdict, so it hits
        # the same tamper-check 403 this test always meant to cover.
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
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=False)
                    for ref in refs
                }
            )
            with pytest.raises(HTTPException) as exc:
                update_mcp_server(
                    server_id,
                    MCPServerUpdate(description="should not land"),
                    current_user=member,
                    db=db,
                )
        assert exc.value.status_code == 403
        assert "shared configuration" in exc.value.detail

    def test_rename_propagates_to_team_agent_selectors(self, db, monkeypatch):
        """A rename by a team editor must rewrite the team's agent
        selectors, exactly as an owner's rename does. Mutation check:
        deleting the ``rename_team_connector`` call turns this red."""
        owner = _make_user(db, 1)
        editor = _make_user(db, 2)
        server = _make_owned_server(db, owner.id, name="old-name")
        server_id = server.id

        calls: list[tuple[str, str]] = []

        def fake_renamed_hook(_db, _user_id, _connector_type, _connector_id, old, new):
            calls.append((old, new))

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
                },
                renamed=fake_renamed_hook,
            )
            update_mcp_server(
                server_id,
                MCPServerUpdate(name="new-name"),
                current_user=editor,
                db=db,
            )

        assert calls == [("old-name", "new-name")]


class TestUserEnvAndIsActiveRejectionForAStandIn:
    def test_user_env_from_a_caller_with_no_personal_row_is_400_not_a_silent_drop(
        self, db
    ):
        owner = _make_user(db, 1)
        editor = _make_user(db, 2)
        server = _make_owned_server(db, owner.id, name="unchanged-name")
        server_id = server.id

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
                }
            )
            with pytest.raises(HTTPException) as exc:
                update_mcp_server(
                    server_id,
                    MCPServerUpdate(user_env={"API_KEY": "x"}),
                    current_user=editor,
                    db=db,
                )

        assert exc.value.status_code == 400
        assert "personal connection" in str(exc.value.detail)

        db.rollback()
        refreshed = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert refreshed.name == "unchanged-name"
        assert (
            db.query(UserMCPServer).filter(UserMCPServer.user_id == editor.id).first()
            is None
        )

    def test_is_active_from_a_caller_with_no_personal_row_is_400_not_a_silent_drop(
        self, db
    ):
        owner = _make_user(db, 1)
        editor = _make_user(db, 2)
        server = _make_owned_server(db, owner.id, name="still-unchanged")
        server_id = server.id

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
                }
            )
            with pytest.raises(HTTPException) as exc:
                update_mcp_server(
                    server_id,
                    MCPServerUpdate(is_active=False),
                    current_user=editor,
                    db=db,
                )

        assert exc.value.status_code == 400

        db.rollback()
        refreshed = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert refreshed.name == "still-unchanged"
        assert (
            db.query(UserMCPServer).filter(UserMCPServer.user_id == editor.id).first()
            is None
        )


class TestAnExplicitNullPersonalFieldIsRefused:
    """``MCPServerUpdate`` accepts an explicit ``null`` for ``user_env`` and
    ``is_active`` the same as it accepts any other value, and Pydantic
    records that in ``model_fields_set`` -- so ``{"user_env": null}`` carries
    the field and must be refused the same way any other value is, on both
    sides of the definition-row lock. Testing the value instead of presence
    let such a payload through to a 200 that stored nothing, and -- mixed
    with a shared field -- to a 200 that applied the shared half while
    silently dropping the personal one.
    """

    @pytest.mark.parametrize("field", ["user_env", "is_active"])
    def test_an_explicit_null_personal_field_is_refused(self, db, field):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        server = _make_owned_server(db, owner.id, name="explicit-null-pre-lock")
        server_id = server.id
        payload = MCPServerUpdate.model_validate({field: None})

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
                }
            )
            with pytest.raises(HTTPException) as exc:
                update_mcp_server(server_id, payload, current_user=member, db=db)

        assert exc.value.status_code == 400
        assert exc.value.detail == (
            "No personal connection exists to configure user_env or "
            "is_active for this server"
        )

    @pytest.mark.parametrize("field", ["user_env", "is_active"])
    def test_an_explicit_null_personal_field_is_refused_after_the_lock_too(
        self, db, field
    ):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        member_id = member.id
        server = _make_owned_server(db, owner.id, name="explicit-null-post-lock")
        server.description = "original"
        server_id = server.id
        db.add(
            UserMCPServer(
                user_id=member_id,
                mcpserver_id=server_id,
                is_owner=False,
                is_active=True,
            )
        )
        db.commit()
        payload = MCPServerUpdate.model_validate(
            {field: None, "description": "should-not-land"}
        )

        real_query = db.query
        fired = False

        def query_and_delete_on_first_recheck(*entities, **kwargs):
            nonlocal fired
            if entities == (UserMCPServer,) and not fired:
                fired = True
                db.execute(
                    sa.delete(UserMCPServer).where(
                        UserMCPServer.user_id == member_id,
                        UserMCPServer.mcpserver_id == server_id,
                    )
                )
                db.commit()
            return real_query(*entities, **kwargs)

        db.query = query_and_delete_on_first_recheck
        try:
            with snapshot_connector_team_hooks():
                set_connector_team_hooks(
                    access=lambda db, user_id, refs: {
                        ref: ConnectorAccess(team_owned=True, can_edit=True)
                        for ref in refs
                    }
                )
                with pytest.raises(HTTPException) as exc:
                    update_mcp_server(server_id, payload, current_user=member, db=db)
        finally:
            db.query = real_query

        assert fired
        assert exc.value.status_code == 400
        assert exc.value.detail == (
            "No personal connection exists to configure user_env or "
            "is_active for this server"
        )

        db.rollback()
        refreshed = real_query(MCPServer).filter(MCPServer.id == server_id).one()
        assert refreshed.description == "original"

    def test_a_denied_stand_in_with_an_explicit_null_still_gets_the_personal_field_400(
        self, db
    ):
        """The lock-side personal-field 400 fires before the lock-side
        denied-stand-in 404 does, even for a caller whose re-derived verdict
        denies edit outright. Both guards would refuse this request, so only
        running the mutation that swaps their order (moving the 404 ahead of
        the 400) can tell whether the ordering the docstring above the 404
        guard promises is real rather than untested."""
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        member_id = member.id
        server = _make_owned_server(db, owner.id, name="explicit-null-denied-post-lock")
        server.description = "original"
        server_id = server.id
        db.add(
            UserMCPServer(
                user_id=member_id,
                mcpserver_id=server_id,
                is_owner=False,
                is_active=True,
            )
        )
        db.commit()
        payload = MCPServerUpdate.model_validate(
            {"is_active": None, "description": "should-not-land"}
        )

        real_query = db.query
        fired = False

        def query_and_delete_on_first_recheck(*entities, **kwargs):
            nonlocal fired
            if entities == (UserMCPServer,) and not fired:
                fired = True
                db.execute(
                    sa.delete(UserMCPServer).where(
                        UserMCPServer.user_id == member_id,
                        UserMCPServer.mcpserver_id == server_id,
                    )
                )
                db.commit()
            return real_query(*entities, **kwargs)

        db.query = query_and_delete_on_first_recheck
        try:
            with snapshot_connector_team_hooks():
                set_connector_team_hooks(
                    access=lambda db, user_id, refs: {
                        ref: ConnectorAccess(team_owned=True, can_edit=False)
                        for ref in refs
                    }
                )
                with pytest.raises(HTTPException) as exc:
                    update_mcp_server(server_id, payload, current_user=member, db=db)
        finally:
            db.query = real_query

        assert fired
        assert exc.value.status_code == 400
        assert exc.value.detail == (
            "No personal connection exists to configure user_env or "
            "is_active for this server"
        )

        db.rollback()
        refreshed = real_query(MCPServer).filter(MCPServer.id == server_id).one()
        assert refreshed.description == "original"


class TestARowThatVanishesUnderTheLockIsTheGatesOwn404:
    """A caller whose personal association row is deleted while this request
    waits for the definition-row lock is answered the same way a caller who
    never had one is: 404, nothing written. A platform admin is not exempt --
    the gate 404s an admin with neither a personal row nor a verdict, and this
    answers the same population the same way once the row is gone.

    The deletion fires on this route's first single-entity ``UserMCPServer``
    query, which is the re-read the lock is taken for: the gate's own read
    joins that table to ``MCPServer``, so a bare ``(UserMCPServer,)`` can only
    be the post-lock re-read.
    """

    @pytest.mark.parametrize(
        "is_admin", [False, True], ids=["member", "platform-admin"]
    )
    def test_a_personal_row_deleted_during_the_lock_wait_is_the_gates_own_404(
        self, db, is_admin
    ):
        caller = _make_user(db, 2, is_admin=is_admin)
        server = MCPServer(
            name="shared-server", transport="stdio", managed="external", command="true"
        )
        db.add(server)
        db.commit()
        server_id = server.id
        db.add(
            UserMCPServer(
                user_id=caller.id,
                mcpserver_id=server_id,
                is_owner=False,
                is_active=True,
            )
        )
        db.commit()

        real_query = db.query
        fired = False

        def query_and_delete_on_first_recheck(*entities, **kwargs):
            nonlocal fired
            if entities == (UserMCPServer,) and not fired:
                fired = True
                db.execute(
                    sa.delete(UserMCPServer).where(
                        UserMCPServer.user_id == caller.id,
                        UserMCPServer.mcpserver_id == server_id,
                    )
                )
                db.commit()
            return real_query(*entities, **kwargs)

        db.query = query_and_delete_on_first_recheck
        try:
            with pytest.raises(HTTPException) as exc:
                update_mcp_server(
                    server_id,
                    MCPServerUpdate(name="renamed"),
                    current_user=caller,
                    db=db,
                )
        finally:
            db.query = real_query

        assert fired
        assert exc.value.status_code == 404
        assert exc.value.detail == "MCP server not found"
        db.rollback()
        refreshed = real_query(MCPServer).filter(MCPServer.id == server_id).one()
        assert refreshed.name == "shared-server"


class TestADeniedStandInIsRefusedAfterTheLock:
    """The lock-side counterpart of
    ``TestADenyingStandInIsRefusedRatherThanReportedSuccessful``: a caller
    admitted on a real non-owner association row, holding a verdict that
    links the connector but denies edit, whose row is then deleted while
    this request waits for the definition-row lock. That caller reaches the
    lock as a stand-in whose re-derived answer is still "no" -- refused
    before the name is read, before the tamper check, and before anything
    is rebuilt or committed, the same way the gate would have refused a
    stand-in with a denying verdict from the start.
    """

    @pytest.mark.parametrize(
        "make_payload",
        [
            lambda server: MCPServerUpdate(config={"env": {"K": "v"}}),
            lambda server: MCPServerUpdate(description=server.description),
        ],
        ids=["secrets-only-payload", "resubmit-current-value-payload"],
    )
    def test_a_denied_stand_in_after_the_lock_is_refused_before_anything_is_written(
        self, db, make_payload
    ):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        member_id = member.id
        server = _make_owned_server(db, owner.id, name="denied-stand-in-after-the-lock")
        server.description = "the connector's current description"
        server_id = server.id
        db.add(
            UserMCPServer(
                user_id=member_id,
                mcpserver_id=server_id,
                is_owner=False,
                is_active=True,
            )
        )
        db.commit()
        payload = make_payload(server)

        real_query = db.query
        fired = False

        def query_and_delete_on_first_recheck(*entities, **kwargs):
            nonlocal fired
            if entities == (UserMCPServer,) and not fired:
                fired = True
                db.execute(
                    sa.delete(UserMCPServer).where(
                        UserMCPServer.user_id == member_id,
                        UserMCPServer.mcpserver_id == server_id,
                    )
                )
                db.commit()
            return real_query(*entities, **kwargs)

        db.query = query_and_delete_on_first_recheck
        try:
            with snapshot_connector_team_hooks():
                set_connector_team_hooks(
                    access=lambda db, user_id, refs: {
                        ref: ConnectorAccess(team_owned=True, can_edit=False)
                        for ref in refs
                    }
                )
                with pytest.raises(HTTPException) as exc:
                    update_mcp_server(server_id, payload, current_user=member, db=db)
        finally:
            db.query = real_query

        assert fired
        assert exc.value.status_code == 404
        assert exc.value.detail == "MCP server not found"

        db.rollback()
        refreshed = real_query(MCPServer).filter(MCPServer.id == server_id).one()
        # A legacy NULL, still unnormalized: the rebuild that would turn it
        # into ``[]`` never ran.
        assert refreshed.concurrent_tools is None


class TestTypedErrorArm:
    """A raising hook still surfaces its declared status for a caller whose
    own personal row does not already decide the answer -- the verdict is
    genuinely the gate for that population, and must stay fail-closed. An
    owner's row already decides the answer on its own, so a hook is never
    called for it at all; that population is pinned separately, below, in
    ``TestOwnerIsImmuneToAHookFailure``."""

    def test_get_surfaces_a_raising_hooks_declared_status(self, db):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        server = _make_owned_server(db, owner.id)

        def boom(*_a, **_k):
            raise ValueError("hook exploded")

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=boom)
            with pytest.raises(HTTPException) as exc:
                get_mcp_server(server.id, current_user=member, db=db)

        assert exc.value.status_code == 503

    def test_put_surfaces_a_raising_hooks_declared_status_and_leaves_the_row_unchanged(
        self, db
    ):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        server = _make_owned_server(db, owner.id, name="pristine")
        server_id = server.id

        def boom(*_a, **_k):
            raise ValueError("hook exploded")

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=boom)
            with pytest.raises(HTTPException) as exc:
                update_mcp_server(
                    server_id,
                    MCPServerUpdate(name="should-not-land"),
                    current_user=member,
                    db=db,
                )

        assert exc.value.status_code == 503

        db.rollback()
        refreshed = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert refreshed.name == "pristine"

    def test_put_passes_through_a_planted_connector_runtime_error_by_its_own_status(
        self, db
    ):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        server = _make_owned_server(db, owner.id)

        def boom(*_a, **_k):
            raise ConnectorRuntimeError("planted", "planted failure", status_code=409)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=boom)
            with pytest.raises(HTTPException) as exc:
                update_mcp_server(
                    server.id,
                    MCPServerUpdate(name="irrelevant"),
                    current_user=member,
                    db=db,
                )

        assert exc.value.status_code == 409
        assert exc.value.detail == "planted failure"


class TestAnAlreadyAssociatedCallersEditSurvivesAHookOutage:
    """A caller who already holds a personal association row was admitted
    onto this route by that row, not by the team verdict -- an owner and a
    platform administrator each decide the edit branch before a verdict is
    ever read, and a non-owner member writing only their own association
    fields writes nothing the verdict governs. For all of them a hook
    failure degrades the reported edit right rather than answering the
    whole request with the hook's own outage; see
    ``TestTypedErrorArm`` above for the population the verdict genuinely
    gates, which still fails closed.
    """

    def test_an_admin_with_a_non_owner_row_still_writes_when_the_hook_fails(self, db):
        owner = _make_user(db, 1)
        admin = _make_user(db, 2, is_admin=True)
        server = _make_owned_server(db, owner.id, name="admin-writes-through-outage")
        server_id = server.id
        db.add(
            UserMCPServer(
                user_id=admin.id,
                mcpserver_id=server_id,
                is_owner=False,
                is_active=True,
            )
        )
        db.commit()

        def boom(*_a, **_k):
            raise ConnectorRuntimeError("planted", "planted failure", status_code=503)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=boom)
            response = update_mcp_server(
                server_id,
                MCPServerUpdate(description="admin edits the shared row"),
                current_user=admin,
                db=db,
            )

        assert response.description == "admin edits the shared row"

        db.rollback()
        refreshed = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert refreshed.description == "admin edits the shared row"

    def test_a_member_setting_only_their_own_user_env_still_writes_when_the_hook_fails(
        self, db
    ):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        member_id = member.id
        server = _make_owned_server(db, owner.id, name="member-writes-through-outage")
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

        def boom(*_a, **_k):
            raise ConnectorRuntimeError("planted", "planted failure", status_code=503)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=boom)
            response = update_mcp_server(
                server_id,
                MCPServerUpdate(user_env={"API_KEY": "widened-by-a-member"}),
                current_user=member,
                db=db,
            )

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
        assert stored.env is not None


class TestOwnerIsImmuneToAHookFailure:
    """An owner's row already decides the edit answer on its own -- the
    edit branch returns True on ``is_owner`` without ever consulting a
    verdict -- so ``GET``/``PUT`` never call the hook for an owner's row at
    all. A hook that would raise must therefore never surface: both routes
    return their normal success status, unaffected by whatever the hook
    would have done."""

    def test_get_and_put_succeed_for_an_owner_even_though_the_hook_would_raise(
        self, db
    ):
        owner = _make_user(db, 1)
        server = _make_owned_server(db, owner.id, name="owner-immune")
        server_id = server.id

        def boom(*_a, **_k):
            raise ValueError("hook exploded")

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=boom)
            get_response = get_mcp_server(server_id, current_user=owner, db=db)
            put_response = update_mcp_server(
                server_id,
                MCPServerUpdate(description="edited by the owner"),
                current_user=owner,
                db=db,
            )

        assert get_response.can_edit_global is True
        assert put_response.can_edit_global is True
        assert put_response.description == "edited by the owner"


def _make_catalog_app(db, app_id: str) -> None:
    db.add(
        PublicMCPApp(
            app_id=app_id,
            name=app_id,
            transport="stdio",
            launch_config={"command": "true", "args": []},
        )
    )
    db.commit()


def _make_catalog_app_with_display_name(
    db, app_id: str, display_name: str, *, transport: str = "stdio", launch_config=None
) -> None:
    """A catalog app written the way the real registry writes one: the
    display name is NOT the app_id. A test that seeds name == app_id would
    let a name-only implementation pass for the wrong reason.
    """
    db.add(
        PublicMCPApp(
            app_id=app_id,
            name=display_name,
            transport=transport,
            launch_config=launch_config or {"command": "true", "args": []},
        )
    )
    db.commit()


def _make_catalog_server_row(
    db,
    *,
    name: str,
    transport: str = "stdio",
    command: str | None = "true",
    args: list | None = None,
    url: str | None = None,
    auth: dict | None = None,
    env: dict | None = None,
) -> MCPServer:
    """A shared server row shaped the way a catalog provisioning helper
    would write it, constructed directly rather than through connect/OAuth
    so a test can pick exactly which catalog shape it needs (api_key,
    mcp_oauth, or a renamed builtin_oauth row)."""
    server = MCPServer(
        name=name,
        transport=transport,
        managed="external",
        command=command,
        args=args if args is not None else [],
        url=url,
        auth=auth,
        env=env,
    )
    db.add(server)
    db.flush()
    return server


class TestADenyingStandInIsRefusedRatherThanReportedSuccessful:
    """A stand-in (no personal association row) whose verdict denies edit
    has an empty writable field set on this route: the personal-field
    guard refuses user_env/is_active (there is no personal row to hold
    them), the tamper check refuses every shared field it can compare, and
    the fields it cannot compare (secrets) are silently emptied out of the
    payload rather than written. Every payload such a caller can send was
    therefore already a no-op before this guard existed -- a 200 for it
    reported success for a write that never happened. All three payload
    shapes below are the ones that used to slip past the tamper check
    specifically (an unset payload, a secret-only payload the tamper check
    deliberately does not compare, and a payload that resubmits the
    connector's current value) and confirm none of them can still commit
    anything even with the new guard in place.
    """

    def _stand_in(self, db):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        server = _make_owned_server(db, owner.id, name="denying-stand-in-target")
        server_id = server.id

        def _run(payload):
            # Captured as plain values, not read off ``server`` after the
            # call: ``server`` and the ``refreshed`` row below share the
            # same identity-mapped Python object in this session, so
            # comparing one against the other after the call is comparing
            # the object with itself and can never fail.
            original_name = str(server.name)
            original_description = (
                str(server.description) if server.description is not None else None
            )

            with snapshot_connector_team_hooks():
                set_connector_team_hooks(
                    access=lambda db, user_id, refs: {
                        ref: ConnectorAccess(team_owned=True, can_edit=False)
                        for ref in refs
                    }
                )
                with pytest.raises(HTTPException) as exc:
                    update_mcp_server(server_id, payload, current_user=member, db=db)
            assert exc.value.status_code == 403

            db.rollback()
            refreshed = db.query(MCPServer).filter(MCPServer.id == server_id).one()
            assert refreshed.name == original_name
            assert refreshed.description == original_description
            assert (
                db.query(UserMCPServer)
                .filter(UserMCPServer.user_id == member.id)
                .count()
                == 0
            )
            return server, refreshed

        return server_id, _run

    def test_an_empty_payload_is_refused(self, db):
        _server_id, run = self._stand_in(db)
        run(MCPServerUpdate())

    def test_a_secrets_only_payload_the_tamper_check_never_compares_is_refused(
        self, db
    ):
        _server_id, run = self._stand_in(db)
        run(MCPServerUpdate(config={"env": {"K": "v"}}))

    def test_resubmitting_the_current_value_is_refused(self, db):
        server_id, run = self._stand_in(db)
        server = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        server.description = "the connector's current description"
        db.commit()
        run(MCPServerUpdate(description=server.description))


class TestCatalogRowsAreNeverTeamEditable:
    """A team verdict that grants edit is downgraded to ``can_edit=False``
    whenever the row it names is some platform catalog app's shared row --
    across every kind of catalog row (api_key, mcp_oauth, a builtin_oauth
    row an administrator renamed) reached through the GET/PUT gate. A
    self-built connector that happens to squat a catalog id is deliberately
    NOT exempted from this: its creator keeps their own edit right in full
    (``is_owner`` decides that outright), but a teammate editing it on the
    owner's behalf is not.

    ``api-key-row-with-no-platform-key``: same shape as
    ``api-key-row-with-platform-key``, except this row carries no platform
    fallback key in ``env`` at all -- the one distinction that matters if
    the downgrade were (wrongly) gated on ``_catalog_server_has_platform_key``
    instead of catalog membership: that function reads False here, but the
    row is still the platform's, not this team's, to hand out edit rights on.

    ``renamed-builtin-oauth-row``: a builtin_oauth row an administrator
    renamed away from the catalog's display name still carries its
    ``app_id`` in ``auth`` -- the one shape ``_is_reserved_catalog_name``
    (name-only) would miss, which is why that function must not be the
    downgrade's predicate.
    """

    @pytest.mark.parametrize(
        "catalog_app, catalog_row, payload, unchanged_field",
        [
            (
                lambda db: _make_catalog_app_with_display_name(db, "stripe", "Stripe"),
                lambda db: _make_catalog_server_row(
                    db,
                    name="stripe",
                    transport="stdio",
                    command="python",
                    args=["-m", "xagent.web.tools.mcp.stripe"],
                ),
                MCPServerUpdate(config={"command": "evil", "args": []}),
                "command",
            ),
            (
                lambda db: _make_catalog_app_with_display_name(
                    db,
                    "notion",
                    "Notion",
                    transport="streamable_http",
                    launch_config={
                        "url": "https://mcp.notion.com/mcp",
                        "auth": {"type": "mcp_oauth"},
                    },
                ),
                lambda db: _make_catalog_server_row(
                    db,
                    name="notion",
                    transport="streamable_http",
                    command=None,
                    url="https://mcp.notion.com/mcp",
                    auth={"type": "mcp_oauth"},
                ),
                MCPServerUpdate(config={"url": "https://evil.example/mcp"}),
                "url",
            ),
            (
                lambda db: _make_catalog_app_with_display_name(
                    db,
                    "acme-books",
                    "Acme Books",
                    transport="stdio",
                    launch_config={
                        "command": "python",
                        "args": ["-m", "acme_books"],
                        "required_env": ["ACME_BOOKS_API_KEY"],
                    },
                ),
                lambda db: _make_catalog_server_row(
                    db,
                    name="acme-books",
                    transport="stdio",
                    command="python",
                    args=["-m", "acme_books"],
                    env=None,
                ),
                MCPServerUpdate(config={"command": "evil", "args": []}),
                None,
            ),
            (
                lambda db: _make_catalog_app_with_display_name(db, "gmail", "Gmail"),
                lambda db: _make_catalog_server_row(
                    db,
                    name="team-mail-renamed",
                    transport="oauth",
                    command=None,
                    auth={"app_id": "gmail"},
                ),
                MCPServerUpdate(description="edited by a teammate"),
                None,
            ),
        ],
        ids=[
            "api-key-row-with-platform-key",
            "mcp-oauth-row",
            "api-key-row-with-no-platform-key",
            "renamed-builtin-oauth-row",
        ],
    )
    def test_team_stand_in_cannot_rewrite_a_catalog_row(
        self, db, catalog_app, catalog_row, payload, unchanged_field
    ):
        catalog_app(db)
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        server = catalog_row(db)
        db.add(
            UserMCPServer(
                user_id=owner.id, mcpserver_id=server.id, is_owner=True, is_active=True
            )
        )
        db.commit()
        server_id = server.id
        original_value = getattr(server, unchanged_field) if unchanged_field else None
        hook = _sequenced_access_hook(ConnectorAccess(team_owned=True, can_edit=True))

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=hook)
            with pytest.raises(HTTPException) as exc:
                update_mcp_server(
                    server_id,
                    payload,
                    current_user=member,
                    db=db,
                )

        assert exc.value.status_code == 403
        assert "You do not have permission to edit this MCP server" in exc.value.detail
        # Exactly one hook call: the refusal comes from the downgrade
        # applied when the verdict is first resolved, before any personal
        # row exists to hold an edit right -- not from the post-lock
        # recheck catching it a step later (that would be two calls).
        assert len(hook.calls) == 1

        if unchanged_field is not None:
            db.rollback()
            refreshed = db.query(MCPServer).filter(MCPServer.id == server_id).one()
            assert getattr(refreshed, unchanged_field) == original_value

    def test_a_self_built_row_with_no_name_collision_is_still_team_editable(self, db):
        _make_catalog_app_with_display_name(db, "stripe", "Stripe")
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        server = _make_owned_server(db, owner.id, name="my-custom-tool")
        server_id = server.id
        hook = _sequenced_access_hook(ConnectorAccess(team_owned=True, can_edit=True))

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=hook)
            response = update_mcp_server(
                server_id,
                MCPServerUpdate(description="edited by the team"),
                current_user=member,
                db=db,
            )

        assert response.can_edit_global is True

        db.rollback()
        refreshed = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert refreshed.description == "edited by the team"

    def test_the_catalog_rows_owner_can_still_edit_it_themselves(self, db):
        """A builtin_oauth connect writes ``is_owner=True`` on the
        connecting user's association -- unlike the key-based/mcp_oauth
        paths, which never do. The owner's edit right must not move: no
        verdict is even consulted for it, so the hook installed here must
        never be called at all.

        Uses the same stdio/api_key catalog shape as the tests above rather
        than an actual oauth-transport row: ``update_mcp_server`` rebuilds
        and revalidates the transport-specific config on every call
        (including a description-only one), and ``MCPServerConfig`` does
        not accept ``transport="oauth"`` at all -- a pre-existing
        limitation of this route, unrelated to catalog membership. What
        this test pins is the ownership bypass itself, which does not
        depend on which catalog shape carries it.
        """
        _make_catalog_app_with_display_name(db, "stripe", "Stripe")
        owner = _make_user(db, 1)
        server = _make_catalog_server_row(
            db,
            name="stripe",
            transport="stdio",
            command="python",
            args=["-m", "xagent.web.tools.mcp.stripe"],
        )
        db.add(
            UserMCPServer(
                user_id=owner.id, mcpserver_id=server.id, is_owner=True, is_active=True
            )
        )
        db.commit()
        server_id = server.id

        def hook_must_not_be_called(*_a, **_k):
            raise AssertionError("the access hook must not be called for an owner")

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=hook_must_not_be_called)
            response = update_mcp_server(
                server_id,
                MCPServerUpdate(description="edited by its owner"),
                current_user=owner,
                db=db,
            )

        assert response.can_edit_global is True

    def test_get_on_a_catalog_row_still_reaches_it_but_reports_no_edit_right(self, db):
        _make_catalog_app_with_display_name(db, "stripe", "Stripe")
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        server = _make_catalog_server_row(
            db, name="stripe", transport="stdio", command="python"
        )
        db.add(
            UserMCPServer(
                user_id=owner.id, mcpserver_id=server.id, is_owner=True, is_active=True
            )
        )
        db.commit()
        server_id = server.id

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
                }
            )
            response = get_mcp_server(server_id, current_user=member, db=db)

        # What this pins is "reachable and readable, with no edit right":
        # reaching this assertion at all means no 404 was raised. It does not
        # distinguish clearing can_edit from dropping the verdict entirely --
        # past the 404 test above, those two are indistinguishable here.
        assert response.can_edit_global is False

    def test_a_self_built_row_that_squats_a_catalog_id_is_not_team_editable_but_its_owner_still_edits_it(
        self, db
    ):
        _make_catalog_app_with_display_name(db, "widget-sync", "Widget Sync")
        creator = _make_user(db, 1)
        teammate = _make_user(db, 2)
        # Built directly, the way this test file builds every row -- not
        # through connect/create, which would refuse this name outright
        # (_is_reserved_catalog_name). This is the row create/rename block
        # today, arriving here as if it predated the catalog app, or as if
        # the reserved-name gate had a bug; the point of this test is what
        # happens to a row in this shape once it exists, not how one could
        # come to exist.
        server = _make_catalog_server_row(
            db,
            name="widget-sync",
            transport="stdio",
            command="a-command-the-creator-chose",
        )
        db.add(
            UserMCPServer(
                user_id=creator.id,
                mcpserver_id=server.id,
                is_owner=True,
                is_active=True,
            )
        )
        db.commit()
        server_id = server.id
        hook = _sequenced_access_hook(ConnectorAccess(team_owned=True, can_edit=True))

        # (a) A teammate editing it on the owner's behalf is refused --
        # the catalog claims this name, and the row's own creation history
        # is not something this schema records today.
        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=hook)
            with pytest.raises(HTTPException) as exc:
                update_mcp_server(
                    server_id,
                    MCPServerUpdate(description="edited by a teammate"),
                    current_user=teammate,
                    db=db,
                )
        assert exc.value.status_code == 403

        # (b) Its own creator is unaffected -- is_owner decides the edit
        # branch outright, before any verdict (downgraded or not) is read.
        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=hook)
            response = update_mcp_server(
                server_id,
                MCPServerUpdate(description="edited by its creator"),
                current_user=creator,
                db=db,
            )
        assert response.can_edit_global is True


class TestOwnershipWithholdsTheTeamEditRightFromAnOwnerlessRow:
    """A team verdict that grants edit is also downgraded when the
    definition row it names has no ``is_owner=True`` association at all --
    independent of, and in addition to, the catalog-key test above. The
    catalog-key test alone cannot see a non-``oauth`` row an administrator
    renamed away from its key, because for that row the key IS the current
    name and this same route can change it. The ownership test does not
    depend on any field this route can write, so it still catches such a
    row after the rename.
    """

    @pytest.mark.parametrize(
        "catalog_app, catalog_row, tamper_payload, unchanged_field",
        [
            (
                lambda db: _make_catalog_app_with_display_name(
                    db, "billing-api", "Billing API"
                ),
                lambda db: _make_catalog_server_row(
                    db,
                    name="billing-api",
                    transport="stdio",
                    command="python",
                    args=["-m", "xagent.web.tools.mcp.billing_api"],
                ),
                MCPServerUpdate(config={"command": "evil", "args": []}),
                "command",
            ),
            (
                lambda db: _make_catalog_app_with_display_name(
                    db, "browser-tool", "Browser Tool"
                ),
                lambda db: _make_catalog_server_row(
                    db,
                    name="browser-tool",
                    transport="stdio",
                    command="npx",
                    args=["-y", "@browser/tool"],
                ),
                MCPServerUpdate(config={"command": "evil", "args": []}),
                "command",
            ),
            (
                lambda db: _make_catalog_app_with_display_name(
                    db,
                    "docs-oauth",
                    "Docs OAuth",
                    transport="streamable_http",
                    launch_config={
                        "url": "https://mcp.docs.example/mcp",
                        "auth": {"type": "mcp_oauth"},
                    },
                ),
                lambda db: _make_catalog_server_row(
                    db,
                    name="docs-oauth",
                    transport="streamable_http",
                    command=None,
                    url="https://mcp.docs.example/mcp",
                    auth={"type": "mcp_oauth"},
                ),
                MCPServerUpdate(config={"url": "https://evil.example/mcp"}),
                "url",
            ),
        ],
        ids=["api_key", "keyless", "mcp_oauth"],
    )
    def test_a_renamed_catalog_row_with_no_owner_is_not_team_editable(
        self, db, catalog_app, catalog_row, tamper_payload, unchanged_field
    ):
        catalog_app(db)
        admin = _make_user(db, 1, is_admin=True)
        member = _make_user(db, 2)
        server = catalog_row(db)
        server_id = server.id
        # Production shape: the administrator who connected this catalog row
        # holds an ordinary non-owner association -- connect never marks one
        # is_owner=True, so this row has an association but no owner.
        db.add(
            UserMCPServer(
                user_id=admin.id,
                mcpserver_id=server_id,
                is_owner=False,
                is_active=True,
            )
        )
        db.commit()

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
                }
            )
            update_mcp_server(
                server_id,
                MCPServerUpdate(name="renamed-away-from-the-catalog-key"),
                current_user=admin,
                db=db,
            )

            original_value = getattr(
                db.query(MCPServer).filter(MCPServer.id == server_id).one(),
                unchanged_field,
            )
            with pytest.raises(HTTPException) as exc:
                update_mcp_server(server_id, tamper_payload, current_user=member, db=db)

        assert exc.value.status_code == 403
        assert "You do not have permission to edit this MCP server" in exc.value.detail

        db.rollback()
        refreshed = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert getattr(refreshed, unchanged_field) == original_value

    def test_a_row_whose_owner_is_gone_is_not_team_editable(self, db):
        """A definition row with no owner and no catalog-key collision
        either -- the shape left behind once a connector's creator account
        has been deleted, since association rows cascade with the user.
        This is the conservative direction stated in the docstring: a
        wrong answer here refuses the edit rather than granting one."""
        member = _make_user(db, 2)
        server = MCPServer(
            name="orphaned-connector",
            transport="stdio",
            managed="external",
            command="true",
        )
        db.add(server)
        db.commit()
        server_id = server.id

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
                }
            )
            with pytest.raises(HTTPException) as exc:
                update_mcp_server(
                    server_id,
                    MCPServerUpdate(description="should not land"),
                    current_user=member,
                    db=db,
                )

        assert exc.value.status_code == 403
        assert "You do not have permission to edit this MCP server" in exc.value.detail

    def test_an_owned_row_still_gets_the_team_edit_right(self, db):
        """Reverse anchor, guarding against an over-broad fix: a row that
        does have an owner -- not a catalog row -- must keep its team edit
        right. Written so a change that downgrades every row
        unconditionally, not only ownerless ones, cannot pass by refusing
        everything."""
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        server = _make_owned_server(db, owner.id, name="owned-row-still-editable")
        server_id = server.id

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
                }
            )
            response = update_mcp_server(
                server_id,
                MCPServerUpdate(description="edited by the team"),
                current_user=member,
                db=db,
            )

        assert response.can_edit_global is True
        db.rollback()
        refreshed = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert refreshed.description == "edited by the team"


_SEAM_MODULE = "xagent.web.api.mcp"

# The arms that answer a failed access verdict with a warning instead of
# re-raising it. Pinned as a count so the enumeration below cannot pass by
# finding nothing, and so a second arm has to come here before it can skip the
# invariant.
_DEGRADING_CONNECTOR_RUNTIME_HANDLERS = 1


def _connector_runtime_handlers_that_log() -> list[ast.ExceptHandler]:
    """Every ``except ConnectorRuntimeError`` arm in this module that answers
    the failure with a warning rather than re-raising it."""
    module = importlib.import_module(_SEAM_MODULE)
    tree = ast.parse(inspect.getsource(module))
    handlers = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if not (
            isinstance(node.type, ast.Name) and node.type.id == "ConnectorRuntimeError"
        ):
            continue
        if any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "warning"
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == "logger"
            for child in ast.walk(node)
        ):
            handlers.append(node)
    return handlers


def test_the_degrading_handler_enumeration_is_not_vacuous():
    """Pins the enumeration itself, so the assertion below cannot pass by
    finding nothing."""
    assert (
        len(_connector_runtime_handlers_that_log())
        == _DEGRADING_CONNECTOR_RUNTIME_HANDLERS
    )


def test_every_degrading_handler_logs_the_failure_it_degraded_on():
    """An arm that answers a failed verdict with a warning leaves the response
    at 200, so that warning is the only record of why the caller lost a
    reported edit right. It has to name the failure, which means formatting
    the caught ``ConnectorRuntimeError`` -- whose ``str`` is
    ``"<code>: <safe message>"`` -- into the line.

    Pinned in the source rather than only per route: the behavioural pin in
    ``test_mcp_reported_edit_permission.py`` can exercise one arm per test,
    and an arm added later would inherit neither that pin nor this reasoning.
    """
    offenders = []
    for handler in _connector_runtime_handlers_that_log():
        for call in [
            child
            for child in ast.walk(handler)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "warning"
        ]:
            carries_the_exception = handler.name is not None and any(
                isinstance(arg, ast.Name) and arg.id == handler.name
                for arg in call.args
            )
            if not carries_the_exception:
                offenders.append(f"line {call.lineno}")
    assert offenders == [], (
        "these degrade arms log a warning that never formats the caught "
        f"ConnectorRuntimeError, so the failure has no identity: {offenders}"
    )
