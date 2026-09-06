"""The edit right on a team-linked MCP connector: ``GET``/``PUT
/api/mcp/servers/{server_id}`` resolve a caller with no personal row
through the connector access hook instead of 404ing outright, the edit
branch of ``_check_mcp_permission`` falls back to that verdict, the two
per-user fields reject outright for a caller with no row to hold them, a
raising hook surfaces as its declared status rather than a 500, a
stand-in whose verdict denies edit is refused outright rather than
reported as an empty success, and the verdict is re-resolved once more
after the definition row's lock is taken, refusing the write if it no
longer grants what the pre-lock answer granted.

Every test installs the access hook through
``snapshot_connector_team_hooks`` so no hook state leaks between tests or
into suites that run after this one.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from xagent.web.api import mcp as mcp_module
from xagent.web.api.mcp import (
    MCPAppConnectRequest,
    MCPServerUpdate,
    _check_mcp_permission,
    connect_mcp_app,
    get_mcp_server,
    get_mcp_servers,
    toggle_mcp_server,
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


class TestDecorationDegradesAfterTheWriteCommits:
    """``toggle`` and ``connect`` both commit their write before resolving
    the verdict, purely to decorate the response's ``can_edit_global`` --
    a hook failure there must degrade that field to False rather than fail
    a request whose write already landed."""

    def test_toggle_degrades_and_keeps_its_effect_when_the_hook_raises(
        self, db, monkeypatch
    ):
        # A non-owner personal row, not the owner's: an owner's
        # can_edit_global cannot be moved by any verdict at all (is_owner
        # wins outright), so only a non-owner's reported field actually
        # depends on whether the verdict resolved or degraded.
        owner = _make_user(db, 1)
        editor = _make_user(db, 2)
        server = _make_owned_server(db, owner.id)
        server_id = server.id
        db.add(
            UserMCPServer(
                user_id=editor.id,
                mcpserver_id=server_id,
                is_owner=False,
                is_active=True,
            )
        )
        db.commit()
        before = (
            db.query(UserMCPServer)
            .filter(
                UserMCPServer.user_id == editor.id,
                UserMCPServer.mcpserver_id == server_id,
            )
            .one()
            .is_active
        )

        def boom(*_a, **_k):
            raise ValueError("hook exploded")

        fake_logger = MagicMock()
        monkeypatch.setattr(mcp_module, "logger", fake_logger)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=boom)
            response = toggle_mcp_server(server_id, current_user=editor, db=db)

        assert response.can_edit_global is False
        fake_logger.warning.assert_called_once()

        db.rollback()
        refreshed = (
            db.query(UserMCPServer)
            .filter(
                UserMCPServer.user_id == editor.id,
                UserMCPServer.mcpserver_id == server_id,
            )
            .one()
        )
        assert refreshed.is_active is (not before)

    def test_connect_degrades_and_keeps_its_effect_when_the_hook_raises(
        self, db, monkeypatch
    ):
        user = _make_user(db, 1)
        _make_catalog_app(db, "decorate-only-app")

        def boom(*_a, **_k):
            raise ValueError("hook exploded")

        fake_logger = MagicMock()
        monkeypatch.setattr(mcp_module, "logger", fake_logger)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=boom)
            response = connect_mcp_app(
                "decorate-only-app",
                MCPAppConnectRequest(),
                current_user=user,
                db=db,
            )

        assert response.can_edit_global is False
        fake_logger.warning.assert_called_once()

        db.rollback()
        server = db.query(MCPServer).filter(MCPServer.name == "decorate-only-app").one()
        assoc = (
            db.query(UserMCPServer)
            .filter(
                UserMCPServer.user_id == user.id,
                UserMCPServer.mcpserver_id == server.id,
            )
            .one()
        )
        assert assoc.is_owner is False


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


class TestTheVerdictIsRevalidatedUnderTheDefinitionLock:
    """The verdict that granted a stand-in edit access is resolved before
    this route's own row lock exists. The installing application can
    revoke the team's link to this connector at any moment in between --
    it writes its own tables, which this lock does not cover -- so the
    route re-resolves the verdict once more after taking the lock, and
    refuses (with zero side effects) if the answer no longer grants edit.
    This narrows the window between resolving the verdict and committing
    the write; it does not close it, since the caller's own definition-row
    lock has nothing to say about a revoke the installing application makes
    through its own tables.
    """

    def _run(self, db, *, hook):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        server = _make_owned_server(db, owner.id, name="revalidated-under-lock")
        server_id = server.id
        # Captured as plain values before the call, not read off ``server``
        # afterwards: ``server`` and the requery below share the same
        # identity-mapped Python object in this session, so comparing one
        # against the other after the call would be comparing the object
        # with itself and could never fail.
        original_name = str(server.name)
        original_description = (
            str(server.description) if server.description is not None else None
        )

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=hook)
            result = {}
            try:
                result["response"] = update_mcp_server(
                    server_id,
                    MCPServerUpdate(description="edited-while-in-flight"),
                    current_user=member,
                    db=db,
                )
            except HTTPException as exc:
                result["error"] = exc
            return server, server_id, result, original_name, original_description

    def test_revoked_between_resolution_and_lock_is_refused(self, db):
        hook = _sequenced_access_hook(
            ConnectorAccess(team_owned=True, can_edit=True), None
        )
        _server, server_id, result, original_name, original_description = self._run(
            db, hook=hook
        )

        assert result["error"].status_code == 403
        db.rollback()
        refreshed = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert refreshed.name == original_name
        assert refreshed.description == original_description
        assert db.query(UserMCPServer).filter(UserMCPServer.user_id == 2).count() == 0

    def test_downgraded_to_not_editable_between_resolution_and_lock_is_refused(
        self, db
    ):
        hook = _sequenced_access_hook(
            ConnectorAccess(team_owned=True, can_edit=True),
            ConnectorAccess(team_owned=True, can_edit=False),
        )
        _server, server_id, result, original_name, original_description = self._run(
            db, hook=hook
        )

        assert result["error"].status_code == 403
        db.rollback()
        refreshed = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert refreshed.name == original_name
        assert refreshed.description == original_description
        assert db.query(UserMCPServer).filter(UserMCPServer.user_id == 2).count() == 0

    def test_still_granted_on_recheck_commits_durably(self, db):
        hook = _sequenced_access_hook(
            ConnectorAccess(team_owned=True, can_edit=True),
            ConnectorAccess(team_owned=True, can_edit=True),
        )
        _server, server_id, result, _original_name, _original_description = self._run(
            db, hook=hook
        )

        assert "error" not in result
        assert result["response"].description == "edited-while-in-flight"

        # Durability, not staging.
        db.rollback()
        refreshed = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert refreshed.description == "edited-while-in-flight"

    def test_a_verdict_that_changed_under_the_lock_is_the_one_reported(self, db):
        """The pre-lock answer granted edit; the post-lock answer still
        grants it but is a different object. The 200's can_edit_global
        must come from the answer the write was authorized on, not from
        the one resolved before the lock existed.

        This test alone cannot distinguish "reports the recheck" from
        "reports the pre-lock answer": both objects grant edit, so either
        one reported here yields the same True. What it pins is that the
        recheck running does not accidentally break the response -- for
        example by reassigning team_access in the refusal branch, or by
        setting it to None.
        """
        hook = _sequenced_access_hook(
            ConnectorAccess(team_owned=True, can_edit=True),
            ConnectorAccess(team_owned=True, can_edit=True),
        )
        _server, _server_id, result, _name, _description = self._run(db, hook=hook)

        assert "error" not in result
        assert result["response"].can_edit_global is True

    def test_recheck_that_raises_surfaces_the_hooks_own_status_with_zero_side_effects(
        self, db
    ):
        hook = _sequenced_access_hook(
            ConnectorAccess(team_owned=True, can_edit=True),
            ValueError("hook exploded during recheck"),
        )
        _server, server_id, result, original_name, original_description = self._run(
            db, hook=hook
        )

        assert result["error"].status_code == 503
        db.rollback()
        refreshed = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert refreshed.name == original_name
        assert refreshed.description == original_description
        assert db.query(UserMCPServer).filter(UserMCPServer.user_id == 2).count() == 0

    def test_a_recheck_hook_that_ends_the_transaction_is_refused_not_trusted(self, db):
        """The recheck runs under the definition-row lock, so it declares
        ``caller_holds_lock=True`` and is checked by the seam's session
        boundary guard (see ``connector_team_scope``'s module docstring).
        A hook that rolls back this session and still answers normally --
        rather than raising -- must not be trusted just because it returned
        a well-formed answer: the route's row lock is already gone by the
        time it does.

        The route has no dedicated handler for the seam's own
        ``ConnectorHookSessionBoundaryError``, so it falls through to this
        route's generic ``except Exception`` and comes back as that
        handler's own 500, not the application-level boundary handler's
        message -- a known, accepted divergence from the other five
        call sites."""
        hook_calls: list[object] = []

        def hook(hook_db, user_id, refs):
            del user_id
            hook_calls.append(refs)
            if len(hook_calls) == 1:
                return {
                    ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
                }
            hook_db.rollback()
            return {
                ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
            }

        member = _make_user(db, 2)
        server = MCPServer(
            name="recheck-boundary-target",
            transport="stdio",
            managed="external",
            command="true",
        )
        db.add(server)
        db.commit()
        server_id = server.id

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=hook)
            with pytest.raises(HTTPException) as exc:
                update_mcp_server(
                    server_id,
                    MCPServerUpdate(description="edited-under-boundary-violation"),
                    current_user=member,
                    db=db,
                )

        assert exc.value.status_code == 500
        assert exc.value.detail == (
            "Failed to update MCP server: An installed connector hook ended "
            "the caller's database transaction"
        )
        assert len(hook_calls) == 2
        db.rollback()
        refreshed = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert refreshed.description is None


class TestTheRecheckCostsExactlyOneExtraHookCall:
    """Which populations pay the recheck's extra hook round trip, and which
    do not, spelled out as call counts, for a payload that writes the shared
    definition row:

    - the owner: 0 calls, the owner's own ``is_owner`` decides the edit
      branch and ``still_can_edit`` is already True;
    - a stand-in sharing the write with the definition-row payload: 2 calls,
      one at the gate and one for the recheck;
    - a stand-in who is a platform admin: 1 call, the gate's -- the recheck
      is skipped because ``still_can_edit`` is already True on
      ``is_admin_now``;
    - a stand-in with an empty payload: 1 call, the gate's -- the recheck
      still runs, but there is no shared write left to refuse;
    - a personal row that does not own the server, whose verdict denies
      edit, on a payload that does carry a definition-row field: 2 calls.
      The recheck runs here even though the pre-lock verdict already
      denied edit, because a personal row that gained ``is_owner`` during
      the wait would otherwise be denied on a stale answer.
    """

    def test_a_granting_stand_in_editing_the_shared_config_pays_two_calls(self, db):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        server = _make_owned_server(db, owner.id, name="cost-stand-in-shared")
        server_id = server.id
        hook = _sequenced_access_hook(ConnectorAccess(team_owned=True, can_edit=True))

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=hook)
            update_mcp_server(
                server_id,
                MCPServerUpdate(description="shared-edit"),
                current_user=member,
                db=db,
            )

        assert len(hook.calls) == 2

    def test_a_granting_stand_in_with_an_empty_payload_pays_one_call(self, db):
        """An empty payload's ``model_fields_set`` is the empty set, which
        is a subset of ``{"user_env", "is_active"}`` -- the personal-only
        exemption, not the earlier personal-field 400 guard (that guard
        only fires when ``user_env``/``is_active`` is actually present).
        This is the payload shape that actually reaches the recheck's own
        condition and exercises the exemption, unlike an is_active-only
        payload, which never gets there at all for a stand-in (it 400s
        first)."""
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        server = _make_owned_server(db, owner.id, name="cost-stand-in-personal-only")
        server_id = server.id
        hook = _sequenced_access_hook(ConnectorAccess(team_owned=True, can_edit=True))

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=hook)
            update_mcp_server(server_id, MCPServerUpdate(), current_user=member, db=db)

        assert len(hook.calls) == 1

    def test_a_granting_stand_in_who_is_a_platform_admin_pays_one_call(self, db):
        """A platform admin's write authority never comes from the verdict
        in the first place: ``_check_mcp_permission`` answers True on
        ``is_admin`` before it ever reads one. ``still_can_edit``'s
        ``is_admin_now`` clause exists to skip the recheck for exactly this
        population -- dropping that clause must turn this red (2 calls
        instead of 1)."""
        owner = _make_user(db, 1)
        admin = _make_user(db, 2, is_admin=True)
        server = _make_owned_server(db, owner.id, name="cost-stand-in-admin")
        server_id = server.id
        hook = _sequenced_access_hook(ConnectorAccess(team_owned=True, can_edit=True))

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=hook)
            update_mcp_server(
                server_id,
                MCPServerUpdate(description="admin-edit"),
                current_user=admin,
                db=db,
            )

        assert len(hook.calls) == 1

    def test_an_owner_pays_zero_calls(self, db):
        owner = _make_user(db, 1)
        server = _make_owned_server(db, owner.id, name="cost-owner")
        server_id = server.id
        hook = _sequenced_access_hook(ConnectorAccess(team_owned=True, can_edit=True))

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=hook)
            update_mcp_server(
                server_id,
                MCPServerUpdate(description="owner-edit"),
                current_user=owner,
                db=db,
            )

        assert len(hook.calls) == 0

    def test_a_denying_verdict_on_a_personal_row_pays_one_call(self, db):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        server = _make_owned_server(db, owner.id, name="cost-personal-denied")
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
        hook = _sequenced_access_hook(ConnectorAccess(team_owned=True, can_edit=False))

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=hook)
            update_mcp_server(
                server_id,
                MCPServerUpdate(is_active=False),
                current_user=member,
                db=db,
            )

        assert len(hook.calls) == 1

    def test_a_denying_verdict_on_a_personal_row_pays_two_calls_on_a_shared_field(
        self, db
    ):
        """The counterpart of the test above, on a payload that carries a
        definition-row field instead of only ``is_active``. The pre-lock
        verdict already denies edit, but the recheck still runs regardless
        of what that verdict said, because a personal row that gains
        ``is_owner`` during the wait must still be re-asked. This caller's
        row does not change, so the second call changes nothing about the
        outcome -- it just pays for a recheck that a row which stayed
        exactly as denied still has to go through."""
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        server = _make_owned_server(db, owner.id, name="cost-personal-denied-shared")
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
        hook = _sequenced_access_hook(ConnectorAccess(team_owned=True, can_edit=False))

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=hook)
            with pytest.raises(HTTPException) as exc:
                update_mcp_server(
                    server_id,
                    MCPServerUpdate(description="shared-edit-denied"),
                    current_user=member,
                    db=db,
                )

        assert exc.value.status_code == 403
        assert len(hook.calls) == 2

    def test_a_member_with_a_personal_row_pays_one_call_on_a_real_personal_field(
        self, db
    ):
        """The personal-only exemption, exercised by a payload that
        actually carries a personal field and by a caller the payload can
        land on. The existing coverage is degenerate in two different
        ways: the empty-payload case (above) never carries a field at all,
        and the denying-verdict case (above) short-circuits one clause
        earlier, on ``team_access.can_edit``, so neither reaches the
        exemption with a real value."""
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        server = _make_owned_server(db, owner.id, name="both-rows-personal-only")
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
        hook = _sequenced_access_hook(ConnectorAccess(team_owned=True, can_edit=True))

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=hook)
            update_mcp_server(
                server_id,
                MCPServerUpdate(is_active=False),
                current_user=member,
                db=db,
            )

        assert len(hook.calls) == 1
        # The personal write the exemption exists to let through actually landed.
        db.rollback()
        refreshed = (
            db.query(UserMCPServer)
            .filter(
                UserMCPServer.user_id == member.id,
                UserMCPServer.mcpserver_id == server_id,
            )
            .one()
        )
        assert refreshed.is_active is False


class TestCatalogRowsAreNeverTeamEditable:
    """A team verdict that grants edit is downgraded to ``can_edit=False``
    whenever the row it names is some platform catalog app's shared row --
    across every kind of catalog row (api_key, mcp_oauth, a builtin_oauth
    row an administrator renamed) and every route that produces or reports
    a verdict (the GET/PUT gate, the list endpoint's two loops, connect,
    and toggle). A self-built connector that happens to squat a catalog id
    is deliberately NOT exempted from this: its creator keeps their own
    edit right in full (``is_owner`` decides that outright), but a
    teammate editing it on the owner's behalf is not.
    """

    def test_team_stand_in_cannot_rewrite_an_api_key_catalog_rows_command(self, db):
        _make_catalog_app_with_display_name(db, "stripe", "Stripe")
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
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
        original_command = server.command
        hook = _sequenced_access_hook(ConnectorAccess(team_owned=True, can_edit=True))

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=hook)
            with pytest.raises(HTTPException) as exc:
                update_mcp_server(
                    server_id,
                    MCPServerUpdate(config={"command": "evil", "args": []}),
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

        db.rollback()
        refreshed = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert refreshed.command == original_command

    def test_team_stand_in_cannot_rewrite_an_mcp_oauth_catalog_rows_url(self, db):
        _make_catalog_app_with_display_name(
            db,
            "notion",
            "Notion",
            transport="streamable_http",
            launch_config={
                "url": "https://mcp.notion.com/mcp",
                "auth": {"type": "mcp_oauth"},
            },
        )
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        server = _make_catalog_server_row(
            db,
            name="notion",
            transport="streamable_http",
            command=None,
            url="https://mcp.notion.com/mcp",
            auth={"type": "mcp_oauth"},
        )
        db.add(
            UserMCPServer(
                user_id=owner.id, mcpserver_id=server.id, is_owner=True, is_active=True
            )
        )
        db.commit()
        server_id = server.id
        original_url = server.url
        hook = _sequenced_access_hook(ConnectorAccess(team_owned=True, can_edit=True))

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=hook)
            with pytest.raises(HTTPException) as exc:
                update_mcp_server(
                    server_id,
                    MCPServerUpdate(config={"url": "https://evil.example/mcp"}),
                    current_user=member,
                    db=db,
                )

        assert exc.value.status_code == 403
        assert "You do not have permission to edit this MCP server" in exc.value.detail
        assert len(hook.calls) == 1

        db.rollback()
        refreshed = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        assert refreshed.url == original_url

    def test_team_stand_in_cannot_rewrite_an_api_key_catalog_row_with_no_platform_key(
        self, db
    ):
        """Same shape as the ``stripe`` case above, except this row carries
        no platform fallback key in ``env`` at all -- the one distinction
        that matters if the downgrade were (wrongly) gated on
        ``_catalog_server_has_platform_key`` instead of catalog membership:
        that function reads False here, but the row is still the
        platform's, not this team's, to hand out edit rights on."""
        _make_catalog_app_with_display_name(
            db,
            "acme-books",
            "Acme Books",
            transport="stdio",
            launch_config={
                "command": "python",
                "args": ["-m", "acme_books"],
                "required_env": ["ACME_BOOKS_API_KEY"],
            },
        )
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        server = _make_catalog_server_row(
            db,
            name="acme-books",
            transport="stdio",
            command="python",
            args=["-m", "acme_books"],
            env=None,
        )
        db.add(
            UserMCPServer(
                user_id=owner.id, mcpserver_id=server.id, is_owner=True, is_active=True
            )
        )
        db.commit()
        server_id = server.id
        hook = _sequenced_access_hook(ConnectorAccess(team_owned=True, can_edit=True))

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=hook)
            with pytest.raises(HTTPException) as exc:
                update_mcp_server(
                    server_id,
                    MCPServerUpdate(config={"command": "evil", "args": []}),
                    current_user=member,
                    db=db,
                )

        assert exc.value.status_code == 403

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

    def test_connecting_a_catalog_app_reports_no_edit_right_even_when_granted(self, db):
        _make_catalog_app_with_display_name(db, "stripe", "Stripe")
        user = _make_user(db, 1)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
                }
            )
            response = connect_mcp_app(
                "stripe",
                MCPAppConnectRequest(),
                current_user=user,
                db=db,
            )

        assert response.can_edit_global is False

    def test_the_list_endpoints_stand_in_row_reports_no_edit_right(self, db):
        _make_catalog_app_with_display_name(
            db,
            "notion",
            "Notion",
            transport="streamable_http",
            launch_config={
                "url": "https://mcp.notion.com/mcp",
                "auth": {"type": "mcp_oauth"},
            },
        )
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        server = _make_catalog_server_row(
            db,
            name="notion",
            transport="streamable_http",
            command=None,
            url="https://mcp.notion.com/mcp",
            auth={"type": "mcp_oauth"},
        )
        db.add(
            UserMCPServer(
                user_id=owner.id, mcpserver_id=server.id, is_owner=True, is_active=True
            )
        )
        db.commit()
        server_id = server.id

        def visibility_hook(_db, _user_id):
            return {"mcp": {server_id}, "custom_api": set()}

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
                },
                visibility=visibility_hook,
            )
            responses = get_mcp_servers(current_user=member, db=db)

        matches = [r for r in responses if r.id == server_id]
        assert len(matches) == 1
        assert matches[0].can_edit_global is False

    def test_the_list_endpoints_personal_row_on_a_catalog_server_reports_no_edit_right(
        self, db
    ):
        """Same downgrade as the stand-in case above, but for the other of
        the list endpoint's two append loops: a caller who has their own
        (non-owner) personal row on a catalog server, rather than no
        personal row at all. Both loops call ``_team_access_for_shared_row``
        independently, so each needs its own test pinning it.
        """
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

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
                }
            )
            responses = get_mcp_servers(current_user=member, db=db)

        matches = [r for r in responses if r.id == server_id]
        assert len(matches) == 1
        assert matches[0].can_edit_global is False

    def test_toggle_on_a_catalog_row_reports_no_edit_right(self, db):
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

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
                }
            )
            response = toggle_mcp_server(server_id, current_user=member, db=db)

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

    def test_team_stand_in_cannot_rewrite_a_renamed_builtin_oauth_catalog_row(self, db):
        """A builtin_oauth row an administrator renamed away from the
        catalog's display name still carries its ``app_id`` in ``auth`` --
        the one shape ``_is_reserved_catalog_name`` (name-only) would miss,
        which is why that function must not be the downgrade's predicate.
        """
        _make_catalog_app_with_display_name(db, "gmail", "Gmail")
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        server = _make_catalog_server_row(
            db,
            name="team-mail-renamed",
            transport="oauth",
            command=None,
            auth={"app_id": "gmail"},
        )
        db.add(
            UserMCPServer(
                user_id=owner.id, mcpserver_id=server.id, is_owner=True, is_active=True
            )
        )
        db.commit()
        server_id = server.id
        hook = _sequenced_access_hook(ConnectorAccess(team_owned=True, can_edit=True))

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=hook)
            with pytest.raises(HTTPException) as exc:
                update_mcp_server(
                    server_id,
                    MCPServerUpdate(description="edited by a teammate"),
                    current_user=member,
                    db=db,
                )

        assert exc.value.status_code == 403
        assert "You do not have permission to edit this MCP server" in exc.value.detail
        assert len(hook.calls) == 1


class TestCatalogCheckQueryBudget:
    """The per-request cost of the catalog downgrade: a deployment with no
    granting verdict in a listing response pays nothing extra at all, and
    one that does pays exactly one additional statement -- a single
    catalog-keys SELECT shared across every row in the response -- not one
    per row. Pinned across two population sizes so that a per-row cost
    shows up as a difference between them rather than as an absolute
    number nobody can read.
    """

    def _list_query_count(self, db, *, num_rows: int, grant_edit: bool) -> int:
        suffix = f"{grant_edit}-{num_rows}"
        owner = _make_user(db, 2000 + num_rows * 10 + (1 if grant_edit else 0))
        caller = _make_user(db, 2050 + num_rows * 10 + (1 if grant_edit else 0))
        stand_in = [
            _make_owned_server(db, owner.id, name=f"budget-{suffix}-{i}")
            for i in range(num_rows)
        ]
        # Read before the query listener attaches: these ids were expired
        # by their own setup commits, and reading them for the first time
        # inside the measured window would count as a query this test's
        # setup causes, not one the endpoint itself issues.
        _ = caller.id
        stand_in_ids = {s.id for s in stand_in}

        def visibility_hook(_db, _user_id):
            return {"mcp": set(stand_in_ids), "custom_api": set()}

        def access_hook(hook_db, user_id, refs):
            del hook_db, user_id
            return {
                ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
            }

        queries: list[str] = []

        def record_query(conn, cursor, statement, parameters, context, executemany):
            del conn, cursor, parameters, context, executemany
            queries.append(statement)

        engine = db.get_bind()
        event.listen(engine, "before_cursor_execute", record_query)
        try:
            with snapshot_connector_team_hooks():
                set_connector_team_hooks(
                    visibility=visibility_hook,
                    access=access_hook if grant_edit else None,
                )
                get_mcp_servers(current_user=caller, db=db)
        finally:
            event.remove(engine, "before_cursor_execute", record_query)
        return len(queries)

    def test_a_deployment_with_no_access_hook_pays_nothing_regardless_of_row_count(
        self, db
    ):
        counts = {
            n: self._list_query_count(db, num_rows=n, grant_edit=False) for n in (2, 6)
        }
        assert counts[2] == counts[6], counts

    def test_a_granting_access_hook_costs_exactly_one_more_query_regardless_of_row_count(
        self, db
    ):
        without_hook = {
            n: self._list_query_count(db, num_rows=n, grant_edit=False) for n in (2, 6)
        }
        with_hook = {
            n: self._list_query_count(db, num_rows=n, grant_edit=True) for n in (2, 6)
        }
        assert with_hook[2] == with_hook[6], with_hook
        assert with_hook[2] == without_hook[2] + 1, (with_hook, without_hook)


_SEAM_MODULE = "xagent.web.api.mcp"

# Every top-level function in this module that can reach an installed
# connector team hook. Written out so the discovery below cannot pass by
# finding nothing.
_SEAM_REACHING_FUNCTIONS = {
    "_local_mcp_can_attach",
    "_resolve_mcp_server_for_request",
    # The coroutine that owns app-scoped teardown, ``teardown_mcp_app_server``,
    # is absent on purpose: it hands this helper to ``asyncio.to_thread``
    # instead of calling it, so the seam runs in a worker thread and the
    # coroutine reaches it on no thread of its own. The discovery below follows
    # plain-name calls, which is exactly the distinction that matters here --
    # turning that dispatch back into a direct call would put the seam back on
    # the event loop, and would also put the coroutine back in this set and in
    # the offender list.
    "_teardown_mcp_app_server_locally",
    "_recheck_team_access_under_definition_lock",
    "connect_mcp_app",
    "delete_mcp_server",
    "get_mcp_server",
    "get_mcp_servers",
    "list_mcp_apps",
    "toggle_mcp_server",
    "update_mcp_server",
}

# The one function that reaches the seam and is still a coroutine, with the
# fact that makes it impossible to convert. Its own await -- one that is not
# the seam call itself -- is asserted below, so this entry cannot be claimed
# by a route whose coroutine is only the seam's doing.
_COROUTINE_EXEMPTIONS = {"delete_mcp_server"}


def _functions_reaching_the_connector_seam() -> dict[str, ast.AST]:
    """Every top-level function in this module that can reach an installed
    connector team hook.

    Seeded on the functions that import ``connector_team_scope`` in their own
    body, which is how every call site in this module reaches the seam, then
    closed transitively over plain-name calls, because one route reaches it
    only through a helper (``get_mcp_server`` through
    ``_resolve_mcp_server_for_request``). A seed-only check would miss exactly
    the route this test exists for.
    """
    module = importlib.import_module(_SEAM_MODULE)
    tree = ast.parse(inspect.getsource(module))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    reaching = {
        name
        for name, node in functions.items()
        if any(
            isinstance(child, ast.ImportFrom)
            and child.module is not None
            and child.module.endswith("connector_team_scope")
            for child in ast.walk(node)
        )
    }
    changed = True
    while changed:
        changed = False
        for name, node in functions.items():
            if name in reaching:
                continue
            called = {
                child.func.id
                for child in ast.walk(node)
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
            }
            if called & reaching:
                reaching.add(name)
                changed = True
    return {name: functions[name] for name in reaching}


def _seam_names_imported_by(node: ast.AST) -> set[str]:
    """The names this function imports from ``connector_team_scope``.

    Read off the function's own body because that is how every call site in
    this module reaches the seam -- the same fact
    ``_functions_reaching_the_connector_seam`` above is seeded on.
    """
    return {
        alias.asname or alias.name
        for child in ast.walk(node)
        if isinstance(child, ast.ImportFrom)
        and child.module is not None
        and child.module.endswith("connector_team_scope")
        for alias in child.names
    }


def test_the_discovery_of_seam_reaching_functions_is_not_vacuous():
    """Pins the enumeration itself, so the assertion below cannot pass by
    finding nothing."""
    assert set(_functions_reaching_the_connector_seam()) == _SEAM_REACHING_FUNCTIONS


def test_no_function_that_reaches_the_connector_seam_is_a_coroutine():
    """An installed connector team hook may be slow -- the seam is designed on
    the assumption that the installing application answers from its own
    tables. FastAPI runs a coroutine route on the event loop thread itself, so
    a slow hook call inside an ``async def`` stalls every other request the
    process is serving, not just this one; a plain ``def`` goes to the
    threadpool instead, where a slow call occupies one worker.

    Enumerated by reachability rather than by a hand-written list of routes:
    an earlier fix for this same risk class swept siblings along the "takes a
    row lock" axis and therefore missed a route that calls a hook without
    taking one.
    """
    offenders = []
    for name, node in _functions_reaching_the_connector_seam().items():
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        if name in _COROUTINE_EXEMPTIONS:
            # An exemption is only legitimate for a function that genuinely
            # cannot be converted, so it must carry an await that is NOT the
            # seam call itself. A function whose only await IS the seam call
            # is a coroutine of the seam's own making -- convertible by making
            # that call synchronous -- and "contains some await" would still
            # wave it through.
            seam_names = _seam_names_imported_by(node)
            non_seam_awaits = [
                child
                for child in ast.walk(node)
                if isinstance(child, ast.Await)
                and not (
                    isinstance(child.value, ast.Call)
                    and isinstance(child.value.func, ast.Name)
                    and child.value.func.id in seam_names
                )
            ]
            assert non_seam_awaits, (
                f"{name} is exempted from this invariant, but every await it "
                "has is a seam call -- the coroutine is the seam's own doing, "
                "so make that call synchronous instead of exempting the route"
            )
            continue
        offenders.append(name)
    assert offenders == [], (
        "these functions can reach an installed connector team hook while "
        f"running on the event loop thread: {sorted(offenders)}"
    )


# The arms that answer a failed access verdict with a warning instead of
# re-raising it. Pinned as a count so the enumeration below cannot pass by
# finding nothing, and so a sixth arm has to come here before it can skip the
# invariant.
_DEGRADING_CONNECTOR_RUNTIME_HANDLERS = 5


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
