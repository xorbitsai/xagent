"""Real-PostgreSQL coverage for the connector access seam restoring a
shared session a hook left with a failed raw statement on it.

``poison_by_raw_statement`` below only actually poisons PostgreSQL: a
failed raw statement aborts the surrounding transaction there, so every
later statement on the same connection is refused until a rollback runs.
SQLite does not enforce that the same way, so a SQLite-backed suite cannot
prove this shape needs the fix -- hence this PostgreSQL-only file.

``test_the_seam_restores_the_session_after_a_raw_statement_failure`` is the
direct, independently mutation-sensitive proof: it calls
``resolve_connector_access_or_raise`` itself with a hook that runs the
poisoning statement, and asserts a fresh query on the same session succeeds
right after. Deleting the rollback call from
``_restore_session_after_hook_failure`` (services/connector_team_scope.py)
turns this test red on this file specifically; it stays green on SQLite
regardless, which is exactly why this shape needs its own PostgreSQL-only
proof.

The four route-level tests below (toggle, connect, the apps listing, the
servers listing) are run here for completeness -- they pin the *correct*
end-to-end behavior (2xx, durable writes) under this exact failure shape on
a real server. Whether each route *call itself* needs the session restored
is a separate question from whether its test does, and the two do not agree
for all four:

The route calls themselves are never independently mutation-sensitive for
this specific shape: each response builder happens to read the connector
row's attributes once *before* the hook ever runs (e.g.
``toggle_mcp_server``'s own log line touches ``server.name``), which loads
those attributes into the ORM instance. A failed raw statement aborts the
underlying transaction without SQLAlchemy's ORM-level "expire everything"
cleanup -- unlike a failed flush, which expires the identity map and so
forces a reload on the next attribute read -- so no attribute on that
already-loaded row needs reloading afterward, and none of the four routes
issues a new statement on the poisoned connection while building its own
response.

The toggle and connect tests are independently mutation-sensitive anyway,
because each queries the database again *after* the route call returns, to
verify what actually landed (``refreshed``/``assoc`` below) -- and that
query runs directly on the same session the hook just poisoned, with no
rollback of the test's own in between. Removing the production restore
turns that query into the first statement that reaches the aborted
transaction, which PostgreSQL refuses. The apps-listing and servers-listing
tests stay non-sensitive: neither issues any further statement after the
route call, so there is nothing left in either test that could reach the
poisoned connection. The seam-level test above is what directly exercises
the poisoned connection regardless of any particular route's shape.

Obtains its database through ``tests/shared/postgres_disposable.py``
(``disposable_database_factory``), the same disposable-CREATE-DATABASE
helper the other ``*_postgresql.py`` suites in this repo use.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.services.connector_team_scope import (
    resolve_connector_access_or_raise,
    set_connector_team_hooks,
    snapshot_connector_team_hooks,
)

pytestmark = pytest.mark.postgresql


def poison_by_raw_statement(db) -> None:
    db.execute(sa.text("select * from no_such_table_at_all"))


@pytest.fixture()
def session_factory():
    with disposable_database_factory("xagent_connector_session_fault") as make_database:
        engine = make_database("session_fault")
        Base.metadata.create_all(bind=engine)
        yield sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture()
def seeded(session_factory):
    """One owner, one owned MCP server (for toggle), one catalog app entry
    (for connect), in their own committed rows."""
    with session_factory() as db:
        owner = User(username="session-fault-owner", password_hash="x", is_admin=False)
        db.add(owner)
        db.flush()
        server = MCPServer(
            name="session-fault-target",
            transport="stdio",
            managed="external",
            command="true",
        )
        db.add(server)
        db.flush()
        db.add(
            UserMCPServer(
                user_id=int(owner.id),
                mcpserver_id=int(server.id),
                is_owner=True,
                is_active=True,
            )
        )
        db.add(
            PublicMCPApp(
                app_id="session-fault-catalog-app",
                name="session-fault-catalog-app",
                description="Session fault test app",
                transport="stdio",
                launch_config={
                    "command": "npx",
                    "args": ["-y", "session-fault-catalog-app"],
                },
            )
        )
        db.commit()
        return int(owner.id), int(server.id)


def test_a_toggle_that_already_committed_still_returns_200_when_the_hook_poisons_the_session(
    session_factory, seeded
) -> None:
    import xagent.web.api.mcp as mcp_api

    owner_id, server_id = seeded
    current_user = SimpleNamespace(id=owner_id, is_admin=False)

    def poisoning_access(db, user_id, refs):
        poison_by_raw_statement(db)
        return {}

    db = session_factory()
    try:
        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=poisoning_access)
            response = mcp_api.toggle_mcp_server(
                server_id, current_user=current_user, db=db
            )
        assert response.can_edit_global is True

        # No rollback here on purpose: the seam's hook door already
        # restored this session, and the query below is the statement that
        # proves it -- on PostgreSQL a poisoned transaction refuses every
        # later statement. Rolling back first would make this test pass
        # with the production restore removed.
        refreshed = (
            db.query(UserMCPServer)
            .filter(
                UserMCPServer.user_id == owner_id,
                UserMCPServer.mcpserver_id == server_id,
            )
            .one()
        )
        assert refreshed.is_active is False
    finally:
        db.close()


def test_connecting_an_app_still_returns_200_when_the_hook_poisons_the_session(
    session_factory, seeded
) -> None:
    import xagent.web.api.mcp as mcp_api

    owner_id, _server_id = seeded
    member = User(username="session-fault-member", password_hash="x", is_admin=False)

    def poisoning_access(db, user_id, refs):
        poison_by_raw_statement(db)
        return {}

    db = session_factory()
    try:
        db.add(member)
        db.commit()
        member_id = int(member.id)
        current_user = SimpleNamespace(id=member_id, is_admin=False)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=poisoning_access)
            response = mcp_api.connect_mcp_app(
                "session-fault-catalog-app",
                mcp_api.MCPAppConnectRequest(),
                current_user=current_user,
                db=db,
            )
        # Connecting never grants ownership -- the same value this route
        # always reported before any verdict existed.
        assert response.can_edit_global is False

        # No rollback here on purpose -- see the same note in the toggle
        # test above: the query below is the proof the seam's hook door
        # restored the session, not just an incidental fresh read.
        assoc = (
            db.query(UserMCPServer)
            .join(MCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
            .filter(
                UserMCPServer.user_id == member_id,
                MCPServer.name == "session-fault-catalog-app",
            )
            .one()
        )
        # ``.one()`` raises when the row is missing, so its own success is
        # the existence assertion. What this line adds is the route's own
        # decision: ``connect_mcp_app`` never grants ownership, and that
        # decision survived the poisoned hook.
        assert assoc.is_owner is False
    finally:
        db.close()


def test_the_apps_listing_still_returns_every_row_when_the_hook_poisons_the_session(
    session_factory, seeded
) -> None:
    import xagent.web.api.mcp as mcp_api

    owner_id, server_id = seeded
    member = User(username="session-fault-apps-member", password_hash="x")

    def poisoning_access(db, user_id, refs):
        poison_by_raw_statement(db)
        return {}

    db = session_factory()
    try:
        db.add(member)
        db.commit()
        member_id = int(member.id)
        current_user = SimpleNamespace(id=member_id, is_admin=False)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=poisoning_access,
                visibility=lambda _db, _uid: {
                    "mcp": {server_id},
                    "custom_api": set(),
                },
            )
            entries = mcp_api.list_mcp_apps(
                location="local", current_user=current_user, db=db
            )

        entry = next(e for e in entries if e["server_id"] == server_id)
        assert entry["can_configure"] is False
    finally:
        db.close()


def test_the_servers_listing_still_returns_every_row_when_the_hook_poisons_the_session(
    session_factory, seeded
) -> None:
    import xagent.web.api.mcp as mcp_api

    owner_id, server_id = seeded
    member = User(username="session-fault-servers-member", password_hash="x")

    def poisoning_access(db, user_id, refs):
        poison_by_raw_statement(db)
        return {}

    db = session_factory()
    try:
        db.add(member)
        db.commit()
        member_id = int(member.id)
        current_user = SimpleNamespace(id=member_id, is_admin=False)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=poisoning_access,
                visibility=lambda _db, _uid: {
                    "mcp": {server_id},
                    "custom_api": set(),
                },
            )
            entries = mcp_api.get_mcp_servers(current_user=current_user, db=db)

        entry = next(e for e in entries if e.id == server_id)
        assert entry.can_edit_global is False
    finally:
        db.close()


def test_the_seam_restores_the_session_after_a_raw_statement_failure(
    session_factory, seeded
) -> None:
    """Direct proof at the seam itself, independent of any particular
    route's attribute-loading order: a hook that runs a raw statement that
    aborts the PostgreSQL transaction still leaves the session usable for
    whatever the caller does next."""
    owner_id, server_id = seeded

    def poisoning_access(db, user_id, refs):
        poison_by_raw_statement(db)
        return {}

    db = session_factory()
    try:
        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=poisoning_access)
            with pytest.raises(ConnectorRuntimeError) as excinfo:
                resolve_connector_access_or_raise(db, owner_id, [("mcp", server_id)])
            assert excinfo.value.status_code == 503

        # The session must be usable again immediately afterward -- not
        # just after an explicit external rollback.
        result = db.execute(sa.select(sa.literal(1))).scalar()
        assert result == 1
    finally:
        db.close()
