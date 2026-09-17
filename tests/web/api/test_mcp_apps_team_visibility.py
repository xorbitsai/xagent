"""`/api/mcp/apps?location=local` must resolve the same connector set as
`/api/mcp/servers` (#1321).

Sharing a connector with a team writes a team link row and no per-member
association, so the picker's personal-association queries resolve nothing for
every member but the creator. Both endpoints overlay the team-owned ids the
`connector_team_scope` visibility hook reports; these tests pin that the
picker's local branch does it too, and that the standalone (no hook) and
remote responses are untouched.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.api.mcp import list_mcp_apps
from xagent.web.models.custom_api import CustomApi, UserCustomApi
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.services import connector_team_scope


@pytest.fixture()
def db_session(tmp_path):
    db_path = tmp_path / "mcp-apps-team.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = SessionLocal()

    creator = User(username="alice", password_hash="x", is_admin=False)
    member = User(username="bob", password_hash="x", is_admin=False)
    db.add_all([creator, member])
    db.commit()
    db.refresh(creator)
    db.refresh(member)

    yield db, creator, member
    db.close()
    engine.dispose()


@pytest.fixture(autouse=True)
def _reset_connector_team_hooks():
    """The hooks are process-global; never leak one into a sibling test."""
    yield
    connector_team_scope.set_connector_team_hooks()


def _install_visibility(mapping: dict[int, dict[str, set[int]]]) -> None:
    """Install a user-keyed visibility hook answering from ``mapping``."""

    def visibility(_db, user_id: int) -> dict[str, set[int]]:
        answer = mapping.get(int(user_id))
        if answer is None:
            return {"mcp": set(), "custom_api": set()}
        return {"mcp": set(answer["mcp"]), "custom_api": set(answer["custom_api"])}

    connector_team_scope.set_connector_team_hooks(visibility=visibility)


def _add_server(
    db, owner: User, name: str = "records", *, is_active: bool = True
) -> MCPServer:
    server = MCPServer.from_config(
        {
            "name": name,
            "description": f"{name} MCP server",
            "managed": "external",
            "transport": "stdio",
            "command": f"{name}-mcp",
        }
    )
    db.add(server)
    db.commit()
    db.refresh(server)
    db.add(
        UserMCPServer(
            user_id=owner.id,
            mcpserver_id=server.id,
            is_owner=True,
            is_active=is_active,
        )
    )
    db.commit()
    return server


def _add_unowned_server(db, name: str) -> MCPServer:
    """A server row with no personal association for anyone — reachable only
    through the team overlay."""
    server = MCPServer.from_config(
        {
            "name": name,
            "description": f"{name} MCP server",
            "managed": "external",
            "transport": "stdio",
            "command": f"{name}-mcp",
        }
    )
    db.add(server)
    db.commit()
    db.refresh(server)
    return server


def _add_oauth_server(db, owner: User, name: str = "notes") -> MCPServer:
    server = MCPServer.from_config(
        {
            "name": name,
            "managed": "external",
            "transport": "streamable_http",
            "url": "https://mcp.example.com/mcp",
            "auth": {
                "type": "mcp_oauth",
                "resource": "https://mcp.example.com/mcp",
                "issuer": "https://auth.example.com",
                "scope": "notes.read",
            },
        }
    )
    db.add(server)
    db.commit()
    db.refresh(server)
    db.add(
        UserMCPServer(
            user_id=owner.id,
            mcpserver_id=server.id,
            is_owner=True,
            is_active=True,
        )
    )
    db.commit()
    return server


def _add_custom_api(db, owner: User, name: str = "billing") -> CustomApi:
    api = CustomApi(
        name=name,
        description=f"{name} API",
        url="https://api.example.com/v1",
        method="GET",
    )
    db.add(api)
    db.commit()
    db.refresh(api)
    db.add(
        UserCustomApi(
            user_id=owner.id,
            custom_api_id=api.id,
            is_owner=True,
            is_active=True,
        )
    )
    db.commit()
    return api


def _add_catalog_app(
    db, app_id: str = "granola", name: str = "Granola"
) -> PublicMCPApp:
    app = PublicMCPApp(
        app_id=app_id,
        name=name,
        description="A catalog app",
        icon="",
        category="Productivity",
        transport="streamable_http",
        is_visible_in_connector=True,
    )
    db.add(app)
    db.commit()
    db.refresh(app)
    return app


def _local_ids(db, user: User) -> list[str]:
    return [a["id"] for a in list_mcp_apps(location="local", current_user=user, db=db)]


def test_local_branch_lists_a_team_owned_mcp_server_without_a_personal_row(db_session):
    """AC1: the connector the Tools page already shows must reach the picker."""
    db, creator, member = db_session
    server = _add_server(db, creator)
    _install_visibility(
        {int(member.id): {"mcp": {int(server.id)}, "custom_api": set()}}
    )

    entry = next(
        a
        for a in list_mcp_apps(location="local", current_user=member, db=db)
        if a["id"] == "records"
    )
    assert entry["server_id"] == server.id
    assert entry["is_custom"] is True
    assert entry["is_connected"] is True


def test_local_branch_lists_a_team_owned_custom_api_without_a_personal_row(db_session):
    """AC2: the Custom API half of the branch has the same association-only
    shape, so it drops team-owned APIs for the same reason."""
    db, creator, member = db_session
    api = _add_custom_api(db, creator)
    _install_visibility({int(member.id): {"mcp": set(), "custom_api": {int(api.id)}}})

    entry = next(
        a
        for a in list_mcp_apps(location="local", current_user=member, db=db)
        if a["id"] == "billing"
    )
    assert entry["server_id"] == api.id
    assert entry["transport"] == "custom_api"
    assert entry["is_connected"] is True


def test_a_personal_row_plus_a_team_link_lists_the_connector_exactly_once(db_session):
    """AC3: the creator holds both; the overlay must not duplicate the entry."""
    db, creator, _member = db_session
    server = _add_server(db, creator)
    api = _add_custom_api(db, creator)
    _install_visibility(
        {
            int(creator.id): {
                "mcp": {int(server.id)},
                "custom_api": {int(api.id)},
            }
        }
    )

    ids = _local_ids(db, creator)
    assert ids.count("records") == 1
    assert ids.count("billing") == 1


def test_a_connector_owned_by_another_team_is_never_listed(db_session):
    """AC4: the overlay is keyed on the *requesting* user, so a connector a
    live hook reports for someone else must not reach this user's picker.

    `shared` has no personal association for anyone, so it is reachable only
    through the overlay: the creator's assertion fails if the overlay is
    deleted, and the member's fails if it ignores the requesting user. An
    answer that was empty for everyone — or a connector the creator also owned
    personally — would leave both assertions true either way."""
    db, creator, member = db_session
    shared = _add_unowned_server(db, "shared")
    _install_visibility(
        {int(creator.id): {"mcp": {int(shared.id)}, "custom_api": set()}}
    )

    assert _local_ids(db, creator) == ["shared"]
    assert _local_ids(db, member) == []


def test_standalone_response_is_unchanged_with_no_hook_installed(db_session):
    """AC5: `visible_team_connector_ids` resolves empty without a hook, so a
    standalone deployment takes the pre-overlay path. Entry *fields* on that
    path are pinned by the existing tests in test_mcp_oauth_flow.py; what this
    pins is that the overlay adds and removes nothing when no hook is
    installed."""
    db, creator, member = db_session
    _add_server(db, creator)
    _add_custom_api(db, creator)

    assert _local_ids(db, creator) == ["records", "billing"]
    assert _local_ids(db, member) == []


def test_remote_results_are_unchanged_by_the_team_overlay(db_session):
    """AC6: the overlay is scoped to the local branch. A team-owned server must
    not leak into the remote branch's connection-state lookups, which answer
    "is this catalog app connected *for me*" from personal associations only."""
    db, creator, member = db_session
    _add_catalog_app(db)
    server = _add_server(db, creator, name="granola")
    _install_visibility(
        {int(member.id): {"mcp": {int(server.id)}, "custom_api": set()}}
    )

    remote = list_mcp_apps(location="remote", current_user=member, db=db)
    granola = next(a for a in remote if a["id"] == "granola")
    assert granola["is_connected"] is False
    assert "server_id" not in granola


def test_a_team_owned_server_named_like_a_catalog_app_is_skipped(db_session):
    """The local branch excludes servers backing a catalog app so they are not
    listed twice; an overlaid server must obey the same rule."""
    db, creator, member = db_session
    _add_catalog_app(db)
    server = _add_server(db, creator, name="Granola")
    _install_visibility(
        {int(member.id): {"mcp": {int(server.id)}, "custom_api": set()}}
    )

    assert "Granola" not in _local_ids(db, member)


def test_a_team_owned_mcp_oauth_server_carries_no_auth_type(db_session):
    """`auth_type` tells the picker to POST `/{server_id}/oauth/connect`, which
    requires an *active personal association* and 404s without one. A team
    member holds no association at all, so advertising the flow would trade a
    missing entry for a failed popup — the same reasoning that already excludes
    deactivated associations."""
    db, creator, member = db_session
    server = _add_oauth_server(db, creator)
    _install_visibility(
        {int(member.id): {"mcp": {int(server.id)}, "custom_api": set()}}
    )

    entry = next(
        a
        for a in list_mcp_apps(location="local", current_user=member, db=db)
        if a["id"] == "notes"
    )
    assert entry["is_connected"] is False
    assert "auth_type" not in entry


def test_the_search_filter_applies_to_overlaid_connectors(db_session):
    """The overlay feeds the same loops, so the existing search filter must
    still narrow the result rather than being bypassed for team-owned rows.

    No category assertion here: the local branch discards *every* row for any
    category other than "All", so a category assertion holds even with the
    overlay deleted and would pin nothing."""
    db, creator, member = db_session
    records = _add_server(db, creator, name="records")
    shipping = _add_server(db, creator, name="shipping")
    billing = _add_custom_api(db, creator, name="billing")
    _install_visibility(
        {
            int(member.id): {
                "mcp": {int(records.id), int(shipping.id)},
                "custom_api": {int(billing.id)},
            }
        }
    )

    assert [
        a["id"]
        for a in list_mcp_apps(
            location="local", search="bill", current_user=member, db=db
        )
    ] == ["billing"]
    assert [
        a["id"]
        for a in list_mcp_apps(
            location="local", search="record", current_user=member, db=db
        )
    ] == ["records"]


@pytest.mark.parametrize("location", ["local", "all"])
def test_the_overlay_applies_to_both_local_and_all(db_session, location):
    """`location="all"` runs the same branch as "local". The frontend only
    sends "local" today, so pin that the branch is entered for both rather
    than leaving "all" to drift."""
    db, creator, member = db_session
    server = _add_server(db, creator)
    api = _add_custom_api(db, creator)
    _install_visibility(
        {
            int(member.id): {
                "mcp": {int(server.id)},
                "custom_api": {int(api.id)},
            }
        }
    )

    ids = [
        a["id"] for a in list_mcp_apps(location=location, current_user=member, db=db)
    ]
    assert ids == ["records", "billing"]


def test_an_inactive_personal_row_plus_a_team_link_lists_one_entry(db_session):
    """A member who deactivated their own association while the connector stays
    team-owned must still see exactly one row, not one per source.

    The dedup index is built from *all* personal associations regardless of
    `is_active`, matching `get_mcp_servers`' `own_mcp_ids`; an index that
    filtered on `is_active` would overlay a second copy of this server."""
    db, creator, _member = db_session
    server = _add_server(db, creator, is_active=False)
    _install_visibility(
        {int(creator.id): {"mcp": {int(server.id)}, "custom_api": set()}}
    )

    assert _local_ids(db, creator).count("records") == 1


def test_a_hook_answering_string_ids_is_not_silently_resolved(db_session):
    """Regression pin for a fail-open path: SQLite's numeric affinity makes
    `id IN ('1')` match an INTEGER primary key, so a hook answering string ids
    would list a connector the dedup index (int-keyed) considered missing.

    Element types are the hook's contract (`dict[str, set[int]]`). This pins
    today's behavior so a future validator in `connector_team_scope` — which
    must check element types, not just the container shape — has a test that
    changes when the behavior does."""
    db, creator, member = db_session
    server = _add_server(db, creator)
    _install_visibility(
        {int(member.id): {"mcp": {str(server.id)}, "custom_api": set()}}  # type: ignore[dict-item]
    )

    listed = _local_ids(db, member)
    # Documents the current fail-open outcome rather than asserting it is
    # correct: a string id resolves through the IN clause and reaches the
    # picker. A validator rejecting non-int members would turn this into a
    # raised error, and this assertion is where that change surfaces.
    assert listed == ["records"]
