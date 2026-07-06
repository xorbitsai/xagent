from __future__ import annotations

import pytest

from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.user import User

from .conftest import _admin_headers, _direct_db_session, _register_second_user, client

pytestmark = pytest.mark.usefixtures("_test_db")


def _uid(username: str) -> int:
    db = _direct_db_session()
    try:
        return int(db.query(User).filter(User.username == username).one().id)
    finally:
        db.close()


def _make_server_for(user_id: int, name: str) -> None:
    db = _direct_db_session()
    try:
        server = MCPServer(
            name=name, managed="external", transport="stdio", description="d"
        )
        db.add(server)
        db.flush()
        db.add(UserMCPServer(user_id=user_id, mcpserver_id=server.id, is_active=True))
        db.commit()
    finally:
        db.close()


def test_admin_can_list_other_users_mcp_servers():
    admin = _admin_headers()
    _register_second_user("bob", "bobpass1")
    bob_id = _uid("bob")
    _make_server_for(bob_id, "bob-server")

    resp = client.get(f"/api/mcp/servers?user_id={bob_id}", headers=admin)
    assert resp.status_code == 200, resp.text
    names = [s["name"] for s in resp.json()]
    assert "bob-server" in names


def test_non_admin_cannot_pass_user_id():
    _admin_headers()
    bob = _register_second_user("bob", "bobpass1")
    # Target a user id that is definitely not bob's, so the 403 comes from the
    # admin gate rather than falling through to the own-scope branch.
    admin_id = _uid("admin")
    resp = client.get(f"/api/mcp/servers?user_id={admin_id}", headers=bob)
    assert resp.status_code == 403
