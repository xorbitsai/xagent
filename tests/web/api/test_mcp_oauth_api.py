"""Integration tests for OAuth connect/callback/connection endpoints.

These endpoints live in ``xagent.web.api.mcp_oauth`` and are intentionally
NOT part of ``conftest.app_for_tests`` (that shared app only wires up
auth/agents/v1-style routers). We build a small dedicated FastAPI app here
that includes the real ``auth_router`` (so we can log in as a real user via
HTTP, matching how the frontend will authenticate) plus the
``mcp_oauth_router`` under test, and reuse the same ``_override_get_db`` /
``_test_db`` machinery as the rest of the suite so both the test setup and
the app hit the same underlying SQLite DB.
"""

import asyncio
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mcp.shared.auth import OAuthClientInformationFull

from xagent.core.tools.core.mcp.oauth.token_storage import DBTokenStorage
from xagent.core.utils.encryption import decrypt_value
from xagent.web.api.auth import auth_router
from xagent.web.api.mcp import mcp_router
from xagent.web.api.mcp_oauth import router as mcp_oauth_router
from xagent.web.models.database import get_db
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.mcp_oauth import MCPUserOAuthToken
from xagent.web.models.user import User

from .conftest import _override_get_db

# ===== App wiring (separate from conftest's shared `app_for_tests`) =====

app = FastAPI()
app.include_router(auth_router)
app.include_router(mcp_oauth_router)
# `mcp_router` (GET /api/mcp/servers/{server_id}/tools among others) is
# included here too so tests can exercise the tools-preview endpoint
# alongside the OAuth connect/callback/connection endpoints above, which
# both live under the same `/api/mcp` prefix.
app.include_router(mcp_router)
app.dependency_overrides[get_db] = _override_get_db

oauth_client = TestClient(app, raise_server_exceptions=False)


# ===== Local auth helpers (mirrors conftest's, but bound to `oauth_client`) =====


def _setup_admin() -> None:
    status = oauth_client.get("/api/auth/setup-status")
    assert status.status_code == 200
    if status.json().get("needs_setup", True):
        resp = oauth_client.post(
            "/api/auth/setup-admin",
            json={
                "username": "admin",
                "email": "admin@example.com",
                "password": "admin123",
            },
        )
        assert resp.status_code == 200


def _login(username: str = "admin", password: str = "admin123") -> dict[str, str]:
    resp = oauth_client.post(
        "/api/auth/login", json={"username": username, "password": password}
    )
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _admin_headers() -> dict[str, str]:
    _setup_admin()
    return _login()


def _register_second_user(
    username: str = "bob", password: str = "bobpass1"
) -> dict[str, str]:
    """Register a second user via the public endpoint, return their auth header.

    Used by cross-user-isolation tests (e.g. accessing another user's
    server must return 404 without revealing that the server exists).
    """
    resp = oauth_client.post(
        "/api/auth/register",
        json={
            "username": username,
            "email": f"{username}@example.com",
            "password": password,
        },
    )
    assert resp.status_code == 200, resp.text
    return _login(username, password)


def _direct_db_session():
    return next(get_db())


def _seed_mcp_server(owner_username: str = "admin") -> int:
    """Insert a minimal remote MCP server row, link it to ``owner_username``
    via ``UserMCPServer`` (mirroring how server-scoped endpoints elsewhere
    enforce ownership), and return the server id.
    """
    db = _direct_db_session()
    try:
        server = MCPServer(
            name="test-remote-mcp",
            managed="external",
            transport="streamable_http",
            url="https://example-mcp.test/mcp",
        )
        db.add(server)
        db.commit()
        db.refresh(server)

        owner = db.query(User).filter_by(username=owner_username).one_or_none()
        if owner is not None:
            link = UserMCPServer(
                user_id=owner.id,
                mcpserver_id=server.id,
                is_owner=True,
                can_edit=True,
                can_delete=True,
                is_active=True,
            )
            db.add(link)
            db.commit()

        return server.id
    finally:
        db.close()


def _seed_oauth_mcp_server(owner_username: str = "admin") -> int:
    """Insert a remote MCP server row flagged with ``auth={"type": "oauth_mcp"}``,
    linked to ``owner_username`` via ``UserMCPServer``, and return the server id.
    """
    db = _direct_db_session()
    try:
        server = MCPServer(
            name="test-oauth-remote-mcp",
            managed="external",
            transport="streamable_http",
            url="https://example-mcp.test/mcp",
            auth={"type": "oauth_mcp"},
        )
        db.add(server)
        db.commit()
        db.refresh(server)

        owner = db.query(User).filter_by(username=owner_username).one_or_none()
        if owner is not None:
            link = UserMCPServer(
                user_id=owner.id,
                mcpserver_id=server.id,
                is_owner=True,
                can_edit=True,
                can_delete=True,
                is_active=True,
            )
            db.add(link)
            db.commit()

        return server.id
    finally:
        db.close()


FAKE_AS_META = {
    "authorization_endpoint": "https://example-mcp.test/authorize",
    "token_endpoint": "https://example-mcp.test/token",
    "registration_endpoint": "https://example-mcp.test/register",
}


async def _fake_discover_auth_server(server_url, client):
    return dict(FAKE_AS_META)


async def _fake_register_client_dcr(as_meta, redirect_uris, client):
    return {"client_id": "fake-client-id", "client_secret": "fake-client-secret"}


@pytest.fixture(autouse=True)
def _db(_test_db):
    """Pull in the shared per-test sqlite schema fixture from conftest."""


class TestMCPOAuthConnect:
    def test_connect_returns_authorization_url(self, monkeypatch):
        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.discover_auth_server", _fake_discover_auth_server
        )
        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.register_client_dcr", _fake_register_client_dcr
        )

        headers = _admin_headers()
        server_id = _seed_mcp_server()

        resp = oauth_client.post(f"/api/mcp/{server_id}/connect", headers=headers)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "authorization_url" in body
        auth_url = body["authorization_url"]
        assert auth_url.startswith(FAKE_AS_META["authorization_endpoint"])

        parsed = parse_qs(urlparse(auth_url).query)
        assert "state" in parsed
        assert "code_challenge" in parsed

        db = _direct_db_session()
        try:
            row = (
                db.query(MCPUserOAuthToken)
                .filter_by(mcpserver_id=server_id)
                .one_or_none()
            )
            assert row is not None
            assert row.status == "pending"
            assert row.pkce_verifier is not None
            # Should be Fernet-encrypted at rest: decrypting it must round-trip
            # to a non-empty value that differs from the ciphertext stored.
            decrypted_verifier = decrypt_value(row.pkce_verifier)
            assert decrypted_verifier
            assert decrypted_verifier != row.pkce_verifier
        finally:
            db.close()

    def test_connect_requires_auth(self):
        server_id = _seed_mcp_server()
        resp = oauth_client.post(f"/api/mcp/{server_id}/connect")
        assert resp.status_code in (401, 403)

    def test_connect_missing_server_404(self):
        headers = _admin_headers()
        resp = oauth_client.post("/api/mcp/999999/connect", headers=headers)
        assert resp.status_code == 404

    def test_connect_registered_client_is_round_trippable(self, monkeypatch):
        """The ``oauth_client`` written by ``/connect`` must be a full,
        round-trippable ``OAuthClientInformationFull`` shape.

        Regression test: the connect endpoint used to persist a lossy dict
        (only ``client_id``/``client_secret``/``source``) that is missing
        fields the SDK's ``OAuthClientInformationFull`` model requires (e.g.
        ``redirect_uris``). ``DBTokenStorage.get_client_info()`` validates the
        stored dict against that model and swallows the resulting
        ``ValidationError``, silently returning ``None``. With no client
        info, the SDK's ``OAuthClientProvider`` can't refresh access tokens
        and instead forces the user through DCR/reauth again on every expiry.
        """
        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.discover_auth_server", _fake_discover_auth_server
        )
        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.register_client_dcr", _fake_register_client_dcr
        )

        headers = _admin_headers()
        server_id = _seed_mcp_server()

        resp = oauth_client.post(f"/api/mcp/{server_id}/connect", headers=headers)
        assert resp.status_code == 200, resp.text

        db = _direct_db_session()
        try:
            user = db.query(User).filter_by(username="admin").one()
            store = DBTokenStorage(user.id, server_id, db)
            client_info = asyncio.run(store.get_client_info())
            assert client_info is not None, (
                "get_client_info() returned None: the stored oauth_client "
                "dict is not a valid OAuthClientInformationFull, so token "
                "auto-refresh is broken"
            )
            assert isinstance(client_info, OAuthClientInformationFull)
            assert client_info.client_id == "fake-client-id"
            assert client_info.client_secret == "fake-client-secret"
            assert client_info.redirect_uris
        finally:
            db.close()

    def test_connect_forbidden_for_non_owner(self, monkeypatch):
        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.discover_auth_server", _fake_discover_auth_server
        )
        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.register_client_dcr", _fake_register_client_dcr
        )

        # User A (admin) owns the server via a UserMCPServer link.
        _admin_headers()
        server_id = _seed_mcp_server(owner_username="admin")

        # User B is authenticated but has no UserMCPServer link to it.
        bob_headers = _register_second_user()

        resp = oauth_client.post(f"/api/mcp/{server_id}/connect", headers=bob_headers)
        assert resp.status_code == 404


class TestMCPOAuthCallback:
    def _do_connect(self, monkeypatch) -> tuple[dict, int, str]:
        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.discover_auth_server", _fake_discover_auth_server
        )
        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.register_client_dcr", _fake_register_client_dcr
        )
        headers = _admin_headers()
        server_id = _seed_mcp_server()
        resp = oauth_client.post(f"/api/mcp/{server_id}/connect", headers=headers)
        assert resp.status_code == 200, resp.text
        auth_url = resp.json()["authorization_url"]
        state = parse_qs(urlparse(auth_url).query)["state"][0]
        return headers, server_id, state

    def test_callback_stores_tokens(self, monkeypatch):
        headers, server_id, state = self._do_connect(monkeypatch)

        async def fake_exchange(*args, **kwargs):
            return {
                "access_token": "AT",
                "token_type": "Bearer",
                "refresh_token": "RT",
                "expires_in": 3600,
            }

        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.exchange_code_for_tokens", fake_exchange
        )

        resp = oauth_client.get(
            "/api/mcp/oauth/callback", params={"code": "abc", "state": state}
        )
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

        db = _direct_db_session()
        try:
            row = (
                db.query(MCPUserOAuthToken)
                .filter_by(mcpserver_id=server_id)
                .one_or_none()
            )
            assert row is not None
            assert row.status == "connected"
            assert row.access_token is not None
            assert row.access_token != "AT"  # encrypted at rest
            assert row.refresh_token is not None
            assert row.refresh_token != "RT"
            assert row.pkce_verifier is None
            assert row.state is None
            assert row.expires_at is not None
        finally:
            db.close()

        # Connection status should now report connected via the API too.
        status_resp = oauth_client.get(
            f"/api/mcp/{server_id}/connection", headers=headers
        )
        assert status_resp.status_code == 200
        assert status_resp.json() == {"status": "connected"}

    def test_callback_rejects_bad_state(self):
        resp = oauth_client.get(
            "/api/mcp/oauth/callback", params={"code": "abc", "state": "garbage"}
        )
        assert resp.status_code == 400

    def test_callback_rejects_unknown_pending_row(self):
        # Valid, well-formed state but no matching pending row in the DB
        # (e.g. replayed or already-consumed callback).
        from xagent.core.tools.core.mcp.oauth.flow import encode_state
        from xagent.web.models.user import User

        _admin_headers()
        server_id = _seed_mcp_server()
        # Look up the just-created admin's id directly from the DB.
        db = _direct_db_session()
        try:
            user = db.query(User).filter_by(username="admin").one()
            user_id = user.id
        finally:
            db.close()

        stale_state = encode_state(user_id=user_id, mcpserver_id=server_id)
        resp = oauth_client.get(
            "/api/mcp/oauth/callback", params={"code": "abc", "state": stale_state}
        )
        assert resp.status_code == 400


class TestMCPOAuthConnection:
    def test_connection_status_and_disconnect(self, monkeypatch):
        headers, server_id, state = TestMCPOAuthCallback()._do_connect(monkeypatch)

        async def fake_exchange(*args, **kwargs):
            return {"access_token": "AT", "token_type": "Bearer"}

        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.exchange_code_for_tokens", fake_exchange
        )
        cb_resp = oauth_client.get(
            "/api/mcp/oauth/callback", params={"code": "abc", "state": state}
        )
        assert cb_resp.status_code == 200

        status_resp = oauth_client.get(
            f"/api/mcp/{server_id}/connection", headers=headers
        )
        assert status_resp.status_code == 200
        assert status_resp.json() == {"status": "connected"}

        del_resp = oauth_client.delete(
            f"/api/mcp/{server_id}/connection", headers=headers
        )
        assert del_resp.status_code == 200
        assert del_resp.json() == {"status": "not_connected"}

        status_resp_after = oauth_client.get(
            f"/api/mcp/{server_id}/connection", headers=headers
        )
        assert status_resp_after.status_code == 200
        assert status_resp_after.json() == {"status": "not_connected"}

    def test_connection_status_not_connected_by_default(self):
        headers = _admin_headers()
        server_id = _seed_mcp_server()
        resp = oauth_client.get(f"/api/mcp/{server_id}/connection", headers=headers)
        assert resp.status_code == 200
        assert resp.json() == {"status": "not_connected"}


class TestMCPToolsPreviewOAuth:
    def test_tools_preview_attaches_oauth_provider(self, monkeypatch):
        """GET /api/mcp/servers/{server_id}/tools must route an oauth_mcp
        server's connection through ``attach_oauth_provider_if_needed`` before
        opening a live MCP session, so it carries a real ``OAuthClientProvider``
        instead of the inert ``oauth_mcp`` marker (which live MCP sessions do
        not understand and would fail to authenticate with).
        """
        from mcp.client.auth import OAuthClientProvider

        captured: dict = {}

        async def fake_load_mcp_tools_as_agent_tools(connections_dict, **kwargs):
            captured.update(connections_dict)
            return []

        monkeypatch.setattr(
            "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
            fake_load_mcp_tools_as_agent_tools,
        )

        headers = _admin_headers()
        server_id = _seed_oauth_mcp_server()

        resp = oauth_client.get(
            f"/api/mcp/servers/{server_id}/tools", headers=headers
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["tools"] == []

        assert "test-oauth-remote-mcp" in captured
        connection = captured["test-oauth-remote-mcp"]
        assert "oauth_mcp" not in connection
        assert isinstance(connection.get("auth"), OAuthClientProvider)
