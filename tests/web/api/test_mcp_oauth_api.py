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
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mcp.shared.auth import OAuthClientInformationFull

from xagent.core.utils.encryption import decrypt_value
from xagent.web.api.auth import auth_router
from xagent.web.api.mcp import mcp_router
from xagent.web.api.mcp_oauth import router as mcp_oauth_router
from xagent.web.models.database import get_db
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.mcp_oauth import MCPUserOAuthToken
from xagent.web.models.user import User
from xagent.web.services.mcp_oauth_token_storage import DBTokenStorage

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


def _seed_oauth_mcp_server(
    owner_username: str = "admin",
    *,
    transport: str = "streamable_http",
    url: str = "https://example-mcp.test/mcp",
) -> int:
    """Insert a remote MCP server row flagged with ``auth={"type": "oauth_mcp"}``,
    linked to ``owner_username`` via ``UserMCPServer``, and return the server id.
    """
    db = _direct_db_session()
    try:
        server = MCPServer(
            name="test-oauth-remote-mcp",
            managed="external",
            transport=transport,
            url=url,
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


def _share_mcp_server(
    server_id: int,
    username: str,
    *,
    is_owner: bool = False,
    can_edit: bool = False,
) -> None:
    db = _direct_db_session()
    try:
        user = db.query(User).filter_by(username=username).one()
        db.add(
            UserMCPServer(
                user_id=user.id,
                mcpserver_id=server_id,
                is_owner=is_owner,
                can_edit=can_edit,
                can_delete=False,
                is_active=True,
            )
        )
        db.commit()
    finally:
        db.close()


FAKE_AS_META = {
    "authorization_endpoint": "https://example-mcp.test/authorize",
    "token_endpoint": "https://example-mcp.test/token",
    "registration_endpoint": "https://example-mcp.test/register",
}


async def _fake_discover_auth_server(server_url, client):
    return dict(FAKE_AS_META)


async def _fake_register_client_dcr(as_meta, client_metadata, client):
    assert client_metadata["client_name"] == "xagent"
    assert client_metadata["redirect_uris"]
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
        server_id = _seed_oauth_mcp_server()

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

    def test_connect_rejects_non_oauth_mcp_server(self, monkeypatch):
        async def fail_discovery(*args, **kwargs):
            raise AssertionError("plain MCP servers must not run OAuth discovery")

        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.discover_auth_server", fail_discovery
        )

        headers = _admin_headers()
        server_id = _seed_mcp_server()

        resp = oauth_client.post(f"/api/mcp/{server_id}/connect", headers=headers)
        assert resp.status_code == 400
        assert "not configured for OAuth" in resp.json()["detail"]

    def test_connect_rejects_oauth_mcp_on_websocket(self):
        headers = _admin_headers()
        server_id = _seed_oauth_mcp_server(
            transport="websocket", url="wss://example-mcp.test/ws"
        )

        resp = oauth_client.post(f"/api/mcp/{server_id}/connect", headers=headers)

        assert resp.status_code == 400
        assert "only supported for SSE" in resp.json()["detail"]

    def test_connect_discovery_error_is_readable(self, monkeypatch):
        async def fail_discovery(*args, **kwargs):
            raise ValueError("metadata missing")

        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.discover_auth_server", fail_discovery
        )

        headers = _admin_headers()
        server_id = _seed_oauth_mcp_server()

        resp = oauth_client.post(f"/api/mcp/{server_id}/connect", headers=headers)
        assert resp.status_code == 422
        assert "Authorization server discovery failed" in resp.json()["detail"]

    def test_connect_discovery_ssrf_is_readable(self, monkeypatch):
        """A discovery attempt blocked by the SSRF guard (e.g. the server's
        metadata resolves to a private/internal address) must produce the
        same kind of clean, friendly error as any other discovery failure --
        not an unhandled 500.
        """
        from xagent.core.tools.core.mcp.oauth.ssrf_guard import (
            UnsafeOAuthEndpointError,
        )

        async def fail_discovery(*args, **kwargs):
            raise UnsafeOAuthEndpointError("blocked: internal address")

        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.discover_auth_server", fail_discovery
        )

        headers = _admin_headers()
        server_id = _seed_oauth_mcp_server()

        resp = oauth_client.post(f"/api/mcp/{server_id}/connect", headers=headers)
        assert resp.status_code == 422
        assert "disallowed address" in resp.json()["detail"]

    def test_connect_validates_callback_base_url_before_discovery(self, monkeypatch):
        async def fail_discovery(*args, **kwargs):
            raise AssertionError("invalid callback URL must fail before discovery")

        monkeypatch.setenv("XAGENT_OAUTH_CALLBACK_BASE_URL", "localhost:8000")
        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.discover_auth_server", fail_discovery
        )

        headers = _admin_headers()
        server_id = _seed_oauth_mcp_server()

        resp = oauth_client.post(f"/api/mcp/{server_id}/connect", headers=headers)

        assert resp.status_code == 422
        assert "http(s) scheme and host" in resp.json()["detail"]

    def test_connect_reuses_pending_authorization_url(self, monkeypatch):
        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.discover_auth_server", _fake_discover_auth_server
        )
        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.register_client_dcr", _fake_register_client_dcr
        )

        headers = _admin_headers()
        server_id = _seed_oauth_mcp_server()

        first = oauth_client.post(f"/api/mcp/{server_id}/connect", headers=headers)
        second = oauth_client.post(f"/api/mcp/{server_id}/connect", headers=headers)

        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        assert second.json()["authorization_url"] == first.json()["authorization_url"]

        db = _direct_db_session()
        try:
            assert (
                db.query(MCPUserOAuthToken).filter_by(mcpserver_id=server_id).count()
                == 1
            )
        finally:
            db.close()

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
        server_id = _seed_oauth_mcp_server()

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
            assert client_info.client_name == "xagent"
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
        server_id = _seed_oauth_mcp_server(owner_username="admin")

        # User B is authenticated but has no UserMCPServer link to it.
        bob_headers = _register_second_user()

        resp = oauth_client.post(f"/api/mcp/{server_id}/connect", headers=bob_headers)
        assert resp.status_code == 404

    def test_connect_forbids_first_time_dcr_for_non_edit_shared_user(self, monkeypatch):
        """A user with mere (read-only) access via sharing -- not edit/owner
        rights -- must not be able to trigger first-time Dynamic Client
        Registration, since it permanently writes a shared, server-wide
        OAuth client onto the MCPServer row.
        """

        async def fail_dcr(*args, **kwargs):
            raise AssertionError("DCR must not run without edit permission")

        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.discover_auth_server", _fake_discover_auth_server
        )
        monkeypatch.setattr("xagent.web.api.mcp_oauth.register_client_dcr", fail_dcr)

        _admin_headers()
        server_id = _seed_oauth_mcp_server(owner_username="admin")

        bob_headers = _register_second_user()
        _share_mcp_server(server_id, "bob", is_owner=False, can_edit=False)

        resp = oauth_client.post(f"/api/mcp/{server_id}/connect", headers=bob_headers)
        assert resp.status_code == 403

        db = _direct_db_session()
        try:
            server = db.query(MCPServer).filter_by(id=server_id).one()
            assert server.oauth_client is None
        finally:
            db.close()

    def test_connect_allows_first_time_dcr_for_edit_user(self, monkeypatch):
        """An owner/edit user must still be able to trigger first-time DCR."""
        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.discover_auth_server", _fake_discover_auth_server
        )
        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.register_client_dcr", _fake_register_client_dcr
        )

        _admin_headers()
        server_id = _seed_oauth_mcp_server(owner_username="admin")

        bob_headers = _register_second_user()
        _share_mcp_server(server_id, "bob", is_owner=False, can_edit=True)

        resp = oauth_client.post(f"/api/mcp/{server_id}/connect", headers=bob_headers)
        assert resp.status_code == 200, resp.text

        db = _direct_db_session()
        try:
            server = db.query(MCPServer).filter_by(id=server_id).one()
            assert server.oauth_client is not None
        finally:
            db.close()

    def test_connect_allows_non_edit_user_when_client_already_registered(
        self, monkeypatch
    ):
        """The permission gate only applies to first-time DCR -- a user with
        mere access must still be able to connect (start the authorization
        flow) once an oauth_client already exists on the server.
        """

        async def fail_dcr(*args, **kwargs):
            raise AssertionError("DCR must not run when a client already exists")

        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.discover_auth_server", _fake_discover_auth_server
        )
        monkeypatch.setattr("xagent.web.api.mcp_oauth.register_client_dcr", fail_dcr)

        _admin_headers()
        server_id = _seed_oauth_mcp_server(owner_username="admin")

        db = _direct_db_session()
        try:
            server = db.query(MCPServer).filter_by(id=server_id).one()
            server.oauth_client = {
                "client_id": "already-registered",
                "redirect_uris": ["https://xagent.test/oauth/callback"],
                "client_name": "xagent",
            }
            db.commit()
        finally:
            db.close()

        bob_headers = _register_second_user()
        _share_mcp_server(server_id, "bob", is_owner=False, can_edit=False)

        resp = oauth_client.post(f"/api/mcp/{server_id}/connect", headers=bob_headers)
        assert resp.status_code == 200, resp.text

    def test_connect_requests_discovered_scopes(self, monkeypatch):
        """When discovery advertises scopes_supported, the authorization URL
        must request them; when it doesn't, no scope param should appear.
        """

        async def discover_with_scopes(server_url, client):
            return dict(FAKE_AS_META, scopes_supported=["read", "write"])

        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.discover_auth_server", discover_with_scopes
        )
        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.register_client_dcr", _fake_register_client_dcr
        )

        headers = _admin_headers()
        server_id = _seed_oauth_mcp_server()

        resp = oauth_client.post(f"/api/mcp/{server_id}/connect", headers=headers)
        assert resp.status_code == 200, resp.text
        auth_url = resp.json()["authorization_url"]

        parsed = parse_qs(urlparse(auth_url).query)
        assert parsed["scope"] == ["read write"]

    def test_connect_omits_scope_when_not_advertised(self, monkeypatch):
        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.discover_auth_server", _fake_discover_auth_server
        )
        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.register_client_dcr", _fake_register_client_dcr
        )

        headers = _admin_headers()
        server_id = _seed_oauth_mcp_server()

        resp = oauth_client.post(f"/api/mcp/{server_id}/connect", headers=headers)
        assert resp.status_code == 200, resp.text
        auth_url = resp.json()["authorization_url"]

        parsed = parse_qs(urlparse(auth_url).query)
        assert "scope" not in parsed


class TestMCPOAuthCallback:
    def _do_connect(self, monkeypatch) -> tuple[dict, int, str]:
        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.discover_auth_server", _fake_discover_auth_server
        )
        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.register_client_dcr", _fake_register_client_dcr
        )
        headers = _admin_headers()
        server_id = _seed_oauth_mcp_server()
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

    def test_callback_error_marks_pending_row_error(self, monkeypatch):
        _, server_id, state = self._do_connect(monkeypatch)

        resp = oauth_client.get(
            "/api/mcp/oauth/callback",
            params={
                "error": "access_denied",
                "error_description": "User denied access",
                "state": state,
            },
        )
        assert resp.status_code == 400
        assert "Authorization failed" in resp.text

        db = _direct_db_session()
        try:
            row = (
                db.query(MCPUserOAuthToken)
                .filter_by(mcpserver_id=server_id)
                .one_or_none()
            )
            assert row is not None
            assert row.status == "error"
            assert row.pkce_verifier is None
            assert row.state is None
        finally:
            db.close()

    def test_callback_malformed_token_response_marks_pending_row_error(
        self, monkeypatch
    ):
        """A 200 response from the token endpoint missing required fields
        (e.g. ``access_token``) makes ``OAuthToken.model_validate`` raise a
        pydantic ValidationError. This must be handled the same clean way as
        a transport-level (httpx.HTTPError) failure -- not leak an
        unhandled 500 with a stuck pending row.
        """
        _, server_id, state = self._do_connect(monkeypatch)

        async def fake_exchange_missing_access_token(*args, **kwargs):
            return {"token_type": "Bearer"}  # no access_token

        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.exchange_code_for_tokens",
            fake_exchange_missing_access_token,
        )

        resp = oauth_client.get(
            "/api/mcp/oauth/callback", params={"code": "abc", "state": state}
        )
        assert resp.status_code == 400
        assert "Authorization failed" in resp.text

        db = _direct_db_session()
        try:
            row = (
                db.query(MCPUserOAuthToken)
                .filter_by(mcpserver_id=server_id)
                .one_or_none()
            )
            assert row is not None
            assert row.status == "error"
            assert row.pkce_verifier is None
            assert row.state is None
        finally:
            db.close()

    def test_callback_does_not_recreate_deleted_pending_row(self, monkeypatch):
        _, server_id, state = self._do_connect(monkeypatch)

        async def fake_exchange(*args, **kwargs):
            db = _direct_db_session()
            try:
                db.query(MCPUserOAuthToken).filter_by(mcpserver_id=server_id).delete()
                db.commit()
            finally:
                db.close()
            return {"access_token": "AT", "token_type": "Bearer"}

        monkeypatch.setattr(
            "xagent.web.api.mcp_oauth.exchange_code_for_tokens", fake_exchange
        )

        resp = oauth_client.get(
            "/api/mcp/oauth/callback", params={"code": "abc", "state": state}
        )

        assert resp.status_code == 400
        assert "canceled or expired" in resp.text

        db = _direct_db_session()
        try:
            assert (
                db.query(MCPUserOAuthToken).filter_by(mcpserver_id=server_id).count()
                == 0
            )
        finally:
            db.close()

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

    def test_connection_status_reports_expired_token_without_refresh(self):
        headers = _admin_headers()
        server_id = _seed_oauth_mcp_server()

        db = _direct_db_session()
        try:
            user = db.query(User).filter_by(username="admin").one()
            db.add(
                MCPUserOAuthToken(
                    user_id=user.id,
                    mcpserver_id=server_id,
                    status="connected",
                    access_token="old-token",
                    expires_at=datetime.now(timezone.utc) - timedelta(minutes=5),
                )
            )
            db.commit()
        finally:
            db.close()

        resp = oauth_client.get(f"/api/mcp/{server_id}/connection", headers=headers)

        assert resp.status_code == 200
        assert resp.json() == {"status": "expired"}


class TestMCPServerUpdateOAuthReview:
    def test_update_url_clears_shared_oauth_state_and_tokens(self):
        headers = _admin_headers()
        server_id = _seed_oauth_mcp_server()

        db = _direct_db_session()
        try:
            user = db.query(User).filter_by(username="admin").one()
            server = db.query(MCPServer).filter_by(id=server_id).one()
            server.oauth_client = {"client_id": "old-client"}
            server.auth_server_metadata = {"issuer": "https://old.example"}
            db.add(
                MCPUserOAuthToken(
                    user_id=user.id,
                    mcpserver_id=server_id,
                    status="connected",
                    access_token="old-token",
                )
            )
            db.commit()
        finally:
            db.close()

        resp = oauth_client.put(
            f"/api/mcp/servers/{server_id}",
            headers=headers,
            json={
                "config": {
                    "url": "https://new-mcp.test/mcp",
                    "auth": {"type": "oauth_mcp"},
                }
            },
        )
        assert resp.status_code == 200, resp.text

        db = _direct_db_session()
        try:
            server = db.query(MCPServer).filter_by(id=server_id).one()
            assert server.url == "https://new-mcp.test/mcp"
            assert server.oauth_client is None
            assert server.auth_server_metadata is None
            assert (
                db.query(MCPUserOAuthToken).filter_by(mcpserver_id=server_id).count()
                == 0
            )
        finally:
            db.close()

    def test_sensitive_config_update_requires_edit_permission(self):
        _admin_headers()
        server_id = _seed_oauth_mcp_server()
        bob_headers = _register_second_user()
        _share_mcp_server(server_id, "bob")

        resp = oauth_client.put(
            f"/api/mcp/servers/{server_id}",
            headers=bob_headers,
            json={"config": {"url": "https://new-mcp.test/mcp"}},
        )
        assert resp.status_code == 403

    def test_name_update_requires_edit_permission(self):
        _admin_headers()
        server_id = _seed_oauth_mcp_server()
        bob_headers = _register_second_user()
        _share_mcp_server(server_id, "bob")

        resp = oauth_client.put(
            f"/api/mcp/servers/{server_id}",
            headers=bob_headers,
            json={"name": "renamed-by-bob"},
        )

        assert resp.status_code == 403


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
            return [], []

        monkeypatch.setattr(
            "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
            fake_load_mcp_tools_as_agent_tools,
        )

        headers = _admin_headers()
        server_id = _seed_oauth_mcp_server()

        resp = oauth_client.get(f"/api/mcp/servers/{server_id}/tools", headers=headers)
        assert resp.status_code == 200, resp.text
        assert resp.json()["tools"] == []

        assert "test-oauth-remote-mcp" in captured
        connection = captured["test-oauth-remote-mcp"]
        assert "oauth_mcp" not in connection
        assert isinstance(connection.get("auth"), OAuthClientProvider)

    def test_tools_preview_returns_reconnect_response(self, monkeypatch):
        """``load_mcp_tools_as_agent_tools`` no longer raises
        ``MCPReauthorizationRequired`` directly (it collects per-server
        failures in ``reauth_failures`` so a batch caller's other, healthy
        servers still get their tools). The single-server tools-preview
        endpoint re-raises the failure itself to preserve this 409
        response."""
        from xagent.core.tools.core.mcp.oauth.errors import (
            MCPReauthorizationRequired,
        )

        async def fake_load_mcp_tools_as_agent_tools(*args, **kwargs):
            return [], [MCPReauthorizationRequired("test-oauth-remote-mcp", 7)]

        monkeypatch.setattr(
            "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
            fake_load_mcp_tools_as_agent_tools,
        )

        headers = _admin_headers()
        server_id = _seed_oauth_mcp_server()

        resp = oauth_client.get(f"/api/mcp/servers/{server_id}/tools", headers=headers)

        assert resp.status_code == 409
        assert resp.json()["detail"]["code"] == "mcp_reauthorization_required"
        assert "reconnect" in resp.json()["detail"]["message"].lower()


@pytest.mark.asyncio
async def test_discover_auth_server_falls_back_to_default_endpoints(monkeypatch):
    from xagent.core.tools.core.mcp.oauth.flow import discover_auth_server

    # Pure HTTP-layer test against a MockTransport with a placeholder
    # ".example" host -- bypass the real-DNS SSRF guard so it doesn't depend
    # on the test environment's DNS behavior for non-existent domains.
    async def noop(url):
        return None

    monkeypatch.setattr(
        "xagent.core.tools.core.mcp.oauth.flow.assert_public_endpoint", noop
    )

    def handler(request):
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        meta = await discover_auth_server("https://mcp.example", client=client)

    assert meta["authorization_endpoint"] == "https://mcp.example/authorize"
    assert meta["token_endpoint"] == "https://mcp.example/token"
    assert meta["registration_endpoint"] == "https://mcp.example/register"
