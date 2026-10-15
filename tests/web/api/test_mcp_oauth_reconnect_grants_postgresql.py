"""PostgreSQL checks for a reconnect that replaces an owner's earlier grants.

``uq_mcp_oauth_grants_lookup`` covers every grant row whatever its status, and
the lookup hash includes the granted scope. The callback deletes the grants it
replaces before it looks up the row for the new scope, so neither a later
refresh nor a refresh running at the same time collides with that constraint.
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import timedelta
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

from tests.shared.async_waits import DB_PROGRESS_TIMEOUT
from tests.shared.postgres_disposable import disposable_database_factory
from xagent.core.utils.encryption import decrypt_value, encrypt_value
from xagent.web.api import mcp as mcp_api
from xagent.web.models import MCPOAuthClient, MCPOAuthFlowState, MCPOAuthGrant
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.services import mcp_oauth as mcp_oauth_service
from xagent.web.services.mcp_oauth import resolve_mcp_oauth_runtime_auth

pytestmark = pytest.mark.postgresql

MCP_URL = "https://mcp.example.com/mcp"
ISSUER = "https://auth.example.com"
OWNER = "actor:workspace:alice"
FULL = "files.read records.read records.write"
NARROW = "records.read records.write"
AUTH = {"type": "mcp_oauth", "scope": FULL, "required_scope": "records.read"}
STATE = "postgres-reconnect-state"


@pytest.fixture()
def factory():
    with disposable_database_factory("xagent_mcp_oauth_reconnect") as make:
        engine = make("grants")
        Base.metadata.create_all(
            engine,
            tables=[
                User.__table__,
                MCPServer.__table__,
                UserMCPServer.__table__,
                PublicMCPApp.__table__,
                MCPOAuthClient.__table__,
                MCPOAuthGrant.__table__,
                MCPOAuthFlowState.__table__,
            ],
        )
        yield sessionmaker(bind=engine, autoflush=False, autocommit=False)


@pytest.fixture(autouse=True)
def provider(monkeypatch):
    """Answer the code exchange and keep revocations away from the network."""

    async def exchange_code(**_kwargs):
        return {
            "access_token": "issued-access-token",
            "refresh_token": "issued-refresh-token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": NARROW,
        }

    async def no_revocation(_snapshot):
        return None

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", exchange_code)
    monkeypatch.setattr(
        mcp_api, "_revoke_mcp_oauth_issued_token_externally", no_revocation
    )
    monkeypatch.setattr(
        mcp_api, "_revoke_mcp_oauth_grant_snapshot_externally", no_revocation
    )


def _seed(factory, *, earlier_expired: bool = False) -> dict[str, int]:
    with factory() as db:
        user = User(username="postgres-reconnect-alice", password_hash="x")
        server = MCPServer.from_config(
            {
                "name": "postgres-reconnect-records",
                "managed": "external",
                "transport": "streamable_http",
                "url": MCP_URL,
                "auth": AUTH,
            }
        )
        db.add_all([user, server])
        db.flush()
        association = UserMCPServer(
            user_id=user.id,
            mcpserver_id=server.id,
            is_owner=False,
            is_active=True,
        )
        client = MCPOAuthClient(
            mcp_server_id=server.id,
            issuer=ISSUER,
            authorization_endpoint=f"{ISSUER}/authorize",
            token_endpoint=f"{ISSUER}/token",
            client_id="postgres-client",
            token_endpoint_auth_method="none",
            redirect_uri="https://xagent.example.com/api/mcp/oauth/callback",
        )
        db.add_all([association, client])
        db.flush()
        earlier = MCPOAuthGrant(
            mcp_server_id=server.id,
            user_id=user.id,
            mcp_oauth_client_id=client.id,
            resource_owner_key=OWNER,
            issuer=ISSUER,
            resource=MCP_URL,
            scope=FULL,
            access_token=encrypt_value("earlier-access-token"),
            refresh_token=encrypt_value("earlier-refresh-token"),
            token_type="Bearer",
            status="active",
            expires_at=mcp_api._utc_now()
            + (timedelta(minutes=-1) if earlier_expired else timedelta(hours=1)),
        )
        flow = MCPOAuthFlowState(
            state=STATE,
            mcp_server_id=server.id,
            user_id=user.id,
            association_lifecycle_generation=association.lifecycle_generation,
            mcp_oauth_client_id=client.id,
            resource_owner_key=OWNER,
            issuer=ISSUER,
            resource=MCP_URL,
            scope=FULL,
            code_verifier=encrypt_value("postgres-verifier"),
            redirect_after="/tools",
            expires_at=mcp_api._utc_now() + timedelta(minutes=10),
        )
        db.add_all([earlier, flow])
        db.commit()
        return {
            "user_id": int(user.id),
            "server_id": int(server.id),
            "earlier_id": int(earlier.id),
        }


def _callback_request() -> Request:
    cookie = (
        f"{mcp_api.MCP_OAUTH_STATE_COOKIE}="
        f"{mcp_api._mcp_oauth_state_cookie_value(STATE)}"
    )
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/mcp/oauth/callback",
            "query_string": f"code=provider-code&state={STATE}".encode(),
            "headers": [(b"cookie", cookie.encode())],
        }
    )


def _reconnect(factory) -> dict[str, list[str]]:
    with factory() as db:
        response = asyncio.run(mcp_api.mcp_oauth_callback(_callback_request(), db))
    return parse_qs(urlparse(response.headers["location"]).query)


def _owner_grants(factory) -> list[tuple[int, str, str, str]]:
    with factory() as db:
        return [
            (
                int(grant.id),
                grant.scope,
                grant.status,
                decrypt_value(str(grant.access_token)),
            )
            for grant in db.query(MCPOAuthGrant)
            .filter(MCPOAuthGrant.resource_owner_key == OWNER)
            .order_by(MCPOAuthGrant.id)
        ]


def _refresh_returning(monkeypatch, scope: str) -> None:
    async def fake_refresh(_client, *, refresh_token, resource, client=None):
        return {
            "access_token": "refreshed-access-token",
            "refresh_token": "rotated-refresh-token",
            "expires_in": 3600,
            "scope": scope,
        }

    monkeypatch.setattr(mcp_oauth_service, "_refresh_mcp_oauth_grant", fake_refresh)


def _resolve(factory, seed: dict[str, int]) -> Any:
    with factory() as db:
        return asyncio.run(
            resolve_mcp_oauth_runtime_auth(
                db,
                server_id=seed["server_id"],
                user_id=seed["user_id"],
                auth_config=AUTH,
                resource_owner_key=OWNER,
            )
        )


def test_refresh_back_to_the_replaced_scope_does_not_collide(factory, monkeypatch):
    seed = _seed(factory)

    assert _reconnect(factory) == {"mcp_oauth_success": ["1"]}
    [(grant_id, scope, status, _token)] = _owner_grants(factory)
    assert (scope, status) == (NARROW, "active")
    assert grant_id != seed["earlier_id"]

    with factory() as db:
        db.get(MCPOAuthGrant, grant_id).expires_at = mcp_api._utc_now() - timedelta(
            minutes=1
        )
        db.commit()
    _refresh_returning(monkeypatch, FULL)
    runtime_auth = _resolve(factory, seed)

    assert (runtime_auth.grant_id, runtime_auth.scope) == (grant_id, FULL)
    assert runtime_auth.refreshed is True
    assert _owner_grants(factory) == [
        (grant_id, FULL, "active", "refreshed-access-token")
    ]


def _wait_for_a_lock_waiter(factory) -> None:
    deadline = time.monotonic() + DB_PROGRESS_TIMEOUT
    with factory() as observer:
        while time.monotonic() < deadline:
            waiting = observer.execute(
                text(
                    "SELECT count(*) FROM pg_stat_activity "
                    "WHERE datname = current_database() AND wait_event_type = 'Lock'"
                )
            ).scalar_one()
            observer.rollback()
            if waiting:
                return
            time.sleep(0.01)
    raise AssertionError("the callback never waited for the refreshing grant")


@pytest.mark.parametrize(
    ("refreshed_scope", "keeps_the_earlier_row"),
    (
        pytest.param(NARROW, True, id="refresh-moves-it-to-the-new-scope"),
        pytest.param(FULL, False, id="refresh-keeps-its-scope"),
    ),
)
def test_reconnect_waits_for_a_refresh_of_the_grant_it_replaces(
    factory, monkeypatch, refreshed_scope, keeps_the_earlier_row
):
    """The refresh holds the earlier grant's row lock. The callback's delete
    waits for it and then sees the refreshed row: one with the new scope is
    the row the callback updates, any other is deleted."""
    seed = _seed(factory, earlier_expired=True)
    _refresh_returning(monkeypatch, refreshed_scope)
    refresh_has_the_row = threading.Event()
    finish_refresh = threading.Event()
    real_expires_at = mcp_oauth_service.oauth_token_expires_at

    def hold_the_row(token_data):
        if threading.current_thread().name == "postgres-refresh":
            refresh_has_the_row.set()
            assert finish_refresh.wait(timeout=DB_PROGRESS_TIMEOUT)
        return real_expires_at(token_data)

    monkeypatch.setattr(mcp_oauth_service, "oauth_token_expires_at", hold_the_row)
    results: dict[str, Any] = {}
    errors: list[BaseException] = []

    def run(name: str, work) -> threading.Thread:
        def target() -> None:
            try:
                results[name] = work()
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=target, name=f"postgres-{name}")
        thread.start()
        return thread

    refresh = run("refresh", lambda: _resolve(factory, seed))
    assert refresh_has_the_row.wait(timeout=DB_PROGRESS_TIMEOUT)
    callback = run("callback", lambda: _reconnect(factory))
    try:
        _wait_for_a_lock_waiter(factory)
    finally:
        finish_refresh.set()
        refresh.join(timeout=DB_PROGRESS_TIMEOUT)
        callback.join(timeout=DB_PROGRESS_TIMEOUT)

    assert errors == []
    assert not refresh.is_alive() and not callback.is_alive()
    assert results["refresh"].grant_id == seed["earlier_id"]
    assert results["callback"] == {"mcp_oauth_success": ["1"]}
    [(grant_id, scope, status, token)] = _owner_grants(factory)
    assert (scope, status, token) == (NARROW, "active", "issued-access-token")
    assert (grant_id == seed["earlier_id"]) is keeps_the_earlier_row
