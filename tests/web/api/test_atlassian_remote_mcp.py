"""The remote Atlassian connector requests a fixed scope subset.

Atlassian's protected-resource metadata lists more scopes than fit in one MCP
OAuth scope value, so the catalog row pins a subset in ``auth.scope``. These
tests drive discovery and dynamic client registration against a copy of the
vendor's public metadata (fetched 2026-10-13) instead of stubbing the whole
connect step, so the scope check runs for real.
"""

from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.api import mcp as mcp_api
from xagent.web.api.mcp import MCPOAuthConnectRequest
from xagent.web.builtin_mcp_registry import (
    ATLASSIAN_MCP_OAUTH_SCOPE,
    ATLASSIAN_MCP_OAUTH_SCOPES,
    get_builtin_public_mcp_app,
)
from xagent.web.mcp_apps import (
    RemoteOAuthServerDefinitionError,
    classify_actor_remote_oauth_server,
)
from xagent.web.models import MCPOAuthFlowState
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.services import mcp_oauth as mcp_oauth_service
from xagent.web.services.mcp_oauth import (
    MCP_OAUTH_SCOPE_MAX_LENGTH,
    MCPOAuthDiscoveryError,
    normalize_mcp_oauth_scope,
)

MCP_URL = "https://mcp.atlassian.com/v2/mcp"
PRM_URL = "https://mcp.atlassian.com/.well-known/oauth-protected-resource/v2/mcp"
ISSUER = "https://auth.atlassian.com/VCeDsk8ZHncYF1g234fKtc4lNipbBhu3"
AS_METADATA_URL = (
    "https://auth.atlassian.com/.well-known/oauth-authorization-server/"
    "VCeDsk8ZHncYF1g234fKtc4lNipbBhu3"
)
REGISTRATION_URL = f"{ISSUER}/dcr/register"

# scopes_supported from Atlassian's public protected-resource metadata, in the
# vendor's order (anonymous GET of PRM_URL on 2026-10-13).
ATLASSIAN_SCOPES_SUPPORTED = (
    "read:me",
    "read:account",
    "offline_access",
    "email",
    "read:jira:agent-interface",
    "write:jira:agent-interface",
    "search:jira:agent-interface",
    "delete:jira:agent-interface",
    "manage:jira:agent-interface",
    "read:confluence:agent-interface",
    "write:confluence:agent-interface",
    "search:confluence:agent-interface",
    "search:rovo:agent-interface",
    "search:code:agent-interface",
    "read:all:twg",
    "write:all:twg",
    "read:goals:agent-interface",
    "write:goals:agent-interface",
    "read:projects:agent-interface",
    "write:projects:agent-interface",
    "read:bitbucket:agent-interface",
    "write:bitbucket:agent-interface",
    "read:loom:agent-interface",
    "write:loom:agent-interface",
    "read:talent:agent-interface",
    "write:talent:agent-interface",
    "read:jira-align:agent-interface",
    "write:jira-align:agent-interface",
    "read:teams:agent-interface",
    "write:teams:agent-interface",
    "read:artifacts:agent-interface",
    "write:artifacts:agent-interface",
    "read:capacity-planning:agent-interface",
    "write:capacity-planning:agent-interface",
    "read:focus:agent-interface",
    "write:focus:agent-interface",
    "read:assets:agent-interface",
    "write:assets:agent-interface",
)
# Atlassian grants only the scopes of the products a site has, so a grant must
# include just the basic identity scopes; the rest of the subset is requested
# but optional.
REQUIRED_SCOPE = "email offline_access read:account read:me"
PROTECTED_RESOURCE_METADATA = {
    "resource": MCP_URL,
    "authorization_servers": [ISSUER],
    "scopes_supported": list(ATLASSIAN_SCOPES_SUPPORTED),
    "bearer_methods_supported": ["header"],
    "resource_documentation": "https://www.atlassian.com/platform/remote-mcp-server",
}
AUTHORIZATION_SERVER_METADATA = {
    "issuer": ISSUER,
    "authorization_endpoint": "https://auth.atlassian.com/authorize",
    "token_endpoint": "https://auth.atlassian.com/oauth/token",
    "registration_endpoint": REGISTRATION_URL,
    "code_challenge_methods_supported": ["S256"],
    "response_types_supported": ["code"],
    "grant_types_supported": ["authorization_code", "refresh_token"],
    "token_endpoint_auth_methods_supported": ["none"],
}


def _vendor_handler(requests: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        url = str(request.url)
        if request.method == "GET" and url == MCP_URL:
            # The vendor's challenge names the metadata but carries no scope.
            return httpx.Response(
                401,
                headers={"WWW-Authenticate": f'Bearer resource_metadata="{PRM_URL}"'},
            )
        if request.method == "GET" and url == PRM_URL:
            return httpx.Response(200, json=PROTECTED_RESOURCE_METADATA)
        if request.method == "GET" and url == AS_METADATA_URL:
            return httpx.Response(200, json=AUTHORIZATION_SERVER_METADATA)
        if request.method == "POST" and url == REGISTRATION_URL:
            return httpx.Response(
                201,
                json={
                    "client_id": "atlassian-dynamic-client",
                    "token_endpoint_auth_method": "none",
                },
            )
        return httpx.Response(404)

    return handler


@pytest.fixture()
def vendor_requests(monkeypatch) -> list[httpx.Request]:
    requests: list[httpx.Request] = []
    handler = _vendor_handler(requests)
    monkeypatch.setattr(
        mcp_oauth_service,
        "create_mcp_oauth_http_client",
        lambda **_kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.xagent.test/")
    return requests


@pytest.fixture()
def db_session(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'atlassian.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    user = User(username="alice", password_hash="x", is_admin=False)
    session.add(user)
    session.commit()
    session.refresh(user)
    yield session, user
    session.close()
    engine.dispose()


def _add_atlassian_catalog_row(db) -> dict:
    app = get_builtin_public_mcp_app("atlassian")
    assert app is not None
    db.add(
        PublicMCPApp(
            app_id=app["app_id"],
            name=app["name"],
            description=app["description"],
            icon=app["icon"],
            transport=app["transport"],
            provider_name=app["provider_name"],
            category=app["category"],
            oauth_scopes=app["oauth_scopes"],
            launch_config=app["launch_config"],
            is_visible_in_connector=app["is_visible_in_connector"],
        )
    )
    db.commit()
    return app


def test_requesting_every_advertised_scope_exceeds_the_scope_limit():
    """Why the catalog pins a subset: the full list is 1023 characters."""
    assert len(ATLASSIAN_SCOPES_SUPPORTED) == 38
    assert len(" ".join(sorted(ATLASSIAN_SCOPES_SUPPORTED))) == 1023

    with pytest.raises(MCPOAuthDiscoveryError) as exc_info:
        normalize_mcp_oauth_scope(ATLASSIAN_SCOPES_SUPPORTED)

    assert exc_info.value.code == "invalid_scope"


def test_catalog_scope_is_a_short_canonical_subset_of_the_advertised_scopes():
    requested = ATLASSIAN_MCP_OAUTH_SCOPE.split()

    assert set(requested) <= set(ATLASSIAN_SCOPES_SUPPORTED)
    assert len(requested) == len(set(ATLASSIAN_MCP_OAUTH_SCOPES))
    assert len(ATLASSIAN_MCP_OAUTH_SCOPE) < MCP_OAUTH_SCOPE_MAX_LENGTH
    # Already in the canonical form, so the catalog, the server row and the
    # persisted flow scope all hold the same string.
    assert normalize_mcp_oauth_scope(ATLASSIAN_MCP_OAUTH_SCOPE) == (
        ATLASSIAN_MCP_OAUTH_SCOPE
    )
    # Identity and refresh are always requested; destructive scopes never are.
    assert {"read:me", "offline_access"} <= set(requested)
    assert not [
        scope for scope in requested if scope.startswith(("delete:", "manage:"))
    ]
    # Only the basic identity scopes are required of a grant; they are part of
    # the request and kept in the same canonical form.
    assert set(REQUIRED_SCOPE.split()) < set(requested)
    assert normalize_mcp_oauth_scope(REQUIRED_SCOPE) == REQUIRED_SCOPE

    app = get_builtin_public_mcp_app("atlassian")
    assert app is not None
    assert app["launch_config"]["auth"] == {
        "type": "mcp_oauth",
        "scope": ATLASSIAN_MCP_OAUTH_SCOPE,
        "required_scope": REQUIRED_SCOPE,
    }


@pytest.mark.asyncio
async def test_without_the_pinned_scope_discovery_selects_the_whole_list(
    vendor_requests,
):
    """Control for the connect test: discovery returns every advertised
    scope (the challenge has none), which the connect rejects as too long."""
    discovery = await mcp_oauth_service.discover_mcp_oauth_metadata(MCP_URL)

    assert discovery.scopes == ATLASSIAN_SCOPES_SUPPORTED
    with pytest.raises(HTTPException) as exc_info:
        mcp_api._scope_string(discovery.scopes)
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "invalid_scope"


def test_a_server_row_with_the_previous_catalog_auth_is_not_canonical(db_session):
    """A shared row created before the catalog pinned the scope no longer
    matches the catalog, so the actor projection rejects it until its auth is
    brought up to date: by the migration (20261013_atlassian_mcp_scope) or by
    the next connect, which syncs the row's auth in place."""
    db, _user = db_session
    app = _add_atlassian_catalog_row(db)
    db.add(
        MCPServer.from_config(
            {
                "name": "atlassian",
                "managed": "external",
                "transport": app["transport"],
                "url": MCP_URL,
                "auth": {"type": "mcp_oauth"},
            }
        )
    )
    db.commit()
    server = db.query(MCPServer).filter(MCPServer.name == "atlassian").one()

    with pytest.raises(RemoteOAuthServerDefinitionError, match="auth"):
        classify_actor_remote_oauth_server(db, server)


@pytest.mark.asyncio
async def test_connect_requests_the_subset_and_keeps_the_required_scope_on_the_row(
    db_session, vendor_requests
):
    """The catalog's required_scope only narrows what a grant must include:
    the authorization still requests the whole subset, registration carries no
    scope, and the shared row keeps the catalog auth exactly, so the actor
    projection still lists the app."""
    db, user = db_session
    app = _add_atlassian_catalog_row(db)
    catalog_auth = app["launch_config"]["auth"]
    assert catalog_auth["required_scope"] == (
        "email offline_access read:account read:me"
    )

    response = await mcp_api.connect_mcp_oauth_app_for_owner(
        "atlassian",
        MCPOAuthConnectRequest(redirect_after="/settings/mcp"),
        user,
        db,
        resource_owner_key="actor:workspace:alice",
        accept="application/json",
    )

    assert response.status_code == 200
    authorization_url = json.loads(response.body)["authorization_url"]
    query = parse_qs(urlparse(authorization_url).query)
    assert query["scope"] == [ATLASSIAN_MCP_OAUTH_SCOPE]
    assert "required_scope" not in query
    assert query["client_id"] == ["atlassian-dynamic-client"]
    assert query["resource"] == [MCP_URL]
    assert [(r.method, str(r.url)) for r in vendor_requests] == [
        ("GET", MCP_URL),
        ("GET", PRM_URL),
        ("GET", AS_METADATA_URL),
        ("POST", REGISTRATION_URL),
    ]
    registration = next(r for r in vendor_requests if r.method == "POST")
    registration_body = json.loads(registration.content)
    assert "scope" not in registration_body
    assert "required_scope" not in registration_body
    assert db.query(MCPOAuthFlowState).one().scope == ATLASSIAN_MCP_OAUTH_SCOPE

    server = db.query(MCPServer).filter(MCPServer.name == "atlassian").one()
    assert server.auth == catalog_auth
    app_info = classify_actor_remote_oauth_server(db, server)
    assert app_info is not None
    assert app_info["id"] == "atlassian"
