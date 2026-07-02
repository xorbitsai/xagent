"""End-to-end integration test for the OAuth remote-MCP connector.

Exercises the seam that matters at execution time: a connected user's token
is attached to outbound requests by the real SDK ``OAuthClientProvider``
(backed by the real ``DBTokenStorage``), and when no usable token exists the
provider's interactive fallback surfaces ``MCPReauthorizationRequired`` (the
"reconnect" signal) instead of hanging or raising an opaque SDK error.

Approach (see Step 0 investigation): ``OAuthClientProvider.async_auth_flow``
is an ``httpx.Auth`` async generator. It is driven by yielding a request,
sending it over a transport, and feeding the response back in via
``asend()``. On a 401 from the protected resource it walks through RFC 9728
protected-resource discovery, RFC 8414 authorization-server discovery, and
dynamic client registration (DCR) before finally calling
``context.redirect_handler`` / ``context.callback_handler`` to obtain an
authorization code. Our ``build_execution_oauth_provider`` wires those two
handlers to always raise ``MCPReauthorizationRequired`` (there is no
interactive user mid-execution), and the SDK's blanket
``except Exception: raise`` inside ``async_auth_flow`` re-raises it
unchanged, so it propagates directly out of ``flow.asend(...)`` /
``flow.__anext__()`` to the caller. This test drives the flow for real
against a self-contained ``httpx.MockTransport`` (no network) and asserts
that exact, real, un-mocked-provider behavior.
"""

from __future__ import annotations

import httpx
import pytest

from mcp.shared.auth import OAuthToken

from xagent.core.tools.core.mcp.oauth.errors import MCPReauthorizationRequired
from xagent.core.tools.core.mcp.oauth.provider import build_execution_oauth_provider
from xagent.core.tools.core.mcp.oauth.token_storage import DBTokenStorage

SERVER_URL = "https://mcp.example/notion"


async def _drive(flow, transport: httpx.MockTransport, first_request: httpx.Request):
    """Drive an ``async_auth_flow`` generator to completion against a mock
    transport, returning the list of requests actually sent over the wire."""
    sent: list[httpx.Request] = []
    async with httpx.AsyncClient(transport=transport) as client:
        request = first_request
        while True:
            sent.append(request)
            response = await client.send(request)
            try:
                request = await flow.asend(response)
            except StopAsyncIteration:
                break
    return sent


@pytest.mark.asyncio
async def test_connected_token_is_used_without_interactive_auth(
    db_session, seed_user_and_server
):
    """(A) A connected token is attached as a Bearer Authorization header,
    and the provider never needs to touch the redirect/callback handlers."""
    user_id, server_id = seed_user_and_server

    storage = DBTokenStorage(user_id=user_id, mcpserver_id=server_id, db=db_session)
    await storage.set_tokens(OAuthToken(access_token="AT", token_type="Bearer"))

    provider = build_execution_oauth_provider(
        server_url=SERVER_URL,
        server_name="notion",
        user_id=user_id,
        mcpserver_id=server_id,
        db=db_session,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        # The real resource server: success means the provider never had to
        # fall back to discovery/registration/interactive auth.
        assert request.headers.get("authorization") == "Bearer AT"
        return httpx.Response(200)

    original_request = httpx.Request("GET", SERVER_URL)
    flow = provider.async_auth_flow(original_request)

    first_request = await flow.__anext__()
    assert first_request.headers.get("authorization") == "Bearer AT"

    sent = await _drive(flow, httpx.MockTransport(handler), first_request)
    assert (
        len(sent) == 1
    )  # only the resource request; no discovery/DCR/auth round trips


@pytest.mark.asyncio
async def test_reauth_required_when_no_connected_token(
    db_session, seed_user_and_server
):
    """(B) With no connected token, the resource server's 401 drives the
    provider through protected-resource discovery, auth-server discovery,
    and dynamic client registration, at which point it needs interactive
    authorization -- our wired handlers raise MCPReauthorizationRequired,
    and that exception propagates out of the auth flow unchanged."""
    user_id, server_id = seed_user_and_server

    # No DBTokenStorage.set_tokens() call: get_tokens() will return None.
    provider = build_execution_oauth_provider(
        server_url=SERVER_URL,
        server_name="notion",
        user_id=user_id,
        mcpserver_id=server_id,
        db=db_session,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/notion":
            return httpx.Response(401)
        if path == "/.well-known/oauth-protected-resource":
            return httpx.Response(
                200,
                json={
                    "resource": SERVER_URL,
                    "authorization_servers": ["https://mcp.example"],
                },
            )
        if path in (
            "/.well-known/oauth-authorization-server",
            "/.well-known/openid-configuration",
        ):
            # No separate authorization server metadata: fall back to the
            # provider's default /authorize, /token, /register endpoints.
            return httpx.Response(404)
        if path == "/register":
            return httpx.Response(
                201,
                json={
                    "client_id": "test-client-id",
                    "redirect_uris": ["https://localhost/callback"],
                },
            )
        return httpx.Response(404)

    original_request = httpx.Request("GET", SERVER_URL)
    flow = provider.async_auth_flow(original_request)

    first_request = await flow.__anext__()
    # No token stored, so the provider does not attach an Authorization header.
    assert "authorization" not in first_request.headers

    with pytest.raises(MCPReauthorizationRequired) as exc_info:
        await _drive(flow, httpx.MockTransport(handler), first_request)

    assert exc_info.value.server_name == "notion"
    assert exc_info.value.mcpserver_id == server_id

    # Confirm the seam directly: the storage the provider reads from has no
    # usable token for this (user, server) pair.
    storage = DBTokenStorage(user_id=user_id, mcpserver_id=server_id, db=db_session)
    assert await storage.get_tokens() is None
