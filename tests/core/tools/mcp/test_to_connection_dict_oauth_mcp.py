from xagent.web.models.mcp import MCPServer


def test_oauth_mcp_does_not_bake_authorization_header():
    s = MCPServer(
        name="notion",
        managed="external",
        transport="streamable_http",
        url="https://mcp.example/notion",
        auth={"type": "oauth_mcp"},
    )
    conn = s.to_connection_dict()
    headers = conn.get("headers") or {}
    assert "Authorization" not in headers
    # marker tells the loader to attach an OAuthClientProvider
    assert conn.get("oauth_mcp") is True


def test_bearer_still_bakes_header():
    # regression guard: existing bearer behavior unchanged
    s = MCPServer(
        name="b",
        managed="external",
        transport="streamable_http",
        url="https://x",
        auth={"type": "bearer", "bearer_token": "T"},
    )
    conn = s.to_connection_dict()
    assert conn["headers"]["Authorization"] == "Bearer T"
    assert "oauth_mcp" not in conn
