from mcp.client.auth import OAuthClientProvider

from xagent.web.tools.config import attach_oauth_provider_if_needed


def test_attaches_provider_for_oauth_mcp(db_session, seed_user_and_server):
    user_id, server_id = seed_user_and_server
    conn = {
        "name": "notion",
        "transport": "streamable_http",
        "url": "https://mcp.example/notion",
        "oauth_mcp": True,
    }
    out = attach_oauth_provider_if_needed(
        conn, user_id=user_id, mcpserver_id=server_id, db=db_session
    )
    assert isinstance(out["auth"], OAuthClientProvider)
    assert "oauth_mcp" not in out  # marker consumed


def test_noop_without_marker(db_session, seed_user_and_server):
    user_id, server_id = seed_user_and_server
    conn = {"name": "x", "transport": "streamable_http", "url": "https://x"}
    out = attach_oauth_provider_if_needed(
        conn, user_id=user_id, mcpserver_id=server_id, db=db_session
    )
    assert "auth" not in out


def test_noop_for_websocket_marker(db_session, seed_user_and_server):
    user_id, server_id = seed_user_and_server
    conn = {
        "name": "ws",
        "transport": "websocket",
        "url": "ws://mcp.example/ws",
        "oauth_mcp": True,
    }
    out = attach_oauth_provider_if_needed(
        conn, user_id=user_id, mcpserver_id=server_id, db=db_session
    )
    assert "oauth_mcp" not in out
    assert "auth" not in out


def test_oauth_mcp_requires_url(db_session, seed_user_and_server):
    user_id, server_id = seed_user_and_server
    conn = {
        "name": "missing-url",
        "transport": "streamable_http",
        "oauth_mcp": True,
    }

    try:
        attach_oauth_provider_if_needed(
            conn, user_id=user_id, mcpserver_id=server_id, db=db_session
        )
    except ValueError as exc:
        assert "requires a URL" in str(exc)
    else:
        raise AssertionError("OAuth MCP connections without URL must fail")
