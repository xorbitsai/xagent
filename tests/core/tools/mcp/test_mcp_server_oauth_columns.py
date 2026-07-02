from xagent.web.models.mcp import MCPServer


def test_oauth_client_columns_exist():
    cols = set(MCPServer.__table__.columns.keys())
    assert "oauth_client" in cols
    assert "auth_server_metadata" in cols
