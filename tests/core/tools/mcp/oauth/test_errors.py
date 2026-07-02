from xagent.core.tools.core.mcp.oauth.errors import MCPReauthorizationRequired


def test_carries_server_identity():
    err = MCPReauthorizationRequired(server_name="notion", mcpserver_id=7)
    assert err.server_name == "notion"
    assert err.mcpserver_id == 7
    assert "reconnect" in str(err).lower()
