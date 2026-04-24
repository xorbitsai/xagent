from xagent.core.tools.core.mcp.data_config import MCPServerConfig

try:
    c = MCPServerConfig(
        name="test",
        transport="sse",
        url="http://1",
        client_id="123",
        managed="external",
    )
    print(c.model_dump())
except Exception as e:
    print(e)
