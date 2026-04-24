from xagent.web.api.mcp import _build_server_config
from xagent.web.schemas.model import MCPServerCreate

try:
    c = MCPServerCreate(
        name="test", transport="sse", config={"url": "http://1", "client_id": "123"}
    )
    print(_build_server_config(c))
except Exception as e:
    print(e)
