"""OAuth-MCP connector errors."""

from typing import Optional


class MCPReauthorizationRequired(RuntimeError):
    """Raised at execution time when a connector's token is missing or cannot
    be refreshed, so the user must reconnect the server from settings."""

    def __init__(self, server_name: str, mcpserver_id: Optional[int] = None) -> None:
        self.server_name = server_name
        self.mcpserver_id = mcpserver_id
        super().__init__(
            f"MCP server '{server_name}' requires authorization. "
            f"Please reconnect it in settings."
        )
