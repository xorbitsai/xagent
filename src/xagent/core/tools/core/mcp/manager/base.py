import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Union

from ..data_config import MCPServerConfig, MCPServerData

logger = logging.getLogger(__name__)


class MCPServerManager(Protocol):
    """
    Protocol defining the interface for MCP Server Manager implementations.

    This uses Python's Protocol for structural subtyping, allowing any class
    that implements these methods to be used as an MCPServerManager without
    explicit inheritance.
    """

    def load_config(self) -> Dict[str, Dict[str, Any]]:
        """
        Load MCP server configurations.

        Returns:
            Dictionary with 'servers' key containing server configurations.
        """
        ...

    def save_config(self) -> None:
        """Save MCP server configurations to storage."""
        ...

    def get_connections(
        self, filter_func: Optional[Callable[..., Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get MCP server connections in the format expected by MCP tools.

        Args:
            filter_func: Optional filter function for implementations that support it.

        Returns:
            Dictionary mapping server names to connection configurations.
        """
        ...

    def add_server(self, config: MCPServerConfig) -> None:
        """
        Add a new MCP server configuration.

        Args:
            config: MCPServerConfig object to add.

        Raises:
            ValueError: If server name already exists.
        """
        ...

    def add_server_from_request(self, request_data: Dict[str, Any]) -> MCPServerConfig:
        """
        Add a new MCP server configuration from request data.

        Args:
            request_data: Dictionary containing server configuration data.

        Returns:
            MCPServerConfig: The created configuration object.

        Raises:
            KeyError: If required parameters are missing.
            ValueError: If server name already exists or validation fails.
        """
        ...

    def remove_server(self, name: str) -> bool:
        """
        Remove an MCP server configuration.

        Args:
            name: Server name to remove.

        Returns:
            True if server was removed, False if it didn't exist.
        """
        ...

    def list_servers(self) -> List[MCPServerData]:
        """
        List all configured MCP servers with updated status.

        Returns:
            List of MCPServerData objects.
        """
        ...

    def get_server(self, name: str) -> Optional[MCPServerData]:
        """
        Get configuration and up to date status for a specific server.

        Args:
            name: Server name to retrieve.

        Returns:
            MCPServerData object or None if not found.
        """
        ...

    def start_server(self, name: str) -> bool:
        """
        Start a server (only for internal managed servers).

        Args:
            name: Server name to start.

        Returns:
            True if successfully started.

        Raises:
            ValueError: If server is external or not found.
        """
        ...

    def stop_server(self, name: str) -> bool:
        """
        Stop a server (only for internal managed servers).

        Args:
            name: Server name to stop.

        Returns:
            True if successfully stopped.

        Raises:
            ValueError: If server is external or not found.
        """
        ...

    def restart_server(self, name: str) -> bool:
        """
        Restart a server (only for internal managed servers).

        Args:
            name: Server name to restart.

        Returns:
            True if successfully restarted.

        Raises:
            ValueError: If server is external or not found.
        """
        ...

    def get_logs(self, name: str, lines: int = 100) -> Optional[List[str]]:
        """
        Get logs for a server (only for internal managed servers).

        Args:
            name: Server name.
            lines: Number of log lines to retrieve (1-1000).

        Returns:
            List of log lines or None if not available.
        """
        ...

    def create_config(
        self,
        name: str,
        transport: Literal["stdio", "sse", "websocket", "streamable_http"],
        managed: Literal["internal", "external"] = "external",
        description: Optional[str] = None,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[Union[str, Path]] = None,
        url: Optional[str] = None,
        headers: Optional[Dict[str, Any]] = None,
        docker_image: Optional[str] = None,
        docker_url: Optional[str] = None,
        bind_ports: Optional[Dict[str, Union[int, str]]] = None,
        volumes: Optional[List[str]] = None,
        restart_policy: str = "no",
        **kwargs: Any,
    ) -> MCPServerConfig:
        """
        Create a unified MCP server configuration for any transport type.

        Args:
            name: Server name/identifier
            transport: Transport type (stdio, sse, websocket, streamable_http)
            managed: Whether server is internally or externally managed
            description: Optional description
            command: Command to run (stdio transport)
            args: Command arguments (stdio transport)
            env: Environment variables
            cwd: Working directory (stdio transport)
            url: Server URL (HTTP-based transports)
            headers: HTTP headers (HTTP-based transports)
            docker_image: Docker image (internal managed only)
            docker_url: Docker daemon URL (internal managed only)
            bind_ports: Docker port bindings (internal managed only)
            volumes: Docker volume mappings (internal managed only)
            restart_policy: Docker restart policy (internal managed only)
            **kwargs: Additional configuration parameters

        Returns:
            MCPServerConfig object.

        Raises:
            ValueError: If required parameters for the transport type are missing.
        """
        ...
