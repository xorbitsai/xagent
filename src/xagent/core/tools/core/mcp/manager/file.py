import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import yaml

from ..data_config import (
    ContainerLogs,
    ContainerStatus,
    MCPServerConfig,
    MCPServerData,
    MCPServerStatus,
)
from ..docker_manager import DockerManager

logger = logging.getLogger(__name__)


class YamlMCPServerManager:
    """
    YAML-based MCP Server Manager implementation.

    Stores MCP server configurations in a YAML file.
    Implements the MCPServerManager protocol.
    """

    def __init__(self, storage_root: str) -> None:
        """
        Initialize the YAML-based MCP server manager.

        Args:
            storage_root: Root directory where MCP configuration files are stored.
        """
        self.storage_root = Path(storage_root)
        self.config_file = self.storage_root / "mcp_servers.yaml"
        self._servers: Dict[str, MCPServerData] = {}
        self._docker_manager = DockerManager()
        self._ensure_config_directory()
        self.load_config()

    def __del__(self) -> None:
        """Cleanup Docker clients on destruction."""
        if hasattr(self, "_docker_manager"):
            self._docker_manager.close_all()

    def _ensure_config_directory(self) -> None:
        """Ensure the configuration directory exists."""
        self.storage_root.mkdir(parents=True, exist_ok=True)

    def _serialize_config_dict(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively convert Path objects and other non-serializable types to strings
        for YAML serialization.

        Args:
            config_dict: Configuration dictionary to serialize.

        Returns:
            Dictionary with Path objects converted to strings.
        """

        def convert_value(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_value(item) for item in value]
            else:
                return value

        return {k: convert_value(v) for k, v in config_dict.items()}

    def load_config(self) -> Dict[str, Dict[str, Any]]:
        """Load MCP server configurations from YAML file."""
        if not self.config_file.exists():
            self._servers = {}
            return {"servers": {}}

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                servers_config: Dict[str, Dict[str, Any]] = data.get("servers", {})

                if not isinstance(servers_config, dict):
                    logger.error(
                        f"Invalid servers configuration: expected dictionary, got {type(servers_config).__name__}"
                    )
                    self._servers = {}
                    return {"servers": {}}

                self._servers = {}
                for name, config_dict in servers_config.items():
                    try:
                        if not isinstance(config_dict, dict):
                            logger.error(
                                f"Failed to load server config '{name}': configuration must be a dictionary, got {type(config_dict).__name__}"
                            )
                            continue

                        config_dict["name"] = name
                        config = MCPServerConfig.model_validate(config_dict)
                        status = MCPServerStatus(
                            status=ContainerStatus.UNKNOWN, last_check=datetime.now()
                        )
                        server_data = MCPServerData(
                            config=config, status=status, container_info=None
                        )
                        self._servers[name] = server_data
                    except Exception as e:
                        logger.error(f"Failed to load server config '{name}': {e}")

                logger.info(
                    f"Loaded {len(self._servers)} MCP server configurations from YAML"
                )
                return data

        except (yaml.YAMLError, IOError) as e:
            logger.error(f"Failed to load MCP configuration: {e}")
            self._servers = {}
            return {"servers": {}}

    def save_config(self) -> None:
        """Save MCP server configurations to YAML file."""
        try:
            servers_config = {}
            for name, server_data in self._servers.items():
                config_dict = server_data.config.model_dump(
                    exclude={"name"}, exclude_none=True
                )
                config_dict = self._serialize_config_dict(config_dict)
                servers_config[name] = config_dict

            data = {"servers": servers_config}
            with open(self.config_file, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=True)
        except (yaml.YAMLError, IOError) as e:
            raise ValueError(f"Failed to save MCP configuration: {e}") from e

    def get_connections(
        self, filter_func: Optional[Callable[..., Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get all server configurations in Connection format.

        Args:
            filter_func: Not used in YAML implementation (kept for protocol compatibility).

        Returns:
            Dictionary mapping server names to connection configurations.
        """
        connections = {}
        for name, server_data in self._servers.items():
            connections[name] = server_data.config.to_connection()
        return connections

    def add_server(self, config: MCPServerConfig) -> None:
        """Add a new MCP server configuration."""
        if config.name in self._servers:
            raise ValueError(f"Server '{config.name}' already exists")

        status = MCPServerStatus(
            status=ContainerStatus.UNKNOWN, last_check=datetime.now()
        )
        server_data = MCPServerData(config=config, status=status, container_info=None)

        self._servers[config.name] = server_data
        self.save_config()
        logger.info(f"Added server '{config.name}'")

    def add_server_from_request(self, request_data: Dict[str, Any]) -> MCPServerConfig:
        """Add a new MCP server configuration from request data."""
        if "name" not in request_data:
            raise KeyError("Required parameter 'name' is missing from request data")
        if "transport" not in request_data:
            raise KeyError(
                "Required parameter 'transport' is missing from request data"
            )

        known_params = {
            "name",
            "transport",
            "managed",
            "description",
            "command",
            "args",
            "env",
            "cwd",
            "url",
            "headers",
            "concurrency_safe",
            "concurrent_tools",
            "docker_image",
            "docker_url",
            "volumes",
            "restart_policy",
            "bind_ports",
        }

        config_kwargs = {}
        extra_kwargs = {}

        for key, value in request_data.items():
            if key in known_params:
                config_kwargs[key] = value
            else:
                extra_kwargs[key] = value

        config_kwargs.setdefault("managed", "external")
        config_kwargs.setdefault("restart_policy", "no")

        config = self.create_config(**config_kwargs, **extra_kwargs)
        self.add_server(config)
        return config

    def remove_server(self, name: str) -> bool:
        """Remove an MCP server configuration."""
        if name not in self._servers:
            return False

        del self._servers[name]
        self.save_config()
        logger.info(f"Removed server '{name}'")
        return True

    def list_servers(self) -> List[MCPServerData]:
        """List all configured MCP servers with updated status."""
        servers_data = list(self._servers.values())
        for server_data in servers_data:
            self._update_server_status(server_data)
        return servers_data

    def get_server(self, name: str) -> Optional[MCPServerData]:
        """Get configuration and up to date status for a specific server."""
        server_data = self._servers.get(name)
        if server_data:
            self._update_server_status(server_data)
        return server_data

    def _update_server_status(self, server_data: MCPServerData) -> None:
        """Update the status of a server data object."""
        config = server_data.config

        if config.managed == "internal":
            status, resource_usage, uptime = self._docker_manager.get_container_status(
                config.name, config.docker_url
            )
            server_data.status.status = status
            server_data.status.resource_usage = resource_usage
            server_data.status.uptime = uptime
        elif config.managed == "external":
            if config.url:
                status, health_message = (
                    self._docker_manager.check_external_server_health(config.url)
                )
                server_data.status.status = status
                server_data.status.health_status = health_message
            else:
                server_data.status.status = ContainerStatus.UNKNOWN

        server_data.status.update_check_time()

    def start_server(self, name: str) -> bool:
        """Start a server (only for internal managed servers)."""
        server_data = self.get_server(name)
        if not server_data:
            raise ValueError(f"Server '{name}' not found")

        if server_data.config.managed == "external":
            raise ValueError(f"Cannot start external server '{name}'")

        config = server_data.config

        try:
            if not config.docker_image:
                raise ValueError(f"No docker_image specified for server '{name}'")

            container_info = self._docker_manager.start_container(
                name=name,
                docker_url=config.docker_url,
                image=config.docker_image,
                volumes=config.volumes,
                bind_ports=config.bind_ports,
                environment=config.docker_environment,
                working_dir=config.docker_working_dir,
                restart_policy=config.restart_policy,
            )

            server_data.container_info = container_info
            server_data.status.status = ContainerStatus.RUNNING
            server_data.status.update_check_time()

            logger.info(
                f"Started server '{name}' with container ID {container_info.container_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to start server '{name}': {e}")
            server_data.status.status = ContainerStatus.ERROR
            server_data.status.update_check_time()
            raise

    def stop_server(self, name: str) -> bool:
        """Stop a server (only for internal managed servers)."""
        server_data = self.get_server(name)
        if not server_data:
            raise ValueError(f"Server '{name}' not found")

        if server_data.config.managed == "external":
            raise ValueError(f"Cannot stop external server '{name}'")

        config = server_data.config

        try:
            success = self._docker_manager.stop_container(name, config.docker_url)
            server_data.status.status = ContainerStatus.STOPPED
            server_data.status.update_check_time()

            logger.info(f"Stopped server '{name}'")
            return success

        except Exception as e:
            logger.error(f"Failed to stop server '{name}': {e}")
            server_data.status.status = ContainerStatus.ERROR
            server_data.status.update_check_time()
            raise

    def restart_server(self, name: str) -> bool:
        """Restart a server (only for internal managed servers)."""
        server_data = self.get_server(name)
        if not server_data:
            raise ValueError(f"Server '{name}' not found")

        if server_data.config.managed == "external":
            raise ValueError(f"Cannot restart external server '{name}'")

        config = server_data.config

        try:
            success = self._docker_manager.restart_container(name, config.docker_url)
            server_data.status.status = ContainerStatus.RUNNING
            server_data.status.update_check_time()

            logger.info(f"Restarted server '{name}'")
            return success

        except Exception as e:
            logger.error(f"Failed to restart server '{name}': {e}")
            server_data.status.status = ContainerStatus.ERROR
            server_data.status.update_check_time()
            raise

    def get_logs(self, name: str, lines: int = 100) -> Optional[List[str]]:
        """Get logs for a server (only for internal managed servers)."""
        if not (1 <= lines <= 1000):
            raise ValueError(f"lines must be between 1 and 1000, got {lines}")

        server_data = self.get_server(name)
        if not server_data:
            return None

        if server_data.config.managed == "external":
            return None

        config = server_data.config

        try:
            log_lines = self._docker_manager.get_container_logs(
                name, config.docker_url, lines
            )

            if log_lines:
                server_data.status.container_logs = ContainerLogs(logs=log_lines)

            return log_lines

        except Exception as e:
            logger.error(f"Failed to get logs for server '{name}': {e}")
            return []

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
        """Create a unified MCP server configuration for any transport type."""
        return MCPServerConfig(
            name=name,
            transport=transport,
            managed=managed,
            description=description,
            command=command,
            args=args,
            env=env,
            cwd=cwd,
            url=url,
            headers=headers,
            docker_image=docker_image,
            docker_url=docker_url,
            bind_ports=bind_ports,
            volumes=volumes,
            restart_policy=restart_policy,
            **kwargs,
        )
