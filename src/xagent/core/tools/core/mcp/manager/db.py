import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Union

from mcp.types import AudioContent, EmbeddedResource, ImageContent, ResourceLink
from sqlalchemy.orm import Query, Session

from .....storage.manager import Base
from ..data_config import (
    ContainerInfo,
    ContainerLogs,
    ContainerStatus,
    MCPServerConfig,
    MCPServerData,
    MCPServerStatus,
)
from ..docker_manager import DockerManager
from ..model import create_mcp_server_table

logger = logging.getLogger(__name__)

NonTextContent = ImageContent | AudioContent | ResourceLink | EmbeddedResource

if TYPE_CHECKING:
    from sqlalchemy import JSON, Boolean, Column, DateTime, Integer, String, Text
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.sql import func

    Base = declarative_base()

    class MCPServer(Base):  # type: ignore[no-any-unimported]
        """MCP server configuration model for storing user-specific MCP server settings."""

        __tablename__ = "mcp_servers"

        id = Column(Integer, primary_key=True, index=True)
        name = Column(String(100), nullable=False, unique=True)
        description = Column(Text, nullable=True)

        # Connection parameters
        transport = Column(String(50), nullable=False)
        command = Column(String(500), nullable=True)
        args = Column(JSON, nullable=True)  # List[str]
        url = Column(String(500), nullable=True)
        env = Column(JSON, nullable=True)  # Dict[str, str]
        cwd = Column(String(500), nullable=True)
        headers = Column(JSON, nullable=True)  # Dict[str, Any]

        # Management type: 'internal' or 'external'
        managed = Column(String(20), nullable=False)

        # Container management parameters (internal only)
        docker_url = Column(String(500), nullable=True)
        docker_image = Column(String(200), nullable=True)
        docker_environment = Column(JSON, nullable=True)  # Dict[str, str]
        docker_working_dir = Column(String(500), nullable=True)
        volumes = Column(JSON, nullable=True)  # List[str]
        bind_ports = Column(JSON, nullable=True)  # Dict[str, Union[int, str]]
        restart_policy = Column(String(50), nullable=False, default="no")
        auto_start = Column(Boolean, nullable=True)

        # Container runtime info (populated when container is running)
        container_id = Column(String(100), nullable=True)
        container_name = Column(String(200), nullable=True)
        container_logs = Column(JSON, nullable=True)  # List[str]

        # Timestamps
        created_at = Column(DateTime(timezone=True), server_default=func.now())
        updated_at = Column(
            DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
        )
else:
    MCPServer = create_mcp_server_table(Base)


class DatabaseMCPServerManager:
    """
    Database-backed MCP Server Manager implementation.

    Stores MCP server configurations in a database using SQLAlchemy.
    Implements the MCPServerManager protocol.
    """

    def __init__(self, db_session: Session):
        """
        Initialize with database session.

        Args:
            db_session: SQLAlchemy database session
        """
        self.db = db_session
        self._docker_manager = DockerManager()
        self._servers_cache: Dict[str, MCPServerData] = {}

    def __del__(self) -> None:
        """Cleanup Docker clients on destruction."""
        if hasattr(self, "_docker_manager"):
            self._docker_manager.close_all()

    def _db_to_config(self, db_server: MCPServer) -> MCPServerConfig:
        """Convert database MCPServer to MCPServerConfig."""
        return MCPServerConfig.model_validate(db_server.to_config_dict())

    def _db_to_server_data(self, db_server: MCPServer) -> MCPServerData:
        """Convert database MCPServer to MCPServerData."""
        config = self._db_to_config(db_server)

        container_logs = ContainerLogs(logs=db_server.container_logs or [])
        status = MCPServerStatus(
            status=ContainerStatus.UNKNOWN,
            last_check=datetime.now(),
            container_logs=container_logs,
        )

        container_info = None
        if db_server.container_id and db_server.container_name:
            container_info = ContainerInfo(
                container_id=db_server.container_id,
                container_name=db_server.container_name,
            )

        return MCPServerData(
            config=config, status=status, container_info=container_info
        )

    def load_config(self) -> Dict[str, Dict[str, Any]]:
        """Load MCP server configurations from database."""
        config: Dict[str, Dict[str, Any]] = {"servers": {}}

        try:
            query = self.db.query(MCPServer)
            mcp_servers = query.all()

            for server in mcp_servers:
                server_name = str(server.name)
                server_config = server.to_config_dict()
                config["servers"].update({server_name: server_config})

            logger.info(
                f"Loaded {len(mcp_servers)} MCP server configurations from database"
            )

        except Exception as e:
            logger.error(f"Failed to load MCP configurations from database: {e}")

        return config

    def save_config(self) -> None:
        """
        Save configuration changes to database.

        Note: For database implementation, changes are saved immediately
        on add/remove operations, so this is a no-op.
        """
        pass

    def get_connections(
        self,
        filter_func: Optional[Callable[[Query[MCPServer]], Query[MCPServer]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get MCP server connections in the format expected by MCP tools.

        Args:
            filter_func: Optional function that takes a SQLAlchemy Query object
                        and returns a filtered Query object.

        Returns:
            Dictionary mapping server names to connection configurations.
        """
        connections: Dict[str, Dict[str, Any]] = {}

        try:
            query = self.db.query(MCPServer)

            if filter_func:
                query = filter_func(query)

            mcp_servers = query.all()

            for server in mcp_servers:
                if isinstance(server, tuple):
                    server = server[0]

                server_name = server.name
                if isinstance(server_name, str):
                    if self._requires_runtime_mcp_oauth(server):
                        logger.warning(
                            "Skipping MCP OAuth server '%s' in direct database "
                            "connection manager; use a runtime resolver instead",
                            server_name,
                        )
                        continue
                    connections[server_name] = server.to_connection_dict()

            logger.info(f"Created {len(connections)} MCP connections from database")

        except Exception as e:
            logger.error(f"Failed to create MCP connections from database: {e}")

        return connections

    @staticmethod
    def _requires_runtime_mcp_oauth(server: Any) -> bool:
        decrypt_auth = getattr(server, "_decrypt_auth_config", None)
        if not callable(decrypt_auth):
            return False
        auth_config = decrypt_auth(getattr(server, "auth", None))
        return (
            getattr(server, "transport", None)
            in {"sse", "websocket", "streamable_http"}
            and isinstance(auth_config, dict)
            and auth_config.get("type") == "mcp_oauth"
        )

    def add_server(self, config: MCPServerConfig) -> None:
        """Add a new MCP server configuration to database."""
        try:
            existing = (
                self.db.query(MCPServer).filter(MCPServer.name == config.name).first()
            )

            if existing:
                raise ValueError(f"Server '{config.name}' already exists")

            mcp_server = MCPServer.from_config(config.model_dump())

            self.db.add(mcp_server)
            self.db.commit()
            self.db.refresh(mcp_server)

            logger.info(f"Added MCP server '{config.name}'")

        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to add MCP server '{config.name}': {e}")
            raise

    def add_server_from_request(self, request_data: Dict[str, Any]) -> MCPServerConfig:
        """Add a new MCP server configuration from request data."""
        if "name" not in request_data:
            raise KeyError("Required parameter 'name' is missing from request data")
        if "transport" not in request_data:
            raise KeyError(
                "Required parameter 'transport' is missing from request data"
            )

        request_data.setdefault("managed", "external")
        request_data.setdefault("restart_policy", "no")

        config = self.create_config(**request_data)
        self.add_server(config)
        return config

    def remove_server(self, name: str) -> bool:
        """Remove an MCP server configuration from database."""
        try:
            mcp_server = self.db.query(MCPServer).filter(MCPServer.name == name).first()

            if not mcp_server:
                return False

            self.db.delete(mcp_server)
            self.db.commit()

            self._servers_cache.pop(name, None)

            logger.info(f"Removed MCP server '{name}'")
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to remove MCP server '{name}': {e}")
            raise

    def list_servers(self) -> List[MCPServerData]:
        """List all configured MCP servers with updated status."""
        query = self.db.query(MCPServer)
        db_servers = query.order_by(MCPServer.created_at.desc()).all()

        servers_data = []
        for db_server in db_servers:
            server_data = self._db_to_server_data(db_server)
            self._update_server_status(server_data)
            servers_data.append(server_data)

        return servers_data

    def get_server(self, name: str) -> Optional[MCPServerData]:
        """Get configuration and up to date status for a specific server."""
        query = self.db.query(MCPServer).filter(MCPServer.name == name)
        db_server = query.first()

        if not db_server:
            return None

        server_data = self._db_to_server_data(db_server)
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

            db_server = self.db.query(MCPServer).filter(MCPServer.name == name).first()
            if db_server:
                if container_info.container_id is not None:
                    db_server.container_id = container_info.container_id  # type: ignore[assignment]

                if container_info.container_name is not None:
                    db_server.container_name = container_info.container_name  # type: ignore[assignment]

                self.db.commit()

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

            server_data.status.container_logs = ContainerLogs(logs=log_lines or [])

            db_server = self.db.query(MCPServer).filter(MCPServer.name == name).first()
            if db_server:
                db_server.container_logs = log_lines or []  # type: ignore[assignment]
                self.db.commit()
                logger.debug(
                    f"Persisted {len(log_lines or [])} log lines for server '{name}'"
                )

            return log_lines

        except Exception as e:
            logger.error(f"Failed to get logs for server '{name}': {e}")
            db_server = self.db.query(MCPServer).filter(MCPServer.name == name).first()
            if db_server and db_server.container_logs:
                logger.info(f"Returning persisted logs for server '{name}' after error")
                return db_server.container_logs  # type: ignore[return-value]
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
