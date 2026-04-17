"""
Unified MCP Server Management Data Models

This module contains the unified data models for MCP server configuration,
runtime status, and management. It combines connection details with lifecycle
management information to provide a single source of truth for all MCP servers.
"""

import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class MCPServerConfig(BaseModel):
    """
    Unified MCP Server Configuration Model.
    This model serves as the single source of truth for all server configurations,
    covering both externally managed and internally managed (e.g., Docker) servers.
    """

    name: str = Field(..., description="Unique name for the MCP server", min_length=1)
    description: Optional[str] = Field(
        None, description="Optional description of the server"
    )

    # Management Type:
    # 'internal': Lifecycle is managed by xagent (e.g., starting/stopping a Docker container).
    # 'external': An already running, externally managed service (e.g., a remote API endpoint).
    managed: Literal["internal", "external"] = Field(
        ..., description="Management type for the server"
    )

    # --- Connection Parameters (for all server types) ---
    transport: str = Field(
        ..., description="Transport protocol (e.g., 'stdio', 'sse', 'websocket')"
    )
    command: Optional[str] = Field(
        None, description="Command to execute (stdio transport)"
    )
    args: Optional[List[str]] = Field(
        None, description="Command line arguments (stdio transport)"
    )
    url: Optional[str] = Field(
        None, description="Server URL (sse, websocket, streamable_http transports)"
    )
    env: Optional[Dict[str, str]] = Field(
        None, description="Environment variables (stdio transport)"
    )
    cwd: Optional[Union[str, Path]] = Field(
        None, description="Working directory (stdio transport)"
    )
    headers: Optional[Dict[str, Any]] = Field(
        None, description="HTTP headers (sse, streamable_http transports)"
    )

    # --- Container Management Parameters (only for 'internal' type) ---
    docker_url: Optional[str] = Field(None, description="URL to connect docker")
    docker_image: Optional[str] = Field(None, description="Docker image name")
    docker_environment: Optional[Dict[str, str]] = Field(
        None, description="Environment variables for Docker container"
    )
    docker_working_dir: Optional[str] = Field(
        None, description="Working directory for Docker container"
    )
    volumes: Optional[List[str]] = Field(None, description="Docker volume mappings")
    bind_ports: Optional[Dict[str, Union[int, str]]] = Field(
        None,
        description="Port mappings for Docker container (format: {'container_port': host_port})",
    )
    restart_policy: str = Field("no", description="Docker container restart policy")
    auto_start: Optional[bool] = Field(
        None,
        description="Whether to start this server automatically on startup (internal only)",
    )

    # --- Timestamps ---
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate server name format."""
        if not v or not v.strip():
            raise ValueError("Name cannot be empty")

        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Name can only contain letters, numbers, hyphens and underscores"
            )

        return v.strip()

    @field_validator("transport")
    @classmethod
    def validate_transport(cls, v: str) -> str:
        """Validate transport type."""
        valid_transports = {
            "stdio",
            "sse",
            "websocket",
            "streamable_http",
        }
        if v not in valid_transports:
            raise ValueError(
                f"Invalid transport '{v}'. "
                f"Must be one of: {', '.join(sorted(valid_transports))}"
            )
        return v

    @model_validator(mode="after")
    def validate_config_consistency(self) -> "MCPServerConfig":
        """Validate configuration consistency based on management type and transport."""
        # Validate transport-specific parameters
        if self.transport == "stdio":
            if self.managed == "internal":
                raise ValueError(
                    "internal managed servers cannot use 'stdio' transport"
                )
            if not self.command:
                raise ValueError("stdio transport requires 'command' parameter")
        elif self.transport in {"sse", "websocket", "streamable_http"}:
            if not self.url:
                raise ValueError(f"{self.transport} transport requires 'url' parameter")

        # Validate management-specific parameters
        if self.managed == "internal":
            if not self.docker_image:
                raise ValueError("internal managed servers require 'docker_image'")
            # Set default auto_start for internal servers
            if self.auto_start is None:
                self.auto_start = True
        elif self.managed == "external":
            # External servers should have connection info but no docker config
            docker_fields = [
                self.docker_image,
                self.docker_environment,
                self.docker_working_dir,
            ]
            if any(field is not None for field in docker_fields):
                raise ValueError(
                    "external managed servers should not have docker configuration"
                )
            # External servers don't use auto_start
            if self.auto_start is not None:
                raise ValueError(
                    "external managed servers should not have auto_start parameter"
                )

        return self

    def to_connection(self) -> Dict[str, Any]:
        """Convert configuration to Connection format for MultiServerMCPClient."""
        connection_config = {"transport": self.transport}

        # Add connection-related fields
        connection_fields = {"command", "args", "url", "env", "cwd", "headers"}
        for field_name in connection_fields:
            value = getattr(self, field_name)
            if value is not None:
                if field_name == "cwd" and isinstance(value, Path):
                    connection_config[field_name] = str(value)
                else:
                    connection_config[field_name] = value

        return connection_config


class ContainerStatus(str, Enum):
    """Container/server status enumeration."""

    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    UNKNOWN = "unknown"
    STARTING = "starting"
    STOPPING = "stopping"
    REACHABLE = "reachable"  # For external servers
    UNREACHABLE = "unreachable"  # For external servers

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_docker_status(cls, docker_status: str) -> "ContainerStatus":
        """Convert Docker native status to ContainerStatus."""
        status_map = {
            "running": cls.RUNNING,
            "exited": cls.STOPPED,
            "dead": cls.STOPPED,
            "created": cls.STARTING,
            "restarting": cls.STARTING,
            "paused": cls.UNKNOWN,
            "removing": cls.STOPPING,
        }
        return status_map.get(docker_status, cls.UNKNOWN)


class ContainerInfo(BaseModel):
    """Holds Docker-specific container information."""

    container_id: Optional[str] = Field(default=None, description="Container ID")
    container_name: Optional[str] = Field(default=None, description="Container name")
    image: Optional[str] = Field(default=None, description="Image name")
    image_id: Optional[str] = Field(default=None, description="Image ID")
    ports: Dict[str, List[Dict[str, str]]] = Field(
        default_factory=dict, description="Port mappings"
    )
    mounts: List[Dict[str, Any]] = Field(
        default_factory=list, description="Mount points"
    )


class ContainerLogs(BaseModel):
    """Container logs model."""

    logs: List[str] = Field(default_factory=list, description="Container logs list")


class ResourceUsage(BaseModel):
    """Resource usage information model."""

    cpu_usage: Optional[float] = Field(
        default=None, description="CPU usage percentage", ge=0, le=100
    )
    memory_usage: Optional[int] = Field(
        default=None, description="Memory usage in bytes", ge=0
    )
    network_rx: Optional[int] = Field(
        default=None, description="Network received bytes", ge=0
    )
    network_tx: Optional[int] = Field(
        default=None, description="Network transmitted bytes", ge=0
    )
    block_read: Optional[int] = Field(default=None, description="Disk read bytes", ge=0)
    block_write: Optional[int] = Field(
        default=None, description="Disk write bytes", ge=0
    )
    pids: Optional[int] = Field(default=None, description="Process count", ge=0)


class MCPServerStatus(BaseModel):
    """Holds the dynamic status of a server."""

    status: ContainerStatus = Field(..., description="Running status")
    container_logs: Optional[ContainerLogs] = Field(
        default=None, description="Container logs"
    )
    resource_usage: Optional[ResourceUsage] = Field(
        default=None, description="Resource usage information"
    )
    uptime: Optional[str] = Field(default=None, description="Uptime duration")
    health_status: Optional[str] = Field(default=None, description="Health status")
    last_check: datetime = Field(
        default_factory=datetime.now, description="Last check time"
    )

    def is_running(self) -> bool:
        """Check if server is running."""
        return self.status in {ContainerStatus.RUNNING, ContainerStatus.REACHABLE}

    def is_healthy(self) -> bool:
        """Check if server is healthy."""
        return self.is_running() and self.health_status in [None, "healthy"]

    def update_check_time(self) -> None:
        """Update last check time."""
        self.last_check = datetime.now()


class MCPServerData(BaseModel):
    """
    In-memory representation of a server's complete state, combining
    static configuration with dynamic runtime information.
    """

    config: MCPServerConfig = Field(..., description="Configuration information")
    status: MCPServerStatus = Field(..., description="Status information")
    container_info: Optional[ContainerInfo] = Field(
        default=None, description="Container information (internal servers only)"
    )

    def is_internal(self) -> bool:
        """Check if this is an internally managed server."""
        return self.config.managed == "internal"

    def is_external(self) -> bool:
        """Check if this is an externally managed server."""
        return self.config.managed == "external"


__all__ = [
    "MCPServerConfig",
    "MCPServerData",
    "MCPServerStatus",
    "ContainerInfo",
    "ContainerLogs",
    "ResourceUsage",
    "ContainerStatus",
]
