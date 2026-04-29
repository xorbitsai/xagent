from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Type

from sqlalchemy import JSON, Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.sql import func


def create_mcp_server_table(Base: Type[Any]) -> Type[Any]:
    """
    Factory function to create MCP server table with any SQLAlchemy Base class.

    Args:
        Base: SQLAlchemy declarative base class

    Returns:
        MCPServer class
    """

    class MCPServer(Base):
        """MCP server configuration model for storing user-specific MCP server settings."""

        __tablename__ = "mcp_servers"

        id = Column(Integer, primary_key=True, index=True)
        name = Column(String(100), nullable=False, unique=True)
        description = Column(Text, nullable=True)

        # Management type: 'internal' or 'external'
        managed = Column(String(20), nullable=False)

        # Connection parameters
        transport = Column(String(50), nullable=False)
        command = Column(String(500), nullable=True)
        args = Column(JSON, nullable=True)  # List[str]
        url = Column(String(500), nullable=True)
        env = Column(JSON, nullable=True)  # Dict[str, str]
        cwd = Column(String(500), nullable=True)
        headers = Column(JSON, nullable=True)  # Dict[str, Any]
        timeout = Column(Integer, nullable=True)
        auth = Column(JSON, nullable=True)  # Dict[str, Any]

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

        def __repr__(self) -> str:
            return f"<MCPServer(id={self.id}, name='{self.name}', transport='{self.transport}', managed='{self.managed}')>"

        @property
        def transport_display(self) -> str:
            """Get human-readable transport name."""
            transport_names = {
                "stdio": "STDIO",
                "sse": "Server-Sent Events",
                "websocket": "WebSocket",
                "streamable_http": "Streamable HTTP",
            }
            transport_value = self.transport
            if isinstance(transport_value, str):
                return transport_names.get(transport_value, transport_value.upper())
            return str(transport_value).upper()

        def to_connection_dict(self) -> Dict[str, Any]:
            """Convert to MCP connection format expected by MCP tools."""
            connection: Dict[str, Any] = {
                "name": self.name,
                "transport": self.transport,
            }

            # Add transport-specific fields
            if self.transport == "stdio":
                if self.command:
                    connection["command"] = self.command
                if self.args:
                    connection["args"] = self.args
                if self.env:
                    connection["env"] = self.env
                if self.cwd:
                    connection["cwd"] = self.cwd
            elif self.transport in ["sse", "websocket", "streamable_http"]:
                if self.url:
                    connection["url"] = self.url
                if self.headers:
                    connection["headers"] = self.headers

            if getattr(self, "timeout", None) is not None:
                connection["timeout"] = self.timeout
            if getattr(self, "auth", None) is not None:
                auth_dict = self.auth
                if isinstance(auth_dict, dict):
                    # Decrypt sensitive fields
                    from xagent.core.utils.encryption import decrypt_value

                    decrypted_auth = auth_dict.copy()
                    if (
                        "bearer_token" in decrypted_auth
                        and decrypted_auth["bearer_token"]
                    ):
                        decrypted_auth["bearer_token"] = decrypt_value(
                            decrypted_auth["bearer_token"]
                        )
                    if (
                        "api_key_value" in decrypted_auth
                        and decrypted_auth["api_key_value"]
                    ):
                        decrypted_auth["api_key_value"] = decrypt_value(
                            decrypted_auth["api_key_value"]
                        )
                    if (
                        "client_secret" in decrypted_auth
                        and decrypted_auth["client_secret"]
                    ):
                        decrypted_auth["client_secret"] = decrypt_value(
                            decrypted_auth["client_secret"]
                        )
                    connection["auth"] = decrypted_auth
                else:
                    connection["auth"] = self.auth

            return connection

        def to_config_dict(self) -> Dict[str, Any]:
            """Convert to MCPServerConfig compatible dictionary."""
            config = {
                "name": self.name,
                "description": self.description,
                "managed": self.managed,
                "transport": self.transport,
                "created_at": self.created_at,
            }

            # Connection parameters
            if self.command:
                config["command"] = self.command
            if self.args:
                config["args"] = self.args
            if self.url:
                config["url"] = self.url
            if self.env:
                config["env"] = self.env
            if self.cwd:
                config["cwd"] = self.cwd
            if self.headers:
                config["headers"] = self.headers
            if getattr(self, "timeout", None) is not None:
                config["timeout"] = self.timeout
            if getattr(self, "auth", None) is not None:
                auth_dict = self.auth
                if isinstance(auth_dict, dict):
                    # Decrypt sensitive fields
                    from xagent.core.utils.encryption import decrypt_value

                    decrypted_auth = auth_dict.copy()
                    if (
                        "bearer_token" in decrypted_auth
                        and decrypted_auth["bearer_token"]
                    ):
                        decrypted_auth["bearer_token"] = decrypt_value(
                            decrypted_auth["bearer_token"]
                        )
                    if (
                        "api_key_value" in decrypted_auth
                        and decrypted_auth["api_key_value"]
                    ):
                        decrypted_auth["api_key_value"] = decrypt_value(
                            decrypted_auth["api_key_value"]
                        )
                    if (
                        "client_secret" in decrypted_auth
                        and decrypted_auth["client_secret"]
                    ):
                        decrypted_auth["client_secret"] = decrypt_value(
                            decrypted_auth["client_secret"]
                        )
                    config["auth"] = decrypted_auth
                else:
                    config["auth"] = self.auth

            # Container parameters (internal only)
            if self.managed == "internal":
                if self.docker_url:
                    config["docker_url"] = self.docker_url
                if self.docker_image:
                    config["docker_image"] = self.docker_image
                if self.docker_environment:
                    config["docker_environment"] = self.docker_environment
                if self.docker_working_dir:
                    config["docker_working_dir"] = self.docker_working_dir
                if self.volumes:
                    config["volumes"] = self.volumes
                if self.bind_ports:
                    config["bind_ports"] = self.bind_ports
                config["restart_policy"] = self.restart_policy
                if self.auto_start is not None:
                    config["auto_start"] = self.auto_start

            return config

        @classmethod
        def from_config(cls, config: Dict[str, Any]) -> MCPServer:
            """Create MCPServer instance from MCPServerConfig dictionary."""

            # Encrypt sensitive auth fields before saving
            auth_config = config.get("auth")
            if auth_config and isinstance(auth_config, dict):
                from xagent.core.utils.encryption import encrypt_value

                encrypted_auth = auth_config.copy()
                # Check if it's already encrypted (starts with gAAAAAB...) to avoid double encryption
                # (Fernet tokens always start with gAAAAAB)
                if (
                    "bearer_token" in encrypted_auth
                    and encrypted_auth["bearer_token"]
                    and not encrypted_auth["bearer_token"].startswith("gAAAAAB")
                ):
                    encrypted_auth["bearer_token"] = encrypt_value(
                        encrypted_auth["bearer_token"]
                    )
                if (
                    "api_key_value" in encrypted_auth
                    and encrypted_auth["api_key_value"]
                    and not encrypted_auth["api_key_value"].startswith("gAAAAAB")
                ):
                    encrypted_auth["api_key_value"] = encrypt_value(
                        encrypted_auth["api_key_value"]
                    )
                if (
                    "client_secret" in encrypted_auth
                    and encrypted_auth["client_secret"]
                    and not encrypted_auth["client_secret"].startswith("gAAAAAB")
                ):
                    encrypted_auth["client_secret"] = encrypt_value(
                        encrypted_auth["client_secret"]
                    )
                auth_config = encrypted_auth

            return cls(
                name=config["name"],
                description=config.get("description"),
                managed=config["managed"],
                transport=config["transport"],
                command=config.get("command"),
                args=config.get("args"),
                url=config.get("url"),
                env=config.get("env"),
                cwd=str(config["cwd"])
                if isinstance(config.get("cwd"), Path)
                else config.get("cwd"),
                headers=config.get("headers"),
                timeout=config.get("timeout"),
                auth=auth_config,
                docker_url=config.get("docker_url"),
                docker_image=config.get("docker_image"),
                docker_environment=config.get("docker_environment"),
                docker_working_dir=config.get("docker_working_dir"),
                volumes=config.get("volumes"),
                bind_ports=config.get("bind_ports"),
                restart_policy=config.get("restart_policy", "no"),
                auto_start=config.get("auto_start"),
                container_id=config.get("container_id"),
                container_name=config.get("container_name"),
            )

    return MCPServer
