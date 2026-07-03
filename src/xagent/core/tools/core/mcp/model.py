from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Type

from sqlalchemy import JSON, Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.sql import func

SENSITIVE_AUTH_FIELDS = (
    "bearer_token",
    "api_key_value",
    "client_secret",
    "access_token",
)
MASKED_SECRET_VALUE = "********"


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
        concurrency_safe = Column(Boolean, nullable=False, default=False)
        concurrent_tools = Column(JSON, nullable=True)  # List[str]

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

        @staticmethod
        def _normalize_concurrent_tools(value: Any) -> list[str]:
            """Normalize stored/raw per-tool concurrency allowlists."""
            if value is None:
                return []
            if isinstance(value, str):
                return [item.strip() for item in value.split(",") if item.strip()]
            if isinstance(value, (list, tuple, set)):
                return [str(item).strip() for item in value if str(item).strip()]
            return []

        @staticmethod
        def _decrypt_auth_config(auth_value: Any) -> Any:
            """Decrypt sensitive auth fields while preserving non-dict values."""
            if not isinstance(auth_value, dict):
                return auth_value

            from xagent.core.utils.encryption import decrypt_value

            decrypted_auth = auth_value.copy()
            for key in SENSITIVE_AUTH_FIELDS:
                if key in decrypted_auth and decrypted_auth[key]:
                    decrypted_auth[key] = decrypt_value(decrypted_auth[key])
            return decrypted_auth

        @staticmethod
        def _merge_auth_headers(
            headers: Dict[str, Any] | None, auth_value: Any
        ) -> Dict[str, Any] | None:
            """Translate supported auth config into HTTP headers.

            Explicit custom headers always win over auto-generated auth headers.
            """
            merged_headers = dict(headers or {})
            if not isinstance(auth_value, dict):
                return merged_headers or None

            existing_headers = {
                str(header_name).lower() for header_name in merged_headers
            }
            auth_type = auth_value.get("type")

            if auth_type == "mcp_oauth":
                merged_headers = {
                    str(header_name): header_value
                    for header_name, header_value in merged_headers.items()
                    if str(header_name).lower() != "authorization"
                }
            elif auth_type == "bearer":
                bearer_token = auth_value.get("bearer_token")
                if bearer_token and "authorization" not in existing_headers:
                    merged_headers["Authorization"] = f"Bearer {bearer_token}"
            elif auth_type == "api_key":
                header_name = auth_value.get("api_key_name")
                header_value = auth_value.get("api_key_value")
                if (
                    header_name
                    and header_value
                    and str(header_name).lower() not in existing_headers
                ):
                    merged_headers[str(header_name)] = header_value
            elif auth_type == "oauth2":
                access_token = auth_value.get("access_token") or auth_value.get(
                    "bearer_token"
                )
                token_type = auth_value.get("token_type") or "Bearer"
                if access_token and "authorization" not in existing_headers:
                    merged_headers["Authorization"] = f"{token_type} {access_token}"

            return merged_headers or None

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
            decrypted_auth = self._decrypt_auth_config(getattr(self, "auth", None))
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
                    from xagent.core.utils.encryption import decrypt_env_dict

                    connection["env"] = decrypt_env_dict(getattr(self, "env", None))
                if self.cwd:
                    connection["cwd"] = self.cwd
            elif self.transport in ["sse", "websocket", "streamable_http"]:
                if self.url:
                    connection["url"] = self.url
                raw_headers = getattr(self, "headers", None)
                typed_headers = raw_headers if isinstance(raw_headers, dict) else None
                merged_headers = self._merge_auth_headers(typed_headers, decrypted_auth)
                if merged_headers:
                    connection["headers"] = merged_headers

            if getattr(self, "timeout", None) is not None:
                connection["timeout"] = self.timeout
            if getattr(self, "auth", None) is not None:
                # HTTP transports consume auth via generated headers above; keep non-dict
                # auth values for compatibility with callers that provide httpx.Auth.
                if self.transport not in [
                    "sse",
                    "websocket",
                    "streamable_http",
                ] or not isinstance(decrypted_auth, dict):
                    connection["auth"] = decrypted_auth

            connection["concurrency_safe"] = bool(
                getattr(self, "concurrency_safe", False)
            )
            connection["concurrent_tools"] = self._normalize_concurrent_tools(
                getattr(self, "concurrent_tools", None)
            )

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
                from xagent.core.utils.encryption import decrypt_env_dict

                config["env"] = decrypt_env_dict(getattr(self, "env", None))
            if self.cwd:
                config["cwd"] = self.cwd
            if self.headers:
                config["headers"] = self.headers
            if getattr(self, "timeout", None) is not None:
                config["timeout"] = self.timeout
            if getattr(self, "auth", None) is not None:
                config["auth"] = self._decrypt_auth_config(self.auth)

            config["concurrency_safe"] = bool(getattr(self, "concurrency_safe", False))
            config["concurrent_tools"] = self._normalize_concurrent_tools(
                getattr(self, "concurrent_tools", None)
            )

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
                # encrypt_value is idempotent, so already-encrypted values are
                # left untouched (no double-encryption).
                for key in SENSITIVE_AUTH_FIELDS:
                    if key in encrypted_auth and encrypted_auth[key]:
                        if encrypted_auth[key] == MASKED_SECRET_VALUE:
                            raise ValueError(
                                f"Masked auth value for '{key}' cannot be stored"
                            )
                        encrypted_auth[key] = encrypt_value(encrypted_auth[key])
                auth_config = encrypted_auth

            from xagent.core.utils.encryption import encrypt_env_dict

            return cls(
                name=config["name"],
                description=config.get("description"),
                managed=config["managed"],
                transport=config["transport"],
                command=config.get("command"),
                args=config.get("args"),
                url=config.get("url"),
                env=encrypt_env_dict(config.get("env")),
                cwd=str(config["cwd"])
                if isinstance(config.get("cwd"), Path)
                else config.get("cwd"),
                headers=config.get("headers"),
                timeout=config.get("timeout"),
                auth=auth_config,
                concurrency_safe=bool(config.get("concurrency_safe", False)),
                concurrent_tools=cls._normalize_concurrent_tools(
                    config.get("concurrent_tools")
                ),
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
