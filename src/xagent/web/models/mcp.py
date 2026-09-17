from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Type

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
    event,
    inspect,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ...core.tools.core.mcp.model import create_mcp_server_table
from .generation import RandomUUID

if TYPE_CHECKING:
    from sqlalchemy import JSON, Boolean, Column, DateTime, Integer, String, Text
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.sql import func

    Base = declarative_base()

    class MCPServer(Base):  # type: ignore[valid-type, misc]
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
        runtime_input_schema = Column(JSON, nullable=True)
        runtime_bindings = Column(JSON, nullable=True)
        allow_delegated_authorization = Column(
            Boolean, nullable=False, default=False, server_default="0"
        )

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
    from .database import Base

    MCPServer: Type[Any] = create_mcp_server_table(Base)
# Relationships
MCPServer.user_mcpservers = relationship(
    "UserMCPServer",
    back_populates="mcp_servers",
    cascade="all, delete-orphan",
)


class UserMCPServer(Base):  # type: ignore
    """User-MCPServer relationship table for MCP server ownership and sharing"""

    __tablename__ = "user_mcpservers"
    __table_args__ = (
        UniqueConstraint("user_id", "mcpserver_id", name="uq_user_mcpservers"),
        UniqueConstraint(
            "lifecycle_generation",
            name="uq_user_mcpservers_lifecycle_generation",
        ),
        CheckConstraint(
            "CAST(lifecycle_generation AS VARCHAR) <> ''",
            name="ck_user_mcpservers_lifecycle_generation_nonempty",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    lifecycle_generation = Column(
        Uuid,
        default=uuid.uuid4,
        server_default=RandomUUID(),
        nullable=False,
    )
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    mcpserver_id = Column(
        Integer, ForeignKey("mcp_servers.id", ondelete="CASCADE"), nullable=False
    )
    is_owner = Column(
        Boolean, default=False, nullable=False
    )  # True if user created the model
    can_edit = Column(
        Boolean, default=False, nullable=False
    )  # True if user can edit the model
    can_delete = Column(
        Boolean, default=False, nullable=False
    )  # True if user can delete the model
    is_shared = Column(
        Boolean, default=False, nullable=False
    )  # True if model is shared by admin
    is_active = Column(Boolean, default=True, nullable=False)
    is_default = Column(Boolean, default=False, nullable=False)
    # Per-user env overrides, merged over the global MCPServer.env at runtime
    # (global env acts as fallback). Dict[str, str].
    env = Column(JSON, nullable=True)
    # Which env layer this user picked for the server: "own" | "shared" |
    # "platform". NULL = legacy fallback (global < shared < user). See
    # resolve_stdio_env in services/mcp_runtime.py.
    env_source = Column(String(16), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="user_mcpservers")
    mcp_servers = relationship("MCPServer", back_populates="user_mcpservers")

    def __repr__(self) -> str:
        return f"<UserMCPServer(user_id={self.user_id}, mcpserver_id={self.mcpserver_id}, is_owner={self.is_owner})>"


@event.listens_for(UserMCPServer, "before_update")
def _prevent_user_mcpserver_generation_update(
    _mapper: Any, _connection: Any, target: UserMCPServer
) -> None:
    """Keep an association generation tied to exactly one row lifecycle."""
    if inspect(target).attrs.lifecycle_generation.history.has_changes():
        raise ValueError("UserMCPServer.lifecycle_generation is immutable")
