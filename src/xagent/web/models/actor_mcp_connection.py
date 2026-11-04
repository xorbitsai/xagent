from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
    event,
    inspect,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .database import Base
from .generation import RandomUUID

ACTOR_MCP_RESOURCE_OWNER_KEY_MAX_LENGTH = 512


class ActorMCPServerConnection(Base):  # type: ignore[no-any-unimported]
    """One actor-owned personal connection to a canonical stdio MCP app.

    The encrypted env is intentionally separate from ``UserMCPServer.env``:
    actor connections are not ordinary xagent connector associations and are
    not visible to the existing runtime until a later integration explicitly
    opts into this storage.

    Actor identity and catalog binding are immutable through supported ORM and
    service writes. Database-owner operations, database Core statements, and
    native SQL are trusted and outside this enforcement boundary.
    """

    __tablename__ = "actor_mcp_server_connections"
    __table_args__ = (
        UniqueConstraint(
            "lifecycle_generation",
            name="uq_actor_mcp_server_connections_lifecycle_generation",
        ),
        UniqueConstraint(
            "user_id",
            "resource_owner_key",
            "app_id",
            name="uq_actor_mcp_server_connections_actor_app",
        ),
        UniqueConstraint(
            "user_id",
            "resource_owner_key",
            "catalog_app_generation",
            name="uq_actor_mcp_server_connections_actor_catalog_generation",
        ),
        CheckConstraint(
            "CAST(lifecycle_generation AS VARCHAR) <> ''",
            name="ck_actor_mcp_server_connections_generation_nonempty",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    lifecycle_generation: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        default=uuid.uuid4,
        server_default=RandomUUID(),
        nullable=False,
    )
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    resource_owner_key: Mapped[str] = mapped_column(
        String(ACTOR_MCP_RESOURCE_OWNER_KEY_MAX_LENGTH), nullable=False
    )
    app_id: Mapped[str] = mapped_column(String(100), nullable=False)
    catalog_app_generation: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("public_mcp_apps.generation", ondelete="CASCADE"),
        nullable=False,
    )
    encrypted_env: Mapped[dict[str, str] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user = relationship("User")
    catalog_app = relationship("PublicMCPApp")

    def __repr__(self) -> str:
        return (
            "<ActorMCPServerConnection("
            f"id={self.id}, user_id={self.user_id}, app_id={self.app_id!r}, "
            f"catalog_app_generation={self.catalog_app_generation})>"
        )


@event.listens_for(ActorMCPServerConnection, "before_update")
def _prevent_actor_mcp_connection_generation_update(
    _mapper: Any, _connection: Any, target: ActorMCPServerConnection
) -> None:
    state: Any = inspect(target)
    immutable_fields = (
        "lifecycle_generation",
        "user_id",
        "resource_owner_key",
        "app_id",
        "catalog_app_generation",
    )
    changed = [
        field for field in immutable_fields if state.attrs[field].history.has_changes()
    ]
    if changed:
        raise ValueError(
            "ActorMCPServerConnection identity and catalog binding are immutable "
            "through supported ORM and service writes"
        )
