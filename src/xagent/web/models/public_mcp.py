from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    event,
    inspect,
    true,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .database import Base
from .generation import RandomUUID


class PublicMCPApp(Base):  # type: ignore[no-any-unimported]
    """Registry of official MCP apps available for users to connect to."""

    __tablename__ = "public_mcp_apps"
    __table_args__ = (
        UniqueConstraint("generation", name="uq_public_mcp_apps_generation"),
        CheckConstraint(
            "CAST(generation AS VARCHAR) <> ''",
            name="ck_public_mcp_apps_generation_nonempty",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    generation: Mapped[UUID] = mapped_column(
        Uuid,
        default=uuid4,
        server_default=RandomUUID(),
        nullable=False,
    )
    app_id: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    icon: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    transport: Mapped[str] = mapped_column(String(50), default="oauth", nullable=False)

    # Optional FK to OAuthProvider
    provider_name: Mapped[str | None] = mapped_column(String(50), nullable=True)

    category: Mapped[str | None] = mapped_column(String(100), nullable=True)
    oauth_scopes: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    is_visible_in_connector: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default=true(), nullable=False
    )
    launch_config: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


@event.listens_for(PublicMCPApp, "before_update")
def _prevent_public_mcp_app_generation_update(
    _mapper: Any, _connection: Any, target: PublicMCPApp
) -> None:
    """Keep a catalog generation tied to exactly one row lifecycle."""
    if inspect(target).attrs.generation.history.has_changes():
        raise ValueError("PublicMCPApp.generation is immutable")


class PublicMCPAppAudit(Base):  # type: ignore[no-any-unimported]
    """Immutable admin write history for public MCP catalog apps."""

    __tablename__ = "public_mcp_app_audits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    actor_user_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    action: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    app_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    before_values: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    after_values: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    request_id: Mapped[str | None] = mapped_column(
        String(128), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )
