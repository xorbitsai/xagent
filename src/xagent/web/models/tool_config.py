"""Tool configuration models for database storage."""

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


class ToolConfig(Base):  # type: ignore
    """Tool configuration table for storing tool settings and availability."""

    __tablename__ = "tool_configs"

    id = Column(Integer, primary_key=True, index=True)
    tool_name = Column(String(100), unique=True, index=True, nullable=False)
    tool_type = Column(String(20), nullable=False)  # builtin, vision, image, mcp, file
    category = Column(String(50), nullable=False)  # development, search, ai_tools, etc.
    display_name = Column(String(100), nullable=False)  # User-friendly name
    description = Column(Text, nullable=True)  # Tool description
    enabled = Column(Boolean, default=True)  # Whether the tool is enabled
    requires_configuration = Column(Boolean, default=False, nullable=False)
    config = Column(JSON, nullable=True)  # Tool-specific configuration
    dependencies = Column(JSON, nullable=True)  # Required models/services
    status = Column(
        String(20), default="available"
    )  # available, missing_config, missing_model, error, disabled
    status_reason = Column(String(500), nullable=True)  # Reason for current status
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class ToolUsage(Base):  # type: ignore
    """Tool usage statistics table."""

    __tablename__ = "tool_usage"

    id = Column(Integer, primary_key=True, index=True)
    tool_name = Column(String(100), nullable=False, index=True)
    usage_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class UserToolConfig(Base):  # type: ignore
    __tablename__ = "user_tool_configs"
    __table_args__ = (
        UniqueConstraint("user_id", "tool_name", name="uq_user_tool_config"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tool_name = Column(String(100), nullable=False, index=True)
    config = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    user = relationship("User", back_populates="tool_configs")
