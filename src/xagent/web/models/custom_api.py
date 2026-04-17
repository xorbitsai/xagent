from __future__ import annotations

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    pass


class CustomApi(Base):  # type: ignore[no-any-unimported]
    """Database model for storing Custom API configurations."""

    __tablename__ = "custom_apis"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)

    # Connection parameters
    url = Column(String(500), nullable=True)
    method = Column(String(20), nullable=True, default="GET")
    headers = Column(JSON, nullable=True)  # Dict[str, str]
    env = Column(JSON, nullable=True)  # Dict[str, str] - encrypted values

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    user_custom_apis = relationship(
        "UserCustomApi",
        back_populates="custom_api",
        cascade="all, delete-orphan",
    )


class UserCustomApi(Base):  # type: ignore[no-any-unimported]
    """User-CustomApi relationship table for ownership and sharing."""

    __tablename__ = "user_custom_apis"
    __table_args__ = (
        UniqueConstraint("user_id", "custom_api_id", name="uq_user_custom_apis"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    custom_api_id = Column(
        Integer, ForeignKey("custom_apis.id", ondelete="CASCADE"), nullable=False
    )
    is_owner = Column(Boolean, default=False, nullable=False)
    can_edit = Column(Boolean, default=False, nullable=False)
    can_delete = Column(Boolean, default=False, nullable=False)
    is_shared = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_default = Column(Boolean, default=False, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="user_custom_apis")
    custom_api = relationship("CustomApi", back_populates="user_custom_apis")

    def __repr__(self) -> str:
        return f"<UserCustomApi(user_id={self.user_id}, custom_api_id={self.custom_api_id}, is_owner={self.is_owner})>"
