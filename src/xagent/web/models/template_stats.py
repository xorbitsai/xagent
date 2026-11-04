"""Template statistics model for tracking template usage"""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class TemplateStats(Base):  # type: ignore
    """Template usage statistics model"""

    __tablename__ = "template_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    template_id: Mapped[str] = mapped_column(
        String(200), nullable=False, unique=True, index=True
    )
    views: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    likes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    used_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.now,
        onupdate=datetime.now,
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<TemplateStats(template_id='{self.template_id}', views={self.views}, likes={self.likes})>"


class UserTemplateRelation(Base):  # type: ignore
    """User-template relationship state for likes and future template actions."""

    __tablename__ = "user_template_relations"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "template_id",
            "relation_type",
            name="uq_user_template_relation",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    template_id: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    relation_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user = relationship("User", back_populates="template_relations")

    def __repr__(self) -> str:
        return (
            f"<UserTemplateRelation(user_id={self.user_id}, "
            f"template_id='{self.template_id}', relation_type='{self.relation_type}', "
            f"is_active={self.is_active})>"
        )
