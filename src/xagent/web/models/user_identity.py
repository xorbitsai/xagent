from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


class UserIdentity(Base):  # type: ignore[no-any-unimported]
    """External login identity linked to a local Xagent user."""

    __tablename__ = "user_identities"
    __table_args__ = (
        UniqueConstraint(
            "provider",
            "provider_subject",
            name="uq_user_identity_provider_subject",
        ),
        Index("ix_user_identities_user_id", "user_id"),
        Index("ix_user_identities_provider_subject", "provider", "provider_subject"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    provider = Column(String(50), nullable=False, index=True)
    provider_subject = Column(String(255), nullable=False)
    email = Column(String(320), nullable=True)
    email_verified = Column(Boolean, default=False, nullable=False)
    display_name = Column(String(255), nullable=True)
    picture_url = Column(String(1000), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user = relationship("User", back_populates="identities")
