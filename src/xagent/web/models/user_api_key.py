"""Personal SDK management API keys bound to a user."""

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


class UserApiKey(Base):  # type: ignore
    """User-level SDK key for management endpoints under ``/v1``."""

    __tablename__ = "user_api_keys"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    key_prefix = Column(String(12), nullable=False, unique=True, index=True)
    key_hash = Column(String(128), nullable=False)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user = relationship("User", back_populates="api_keys")

    def __repr__(self) -> str:
        return (
            f"<UserApiKey(id={self.id}, user_id={self.user_id}, "
            f"key_prefix='{self.key_prefix}', revoked={self.revoked_at is not None})>"
        )
