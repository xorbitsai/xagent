from __future__ import annotations

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


class GmailWatchState(Base):  # type: ignore
    """Persisted Gmail watch cursor for a connected Gmail OAuth account."""

    __tablename__ = "gmail_watch_states"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    oauth_account_id = Column(
        Integer,
        ForeignKey("user_oauth.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    email = Column(String(255), nullable=False, index=True)
    history_id = Column(String(255), nullable=False)
    watch_expiration = Column(DateTime(timezone=True), nullable=True, index=True)
    topic_name = Column(String(512), nullable=False)
    last_error = Column(Text, nullable=True)

    callback_id = Column(String(128), nullable=True, unique=True, index=True)
    push_audience = Column(Text, nullable=True)
    previous_push_audience = Column(Text, nullable=True)
    previous_push_audience_expires_at = Column(DateTime(timezone=True), nullable=True)
    subscription_name = Column(String(512), nullable=True)
    status = Column(String(32), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user = relationship("User")
    oauth_account = relationship("UserOAuth")

    def __repr__(self) -> str:
        return (
            f"<GmailWatchState(id={self.id}, user_id={self.user_id}, "
            f"email='{self.email}')>"
        )
