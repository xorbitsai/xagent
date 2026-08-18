from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


class UserOAuth(Base):  # type: ignore[no-any-unimported]
    """User OAuth accounts (e.g. Google Drive, GitHub)"""

    __tablename__ = "user_oauth"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "provider", "provider_user_id", name="uq_user_provider_account"
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    provider = Column(String(50), nullable=False)  # e.g. "google-drive"
    access_token = Column(String, nullable=False)
    refresh_token = Column(String, nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    token_type = Column(String(50), nullable=True)
    scope = Column(String, nullable=True)
    provider_user_id = Column(
        String, nullable=True
    )  # The user's ID in the provider system
    email = Column(String, nullable=True)
    # Salesforce (and no other provider here) returns a per-org API host in
    # the token response instead of using a fixed API domain -- every
    # subsequent API call must go through this URL, not a hardcoded one.
    instance_url = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="oauth_accounts")

    def __repr__(self) -> str:
        return f"<UserOAuth(user_id={self.user_id}, provider='{self.provider}', email='{self.email}')>"
