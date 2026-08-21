from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base

USER_OAUTH_RESOURCE_OWNER_KEY_MAX_LENGTH = 512
_ORDINARY_OWNER_CLAUSE = text("resource_owner_key IS NULL")
_ACTOR_OWNER_CLAUSE = text("resource_owner_key IS NOT NULL")


class UserOAuth(Base):  # type: ignore[no-any-unimported]
    """OAuth credentials owned by either a user or one trusted actor.

    A null ``resource_owner_key`` preserves the ordinary xagent credential
    namespace. A non-null key selects an actor inside the same xagent account.
    The key is server-owned identity metadata, never provider token material.
    """

    __tablename__ = "user_oauth"
    __table_args__ = (
        Index(
            "uq_user_oauth_ordinary_account",
            "user_id",
            "provider",
            "provider_user_id",
            unique=True,
            sqlite_where=_ORDINARY_OWNER_CLAUSE,
            postgresql_where=_ORDINARY_OWNER_CLAUSE,
        ),
        Index(
            "uq_user_oauth_actor_account",
            "user_id",
            "resource_owner_key",
            "provider",
            "provider_user_id",
            unique=True,
            sqlite_where=_ACTOR_OWNER_CLAUSE,
            postgresql_where=_ACTOR_OWNER_CLAUSE,
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    provider = Column(String(50), nullable=False)  # e.g. "google-drive"
    resource_owner_key = Column(
        String(USER_OAUTH_RESOURCE_OWNER_KEY_MAX_LENGTH), nullable=True
    )
    access_token = Column(String, nullable=False)
    refresh_token = Column(String, nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    token_type = Column(String(50), nullable=True)
    scope = Column(String, nullable=True)
    provider_user_id = Column(
        String, nullable=True
    )  # The user's ID in the provider system
    email = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship(
        "User",
        back_populates="oauth_accounts",
        # Keep actor assignment from synchronizing into User.oauth_accounts,
        # whose SQL join intentionally represents ordinary credentials only.
        sync_backref=False,
    )

    def __repr__(self) -> str:
        return f"<UserOAuth(user_id={self.user_id}, provider='{self.provider}', email='{self.email}')>"
