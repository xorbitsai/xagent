"""Explicit shared Google OAuth grant and connector associations."""

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base
from .user_oauth import USER_OAUTH_RESOURCE_OWNER_KEY_MAX_LENGTH

_ORDINARY_OWNER_CLAUSE = text("resource_owner_key IS NULL")
_ACTOR_OWNER_CLAUSE = text("resource_owner_key IS NOT NULL")


class GoogleOAuthGrant(Base):  # type: ignore[no-any-unimported]
    """Identity/client boundary around one shared ``UserOAuth`` credential.

    Token material remains in ``UserOAuth`` so refresh, encryption, reconnect
    tombstones, and key rotation keep one implementation. This row makes the
    dimensions that are unsafe to infer from an app id explicit.

    Shared rows have three lifecycle invariants for the callback/runtime writer:
    update ``provider='google'`` credentials in place instead of deleting them
    (the foreign key intentionally cascades); never expose one through the
    legacy provider-candidate resolver; and route a Gmail association's
    permanent-invalid transition through Gmail watch quiescing even though the
    stored provider is ``google``. The association-writing follow-up must keep
    these invariants before enabling the feature flag.
    """

    __tablename__ = "google_oauth_grants"
    __table_args__ = (
        UniqueConstraint("oauth_account_id", name="uq_google_oauth_grant_account"),
        Index(
            "uq_google_oauth_grant_ordinary_identity",
            "user_id",
            "google_subject",
            "oauth_client_fingerprint",
            unique=True,
            sqlite_where=_ORDINARY_OWNER_CLAUSE,
            postgresql_where=_ORDINARY_OWNER_CLAUSE,
        ),
        Index(
            "uq_google_oauth_grant_actor_identity",
            "user_id",
            "resource_owner_key",
            "google_subject",
            "oauth_client_fingerprint",
            unique=True,
            sqlite_where=_ACTOR_OWNER_CLAUSE,
            postgresql_where=_ACTOR_OWNER_CLAUSE,
        ),
    )

    id = Column(Integer, primary_key=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    resource_owner_key = Column(
        String(USER_OAUTH_RESOURCE_OWNER_KEY_MAX_LENGTH), nullable=True
    )
    google_subject = Column(String(255), nullable=False)
    oauth_client_fingerprint = Column(String(64), nullable=False)
    oauth_account_id = Column(
        Integer,
        ForeignKey("user_oauth.id", ondelete="CASCADE"),
        nullable=False,
    )
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    oauth_account = relationship("UserOAuth", foreign_keys=[oauth_account_id])
    associations = relationship(
        "GoogleOAuthAppAssociation",
        back_populates="grant",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class GoogleOAuthAppAssociation(Base):  # type: ignore[no-any-unimported]
    """Authorize one connector app to resolve one shared Google grant."""

    __tablename__ = "google_oauth_app_associations"
    __table_args__ = (
        UniqueConstraint("grant_id", "app_id", name="uq_google_oauth_app_association"),
        UniqueConstraint(
            "legacy_oauth_account_id",
            name="uq_google_oauth_app_association_legacy_account",
        ),
        Index("ix_google_oauth_app_associations_app_id", "app_id"),
    )

    id = Column(Integer, primary_key=True)
    grant_id = Column(
        Integer,
        ForeignKey("google_oauth_grants.id", ondelete="CASCADE"),
        nullable=False,
    )
    app_id = Column(String(255), nullable=False)
    legacy_oauth_account_id = Column(
        Integer,
        ForeignKey("user_oauth.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    grant = relationship("GoogleOAuthGrant", back_populates="associations")
    legacy_oauth_account = relationship(
        "UserOAuth", foreign_keys=[legacy_oauth_account_id]
    )
