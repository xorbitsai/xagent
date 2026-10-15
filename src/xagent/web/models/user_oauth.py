import logging

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.types import TypeDecorator

from ...core.utils.encryption import (
    EncryptionDecodeError,
    decrypt_value_strict,
    encrypt_value,
)
from .database import Base

logger = logging.getLogger(__name__)


class UndecryptableOAuthToken(str):
    """Empty token marker for ciphertext that needs user reauthorization."""

    requires_reconnect = True


def is_undecryptable_oauth_token(value: object) -> bool:
    """Return whether an ORM token value was unreadable under the current key."""
    return isinstance(value, UndecryptableOAuthToken)


class EncryptedOAuthToken(TypeDecorator[str]):
    """Encrypt OAuth tokens and fail closed if the configured key cannot open one.

    Fernet output is intentionally non-deterministic, so callers must not add
    value-equality filters for token contents. Empty strings and NULL remain
    supported for presence checks and ``IS NULL`` predicates. A token encrypted
    under a rotated key is exposed as an empty reconnect marker in Python, but
    SQL filters still see its ciphertext; query consumers must inspect the
    loaded value before attempting refresh or renewal.
    """

    impl = String
    cache_ok = True

    def process_bind_param(self, value: str | None, dialect: object) -> str | None:
        return encrypt_value(value) if value else value

    def process_result_value(self, value: str | None, dialect: object) -> str | None:
        if not value:
            return value
        try:
            return decrypt_value_strict(value)
        except EncryptionDecodeError:
            # A rotated/missing key must not make every UserOAuth query fail.
            # Returning an empty credential marks this connection for
            # reauthorization while leaving the ciphertext untouched so it
            # can still be recovered with the original key.
            logger.warning(
                "Unable to decrypt a stored UserOAuth token; "
                "connection requires reauthorization"
            )
            return UndecryptableOAuthToken("")


USER_OAUTH_RESOURCE_OWNER_KEY_MAX_LENGTH = 512
_ORDINARY_OWNER_CLAUSE = text("resource_owner_key IS NULL")
_ACTOR_OWNER_CLAUSE = text("resource_owner_key IS NOT NULL")


class UserOAuth(Base):  # type: ignore[no-any-unimported]
    """OAuth credentials owned by either a user or one trusted actor.

    A null ``resource_owner_key`` preserves the ordinary xagent credential
    namespace. A non-null key selects an actor inside the same xagent account.
    This deliberately differs from ``MCPOAuthGrant``'s non-null synthetic
    ``xagent:user:<id>`` owner key so existing UserOAuth rows need no backfill.
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
    access_token = Column(EncryptedOAuthToken, nullable=False)
    refresh_token = Column(EncryptedOAuthToken, nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    token_type = Column(String(50), nullable=True)
    scope = Column(String, nullable=True)
    provider_user_id = Column(
        String, nullable=True
    )  # The user's ID in the provider system
    email = Column(String, nullable=True)
    # A per-connection value some providers need beyond a fixed API domain,
    # reused for different shapes by whichever provider populates it:
    # Salesforce's per-org API host (a real URL), Deputy's per-install API
    # host (a bare hostname, no scheme), MYOB's company-file businessId (a
    # bare GUID, not a URL at all despite the column name) -- see each
    # provider's own handling in api/auth.py for how it's populated and
    # tools/mcp/<provider>.py for how it's used at call time.
    instance_url = Column(String, nullable=True)

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
