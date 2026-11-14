"""Serialize a provider-side grant revocation with OAuth callbacks."""

from datetime import datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class OAuthRevocationFence(Base):  # type: ignore[no-any-unimported]
    """One row per upstream account a builtin OAuth grant has been connected for.

    GitHub and Google revoke a whole grant per upstream account and OAuth
    client, so a revoke also kills tokens a concurrent callback for the same
    account has just been issued. The revoker and that callback both lock
    this row (see ``services.oauth_revocation_fences``): the revoker checks
    for live references and records ``revoking_until`` under the lock, and a
    callback whose token exchange started before ``revoking_until`` is
    refused instead of storing credentials the revoke may already have
    killed.

    Either side inserts the row if missing, so every Google or GitHub
    connect that records an upstream account id creates one (with
    ``revoking_until`` unset), whether or not that account is ever revoked.
    Rows are never deleted -- a row is a lock target that must stay put while
    any callback or revoker may be waiting on it -- so the table grows with
    the number of distinct upstream accounts ever connected. Each row is two
    short keys and a timestamp, which is why nothing prunes it.
    """

    __tablename__ = "oauth_revocation_fences"

    # No FK: the fence outlives every local credential for the account.
    provider: Mapped[str] = mapped_column(String(50), primary_key=True)
    provider_user_id: Mapped[str] = mapped_column(String, primary_key=True)
    # Until when a revocation may still reach the provider; ``None`` while
    # no revocation has ever been opened for this account.
    revoking_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
