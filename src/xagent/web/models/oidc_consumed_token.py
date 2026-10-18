from sqlalchemy import Column, DateTime, String

from .database import Base


class OidcConsumedToken(Base):  # type: ignore[no-any-unimported]
    """Consumed short-lived authentication proofs for replay protection.

    Ordinary OAuth state digests use ``oauth-state:`` and short-lived connector
    navigation IDs use ``oauth-launch:``, distinct from OIDC exchange IDs.
    Each owner bounds retention and cleanup within its own namespace.
    """

    __tablename__ = "oidc_consumed_tokens"

    token_id = Column(String(96), primary_key=True)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
