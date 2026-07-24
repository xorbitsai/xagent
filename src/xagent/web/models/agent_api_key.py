"""API key for SDK-side authentication, bound to an agent or a workforce."""

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


class AgentApiKey(Base):  # type: ignore
    """SDK API key. An owner may hold any number of simultaneously-active keys.

    A key is bound to exactly one owner: either an agent (``agent_id``) or
    a workforce (``workforce_id``). Exactly one of the two FKs is set --
    enforced at the service layer rather than by a CHECK constraint so
    SQLite deployments don't need a table rebuild (same trade-off as
    ``uq_workforce_run_idempotency``). The ``xag_*`` wire format is
    identical for both owner types; resolution branches on which FK is
    populated.

    Rotating (via the legacy single-key endpoints) revokes existing active
    rows and inserts a new one; the multi-key admin endpoints instead let
    callers add keys without touching existing ones. Revoked rows stay in
    the table as an audit trail.
    """

    __tablename__ = "agent_api_keys"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(
        Integer,
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    workforce_id = Column(
        Integer,
        ForeignKey("workforces.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    # Owner-facing display name for the key, e.g. "CI pipeline". Optional.
    label = Column(String(100), nullable=True)
    # Public-safe lookup handle (6-char in practice; column sized to 12 for headroom).
    key_prefix = Column(String(12), nullable=False, unique=True, index=True)
    # bcrypt(full_key, cost=12). Never store the plaintext secret.
    key_hash = Column(String(128), nullable=False)
    # Temporarily disables auth without revoking (audit-preserving pause/resume).
    paused_at = Column(DateTime(timezone=True), nullable=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    # "YYYY-MM" bucket paired with usage_month_calls; lazily reset when a
    # call lands in a new month rather than requiring a scheduled job.
    usage_month = Column(String(7), nullable=True)
    usage_month_calls = Column(Integer, nullable=False, server_default="0")
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    agent = relationship("Agent")
    workforce = relationship("Workforce")

    def __repr__(self) -> str:
        return (
            f"<AgentApiKey(id={self.id}, agent_id={self.agent_id}, "
            f"workforce_id={self.workforce_id}, "
            f"key_prefix='{self.key_prefix}', revoked={self.revoked_at is not None})>"
        )
