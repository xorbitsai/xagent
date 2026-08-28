"""Minimal single-use nonce storage for trusted actor OAuth callbacks."""

from sqlalchemy import Column, DateTime, String

from .database import Base


class ActorOAuthFlowState(Base):  # type: ignore[no-any-unimported]
    """One unexpired browser-bound actor OAuth nonce.

    All actor, user, provider, and application claims remain exclusively in
    signed state. Deleting this row atomically claims the callback flow.
    """

    __tablename__ = "actor_oauth_flow_states"

    nonce = Column(String(64), primary_key=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
