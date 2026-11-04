"""Shared external-deployment configuration for agents and workforces.

One row per (owner_type, owner_id) holds the channel opt-ins and credentials
for exposing that owner outside Xagent (widget embed, public share link).
Workforce deployments use this table from day one; the Agent model still
carries its legacy flattened columns (``widget_enabled`` / ``share_token`` /
...) — migrating those onto this table is a separate effort because it
touches the auth-critical public chat access path.
"""

import enum

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.sql import func

from .database import Base


class DeploymentOwnerType(enum.Enum):
    """Which kind of entity a deployment row exposes."""

    AGENT = "agent"
    WORKFORCE = "workforce"


class Deployment(Base):  # type: ignore[no-any-unimported]
    __tablename__ = "deployments"
    __table_args__ = (
        UniqueConstraint("owner_type", "owner_id", name="uq_deployment_owner"),
    )

    id = Column(Integer, primary_key=True, index=True)
    # No DB ForeignKey: owner_id points at agents.id or workforces.id depending
    # on owner_type, which a single FK cannot express. Ownership/authorization
    # is always resolved through the owner row itself.
    owner_type = Column(String(20), nullable=False, index=True)
    owner_id = Column(Integer, nullable=False, index=True)

    # Widget channel (embeddable chat). ``widget_key`` is the unguessable
    # credential distributed in the embed snippet; ``allowed_domains`` is a
    # browser-level origin restriction on top of it.
    widget_enabled = Column(Boolean, default=False, nullable=False)
    allowed_domains = Column(JSON, nullable=True, default=list)
    widget_key = Column(String(255), nullable=True, unique=True, index=True)

    # Shareable-link channel (public chat page). Like widget_key, the token
    # is a lookup credential and must resolve to at most one deployment.
    share_enabled = Column(Boolean, default=False, nullable=False)
    share_token = Column(String(255), nullable=True, unique=True, index=True)
    share_updated_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
