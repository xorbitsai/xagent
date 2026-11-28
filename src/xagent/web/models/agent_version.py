"""Version rows of an agent's versioned configuration.

One row per recorded state of the eight versioned fields of an ``agents`` row
(``xagent.web.services.agent_versions.VERSIONED_FIELDS``). Rows are written
only by ``apply_agent_change``; afterwards their ``content`` is changed only by
``rewrite_agent_references`` (a connector or knowledge base was renamed) and by
migrations that normalize how a versioned field is spelled.

The current version number of an agent is the largest ``version_no`` of its
rows, or 0 when it has none. Having no rows is a normal state: agents that
existed before versioning, or were written without ``apply_agent_change``,
get their first row the next time that function is called for them.
"""

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.sql import func

from .database import Base

AGENT_VERSION_UNIQUE_CONSTRAINT = "uq_agent_versions_agent_id_version_no"


class AgentVersion(Base):  # type: ignore
    __tablename__ = "agent_versions"
    __table_args__ = (
        # Leading ``agent_id`` also serves the foreign key and the
        # "newest version of this agent" lookup.
        UniqueConstraint(
            "agent_id", "version_no", name=AGENT_VERSION_UNIQUE_CONSTRAINT
        ),
        {"sqlite_autoincrement": True},
    )

    id = Column(Integer, primary_key=True)
    # CASCADE: deleting an agent deletes its history, so a later agent that
    # SQLite gives the same id starts with no rows.
    agent_id = Column(
        Integer, ForeignKey("agents.id", ondelete="CASCADE"), nullable=False
    )
    version_no = Column(Integer, nullable=False)
    # ``initial``, ``save`` or ``untracked``.
    kind = Column(String(32), nullable=False)
    # ``user``, ``api_key``, ``workspace_key``, ``system`` or ``unspecified``.
    actor_kind = Column(String(16), nullable=False)
    # SET NULL keeps the row when the user is deleted; indexed because that
    # delete has to find the rows to clear.
    actor_user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    # Personal key prefix, or the id of a workspace key.
    actor_key_prefix = Column(String(32), nullable=True)
    # Entry point, e.g. ``web``, ``builder_tool``, ``v1``, ``import``,
    # ``baseline``.
    actor_source = Column(String(64), nullable=True)
    # The eight versioned fields as stored on the agent row.
    content = Column(JSON, nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
