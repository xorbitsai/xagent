"""Agent Builder models for creating custom AI agents."""

import enum
from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime
from sqlalchemy import Enum as SQLEnum
from sqlalchemy import ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.orm import relationship

from .database import Base

# Per-user name uniqueness, excluding workforce-generated manager agents
# (which are allowed to share names - see agent_name_exists). Declared here
# so brand-new databases get it via Base.metadata.create_all(); existing
# databases get the same index from the
# 20260728_add_agent_template_id_and_name_uniqueness migration, which also
# dedupes any pre-existing collisions first.
#
# This is keyed on (user_id, name) only, so it mirrors agent_name_exists
# exactly in standalone xagent. When a SaaS team-scope hook is installed,
# agent_name_exists becomes a team-wide check (see agent_team_scope.py) that
# this per-user index does not enforce - it only guarantees a single user
# can't race a duplicate name past themselves, not across teammates.
_NON_WORKFORCE_MANAGER_CLAUSE = text("origin != 'workforce_generated_manager'")

# Shared with agent_management.py, which inspects a raised IntegrityError's
# message for this name to distinguish this specific violation from any other
# IntegrityError (e.g. a widget_key collision or an unrelated FK failure).
AGENT_NAME_UNIQUE_INDEX = "uq_agents_user_id_name_active"

# Scopes the (user_id, template_id) uniqueness below to agents created by the
# /task template quick-access resolve flow specifically - the plain
# POST /from-template create path (workforce-builder UI) deliberately mints
# multiple named instances of one template and must not be constrained by
# it. Also shared with agent_management.py's IntegrityError matching.
_QUICK_ACCESS_ORIGIN = "template_quick_access"
_QUICK_ACCESS_ORIGIN_CLAUSE = text(f"origin = '{_QUICK_ACCESS_ORIGIN}'")
AGENT_TEMPLATE_QUICK_ACCESS_UNIQUE_INDEX = "uq_agents_user_id_template_id_quick_access"


class AgentStatus(enum.Enum):
    """Agent status enumeration"""

    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class AgentOrigin(enum.Enum):
    """Where an agent came from."""

    USER = "user"
    WORKFORCE_GENERATED_MANAGER = "workforce_generated_manager"
    # The /task template quick-access get-or-create flow
    # (AgentManagementService.resolve_agent_from_template). Distinct from
    # USER so its (user_id, template_id) reuse query can't adopt an
    # unrelated agent the workforce-builder UI created from the same
    # template under a user-chosen name (PR review finding B4).
    TEMPLATE_QUICK_ACCESS = _QUICK_ACCESS_ORIGIN


class ExecutionMode(enum.Enum):
    """Agent execution mode enumeration"""

    FLASH = "flash"  # Simple, quick tasks (single_call pattern)
    BALANCED = "balanced"  # Most everyday tasks (react pattern)
    THINK = "think"  # Complex, multi-step tasks (dag_plan_execute pattern)
    AUTO = "auto"  # Let agent choose final answer, ReAct, or DAG


class Agent(Base):  # type: ignore
    """Custom AI Agent model for agent builder"""

    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    # SaaS-overlay team ownership. No DB ForeignKey: the ``teams`` table is a
    # SaaS-only overlay table and does not exist in standalone xagent. When the
    # SaaS agent-team-scope hook is not installed this stays NULL and agents
    # remain purely user-owned.
    team_id = Column(Integer, nullable=True, index=True)
    # Team-internal visibility. "team" (default) = every team member; "admins"
    # = only team admins (and the team API key) can see/manage/run it. Only a
    # team admin may switch this. Standalone xagent (no team-scope hook) ignores
    # it. Distinct from external share_enabled/widget exposure.
    visibility = Column(
        String(20), nullable=False, default="team", server_default="team"
    )
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    instructions = Column(Text, nullable=True)  # System prompt/instructions
    # Built-in template id this agent was instantiated from (e.g. via
    # "/api/agents/from-template"), or NULL for agents built from scratch.
    # Lets create-or-reuse flows key off a stable id instead of the
    # user-editable display name.
    template_id = Column(String(255), nullable=True, index=True)

    # Configuration
    execution_mode = Column(
        String(20), nullable=False, default="balanced"
    )  # Execution mode: flash, balanced, think, auto
    models = Column(
        JSON, nullable=True
    )  # Model config: {general: id, small_fast: id, visual: id, compact: id}
    knowledge_bases = Column(JSON, nullable=True, default=list)  # List of KB names
    skills = Column(JSON, nullable=True, default=list)  # List of skill names
    tool_categories = Column(
        JSON, nullable=True, default=list
    )  # List of tool categories
    suggested_prompts = Column(
        JSON, nullable=True, default=list
    )  # List of suggested prompt examples for users

    # Visual
    logo_url = Column(String(500), nullable=True)

    # Widget Config
    widget_enabled = Column(Boolean, default=True, nullable=False)
    allowed_domains = Column(
        JSON, nullable=True, default=list
    )  # List of allowed domains for the widget
    # Unguessable per-agent credential distributed in the embed snippet; the
    # real access gate for widget guest tokens (allowed_domains is only a
    # browser-level restriction). Owner-visible, rotatable.
    widget_key = Column(String(255), nullable=True, unique=True, index=True)
    share_enabled = Column(Boolean, default=False, nullable=False)
    share_token = Column(String(255), nullable=True, index=True)
    share_updated_at = Column(DateTime(timezone=True), nullable=True)

    # Status
    origin = Column(
        String(50),
        default=AgentOrigin.USER.value,
        nullable=False,
        index=True,
    )
    status: AgentStatus = Column(
        SQLEnum(AgentStatus, values_callable=lambda obj: [e.value for e in obj]),
        default=AgentStatus.DRAFT,
        nullable=False,
    )  # type: ignore[assignment]
    published_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index(
            AGENT_NAME_UNIQUE_INDEX,
            "user_id",
            "name",
            unique=True,
            sqlite_where=_NON_WORKFORCE_MANAGER_CLAUSE,
            postgresql_where=_NON_WORKFORCE_MANAGER_CLAUSE,
        ),
        # Backs the /task template quick-access resolve flow's atomicity:
        # scoped to TEMPLATE_QUICK_ACCESS origin only, so a workforce-builder
        # agent built from the same template (a deliberate, user-named,
        # possibly-multiple instance) never collides with it. See
        # AgentManagementService._resolve_agent_from_template_sync and PR
        # review findings B1/B2/D2/D3.
        Index(
            AGENT_TEMPLATE_QUICK_ACCESS_UNIQUE_INDEX,
            "user_id",
            "template_id",
            unique=True,
            sqlite_where=_QUICK_ACCESS_ORIGIN_CLAUSE,
            postgresql_where=_QUICK_ACCESS_ORIGIN_CLAUSE,
        ),
    )

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.now, nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        default=datetime.now,
        onupdate=datetime.now,
        nullable=False,
    )

    # Relationships
    user = relationship("User", back_populates="agents")
    triggers = relationship(
        "AgentTrigger", back_populates="agent", cascade="all, delete-orphan"
    )

    @property
    def is_workforce_generated_manager(self) -> bool:
        origin = getattr(self.origin, "value", self.origin)
        return bool(origin == AgentOrigin.WORKFORCE_GENERATED_MANAGER.value)

    def __repr__(self) -> str:
        return f"<Agent(id={self.id}, name='{self.name}', status='{self.status}')>"


def is_workforce_generated_manager_agent(agent: object | None) -> bool:
    if agent is None:
        return False

    marker = getattr(agent, "is_workforce_generated_manager", None)
    if isinstance(marker, bool):
        return marker

    origin = getattr(agent, "origin", None)
    origin = getattr(origin, "value", origin)
    return origin == AgentOrigin.WORKFORCE_GENERATED_MANAGER.value
