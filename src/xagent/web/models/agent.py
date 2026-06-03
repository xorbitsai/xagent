"""Agent Builder models for creating custom AI agents."""

import enum
from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime
from sqlalchemy import Enum as SQLEnum
from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from .database import Base


class AgentStatus(enum.Enum):
    """Agent status enumeration"""

    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class AgentOrigin(enum.Enum):
    """Where an agent came from."""

    USER = "user"
    WORKFORCE_GENERATED_MANAGER = "workforce_generated_manager"


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
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    instructions = Column(Text, nullable=True)  # System prompt/instructions

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
