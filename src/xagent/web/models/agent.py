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


class ExecutionMode(enum.Enum):
    """Agent execution mode enumeration"""

    SIMPLE = "simple"  # Reserved: single LLM call (not implemented yet)
    REACT = "react"  # ReAct pattern for reasoning and acting
    GRAPH = "graph"  # DAG/Graph plan-execute pattern for complex tasks


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
        String(20), nullable=False, default="react"
    )  # Execution mode: simple, react, graph
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

    # Status
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

    def __repr__(self) -> str:
        return f"<Agent(id={self.id}, name='{self.name}', status='{self.status}')>"
