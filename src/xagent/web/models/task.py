import enum
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


class TaskStatus(enum.Enum):
    """Task status enumeration"""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class DAGExecutionPhase(enum.Enum):
    """DAG execution phase enumeration"""

    PLANNING = "planning"
    EXECUTING = "executing"
    CHECKING = "checking"
    COMPLETED = "completed"
    FAILED = "failed"


class StepStatus(enum.Enum):
    """Step status enumeration"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ANALYZED = "analyzed"


class ExecutionMode(enum.Enum):
    """Execution mode enumeration"""

    FLASH = "flash"  # Simple, quick tasks (single_call pattern)
    BALANCED = "balanced"  # Most everyday tasks (react pattern)
    THINK = "think"  # Complex, multi-step tasks (dag_plan_execute pattern)


class AgentType(enum.Enum):
    """Agent type enumeration"""

    STANDARD = "standard"  # Standard purpose agent
    TEXT2SQL = "text2sql"  # Text2SQL agent
    # Future agent types can be added here
    # CODE_ASSISTANT = "code_assistant"
    # DATA_ANALYSIS = "data_analysis"


class Task(Base):  # type: ignore
    """Task model"""

    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    status: Any = Column(Enum(TaskStatus), default=TaskStatus.PENDING)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Model configuration
    model_name = Column(String(255), nullable=True)  # Main model used for the task
    small_fast_model_name = Column(
        String(255), nullable=True
    )  # Small/fast model if configured
    visual_model_name = Column(String(255), nullable=True)  # Visual model if configured
    compact_model_name = Column(
        String(255), nullable=True
    )  # Compact model if configured

    # Internal model identifiers (preferred over *_model_name for selection)
    model_id = Column(String(255), nullable=True)
    small_fast_model_id = Column(String(255), nullable=True)
    visual_model_id = Column(String(255), nullable=True)
    compact_model_id = Column(String(255), nullable=True)

    # Agent configuration
    agent_id = Column(
        Integer, ForeignKey("agents.id"), nullable=True
    )  # Agent Builder agent ID
    agent_type = Column(
        String(20), default=AgentType.STANDARD.value, nullable=True
    )  # SQLite compatible
    agent_config = Column(JSON, nullable=True)  # Agent-specific configuration

    # Execution mode configuration
    execution_mode = Column(
        String(20), default=ExecutionMode.BALANCED.value, nullable=True
    )  # "flash" | "balanced" | "think"
    process_description = Column(
        Text, nullable=True
    )  # Process mode: detailed process description
    examples = Column(JSON, nullable=True)  # Process mode: input/output examples

    # Channel configuration
    channel_id = Column(
        Integer, ForeignKey("user_channels.id", ondelete="SET NULL"), nullable=True
    )
    channel_name = Column(String(100), nullable=True)

    # Token usage statistics
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    llm_calls = Column(Integer, default=0)
    token_usage_details = Column(JSON, nullable=True)  # Detailed breakdown

    @property
    def execution_mode_enum(self) -> ExecutionMode:
        """Get execution_mode as enum with fallback"""
        try:
            return (
                ExecutionMode(self.execution_mode)
                if self.execution_mode
                else ExecutionMode.BALANCED
            )
        except ValueError:
            return ExecutionMode.BALANCED

    @execution_mode_enum.setter
    def execution_mode_enum(self, value: ExecutionMode) -> None:
        """Set execution_mode from enum"""
        setattr(self, "execution_mode", value.value if value else None)

    @property
    def agent_type_enum(self) -> AgentType:
        """Get agent_type as enum with fallback"""
        try:
            return AgentType(self.agent_type) if self.agent_type else AgentType.STANDARD
        except ValueError:
            return AgentType.STANDARD

    @agent_type_enum.setter
    def agent_type_enum(self, value: AgentType) -> None:
        """Set agent_type from enum"""
        # Use setattr to avoid mypy Column type checking
        setattr(self, "agent_type", value.value if value else None)

    # Relationships
    user = relationship("User", back_populates="tasks")
    dag_executions = relationship("DAGExecution", back_populates="task")
    chat_messages = relationship(
        "TaskChatMessage",
        back_populates="task",
        cascade="all, delete-orphan",
        order_by="TaskChatMessage.id",
    )
    uploaded_files = relationship("UploadedFile", back_populates="task")

    def __repr__(self) -> str:
        return f"<Task(id={self.id}, title='{self.title}', status='{self.status}')>"


class DAGExecution(Base):  # type: ignore
    """DAG execution status model"""

    __tablename__ = "dag_executions"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False, unique=True)
    phase: Column[DAGExecutionPhase] = Column(
        Enum(DAGExecutionPhase), default=DAGExecutionPhase.PLANNING
    )
    progress_percentage = Column(Float, default=0.0)
    completed_steps = Column(Integer, default=0)
    total_steps = Column(Integer, default=0)
    execution_time = Column(Float)  # Total execution time in seconds
    start_time = Column(DateTime(timezone=True))
    end_time = Column(DateTime(timezone=True))
    current_plan = Column(JSON)  # Store the current plan data
    skipped_steps = Column(JSON)  # Store list of skipped step IDs
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    task = relationship("Task", back_populates="dag_executions")

    def __repr__(self) -> str:
        return f"<DAGExecution(id={self.id}, task_id={self.task_id}, phase='{self.phase}')>"


class TraceEvent(Base):  # type: ignore
    """Unified trace event model for consistent storage and WebSocket transmission"""

    __tablename__ = "trace_events"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    build_id = Column(String(255), nullable=True, index=True)  # Build session ID
    event_id = Column(String(255), nullable=False)  # UUID for the trace event
    event_type = Column(
        String(100), nullable=False
    )  # Event type string (e.g., "dag_execute_start")
    timestamp = Column(DateTime(timezone=True), nullable=False)  # Event timestamp
    step_id = Column(String(255), nullable=True)  # Optional step ID
    parent_event_id = Column(
        String(255), nullable=True
    )  # Parent event ID for hierarchy
    data = Column(JSON, nullable=False)  # Event data payload

    # Relationships
    task = relationship("Task")

    def __repr__(self) -> str:
        return f"<TraceEvent(id={self.id}, event_type='{self.event_type}', task_id={self.task_id})>"
