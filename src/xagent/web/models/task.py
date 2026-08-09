import enum
from typing import Any, Collection

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


class TaskStatus(enum.Enum):
    """Task status enumeration"""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_FOR_USER = "waiting_for_user"
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
    AUTO = "auto"  # Let agent choose final answer, ReAct, or DAG


class AgentType(enum.Enum):
    """Agent type enumeration"""

    STANDARD = "standard"  # Standard purpose agent


class Task(Base):  # type: ignore
    """Task model"""

    __tablename__ = "tasks"
    __table_args__ = (
        Index("ix_tasks_agent_id_source", "agent_id", "source"),
        Index(
            "ix_tasks_status_lease_expires_at",
            "status",
            "lease_expires_at",
            "id",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    # sqlalchemy.Enum(TaskStatus) with no values_callable persists member
    # *names* (e.g. "WAITING_FOR_USER"), not member values
    # ("waiting_for_user"). validate_strings=True rejects a raw string that
    # is not one of those names at bind time (StatementError/LookupError,
    # symmetric on SQLite and PostgreSQL); it is a second-layer guard behind
    # TaskStatusPredicate below and does not change the DDL. It does not
    # cover raw-SQL writes that bypass the ORM/Core bind path -- those stay
    # covered by the storage-layer sentinels in
    # tests/web/services/test_task_status_storage.py.
    status: Any = Column(
        Enum(TaskStatus, validate_strings=True), default=TaskStatus.PENDING
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    runner_id = Column(String(255), nullable=True)
    lease_expires_at = Column(DateTime(timezone=True), nullable=True)
    last_heartbeat_at = Column(DateTime(timezone=True), nullable=True)
    last_checkpoint_event_id = Column(String(255), nullable=True)
    # Exact-row anchor: the primary key of the trace_events row that
    # last_checkpoint_event_id names. Readers that find this set can load the
    # checkpoint by primary key instead of re-resolving the legacy string
    # column against the row set. This FK forms a cycle with
    # trace_events.task_id -> tasks.id: it must be named, or an unnamed
    # constraint in that cycle raises CircularDependencyError on backends
    # whose create_all/drop_all doesn't go through Alembic (e.g. PostgreSQL
    # in this repo's CI and dev paths); and it must be use_alter=True, or
    # SQLAlchemy can't topologically sort DROP order across the cycle and a
    # SQLite database with FK enforcement on (this repo's default -- see
    # apply_sqlite_concurrency_pragmas) fails mid-drop_all with a DROP TABLE
    # error on an unrelated table further down the (corrupted) drop order.
    last_checkpoint_trace_event_id = Column(
        Integer,
        ForeignKey(
            "trace_events.id",
            name="fk_tasks_last_checkpoint_trace_event_id",
            use_alter=True,
        ),
        nullable=True,
    )
    # Monotonic task-control identity. ``run_id`` changes for each new turn,
    # while pause/resume transitions retain it and advance ``state_version``.
    # Clients use the version to ignore stale WebSocket status events.
    run_id = Column(String(64), nullable=True, index=True)
    state_version = Column(Integer, nullable=False, default=0, server_default="0")
    control_state = Column(
        String(32), nullable=False, default="idle", server_default="idle"
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
    connector_runtime_selected_refs = Column(JSON, nullable=True, default=list)

    # Execution mode configuration
    execution_mode = Column(
        String(20), default=ExecutionMode.AUTO.value, nullable=True
    )  # "flash" | "balanced" | "think" | "auto"
    process_description = Column(
        Text, nullable=True
    )  # Process mode: detailed process description
    examples = Column(JSON, nullable=True)  # Process mode: input/output examples

    # Channel configuration
    channel_id = Column(
        Integer, ForeignKey("user_channels.id", ondelete="SET NULL"), nullable=True
    )
    channel_name = Column(String(100), nullable=True)
    # External Telegram sender that owns this channel task. A channel may allow
    # multiple Telegram accounts, so channel_id alone is not an authorization
    # boundary for conversation history.
    telegram_user_id = Column(String(32), nullable=True, index=True)

    # Token usage statistics
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    llm_calls = Column(Integer, default=0)
    token_usage_details = Column(JSON, nullable=True)  # Detailed breakdown

    # ----- SDK surface fields (read/written by /v1/* endpoints) -----
    # The four columns below are populated by SDK-driven task lifecycles
    # (see web/api/v1/tasks.py). Legacy task creation paths
    # (chat.py / websocket.py / widget.py) intentionally leave them at
    # their defaults: legacy consumers don't read these, and SDK
    # consumers only see tasks they themselves created.

    # Latest-turn user input as plaintext. Updated each time the SDK
    # appends a message via POST /v1/chat/tasks/{id}/messages.
    input = Column(Text, nullable=True)

    # Latest-turn assistant output as plaintext, written when the
    # background execution finishes a turn successfully.
    output = Column(Text, nullable=True)

    # Last failure reason when status transitions to FAILED.
    error_message = Column(Text, nullable=True)

    # Call origin classifier: 'internal' (web UI / WS / legacy),
    # 'sdk' (POST /v1/chat/tasks), 'a2a' (agent-to-agent calls),
    # 'trigger' (agent triggers), 'widget' (embedded chat widget), or
    # 'shared_link' (public share chat).
    # Default 'internal' so legacy code paths -- which never specify
    # this field on Task(...) -- are auto-classified correctly. Both
    # ``default`` (Python-level, fires on ORM INSERT) and
    # ``server_default`` (DB-level DDL, fires for raw SQL INSERT and
    # for any future ALTER ADD COLUMN against this Column) are set so
    # the schema produced by ``Base.metadata.create_all()`` (used by
    # dev/test init_db) stays identical to what alembic produces in
    # production. Indexed for adoption-metrics queries
    # (SELECT count(*) FROM tasks WHERE source='sdk' AND created_at>...).
    source = Column(
        String(20),
        default="internal",
        server_default="internal",
        nullable=True,
        index=True,
    )

    # Visibility flag for discovery surfaces such as sidebar/history/search.
    # Hidden tasks still use normal owner/admin access by exact task_id.
    is_visible = Column(
        Boolean,
        default=True,
        server_default="1",
        nullable=False,
        index=True,
    )

    @property
    def execution_mode_enum(self) -> ExecutionMode:
        """Get execution_mode as enum with fallback"""
        try:
            return (
                ExecutionMode(self.execution_mode)
                if self.execution_mode
                else ExecutionMode.AUTO
            )
        except ValueError:
            return ExecutionMode.AUTO

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
    agent = relationship(
        "Agent",
        primaryjoin="Task.agent_id == Agent.id",
        foreign_keys=[agent_id],
        viewonly=True,
    )
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


def _require_task_status(status: Any) -> TaskStatus:
    if not isinstance(status, TaskStatus):
        raise TypeError(f"expected TaskStatus, got {type(status).__name__}")
    return status


def _require_task_status_members(
    statuses: Collection[TaskStatus],
) -> tuple[TaskStatus, ...]:
    if statuses is None or isinstance(statuses, (TaskStatus, str)):
        raise TypeError(
            "expected a collection of TaskStatus members, got "
            f"{type(statuses).__name__}; wrap a single member in a list"
        )
    members = tuple(statuses)
    if not members:
        raise ValueError("task status predicate requires at least one TaskStatus")
    for status in members:
        _require_task_status(status)
    return members


class TaskStatusPredicate:
    """Typed entry points for every SQL predicate and write value against
    ``Task.status``.

    The column stores ``TaskStatus`` member names, not member values (see
    the ``status`` column comment). A raw string literal reaching an
    ORM/Core comparison or write fails at bind time with ``StatementError``
    wrapping ``LookupError`` (see the column comment above for why that
    holds on both backends); raw ``text()`` SQL bypasses that bind layer
    and is covered instead by the storage-layer sentinels in
    ``tests/web/services/test_task_status_storage.py``.

    A typed ``TaskStatus`` member compared directly against ``Task.status``
    is legitimate and common in this codebase; it compiles to exactly what
    the methods below compile to (pinned by the equivalence tests in
    ``test_task_status_storage.py``). The methods below are the required
    entry point for value-sourced or dynamic statuses -- anything that is
    not a literal ``TaskStatus`` member in the source -- because they turn
    a non-``TaskStatus`` input into a construction-time ``TypeError``
    instead of a query-time failure. What is actually enforced repo-wide is
    narrower still: the literal-predicate scan in
    ``tests/web/services/test_task_status_literal_predicates.py`` bans a
    raw string literal placed beside ``Task.status``; adoption of the
    methods below at the safe-but-unconverted typed sites above is not
    enforced and is not claimed.
    """

    @staticmethod
    def eq(status: TaskStatus) -> Any:
        """``Task.status == status``."""
        _require_task_status(status)
        return Task.status == status

    @staticmethod
    def ne(status: TaskStatus) -> Any:
        """``Task.status != status``."""
        _require_task_status(status)
        return Task.status != status

    @staticmethod
    def in_(statuses: Collection[TaskStatus]) -> Any:
        """``Task.status IN (...)``; a single member compiles to ``==``."""
        members = _require_task_status_members(statuses)
        if len(members) == 1:
            return Task.status == members[0]
        return Task.status.in_(members)

    @staticmethod
    def not_in(statuses: Collection[TaskStatus]) -> Any:
        """``Task.status NOT IN (...)``; a single member compiles to ``!=``."""
        members = _require_task_status_members(statuses)
        if len(members) == 1:
            return Task.status != members[0]
        return Task.status.notin_(members)

    @staticmethod
    def value(status: TaskStatus) -> TaskStatus:
        """Validate one status before it is passed to ``.values(status=...)``.

        Every call site today already passes a typed ``TaskStatus``, so this
        check does no runtime work under a type-checked call graph; it exists
        for callers whose status is dynamically sourced, and to give guarded
        write sites a compliant expression to wrap.
        """
        return _require_task_status(status)


task_status_predicate = TaskStatusPredicate()


class TaskConnectorRuntimeContext(Base):  # type: ignore
    """Task-bound non-secret runtime context for one connector."""

    __tablename__ = "task_connector_runtime_contexts"
    __table_args__ = (
        UniqueConstraint(
            "task_id",
            "connector_type",
            "connector_id",
            name="uq_task_connector_runtime_contexts_ref",
        ),
        Index("ix_task_connector_runtime_contexts_task_id", "task_id"),
    )

    id = Column(Integer, primary_key=True)
    task_id = Column(
        Integer,
        ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=False,
    )
    connector_type = Column(String(32), nullable=False)
    connector_id = Column(Integer, nullable=False)
    context = Column(JSON, nullable=False)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    task = relationship("Task")

    def __repr__(self) -> str:
        return (
            "<TaskConnectorRuntimeContext("
            f"task_id={self.task_id}, "
            f"connector_type='{self.connector_type}', "
            f"connector_id={self.connector_id})>"
        )


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
    # Checkpoint pruning and latest-checkpoint loading both filter by
    # task_id + event_type on every checkpoint write/resume.
    __table_args__ = (
        Index("ix_trace_events_task_id_event_type", "task_id", "event_type"),
    )

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

    # Relationships. foreign_keys is explicit because tasks and trace_events
    # now have two FK paths between them (this task_id, and
    # Task.last_checkpoint_trace_event_id pointing back) -- without it,
    # SQLAlchemy can't pick a join condition for this relationship.
    task = relationship("Task", foreign_keys=[task_id])

    def __repr__(self) -> str:
        return f"<TraceEvent(id={self.id}, event_type='{self.event_type}', task_id={self.task_id})>"


class TraceMessageBlob(Base):  # type: ignore
    """Deduplicated message payload referenced by checkpoint trace events."""

    __tablename__ = "trace_message_blobs"
    __table_args__ = (
        UniqueConstraint(
            "task_id",
            "message_hash",
            name="uq_trace_message_blobs_task_hash",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False, index=True)
    execution_id = Column(String(255), nullable=False, index=True)
    message_hash = Column(String(80), nullable=False)
    message_data = Column(JSON, nullable=False)
    message_bytes = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    task = relationship("Task")

    def __repr__(self) -> str:
        return (
            "<TraceMessageBlob("
            f"id={self.id}, task_id={self.task_id}, message_hash='{self.message_hash}'"
            ")>"
        )


class TraceCheckpointBlob(Base):  # type: ignore
    """Deduplicated checkpoint field payload referenced by trace events."""

    __tablename__ = "trace_checkpoint_blobs"
    __table_args__ = (
        UniqueConstraint(
            "task_id",
            "blob_kind",
            "blob_hash",
            name="uq_trace_checkpoint_blobs_task_kind_hash",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False, index=True)
    execution_id = Column(String(255), nullable=False, index=True)
    blob_kind = Column(String(255), nullable=False, index=True)
    blob_hash = Column(String(80), nullable=False)
    blob_data = Column(JSON, nullable=False)
    blob_bytes = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    task = relationship("Task")

    def __repr__(self) -> str:
        return (
            "<TraceCheckpointBlob("
            f"id={self.id}, task_id={self.task_id}, "
            f"blob_kind='{self.blob_kind}', blob_hash='{self.blob_hash}'"
            ")>"
        )
