"""Generic tracing module for tracking events in the xagent system."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class TraceScope(Enum):
    """Defines the scope of trace events for clear task/step attribution."""

    TASK = "task"  # Task-level events
    STEP = "step"  # Step-level events
    ACTION = "action"  # Action-level events (within steps)
    SYSTEM = "system"  # System-level events


class TraceAction(Enum):
    """Defines the action type of trace events."""

    START = "start"
    END = "end"
    UPDATE = "update"
    ERROR = "error"
    INFO = "info"


class TraceCategory(Enum):
    """Defines the category of trace events."""

    DAG = "dag"  # DAG execution events
    DAG_PLAN = "dag_plan"  # DAG planning events
    REACT = "react"  # ReAct pattern events
    LLM = "llm"  # LLM call events
    TOOL = "tool"  # Tool execution events
    VISUALIZATION = "visualization"  # UI update events
    MESSAGE = "message"  # User/AI message events
    MEMORY_GENERATE = "memory_generate"  # Memory generation events
    MEMORY_STORE = "memory_store"  # Memory storage events
    MEMORY_RETRIEVE = "memory_retrieve"  # Memory retrieval events
    COMPACT = "compact"  # Context compaction events
    SKILL = "skill"  # Skill selection events
    GENERAL = "general"  # General events


class TraceEventType:
    """Unified trace event type that combines scope, action, and category."""

    def __init__(self, scope: TraceScope, action: TraceAction, category: TraceCategory):
        self.scope = scope
        self.action = action
        self.category = category

    @property
    def value(self) -> str:
        return f"{self.scope.value}_{self.action.value}_{self.category.value}"

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TraceEventType):
            return False
        return (
            self.scope == other.scope
            and self.action == other.action
            and self.category == other.category
        )

    def __hash__(self) -> int:
        return hash((self.scope, self.action, self.category))


# Predefined event types for convenience
# Task-level events
TASK_START_DAG = TraceEventType(TraceScope.TASK, TraceAction.START, TraceCategory.DAG)
TASK_END_DAG = TraceEventType(TraceScope.TASK, TraceAction.END, TraceCategory.DAG)
TASK_START_REACT = TraceEventType(
    TraceScope.TASK, TraceAction.START, TraceCategory.REACT
)
TASK_END_REACT = TraceEventType(TraceScope.TASK, TraceAction.END, TraceCategory.REACT)
TASK_START_GENERAL = TraceEventType(
    TraceScope.TASK, TraceAction.START, TraceCategory.GENERAL
)
TASK_END_GENERAL = TraceEventType(
    TraceScope.TASK, TraceAction.END, TraceCategory.GENERAL
)
TASK_ERROR = TraceEventType(TraceScope.TASK, TraceAction.ERROR, TraceCategory.GENERAL)

# AI message event (for chat responses)
AI_MESSAGE = TraceEventType(TraceScope.TASK, TraceAction.END, TraceCategory.MESSAGE)

# Step-level events
STEP_START_DAG = TraceEventType(TraceScope.STEP, TraceAction.START, TraceCategory.DAG)
STEP_END_DAG = TraceEventType(TraceScope.STEP, TraceAction.END, TraceCategory.DAG)
STEP_START_REACT = TraceEventType(
    TraceScope.STEP, TraceAction.START, TraceCategory.REACT
)
STEP_END_REACT = TraceEventType(TraceScope.STEP, TraceAction.END, TraceCategory.REACT)
STEP_ERROR = TraceEventType(TraceScope.STEP, TraceAction.ERROR, TraceCategory.GENERAL)

# Memory-related events
MEMORY_GENERATE_START = TraceEventType(
    TraceScope.TASK, TraceAction.START, TraceCategory.MEMORY_GENERATE
)
MEMORY_GENERATE_END = TraceEventType(
    TraceScope.TASK, TraceAction.END, TraceCategory.MEMORY_GENERATE
)
MEMORY_STORE_START = TraceEventType(
    TraceScope.TASK, TraceAction.START, TraceCategory.MEMORY_STORE
)
MEMORY_STORE_END = TraceEventType(
    TraceScope.TASK, TraceAction.END, TraceCategory.MEMORY_STORE
)
MEMORY_RETRIEVE_START = TraceEventType(
    TraceScope.TASK, TraceAction.START, TraceCategory.MEMORY_RETRIEVE
)
MEMORY_RETRIEVE_END = TraceEventType(
    TraceScope.TASK, TraceAction.END, TraceCategory.MEMORY_RETRIEVE
)

# Compact-related events
COMPACT_START = TraceEventType(
    TraceScope.ACTION, TraceAction.START, TraceCategory.COMPACT
)
COMPACT_END = TraceEventType(TraceScope.ACTION, TraceAction.END, TraceCategory.COMPACT)

# Action-level events (consolidated)
ACTION_START_TOOL = TraceEventType(
    TraceScope.ACTION, TraceAction.START, TraceCategory.TOOL
)
ACTION_END_TOOL = TraceEventType(TraceScope.ACTION, TraceAction.END, TraceCategory.TOOL)
ACTION_START_LLM = TraceEventType(
    TraceScope.ACTION, TraceAction.START, TraceCategory.LLM
)
ACTION_END_LLM = TraceEventType(TraceScope.ACTION, TraceAction.END, TraceCategory.LLM)

# System-level events
SYSTEM_VISUALIZATION_UPDATE = TraceEventType(
    TraceScope.SYSTEM, TraceAction.UPDATE, TraceCategory.VISUALIZATION
)
SYSTEM_INFO = TraceEventType(TraceScope.SYSTEM, TraceAction.INFO, TraceCategory.GENERAL)


class TraceEvent:
    """Represents a single trace event with clear task/step attribution."""

    def __init__(
        self,
        event_type: TraceEventType,
        task_id: Optional[str] = None,
        step_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        data: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
    ):
        self.id = str(uuid4())
        self.event_type = event_type
        self.task_id = task_id
        self.step_id = step_id
        self.timestamp = timestamp or datetime.now(timezone.utc).timestamp()
        self.data = data or {}
        self.parent_id = parent_id

        # Validate scope requirements
        self._validate_scope()

    def _validate_scope(self) -> None:
        """Validate that required fields are present based on event scope."""
        if self.event_type.scope == TraceScope.TASK and not self.task_id:
            raise ValueError(
                f"Task-level event {self.event_type.value} requires task_id"
            )
        if self.event_type.scope == TraceScope.STEP and not self.step_id:
            raise ValueError(
                f"Step-level event {self.event_type.value} requires step_id"
            )
        if self.event_type.scope == TraceScope.ACTION and not self.step_id:
            raise ValueError(
                f"Action-level event {self.event_type.value} requires step_id"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace event to dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "scope": self.event_type.scope.value,
            "action": self.event_type.action.value,
            "category": self.event_type.category.value,
            "timestamp": self.timestamp,
            "task_id": self.task_id,
            "step_id": self.step_id,
            "data": self.data,
            "parent_id": self.parent_id,
        }


class TraceHandler(ABC):
    """Abstract base class for trace handlers."""

    @abstractmethod
    async def handle_event(self, event: TraceEvent) -> None:
        """Handle a trace event."""
        pass


class BaseTraceHandler(TraceHandler):
    """Base trace handler with common functionality."""

    def __init__(self) -> None:
        self.event_transformers = {
            TraceScope.TASK: self._handle_task_event,
            TraceScope.STEP: self._handle_step_event,
            TraceScope.ACTION: self._handle_action_event,
            TraceScope.SYSTEM: self._handle_system_event,
        }

    async def handle_event(self, event: TraceEvent) -> None:
        """Handle a trace event by delegating to scope-specific handler."""
        try:
            handler = self.event_transformers.get(event.event_type.scope)
            if handler:
                await handler(event)
            else:
                logger.warning(f"No handler for event scope: {event.event_type.scope}")
        except Exception as e:
            logger.error(f"Error handling event {event.event_type.value}: {e}")
            # Don't re-raise to avoid breaking the main execution

    async def _handle_task_event(self, event: TraceEvent) -> None:
        """Handle task-level events. Override in subclasses."""
        pass

    async def _handle_step_event(self, event: TraceEvent) -> None:
        """Handle step-level events. Override in subclasses."""
        pass

    async def _handle_action_event(self, event: TraceEvent) -> None:
        """Handle action-level events. Override in subclasses."""
        pass

    async def _handle_system_event(self, event: TraceEvent) -> None:
        """Handle system-level events. Override in subclasses."""
        pass


class ConsoleTraceHandler(BaseTraceHandler):
    """Trace handler that logs events to console with clear scope information."""

    async def _handle_task_event(self, event: TraceEvent) -> None:
        """Handle task-level events."""
        logger.info(
            f"[TASK] {event.event_type.action.value.upper()} {event.event_type.category.value.upper()} - Task {event.task_id} - {event.data}"
        )

    async def _handle_step_event(self, event: TraceEvent) -> None:
        """Handle step-level events."""
        logger.info(
            f"[STEP] {event.event_type.action.value.upper()} {event.event_type.category.value.upper()} - Step {event.step_id} - {event.data}"
        )

    async def _handle_action_event(self, event: TraceEvent) -> None:
        """Handle action-level events."""
        logger.info(
            f"[ACTION] {event.event_type.action.value.upper()} {event.event_type.category.value.upper()} - Step {event.step_id} - {event.data}"
        )

    async def _handle_system_event(self, event: TraceEvent) -> None:
        """Handle system-level events."""
        logger.info(
            f"[SYSTEM] {event.event_type.action.value.upper()} {event.event_type.category.value.upper()} - {event.data}"
        )


class DatabaseTraceHandler(BaseTraceHandler):
    """Trace handler that saves events to database."""

    def __init__(self, task_id: Optional[int] = None) -> None:
        super().__init__()
        self.task_id = task_id
        # Import here to avoid circular dependencies
        # Actual database import will be handled in web-specific handler

    async def _handle_task_event(self, event: TraceEvent) -> None:
        """Handle task-level events for database storage."""
        # This will be implemented in the web-specific handler
        logger.debug(
            f"[DB] Task event: {event.event_type.value} for task {event.task_id}"
        )

    async def _handle_step_event(self, event: TraceEvent) -> None:
        """Handle step-level events for database storage."""
        # This will be implemented in the web-specific handler
        logger.debug(
            f"[DB] Step event: {event.event_type.value} for step {event.step_id}"
        )

    async def _handle_action_event(self, event: TraceEvent) -> None:
        """Handle action-level events for database storage."""
        # This will be implemented in the web-specific handler
        logger.debug(
            f"[DB] Action event: {event.event_type.value} for step {event.step_id}"
        )

    async def _handle_system_event(self, event: TraceEvent) -> None:
        """Handle system-level events for database storage."""
        # This will be implemented in the web-specific handler
        logger.debug(f"[DB] System event: {event.event_type.value}")


class Tracer:
    """Main tracing class that manages trace events and handlers."""

    def __init__(self) -> None:
        self.handlers: List[TraceHandler] = []
        # No default handlers - let users add their own

    def add_handler(self, handler: TraceHandler) -> None:
        """Add a trace handler."""
        self.handlers.append(handler)

    async def trace_event(
        self,
        event_type: TraceEventType,
        task_id: Optional[str] = None,
        step_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        """Record a trace event and return its ID."""
        logger.info(
            f"trace_event called: {event_type.value} for task {task_id}, step {step_id} with data keys: {list(data.keys()) if data else []}"
        )

        event = TraceEvent(
            event_type=event_type,
            task_id=task_id,
            step_id=step_id,
            data=data or {},
            parent_id=parent_id,
        )

        # Notify all handlers
        logger.info(
            f"Notifying {len(self.handlers)} handlers for event {event_type.value}"
        )
        for i, handler in enumerate(self.handlers):
            try:
                logger.info(f"Calling handler {i}: {type(handler).__name__}")
                await handler.handle_event(event)
                logger.info(f"Handler {i} completed successfully")
            except Exception as e:
                logger.warning(f"Trace handler {i} failed: {e}")

        logger.info(
            f"trace_event completed for {event_type.value}, event_id: {event.id}"
        )
        return event.id


# Simplified convenience functions for common tracing operations
async def trace_task_start(
    tracer: Tracer,
    task_id: str,
    category: TraceCategory,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace task start event."""
    event_type = TraceEventType(TraceScope.TASK, TraceAction.START, category)
    return await tracer.trace_event(event_type, task_id=task_id, data=data or {})


async def trace_task_end(
    tracer: Tracer,
    task_id: str,
    category: TraceCategory,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace task end event."""
    event_type = TraceEventType(TraceScope.TASK, TraceAction.END, category)
    return await tracer.trace_event(event_type, task_id=task_id, data=data or {})


async def trace_task_completion(
    tracer: Tracer,
    task_id: str,
    result: Any,
    success: bool = True,
) -> str:
    """Trace task completion event with result data."""
    event_type = TASK_END_GENERAL

    data = {
        "result": result,
        "success": success,
    }
    return await tracer.trace_event(event_type, task_id=task_id, data=data)


async def trace_step_start(
    tracer: Tracer,
    task_id: str,
    step_id: str,
    category: TraceCategory,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace step start event."""
    event_type = TraceEventType(TraceScope.STEP, TraceAction.START, category)
    return await tracer.trace_event(
        event_type, task_id=task_id, step_id=step_id, data=data or {}
    )


async def trace_step_end(
    tracer: Tracer,
    task_id: str,
    step_id: str,
    category: TraceCategory,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace step end event."""
    event_type = TraceEventType(TraceScope.STEP, TraceAction.END, category)
    return await tracer.trace_event(
        event_type, task_id=task_id, step_id=step_id, data=data or {}
    )


async def trace_action_start(
    tracer: Tracer,
    task_id: str,
    step_id: str,
    category: TraceCategory,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace action start event."""
    event_type = TraceEventType(TraceScope.ACTION, TraceAction.START, category)
    return await tracer.trace_event(
        event_type, task_id=task_id, step_id=step_id, data=data or {}
    )


async def trace_action_end(
    tracer: Tracer,
    task_id: str,
    step_id: str,
    category: TraceCategory,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace action end event."""
    event_type = TraceEventType(TraceScope.ACTION, TraceAction.END, category)
    return await tracer.trace_event(
        event_type, task_id=task_id, step_id=step_id, data=data or {}
    )


async def trace_error(
    tracer: Tracer,
    task_id: str,
    step_id: Optional[str] = None,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    traceback_str: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace error event with clear scope attribution."""
    event_data = {}
    if error_type:
        event_data["error_type"] = error_type
    if error_message:
        event_data["error_message"] = error_message
    if traceback_str:
        event_data["traceback"] = traceback_str
    if data:
        event_data.update(data)

    # Determine scope based on whether step_id is provided
    scope = TraceScope.STEP if step_id else TraceScope.TASK
    event_type = TraceEventType(scope, TraceAction.ERROR, TraceCategory.GENERAL)

    return await tracer.trace_event(
        event_type, task_id=task_id, step_id=step_id, data=event_data
    )


async def trace_info(
    tracer: Tracer,
    task_id: str,
    step_id: Optional[str] = None,
    category: TraceCategory = TraceCategory.GENERAL,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace info event with flexible scope."""
    scope = TraceScope.STEP if step_id else TraceScope.TASK
    event_type = TraceEventType(scope, TraceAction.INFO, category)

    return await tracer.trace_event(
        event_type, task_id=task_id, step_id=step_id, data=data or {}
    )


async def trace_memory_generate_start(
    tracer: Tracer,
    task_id: str,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace memory generation start."""
    return await tracer.trace_event(
        MEMORY_GENERATE_START, task_id=task_id, data=data or {}
    )


async def trace_memory_generate_end(
    tracer: Tracer,
    task_id: str,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace memory generation end."""
    return await tracer.trace_event(
        MEMORY_GENERATE_END, task_id=task_id, data=data or {}
    )


async def trace_memory_store_start(
    tracer: Tracer,
    task_id: str,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace memory storage start."""
    return await tracer.trace_event(
        MEMORY_STORE_START, task_id=task_id, data=data or {}
    )


async def trace_memory_store_end(
    tracer: Tracer,
    task_id: str,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace memory storage end."""
    return await tracer.trace_event(MEMORY_STORE_END, task_id=task_id, data=data or {})


async def trace_memory_retrieve_start(
    tracer: Tracer,
    task_id: str,
    data: Optional[Dict[str, Any]] = None,
    step_id: Optional[str] = None,
) -> str:
    """Trace memory retrieval start event."""
    return await tracer.trace_event(
        MEMORY_RETRIEVE_START, task_id=task_id, step_id=step_id, data=data or {}
    )


async def trace_memory_retrieve_end(
    tracer: Tracer,
    task_id: str,
    data: Optional[Dict[str, Any]] = None,
    step_id: Optional[str] = None,
) -> str:
    """Trace memory retrieval end event."""
    return await tracer.trace_event(
        MEMORY_RETRIEVE_END, task_id=task_id, step_id=step_id, data=data or {}
    )


async def trace_compact_start(
    tracer: Tracer,
    task_id: str,
    step_id: str,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace context compaction start."""
    return await tracer.trace_event(
        COMPACT_START, task_id=task_id, step_id=step_id, data=data or {}
    )


async def trace_compact_end(
    tracer: Tracer,
    task_id: str,
    step_id: str,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace context compaction end."""
    return await tracer.trace_event(
        COMPACT_END, task_id=task_id, step_id=step_id, data=data or {}
    )


async def trace_system_event(
    tracer: Tracer,
    action: TraceAction,
    category: TraceCategory,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace system-level event."""
    event_type = TraceEventType(TraceScope.SYSTEM, action, category)
    return await tracer.trace_event(event_type, data=data or {})


# Common convenience functions for specific use cases
async def trace_dag_plan_start(
    tracer: Tracer, task_id: str, data: Optional[Dict[str, Any]] = None
) -> str:
    """Trace DAG plan start event."""
    return await trace_task_start(tracer, task_id, TraceCategory.DAG_PLAN, data)


async def trace_dag_plan_end(
    tracer: Tracer, task_id: str, data: Optional[Dict[str, Any]] = None
) -> str:
    """Trace DAG plan end event."""
    return await trace_task_end(tracer, task_id, TraceCategory.DAG_PLAN, data)


async def trace_dag_execution(
    tracer: Tracer,
    task_id: str,
    phase: str,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace DAG execution status event.

    Args:
        tracer: The tracer instance
        task_id: The task ID
        phase: The execution phase ("planning", "executing", "completed", "failed")
        data: Additional data to include in the event

    Returns:
        The event ID
    """
    event_data = data or {}
    event_data["phase"] = phase
    return await tracer.trace_event(
        TraceEventType(TraceScope.TASK, TraceAction.UPDATE, TraceCategory.DAG),
        task_id=task_id,
        data=event_data,
    )


async def trace_dag_step_start(
    tracer: Tracer, task_id: str, step_id: str, data: Optional[Dict[str, Any]] = None
) -> str:
    """Trace DAG step start event."""
    return await trace_step_start(tracer, task_id, step_id, TraceCategory.DAG, data)


async def trace_llm_call_start(
    tracer: Tracer, task_id: str, step_id: str, data: Optional[Dict[str, Any]] = None
) -> str:
    """Trace LLM call start event."""
    return await trace_action_start(tracer, task_id, step_id, TraceCategory.LLM, data)


async def trace_task_llm_call_start(
    tracer: Tracer, task_id: str, data: Optional[Dict[str, Any]] = None
) -> str:
    """Trace task-level LLM call start event (not associated with a specific step)."""
    return await trace_task_start(tracer, task_id, TraceCategory.LLM, data)


async def trace_task_llm_call_end(
    tracer: Tracer, task_id: str, data: Optional[Dict[str, Any]] = None
) -> str:
    """Trace task-level LLM call end event (not associated with a specific step)."""
    return await trace_task_end(tracer, task_id, TraceCategory.LLM, data)


async def trace_tool_execution_start(
    tracer: Tracer,
    task_id: str,
    step_id: str,
    tool_name: str,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace tool execution start event."""
    event_data = {"tool_name": tool_name}
    if data:
        event_data.update(data)
    return await trace_action_start(
        tracer, task_id, step_id, TraceCategory.TOOL, event_data
    )


async def trace_visualization_update(
    tracer: Tracer, data: Optional[Dict[str, Any]] = None
) -> str:
    """Trace visualization update event."""
    return await trace_system_event(
        tracer, TraceAction.UPDATE, TraceCategory.VISUALIZATION, data
    )


async def trace_user_message(
    tracer: Tracer, task_id: str, message: str, data: Optional[Dict[str, Any]] = None
) -> str:
    """Trace user message event."""
    event_data = {"message": message}
    if data:
        event_data.update(data)
    return await trace_task_start(tracer, task_id, TraceCategory.MESSAGE, event_data)


async def trace_ai_message(
    tracer: Tracer, task_id: str, message: str, data: Optional[Dict[str, Any]] = None
) -> str:
    """Trace AI message event."""
    event_data = {"content": message}  # Use 'content' to match frontend expectations
    if data:
        event_data.update(data)
    # Use AI_MESSAGE event type to generate "ai_message" event_type for frontend
    return await tracer.trace_event(AI_MESSAGE, task_id=task_id, data=event_data)


async def trace_skill_select_start(
    tracer: Tracer, task_id: str, data: Optional[Dict[str, Any]] = None
) -> str:
    """Trace skill selection start event."""
    return await trace_task_start(tracer, task_id, TraceCategory.SKILL, data)


async def trace_skill_select_end(
    tracer: Tracer, task_id: str, data: Optional[Dict[str, Any]] = None
) -> str:
    """Trace skill selection end event."""
    return await trace_task_end(tracer, task_id, TraceCategory.SKILL, data)
