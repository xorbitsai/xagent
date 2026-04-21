"""Chat API request and response models"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, model_validator


class ChatMessage(BaseModel):
    """Chat message model"""

    role: str  # "user", "ai", "system"
    content: str
    timestamp: datetime


class ChatSendRequest(BaseModel):
    """Send message request"""

    message: str
    task_id: Optional[int] = None
    context: Optional[Dict[str, Any]] = None


class ChatSendResponse(BaseModel):
    """Send message response"""

    task_id: int
    message_id: int
    status: str
    ai_response: Optional[str] = None


class ChatHistoryResponse(BaseModel):
    """Chat history response"""

    task_id: int
    messages: List[ChatMessage]


class ExampleItem(BaseModel):
    """Input/output example for process mode"""

    input: str
    output: str


class TaskCreateRequest(BaseModel):
    """Create task request"""

    title: str
    description: Optional[str] = None
    agent_id: Optional[int] = None  # Agent Builder agent ID
    files: Optional[List[str]] = None  # List of filenames to associate with the task
    llm_ids: Optional[List[Optional[str]]] = (
        None  # Model identifiers to use: exactly 4 elements in order [default, fast_small, vision, compact]
    )
    memory_similarity_threshold: Optional[float] = (
        1.5  # Memory search similarity threshold
    )
    agent_type: Optional[str] = "standard"  # Agent type: "standard", "text2sql", etc.
    agent_config: Optional[Dict[str, Any]] = None  # Agent-specific configuration

    # Execution mode field
    execution_mode: Optional[str] = "balanced"  # "flash", "balanced", or "think"
    process_description: Optional[str] = (
        None  # Process mode: detailed process description (deprecated)
    )
    examples: Optional[List[ExampleItem]] = (
        None  # Process mode: input/output examples (deprecated)
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_llm_names_to_llm_ids(cls, data: Any) -> Any:
        """Backward compatibility: accept deprecated `llm_names` and map to `llm_ids`."""
        if not isinstance(data, dict):
            return data
        if data.get("llm_ids") is None and data.get("llm_names") is not None:
            data = dict(data)
            data["llm_ids"] = data.get("llm_names")
        return data


class TaskCreateResponse(BaseModel):
    """Create task response"""

    task_id: int
    title: str
    status: str
    created_at: str
    model_id: Optional[str] = None
    small_fast_model_id: Optional[str] = None
    visual_model_id: Optional[str] = None
    compact_model_id: Optional[str] = None
    model_name: Optional[str] = None
    small_fast_model_name: Optional[str] = None
    visual_model_name: Optional[str] = None
    compact_model_name: Optional[str] = None
    execution_mode: Optional[str] = None
    channel_id: Optional[int] = None
    channel_name: Optional[str] = None


class ExecutionStatus(BaseModel):
    """Execution status model"""

    task_id: int
    status: str  # "pending", "running", "completed", "failed"
    current_step: Optional[str] = None
    progress: Optional[float] = None
    steps: List[Dict[str, Any]] = []
    logs: List[Dict[str, Any]] = []
    # Step detailed information with associated logs
    step_details: Optional[Dict[str, Dict[str, Any]]] = None
    # Task information
    task_title: Optional[str] = None
    task_description: Optional[str] = None
    # Final result from AI response
    result: Optional[str] = None


class InterventionRequest(BaseModel):
    """Human intervention request"""

    task_id: int
    step_id: str
    action: str  # "pause", "resume", "modify", "skip"
    data: Optional[Dict[str, Any]] = None


class InterventionResponse(BaseModel):
    """Human intervention response"""

    success: bool
    message: str
    updated_status: Optional[ExecutionStatus] = None
