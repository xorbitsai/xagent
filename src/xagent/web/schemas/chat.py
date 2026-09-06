"""Chat API request and response models"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from ...core.task_runtime import MAX_TASK_RUNTIME_EXTENSIONS
from .connector_runtime import ConnectorRuntimeRequirementsModel

# Only ever read here (TaskCreateRequest.seed_interactions is passed straight
# through to create_task_with_message as-is - see api/chat.py), unlike
# MAX_TASK_RUNTIME_EXTENSIONS above, which is re-checked at bind time too and
# so lives in core.task_runtime instead.
MAX_SEED_INTERACTIONS = 5


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
    agent_type: Optional[str] = "standard"
    agent_config: Optional[Dict[str, Any]] = None  # Agent-specific configuration
    # Transport-shape violations intentionally remain Pydantic 422 responses;
    # registry/semantic validation happens in the service layer as HTTP 400.
    runtime_extensions: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        max_length=MAX_TASK_RUNTIME_EXTENSIONS,
        description=(
            "Runtime extensions to bind this task to, keyed by registered "
            "extension name, each mapping to that provider's configuration "
            "object. Names must be registered in this deployment; an unknown "
            f"name is rejected. At most {MAX_TASK_RUNTIME_EXTENSIONS} entries; "
            "each configuration must be JSON-serializable and is bounded in "
            "size, as is the combined payload. Bindings are recorded on the "
            "task and released again when the task is deleted."
        ),
    )
    is_preview: bool = False  # Backward-compatible alias for is_visible=False.
    is_visible: bool = True
    # Only consumed by entry points that start the first turn inside task
    # creation. Unresolvable names degrade to UTC at the render site rather
    # than 422 here, matching how the websocket path treats the same field.
    # max_length bounds this guest-reachable free-text field to IANA lengths.
    timezone: Optional[str] = Field(
        default=None, max_length=64
    )  # IANA name for the caller's local clock

    # Execution mode field
    execution_mode: Optional[str] = None  # "flash", "balanced", "think", or "auto"
    process_description: Optional[str] = (
        None  # Process mode: detailed process description (deprecated)
    )
    examples: Optional[List[ExampleItem]] = (
        None  # Process mode: input/output examples (deprecated)
    )
    seed_assistant_message: Optional[str] = Field(
        default=None,
        max_length=8000,
        description=(
            "Plain-text assistant message to seed as the task's first chat "
            "history entry, in the same transaction as task creation - lets "
            "an agent 'speak first' (e.g. a marketplace persona's opening "
            "intro) without running the LLM. Never triggers execution or "
            "sets task status to waiting_for_user; it is purely a transcript "
            "row a client reading the task's history will see immediately."
        ),
    )
    seed_interactions: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        max_length=MAX_SEED_INTERACTIONS,
        description=(
            "Structured interaction descriptors (the same shape the "
            "ask_user_question agent tool produces) attached to the "
            "seed_assistant_message's chat history row. Not validated "
            "against a fixed schema here - same permissive contract as "
            "every other interactions list in this codebase - the frontend's "
            "normalizeInteractions drops any entry with an unrecognized type "
            "before it ever renders; the 'unsupported' notice is reachable "
            "only for a recognized type ClarificationForm has no render case "
            "for. Ignored if seed_assistant_message is not also set, since "
            "there is no row to attach interactions to."
        ),
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
    agent_id: Optional[int] = None
    agent_name: Optional[str] = None
    agent_logo_url: Optional[str] = None
    run_id: Optional[str] = None
    state_version: int = 0
    control_state: str = "idle"
    runtime_extensions: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Public metadata published by each bound runtime extension, keyed "
            "by extension name. Decoration only: the bindings are already "
            "persisted when this response is produced, so an empty mapping "
            "does not mean nothing was bound. Re-read the live values from "
            "GET /task/{task_id}/runtime-extensions."
        ),
    )
    runtime_extensions_status: Literal["complete", "truncated", "failed"] = Field(
        default="complete",
        description=(
            "Delivery status of `runtime_extensions`: `complete` when every "
            "provider's metadata is included, `truncated` when some was "
            "dropped to keep the response under its aggregate size cap, and "
            "`failed` when metadata could not be collected at all. The task "
            "was created successfully in every case."
        ),
    )
    runtime_extensions_omitted: List[str] = Field(
        default_factory=list,
        description=(
            "Extension names whose metadata was dropped for the aggregate "
            "size cap, i.e. the names missing from `runtime_extensions` when "
            "the status is `truncated`. Empty otherwise."
        ),
    )
    connector_runtime_requirements: ConnectorRuntimeRequirementsModel | None = Field(
        ...,
        description=(
            "Which runtime inputs this task's connectors still need, and "
            "which of them already have a value. Always present in the "
            "response body -- a client must not treat its absence as "
            "meaning anything. Never includes a stored value itself, only "
            "whether one exists. Present with a report on the web chat "
            "create path (`POST /api/chat/task/create`); `null` on the "
            "public chat and share-link create paths, meaning the "
            "requirements were not evaluated there -- those visitors never "
            "receive connector key names."
        ),
    )


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
