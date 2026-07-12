"""Pydantic request/response models for the ``/v1/chat/tasks/*`` SDK endpoints.

Kept in one module because the shapes are small, cross-referential
(CreateTask / AppendMessage both nest the same ``MessageBody``), and
will all be regenerated as TypeScript / Python SDK types from the
OpenAPI schema together.

Design notes:

  - ``MessageBody.role`` is currently fixed to ``"user"`` (SDK callers
    only push user input on this surface). We accept it as a field
    rather than hard-coding for forward-compatibility with future
    system / function message roles, but reject anything else at
    validation time.

  - ``metadata`` is a free-form passthrough dict the SaaS caller can
    use to round-trip its own correlation IDs (trace_id, request_id,
    etc). The server does not interpret it but persists enough of the
    SDK call shape to support future debugging.

  - Timestamps are tz-aware ``datetime`` so SDK clients deserialize
    into proper datetimes (``datetime.fromisoformat`` works on both
    PG ``timestamptz`` and the ISO 8601 the FastAPI default
    serializer emits).
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class MessageBody(BaseModel):
    """One chat message in the SDK request body.

    Currently the SDK surface only accepts ``role='user'`` -- the SDK
    is for SaaS clients pushing user input, not for replaying
    transcripts. Future-proofed as a string so we don't have to break
    the wire shape when adding ``system`` / ``function`` later.
    """

    role: Literal["user"] = Field(
        default="user",
        description=(
            "Currently must be 'user'. Reserved as a field for future "
            "expansion (system / function roles) without breaking the "
            "wire shape."
        ),
    )
    content: str = Field(
        ...,
        min_length=1,
        description="The user's message text. Must be non-empty.",
    )
    files: Optional[List[str]] = Field(
        default=None,
        description=(
            "file_id values previously returned by ``POST /v1/chat/files``. "
            "The referenced files are attached to this turn and exposed to "
            "the agent via file references."
        ),
    )


class ConnectorRuntimeRefBody(BaseModel):
    """Stable connector identity for per-invocation runtime values."""

    model_config = ConfigDict(extra="forbid")

    connector_type: Literal["mcp", "custom_api"]
    connector_id: int = Field(..., gt=0)


class ConnectorRuntimeContextBody(BaseModel):
    """Per-connector runtime values supplied by a trusted invocation."""

    model_config = ConfigDict(extra="forbid")

    connector_ref: ConnectorRuntimeRefBody
    context: Optional[Dict[str, Any]] = None
    secrets: Optional[Dict[str, Any]] = None
    auth_selector: Optional[Dict[str, Any]] = None


class CreateTaskRequest(BaseModel):
    """Body for ``POST /v1/chat/tasks``.

    ``agent_id`` is required and must match the agent bound to the
    presented API key; the server enforces ``body.agent_id ==
    authed.agent.id`` and returns 404 ``agent_not_found`` on mismatch.
    """

    agent_id: int = Field(
        ...,
        description=(
            "Target agent's primary key. Must match the agent the "
            "presented API key is bound to."
        ),
    )
    message: MessageBody = Field(..., description="First user message of the task.")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Free-form correlation data the SDK caller can pass through "
            "(trace_id, request_id, etc). Not interpreted server-side."
        ),
    )
    connector_runtime_context: Optional[List[ConnectorRuntimeContextBody]] = Field(
        default=None,
        description=(
            "Trusted server-to-server connector runtime values. Values are "
            "validated and applied below the LLM/tool-argument layer."
        ),
    )


class UploadedFileInfo(BaseModel):
    """One stored file returned by ``POST /v1/chat/files``."""

    file_id: str
    filename: str
    file_size: int
    mime_type: Optional[str] = None


class UploadFilesResponse(BaseModel):
    """``POST /v1/chat/files`` -> stored file handles.

    Pass the returned ``file_id`` values in a subsequent
    ``POST /v1/chat/tasks`` (or ``.../messages``) under ``message.files``
    to attach them to a turn.
    """

    files: List[UploadedFileInfo]


class RuntimeKeyResponse(BaseModel):
    """Runtime API key returned once after creation or rotation."""

    full_key: str
    key_prefix: str
    created_at: datetime


class V1AgentCreateRequest(BaseModel):
    """Body for ``POST /v1/agents``."""

    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    instructions: Optional[str] = None
    execution_mode: Optional[str] = "balanced"
    models: Optional[dict[str, Any]] = None
    knowledge_bases: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    tool_categories: list[str] = Field(default_factory=list)
    suggested_prompts: list[str] = Field(default_factory=list)
    generate_runtime_key: bool = True


class V1AgentTemplateCreateRequest(BaseModel):
    """Body for ``POST /v1/agents/from-template``."""

    template_id: str = Field(..., min_length=1)
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = None
    instructions: Optional[str] = None
    execution_mode: Optional[str] = None
    models: Optional[dict[str, Any]] = None
    knowledge_bases: Optional[list[str]] = None
    skills: Optional[list[str]] = None
    tool_categories: Optional[list[str]] = None
    suggested_prompts: Optional[list[str]] = None
    generate_runtime_key: bool = True


class V1AgentSummary(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    logo_url: Optional[str] = None
    status: str
    created_at: str
    updated_at: str
    widget_enabled: bool
    allowed_domains: list[str]
    share_enabled: bool
    share_updated_at: Optional[str]


class V1AgentResponse(BaseModel):
    id: int
    user_id: int
    name: str
    description: Optional[str] = None
    instructions: Optional[str] = None
    execution_mode: str
    models: Optional[dict[str, Any]] = None
    knowledge_bases: list[str]
    skills: list[str]
    tool_categories: list[str]
    suggested_prompts: list[str]
    logo_url: Optional[str] = None
    status: str
    published_at: Optional[str] = None
    created_at: str
    updated_at: str
    widget_enabled: bool
    allowed_domains: list[str]
    share_enabled: bool
    share_updated_at: Optional[str]


class V1AgentCreateResponse(BaseModel):
    agent: V1AgentResponse
    api_key: Optional[RuntimeKeyResponse] = None


class V1TemplateSummary(BaseModel):
    id: str
    name: str
    category: str = ""
    featured: bool = False
    description: str = ""
    features: list[str] = Field(default_factory=list)
    connections: list[dict[str, Any]] = Field(default_factory=list)
    setup_time: str = "5 min setup"
    tags: list[str] = Field(default_factory=list)
    author: str = ""
    version: str = ""


class V1TemplateDetail(V1TemplateSummary):
    agent_config: dict[str, Any]


class CreateTaskResponse(BaseModel):
    """``POST /v1/chat/tasks`` -> 202 Accepted response.

    The task has been persisted, claimed as RUNNING in the same
    transaction, and queued for background execution; callers poll
    ``GET /v1/chat/tasks/{task_id}`` to observe the transition
    running -> completed/failed.
    """

    task_id: int = Field(..., description="Newly created task primary key.")
    agent_id: int = Field(..., description="Agent the task is bound to.")
    status: str = Field(
        ...,
        description=(
            "Initial status, 'running' in the 202 response (the atomic "
            "claim inside POST commits the status flip before the "
            "response is sent). Use GET /v1/chat/tasks/{task_id} to "
            "observe later transitions."
        ),
    )
    created_at: datetime = Field(..., description="UTC creation timestamp.")
    run_id: str = Field(..., description="Identity of the accepted execution run.")
    state_version: int = Field(
        ..., description="Monotonic version of the task control state."
    )
    control_state: str = Field(
        ..., description="Detailed control state, such as running or pause_requested."
    )


class AppendMessageRequest(BaseModel):
    """Body for ``POST /v1/chat/tasks/{task_id}/messages``.

    Same shape as :class:`CreateTaskRequest` minus the lack of a
    ``metadata`` field by default -- callers append a new user
    message to an existing task. ``agent_id`` is required again
    (consistent with the SDK contract: every write carries the
    agent_id explicitly for forward-compat with multi-agent keys).
    """

    agent_id: int = Field(
        ...,
        description=(
            "Target agent's primary key. Must match the agent the "
            "presented API key is bound to and the task's agent_id."
        ),
    )
    message: MessageBody = Field(..., description="Next user message in the task.")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Free-form correlation data passed through unchanged.",
    )
    connector_runtime_context: Optional[List[ConnectorRuntimeContextBody]] = Field(
        default=None,
        description=(
            "Trusted server-to-server connector runtime values for this turn."
        ),
    )


class AppendMessageResponse(BaseModel):
    """``POST /v1/chat/tasks/{task_id}/messages`` -> 202 Accepted response.

    The new user message has been persisted and the next turn queued
    for background execution; callers poll the same way they would
    after the initial POST /v1/chat/tasks.
    """

    task_id: int = Field(..., description="Existing task primary key.")
    agent_id: int = Field(..., description="Agent the task is bound to.")
    status: str = Field(
        ...,
        description=(
            "Initial status of the new turn, 'running' in the 202 "
            "response (the atomic claim inside POST commits the status "
            "flip before the response is sent)."
        ),
    )
    accepted_at: datetime = Field(
        ...,
        description=(
            "UTC timestamp when the server accepted the message and "
            "scheduled background execution. Not the message's stored "
            "created_at (which may differ slightly due to DB clock)."
        ),
    )
    run_id: str = Field(..., description="Identity of the accepted execution run.")
    state_version: int = Field(
        ..., description="Monotonic version of the task control state."
    )
    control_state: str = Field(..., description="Detailed task control state.")


class TaskInfoResponse(BaseModel):
    """``GET /v1/chat/tasks/{task_id}`` response.

    Returns a snapshot of the task's current state from the ``tasks``
    row. ``input`` / ``output`` / ``error`` reflect the **latest** turn
    only -- full transcript history is queryable via the ``/steps``
    endpoint's ``message`` type steps.
    """

    task_id: int
    agent_id: int
    status: str = Field(
        ...,
        description="One of: pending / running / paused / completed / failed.",
    )
    run_id: Optional[str] = Field(None, description="Current execution run identity.")
    state_version: int = Field(
        0, description="Monotonic version of the task control state."
    )
    control_state: str = Field("idle", description="Detailed task control state.")
    input: Optional[str] = Field(
        None,
        description="Latest-turn user input. Null if no message yet recorded.",
    )
    output: Optional[str] = Field(
        None,
        description=(
            "Latest-turn assistant output. Populated when status reaches "
            "'completed'; null while running or pending."
        ),
    )
    error: Optional[str] = Field(
        None,
        description="Last failure reason when status='failed'.",
    )
    created_at: datetime
    completed_at: Optional[datetime] = Field(
        None,
        description=(
            "UTC timestamp when the task reached a terminal state "
            "(completed or failed). Null while still running."
        ),
    )


# ``PublicStep.type`` is the public surface for what was internally one
# of ~32 trace event types. Restricted to four stable values so SDK
# clients can switch on the type without keeping up with internal
# trace-event churn. See ``web/api/v1/_step_mapping.py`` for the full
# internal->public mapping table.
PublicStepType = Literal["thinking", "tool_call", "agent_delegation", "message"]

# ``running`` means a start event was seen with no matching end (the
# task is still mid-step at the time of the GET). ``completed`` /
# ``failed`` reflect the end event's success flag.
PublicStepStatus = Literal["running", "completed", "failed"]


class PublicStep(BaseModel):
    """One step on the public SDK timeline.

    Type-specific shape of the ``data`` dict:

      - ``thinking``: ``{"phase": "planning" | "step" | "action"}``
      - ``tool_call``: ``{"name": str, "args": Any,
                          "result"?: Any, "error"?: str}``
      - ``agent_delegation``: ``{"sub_agent_name": str,
                                  "input"?: Any, "output"?: Any}``
      - ``message``: ``{"role": "user" | "assistant", "content": str}``

    The exact keys are documented but the values are intentionally
    typed as ``Any`` because tools and agents can return arbitrary
    JSON. SDK clients should treat unknown keys as forward-compat
    extensions and ignore them.
    """

    id: str = Field(
        ...,
        description=(
            "Stable identifier for this step within the task. Includes "
            "a type prefix (e.g. 'tool_call:abc123') so SDK clients "
            "can dedupe across re-polls."
        ),
    )
    type: PublicStepType = Field(
        ...,
        description=(
            "One of: thinking, tool_call, agent_delegation, message. "
            "Other internal event types are not surfaced on this "
            "endpoint."
        ),
    )
    status: PublicStepStatus = Field(
        ...,
        description=(
            "running while the corresponding end event has not yet "
            "fired (i.e. the SDK polled mid-step), completed on a "
            "normal end event, failed when the end event carries "
            "success=false."
        ),
    )
    started_at: datetime = Field(
        ...,
        description="UTC timestamp of the start event for this step.",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description=("UTC timestamp of the end event. Null while status is 'running'."),
    )
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Type-specific payload. See class docstring for the keys "
            "expected per step type."
        ),
    )


class StepsResponse(BaseModel):
    """``GET /v1/chat/tasks/{task_id}/steps`` response body.

    Steps are returned in monotonic ``started_at`` order. The endpoint
    is a polling primitive: each call returns the full known history
    so far (including any still-running steps as ``status='running'``)
    so SDK clients can resume after a network blip without state.
    """

    task_id: int = Field(..., description="The task these steps belong to.")
    agent_id: int = Field(..., description="The task's agent.")
    steps: List[PublicStep] = Field(
        default_factory=list,
        description="Public-timeline steps in started_at ascending order.",
    )
