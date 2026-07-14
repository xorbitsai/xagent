from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from http import HTTPStatus
from typing import Any, Mapping
from uuid import uuid4

from fastapi import Request
from fastapi.responses import JSONResponse

from ...config import get_public_api_base_url
from ..models.agent import Agent, AgentStatus
from ..models.task import Task, TaskStatus

A2A_MEDIA_TYPE = "application/a2a+json"
A2A_VERSION = "1.0"
A2A_MAX_MESSAGE_TEXT_LENGTH = 200_000
ALL_TASK_STATES = frozenset(
    {
        "TASK_STATE_UNSPECIFIED",
        "TASK_STATE_SUBMITTED",
        "TASK_STATE_WORKING",
        "TASK_STATE_COMPLETED",
        "TASK_STATE_FAILED",
        "TASK_STATE_CANCELED",
        "TASK_STATE_INPUT_REQUIRED",
        "TASK_STATE_REJECTED",
        "TASK_STATE_AUTH_REQUIRED",
    }
)


class A2AApiError(Exception):
    """A2A REST error represented with the google.rpc.Status JSON shape."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        reason = _error_reason(code)
        metadata = {
            str(key): str(value)
            for key, value in (details or {}).items()
            if value is not None
        }
        self.payload: dict[str, Any] = {
            "code": status_code,
            "status": _rpc_status(status_code, reason),
            "message": message,
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                    "reason": reason,
                    "domain": "a2a-protocol.org",
                    "metadata": metadata,
                }
            ],
        }


def a2a_json_response(content: dict[str, Any], status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        content=content,
        status_code=status_code,
        media_type=A2A_MEDIA_TYPE,
        headers={"A2A-Version": A2A_VERSION},
    )


async def a2a_api_error_handler(request: Request, exc: A2AApiError) -> JSONResponse:
    response = a2a_json_response({"error": exc.payload}, status_code=exc.status_code)
    if exc.status_code == 401:
        response.headers["WWW-Authenticate"] = "Bearer"
    return response


def a2a_error(
    code: str,
    message: str,
    *,
    status_code: int,
    details: dict[str, Any] | None = None,
) -> A2AApiError:
    return A2AApiError(code, message, status_code=status_code, details=details)


def is_published_agent(agent: Agent | None) -> bool:
    if agent is None:
        return False
    status = getattr(agent, "status", None)
    status = getattr(status, "value", status)
    return status == AgentStatus.PUBLISHED.value


def normalize_agent_card_base_url(request: Request, agent_id: int) -> str:
    root = get_public_api_base_url() or str(request.base_url).rstrip("/")
    return f"{root}/api/a2a/agents/{agent_id}"


def build_agent_card(agent: Agent, request: Request) -> dict[str, Any]:
    if not is_published_agent(agent):
        raise a2a_error(
            "agent_not_found",
            "Agent not found.",
            status_code=404,
        )

    base_url = normalize_agent_card_base_url(request, int(agent.id))
    # Agent Cards are public discovery documents. Never fall back to the
    # private execution instructions when an owner omitted a description.
    description = str(agent.description or agent.name)
    raw_suggested: Any = agent.suggested_prompts
    suggested_source = raw_suggested if isinstance(raw_suggested, list) else []
    suggested = [
        item for item in suggested_source if isinstance(item, str) and item.strip()
    ]
    skill_id = _slugify(str(agent.name or f"agent-{agent.id}")) or "default"
    card: dict[str, Any] = {
        "name": str(agent.name),
        "description": description,
        "version": "1.0.0",
        "supportedInterfaces": [
            {
                "url": base_url,
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": A2A_VERSION,
            }
        ],
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
            "extendedAgentCard": False,
        },
        "securitySchemes": {
            "xagentAgentApiKey": {
                "httpAuthSecurityScheme": {
                    "scheme": "Bearer",
                    "description": "Xagent agent API key",
                }
            }
        },
        "securityRequirements": [{"schemes": {"xagentAgentApiKey": {}}}],
        "defaultInputModes": ["text/plain", "application/json"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {
                "id": skill_id,
                "name": str(agent.name),
                "description": description,
                "tags": ["xagent", "agent"],
                "examples": suggested[:5],
                "inputModes": ["text/plain", "application/json"],
                "outputModes": ["text/plain"],
            }
        ],
    }
    if agent.logo_url:
        card["iconUrl"] = str(agent.logo_url)
    return card


def extract_message_text(message: Mapping[str, Any]) -> str:
    message_id = message.get("messageId")
    if not isinstance(message_id, str) or not message_id.strip():
        raise a2a_error(
            "invalid_argument",
            "A2A message must include a non-empty messageId.",
            status_code=400,
            details={"field": "message.messageId"},
        )
    if message.get("role") != "ROLE_USER":
        raise a2a_error(
            "invalid_argument",
            "A2A request messages must use role ROLE_USER.",
            status_code=400,
            details={"field": "message.role"},
        )

    parts = message.get("parts")
    if not isinstance(parts, list) or not parts:
        raise a2a_error(
            "invalid_argument",
            "A2A message must include at least one part.",
            status_code=400,
            details={"field": "message.parts"},
        )

    texts: list[str] = []
    total_length = 0

    def append_text(value: str, field: str) -> None:
        nonlocal total_length
        projected_length = total_length + (2 if texts else 0) + len(value)
        if projected_length > A2A_MAX_MESSAGE_TEXT_LENGTH:
            raise a2a_error(
                "resource_exhausted",
                "A2A message content exceeds the supported length limit.",
                status_code=413,
                details={
                    "field": field,
                    "maxLength": A2A_MAX_MESSAGE_TEXT_LENGTH,
                },
            )
        texts.append(value)
        total_length = projected_length

    for index, part in enumerate(parts):
        if not isinstance(part, Mapping):
            raise a2a_error(
                "invalid_argument",
                "Each A2A message part must be an object.",
                status_code=400,
                details={"field": f"message.parts[{index}]"},
            )
        content_fields = [
            field for field in ("text", "raw", "url", "data") if field in part
        ]
        if len(content_fields) != 1:
            raise a2a_error(
                "invalid_argument",
                "Each A2A message part must contain exactly one content field.",
                status_code=400,
                details={"field": f"message.parts[{index}]"},
            )
        field = content_fields[0]
        media_type = part.get("mediaType")
        if field == "text":
            text = part.get("text")
            if not isinstance(text, str) or not text.strip():
                raise a2a_error(
                    "invalid_argument",
                    "A2A text parts must contain non-empty text.",
                    status_code=400,
                    details={"field": f"message.parts[{index}].text"},
                )
            if media_type and not str(media_type).lower().startswith("text/"):
                raise a2a_error(
                    "content_type_not_supported",
                    f"Unsupported text media type: {media_type}",
                    status_code=400,
                    details={"mediaType": media_type},
                )
            append_text(text.strip(), f"message.parts[{index}].text")
            continue
        if field == "data":
            if media_type and str(media_type).lower() != "application/json":
                raise a2a_error(
                    "content_type_not_supported",
                    f"Unsupported data media type: {media_type}",
                    status_code=400,
                    details={"mediaType": media_type},
                )
            append_text(
                json.dumps(part.get("data"), ensure_ascii=False),
                f"message.parts[{index}].data",
            )
            continue
        raise a2a_error(
            "content_type_not_supported",
            "Xagent A2A agents currently accept text and JSON data parts only.",
            status_code=400,
            details={"field": f"message.parts[{index}].{field}"},
        )
    if not texts:
        raise a2a_error(
            "invalid_argument",
            "A2A message must include at least one text or data part.",
            status_code=400,
        )
    return "\n\n".join(texts)


def message_context_id(
    message: Mapping[str, Any], body: Mapping[str, Any]
) -> str | None:
    context_id = message.get("contextId") or body.get("contextId")
    if isinstance(context_id, str) and context_id.strip():
        return context_id.strip()
    return None


def new_context_id() -> str:
    return str(uuid4())


def message_task_id(message: Mapping[str, Any], body: Mapping[str, Any]) -> int | None:
    raw = message.get("taskId") or body.get("taskId")
    if raw is None:
        return None
    if isinstance(raw, int) and not isinstance(raw, bool):
        if raw > 0:
            return raw
    if isinstance(raw, str) and raw.isdecimal():
        value = int(raw)
        if value > 0:
            return value
    raise a2a_error(
        "invalid_argument",
        "taskId must reference a valid Xagent A2A task ID.",
        status_code=400,
        details={"field": "message.taskId"},
    )


def task_context_id(task: Task) -> str:
    agent_config: dict[str, Any] = (
        task.agent_config if isinstance(task.agent_config, dict) else {}
    )
    context_id = agent_config.get("a2a_context_id")
    if isinstance(context_id, str) and context_id:
        return context_id
    return str(task.id)


def task_to_a2a(task: Task, *, include_artifacts: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": str(task.id),
        "contextId": task_context_id(task),
        "status": {
            "state": task_state(task),
            "timestamp": _iso_timestamp(task.updated_at),
        },
    }
    if task.error_message:
        result["status"]["message"] = _agent_message(str(task.error_message), task)
    if include_artifacts and task.output and task_state(task) != "TASK_STATE_CANCELED":
        output = str(task.output)
        result["artifacts"] = [
            {
                "artifactId": f"task-{task.id}-output",
                "name": "final_output",
                "parts": [{"text": output}],
            }
        ]
    return result


def task_state(task_or_status: Task | Any) -> str:
    task = task_or_status if isinstance(task_or_status, Task) else None
    status = task.status if task is not None else task_or_status
    if task is not None:
        agent_config: dict[str, Any] = (
            task.agent_config if isinstance(task.agent_config, dict) else {}
        )
        override = agent_config.get("a2a_state")
        if override in {
            "TASK_STATE_CANCELED",
            "TASK_STATE_REJECTED",
            "TASK_STATE_AUTH_REQUIRED",
        }:
            return str(override)
    if isinstance(status, str):
        try:
            status = TaskStatus(status)
        except ValueError:
            return "TASK_STATE_UNSPECIFIED"
    if status == TaskStatus.PENDING:
        return "TASK_STATE_SUBMITTED"
    if status == TaskStatus.RUNNING:
        return "TASK_STATE_WORKING"
    if status in {TaskStatus.PAUSED, TaskStatus.WAITING_FOR_USER}:
        return "TASK_STATE_INPUT_REQUIRED"
    if status == TaskStatus.COMPLETED:
        return "TASK_STATE_COMPLETED"
    if status == TaskStatus.FAILED:
        return "TASK_STATE_FAILED"
    return "TASK_STATE_UNSPECIFIED"


def sse_task_snapshot(task: Task) -> str:
    return _sse_event({"task": task_to_a2a(task)})


def sse_task_update(task: Task) -> str:
    return _sse_event(
        {
            "statusUpdate": {
                "status": task_to_a2a(task, include_artifacts=False)["status"],
                "taskId": str(task.id),
                "contextId": task_context_id(task),
            }
        }
    )


def sse_task_artifacts(
    task: Task,
    *,
    text: str | None = None,
    append: bool = False,
    last_chunk: bool = True,
) -> str | None:
    output = str(task.output or "") if text is None else text
    if not output or task_state(task) == "TASK_STATE_CANCELED":
        return None
    return _sse_event(
        {
            "artifactUpdate": {
                "taskId": str(task.id),
                "contextId": task_context_id(task),
                "artifact": {
                    "artifactId": f"task-{task.id}-output",
                    "name": "final_output",
                    "parts": [{"text": output}],
                },
                "append": append,
                "lastChunk": last_chunk,
            }
        }
    )


def _agent_message(content: str, task: Task) -> dict[str, Any]:
    return {
        "messageId": f"task-{task.id}-status",
        "contextId": task_context_id(task),
        "taskId": str(task.id),
        "role": "ROLE_AGENT",
        "parts": [{"text": content}],
    }


def _iso_timestamp(value: Any) -> str:
    if not isinstance(value, datetime):
        timestamp = datetime.now(timezone.utc)
    elif value.tzinfo is None:
        timestamp = value.replace(tzinfo=timezone.utc)
    else:
        timestamp = value.astimezone(timezone.utc)
    return timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sse_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _error_reason(code: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", code).strip("_").upper()
    return normalized.removesuffix("_ERROR") or "INTERNAL"


def _rpc_status(status_code: int, reason: str) -> str:
    if reason in {
        "TASK_NOT_CANCELABLE",
        "PUSH_NOTIFICATION_NOT_SUPPORTED",
        "UNSUPPORTED_OPERATION",
        "EXTENDED_AGENT_CARD_NOT_CONFIGURED",
        "EXTENSION_SUPPORT_REQUIRED",
        "VERSION_NOT_SUPPORTED",
    }:
        return "FAILED_PRECONDITION"
    mapping = {
        400: "INVALID_ARGUMENT",
        401: "UNAUTHENTICATED",
        403: "PERMISSION_DENIED",
        404: "NOT_FOUND",
        409: "ABORTED",
        413: "RESOURCE_EXHAUSTED",
        429: "RESOURCE_EXHAUSTED",
        500: "INTERNAL",
        503: "UNAVAILABLE",
    }
    if status_code in mapping:
        return mapping[status_code]
    try:
        return HTTPStatus(status_code).phrase.replace(" ", "_").upper()
    except ValueError:
        return "UNKNOWN"


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_")
    return normalized.lower()
