from __future__ import annotations

import asyncio
import json
import logging
import re
import socket
import time
from asyncio import sleep, timeout
from typing import Any, Mapping, Optional, Type
from urllib.parse import quote, urlsplit
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field

from ....utils.security import reject_private_network_host
from .base import AbstractBaseTool, ToolCategory, ToolVisibility
from .config import BaseToolConfig
from .factory import register_tool

A2A_MEDIA_TYPE = "application/a2a+json"
A2A_VERSION = "1.0"
A2A_TOOL_ERROR_MESSAGE = "Remote A2A agent call failed."
_TERMINAL_STATES = {
    "TASK_STATE_COMPLETED",
    "TASK_STATE_FAILED",
    "TASK_STATE_CANCELED",
    "TASK_STATE_CANCELLED",
    "TASK_STATE_REJECTED",
}
_INTERRUPTED_STATES = {
    "TASK_STATE_INPUT_REQUIRED",
    "TASK_STATE_AUTH_REQUIRED",
}
_STOP_STATES = _TERMINAL_STATES | _INTERRUPTED_STATES
_FAILURE_STATES = {
    "TASK_STATE_FAILED",
    "TASK_STATE_CANCELED",
    "TASK_STATE_CANCELLED",
    "TASK_STATE_REJECTED",
}

logger = logging.getLogger(__name__)


class A2AAgentToolArgs(BaseModel):
    """Arguments for invoking a remote A2A agent."""

    task: str = Field(description="Task or message to send to the remote A2A agent")
    context_id: Optional[str] = Field(
        default=None,
        description="Optional remote A2A context ID for continuing a conversation",
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Optional remote A2A task ID for continuing an interrupted task",
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional metadata to pass through in the A2A message",
    )


class A2AAgentToolResult(BaseModel):
    """Result from a remote A2A agent invocation."""

    success: bool = Field(description="Whether the remote A2A call completed")
    task_id: Optional[str] = Field(default=None, description="Remote A2A task ID")
    context_id: Optional[str] = Field(default=None, description="Remote A2A context ID")
    state: Optional[str] = Field(default=None, description="Remote A2A task state")
    response: str = Field(description="Text extracted from remote response")
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    raw: Optional[dict[str, Any]] = Field(
        default=None, description="Raw A2A response payload"
    )
    error: Optional[str] = Field(default=None, description="Remote call error")


class A2AAgentTool(AbstractBaseTool):
    """Tool wrapper for a remote A2A HTTP+JSON agent."""

    category: ToolCategory = ToolCategory.AGENT
    concurrency_safe = True

    def __init__(
        self,
        *,
        name: str,
        description: str | None = None,
        endpoint_url: str | None = None,
        agent_card_url: str | None = None,
        headers: Mapping[str, Any] | None = None,
        auth_token: str | None = None,
        timeout_seconds: float = 60.0,
        allow_private_networks: bool = False,
    ):
        self._name = _tool_name(name)
        self._description = description or (
            "Call a remote A2A agent and return its response."
        )
        self._allow_private_networks = allow_private_networks is True
        self._endpoint_url = _clean_url(
            endpoint_url,
            allow_private_networks=self._allow_private_networks,
        )
        self._agent_card_url = _clean_url(
            agent_card_url,
            allow_private_networks=self._allow_private_networks,
        )
        self._headers = _string_headers(headers)
        self._auth_token = auth_token.strip() if isinstance(auth_token, str) else None
        self._timeout_seconds = max(float(timeout_seconds or 60.0), 1.0)
        self._visibility = ToolVisibility.PUBLIC

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def tags(self) -> list[str]:
        return ["a2a", "agent", "remote"]

    def args_type(self) -> Type[BaseModel]:
        return A2AAgentToolArgs

    def return_type(self) -> Type[BaseModel]:
        return A2AAgentToolResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError("A2AAgentTool only supports async execution.")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        parsed = A2AAgentToolArgs(**dict(args))
        task_text = parsed.task.strip()
        if not task_text:
            return A2AAgentToolResult(
                success=False,
                response="",
                error="Task must not be empty.",
            ).model_dump()

        try:
            async with timeout(self._timeout_seconds):
                async with httpx.AsyncClient(
                    timeout=self._timeout_seconds,
                    transport=_PinnedA2ATransport(
                        allow_private_networks=self._allow_private_networks
                    ),
                    follow_redirects=False,
                ) as client:
                    deadline = time.monotonic() + self._timeout_seconds
                    endpoint_url = await self._resolve_endpoint_url(client)
                    response = await client.post(
                        _message_send_url(endpoint_url),
                        json=_message_send_payload(parsed),
                        headers=self._request_headers(),
                    )
                    response.raise_for_status()
                    payload = response.json()
                    task = _extract_task_payload(payload)
                    if task is not None and _should_poll_task(task):
                        task, payload = await self._poll_task_until_terminal(
                            client=client,
                            endpoint_url=endpoint_url,
                            task=task,
                            raw_payload=payload,
                            deadline=deadline,
                        )
        except TimeoutError:
            return A2AAgentToolResult(
                success=False,
                response="",
                error=f"A2A call timed out after {self._timeout_seconds:g}s.",
            ).model_dump()
        except Exception:  # noqa: BLE001
            logger.exception("A2A agent tool %s call failed", self._name)
            return A2AAgentToolResult(
                success=False,
                response="",
                error=A2A_TOOL_ERROR_MESSAGE,
            ).model_dump()

        task = _extract_task_payload(payload)
        if task is None:
            message = _extract_message_payload(payload)
            return A2AAgentToolResult(
                success=True,
                response=(
                    "\n\n".join(_text_parts(message.get("parts"))).strip()
                    if message is not None
                    else _json_text(payload)
                ),
                raw=payload if isinstance(payload, dict) else {"response": payload},
            ).model_dump()

        state = _task_state(task)
        response_text = _task_text(task)
        normalized_state = _normalized_state(state)
        failed = normalized_state in _FAILURE_STATES
        interrupted = normalized_state in _INTERRUPTED_STATES
        completed = normalized_state == "TASK_STATE_COMPLETED"
        error = None
        if failed or interrupted:
            error = response_text or f"Remote A2A task stopped in {state}."
        elif not completed:
            error = f"Remote A2A task did not reach a completed state ({state})."
        result = A2AAgentToolResult(
            success=completed,
            task_id=_optional_str(task.get("id")),
            context_id=_optional_str(task.get("contextId")),
            state=state,
            response=response_text,
            artifacts=_artifacts(task),
            raw=payload if isinstance(payload, dict) else {"response": payload},
            error=error,
        )
        return result.model_dump()

    async def _poll_task_until_terminal(
        self,
        *,
        client: httpx.AsyncClient,
        endpoint_url: str,
        task: dict[str, Any],
        raw_payload: Any,
        deadline: float,
    ) -> tuple[dict[str, Any], Any]:
        task_id = _optional_str(task.get("id"))
        if not task_id:
            return task, raw_payload

        while time.monotonic() < deadline:
            await sleep(min(1.0, max(deadline - time.monotonic(), 0.0)))
            response = await client.get(
                _task_url(endpoint_url, task_id),
                headers=self._request_headers(),
            )
            response.raise_for_status()
            payload = response.json()
            fresh = _extract_task_payload(payload)
            if fresh is None:
                return task, raw_payload
            task = fresh
            raw_payload = payload
            if not _should_poll_task(task):
                return task, raw_payload
        raise TimeoutError(
            f"A2A task {task_id} did not finish within {self._timeout_seconds:g}s."
        )

    async def _resolve_endpoint_url(self, client: httpx.AsyncClient) -> str:
        if self._endpoint_url:
            return self._endpoint_url
        if not self._agent_card_url:
            raise ValueError("A2A tool requires endpoint_url or agent_card_url.")

        response = await client.get(
            self._agent_card_url, headers=self._request_headers()
        )
        response.raise_for_status()
        card = response.json()
        if not isinstance(card, Mapping):
            raise ValueError("A2A agent card response must be a JSON object.")

        endpoint_url = _endpoint_from_card(card)
        if not endpoint_url:
            raise ValueError("A2A agent card does not expose an HTTP+JSON endpoint.")
        endpoint_url = _clean_url(
            endpoint_url,
            allow_private_networks=self._allow_private_networks,
        )
        if endpoint_url is None:
            raise ValueError("A2A agent card does not expose a valid endpoint URL.")
        if not _same_origin(self._agent_card_url, endpoint_url):
            raise ValueError(
                "A2A agent card endpoint must use the same origin as the card URL; "
                "configure endpoint_url explicitly for a cross-origin endpoint."
            )
        self._endpoint_url = endpoint_url
        if self._description == "Call a remote A2A agent and return its response.":
            description = card.get("description")
            if isinstance(description, str) and description.strip():
                self._description = description.strip()
        return endpoint_url

    def _request_headers(self) -> dict[str, str]:
        headers = dict(self._headers)
        headers["Accept"] = A2A_MEDIA_TYPE
        headers["Content-Type"] = A2A_MEDIA_TYPE
        headers["A2A-Version"] = A2A_VERSION
        if self._auth_token and "authorization" not in {key.lower() for key in headers}:
            headers["Authorization"] = f"Bearer {self._auth_token}"
        return headers


def create_a2a_agent_tools_from_configs(
    configs: list[dict[str, Any]],
) -> list[A2AAgentTool]:
    tools: list[A2AAgentTool] = []
    for item in configs:
        if not isinstance(item, dict):
            continue
        raw_name = item.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            continue
        timeout_seconds = item.get("timeout_seconds", 60.0)
        try:
            request_timeout_seconds = float(str(timeout_seconds))
        except (TypeError, ValueError):
            request_timeout_seconds = 60.0
        tool = A2AAgentTool(
            name=raw_name,
            description=_optional_str(item.get("description")),
            endpoint_url=_optional_str(item.get("endpoint_url")),
            agent_card_url=_optional_str(item.get("agent_card_url")),
            headers=item.get("headers")
            if isinstance(item.get("headers"), dict)
            else {},
            auth_token=_optional_str(item.get("auth_token")),
            timeout_seconds=request_timeout_seconds,
            allow_private_networks=item.get("allow_private_networks") is True,
        )
        tools.append(tool)
    return tools


@register_tool(categories={"agent"})
async def create_a2a_agent_tools(config: BaseToolConfig) -> list[AbstractBaseTool]:
    configs = config.get_a2a_agent_configs()
    if not configs:
        return []
    tools: list[AbstractBaseTool] = []
    tools.extend(create_a2a_agent_tools_from_configs(configs))
    return tools


def _message_send_payload(args: A2AAgentToolArgs) -> dict[str, Any]:
    message: dict[str, Any] = {
        "messageId": str(uuid4()),
        "role": "ROLE_USER",
        "parts": [{"text": args.task}],
    }
    if args.context_id:
        message["contextId"] = args.context_id
    if args.task_id:
        message["taskId"] = args.task_id
    if args.metadata:
        message["metadata"] = args.metadata
    return {
        "message": message,
        "configuration": {
            "acceptedOutputModes": ["text/plain"],
            "returnImmediately": True,
        },
    }


def _message_send_url(endpoint_url: str) -> str:
    endpoint_url = endpoint_url.rstrip("/")
    if endpoint_url.endswith("/message:send"):
        return endpoint_url
    return f"{endpoint_url}/message:send"


def _task_url(endpoint_url: str, task_id: str) -> str:
    endpoint_url = endpoint_url.rstrip("/")
    if endpoint_url.endswith("/message:send"):
        endpoint_url = endpoint_url.removesuffix("/message:send")
    return f"{endpoint_url}/tasks/{quote(task_id, safe='')}"


def _endpoint_from_card(card: Mapping[str, Any]) -> str | None:
    interfaces = card.get("supportedInterfaces")
    if isinstance(interfaces, list):
        for item in interfaces:
            if not isinstance(item, Mapping):
                continue
            url = _optional_str(item.get("url"))
            if not url:
                continue
            binding = _optional_str(item.get("protocolBinding"))
            protocol_version = _optional_str(item.get("protocolVersion"))
            if (
                binding is not None
                and binding.upper().replace("_", "+") == "HTTP+JSON"
                and protocol_version is not None
                and protocol_version.split(".", maxsplit=1)[0] == "1"
            ):
                return url
    return None


def _extract_task_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    if isinstance(payload.get("status"), Mapping) and payload.get("id") is not None:
        return dict(payload)
    direct_task = payload.get("task")
    if isinstance(direct_task, dict):
        return direct_task
    result = payload.get("result")
    if isinstance(result, dict):
        nested_task = result.get("task")
        if isinstance(nested_task, dict):
            return nested_task
        if "status" in result:
            return result
    return None


def _extract_message_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    if payload.get("role") in {"ROLE_AGENT", "ROLE_USER"} and isinstance(
        payload.get("parts"), list
    ):
        return dict(payload)
    message = payload.get("message")
    if isinstance(message, Mapping):
        return dict(message)
    result = payload.get("result")
    if isinstance(result, Mapping):
        nested = result.get("message")
        if isinstance(nested, Mapping):
            return dict(nested)
    return None


def _task_state(task: Mapping[str, Any]) -> str | None:
    status = task.get("status")
    if isinstance(status, Mapping):
        return _optional_str(status.get("state"))
    return None


def _should_poll_task(task: Mapping[str, Any]) -> bool:
    return bool(
        _optional_str(task.get("id"))
        and _normalized_state(_task_state(task)) not in _STOP_STATES
    )


def _normalized_state(state: str | None) -> str | None:
    if not state:
        return None
    normalized = state.upper()
    if not normalized.startswith("TASK_STATE_"):
        normalized = f"TASK_STATE_{normalized}"
    return normalized


def _task_text(task: Mapping[str, Any]) -> str:
    texts: list[str] = []
    status = task.get("status")
    if isinstance(status, Mapping):
        message = status.get("message")
        if isinstance(message, Mapping):
            texts.extend(_text_parts(message.get("parts")))
    for artifact in _artifacts(task):
        texts.extend(_text_parts(artifact.get("parts")))
    return "\n\n".join(text for text in texts if text).strip()


def _artifacts(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifacts = task.get("artifacts")
    if not isinstance(artifacts, list):
        return []
    return [dict(item) for item in artifacts if isinstance(item, Mapping)]


def _text_parts(parts: Any) -> list[str]:
    texts: list[str] = []
    if not isinstance(parts, list):
        return texts
    for part in parts:
        if not isinstance(part, Mapping):
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
            continue
        if "data" in part:
            texts.append(json.dumps(part.get("data"), ensure_ascii=False))
    return texts


def _json_text(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False)
    except TypeError:
        return str(payload)


def _tool_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip()).strip("_").lower()
    if not normalized:
        normalized = "remote_agent"
    if not normalized.startswith("a2a_"):
        normalized = f"a2a_{normalized}"
    return normalized


class _PinnedA2ATransport(httpx.AsyncBaseTransport):
    """Resolve, validate, and pin A2A hosts before opening a connection."""

    def __init__(self, *, allow_private_networks: bool):
        self._allow_private_networks = allow_private_networks
        self._transport = httpx.AsyncHTTPTransport(trust_env=False, http2=False)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self._allow_private_networks:
            return await self._transport.handle_async_request(request)

        original_url = request.url
        original_host = request.headers.get("Host")
        addresses = await _resolve_public_addresses(str(original_url))
        last_error: httpx.TransportError | None = None
        for index, address in enumerate(addresses):
            request.url = original_url.copy_with(host=address)
            request.headers["Host"] = _host_header_value(str(original_url))
            if original_url.scheme == "https":
                request.extensions["sni_hostname"] = _hostname_for_url(
                    str(original_url)
                )
            try:
                return await self._transport.handle_async_request(request)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_error = exc
                if index == len(addresses) - 1:
                    raise
            finally:
                request.url = original_url
                if original_host is None:
                    request.headers.pop("Host", None)
                else:
                    request.headers["Host"] = original_host
        if last_error is not None:
            raise last_error
        raise httpx.ConnectError(
            "A2A endpoint host could not be resolved.", request=request
        )

    async def aclose(self) -> None:
        await self._transport.aclose()


async def _resolve_public_addresses(value: str) -> list[str]:
    parsed = urlsplit(value)
    hostname = (parsed.hostname or "").rstrip(".").lower()
    reject_private_network_host(hostname)
    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    try:
        loop = asyncio.get_running_loop()
        records = await loop.run_in_executor(
            None,
            socket.getaddrinfo,
            hostname,
            port,
            socket.AF_UNSPEC,
            socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise httpx.ConnectError("A2A endpoint host could not be resolved.") from exc

    addresses: list[str] = []
    for record in records:
        sockaddr = record[4]
        if not sockaddr:
            continue
        address = str(sockaddr[0])
        reject_private_network_host(address)
        if address not in addresses:
            addresses.append(address)
    if not addresses:
        raise httpx.ConnectError("A2A endpoint host could not be resolved.")
    return addresses


def _hostname_for_url(value: str) -> str:
    hostname = urlsplit(value).hostname
    if not hostname:
        raise ValueError("A2A URL must include a hostname.")
    return hostname.rstrip(".").lower()


def _host_header_value(value: str) -> str:
    parsed = urlsplit(value)
    hostname = _hostname_for_url(value)
    host = f"[{hostname}]" if ":" in hostname else hostname
    port = parsed.port
    if port and not (
        (parsed.scheme.lower() == "http" and port == 80)
        or (parsed.scheme.lower() == "https" and port == 443)
    ):
        return f"{host}:{port}"
    return host


def _clean_url(
    value: str | None,
    *,
    allow_private_networks: bool = False,
) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or not parsed.hostname
    ):
        raise ValueError("A2A URLs must be absolute HTTP(S) URLs.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("A2A URLs must not contain embedded credentials.")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("A2A URLs must contain a valid port.") from exc
    if not allow_private_networks:
        reject_private_network_host(parsed.hostname)
    return value


def _same_origin(left: str, right: str) -> bool:
    return _url_origin(left) == _url_origin(right)


def _url_origin(value: str) -> tuple[str, str, int]:
    parsed = urlsplit(value)
    default_port = 443 if parsed.scheme.lower() == "https" else 80
    return (
        parsed.scheme.lower(),
        str(parsed.hostname).lower(),
        parsed.port or default_port,
    )


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _string_headers(headers: Mapping[str, Any] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    if not isinstance(headers, Mapping):
        return result
    for key, value in headers.items():
        if isinstance(key, str) and key and isinstance(value, str):
            result[key] = value
    return result
