from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..context import ExecutionContext

REQUIRED_TOOL_CALL_FAILURE_REASON = "missing_required_tool_call"


def truncate_prompt_preview(value: str, *, limit: int = 1200) -> str:
    stripped = value.strip()
    if len(stripped) <= limit:
        return stripped
    return f"{stripped[:limit]}... [truncated]"


class RequiredToolCallError(ValueError):
    """Raised when an LLM response omits a tool call the caller requires."""

    def __init__(
        self,
        *,
        owner: str,
        tool_name: str,
        attempts: int = 1,
        user_message: str | None = None,
    ) -> None:
        self.owner = owner
        self.tool_name = tool_name
        self.attempts = attempts
        self.failure_reason = REQUIRED_TOOL_CALL_FAILURE_REASON
        self.user_message = user_message or (
            "The model did not return the required tool call. Please retry."
        )
        super().__init__(
            f"{owner} did not receive the required {tool_name} tool call "
            f"after {attempts} attempt(s)."
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "failure_reason": self.failure_reason,
            "required_tool_name": self.tool_name,
            "attempts": self.attempts,
        }


def extract_required_tool_arguments(
    response: Any,
    *,
    tool_name: str,
    owner: str,
    attempts: int = 1,
    user_message: str | None = None,
) -> Any:
    """Return arguments for a named required tool call or raise a structured error."""

    for function_payload in iter_tool_function_payloads(response):
        if function_payload.get("name") == tool_name:
            return function_payload.get("arguments", {})
    raise RequiredToolCallError(
        owner=owner,
        tool_name=tool_name,
        attempts=attempts,
        user_message=user_message,
    )


def append_user_message_preserving_turns(
    messages: list[dict[str, Any]],
    *,
    content: str,
    section_title: str | None = None,
) -> list[dict[str, Any]]:
    """Append user-scoped pattern feedback without creating adjacent user turns."""

    updated = [dict(message) for message in messages]
    if not updated or updated[-1].get("role") != "user":
        updated.append({"role": "user", "content": content})
        return updated

    last_message = dict(updated[-1])
    previous_content = last_message.get("content")
    if isinstance(previous_content, str):
        title = section_title or "Additional instruction"
        last_message["content"] = (
            f"{previous_content.rstrip()}\n\n{title}:\n{content}"
            if previous_content.strip()
            else content
        )
        updated[-1] = last_message
        return updated

    updated.append(
        {
            "role": "assistant",
            "content": "Acknowledged. I will follow the next instruction.",
        }
    )
    updated.append({"role": "user", "content": content})
    return updated


def iter_tool_function_payloads(response: Any) -> list[dict[str, Any]]:
    tool_calls = _response_tool_calls(response)
    function_payloads: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        function_payload = _function_payload(tool_call)
        if function_payload:
            function_payloads.append(function_payload)
    return function_payloads


def _response_tool_calls(response: Any) -> list[Any]:
    if isinstance(response, dict):
        return list(response.get("tool_calls") or [])
    return list(getattr(response, "tool_calls", []) or [])


def _function_payload(tool_call: Any) -> dict[str, Any] | None:
    if isinstance(tool_call, dict):
        function_payload = tool_call.get("function")
        return function_payload if isinstance(function_payload, dict) else None
    function_payload = getattr(tool_call, "function", None)
    if function_payload is None:
        return None
    return {
        "name": getattr(function_payload, "name", None),
        "arguments": getattr(function_payload, "arguments", {}),
    }


@dataclass
class PatternResult:
    """Standard result envelope returned by v2 patterns."""

    success: bool
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"success": self.success}
        if self.output is not None:
            result["output"] = self.output
        if self.error is not None:
            result["error"] = self.error
        if self.metadata:
            result.update(self.metadata)
        return result


class AgentPattern(ABC):
    """Abstract interface for agent execution patterns."""

    @abstractmethod
    async def run(
        self,
        context: ExecutionContext,
        tools: list[Any],
        llm: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute the pattern against an execution context."""
