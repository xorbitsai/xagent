from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

from json_repair import loads as repair_json_loads

from ..tool_protocol import (
    ToolProtocolViolation,
    tool_protocol_error_response,
)
from ..types import PROVIDER_STATE_METADATA_KEY, ChunkType, StreamChunk

logger = logging.getLogger(__name__)

_PROVIDER = "deepseek"

# DeepSeek's mandatory reasoning-replay contract, shared by any client that
# talks the DeepSeek tool protocol (``DeepSeekLLM`` directly, and
# ``OpenRouterLLM`` for OpenRouter's deepseek-authored slugs): a response that
# carries reasoning content on a tool-call message must have that exact
# content replayed on the assistant message the next time it is sent back.
# The namespace/key below are this client's own internal vocabulary for that
# state, not necessarily the wire field name a given provider used to send it
# (see ``deepseek_reasoning_provider_state``'s ``fields`` argument).
DEEPSEEK_PROVIDER_STATE_NAMESPACE = "deepseek"
DEEPSEEK_REASONING_CONTENT_STATE_KEY = "reasoning_content"
_SERIALIZED_TOOL_CALL_RE = re.compile(
    r"<[^>\n]*dsml[^>\n]*tool_calls",
    re.IGNORECASE,
)
_PARTIAL_MARKER_TARGET = "dsmltool_calls"
_PARTIAL_MARKER_SCAN_LIMIT = 64
_MARKER_SEPARATOR_RE = re.compile(r"[\s|｜]")
_ARGUMENTS_PREVIEW_LIMIT = 4096
_ERROR_PREVIEW_LIMIT = 512


def normalize_deepseek_response(
    response: Any,
    *,
    tools: list[dict[str, Any]] | None,
) -> Any:
    """Validate a response, repairing malformed tool arguments in place."""

    if not tools:
        return response
    violation = _response_violation(response, tools=tools)
    if violation is None:
        return response
    raw = response.get("raw") if isinstance(response, dict) else None
    error_response = tool_protocol_error_response(violation, raw=raw)
    # Preserve the top-level usage stamp the adapter put on the original
    # envelope so token accounting survives the error rebuild.
    if isinstance(response, dict) and response.get("usage") is not None:
        error_response["usage"] = response["usage"]
    return error_response


async def adapt_deepseek_stream(
    stream: AsyncIterator[StreamChunk],
    *,
    tools: list[dict[str, Any]] | None,
) -> AsyncIterator[StreamChunk]:
    if not tools:
        async for chunk in stream:
            yield chunk
        return

    text = ""
    emitted_text_length = 0
    buffered_tool_chunk: StreamChunk | None = None
    terminal_chunk: StreamChunk | None = None
    usage_chunks: list[StreamChunk] = []
    violation: ToolProtocolViolation | None = None
    withheld_tool_tail = False

    async for chunk in stream:
        if chunk.is_token():
            delta = str(chunk.delta or chunk.content or "")
            text += delta
            if violation is None:
                violation = _serialized_content_violation(text)
            if violation is not None:
                continue
            safe_length = _safe_streaming_text_length(text)
            if safe_length > emitted_text_length:
                safe_delta = text[emitted_text_length:safe_length]
                emitted_text_length = safe_length
                yield StreamChunk(
                    type=ChunkType.TOKEN,
                    content=safe_delta,
                    delta=safe_delta,
                    raw=chunk.raw,
                )
            continue

        if chunk.is_tool_call():
            buffered_tool_chunk = chunk
            if violation is None:
                violation = _streaming_tool_call_violation(chunk)
            if violation is None:
                violation = _complete_streaming_tool_call_violation(
                    chunk,
                    tools=tools,
                )
            if violation is None:
                safe_chunk, withheld_tool_tail = _safe_streaming_tool_chunk(chunk)
                yield safe_chunk
            continue
        if chunk.is_usage():
            usage_chunks.append(chunk)
            continue
        if chunk.is_end():
            terminal_chunk = chunk
            continue
        yield chunk

    if violation is None and buffered_tool_chunk is not None:
        violation = _response_violation(
            {
                "content": text,
                "tool_calls": buffered_tool_chunk.tool_calls,
            },
            tools=tools,
        )

    if violation is not None:
        raw = (
            buffered_tool_chunk.raw
            if buffered_tool_chunk is not None
            else terminal_chunk.raw
            if terminal_chunk is not None
            else None
        )
        yield StreamChunk(
            type=ChunkType.PROTOCOL_ERROR,
            protocol_error=violation.to_dict(),
            raw=raw,
        )
    else:
        if emitted_text_length < len(text):
            delta = text[emitted_text_length:]
            yield StreamChunk(
                type=ChunkType.TOKEN,
                content=delta,
                delta=delta,
                raw=terminal_chunk.raw if terminal_chunk is not None else None,
            )
        if buffered_tool_chunk is not None and withheld_tool_tail:
            yield buffered_tool_chunk
        elif terminal_chunk is not None:
            yield terminal_chunk

    for chunk in usage_chunks:
        yield chunk


def deepseek_reasoning_provider_state_payload(reasoning_content: Any) -> dict[str, Any]:
    """Wrap already-extracted reasoning text in the shared provider-state shape."""
    return {
        DEEPSEEK_PROVIDER_STATE_NAMESPACE: {
            DEEPSEEK_REASONING_CONTENT_STATE_KEY: reasoning_content,
        },
    }


def deepseek_reasoning_provider_state(
    result: dict[str, Any],
    *,
    fields: tuple[str, ...],
) -> dict[str, Any]:
    """Capture DeepSeek-protocol reasoning content into a provider-state payload.

    ``fields`` lists the wire field spellings to look for, in precedence
    order. Checks ``result`` itself first (the shape the transport layer
    already normalizes onto), then falls back to the raw response's message
    payload (``result["raw"]["choices"][0]["message"]``) for a provider whose
    SDK preserves a field spelling the transport layer does not normalize.
    A field counts as present only when its value is not ``None`` -- an
    explicit empty string is a valid, must-preserve value; only genuine
    absence should return ``{}``.

    Direct DeepSeek only ever sends ``reasoning_content``, so it passes a
    single-element ``fields`` tuple and the raw fallback is never reached in
    practice; OpenRouter's deepseek-authored slugs may use either
    ``reasoning_content`` or the ``reasoning`` alias, so it passes both.
    """
    value, found = _first_reasoning_value(result, fields)
    if not found:
        message = _raw_response_message(
            result.get("raw") if isinstance(result, dict) else None
        )
        if message is not None:
            value, found = _first_reasoning_value(message, fields)
    if not found:
        return {}
    return deepseek_reasoning_provider_state_payload(value)


def restore_deepseek_reasoning_content(
    messages: list[dict[str, Any]],
    *,
    model_name: str,
) -> list[dict[str, Any]]:
    """Replay captured DeepSeek reasoning content onto assistant history.

    An assistant message that previously captured ``reasoning_content`` under
    the shared provider-state marker gets it translated back to the
    ``reasoning_content`` field DeepSeek's API expects. An assistant message
    with tool calls but no captured state gets an empty-string fallback
    instead, so the history stays structurally valid for a DeepSeek request
    (older context may predate capture, or come from a rebuilt session).

    This is unconditional: it does not look at whether thinking is enabled
    for the *current* request, because the replay requirement is about the
    history's own shape, not this call's configuration.

    Logs one INFO summary per call (only when at least one assistant
    tool-call message actually needed replay) counting how many messages
    replayed real captured content versus how many hit the empty-string
    fallback. A rising fallback count is the signal that something upstream
    is losing captured state (a rebuilt task history, a synthesized
    tool-call message) before it reaches here -- that loss does not raise on
    its own, since the fallback keeps the request valid. The log carries
    only the counts and ``model_name``, never the reasoning text itself.
    """
    prepared: list[dict[str, Any]] = []
    replayed_count = 0
    fallback_count = 0
    for message in messages:
        prepared_message = dict(message)
        provider_state = prepared_message.get(PROVIDER_STATE_METADATA_KEY)
        replayed_captured_content = False
        if isinstance(provider_state, dict):
            deepseek_metadata = provider_state.get(DEEPSEEK_PROVIDER_STATE_NAMESPACE)
            if (
                isinstance(deepseek_metadata, dict)
                and DEEPSEEK_REASONING_CONTENT_STATE_KEY in deepseek_metadata
            ):
                prepared_message[DEEPSEEK_REASONING_CONTENT_STATE_KEY] = (
                    deepseek_metadata[DEEPSEEK_REASONING_CONTENT_STATE_KEY]
                )
                replayed_captured_content = True
        if prepared_message.get("role") == "assistant" and prepared_message.get(
            "tool_calls"
        ):
            if replayed_captured_content:
                replayed_count += 1
            elif DEEPSEEK_REASONING_CONTENT_STATE_KEY not in prepared_message:
                prepared_message[DEEPSEEK_REASONING_CONTENT_STATE_KEY] = ""
                fallback_count += 1
        prepared.append(prepared_message)
    if replayed_count or fallback_count:
        logger.info(
            "DeepSeek reasoning replay for model %s: %d assistant message(s) "
            "replayed captured reasoning content, %d used the empty-string "
            "fallback",
            model_name,
            replayed_count,
            fallback_count,
        )
    return prepared


def reasoning_field_names(result: Any) -> tuple[str, ...]:
    """Return every reasoning-related key name observed on a captured result.

    Checks the same two places ``deepseek_reasoning_provider_state`` does --
    the ``result`` dict itself, then its raw response message -- and returns
    the union of keys starting with ``"reasoning"`` found there. For
    observability only: the caller logs these key *names*, never their
    values, so a provider that renamed its reasoning field can be diagnosed
    without ever putting reasoning content itself into a log line.
    """
    names: list[str] = []
    if isinstance(result, dict):
        names.extend(key for key in result if key.startswith("reasoning"))
        message = _raw_response_message(result.get("raw"))
        if message is not None:
            names.extend(key for key in message if key.startswith("reasoning"))
    return tuple(dict.fromkeys(names))


def _first_reasoning_value(source: Any, fields: tuple[str, ...]) -> tuple[Any, bool]:
    if not isinstance(source, dict):
        return None, False
    for field_name in fields:
        if field_name in source and source[field_name] is not None:
            return source[field_name], True
    return None, False


def _raw_response_message(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None
    message = first_choice.get("message")
    return message if isinstance(message, dict) else None


def _response_violation(
    response: Any,
    *,
    tools: list[dict[str, Any]] | None,
) -> ToolProtocolViolation | None:
    content = _response_content(response)
    violation = _serialized_content_violation(content)
    if violation is not None:
        return violation

    for tool_call in _response_tool_calls(response):
        violation = _tool_call_violation(tool_call, tools=tools)
        if violation is not None:
            return violation
    return None


def _serialized_content_violation(content: Any) -> ToolProtocolViolation | None:
    if _serialized_tool_call_start(content) is None:
        return None
    return ToolProtocolViolation(
        provider=_PROVIDER,
        code="serialized_tool_call_content",
        message="DeepSeek returned serialized tool-call markup in assistant content.",
    )


def _streaming_tool_call_violation(
    chunk: StreamChunk,
) -> ToolProtocolViolation | None:
    for tool_call in chunk.tool_calls:
        function = _function_payload(tool_call)
        name = function.get("name")
        arguments = function.get("arguments")
        if _contains_serialized_tool_call(arguments):
            return ToolProtocolViolation(
                provider=_PROVIDER,
                code="nested_serialized_tool_call",
                message=(
                    "DeepSeek embedded serialized tool-call markup inside "
                    f"{name or 'a tool call'!r}."
                ),
            )
    return None


def _complete_streaming_tool_call_violation(
    chunk: StreamChunk,
    *,
    tools: list[dict[str, Any]] | None,
) -> ToolProtocolViolation | None:
    """Validate complete accumulated calls without rejecting partial chunks."""
    for tool_call in chunk.tool_calls:
        arguments = _function_payload(tool_call).get("arguments", {})
        if not _arguments_are_ready_for_validation(arguments):
            continue
        violation = _tool_call_violation(tool_call, tools=tools)
        if violation is not None:
            return violation
    return None


def _safe_streaming_tool_chunk(
    chunk: StreamChunk,
) -> tuple[StreamChunk, bool]:
    safe_tool_calls: list[dict[str, Any]] = []
    withheld = False
    for tool_call in chunk.tool_calls:
        if not isinstance(tool_call, dict):
            safe_tool_calls.append(tool_call)
            continue
        safe_tool_call = dict(tool_call)
        function = tool_call.get("function")
        if isinstance(function, dict):
            safe_function = dict(function)
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                safe_length = _safe_streaming_text_length(arguments)
                if safe_length < len(arguments):
                    safe_function["arguments"] = arguments[:safe_length]
                    withheld = True
            safe_tool_call["function"] = safe_function
        safe_tool_calls.append(safe_tool_call)
    return replace(chunk, tool_calls=safe_tool_calls), withheld


def _tool_call_violation(
    tool_call: Any,
    *,
    tools: list[dict[str, Any]] | None,
) -> ToolProtocolViolation | None:
    function = _function_payload(tool_call)
    name = function.get("name")
    if not isinstance(name, str) or not name:
        return ToolProtocolViolation(
            provider=_PROVIDER,
            code="malformed_tool_call",
            message="DeepSeek returned a tool call without a function name.",
        )

    arguments = function.get("arguments", {})
    if isinstance(arguments, str) and not arguments.strip():
        # Rebind the local for validation only. `function["arguments"]` keeps the
        # raw blank string: normalizing it to `"{}"` would defeat pass-through
        # and crash the auto/DAG paths on a syntactically valid empty plan.
        arguments = {}
    argument_details: dict[str, Any] | None = None
    if isinstance(arguments, str):
        original_arguments = arguments
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as original_error:
            argument_details = _argument_diagnostics(
                original_arguments,
                json_error=original_error,
            )
            if not _is_structurally_complete_json_object(original_arguments):
                argument_details["repair_status"] = "skipped_incomplete"
                return _malformed_arguments_violation(
                    name,
                    details=argument_details,
                )
            try:
                arguments = repair_json_loads(
                    original_arguments,
                    logging=False,
                )
            except Exception as repair_error:  # noqa: BLE001
                argument_details.update(
                    {
                        "repair_status": "failed",
                        "repair_error": _bounded_text(repair_error),
                    }
                )
                return _malformed_arguments_violation(
                    name,
                    details=argument_details,
                )
            if not isinstance(arguments, dict):
                argument_details["repair_status"] = "failed_non_dict"
                return _malformed_arguments_violation(
                    name,
                    message=f"DeepSeek returned non-object arguments for {name!r}.",
                    details=argument_details,
                )
            argument_details["repair_status"] = "repaired"
            applied = _set_tool_call_arguments(
                tool_call,
                json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
            )
            if not applied:
                argument_details["repair_status"] = "repair_application_failed"
                return _malformed_arguments_violation(
                    name,
                    details=argument_details,
                )
    if not isinstance(arguments, dict):
        details = argument_details or _argument_diagnostics(arguments)
        details.setdefault("repair_status", "not_applicable")
        return _malformed_arguments_violation(
            name,
            message=f"DeepSeek returned non-object arguments for {name!r}.",
            details=details,
        )

    if _contains_serialized_tool_call(arguments):
        return ToolProtocolViolation(
            provider=_PROVIDER,
            code="nested_serialized_tool_call",
            message=f"DeepSeek embedded serialized tool-call markup inside {name!r}.",
            details=argument_details,
        )

    schemas = _tool_schema_by_name(tools)
    if not schemas:
        return None
    schema = schemas.get(name)
    if schema is None:
        return ToolProtocolViolation(
            provider=_PROVIDER,
            code="unavailable_tool_call",
            message=f"DeepSeek returned unavailable tool call {name!r}.",
            details=argument_details,
        )

    parameters = schema.get("parameters")
    if isinstance(parameters, dict) and parameters.get("additionalProperties") is False:
        properties = parameters.get("properties")
        allowed = set(properties) if isinstance(properties, dict) else set()
        unexpected = set(arguments) - allowed
        if unexpected:
            return ToolProtocolViolation(
                provider=_PROVIDER,
                code="unexpected_tool_arguments",
                message=(
                    f"DeepSeek returned unexpected arguments for {name!r}: "
                    f"{', '.join(sorted(unexpected))}."
                ),
                details=argument_details,
            )
    return None


def _malformed_arguments_violation(
    name: str,
    *,
    message: str | None = None,
    details: dict[str, Any],
) -> ToolProtocolViolation:
    return ToolProtocolViolation(
        provider=_PROVIDER,
        code="malformed_tool_arguments",
        message=message or f"DeepSeek returned malformed arguments for {name!r}.",
        details=details,
    )


def _arguments_are_ready_for_validation(arguments: Any) -> bool:
    if not isinstance(arguments, str):
        return True
    try:
        json.loads(arguments)
    except json.JSONDecodeError:
        return _is_structurally_complete_json_object(arguments)
    return True


def _is_structurally_complete_json_object(arguments: str) -> bool:
    stripped = arguments.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        return False

    pairs = {"}": "{", "]": "["}
    stack: list[str] = []
    string_quote: str | None = None
    escaped = False
    for char in stripped:
        if string_quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == string_quote:
                string_quote = None
            continue

        if char in {'"', "'"}:
            string_quote = char
        elif char in "{[":
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return False

    return string_quote is None and not escaped and not stack


def _set_tool_call_arguments(tool_call: Any, arguments: str) -> bool:
    if isinstance(tool_call, dict):
        function = tool_call.get("function")
        if isinstance(function, dict):
            function["arguments"] = arguments
            return True
        if "args" in tool_call:
            tool_call["args"] = arguments
            return True
        tool_call["arguments"] = arguments
        return True

    function = getattr(tool_call, "function", None)
    try:
        setattr(function, "arguments", arguments)
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def _argument_diagnostics(
    arguments: Any,
    *,
    json_error: json.JSONDecodeError | None = None,
) -> dict[str, Any]:
    original = arguments if isinstance(arguments, str) else repr(arguments)
    preview = original[:_ARGUMENTS_PREVIEW_LIMIT]
    details: dict[str, Any] = {
        "original_arguments_preview": preview,
        "original_arguments_length": len(original),
        "original_arguments_truncated": len(original) > len(preview),
    }
    if json_error is not None:
        details["json_error"] = _bounded_text(json_error)
    return details


def _bounded_text(value: Any) -> str:
    return str(value)[:_ERROR_PREVIEW_LIMIT]


def _contains_serialized_tool_call(value: Any) -> bool:
    if isinstance(value, str):
        return _serialized_tool_call_start(value) is not None
    if isinstance(value, dict):
        return any(_contains_serialized_tool_call(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_serialized_tool_call(item) for item in value)
    return False


def _serialized_tool_call_start(content: Any) -> int | None:
    if not isinstance(content, str):
        return None
    match = _SERIALIZED_TOOL_CALL_RE.search(content)
    return match.start() if match is not None else None


def _safe_streaming_text_length(content: str) -> int:
    serialized_start = _serialized_tool_call_start(content)
    if serialized_start is not None:
        return serialized_start

    for match in re.finditer("<", content):
        tail_start = match.start() + 1
        tail = content[tail_start : tail_start + _PARTIAL_MARKER_SCAN_LIMIT]
        if "\n" in tail or ">" in tail:
            continue
        normalized_tail = _MARKER_SEPARATOR_RE.sub("", tail).casefold()
        if _PARTIAL_MARKER_TARGET.startswith(normalized_tail):
            return match.start()
    return len(content)


def _response_content(response: Any) -> Any:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return response.get("content")
    return None


def _response_tool_calls(response: Any) -> list[Any]:
    if isinstance(response, dict):
        return list(response.get("tool_calls") or [])
    return []


def _function_payload(tool_call: Any) -> dict[str, Any]:
    if isinstance(tool_call, dict):
        function = tool_call.get("function")
        if isinstance(function, dict):
            return function
        return {
            "name": tool_call.get("name"),
            "arguments": tool_call.get("args", tool_call.get("arguments", {})),
        }
    function = getattr(tool_call, "function", None)
    return {
        "name": getattr(function, "name", None),
        "arguments": getattr(function, "arguments", {}),
    }


def _tool_schema_by_name(
    tools: list[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    schemas: dict[str, dict[str, Any]] = {}
    for tool in tools or []:
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if isinstance(name, str) and name:
            schemas[name] = function
    return schemas
