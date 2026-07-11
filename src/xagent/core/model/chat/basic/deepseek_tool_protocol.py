from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

from ..tool_protocol import (
    ToolProtocolViolation,
    tool_protocol_error_response,
)
from ..types import ChunkType, StreamChunk

_PROVIDER = "deepseek"
_SERIALIZED_TOOL_CALL_RE = re.compile(
    r"<[^>\n]*dsml[^>\n]*tool_calls",
    re.IGNORECASE,
)
_PARTIAL_MARKER_TARGET = "dsmltool_calls"
_PARTIAL_MARKER_SCAN_LIMIT = 64
_MARKER_SEPARATOR_RE = re.compile(r"[\s|｜]")


def normalize_deepseek_response(
    response: Any,
    *,
    tools: list[dict[str, Any]] | None,
) -> Any:
    if not tools:
        return response
    violation = _response_violation(response, tools=tools)
    if violation is None:
        return response
    raw = response.get("raw") if isinstance(response, dict) else None
    return tool_protocol_error_response(violation, raw=raw)


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
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return ToolProtocolViolation(
                provider=_PROVIDER,
                code="malformed_tool_arguments",
                message=f"DeepSeek returned malformed arguments for {name!r}.",
            )
    if not isinstance(arguments, dict):
        return ToolProtocolViolation(
            provider=_PROVIDER,
            code="malformed_tool_arguments",
            message=f"DeepSeek returned non-object arguments for {name!r}.",
        )

    if _contains_serialized_tool_call(arguments):
        return ToolProtocolViolation(
            provider=_PROVIDER,
            code="nested_serialized_tool_call",
            message=f"DeepSeek embedded serialized tool-call markup inside {name!r}.",
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
            )
    return None


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
