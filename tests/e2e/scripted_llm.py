from __future__ import annotations

import json
from collections import deque
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.chat.types import ChunkType, StreamChunk

ScriptedResponse = str | dict[str, Any]


class ScriptedLLM(BaseLLM):
    def __init__(
        self,
        responses: list[ScriptedResponse],
        *,
        model_name: str = "scripted-test-llm",
    ):
        self._responses = deque(responses)
        self._model_name = model_name

    @property
    def abilities(self) -> list[str]:
        return ["chat", "tool_calling"]

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def supports_thinking_mode(self) -> bool:
        return False

    async def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        output_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str | dict[str, Any]:
        del (
            messages,
            temperature,
            max_tokens,
            tools,
            tool_choice,
            response_format,
            thinking,
            output_config,
            kwargs,
        )
        if not self._responses:
            raise AssertionError("ScriptedLLM received more calls than expected")
        return self._responses.popleft()

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        output_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        response = await self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking=thinking,
            output_config=output_config,
            **kwargs,
        )
        if isinstance(response, dict):
            yield StreamChunk(
                type=ChunkType.TOOL_CALL,
                tool_calls=response.get("tool_calls", []),
                raw=response,
            )
        else:
            yield StreamChunk(type=ChunkType.TOKEN, content=response, delta=response)
        yield StreamChunk(
            type=ChunkType.USAGE,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
        yield StreamChunk(type=ChunkType.END)


def _json_text_response(entry: dict[str, Any], path: Path, index: int) -> str:
    content = entry.get("content")
    if not isinstance(content, (dict, list)):
        raise ValueError(
            f"Scripted LLM response {index} json_text content must be an object or array: {path}"
        )
    serialized = json.dumps(content)
    if isinstance(content, dict) and content.get("type") == "tool_call":
        return {"content": serialized, "done": False}
    return serialized


def _tool_call_response(entry: dict[str, Any]) -> dict[str, Any]:
    tool_calls = []
    for tool_call in entry.get("tool_calls", []):
        normalized_tool_call = dict(tool_call)
        function = normalized_tool_call.get("function")
        if isinstance(function, dict) and not isinstance(
            function.get("arguments"), str
        ):
            normalized_function = dict(function)
            normalized_function["arguments"] = json.dumps(
                normalized_function.get("arguments", {})
            )
            normalized_tool_call["function"] = normalized_function
        tool_calls.append(normalized_tool_call)

    return {
        "type": "tool_call",
        "tool_calls": tool_calls,
    }


def _normalize_scripted_response(
    response: Any,
    *,
    path: Path,
    index: int,
) -> ScriptedResponse:
    if isinstance(response, str):
        return response
    if not isinstance(response, dict):
        raise ValueError(
            f"Scripted LLM response {index} must be a string or object: {path}"
        )

    kind = response.get("kind")
    if kind == "json_text":
        return _json_text_response(response, path, index)
    if kind == "tool_call":
        return _tool_call_response(response)
    return response


def load_scripted_responses(path: Path) -> list[ScriptedResponse]:
    with path.open(encoding="utf-8") as handle:
        responses = json.load(handle)
    if not isinstance(responses, list):
        raise ValueError(f"Scripted LLM responses must be a list: {path}")
    return [
        _normalize_scripted_response(response, path=path, index=index)
        for index, response in enumerate(responses)
    ]


def build_scripted_llm_from_json(
    path: Path,
    *,
    model_name: str = "scripted-test-llm",
) -> ScriptedLLM:
    return ScriptedLLM(load_scripted_responses(path), model_name=model_name)
