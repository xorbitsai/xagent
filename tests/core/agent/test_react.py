from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import unicodedata
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.tools import tool as langchain_tool
from pydantic import BaseModel

from xagent.core.agent import (
    ExecutionContext,
    PatternRuntime,
    ReActPattern,
    ReActReasoningMode,
    ToolCallInterrupted,
    ToolCallRecord,
)
from xagent.core.agent.context.execution import CLOCK_TIMEZONE_METADATA_KEY
from xagent.core.agent.pattern.react.react import (
    _INTERACTION_TRIM_CHARS,
    _normalize_ask_user_interactions,
)
from xagent.core.agent.result import tool_result_succeeded
from xagent.core.file_ref import WORKSPACE_OUTPUT_FILES_TOOL_NAME
from xagent.core.model.chat.basic.router import RouterLLM
from xagent.core.model.chat.exceptions import LLMToolProtocolError
from xagent.core.model.chat.tool_protocol import (
    ToolProtocolViolation,
    tool_protocol_error_response,
)
from xagent.core.model.chat.types import ChunkType, StreamChunk


class CalculatorArgs(BaseModel):
    expression: str


class WriteFileArgs(BaseModel):
    file_path: str
    content: str


class SearchArgs(BaseModel):
    query: str
    count: int = 10


class EmptyArgs(BaseModel):
    pass


class FakeTool:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

        class Metadata:
            name = "calculator"
            description = "Evaluate math expressions."

        self.metadata = Metadata()

    def args_type(self) -> type[BaseModel]:
        return CalculatorArgs

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        self.calls.append(args)
        expression = args["expression"]
        return {"result": eval(expression), "expression": expression}  # noqa: S307


class FakeWorkspaceOutputTool:
    def __init__(self) -> None:
        class Metadata:
            name = WORKSPACE_OUTPUT_FILES_TOOL_NAME
            description = "List output files from the current workspace."

        self.metadata = Metadata()

    def args_type(self) -> type[BaseModel]:
        return EmptyArgs


class FakeWriteFileTool:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

        class Metadata:
            name = "write_file"
            description = "Write file content in workspace."

        self.metadata = Metadata()

    def args_type(self) -> type[BaseModel]:
        return WriteFileArgs

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        self.calls.append(args)
        path = args["file_path"]
        return {
            "success": True,
            "filename": path.split("/")[-1],
            "relative_path": f"output/{path.split('/')[-1]}",
            "file_path": f"/workspace/output/{path.split('/')[-1]}",
        }


class FakeSearchTool:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

        class Metadata:
            name = "zhipu_web_search"
            description = "Search the web."

        self.metadata = Metadata()

    def args_type(self) -> type[BaseModel]:
        return SearchArgs

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        self.calls.append(args)
        return {
            "results": [
                {
                    "title": "用不了NotebookLM,试试这个国产知识库工具",
                    "link": "https://example.com/a",
                },
                {
                    "title": "AnyGen真能取代NotebookLM?",
                    "link": "https://example.com/b",
                },
                {
                    "title": "Open NotebookLM",
                    "link": "https://example.com/c",
                },
            ]
        }


class FakeTraceSanitizingTool:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

        class Metadata:
            name = "custom_api"
            description = "Call a custom API."

        self.metadata = Metadata()

    def args_type(self) -> type[BaseModel]:
        return BaseModel

    def sanitize_tool_args_for_trace(self, args: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in args.items() if key != "headers"}

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        self.calls.append(args)
        return {"success": True, "args": args}


class FakeInPlaceTraceSanitizingTool(FakeTraceSanitizingTool):
    def sanitize_tool_args_for_trace(self, args: dict[str, Any]) -> dict[str, Any]:
        args.pop("headers", None)
        if isinstance(args.get("body"), dict):
            args["body"].pop("secret", None)
        return args


class FakeGroupedTool:
    def __init__(self, name: str, category: str) -> None:
        self.calls: list[dict[str, Any]] = []

        class Metadata:
            description = "Run grouped work."

        self.metadata = Metadata()
        self.metadata.name = name
        self.metadata.category = category

    def args_type(self) -> type[BaseModel]:
        return SearchArgs

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        self.calls.append(args)
        return {
            "results": [{"title": self.metadata.name, "link": "https://example.com"}]
        }


class FakeBrowserNavigateTool:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

        class Metadata:
            name = "browser_navigate"
            description = "Navigate a browser page."

        self.metadata = Metadata()

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        self.calls.append(args)
        return {
            "success": True,
            "session_id": args.get("session_id", ""),
            "url": args["url"],
            "title": "",
            "message": "ok",
        }


class BrokenTool:
    def __init__(self) -> None:
        class Metadata:
            name = "broken"
            description = "Always fails."

        self.metadata = Metadata()

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        raise RuntimeError(f"broken with {args}")


class FailingResultTool:
    def __init__(self) -> None:
        class Metadata:
            name = "failing_result"
            description = "Returns a failed tool result."

        self.metadata = Metadata()

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        return {"success": False, "output": "", "error": f"failed with {args}"}


class StatusErrorResultTool:
    def __init__(self) -> None:
        class Metadata:
            name = "status_error_result"
            description = "Returns a status=error tool result."

        self.metadata = Metadata()

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        return {"status": "error", "message": f"failed with {args}"}


class IsErrorResultTool:
    def __init__(self) -> None:
        class Metadata:
            name = "is_error_result"
            description = "Returns an MCP-style is_error result."

        self.metadata = Metadata()

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        return {
            "is_error": True,
            "content": [{"text": f"failed with {args}"}],
        }


class ClassifiedFailureResultTool:
    def __init__(self) -> None:
        class Metadata:
            name = "classified_failure_result"
            description = "Returns a classified delegated-agent failure."

        self.metadata = Metadata()

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        return {
            "success": False,
            "is_error": True,
            "status": "error",
            "failure_code": "unsupported_nested_interaction",
            "error": "Nested agent calls cannot forward interactive prompts.",
            "output": "Nested agent calls cannot forward interactive prompts.",
            "response": "Nested agent calls cannot forward interactive prompts.",
        }


class FakeAskUserTool:
    def __init__(self) -> None:
        class Metadata:
            name = "ask_user_question"
            description = "Legacy ask user tool."

        self.metadata = Metadata()

    def args_type(self) -> type[BaseModel]:
        return BaseModel

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        return args


class LegacySchemaArgs:
    @classmethod
    def schema(cls) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        }


class LegacySchemaTool:
    def __init__(self) -> None:
        class Metadata:
            name = "legacy_schema"
            description = "Legacy schema tool."

        self.metadata = Metadata()

    def args_type(self) -> type[LegacySchemaArgs]:
        return LegacySchemaArgs

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        return {"value": args["value"]}


class FakeLLM:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeNamedLLM(FakeLLM):
    """FakeLLM carrying a ``model_name``, for tests that assert on it in logs."""

    model_name = "fake-model"


class StreamingFinalAnswerLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        raise AssertionError("streaming ReAct path should not call chat()")

    async def stream_chat(self, **kwargs: Any) -> Any:
        self.stream_calls.append(kwargs)
        tool_names = [
            tool["function"]["name"] for tool in list(kwargs.get("tools") or [])
        ]
        if tool_names == ["final_answer"]:
            yield StreamChunk(
                type=ChunkType.TOOL_CALL,
                tool_calls=[
                    {
                        "id": "call_final",
                        "function": {
                            "name": "final_answer",
                            "arguments": (
                                '{"response_language":"English",'
                                '"answer":"The result is 4."}'
                            ),
                        },
                    }
                ],
            )
            yield StreamChunk(type=ChunkType.END)
            return
        if tool_names:
            yield StreamChunk(
                type=ChunkType.TOOL_CALL,
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                        },
                    }
                ],
            )
            yield StreamChunk(type=ChunkType.END)
            return
        yield StreamChunk(type=ChunkType.TOKEN, delta="The result")
        yield StreamChunk(type=ChunkType.TOKEN, delta=" is 4.")
        yield StreamChunk(type=ChunkType.END)


class StreamingInvalidToolProtocolFinalAnswerLLM:
    def __init__(
        self,
        *,
        structured_work_tool: bool = False,
        partial_final_before_work_tool: bool = False,
        invalid_retry: bool = False,
    ) -> None:
        self.structured_work_tool = structured_work_tool
        self.partial_final_before_work_tool = partial_final_before_work_tool
        self.invalid_retry = invalid_retry
        self.stream_calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        raise AssertionError("invalid tool protocol path should stay streaming")

    async def stream_chat(
        self, messages: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> Any:
        if messages is not None:
            kwargs["messages"] = messages
        self.stream_calls.append(kwargs)
        call_index = len(self.stream_calls) - 1
        if call_index == 0:
            yield StreamChunk(
                type=ChunkType.TOOL_CALL,
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                        },
                    }
                ],
            )
        elif call_index == 1:
            if self.partial_final_before_work_tool:
                yield StreamChunk(
                    type=ChunkType.TOOL_CALL,
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "call_partial_final",
                            "function": {
                                "name": "final_answer",
                                "arguments": (
                                    '{"response_language":"English",'
                                    '"answer":"Draft answer"}'
                                ),
                            },
                        }
                    ],
                )
                yield StreamChunk(
                    type=ChunkType.TOOL_CALL,
                    tool_calls=[
                        {
                            "index": 1,
                            "id": "call_unavailable_work_tool",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression":"3+3"}',
                            },
                        }
                    ],
                )
            elif self.structured_work_tool:
                yield StreamChunk(
                    type=ChunkType.TOOL_CALL,
                    tool_calls=[
                        {
                            "id": "call_unavailable_work_tool",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression":"3+3"}',
                            },
                        }
                    ],
                )
            else:
                yield StreamChunk(
                    type=ChunkType.PROTOCOL_ERROR,
                    protocol_error={
                        "provider": "deepseek",
                        "code": "serialized_tool_call_content",
                        "message": "Invalid provider tool protocol.",
                    },
                )
        elif self.invalid_retry:
            yield StreamChunk(
                type=ChunkType.TOOL_CALL,
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_retry_partial_final",
                        "function": {
                            "name": "final_answer",
                            "arguments": (
                                '{"response_language":"English","answer":"Retry draft"}'
                            ),
                        },
                    }
                ],
            )
            yield StreamChunk(
                type=ChunkType.TOOL_CALL,
                tool_calls=[
                    {
                        "index": 1,
                        "id": "call_retry_unavailable_work_tool",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"4+4"}',
                        },
                    }
                ],
            )
        else:
            yield StreamChunk(
                type=ChunkType.TOOL_CALL,
                tool_calls=[
                    {
                        "id": "call_final",
                        "function": {
                            "name": "final_answer",
                            "arguments": json.dumps(
                                {
                                    "response_language": "Simplified Chinese",
                                    "answer": "目前没有找到可直接下载的音频片段。",
                                },
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            )
        yield StreamChunk(type=ChunkType.END)


class StreamingUnavailableToolRecoveryLLM:
    def __init__(self) -> None:
        self.stream_calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        raise AssertionError("unavailable-tool recovery path should stay streaming")

    async def stream_chat(self, **kwargs: Any) -> Any:
        self.stream_calls.append(kwargs)
        call_index = len(self.stream_calls) - 1
        if call_index == 0:
            tool_name = "calculator"
            arguments = '{"expression":"2+2"}'
            tool_call_id = "call_first_work"
        elif call_index == 1:
            raise LLMToolProtocolError(
                provider="deepseek",
                code="unavailable_tool_call",
                message="DeepSeek returned unavailable tool call 'calculator'.",
            )
        elif call_index == 2:
            tool_name = "calculator"
            arguments = '{"expression":"3+3"}'
            tool_call_id = "call_recovered_work"
        else:
            tool_name = "final_answer"
            arguments = json.dumps(
                {
                    "response_language": "English",
                    "answer": "The results are 4 and 6.",
                }
            )
            tool_call_id = "call_final"
        yield StreamChunk(
            type=ChunkType.TOOL_CALL,
            tool_calls=[
                {
                    "id": tool_call_id,
                    "function": {
                        "name": tool_name,
                        "arguments": arguments,
                    },
                }
            ],
        )
        yield StreamChunk(type=ChunkType.END)


class StreamingMalformedFinalAnswerRecoveryLLM:
    def __init__(self) -> None:
        self.stream_calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        raise AssertionError("malformed-tool recovery path should stay streaming")

    async def stream_chat(self, **kwargs: Any) -> Any:
        self.stream_calls.append(kwargs)
        call_index = len(self.stream_calls) - 1
        if call_index == 0:
            yield StreamChunk(
                type=ChunkType.TOOL_CALL,
                tool_calls=[
                    {
                        "id": "call_transcribe",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                        },
                    }
                ],
            )
            yield StreamChunk(type=ChunkType.END)
            return
        if call_index == 1:
            raise LLMToolProtocolError(
                provider="deepseek",
                code="malformed_tool_arguments",
                message="DeepSeek returned malformed arguments for 'final_answer'.",
                details={
                    "original_arguments_preview": '{"answer":',
                    "original_arguments_length": 10,
                    "repair_status": "skipped_incomplete",
                },
            )
        yield StreamChunk(
            type=ChunkType.TOOL_CALL,
            tool_calls=[
                {
                    "id": "call_final_repaired",
                    "function": {
                        "name": "final_answer",
                        "arguments": json.dumps(
                            {
                                "response_language": "Simplified Chinese",
                                "answer": "音频主要讲运动如何缓解精神疲劳。",
                            },
                            ensure_ascii=False,
                        ),
                    },
                }
            ],
        )
        yield StreamChunk(type=ChunkType.END)


class StreamingPlainTextFinalAnswerLLM:
    def __init__(self) -> None:
        self.stream_calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        raise AssertionError("streaming ReAct final answer should not call chat()")

    async def stream_chat(self, **kwargs: Any) -> Any:
        self.stream_calls.append(kwargs)
        yield StreamChunk(type=ChunkType.TOKEN, delta="Plain")
        yield StreamChunk(type=ChunkType.TOKEN, delta=" final.")
        yield StreamChunk(type=ChunkType.END)


class StreamingPreambleToolCallLLM:
    def __init__(self) -> None:
        self.stream_calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        raise AssertionError("streaming ReAct preamble path should not call chat()")

    async def stream_chat(self, **kwargs: Any) -> Any:
        self.stream_calls.append(kwargs)
        yield StreamChunk(type=ChunkType.TOKEN, delta="I will use a tool first.")
        yield StreamChunk(
            type=ChunkType.TOOL_CALL,
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression":"2+2"}',
                    },
                }
            ],
        )
        yield StreamChunk(type=ChunkType.END)


class StreamingFinalAnswerToolLLM:
    def __init__(self) -> None:
        self.stream_calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        raise AssertionError("streaming ReAct final_answer tool should not call chat()")

    async def stream_chat(self, **kwargs: Any) -> Any:
        self.stream_calls.append(kwargs)
        prefix = '{"answer":"'
        for arguments in [
            prefix + "Hi",
            prefix + "Hi there",
            prefix + 'Hi there."}',
        ]:
            yield StreamChunk(
                type=ChunkType.TOOL_CALL,
                tool_calls=[
                    {
                        "id": "call_final",
                        "function": {
                            "name": "final_answer",
                            "arguments": arguments,
                        },
                    }
                ],
            )
        yield StreamChunk(type=ChunkType.END)


class StreamingMixedFinalAnswerAndToolLLM:
    """First call bundles final_answer with a work tool; the retry answers for real."""

    def __init__(self) -> None:
        self.stream_calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        raise AssertionError("streaming mixed tool path should not call chat()")

    async def stream_chat(self, **kwargs: Any) -> Any:
        self.stream_calls.append(kwargs)
        if len(self.stream_calls) == 1:
            yield StreamChunk(
                type=ChunkType.TOOL_CALL,
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_final",
                        "function": {
                            "name": "final_answer",
                            "arguments": '{"answer":"Candidate"}',
                        },
                    },
                    {
                        "index": 1,
                        "id": "call_calc",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                        },
                    },
                ],
            )
            yield StreamChunk(type=ChunkType.END)
            return
        yield StreamChunk(
            type=ChunkType.TOOL_CALL,
            tool_calls=[
                {
                    "id": "call_final_2",
                    "function": {
                        "name": "final_answer",
                        "arguments": '{"answer":"4"}',
                    },
                }
            ],
        )
        yield StreamChunk(type=ChunkType.END)


class StreamingAnswerThenWorkToolLLM:
    """Streams a final_answer candidate before the batch also names a work tool.

    Exercises the case where the answer streamer has already emitted content
    by the time the work tool's name shows up in a later chunk of the same
    response, so the strip point is the only place left to close the stream.
    """

    def __init__(self) -> None:
        self.stream_calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        raise AssertionError("streaming answer-then-tool path should not call chat()")

    async def stream_chat(self, **kwargs: Any) -> Any:
        self.stream_calls.append(kwargs)
        if len(self.stream_calls) == 1:
            prefix = '{"answer":"'
            for arguments in [
                prefix + "Looking",
                prefix + "Looking that up now.",
                prefix + 'Looking that up now."}',
            ]:
                yield StreamChunk(
                    type=ChunkType.TOOL_CALL,
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "call_final",
                            "function": {
                                "name": "final_answer",
                                "arguments": arguments,
                            },
                        }
                    ],
                )
            yield StreamChunk(
                type=ChunkType.TOOL_CALL,
                tool_calls=[
                    {
                        "index": 1,
                        "id": "call_calc",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                        },
                    }
                ],
            )
            yield StreamChunk(type=ChunkType.END)
            return
        yield StreamChunk(
            type=ChunkType.TOOL_CALL,
            tool_calls=[
                {
                    "id": "call_final_2",
                    "function": {
                        "name": "final_answer",
                        "arguments": '{"answer":"4"}',
                    },
                }
            ],
        )
        yield StreamChunk(type=ChunkType.END)


class StreamingRepeatedGuardFinalAnswerLLM:
    def __init__(self) -> None:
        self.stream_calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        raise AssertionError("streaming repeated guard path should not call chat()")

    async def stream_chat(self, **kwargs: Any) -> Any:
        self.stream_calls.append(kwargs)
        call_index = len(self.stream_calls) - 1
        if call_index == 0:
            yield StreamChunk(
                type=ChunkType.TOOL_CALL,
                tool_calls=[
                    {
                        "id": "search_1",
                        "function": {
                            "name": "zhipu_web_search",
                            "arguments": '{"query":"AI news","count":10}',
                        },
                    }
                ],
            )
        elif call_index == 1:
            yield StreamChunk(
                type=ChunkType.TOOL_CALL,
                tool_calls=[
                    {
                        "id": "search_2",
                        "function": {
                            "name": "zhipu_web_search",
                            "arguments": '{"query":"AI news latest","count":5}',
                        },
                    }
                ],
            )
        elif call_index == 2:
            yield StreamChunk(
                type=ChunkType.TOOL_CALL,
                tool_calls=[
                    {
                        "id": "decision_1",
                        "function": {
                            "name": "react_decision",
                            "arguments": (
                                '{"action":"final_answer",'
                                '"reason":"已有搜索结果足够回答。"}'
                            ),
                        },
                    }
                ],
            )
        else:
            yield StreamChunk(
                type=ChunkType.TOOL_CALL,
                tool_calls=[
                    {
                        "id": "call_final",
                        "function": {
                            "name": "final_answer",
                            "arguments": (
                                '{"response_language":"English",'
                                '"answer":"Fallback answer."}'
                            ),
                        },
                    }
                ],
            )
        yield StreamChunk(type=ChunkType.END)


class OutboundCollector:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def __call__(self, payload: dict[str, Any]) -> None:
        self.events.append(payload)


class BlockingLLM:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = False

    async def chat(self, **kwargs: Any) -> Any:
        del kwargs
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class BlockingProtocolRetryLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.retry_started = asyncio.Event()
        self.cancelled = False

    async def chat(self, **kwargs: Any) -> Any:
        del kwargs
        self.calls += 1
        if self.calls == 1:
            return tool_protocol_error_response(
                ToolProtocolViolation(
                    provider="deepseek",
                    code="serialized_tool_call_content",
                    message="Invalid provider tool protocol.",
                )
            )
        self.retry_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class FakeTracer:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def start_trace(self, **kwargs: Any) -> None:
        self.events.append(("start_trace", kwargs))

    async def finish_trace(self, **kwargs: Any) -> None:
        self.events.append(("finish_trace", kwargs))

    async def start_span(self, **kwargs: Any) -> None:
        self.events.append(("start_span", kwargs))

    async def finish_span(self, **kwargs: Any) -> None:
        self.events.append(("finish_span", kwargs))


class TraceEventRecorder:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def trace_event(
        self,
        event_type: Any,
        *,
        task_id: str | None = None,
        step_id: str | None = None,
        data: dict[str, Any] | None = None,
        **_: Any,
    ) -> str:
        self.events.append(
            {
                "event_type": getattr(event_type, "value", str(event_type)),
                "task_id": task_id,
                "step_id": step_id,
                "data": data or {},
            }
        )
        return str(len(self.events))


class MemoryNote:
    content = "Use the stored project preference."
    keywords = ["project", "preference"]
    metadata = {"source": "test"}
    category = "react_memory"


class FakeMemoryStore:
    def __init__(self) -> None:
        self.searches: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> list[MemoryNote]:
        self.searches.append(kwargs)
        return [MemoryNote()]


class FakeSkillManager:
    def __init__(self) -> None:
        self.loaded: list[str] = []

    async def list_skills(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "test-skill",
                "description": "A test skill",
                "when_to_use": "Testing",
            }
        ]

    async def get_skill(self, name: str) -> dict[str, Any] | None:
        if name != "test-skill":
            return None
        self.loaded.append(name)
        return {
            "name": "test-skill",
            "description": "A test skill",
            "content": "Follow the selected skill instructions.",
        }


@pytest.mark.asyncio
async def test_react_pattern_runs_tool_call_then_final_answer() -> None:
    llm = FakeLLM(
        responses=[
            {
                "content": "I should calculate this first.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                        },
                    }
                ],
                "done": False,
            },
            {"content": "The result is 4.", "done": True},
        ]
    )
    pattern = ReActPattern(max_iterations=3)
    tool = FakeTool()
    context = ExecutionContext(system_prompt="You are helpful.")
    context.add_user_message("Calculate 2+2")

    result = await pattern.run(context=context, tools=[tool], llm=llm)

    assert result["success"] is True
    assert result["response"] == "The result is 4."
    assert tool.calls == [{"expression": "2+2"}]
    assert [message.role for message in context.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert context.messages[1].tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "calculator",
                "arguments": json.dumps({"expression": "2+2"}),
            },
        }
    ]
    assert llm.calls[0]["tools"][0]["function"]["name"] == "calculator"
    system_prompt = llm.calls[0]["messages"][0]["content"]
    assert "latest user message" in system_prompt
    assert re.search(r"Turn-start date \(UTC\): \d{4}-\d{2}-\d{2}", system_prompt)
    assert "use this date when forming search queries" in system_prompt
    assert "not supported by the conversation" in system_prompt
    assert "available context is insufficient" in system_prompt
    assert "quantitative data" in system_prompt
    assert "Do not write assistant text in the same response as a work tool call" in (
        system_prompt
    )


@pytest.mark.asyncio
async def test_react_pattern_carries_opaque_provider_state() -> None:
    provider_state = {"provider": {"field": ""}}
    llm = FakeLLM(
        responses=[
            {
                "content": "",
                "_xagent_provider_state": provider_state,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                        },
                    }
                ],
                "done": False,
            },
            {"content": "The result is 4.", "done": True},
        ]
    )
    pattern = ReActPattern(max_iterations=3)
    tool = FakeTool()
    context = ExecutionContext()
    context.add_user_message("Calculate 2+2")

    result = await pattern.run(context=context, tools=[tool], llm=llm)

    assert result["success"] is True
    assistant_messages = [
        message
        for message in llm.calls[1]["messages"]
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]
    assert assistant_messages[0]["_xagent_provider_state"] == provider_state
    assert "reasoning_content" not in assistant_messages[0]
    stored_assistant = next(
        message
        for message in context.messages
        if message.role == "assistant" and message.tool_calls
    )
    assert stored_assistant.metadata["_xagent_provider_state"] == provider_state


@pytest.mark.asyncio
async def test_react_pattern_does_not_persist_invalid_raw_tool_calls() -> None:
    llm = FakeLLM(
        responses=[
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_search",
                        "type": "function",
                        "function": {
                            "name": "zhipu_web_search",
                            "arguments": (
                                '{"query": "Germany vs Ivory Coast live score"'
                            ),
                        },
                    },
                    {
                        "id": "",
                        "type": "function",
                        "function": {
                            "arguments": (
                                '}{"query":"德国 科特迪瓦 世界杯 比分 2026 6月"}'
                            ),
                        },
                    },
                ],
            },
            {"content": "No live score found.", "done": True},
        ]
    )
    pattern = ReActPattern(max_iterations=3)
    context = ExecutionContext()
    context.add_user_message("搜索啊")

    result = await pattern.run(context=context, tools=[FakeSearchTool()], llm=llm)

    assert result["success"] is True
    assistant_tool_calls = context.messages[1].tool_calls
    assert assistant_tool_calls == [
        {
            "id": "call_search",
            "type": "function",
            "function": {
                "name": "zhipu_web_search",
                "arguments": json.dumps(
                    {"input": ('{"query": "Germany vs Ivory Coast live score"')},
                    ensure_ascii=False,
                ),
            },
        }
    ]
    second_prompt_tool_calls = [
        message["tool_calls"]
        for message in llm.calls[1]["messages"]
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]
    assert second_prompt_tool_calls == [assistant_tool_calls]


@pytest.mark.asyncio
async def test_react_pattern_streams_only_final_answer_after_tool_call() -> None:
    llm = StreamingFinalAnswerLLM()
    pattern = ReActPattern(max_iterations=3, finalize_after_tool_result=True)
    tool = FakeTool()
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="task-1")
    context.add_user_message("Calculate 2+2")
    outbound = OutboundCollector()
    runtime = PatternRuntime(execution_id="task-1", outbound_message_handler=outbound)

    result = await pattern.run(context=context, tools=[tool], llm=llm, runtime=runtime)

    assert result["success"] is True
    assert result["response"] == "The result is 4."
    assert tool.calls == [{"expression": "2+2"}]
    assert len(llm.calls) == 0
    assert len(llm.stream_calls) == 2
    assert llm.stream_calls[0]["tools"][0]["function"]["name"] == "calculator"
    assert [tool["function"]["name"] for tool in llm.stream_calls[1]["tools"]] == [
        "final_answer"
    ]
    forced_answer_description = llm.stream_calls[1]["tools"][0]["function"][
        "parameters"
    ]["properties"]["answer"]["description"]
    assert "get_workspace_output_files" not in forced_answer_description
    assert llm.stream_calls[1]["tool_choice"] == "required"
    assert [event["type"] for event in outbound.events] == [
        "final_answer_start",
        "final_answer_delta",
        "final_answer_end",
    ]


@pytest.mark.asyncio
async def test_react_recovers_unavailable_forced_final_with_full_tool_set() -> None:
    llm = StreamingUnavailableToolRecoveryLLM()
    pattern = ReActPattern(max_iterations=4, finalize_after_tool_result=True)
    tool = FakeTool()
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="task-1")
    context.add_user_message("Calculate 2+2 and 3+3")
    tracer = TraceEventRecorder()
    runtime = PatternRuntime(execution_id="task-1", tracer=tracer)

    result = await pattern.run(context=context, tools=[tool], llm=llm, runtime=runtime)

    assert result["success"] is True
    assert result["response"] == "The results are 4 and 6."
    assert tool.calls == [{"expression": "2+2"}, {"expression": "3+3"}]
    assert len(llm.stream_calls) == 4
    assert [schema["function"]["name"] for schema in llm.stream_calls[1]["tools"]] == [
        "final_answer"
    ]
    recovery_tool_names = [
        schema["function"]["name"] for schema in llm.stream_calls[2]["tools"]
    ]
    assert "calculator" in recovery_tool_names
    assert "final_answer" in recovery_tool_names
    assert (
        "Re-decide this turn using the complete current tool set"
        in llm.stream_calls[2]["messages"][0]["content"]
    )
    # stream_calls[1] is the forced final-answer turn. Assert the no-tools
    # grounding variant specifically: both branches carry the rule, so only the
    # can_call_tools=False wording proves run() reached the forced branch.
    forced_final_prompt = llm.stream_calls[1]["messages"][0]["content"]
    assert "quantitative data" in forced_final_prompt
    assert "invented values" in forced_final_prompt
    assert "use an appropriate tool" not in forced_final_prompt
    recovery_starts = [
        event
        for event in tracer.events
        if event["event_type"] == "action_start_llm"
        and event["data"].get("phase") == "unavailable_tool_call_recovery"
    ]
    assert len(recovery_starts) == 1


@pytest.mark.asyncio
async def test_react_repairs_malformed_forced_final_answer_arguments() -> None:
    llm = StreamingMalformedFinalAnswerRecoveryLLM()
    pattern = ReActPattern(max_iterations=3, finalize_after_tool_result=True)
    tool = FakeTool()
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="780")
    context.add_user_message("这个音频讲了什么？")
    tracer = TraceEventRecorder()
    runtime = PatternRuntime(execution_id="780", tracer=tracer)

    result = await pattern.run(context=context, tools=[tool], llm=llm, runtime=runtime)

    assert result["success"] is True
    assert result["response"] == "音频主要讲运动如何缓解精神疲劳。"
    assert tool.calls == [{"expression": "2+2"}]
    assert len(llm.stream_calls) == 3
    assert [schema["function"]["name"] for schema in llm.stream_calls[2]["tools"]] == [
        "final_answer"
    ]
    retry_prompt = llm.stream_calls[2]["messages"][0]["content"]
    assert "malformed JSON arguments" in retry_prompt
    assert "one complete JSON object" in retry_prompt
    recovery_starts = [
        event
        for event in tracer.events
        if event["event_type"] == "action_start_llm"
        and event["data"].get("phase") == "malformed_tool_arguments_recovery"
    ]
    assert len(recovery_starts) == 1
    protocol_errors = [
        event
        for event in tracer.events
        if event["event_type"] == "action_error_llm"
        and event["data"].get("protocol_code") == "malformed_tool_arguments"
    ]
    assert len(protocol_errors) == 1
    assert protocol_errors[0]["data"]["protocol_details"] == {
        "original_arguments_preview": '{"answer":',
        "original_arguments_length": 10,
        "repair_status": "skipped_incomplete",
    }


@pytest.mark.parametrize("structured_work_tool", [False, True])
@pytest.mark.asyncio
async def test_react_retries_invalid_forced_final_as_final_answer_tool(
    structured_work_tool: bool,
) -> None:
    llm = StreamingInvalidToolProtocolFinalAnswerLLM(
        structured_work_tool=structured_work_tool
    )
    pattern = ReActPattern(max_iterations=3, finalize_after_tool_result=True)
    tool = FakeTool()
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="task-1")
    context.add_user_message("Find an audio clip")
    outbound = OutboundCollector()
    runtime = PatternRuntime(execution_id="task-1", outbound_message_handler=outbound)

    result = await pattern.run(
        context=context,
        tools=[tool],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert result["response"] == "目前没有找到可直接下载的音频片段。"
    assert tool.calls == [{"expression": "2+2"}]
    assert len(llm.stream_calls) == 3
    assert [tool["function"]["name"] for tool in llm.stream_calls[1]["tools"]] == [
        "final_answer"
    ]
    assert llm.stream_calls[1]["tool_choice"] == "required"
    retry_tools = llm.stream_calls[2]["tools"]
    retry_tool_names = [tool["function"]["name"] for tool in retry_tools]
    if structured_work_tool:
        assert "calculator" in retry_tool_names
        assert "final_answer" in retry_tool_names
    else:
        assert retry_tool_names == ["final_answer"]
    assert llm.stream_calls[2]["tool_choice"] == "required"
    retry_prompt = llm.stream_calls[2]["messages"][0]["content"]
    if structured_work_tool:
        assert "complete current tool set" in retry_prompt
    else:
        assert "calling the final_answer control tool exactly once" in retry_prompt
    outbound_text = "".join(
        str(event.get("delta", "")) + str(event.get("content", ""))
        for event in outbound.events
    )
    assert "DSML" not in outbound_text
    assert [event["type"] for event in outbound.events] == [
        "final_answer_start",
        "final_answer_delta",
        "final_answer_end",
    ]


@pytest.mark.asyncio
async def test_react_retries_provider_tool_protocol_error_with_work_tools() -> None:
    llm = FakeLLM(
        responses=[
            tool_protocol_error_response(
                ToolProtocolViolation(
                    provider="deepseek",
                    code="nested_serialized_tool_call",
                    message="Invalid provider tool protocol.",
                )
            ),
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_write",
                        "function": {
                            "name": "write_file",
                            "arguments": json.dumps(
                                {
                                    "file_path": "podcast.md",
                                    "content": "# Podcast script\nComplete script body.",
                                }
                            ),
                        },
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_final",
                        "function": {
                            "name": "final_answer",
                            "arguments": json.dumps(
                                {
                                    "response_language": "English",
                                    "answer": "The podcast script is ready.",
                                }
                            ),
                        },
                    }
                ],
            },
        ]
    )
    pattern = ReActPattern(max_iterations=2)
    tool = FakeWriteFileTool()
    context = ExecutionContext()
    context.add_user_message("Create a podcast script and synthesize it.")
    tracer = TraceEventRecorder()
    runtime = PatternRuntime(tracer=tracer)

    result = await pattern.run(context=context, tools=[tool], llm=llm, runtime=runtime)

    assert result["success"] is True
    assert result["response"] == "The podcast script is ready."
    assert tool.calls == [
        {
            "file_path": "podcast.md",
            "content": "# Podcast script\nComplete script body.",
        }
    ]
    assert len(llm.calls) == 3
    retry_tool_names = [schema["function"]["name"] for schema in llm.calls[1]["tools"]]
    assert "write_file" in retry_tool_names
    assert "final_answer" in retry_tool_names
    assert (
        "If work remains, call the appropriate available work tool directly"
        in llm.calls[1]["messages"][0]["content"]
    )
    assert all(
        message.content != "Invalid provider tool protocol."
        for message in context.messages
        if isinstance(message.content, str)
    )
    llm_end_events = [
        event for event in tracer.events if event["event_type"] == "action_end_llm"
    ]
    assert llm_end_events[0]["data"]["success"] is False
    assert llm_end_events[0]["data"]["phase"] == ("discarded_invalid_tool_protocol")
    assert llm_end_events[1]["data"]["success"] is True


def test_react_accepts_null_tool_calls_in_forced_final_response() -> None:
    pattern = ReActPattern()

    assert not pattern._response_requires_tool_protocol_retry(
        {"content": "Done.", "tool_calls": None},
        force_final_answer=True,
    )


def test_react_grounding_rule_present_in_both_answer_paths() -> None:
    pattern = ReActPattern()
    context = ExecutionContext(system_prompt="You are helpful.")
    context.add_user_message("Build a KPI report")

    tool_prompt = pattern._messages_for_llm(
        context, has_tools=True, tool_names=["calculator"]
    )[0]["content"]
    lookup_tool_prompt = pattern._messages_for_llm(
        context,
        has_tools=True,
        tool_names=[WORKSPACE_OUTPUT_FILES_TOOL_NAME, "final_answer"],
    )[0]["content"]
    forced_prompt = pattern._messages_for_llm(
        context, has_tools=True, force_final_answer=True, tool_names=["final_answer"]
    )[0]["content"]

    for prompt in (tool_prompt, lookup_tool_prompt, forced_prompt):
        assert "quantitative data" in prompt
        assert "illustrative placeholders" in prompt
    assert "use an appropriate tool to verify" in tool_prompt
    assert "use an appropriate tool" not in forced_prompt
    assert "## FINAL DELIVERABLE FILE REFERENCES" not in tool_prompt
    assert "exact markdown_link" in tool_prompt
    assert "lookup is unavailable" in tool_prompt
    assert "call get_workspace_output_files once before finalizing" not in tool_prompt
    assert (
        "call get_workspace_output_files once before finalizing" in lookup_tool_prompt
    )
    assert forced_prompt.count("## FINAL DELIVERABLE FILE REFERENCES") == 1
    assert "call get_workspace_output_files once before finalizing" not in forced_prompt


@pytest.mark.asyncio
async def test_react_closes_invalid_initial_final_answer_stream_before_retry() -> None:
    llm = StreamingInvalidToolProtocolFinalAnswerLLM(
        partial_final_before_work_tool=True
    )
    pattern = ReActPattern(max_iterations=3, finalize_after_tool_result=True)
    tool = FakeTool()
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="task-1")
    context.add_user_message("Find an audio clip")
    outbound = OutboundCollector()
    runtime = PatternRuntime(execution_id="task-1", outbound_message_handler=outbound)

    result = await pattern.run(
        context=context,
        tools=[tool],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert result["response"] == "目前没有找到可直接下载的音频片段。"
    assert tool.calls == [{"expression": "2+2"}]
    assert [event["type"] for event in outbound.events] == [
        "final_answer_start",
        "final_answer_delta",
        "final_answer_error",
        "final_answer_start",
        "final_answer_delta",
        "final_answer_end",
    ]
    assert outbound.events[2]["error"] == "invalid tool protocol, retrying"


@pytest.mark.asyncio
async def test_react_reuses_resolved_route_for_tool_protocol_retry() -> None:
    downstream = StreamingInvalidToolProtocolFinalAnswerLLM(
        partial_final_before_work_tool=True
    )
    selected_models: list[str] = []
    router = RouterLLM(downstream_resolver=lambda _model_id: downstream)
    router.context_window = 1_048_576

    async def select_model(_prompt: str) -> str:
        selected = f"test/model-{len(selected_models) + 1}"
        selected_models.append(selected)
        return selected

    router._select_model = select_model  # type: ignore[assignment]
    pattern = ReActPattern(max_iterations=3, finalize_after_tool_result=True)
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="task-1")
    context.add_user_message("Find an audio clip")
    tracer = TraceEventRecorder()
    runtime = PatternRuntime(execution_id="task-1", tracer=tracer)

    result = await pattern.run(
        context=context,
        tools=[FakeTool()],
        llm=router,
        runtime=runtime,
    )

    assert result["success"] is True
    assert selected_models == ["test/model-1", "test/model-2"]
    retry_starts = [
        event
        for event in tracer.events
        if event["event_type"] == "action_start_llm"
        and event["data"].get("phase") == "unavailable_tool_call_recovery"
    ]
    assert len(retry_starts) == 1
    assert retry_starts[0]["data"]["selected_model"] == "test/model-2"
    assert retry_starts[0]["data"]["context_window"] == 1_048_576


@pytest.mark.asyncio
async def test_react_closes_invalid_retry_final_answer_stream_before_failure() -> None:
    llm = StreamingInvalidToolProtocolFinalAnswerLLM(
        structured_work_tool=True,
        invalid_retry=True,
    )
    pattern = ReActPattern(max_iterations=3, finalize_after_tool_result=True)
    tool = FakeTool()
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="task-1")
    context.add_user_message("Find an audio clip")
    outbound = OutboundCollector()
    tracer = TraceEventRecorder()
    runtime = PatternRuntime(
        execution_id="task-1",
        tracer=tracer,
        outbound_message_handler=outbound,
    )

    result = await pattern.run(
        context=context,
        tools=[tool],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is False
    assert result["status"] == "invalid_tool_protocol"
    assert tool.calls == [{"expression": "2+2"}]
    assert [event["type"] for event in outbound.events] == [
        "final_answer_start",
        "final_answer_delta",
        "final_answer_error",
    ]
    assert outbound.events[2]["error"] == "invalid tool protocol after retry"
    invalid_end_events = [
        event
        for event in tracer.events
        if event["event_type"] == "action_end_llm"
        and event["data"].get("success") is False
    ]
    assert [event["data"]["phase"] for event in invalid_end_events] == [
        "discarded_invalid_tool_protocol",
        "discarded_invalid_tool_protocol_retry",
    ]


@pytest.mark.asyncio
async def test_react_pattern_does_not_stream_plain_text_when_tool_protocol_is_ignored() -> (
    None
):
    llm = StreamingPlainTextFinalAnswerLLM()
    pattern = ReActPattern(max_iterations=1)
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="task-1")
    context.add_user_message("Answer directly")
    outbound = OutboundCollector()
    runtime = PatternRuntime(execution_id="task-1", outbound_message_handler=outbound)

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert result["success"] is True
    assert result["response"] == "Plain final."
    assert llm.stream_calls[0]["tools"] is not None
    assert llm.stream_calls[0]["tool_choice"] == "required"
    assert outbound.events == []


@pytest.mark.asyncio
async def test_react_pattern_does_not_stream_tool_call_preamble() -> None:
    llm = StreamingPreambleToolCallLLM()
    pattern = ReActPattern(max_iterations=1)
    tool = FakeTool()
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="task-1")
    context.add_user_message("Calculate 2+2")
    outbound = OutboundCollector()
    tracer = TraceEventRecorder()
    runtime = PatternRuntime(
        execution_id="task-1",
        tracer=tracer,
        outbound_message_handler=outbound,
    )

    result = await pattern.run(context=context, tools=[tool], llm=llm, runtime=runtime)

    assert result["success"] is False
    assert tool.calls == [{"expression": "2+2"}]
    assert llm.stream_calls[0]["tools"][0]["function"]["name"] == "calculator"
    assert outbound.events == []
    tool_start_event = next(
        event for event in tracer.events if event["event_type"] == "action_start_tool"
    )
    assert tool_start_event["data"]["assistant_content"] == ("I will use a tool first.")


@pytest.mark.asyncio
async def test_react_pattern_streams_final_answer_control_tool() -> None:
    llm = StreamingFinalAnswerToolLLM()
    pattern = ReActPattern(max_iterations=1)
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="task-1")
    context.add_user_message("Answer directly")
    outbound = OutboundCollector()
    runtime = PatternRuntime(execution_id="task-1", outbound_message_handler=outbound)

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert result["success"] is True
    assert result["response"] == "Hi there."
    assert [event["type"] for event in outbound.events] == [
        "final_answer_start",
        "final_answer_delta",
        "final_answer_delta",
        "final_answer_delta",
        "final_answer_end",
    ]
    assert [event["delta"] for event in outbound.events[1:-1]] == [
        "Hi",
        " there",
        ".",
    ]
    assert outbound.events[-1]["content"] == "Hi there."


@pytest.mark.asyncio
async def test_react_strips_final_answer_bundled_before_work_tool() -> None:
    """I-2: a final_answer bundled before a work tool is stripped too - the
    work tool is no longer silently discarded, and the candidate answer text
    is never streamed to the frontend."""

    llm = StreamingMixedFinalAnswerAndToolLLM()
    pattern = ReActPattern(max_iterations=3)
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="task-1")
    context.add_user_message("Calculate 2+2")
    outbound = OutboundCollector()
    runtime = PatternRuntime(execution_id="task-1", outbound_message_handler=outbound)
    tool = FakeTool()

    result = await pattern.run(
        context=context,
        tools=[tool],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert result["response"] == "4"
    assert tool.calls == [{"expression": "2+2"}]
    assert not any(
        event.get("content") == "Candidate" or event.get("delta") == "Candidate"
        for event in outbound.events
    )


@pytest.mark.asyncio
async def test_react_strips_final_answer_bundled_after_work_tool(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """I-1/I-3/I-5: a final_answer bundled after a work tool is stripped, the
    work tool executes and its result reaches the next turn, and the strip is
    logged once with the model name and the pre-strip tool list, bounded and
    escaped."""

    llm = FakeNamedLLM(
        responses=[
            {
                "tool_calls": [
                    {
                        "id": "call_calc",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                        },
                    },
                    {
                        "id": "call_final",
                        "function": {
                            "name": "final_answer",
                            "arguments": '{"answer":"Looking that up now."}',
                        },
                    },
                ],
            },
            {
                "tool_calls": [
                    {
                        "id": "call_final_2",
                        "function": {
                            "name": "final_answer",
                            "arguments": '{"answer":"4"}',
                        },
                    }
                ],
            },
        ]
    )
    tool = FakeTool()
    pattern = ReActPattern(max_iterations=3)
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="task-1")
    context.add_user_message("Calculate 2+2")
    runtime = PatternRuntime(execution_id="task-1")

    with caplog.at_level(
        logging.WARNING, logger="xagent.core.agent.pattern.react.react"
    ):
        result = await pattern.run(
            context=context,
            tools=[tool],
            llm=llm,
            runtime=runtime,
        )

    assert result["success"] is True
    assert result["response"] == "4"
    assert tool.calls == [{"expression": "2+2"}]

    # I-3: the stripped final_answer leaves no trace in the assistant
    # message, the tool-result messages, or the ledger.
    assistant_messages = context.get_messages_by_role("assistant")
    first_turn_tool_names = [
        call["function"]["name"] for call in (assistant_messages[0].tool_calls or [])
    ]
    assert first_turn_tool_names == ["calculator"]
    tool_result_ids = {
        message.tool_call_id for message in context.get_messages_by_role("tool")
    }
    assert "call_final" not in tool_result_ids
    assert "call_final" not in pattern.tool_ledger

    # Regression-only guard, not a mutation-effective assertion on its own:
    # no orphaned tool_call may ever appear in history.
    assistant_tool_call_ids = {
        call["id"]
        for message in assistant_messages
        for call in (message.tool_calls or [])
    }
    assert assistant_tool_call_ids == tool_result_ids

    all_content = json.dumps(
        [message.content for message in context.messages], default=str
    )
    assert "Looking that up now." not in all_content

    assert len(llm.calls) == 2
    assert llm.calls[1]["messages"][-1]["role"] == "tool"

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert "final_answer" in messages[0]
    assert "calculator" in messages[0]
    assert "fake-model" in messages[0]

    # I-5 (bound): the tool-name list in the warning is capped, escaped, and
    # never exposes an unbounded or unescaped model-controlled string.
    caplog.clear()
    long_name = "x" * 200 + "\n" + "y"
    overflow_tool_calls = [
        {"id": "call_long", "function": {"name": long_name, "arguments": "{}"}},
        *(
            {
                "id": f"call_work_{index}",
                "function": {"name": f"work_tool_{index}", "arguments": "{}"},
            }
            for index in range(10)
        ),
        {
            "id": "call_overflow_final",
            "function": {
                "name": "final_answer",
                "arguments": '{"answer":"Looking that up now."}',
            },
        },
    ]
    overflow_llm = FakeNamedLLM(responses=[{"tool_calls": overflow_tool_calls}])
    overflow_pattern = ReActPattern(max_iterations=1)
    overflow_context = ExecutionContext(
        system_prompt="You are helpful.", execution_id="task-2"
    )
    overflow_context.add_user_message("Run many tools")
    overflow_runtime = PatternRuntime(execution_id="task-2")

    with caplog.at_level(
        logging.WARNING, logger="xagent.core.agent.pattern.react.react"
    ):
        await overflow_pattern.run(
            context=overflow_context,
            tools=[],
            llm=overflow_llm,
            runtime=overflow_runtime,
        )

    overflow_messages = [record.getMessage() for record in caplog.records]
    assert len(overflow_messages) == 1
    overflow_message = overflow_messages[0]
    assert "(+4 more)" in overflow_message
    assert "\n" not in overflow_message
    assert ("x" * 64) in overflow_message
    assert ("x" * 65) not in overflow_message


def test_tool_names_for_log_tolerates_non_mapping_entries() -> None:
    """Matches the isinstance(dict) defense in _batch_carries_work_tool and
    _strip_final_answer_bundled_with_work_tools: a non-mapping batch entry
    must render a placeholder instead of crashing the log line it appears
    in."""

    pattern = ReActPattern()

    rendered = pattern._tool_names_for_log(
        [
            {"id": "call_1", "name": "calculator"},
            "not-a-tool-call",
            None,
        ]
    )

    assert "calculator" in rendered
    assert "non-mapping tool_call: str" in rendered
    assert "non-mapping tool_call: NoneType" in rendered


@pytest.mark.asyncio
async def test_react_strips_every_final_answer_in_a_mixed_batch() -> None:
    """I-4: every final_answer in a batch is stripped, not only the first."""

    llm = FakeLLM(
        responses=[
            {
                "tool_calls": [
                    {
                        "id": "call_final_a",
                        "function": {
                            "name": "final_answer",
                            "arguments": '{"answer":"A"}',
                        },
                    },
                    {
                        "id": "call_calc",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                        },
                    },
                    {
                        "id": "call_final_b",
                        "function": {
                            "name": "final_answer",
                            "arguments": '{"answer":"B"}',
                        },
                    },
                ],
            },
            {
                "tool_calls": [
                    {
                        "id": "call_final_2",
                        "function": {
                            "name": "final_answer",
                            "arguments": '{"answer":"4"}',
                        },
                    }
                ],
            },
        ]
    )
    tool = FakeTool()
    pattern = ReActPattern(max_iterations=3)
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="task-1")
    context.add_user_message("Calculate 2+2")
    runtime = PatternRuntime(execution_id="task-1")

    result = await pattern.run(
        context=context,
        tools=[tool],
        llm=llm,
        runtime=runtime,
    )

    assert result["response"] not in {"A", "B"}
    assert result["response"] == "4"
    assistant_messages = context.get_messages_by_role("assistant")
    first_turn_tool_calls = assistant_messages[0].tool_calls or []
    assert len(first_turn_tool_calls) == 1
    assert first_turn_tool_calls[0]["function"]["name"] == "calculator"


@pytest.mark.asyncio
async def test_react_closes_open_answer_stream_when_bundled_final_answer_is_stripped() -> (
    None
):
    """I-6: an answer stream already open when the batch turns out to bundle
    a work tool is closed explicitly instead of left open forever, and it is
    never closed as though the candidate text were the real final answer."""

    llm = StreamingAnswerThenWorkToolLLM()
    pattern = ReActPattern(max_iterations=3)
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="task-1")
    context.add_user_message("Calculate 2+2")
    outbound = OutboundCollector()
    runtime = PatternRuntime(execution_id="task-1", outbound_message_handler=outbound)
    tool = FakeTool()

    result = await pattern.run(
        context=context,
        tools=[tool],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert result["response"] == "4"
    assert tool.calls == [{"expression": "2+2"}]

    started_id = next(
        event["message_id"]
        for event in outbound.events
        if event["type"] == "final_answer_start"
    )
    first_stream_events = []
    for event in outbound.events:
        if event.get("message_id") != started_id:
            continue
        first_stream_events.append(event)
        if event["type"] == "final_answer_error":
            break

    assert [event["type"] for event in first_stream_events] == [
        "final_answer_start",
        "final_answer_delta",
        "final_answer_delta",
        "final_answer_error",
    ]
    assert not any(
        event["type"] == "final_answer_end"
        and event.get("content") == "Looking that up now."
        for event in outbound.events
    )
    assert outbound.events[-1]["type"] == "final_answer_end"
    assert outbound.events[-1]["content"] == "4"


@pytest.mark.asyncio
async def test_react_keeps_send_message_bundled_with_work_tool(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Guards against a future change that widens the strip from final_answer
    to every control tool. This test stays green if the strip is removed
    entirely - it pins behavior that must not change, not behavior this fix
    introduces."""

    llm = FakeLLM(
        responses=[
            {
                "tool_calls": [
                    {
                        "id": "call_message",
                        "function": {
                            "name": "send_message",
                            "arguments": (
                                '{"message":"Still working",'
                                '"message_type":"progress","expect_response":false}'
                            ),
                        },
                    },
                    {
                        "id": "call_calc",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                        },
                    },
                ],
            },
            {
                "tool_calls": [
                    {
                        "id": "call_final",
                        "function": {
                            "name": "final_answer",
                            "arguments": '{"answer":"4"}',
                        },
                    }
                ],
            },
        ]
    )
    tool = FakeTool()
    pattern = ReActPattern(max_iterations=3)
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="task-1")
    context.add_user_message("Calculate 2+2")
    runtime = PatternRuntime(execution_id="task-1")

    with caplog.at_level(
        logging.WARNING, logger="xagent.core.agent.pattern.react.react"
    ):
        result = await pattern.run(
            context=context,
            tools=[tool],
            llm=llm,
            runtime=runtime,
        )

    assert result["success"] is True
    assert result["response"] == "4"
    assert tool.calls == [{"expression": "2+2"}]
    assert len(runtime.outbound_messages) == 1
    assert runtime.outbound_messages[0]["message"] == "Still working"
    assert caplog.records == []


@pytest.mark.asyncio
async def test_react_keeps_control_only_final_answer_batch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Same guard shape as the send_message case: a batch of control tools
    only has no result the answer could be missing, so the strip must not
    touch it. Green with or without the strip."""

    llm = FakeLLM(
        responses=[
            {
                "tool_calls": [
                    {
                        "id": "call_message",
                        "function": {
                            "name": "send_message",
                            "arguments": (
                                '{"message":"Still working",'
                                '"message_type":"progress","expect_response":false}'
                            ),
                        },
                    },
                    {
                        "id": "call_final",
                        "function": {
                            "name": "final_answer",
                            "arguments": '{"answer":"Done."}',
                        },
                    },
                ],
            },
        ]
    )
    pattern = ReActPattern(max_iterations=1)
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="task-1")
    context.add_user_message("Do the thing")
    runtime = PatternRuntime(execution_id="task-1")

    with caplog.at_level(
        logging.WARNING, logger="xagent.core.agent.pattern.react.react"
    ):
        result = await pattern.run(
            context=context,
            tools=[],
            llm=llm,
            runtime=runtime,
        )

    assert result["success"] is True
    assert result["response"] == "Done."
    assert caplog.records == []


@pytest.mark.asyncio
async def test_react_forced_final_answer_recovers_full_tool_set_after_mixed_batch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """I-9a: guards the forced-final-answer turn's existing recovery path.
    This test stays green if the strip is removed entirely - it pins
    behavior that must not change, not behavior this fix introduces."""

    llm = FakeLLM(
        responses=[
            {
                "tool_calls": [
                    {
                        "id": "call_calc_1",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                        },
                    },
                    {
                        "id": "call_final_1",
                        "function": {
                            "name": "final_answer",
                            "arguments": '{"answer":"Looking that up now."}',
                        },
                    },
                ],
            },
            {
                "tool_calls": [
                    {
                        "id": "call_calc_2",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                        },
                    }
                ],
            },
            {
                "tool_calls": [
                    {
                        "id": "call_final_2",
                        "function": {
                            "name": "final_answer",
                            "arguments": '{"answer":"4"}',
                        },
                    }
                ],
            },
        ]
    )
    tool = FakeTool()
    pattern = ReActPattern(max_iterations=3)
    pattern.force_final_answer_next = True
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="task-1")
    context.add_user_message("Calculate 2+2")
    runtime = PatternRuntime(execution_id="task-1")

    with caplog.at_level(
        logging.WARNING, logger="xagent.core.agent.pattern.react.react"
    ):
        result = await pattern.run(
            context=context,
            tools=[tool],
            llm=llm,
            runtime=runtime,
        )

    assert result["success"] is True
    assert result["response"] == "4"
    second_call_tool_names = {
        schema["function"]["name"] for schema in llm.calls[1]["tools"]
    }
    assert "calculator" in second_call_tool_names
    assert pattern.force_final_answer_next is False
    assert tool.calls == [{"expression": "2+2"}]
    assert caplog.records == []


@pytest.mark.asyncio
async def test_react_forced_final_answer_fails_when_recovery_retry_bundles_again(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """I-9b: anchor test - pins the strip's position, not its existence.
    Deleting the strip helper entirely leaves this green. Moving the strip
    call to run ahead of either tool-protocol-retry guard check instead of
    after both turns it red: on the first guard, because the strip fires and
    logs before that response is discarded wholesale for an unrelated reason,
    leaving a stray "discarding" warning behind; on the second (retry) guard,
    because stripping the retried batch's final_answer erases the very
    mixed-call shape that guard exists to reject, so the run no longer fails
    at all - it runs the retried calculator and needs a third response the
    fixture never provides."""

    llm = FakeLLM(
        responses=[
            {
                "tool_calls": [
                    {
                        "id": "call_calc_1",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                        },
                    },
                    {
                        "id": "call_final_1",
                        "function": {
                            "name": "final_answer",
                            "arguments": '{"answer":"Looking that up now."}',
                        },
                    },
                ],
            },
            {
                "tool_calls": [
                    {
                        "id": "call_calc_2",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                        },
                    },
                    {
                        "id": "call_final_2",
                        "function": {
                            "name": "final_answer",
                            "arguments": '{"answer":"Looking that up now."}',
                        },
                    },
                ],
            },
        ]
    )
    tool = FakeTool()
    pattern = ReActPattern(max_iterations=3)
    pattern.force_final_answer_next = True
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="task-1")
    context.add_user_message("Calculate 2+2")
    runtime = PatternRuntime(execution_id="task-1")

    with caplog.at_level(
        logging.WARNING, logger="xagent.core.agent.pattern.react.react"
    ):
        result = await pattern.run(
            context=context,
            tools=[tool],
            llm=llm,
            runtime=runtime,
        )

    assert result["success"] is False
    assert result["status"] == "invalid_tool_protocol"
    assert tool.calls == []
    # The run fails through the existing invalid-protocol path, which logs
    # its own unrelated warning; only the strip's warning must be absent.
    assert not any("discarding" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_react_strips_empty_final_answer_from_a_retried_batch_when_not_reforcing() -> (
    None
):
    """Guards the retry recheck's accepted behavior change: a retried batch
    that is neither a forced turn nor itself rejecting mixed control calls
    (reject_mixed_control_calls=False from a provider protocol error on the
    first response, force_final_answer=False because this was never a forced
    turn) now lets a bundled empty final_answer strip and the work tool run,
    instead of hard-failing as "invalid tool protocol after retry". The
    first response's provider-level protocol error is what puts the run on
    the retry path at all; the retried response is the one carrying the
    mixed batch this test is pinning."""

    llm = FakeLLM(
        responses=[
            tool_protocol_error_response(
                ToolProtocolViolation(
                    provider="deepseek",
                    code="serialized_tool_call_content",
                    message="Invalid provider tool protocol.",
                )
            ),
            {
                "tool_calls": [
                    {
                        "id": "call_work",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                        },
                    },
                    {
                        "id": "call_final",
                        "function": {
                            "name": "final_answer",
                            "arguments": '{"answer":""}',
                        },
                    },
                ],
            },
            {
                "tool_calls": [
                    {
                        "id": "call_final_2",
                        "function": {
                            "name": "final_answer",
                            "arguments": '{"answer":"4"}',
                        },
                    }
                ],
            },
        ]
    )
    tool = FakeTool()
    pattern = ReActPattern(max_iterations=3)
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="task-1")
    context.add_user_message("Calculate 2+2")
    runtime = PatternRuntime(execution_id="task-1")

    result = await pattern.run(
        context=context,
        tools=[tool],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert result["response"] == "4"
    assert tool.calls == [{"expression": "2+2"}]
    assert len(llm.calls) == 3


@pytest.mark.asyncio
async def test_react_strips_empty_final_answer_bundled_with_work_tool(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """I-11: an empty-answer final_answer bundled with a work tool is
    stripped like any other bundled final_answer, instead of discarding the
    whole response the way a control-only empty answer does."""

    llm = FakeLLM(
        responses=[
            {
                "tool_calls": [
                    {
                        "id": "call_calc",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                        },
                    },
                    {
                        "id": "call_final",
                        "function": {
                            "name": "final_answer",
                            "arguments": '{"answer":""}',
                        },
                    },
                ],
            },
            {
                "tool_calls": [
                    {
                        "id": "call_final_2",
                        "function": {
                            "name": "final_answer",
                            "arguments": '{"answer":"4"}',
                        },
                    }
                ],
            },
        ]
    )
    tool = FakeTool()
    pattern = ReActPattern(max_iterations=3)
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="task-1")
    context.add_user_message("Calculate 2+2")
    runtime = PatternRuntime(execution_id="task-1")

    with caplog.at_level(
        logging.WARNING, logger="xagent.core.agent.pattern.react.react"
    ):
        result = await pattern.run(
            context=context,
            tools=[tool],
            llm=llm,
            runtime=runtime,
        )

    assert result["success"] is True
    assert result["response"] == "4"
    assert tool.calls == [{"expression": "2+2"}]
    assert len(llm.calls) == 2

    messages = [record.getMessage() for record in caplog.records]
    assert not any("carried no answer text" in message for message in messages)
    assert any(
        "final_answer" in message and "calculator" in message for message in messages
    )


@pytest.mark.asyncio
async def test_react_pauses_for_user_after_stripping_bundled_final_answer() -> None:
    """I-12: a batch that turns out to need user input after the bundled
    final_answer is stripped still pauses cleanly, with the unexecuted work
    tool cancelled rather than silently dropped."""

    llm = FakeLLM(
        responses=[
            {
                "tool_calls": [
                    {
                        "id": "call_final",
                        "function": {
                            "name": "final_answer",
                            "arguments": '{"answer":"Done."}',
                        },
                    },
                    {
                        "id": "call_ask",
                        "function": {
                            "name": "ask_user_question",
                            "arguments": '{"message":"Which one?"}',
                        },
                    },
                    {
                        "id": "call_calc",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                        },
                    },
                ],
            },
        ]
    )
    tool = FakeTool()
    pattern = ReActPattern(max_iterations=1)
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="task-1")
    context.add_user_message("Calculate 2+2 then ask me something")
    runtime = PatternRuntime(execution_id="task-1")

    result = await pattern.run(
        context=context,
        tools=[tool],
        llm=llm,
        runtime=runtime,
    )

    assert result["status"] == "waiting_for_user"
    assert tool.calls == []
    assert "call_final" not in pattern.tool_ledger
    assert pattern.tool_ledger["call_calc"].status == "cancelled"
    assert pattern.tool_ledger["call_ask"].status == "completed"

    assistant_messages = context.get_messages_by_role("assistant")
    assistant_tool_call_ids = {
        call["id"]
        for message in assistant_messages
        for call in (message.tool_calls or [])
    }
    tool_result_ids = {
        message.tool_call_id for message in context.get_messages_by_role("tool")
    }
    assert assistant_tool_call_ids == {"call_ask", "call_calc"}
    assert assistant_tool_call_ids <= tool_result_ids


@pytest.mark.asyncio
async def test_react_passes_runtime_step_to_browser_tool_call() -> None:
    pattern = ReActPattern()
    runtime = PatternRuntime()
    runtime.active_react_step_id = "render_english_poster"
    tool = FakeBrowserNavigateTool()

    result = await pattern._execute_tool_safely(
        {
            "id": "call-browser",
            "name": "browser_navigate",
            "args": {"url": "poster_en.html"},
        },
        [tool],
        runtime,
    )

    assert result["success"] is True
    assert tool.calls == [
        {
            "url": "poster_en.html",
            "_xagent_step_id": "render_english_poster",
        }
    ]


@pytest.mark.asyncio
async def test_react_passes_runtime_step_to_computer_tool_call() -> None:
    class FakeComputerTool:
        name = "computer"

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def run_json_async(self, args: dict[str, Any]) -> dict[str, Any]:
            self.calls.append(args)
            return {"success": True}

    pattern = ReActPattern()
    runtime = PatternRuntime()
    runtime.active_react_step_id = "inspect_browser"
    tool = FakeComputerTool()

    result = await pattern._execute_tool_safely(
        {
            "id": "call-computer",
            "name": "computer",
            "args": {"actions": [{"type": "screenshot"}]},
        },
        [tool],
        runtime,
    )

    assert result["success"] is True
    assert tool.calls == [
        {
            "actions": [{"type": "screenshot"}],
            "_xagent_step_id": "inspect_browser",
        }
    ]


@pytest.mark.asyncio
async def test_react_interrupt_cancels_in_flight_tool() -> None:
    class SlowVisionTool:
        name = "understand_images"
        description = "Analyze an image."

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def ainvoke(self, _args: dict[str, Any]) -> Any:
            self.started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            return {"success": True, "answer": "never"}

    pattern = ReActPattern()
    runtime = PatternRuntime()
    tool = SlowVisionTool()
    tool_call = {
        "id": "call-vision",
        "name": "understand_images",
        "args": {"images": "file-id", "question": "What is shown?"},
    }
    task = asyncio.create_task(pattern._execute_tool_safely(tool_call, [tool], runtime))
    await tool.started.wait()

    runtime.request_interrupt("paused by websocket")

    with pytest.raises(ToolCallInterrupted, match="paused by websocket"):
        await task
    assert tool.cancelled.is_set()
    assert pattern.tool_ledger["call-vision"].status == "interrupted"
    assert pattern.tool_ledger["call-vision"].error == "paused by websocket"


@pytest.mark.asyncio
async def test_react_tool_interrupt_checkpoint_uses_tool_safe_point() -> None:
    class SlowVisionTool:
        name = "understand_images"
        description = "Analyze an image."

        def __init__(self) -> None:
            self.started = asyncio.Event()

        async def ainvoke(self, _args: dict[str, Any]) -> Any:
            self.started.set()
            await asyncio.sleep(60)
            return {"success": True, "answer": "never"}

    pattern = ReActPattern(max_iterations=2)
    runtime = PatternRuntime(execution_id="react-tool-safe-point")
    context = ExecutionContext(execution_id="react-tool-safe-point")
    context.add_user_message("Inspect this image.")
    tool = SlowVisionTool()
    llm = FakeLLM(
        [
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "vision-1",
                        "function": {
                            "name": "understand_images",
                            "arguments": json.dumps(
                                {
                                    "images": "file-id",
                                    "question": "What is shown?",
                                }
                            ),
                        },
                    }
                ],
            }
        ]
    )

    task = asyncio.create_task(
        pattern.run(context=context, tools=[tool], llm=llm, runtime=runtime)
    )
    await tool.started.wait()
    runtime.request_interrupt("paused by websocket")
    result = await task

    assert result["status"] == "interrupted"
    checkpoint = next(
        checkpoint
        for checkpoint in reversed(runtime.checkpoints)
        if checkpoint["label"] == "interrupted"
    )
    assert checkpoint["metadata"] == {
        "safe_point": "during_tool",
        "reason": "paused by websocket",
    }


@pytest.mark.asyncio
async def test_react_sanitizes_tool_args_before_trace_and_execution() -> None:
    class CapturingRuntime(PatternRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.started_args: dict[str, Any] | None = None

        async def on_tool_start(self, *, tool_call: dict[str, Any]) -> None:
            self.started_args = dict(tool_call.get("args") or {})

    pattern = ReActPattern()
    runtime = CapturingRuntime()
    tool = FakeTraceSanitizingTool()

    result = await pattern._execute_tool_safely(
        {
            "id": "call-custom-api",
            "name": "custom_api",
            "args": {
                "headers": {"Authorization": "Bearer caller-token"},
                "params": {"q": "client"},
            },
        },
        [tool],
        runtime,
    )

    assert result["success"] is True
    assert runtime.started_args == {"params": {"q": "client"}}
    assert tool.calls == [{"params": {"q": "client"}}]
    record = pattern.tool_ledger["call-custom-api"]
    assert record.args == {"params": {"q": "client"}}
    assert "caller-token" not in repr(record.to_dict())


@pytest.mark.asyncio
async def test_react_trace_safe_tool_args_handles_null_args() -> None:
    pattern = ReActPattern()
    runtime = PatternRuntime()
    tool = FakeTraceSanitizingTool()

    result = await pattern._execute_tool_safely(
        {
            "id": "call-custom-api",
            "name": "custom_api",
            "args": None,
        },
        [tool],
        runtime,
    )

    assert result["success"] is True
    assert tool.calls == [{}]


def test_react_trace_safe_tool_args_ignores_non_mapping_args() -> None:
    pattern = ReActPattern()
    tool = FakeTraceSanitizingTool()
    tool_call = {
        "id": "call-custom-api",
        "name": "custom_api",
        "args": ["not", "a", "mapping"],
    }

    result = pattern._with_trace_safe_tool_args(tool_call, [tool])

    assert result is tool_call


@pytest.mark.parametrize("raw_args", [["not", "a", "mapping"], "not-a-mapping"])
@pytest.mark.asyncio
async def test_react_rejects_non_mapping_tool_args_without_crashing_ledger(
    raw_args: Any,
) -> None:
    pattern = ReActPattern()
    runtime = PatternRuntime()
    tool = FakeTraceSanitizingTool()

    result = await pattern._execute_tool_safely(
        {
            "id": "call-custom-api",
            "name": "custom_api",
            "args": raw_args,
        },
        [tool],
        runtime,
    )

    assert result["success"] is False
    assert result["error"] == "Tool call args must be a JSON object."
    assert tool.calls == []
    record = pattern.tool_ledger["call-custom-api"]
    assert record.status == "failed"
    assert record.args == {}
    assert record.error == "Tool call args must be a JSON object."


def test_react_trace_safe_tool_args_detects_in_place_sanitizer() -> None:
    pattern = ReActPattern()
    tool = FakeInPlaceTraceSanitizingTool()
    tool_call = {
        "id": "call-custom-api",
        "name": "custom_api",
        "args": {
            "headers": {"Authorization": "Bearer caller-token"},
            "body": {"secret": "body-secret", "safe": "value"},
        },
    }

    result = pattern._with_trace_safe_tool_args(tool_call, [tool])

    assert result is not tool_call
    assert result["args"] == {"body": {"safe": "value"}}
    assert tool_call["args"] == {
        "headers": {"Authorization": "Bearer caller-token"},
        "body": {"secret": "body-secret", "safe": "value"},
    }
    assert "caller-token" not in repr(result)
    assert "body-secret" not in repr(result)


@pytest.mark.asyncio
async def test_react_pattern_unwraps_textual_final_answer_json() -> None:
    llm = FakeLLM(
        responses=[
            '```json\n{"action":"final_answer","action_input":"Done cleanly."}\n```'
        ]
    )
    pattern = ReActPattern(max_iterations=1)
    context = ExecutionContext()
    context.add_user_message("Finish")

    result = await pattern.run(context=context, tools=[], llm=llm)

    assert result["success"] is True
    assert result["response"] == "Done cleanly."
    assert context.messages[-1].content == "Done cleanly."
    assert "action_input" not in context.messages[-1].content


@pytest.mark.asyncio
async def test_react_pattern_finalizes_with_completion_evidence() -> None:
    llm = FakeLLM(
        responses=[
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_write",
                        "function": {
                            "name": "write_file",
                            "arguments": (
                                '{"file_path":"en_poster.html","content":"<html></html>"}'
                            ),
                        },
                    }
                ],
                "done": False,
            },
            {"content": "Created output/en_poster.html."},
        ]
    )
    pattern = ReActPattern(max_iterations=3, finalize_after_tool_result=True)
    tool = FakeWriteFileTool()
    context = ExecutionContext(system_prompt="You are helpful.")
    context.add_user_message(
        "Create the English poster HTML.",
        metadata={
            "dag_completion_evidence": (
                "The writer returned success=true for the requested output path."
            )
        },
    )

    result = await pattern.run(context=context, tools=[tool], llm=llm)

    assert result["success"] is True
    assert result["response"] == "Created output/en_poster.html."
    assert tool.calls == [{"file_path": "en_poster.html", "content": "<html></html>"}]
    assert [schema["function"]["name"] for schema in llm.calls[1]["tools"]] == [
        "final_answer"
    ]
    assert llm.calls[1]["tool_choice"] == "required"
    assert (
        "Step completion evidence: The writer returned success=true"
        in (llm.calls[1]["messages"][0]["content"])
    )
    assert "Do not repeat the same work" in llm.calls[1]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_react_pattern_uses_decision_for_repeated_tools() -> None:
    llm = FakeLLM(
        responses=[
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "search_1",
                        "function": {
                            "name": "zhipu_web_search",
                            "arguments": '{"query":"AI news May 2026","count":10}',
                        },
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "search_2",
                        "function": {
                            "name": "zhipu_web_search",
                            "arguments": '{"query":"AI news May 2026","count":5}',
                        },
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "decision_1",
                        "function": {
                            "name": "react_decision",
                            "arguments": (
                                '{"action":"final_answer",'
                                '"reason":"已有结果足够回答。"}'
                            ),
                        },
                    }
                ],
            },
            "可以基于已有搜索结果回答。",
        ]
    )
    pattern = ReActPattern(
        max_iterations=3,
        repeated_tool_decision_after_consecutive_tool_calls=2,
    )
    tool = FakeSearchTool()
    context = ExecutionContext()
    context.add_user_message("最近 AI 新闻")

    result = await pattern.run(context=context, tools=[tool], llm=llm)

    assert result["success"] is True
    assert result["response"] == "可以基于已有搜索结果回答。"
    assert len(llm.calls) == 4
    assert len(tool.calls) == 2
    tool_names = [schema["function"]["name"] for schema in llm.calls[2]["tools"]]
    assert tool_names == ["react_decision"]
    decision_prompt = llm.calls[2]["messages"][-1]["content"]
    assert "Latest user request text" in decision_prompt
    assert "最近 AI 新闻" in decision_prompt
    assert "Do not put user-facing final answer text in this decision" in (
        decision_prompt
    )
    assert "private observations are evidence for reasoning" in decision_prompt
    assert "requested computer screenshot" in decision_prompt
    assert "artifact, file_ref, or markdown_link" in decision_prompt
    assert "future tool action" in decision_prompt
    assert "response_language" not in decision_prompt
    decision_schema = llm.calls[2]["tools"][0]["function"]["parameters"]
    assert "response_language" not in decision_schema["properties"]
    assert "response_language" not in decision_schema["required"]
    assert [schema["function"]["name"] for schema in llm.calls[3]["tools"]] == [
        "final_answer"
    ]
    assert pattern.pending_tool_calls == []


def test_react_repeated_decision_anchors_latest_user_text_not_cached_task() -> None:
    pattern = ReActPattern()
    pattern.task_text = "search AI news"
    context = ExecutionContext()
    context.add_user_message("search AI news")
    tail = "TAIL_SHOULD_NOT_BE_IN_LANGUAGE_ANCHOR"
    context.add_user_message(f"请用简体中文回答 {'x' * 430}{tail}")

    messages = pattern._messages_for_repeated_tool_decision(
        context,
        {
            "tool_name": "zhipu_web_search",
            "latest_tool_name": "zhipu_web_search",
            "consecutive_tool_calls": 2,
        },
    )

    prompt = messages[-1]["content"]
    anchor_start = prompt.index("Latest user request text")
    anchor_end = prompt.index("Use this as the controlling request", anchor_start)
    anchor = prompt[anchor_start:anchor_end]
    assert "请用简体中文回答" in anchor
    assert "search AI news" not in anchor
    assert "... [truncated]" in anchor
    assert tail not in anchor
    assert "response_language" not in prompt


@pytest.mark.asyncio
async def test_react_repeated_decision_drains_current_tool_call_batch() -> None:
    llm = FakeLLM(
        responses=[
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "search_1",
                        "function": {
                            "name": "zhipu_web_search",
                            "arguments": '{"query":"AI news May 2026","count":10}',
                        },
                    },
                    {
                        "id": "search_2",
                        "function": {
                            "name": "zhipu_web_search",
                            "arguments": '{"query":"OpenAI news May 2026","count":5}',
                        },
                    },
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "decision_1",
                        "function": {
                            "name": "react_decision",
                            "arguments": (
                                '{"action":"final_answer",'
                                '"reason":"The current batch is enough."}'
                            ),
                        },
                    }
                ],
            },
            "Both pending searches were executed.",
        ]
    )
    pattern = ReActPattern(
        max_iterations=3,
        repeated_tool_decision_after_consecutive_tool_calls=1,
    )
    tool = FakeSearchTool()
    context = ExecutionContext()
    context.add_user_message("最近 AI 新闻")

    result = await pattern.run(context=context, tools=[tool], llm=llm)

    assert result["success"] is True
    assert result["response"] == "Both pending searches were executed."
    assert len(tool.calls) == 2
    assert len(llm.calls) == 3
    tool_result_ids = [
        message.tool_call_id
        for message in context.messages
        if message.role == "tool"
        and (message.metadata or {}).get("tool_name") == "zhipu_web_search"
    ]
    assert tool_result_ids[-2:] == ["search_1", "search_2"]
    tool_names = [schema["function"]["name"] for schema in llm.calls[1]["tools"]]
    assert tool_names == ["react_decision"]
    assert [schema["function"]["name"] for schema in llm.calls[2]["tools"]] == [
        "final_answer"
    ]
    assert pattern.pending_tool_calls == []


@pytest.mark.asyncio
async def test_react_pattern_uses_decision_after_cross_tool_attempts() -> None:
    llm = FakeLLM(
        responses=[
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "failed_1",
                        "function": {
                            "name": "failing_result",
                            "arguments": '{"input":"bad image ref"}',
                        },
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "search_1",
                        "function": {
                            "name": "zhipu_web_search",
                            "arguments": '{"query":"Vadim Nicolai xinference","count":5}',
                        },
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "browser_1",
                        "function": {
                            "name": "browser_navigate",
                            "arguments": '{"url":"https://example.com/profile"}',
                        },
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "decision_1",
                        "function": {
                            "name": "react_decision",
                            "arguments": (
                                '{"action":"final_answer",'
                                '"reason":"Enough cross-tool attempts."}'
                            ),
                        },
                    }
                ],
            },
            "已有跨工具结果，直接回答。",
        ]
    )
    pattern = ReActPattern(
        max_iterations=4,
        repeated_tool_decision_after_consecutive_tool_calls=None,
        repeated_tool_decision_after_consecutive_work_tool_calls=3,
    )
    search_tool = FakeSearchTool()
    browser_tool = FakeBrowserNavigateTool()
    context = ExecutionContext()
    context.add_user_message("判断这封邮件是不是广撒网")

    result = await pattern.run(
        context=context,
        tools=[FailingResultTool(), search_tool, browser_tool],
        llm=llm,
    )

    assert result["success"] is True
    assert result["response"] == "已有跨工具结果，直接回答。"
    assert len(llm.calls) == 5
    assert len(search_tool.calls) == 1
    assert len(browser_tool.calls) == 1
    tool_names = [schema["function"]["name"] for schema in llm.calls[3]["tools"]]
    assert tool_names == ["react_decision"]
    decision_prompt = llm.calls[3]["messages"][-1]["content"]
    assert "3 consecutive work-tool calls without a final answer" in decision_prompt
    assert "latest work tool was browser_navigate" in decision_prompt
    assert [schema["function"]["name"] for schema in llm.calls[4]["tools"]] == [
        "final_answer"
    ]


@pytest.mark.asyncio
async def test_react_pattern_uses_decision_after_tool_group_successes() -> None:
    llm = FakeLLM(
        responses=[
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "search_1",
                        "function": {
                            "name": "search_source",
                            "arguments": '{"query":"AirPods 4 charging port"}',
                        },
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "fetch_1",
                        "function": {
                            "name": "fetch_source",
                            "arguments": '{"query":"Apple AirPods 4 page"}',
                        },
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "search_2",
                        "function": {
                            "name": "search_source",
                            "arguments": '{"query":"AirPods 4 USB-C"}',
                        },
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "decision_1",
                        "function": {
                            "name": "react_decision",
                            "arguments": (
                                '{"action":"final_answer",'
                                '"reason":"Enough web research."}'
                            ),
                        },
                    }
                ],
            },
            "AirPods 4 uses USB-C.",
        ]
    )
    pattern = ReActPattern(
        max_iterations=4,
        repeated_tool_decision_after_consecutive_tool_calls=3,
        repeated_tool_decision_after_consecutive_work_tool_calls=10,
    )
    search_tool = FakeGroupedTool("search_source", "research")
    fetch_tool = FakeGroupedTool("fetch_source", "research")
    context = ExecutionContext()
    context.add_user_message("airpods 4 是什么接口")

    result = await pattern.run(
        context=context,
        tools=[search_tool, fetch_tool],
        llm=llm,
    )

    assert result["success"] is True
    assert result["response"] == "AirPods 4 uses USB-C."
    assert len(llm.calls) == 5
    assert len(search_tool.calls) == 2
    assert len(fetch_tool.calls) == 1
    tool_names = [schema["function"]["name"] for schema in llm.calls[3]["tools"]]
    assert tool_names == ["react_decision"]
    decision_prompt = llm.calls[3]["messages"][-1]["content"]
    assert (
        "3 consecutive successful calls in the research tool group" in decision_prompt
    )
    assert "latest tool was search_source" in decision_prompt
    assert [schema["function"]["name"] for schema in llm.calls[4]["tools"]] == [
        "final_answer"
    ]
    assert pattern.pending_tool_calls == []


@pytest.mark.asyncio
async def test_react_pattern_accepts_legacy_auto_reroute_kwarg() -> None:
    llm = FakeLLM(
        responses=[
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "search_1",
                        "function": {
                            "name": "zhipu_web_search",
                            "arguments": '{"query":"AI news May 2026","count":10}',
                        },
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "search_2",
                        "function": {
                            "name": "zhipu_web_search",
                            "arguments": '{"query":"AI news May 2026","count":5}',
                        },
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "decision_1",
                        "function": {
                            "name": "react_decision",
                            "arguments": (
                                '{"action":"final_answer",'
                                '"reason":"已有结果足够回答。"}'
                            ),
                        },
                    }
                ],
            },
            "可以基于已有搜索结果回答。",
        ]
    )
    pattern = ReActPattern(
        max_iterations=3,
        repeated_tool_decision_after_consecutive_tool_calls=2,
    )
    tool = FakeSearchTool()
    context = ExecutionContext()
    context.add_user_message("最近 AI 新闻")

    result = await pattern.run(
        context=context,
        tools=[tool],
        llm=llm,
        allow_auto_reroute=True,
    )

    assert result["success"] is True
    assert result["response"] == "可以基于已有搜索结果回答。"
    assert len(llm.calls) == 4
    assert len(tool.calls) == 2
    next_tool_names = [
        tool_schema["function"]["name"] for tool_schema in llm.calls[2]["tools"]
    ]
    assert next_tool_names == ["react_decision"]
    assert (
        "action must be final_answer or tool_call"
        in (llm.calls[2]["messages"][-1]["content"])
    )
    assert llm.calls[2]["tool_choice"] == "required"
    assert [schema["function"]["name"] for schema in llm.calls[3]["tools"]] == [
        "final_answer"
    ]
    assert pattern.repeated_tool_decision is None


@pytest.mark.asyncio
async def test_react_pattern_repeated_guard_can_switch_to_non_repeated_tool() -> None:
    llm = FakeLLM(
        responses=[
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "search_1",
                        "function": {
                            "name": "zhipu_web_search",
                            "arguments": '{"query":"AI news May 2026","count":10}',
                        },
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "search_2",
                        "function": {
                            "name": "zhipu_web_search",
                            "arguments": '{"query":"Google AI news May 2026","count":5}',
                        },
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "decision_1",
                        "function": {
                            "name": "react_decision",
                            "arguments": (
                                '{"action":"tool_call",'
                                '"reason":"Need to write the output file.",'
                                '"missing_verification":"write summarized search results"}'
                            ),
                        },
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "write_1",
                        "function": {
                            "name": "write_file",
                            "arguments": (
                                '{"file_path":"ai_news.md",'
                                '"content":"summarized search results"}'
                            ),
                        },
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "final_1",
                        "function": {
                            "name": "final_answer",
                            "arguments": (
                                '{"response_language":"Simplified Chinese",'
                                '"answer":"搜索后已写入文件。"}'
                            ),
                        },
                    }
                ],
            },
        ]
    )
    pattern = ReActPattern(
        max_iterations=5,
        repeated_tool_decision_after_consecutive_tool_calls=2,
    )
    search_tool = FakeSearchTool()
    write_tool = FakeWriteFileTool()
    context = ExecutionContext()
    context.add_user_message("最近 AI 新闻")

    result = await pattern.run(
        context=context,
        tools=[search_tool, write_tool],
        llm=llm,
    )

    assert result["success"] is True
    assert result["response"] == "搜索后已写入文件。"
    assert len(search_tool.calls) == 2
    assert len(write_tool.calls) == 1
    assert len(llm.calls) == 5
    decision_tool_names = [
        schema["function"]["name"] for schema in llm.calls[2]["tools"]
    ]
    assert decision_tool_names == ["react_decision"]
    normal_tool_names = [schema["function"]["name"] for schema in llm.calls[3]["tools"]]
    assert "zhipu_web_search" in normal_tool_names
    assert "write_file" in normal_tool_names
    assert "write summarized search results" in "\n".join(
        str(message.get("content") or "") for message in llm.calls[3]["messages"]
    )
    final_tool_names = [schema["function"]["name"] for schema in llm.calls[4]["tools"]]
    assert "final_answer" in final_tool_names
    assert "react_decision" not in final_tool_names


@pytest.mark.asyncio
async def test_react_pattern_repeated_guard_can_fire_twice_in_single_run() -> None:
    llm = FakeLLM(
        responses=[
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "search_1",
                        "function": {
                            "name": "zhipu_web_search",
                            "arguments": '{"query":"AI news May 2026","count":10}',
                        },
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "search_2",
                        "function": {
                            "name": "zhipu_web_search",
                            "arguments": '{"query":"Google AI news May 2026","count":5}',
                        },
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "decision_1",
                        "function": {
                            "name": "react_decision",
                            "arguments": (
                                '{"action":"tool_call",'
                                '"reason":"Need one more source.",'
                                '"missing_verification":"official source"}'
                            ),
                        },
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "search_3",
                        "function": {
                            "name": "zhipu_web_search",
                            "arguments": (
                                '{"query":"OpenAI news May 2026 official","count":3}'
                            ),
                        },
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "decision_2",
                        "function": {
                            "name": "react_decision",
                            "arguments": (
                                '{"action":"final_answer",'
                                '"reason":"Now enough sources."}'
                            ),
                        },
                    }
                ],
            },
            "第三次搜索后信息足够。",
        ]
    )
    pattern = ReActPattern(
        max_iterations=5,
        repeated_tool_decision_after_consecutive_tool_calls=2,
    )
    tool = FakeSearchTool()
    context = ExecutionContext()
    context.add_user_message("最近 AI 新闻")

    result = await pattern.run(context=context, tools=[tool], llm=llm)

    assert result["success"] is True
    assert result["response"] == "第三次搜索后信息足够。"
    assert len(tool.calls) == 3
    assert len(llm.calls) == 6
    assert [schema["function"]["name"] for schema in llm.calls[2]["tools"]] == [
        "react_decision"
    ]
    assert "official source" in "\n".join(
        str(message.get("content") or "") for message in llm.calls[3]["messages"]
    )
    assert [schema["function"]["name"] for schema in llm.calls[4]["tools"]] == [
        "react_decision"
    ]
    assert [schema["function"]["name"] for schema in llm.calls[5]["tools"]] == [
        "final_answer"
    ]


@pytest.mark.asyncio
async def test_react_pattern_defers_repeated_final_decision_to_normal_loop() -> None:
    llm = FakeLLM(
        responses=[
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "search_1",
                        "function": {
                            "name": "zhipu_web_search",
                            "arguments": '{"query":"locate source file","count":10}',
                        },
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "search_2",
                        "function": {
                            "name": "zhipu_web_search",
                            "arguments": '{"query":"find pptx input path","count":5}',
                        },
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "decision_1",
                        "function": {
                            "name": "react_decision",
                            "arguments": (
                                '{"action":"tool_call",'
                                '"reason":"Need to perform the actual file edit.",'
                                '"missing_verification":"remove the NotebookLM watermark from the PPTX"}'
                            ),
                        },
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "write_1",
                        "function": {
                            "name": "write_file",
                            "arguments": (
                                '{"file_path":"MegaCube_Infinite_AI2.pptx",'
                                '"content":"watermark removed"}'
                            ),
                        },
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "final_1",
                        "function": {
                            "name": "final_answer",
                            "arguments": (
                                '{"response_language":"Simplified Chinese",'
                                '"answer":"已完成处理。"}'
                            ),
                        },
                    }
                ],
            },
        ]
    )
    pattern = ReActPattern(
        max_iterations=5,
        repeated_tool_decision_after_consecutive_tool_calls=2,
    )
    search_tool = FakeSearchTool()
    write_tool = FakeWriteFileTool()
    context = ExecutionContext()
    context.add_user_message("去除右下角 notebooklm 水印")
    runtime = PatternRuntime()

    result = await pattern.run(
        context=context,
        tools=[search_tool, write_tool],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert result["response"] == "已完成处理。"
    assert len(search_tool.calls) == 2
    assert len(write_tool.calls) == 1
    assistant_messages = [
        message.content for message in context.messages if message.role == "assistant"
    ]
    assert "已找到源文件路径。正在处理去除水印，请稍候..." not in assistant_messages
    assert assistant_messages[-1] == "已完成处理。"
    continue_checkpoints = [
        checkpoint
        for checkpoint in runtime.checkpoints
        if checkpoint["label"] == "repeated_tool_decision_continue"
    ]
    assert len(continue_checkpoints) == 1
    assert any(
        message.role == "system"
        and "Repeated tool decision continuation guidance" in message.content
        and "remove the NotebookLM watermark from the PPTX" in message.content
        for message in context.messages
    )


@pytest.mark.asyncio
async def test_react_pattern_streams_final_answer_after_repeated_guard() -> None:
    llm = StreamingRepeatedGuardFinalAnswerLLM()
    pattern = ReActPattern(
        max_iterations=4,
        repeated_tool_decision_after_consecutive_tool_calls=2,
    )
    tool = FakeSearchTool()
    context = ExecutionContext(execution_id="task-1")
    context.add_user_message("Search AI news")
    outbound = OutboundCollector()
    runtime = PatternRuntime(execution_id="task-1", outbound_message_handler=outbound)

    result = await pattern.run(
        context=context,
        tools=[tool],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert result["response"] == "Fallback answer."
    assert [event["type"] for event in outbound.events] == [
        "final_answer_start",
        "final_answer_delta",
        "final_answer_end",
    ]
    assert outbound.events[-1]["content"] == "Fallback answer."


@pytest.mark.asyncio
async def test_react_pattern_supports_plain_function_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    to_thread_calls: list[dict[str, Any]] = []

    async def fake_to_thread(fn: Any, /, *args: Any, **kwargs: Any) -> Any:
        to_thread_calls.append({"fn": fn, "args": args, "kwargs": kwargs})
        return fn(*args, **kwargs)

    def double_number(value: int) -> dict[str, Any]:
        """Double a numeric input."""
        return {"result": value * 2}

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    llm = FakeLLM(
        responses=[
            {
                "content": "Use the plain function.",
                "tool_calls": [
                    {
                        "id": "call_plain",
                        "function": {
                            "name": "double_number",
                            "arguments": '{"value":4}',
                        },
                    }
                ],
            },
            {"content": "The result is 8.", "done": True},
        ]
    )
    pattern = ReActPattern(max_iterations=3)
    context = ExecutionContext(system_prompt="You are helpful.")
    context.add_user_message("Double 4")

    result = await pattern.run(context=context, tools=[double_number], llm=llm)

    assert result["success"] is True
    assert result["response"] == "The result is 8."
    assert to_thread_calls
    assert to_thread_calls[0]["kwargs"] == {"value": 4}
    tool_schema = next(
        schema
        for schema in llm.calls[0]["tools"]
        if schema["function"]["name"] == "double_number"
    )
    assert tool_schema["function"]["description"] == "Double a numeric input."
    parameters = tool_schema["function"]["parameters"]
    assert parameters["properties"]["value"]["type"] == "integer"
    assert parameters["required"] == ["value"]
    assert context.messages[2].content == "Tool double_number returned: {'result': 8}"


@pytest.mark.asyncio
async def test_react_pattern_uses_langchain_tool_schema() -> None:
    @langchain_tool
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    llm = FakeLLM(
        responses=[
            {
                "tool_calls": [
                    {
                        "id": "call_add",
                        "function": {
                            "name": "add_numbers",
                            "arguments": '{"a":2,"b":3}',
                        },
                    }
                ],
            },
            {"content": "The result is 5.", "done": True},
        ]
    )
    pattern = ReActPattern(max_iterations=3)
    context = ExecutionContext()
    context.add_user_message("Add 2 and 3")

    result = await pattern.run(context=context, tools=[add_numbers], llm=llm)

    assert result["success"] is True
    tool_schema = next(
        schema
        for schema in llm.calls[0]["tools"]
        if schema["function"]["name"] == "add_numbers"
    )
    parameters = tool_schema["function"]["parameters"]
    assert parameters["properties"]["a"]["type"] == "integer"
    assert parameters["properties"]["b"]["type"] == "integer"
    assert parameters["required"] == ["a", "b"]
    assert context.get_messages_by_role("tool")[-1].content == (
        "Tool add_numbers returned: 5"
    )


@pytest.mark.asyncio
async def test_react_pattern_supports_legacy_args_type_schema() -> None:
    llm = FakeLLM(responses=[{"content": "Done.", "done": True}])
    pattern = ReActPattern(max_iterations=1)
    context = ExecutionContext()
    context.add_user_message("Inspect schema")

    result = await pattern.run(context=context, tools=[LegacySchemaTool()], llm=llm)

    assert result["success"] is True
    tool_schema = next(
        schema
        for schema in llm.calls[0]["tools"]
        if schema["function"]["name"] == "legacy_schema"
    )
    parameters = tool_schema["function"]["parameters"]
    assert parameters["properties"]["value"]["type"] == "integer"
    assert parameters["required"] == ["value"]


def test_react_compacts_provider_tool_schema_without_losing_named_fields() -> None:
    class CompactSchemaArgs:
        @staticmethod
        def model_json_schema() -> dict[str, Any]:
            return {
                "title": "CompactSchemaArgs",
                "type": "object",
                "properties": {
                    "title": {
                        "title": "Title",
                        "type": "string",
                        "description": "First line.\n\n  Second line.",
                        "default": "draft",
                    },
                    "properties": {
                        "title": "Properties",
                        "type": "object",
                        "properties": {
                            "value": {
                                "title": "Value",
                                "type": "integer",
                            }
                        },
                    },
                },
                "required": ["title", "properties"],
            }

    class CompactSchemaTool:
        def __init__(self) -> None:
            class Metadata:
                name = "compact_schema"
                description = "Compact\n\n  - first   step\n  - second step."

            self.metadata = Metadata()

        def args_type(self) -> type[CompactSchemaArgs]:
            return CompactSchemaArgs

    schema = ReActPattern()._build_tool_schema(CompactSchemaTool())

    assert schema["function"]["description"] == (
        "Compact\n\n- first step\n- second step."
    )
    parameters = schema["function"]["parameters"]
    assert "title" not in parameters
    assert parameters["properties"]["title"] == {
        "type": "string",
        "description": "First line.\n\nSecond line.",
        "default": "draft",
    }
    assert parameters["properties"]["properties"] == {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
    }
    assert parameters["required"] == ["title", "properties"]


@pytest.mark.asyncio
async def test_react_pattern_injects_memory_context_and_skill_index() -> None:
    llm = FakeLLM(
        responses=[
            {
                "content": "Loading the skill first.",
                "tool_calls": [
                    {
                        "id": "call_skill",
                        "function": {
                            "name": "load_skill",
                            "arguments": '{"skill_name":"test-skill"}',
                        },
                    }
                ],
                "done": False,
            },
            {"content": "Done.", "done": True},
        ]
    )
    memory_store = FakeMemoryStore()
    skill_manager = FakeSkillManager()
    pattern = ReActPattern(max_iterations=3)
    context = ExecutionContext(system_prompt="You are helpful.")
    context.add_user_message("Do the thing")

    result = await pattern.run(
        context=context,
        tools=[],
        llm=llm,
        memory_store=memory_store,
        skill_manager=skill_manager,
        allowed_skills=["test-skill"],
    )

    assert result["success"] is True
    assert [search["filters"]["category"] for search in memory_store.searches] == [
        "react_memory",
        "general",
    ]
    first_system_prompt = llm.calls[0]["messages"][0]["content"]
    assert "Use the stored project preference." in first_system_prompt
    assert "Available skills:" in first_system_prompt
    assert "- test-skill: A test skill" in first_system_prompt
    tool_names = [
        tool["function"]["name"] for tool in list(llm.calls[0].get("tools") or [])
    ]
    assert "load_skill" in tool_names
    # After load_skill, the full guidance appears in the next system prompt.
    assert skill_manager.loaded == ["test-skill"]
    second_system_prompt = llm.calls[1]["messages"][0]["content"]
    assert "Available Skill: test-skill" in second_system_prompt
    assert "Follow the selected skill instructions." in second_system_prompt


@pytest.mark.asyncio
async def test_react_pattern_emits_memory_retrieve_trace_events() -> None:
    llm = FakeLLM(responses=[{"content": "Done.", "done": True}])
    memory_store = FakeMemoryStore()
    tracer = TraceEventRecorder()
    runtime = PatternRuntime(tracer=tracer, execution_id="task-memory-trace")
    pattern = ReActPattern(max_iterations=1, tool_choice="none")
    context = ExecutionContext(execution_id="task-memory-trace")
    context.add_user_message("Use memory")

    result = await pattern.run(
        context=context,
        tools=[],
        llm=llm,
        runtime=runtime,
        memory_store=memory_store,
    )

    assert result["success"] is True
    memory_events = [
        event for event in tracer.events if "memory_retrieve" in event["event_type"]
    ]
    assert [event["event_type"] for event in memory_events] == [
        "task_start_memory_retrieve",
        "task_end_memory_retrieve",
    ]
    assert all(event["task_id"] == "task-memory-trace" for event in memory_events)
    store_events = [
        event for event in tracer.events if "memory_store" in event["event_type"]
    ]
    # store_memory is model-driven now; no tool call means no store events.
    assert store_events == []


@pytest.mark.asyncio
async def test_react_pattern_keeps_tools_available_after_successful_tool() -> None:
    llm = FakeLLM(
        responses=[
            {
                "content": "Need calculation.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                        },
                    }
                ],
                "done": False,
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "final_call",
                        "function": {
                            "name": "final_answer",
                            "arguments": '{"answer":"The result is 4."}',
                        },
                    }
                ],
                "done": False,
            },
        ]
    )
    pattern = ReActPattern(max_iterations=3)
    tool = FakeTool()
    context = ExecutionContext()
    context.add_user_message("Calculate 2+2")

    result = await pattern.run(context=context, tools=[tool], llm=llm)

    assert result["success"] is True
    assert result["response"] == "The result is 4."
    assert tool.calls == [{"expression": "2+2"}]
    assert llm.calls[0]["tools"][0]["function"]["name"] == "calculator"
    second_call_tool_names = [
        schema["function"]["name"] for schema in llm.calls[1]["tools"]
    ]
    assert "calculator" in second_call_tool_names
    assert "final_answer" in second_call_tool_names
    assert "Do not call tools again" not in llm.calls[1]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_react_pattern_reserves_control_tool_names_in_schema() -> None:
    llm = FakeLLM(responses=[{"content": "No tools needed."}])
    pattern = ReActPattern(max_iterations=1)
    context = ExecutionContext()
    context.add_user_message("Say hi")

    result = await pattern.run(
        context=context,
        tools=[FakeAskUserTool()],
        llm=llm,
    )

    assert result["success"] is True
    tool_names = [schema["function"]["name"] for schema in llm.calls[0]["tools"]]
    assert tool_names.count("ask_user_question") == 1
    assert "final_answer" in tool_names
    assert "send_message" in tool_names
    assert "complete_task" not in tool_names
    ask_user_schema = next(
        schema
        for schema in llm.calls[0]["tools"]
        if schema["function"]["name"] == "ask_user_question"
    )
    ask_user_description = ask_user_schema["function"]["description"]
    assert "cannot continue without missing user-provided information" in (
        ask_user_description
    )
    assert "Do not use it to confirm execution strategy" in ask_user_description
    assert "whether to use memory" in ask_user_description
    system_prompt = llm.calls[0]["messages"][0]["content"]
    assert "Only call tools that are present in the current tool schema" in (
        system_prompt
    )
    assert "tool names mentioned in memory" in system_prompt
    assert "call the final_answer tool exactly once" in system_prompt
    assert (
        "Never put final_answer in the same response as any other tool call"
        in system_prompt
    )
    final_answer_schema = next(
        schema
        for schema in llm.calls[0]["tools"]
        if schema["function"]["name"] == "final_answer"
    )["function"]
    assert (
        "same natural language as the current user request"
        in final_answer_schema["description"]
    )
    assert (
        "Call this tool alone: never place it in the same response as any "
        "other tool call" in final_answer_schema["description"]
    )
    assert "response_language" in final_answer_schema["parameters"]["required"]
    assert "outcome" in final_answer_schema["parameters"]["required"]
    assert final_answer_schema["parameters"]["properties"]["outcome"]["enum"] == [
        "completed",
        "partial",
        "blocked",
    ]
    response_language_schema = final_answer_schema["parameters"]["properties"][
        "response_language"
    ]
    assert "Simplified Chinese" in response_language_schema["description"]
    assert "Traditional Chinese" in response_language_schema["description"]
    assert "generic Chinese" in response_language_schema["description"]
    answer_schema = final_answer_schema["parameters"]["properties"]["answer"]
    assert "response_language" in answer_schema["description"]
    assert "tool results, source documents" in answer_schema["description"]
    assert "## FINAL DELIVERABLE FILE REFERENCES" not in answer_schema["description"]
    assert "exact markdown_link" in answer_schema["description"]
    assert "get_workspace_output_files" not in answer_schema["description"]


def test_interaction_type_list_has_one_source() -> None:
    from xagent.core.tools.adapters.vibe.ask_user_tool import InteractionArg
    from xagent.core.tools.adapters.vibe.interaction_types import INTERACTION_TYPES
    from xagent.web.services.task_interaction_service import _V1_INTERACTION_TYPES

    assert INTERACTION_TYPES == (
        "select_one",
        "select_multiple",
        "text_input",
        "file_upload",
        "confirm",
        "number_input",
        "action_cards",
    )
    assert InteractionArg.model_fields["type"].description == (
        "Type of interaction: select_one, select_multiple, text_input, "
        "file_upload, confirm, number_input, action_cards"
    )
    assert frozenset(INTERACTION_TYPES) == _V1_INTERACTION_TYPES


async def test_react_pattern_ask_user_question_schema_derives_its_type_enum_from_one_source() -> (
    None
):
    """The ask_user_question tool's ``interactions[].type`` enum is built
    from ``interaction_types.INTERACTION_TYPES`` rather than a copy written
    out in this schema; a name added or reordered there must show up here
    unchanged."""

    from xagent.core.tools.adapters.vibe.interaction_types import INTERACTION_TYPES

    llm = FakeLLM(responses=[{"content": "No tools needed."}])
    pattern = ReActPattern(max_iterations=1)
    context = ExecutionContext()
    context.add_user_message("Say hi")

    result = await pattern.run(
        context=context,
        tools=[FakeAskUserTool()],
        llm=llm,
    )

    assert result["success"] is True
    ask_user_schema = next(
        schema
        for schema in llm.calls[0]["tools"]
        if schema["function"]["name"] == "ask_user_question"
    )["function"]
    type_enum = ask_user_schema["parameters"]["properties"]["interactions"]["items"][
        "properties"
    ]["type"]["enum"]
    assert type_enum == list(INTERACTION_TYPES)


def test_react_module_pulls_in_no_web_modules() -> None:
    """react.py must not depend on xagent.web. The interaction type list it
    reads lives in an import-free module for exactly this reason -- putting
    it beside InteractionArg in ask_user_tool pulls the tool-registration
    chain and 61 xagent.web modules into every import of this pattern."""
    import subprocess
    import sys

    probe = (
        "import sys; import xagent.core.agent.pattern.react.react; "
        "print(len([m for m in sys.modules if m.startswith('xagent.web')]))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "0"


def test_react_final_answer_lookup_instruction_tracks_active_workspace_tool() -> None:
    pattern = ReActPattern()

    assert not inspect.signature(pattern._final_answer_tool_schema).parameters
    schemas = pattern._tool_schemas_with_builtin_controls([FakeWorkspaceOutputTool()])
    final_answer_schema = next(
        schema for schema in schemas if schema["function"]["name"] == "final_answer"
    )
    answer_description = final_answer_schema["function"]["parameters"]["properties"][
        "answer"
    ]["description"]

    assert "get_workspace_output_files" in answer_description


@pytest.mark.asyncio
async def test_react_pattern_can_finish_with_final_answer_tool() -> None:
    llm = FakeLLM(
        responses=[
            {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                        },
                    }
                ],
            },
            {
                "tool_calls": [
                    {
                        "id": "call_final",
                        "function": {
                            "name": "final_answer",
                            "arguments": '{"answer":"The result is 4."}',
                        },
                    }
                ],
            },
        ]
    )
    pattern = ReActPattern(max_iterations=3)
    tool = FakeTool()
    context = ExecutionContext()
    context.add_user_message("Calculate 2+2")

    result = await pattern.run(context=context, tools=[tool], llm=llm)

    assert result["success"] is True
    assert result["response"] == "The result is 4."
    assert tool.calls == [{"expression": "2+2"}]
    assert context.messages[-2].role == "tool"
    assert context.messages[-2].tool_call_id == "call_final"
    assert context.messages[-2].metadata["tool_name"] == "final_answer"
    assert context.messages[-1].role == "assistant"
    assert context.messages[-1].content == "The result is 4."


@pytest.mark.asyncio
async def test_react_pattern_final_answer_clears_trailing_pending_before_checkpoint() -> (
    None
):
    # The trailing call is another control tool (send_message), not a work
    # tool: a work tool here would instead be stripped from the batch before
    # final_answer ever reaches this point (see the strip tests above), which
    # would defeat what this test is pinning - that a control result of
    # "completed" clears whatever is still queued behind it.
    llm = FakeLLM(
        responses=[
            {
                "tool_calls": [
                    {
                        "id": "call_final",
                        "function": {
                            "name": "final_answer",
                            "arguments": '{"answer":"Done."}',
                        },
                    },
                    {
                        "id": "call_message",
                        "function": {
                            "name": "send_message",
                            "arguments": (
                                '{"message":"Still working",'
                                '"message_type":"progress","expect_response":false}'
                            ),
                        },
                    },
                ],
            },
        ]
    )
    pattern = ReActPattern(max_iterations=2)
    runtime = PatternRuntime()
    tool = FakeTool()
    context = ExecutionContext()
    context.add_user_message("Finish and do not continue")

    result = await pattern.run(
        context=context,
        tools=[tool],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert result["response"] == "Done."
    assert tool.calls == []
    assert pattern.pending_tool_calls == []
    final_checkpoint = next(
        checkpoint
        for checkpoint in runtime.checkpoints
        if checkpoint["label"] == "final"
    )
    assert final_checkpoint["pattern_state"]["status"] == "completed"
    assert final_checkpoint["pattern_state"]["pending_tool_calls"] == []


def test_react_decision_schema_assesses_work_before_choosing_action() -> None:
    schema = ReActPattern()._react_decision_tool_schema()
    parameters = schema["function"]["parameters"]

    assert list(parameters["properties"]) == [
        "reason",
        "missing_verification",
        "action",
    ]
    assert parameters["required"] == [
        "reason",
        "missing_verification",
        "action",
    ]
    assert parameters["additionalProperties"] is False


def test_react_decision_does_not_finalize_with_missing_work() -> None:
    response = {
        "tool_calls": [
            {
                "id": "decision_contradictory",
                "function": {
                    "name": "react_decision",
                    "arguments": json.dumps(
                        {
                            "reason": "One more audio clip must be synthesized.",
                            "missing_verification": "Synthesize the final audio clip.",
                            "action": "final_answer",
                        }
                    ),
                },
            }
        ]
    }

    decision = ReActPattern()._parse_react_decision(response)

    assert decision == {
        "action": "tool_call",
        "reason": "One more audio clip must be synthesized.",
        "missing_verification": "Synthesize the final audio clip.",
    }


@pytest.mark.asyncio
async def test_react_pattern_accepts_plain_text_response() -> None:
    llm = FakeLLM(responses=["Direct answer"])
    pattern = ReActPattern(max_iterations=1)
    context = ExecutionContext()
    context.add_user_message("Say hi")

    result = await pattern.run(context=context, tools=[], llm=llm)

    assert result == {
        "success": True,
        "output": "Direct answer",
        "response": "Direct answer",
        "status": "completed",
        "completion_outcome": "completed",
    }
    assert context.messages[-1].content == "Direct answer"


@pytest.mark.asyncio
async def test_react_pattern_can_run_as_strict_single_call() -> None:
    llm = FakeLLM(responses=["Direct answer"])
    pattern = ReActPattern(max_iterations=1, tool_choice="none")
    context = ExecutionContext()
    context.add_user_message("Say hi")

    result = await pattern.run(context=context, tools=[FakeTool()], llm=llm)

    assert result["success"] is True
    assert result["status"] == "completed"
    assert llm.calls[0]["tools"] is None
    assert llm.calls[0]["tool_choice"] is None


@pytest.mark.asyncio
async def test_react_pattern_errors_without_llm() -> None:
    pattern = ReActPattern()
    context = ExecutionContext()
    context.add_user_message("Hello")

    result = await pattern.run(context=context, tools=[], llm=None)

    assert result["success"] is False
    assert "requires an llm" in result["error"]


@pytest.mark.asyncio
async def test_react_pattern_send_message_without_response_continues() -> None:
    llm = FakeLLM(
        responses=[
            {
                "content": "Sending progress.",
                "tool_calls": [
                    {
                        "id": "call_message",
                        "function": {
                            "name": "send_message",
                            "arguments": '{"message":"Still working","message_type":"progress","expect_response":false}',
                        },
                    }
                ],
            },
            {"content": "All done."},
        ]
    )
    pattern = ReActPattern(max_iterations=3)
    runtime = PatternRuntime()
    context = ExecutionContext()
    context.add_user_message("Work")

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert result["success"] is True
    assert result["response"] == "All done."
    assert len(runtime.outbound_messages) == 1
    outbound_message = runtime.outbound_messages[0]
    assert outbound_message["type"] == "agent_message"
    assert outbound_message["execution_id"] == context.execution_id
    assert outbound_message["message"] == "Still working"
    assert outbound_message["message_type"] == "progress"
    assert outbound_message["expect_response"] is False
    assert outbound_message["visible"] is True
    assert outbound_message["step_id"] == outbound_message["metadata"]["step_id"]
    tool_messages = context.get_messages_by_role("tool")
    assert len(tool_messages) == 1
    assert tool_messages[0].metadata["tool_name"] == "send_message"
    next_call_messages = llm.calls[1]["messages"]
    assert next_call_messages[-1]["role"] == "tool"
    assert all(
        message.get("content") != "Still working" for message in next_call_messages
    )
    assert pattern.tool_ledger["call_message"].status == "completed"


@pytest.mark.asyncio
async def test_react_pattern_send_message_with_response_waits() -> None:
    llm = FakeLLM(
        responses=[
            {
                "content": "Need input.",
                "tool_calls": [
                    {
                        "id": "call_question",
                        "function": {
                            "name": "send_message",
                            "arguments": '{"message":"Choose A or B","message_type":"question","expect_response":true}',
                        },
                    }
                ],
            }
        ]
    )
    pattern = ReActPattern(max_iterations=2)
    sent_messages: list[dict[str, Any]] = []
    runtime = PatternRuntime(outbound_message_handler=sent_messages.append)
    context = ExecutionContext()
    context.add_user_message("Ask")

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert result["success"] is False
    assert result["status"] == "waiting_for_user"
    assert result["message"] == "Choose A or B"
    assert sent_messages == runtime.outbound_messages
    assert sent_messages[0]["message"] == "Choose A or B"
    assert sent_messages[0]["expect_response"] is True
    assert pattern.status == "waiting_for_user"
    tool_messages = context.get_messages_by_role("tool")
    assert tool_messages[0].tool_call_id == "call_question"
    assert tool_messages[0].metadata["raw_result"]["status"] == "waiting_for_user"


@pytest.mark.asyncio
async def test_react_pattern_ask_user_question_pauses_with_structured_payload() -> None:
    llm = FakeLLM(
        responses=[
            {
                "content": "Need structured input.",
                "tool_calls": [
                    {
                        "id": "call_question_form",
                        "function": {
                            "name": "ask_user_question",
                            "arguments": (
                                '{"message":"Pick one","interactions":'
                                '[{"type":"select_one","field":"choice","label":"Choice"}]}'
                            ),
                        },
                    }
                ],
            }
        ]
    )
    pattern = ReActPattern(max_iterations=2)
    runtime = PatternRuntime(execution_id="exec-1")
    context = ExecutionContext()
    context.add_user_message("Ask")

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert result["success"] is False
    assert result["status"] == "waiting_for_user"
    assert result["message"] == "Pick one"
    assert len(runtime.outbound_messages) == 1
    outbound_message = runtime.outbound_messages[0]
    assert outbound_message["type"] == "agent_message"
    assert outbound_message["execution_id"] == "exec-1"
    assert outbound_message["message"] == "Pick one"
    assert outbound_message["message_type"] == "question"
    assert outbound_message["expect_response"] is True
    assert outbound_message["visible"] is True
    assert outbound_message["step_id"] == outbound_message["metadata"]["step_id"]
    assert outbound_message["metadata"]["interactions"] == [
        {
            "type": "select_one",
            "field": "choice",
            "label": "Choice",
        }
    ]
    assert pattern.tool_ledger["call_question_form"].status == "completed"
    tool_messages = context.get_messages_by_role("tool")
    assert tool_messages[0].tool_call_id == "call_question_form"
    assert tool_messages[0].metadata["raw_result"]["status"] == "waiting_for_user"


@pytest.mark.asyncio
async def test_react_pattern_ask_user_question_drops_invalid_options() -> None:
    llm = FakeLLM(
        responses=[
            {
                "content": "Need structured input.",
                "tool_calls": [
                    {
                        "id": "call_question_form",
                        "function": {
                            "name": "ask_user_question",
                            "arguments": (
                                '{"message":"Pick one","interactions":'
                                '[{"type":"select_one","field":"choice",'
                                '"label":"Choice","options":['
                                '{"label":"A","value":"a"},'
                                '{"label":"","value":"empty-label"},'
                                '{"value":"missing-label"},'
                                '{"label":"Missing value"},'
                                '{"label":"B","value":"b","description":"Bee"}'
                                "]}]}"
                            ),
                        },
                    }
                ],
            }
        ]
    )
    pattern = ReActPattern(max_iterations=2)
    runtime = PatternRuntime(execution_id="exec-1")
    context = ExecutionContext()
    context.add_user_message("Ask")

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert result["status"] == "waiting_for_user"
    assert result["interactions"][0]["options"] == [
        {"label": "A", "value": "a"},
        {"label": "B", "value": "b", "description": "Bee"},
    ]


# Independent reference for JavaScript's String.prototype.trim() semantics,
# derived from unicodedata rather than the production _INTERACTION_TRIM_CHARS
# constant -- reusing that constant here would make every assertion below
# self-proving instead of an independent check. Built once at module scope
# and reused, instead of being rebuilt inline in each function that needs it.
_JS_TRIM = {chr(c) for c in range(0x110000) if unicodedata.category(chr(c)) == "Zs"}
_JS_TRIM |= {"\t", "\n", "\v", "\f", "\r", "\ufeff", "\u2028", "\u2029"}


def _js_trim_equivalent(value: str) -> str:
    """Reference JavaScript String.prototype.trim(), via the independent
    _JS_TRIM set above."""
    return value.strip("".join(_JS_TRIM))


def test_trim_table_covers_every_javascript_trimmed_code_point() -> None:
    """Coverage check, superset half: _INTERACTION_TRIM_CHARS must contain
    every code point JavaScript's trim() removes, or writing the normalized
    field back into item["field"] and relying on the frontend's own trim()
    being a no-op on it stops holding."""
    assert _JS_TRIM <= set(_INTERACTION_TRIM_CHARS)


def test_trim_table_python_only_difference_is_exactly_five_code_points() -> None:
    """Coverage check, differential half: pins the five Python-only code
    points exactly. The superset check above alone would still pass if one
    of these five were mistyped (e.g. U+001C typoed into U+001B), since a
    typo like that only shrinks the Python-only difference by one member and
    stays within a superset of the JavaScript table."""
    assert set(_INTERACTION_TRIM_CHARS) - _JS_TRIM == {
        "\x1c",
        "\x1d",
        "\x1e",
        "\x1f",
        "\x85",
    }


def test_python_whitespace_set_is_a_subset_of_the_frozen_trim_table() -> None:
    """task_interaction_service.py's write-side field/option checks use
    plain str.strip() rather than importing _INTERACTION_TRIM_CHARS -- that
    is only safe to reason about at all if every code point Python's own
    str.isspace() treats as whitespace is already inside the frozen table.
    Computed independently (via isspace(), not by re-deriving from
    _INTERACTION_TRIM_CHARS or from _JS_TRIM), so a typo that dropped one of
    the five Python-only code points from the frozen table would be caught
    here even if it happened to still pass the two JS-side checks above."""
    python_whitespace = {chr(c) for c in range(0x110000) if chr(c).isspace()}
    assert python_whitespace <= set(_INTERACTION_TRIM_CHARS)


def test_normalize_keeps_well_formed_options_and_fields() -> None:
    """A normal option and a normal field pass through unchanged."""
    normalized = _normalize_ask_user_interactions(
        [
            {
                "type": "select_one",
                "field": "choice",
                "options": [{"label": "A", "value": "a"}],
            }
        ]
    )
    assert normalized[0]["options"] == [{"label": "A", "value": "a"}]
    assert normalized[0]["field"] == "choice"


def test_normalize_returns_empty_list_for_non_list_interactions() -> None:
    """A top-level interactions value that is not a list -- the model
    sending a dict or a string instead of an array, say -- is rejected
    outright rather than partially processed."""
    assert _normalize_ask_user_interactions("not-a-list") == []
    assert _normalize_ask_user_interactions({"field": "choice"}) == []
    assert _normalize_ask_user_interactions(None) == []


def test_normalize_skips_non_dict_elements_within_the_list() -> None:
    """A non-dict element inside an otherwise well-formed interactions list
    is dropped, and the well-formed interactions around it are still
    normalized -- one malformed element does not fail the whole batch."""
    normalized = _normalize_ask_user_interactions(
        [
            "not-a-dict",
            {
                "type": "select_one",
                "field": "choice",
                "options": [{"label": "A", "value": "a"}],
            },
            42,
            None,
        ]
    )
    assert len(normalized) == 1
    assert normalized[0]["field"] == "choice"


_BLANK_TEXT_CASES = [
    ("v1_empty", ""),
    ("v2_ascii_spaces", "   "),
    ("v3_mixed_whitespace", "\t\n "),
    ("v4_fullwidth_space", "\u3000"),
    ("v5_bom_only", "\ufeff"),
    ("v8_python_only_control", "\x1c"),
    ("v9_js_line_separator", "\u2028"),
    ("v10_nbsp", "\xa0"),
]


@pytest.mark.parametrize("case_id,value", _BLANK_TEXT_CASES)
def test_normalize_drops_blank_options(case_id: str, value: str) -> None:
    """An option whose label or value is blank under
    _INTERACTION_TRIM_CHARS is dropped -- the same treatment the existing
    empty-string-label case already gets (see the regression test above
    for missing/empty label or value), widened from "the empty string" to
    "blank under the trim table"."""
    normalized = _normalize_ask_user_interactions(
        [
            {
                "type": "select_one",
                "field": "choice",
                "options": [
                    {"label": value, "value": "kept-value"},
                    {"label": "kept-label", "value": value},
                    {"label": "A", "value": "a"},
                ],
            }
        ]
    )
    assert normalized[0]["options"] == [{"label": "A", "value": "a"}]


@pytest.mark.parametrize("case_id,value", _BLANK_TEXT_CASES)
def test_normalize_substitutes_blank_field(case_id: str, value: str) -> None:
    """A field name blank under _INTERACTION_TRIM_CHARS falls back to
    response_{index}."""
    normalized = _normalize_ask_user_interactions(
        [{"type": "select_one", "field": value, "label": "Choice"}]
    )
    assert normalized[0]["field"] == "response_0"


def test_normalize_substitutes_blank_field_using_its_own_index() -> None:
    """The fallback is f"response_{index}" for the interaction's own
    position, not a fixed "response_0" -- a well-formed interaction ahead
    of the blank one must not shift what the blank one falls back to."""
    normalized = _normalize_ask_user_interactions(
        [
            {"type": "select_one", "field": "choice", "label": "Choice"},
            {"type": "select_one", "field": "   ", "label": "Other"},
        ]
    )
    assert normalized[0]["field"] == "choice"
    assert normalized[1]["field"] == "response_1"


@pytest.mark.parametrize(
    "case_id,value,expected",
    [
        ("v6_bom_prefix", "\ufeffabc", "abc"),
        ("v7_bom_and_space_both_ends", "\ufeff abc\ufeff", "abc"),
    ],
)
def test_normalize_field_bom_normalizes_to_frontend_equivalent(
    case_id: str, value: str, expected: str
) -> None:
    """A field wrapped in a BOM (with or without interior spacing)
    normalizes to the same string the frontend's own trim() produces. The
    same normalization applies when the value arrives through the field/id/
    name alias chain rather than "field" directly."""
    normalized = _normalize_ask_user_interactions(
        [{"type": "select_one", "field": value, "label": "Choice"}]
    )
    assert normalized[0]["field"] == expected

    normalized_via_alias = _normalize_ask_user_interactions(
        [{"type": "select_one", "id": value, "label": "Choice"}]
    )
    assert normalized_via_alias[0]["field"] == expected


@pytest.mark.parametrize(
    "case_id,value",
    [
        ("v1_empty", ""),
        ("v2_ascii_spaces", "   "),
        ("v3_mixed_whitespace", "\t\n "),
        ("v4_fullwidth_space", "\u3000"),
        ("v5_bom_only", "\ufeff"),
        ("v6_bom_prefix", "\ufeffabc"),
        ("v7_bom_and_space_both_ends", "\ufeff abc\ufeff"),
        ("v8_python_only_control", "\x1c"),
        ("v9_js_line_separator", "\u2028"),
    ],
)
def test_normalized_field_is_stable_under_javascript_trim(
    case_id: str, value: str
) -> None:
    """Whatever field the normalizer produces must already be a fixed
    point of the frontend's own trim(). If it were not, writing the result
    back into item["field"] and trusting the frontend to leave it alone
    would silently stop being true for that input."""
    normalized = _normalize_ask_user_interactions(
        [{"type": "select_one", "field": value, "label": "Choice"}]
    )
    field = normalized[0]["field"]
    assert _js_trim_equivalent(field) == field


def test_normalize_logs_when_all_options_are_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An interaction whose options are all blank keeps the
    interaction (the question still goes out) and logs exactly one warning.
    The warning's extra keys are exactly {dropped, total, interaction_index},
    all ints, and its message format args are also all ints -- no
    model-controlled string (the field name, a label, a value) is allowed
    into either half of this log line."""
    with caplog.at_level(
        logging.WARNING, logger="xagent.core.agent.pattern.react.react"
    ):
        normalized = _normalize_ask_user_interactions(
            [
                {
                    "type": "select_one",
                    "field": "choice",
                    "options": [
                        {"label": "   ", "value": "   "},
                        {"label": "\ufeff", "value": "\ufeff"},
                    ],
                }
            ]
        )

    assert len(normalized) == 1
    assert normalized[0]["options"] == []

    dropped_records = [r for r in caplog.records if "dropped all" in r.getMessage()]
    assert len(dropped_records) == 1
    record = dropped_records[0]

    baseline_attrs = set(
        logging.LogRecord("n", logging.WARNING, "p", 1, "m", None, None).__dict__
    )
    # "taskName" is a LogRecord attribute added in 3.12. "message" and
    # (when some other test module's logging setup is still attached to the
    # root logger) "asctime" are added by a Formatter.format() pass -- set
    # record.message / record.asctime as a side effect of formatting, not by
    # this log call's own extra= payload. All three are excluded so this
    # comparison is stable regardless of Python version, test runner, or
    # which other test modules ran earlier in the same process.
    extra_keys = (
        set(record.__dict__) - baseline_attrs - {"taskName", "message", "asctime"}
    )
    assert extra_keys == {"dropped", "total", "interaction_index"}
    assert all(isinstance(record.__dict__[k], int) for k in extra_keys)
    assert record.args is None or all(isinstance(a, int) for a in record.args)
    # Both options in this interaction were blank, so dropped == total == 2,
    # not just "some int" -- pins the actual count, not merely its type.
    assert record.__dict__["dropped"] == 2
    assert record.__dict__["total"] == 2
    assert record.__dict__["interaction_index"] == 0


def test_normalize_keeps_surviving_option_text_verbatim() -> None:
    """An option that survives the blank filter keeps its label and
    value exactly as given -- normalization only judges blankness, it never
    rewrites content. A BOM-wrapped value is the only kind of input that can
    tell "judge blank" apart from "judge blank and also rewrite": a purely
    blank value would be dropped under either implementation, so it would
    not catch a rewrite regression."""
    raw = {"label": "\ufeffImport", "value": "\ufeffimport"}
    normalized = _normalize_ask_user_interactions(
        [{"type": "select_one", "field": "choice", "options": [raw]}]
    )
    assert normalized[0]["options"] == [raw]


def test_normalize_keeps_colliding_fields_at_single_tool_callsite(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Within one ask_user_question call, a BOM-wrapped field name and
    its plain counterpart now normalize to the same field. This function
    does not deduplicate -- both interactions are kept unchanged, and the
    single-tool call site sends the result on as-is (the multi-tool call
    site has its own dedup, covered separately below). One warning is
    logged, with an integer-only extra/message-args payload."""
    with caplog.at_level(
        logging.WARNING, logger="xagent.core.agent.pattern.react.react"
    ):
        normalized = _normalize_ask_user_interactions(
            [
                {"type": "select_one", "field": "\ufeffchoice"},
                {"type": "select_one", "field": "choice"},
            ]
        )

    assert [item["field"] for item in normalized] == ["choice", "choice"]

    collision_records = [
        r for r in caplog.records if "colliding field name" in r.getMessage()
    ]
    assert len(collision_records) == 1
    record = collision_records[0]

    baseline_attrs = set(
        logging.LogRecord("n", logging.WARNING, "p", 1, "m", None, None).__dict__
    )
    # "taskName" is a LogRecord attribute added in 3.12. "message" and
    # (when some other test module's logging setup is still attached to the
    # root logger) "asctime" are added by a Formatter.format() pass -- set
    # record.message / record.asctime as a side effect of formatting, not by
    # this log call's own extra= payload. All three are excluded so this
    # comparison is stable regardless of Python version, test runner, or
    # which other test modules ran earlier in the same process.
    extra_keys = (
        set(record.__dict__) - baseline_attrs - {"taskName", "message", "asctime"}
    )
    assert extra_keys == {"colliding_field_count", "total"}
    assert all(isinstance(record.__dict__[k], int) for k in extra_keys)
    assert record.args is None or all(isinstance(a, int) for a in record.args)


def test_normalize_blank_alias_does_not_fall_through_to_next_key() -> None:
    """The field/id/name alias chain keeps its raw truthiness
    check on purpose. A BOM-only field is truthy before normalization, so it
    wins the alias chain and is only normalized away to blank afterward --
    id="ok" is never reached. Widening the blankness judgment to include BOM
    does not create a new alias-chain outcome, it only adds a new way to
    reach the one a plain whitespace field already produced (the second
    case below, pinned as a regression)."""
    normalized_bom = _normalize_ask_user_interactions(
        [{"type": "select_one", "field": "\ufeff", "id": "ok"}]
    )
    assert normalized_bom[0]["field"] == "response_0"

    normalized_whitespace = _normalize_ask_user_interactions(
        [{"type": "select_one", "field": "   ", "id": "ok"}]
    )
    assert normalized_whitespace[0]["field"] == "response_0"


@pytest.mark.parametrize(
    "case_id,raw",
    [
        (
            "case1_alias_only",
            {
                "type": "select_one",
                "field": "f",
                "actions": [{"label": "A", "value": "a"}],
            },
        ),
        (
            "case2_options_and_unrelated_actions",
            {
                "type": "select_one",
                "field": "f",
                "options": [{"label": "A", "value": "a"}],
                "actions": [{"label": "X", "value": "x"}],
            },
        ),
        (
            "case3_options_not_a_list",
            {
                "type": "select_one",
                "field": "f",
                "options": "auto",
                "actions": [{"label": "B", "value": "b"}],
            },
        ),
        (
            "case4_actions_not_a_list",
            {
                "type": "select_one",
                "field": "f",
                "options": "auto",
                "actions": "bad",
            },
        ),
    ],
)
def test_normalize_strips_actions_alias_from_output(case_id: str, raw: dict) -> None:
    """The output never carries an actions key, unconditionally --
    whether options was missing, a well-formed list, or a malformed
    non-list value, and whether actions itself was a list or not. Case 1
    additionally asserts the alias survived into options: without that
    second assertion, moving the pop above the alias branch would still
    pass this case (actions is gone either way, just for the wrong reason)
    while silently losing the alias's only copy of the data -- a gap only
    test_normalize_aliases_actions_when_options_is_not_a_list below would
    otherwise catch alone."""
    normalized = _normalize_ask_user_interactions([raw])
    assert "actions" not in normalized[0]
    if case_id == "case1_alias_only":
        assert normalized[0]["options"] == [{"label": "A", "value": "a"}]


def test_normalize_aliases_actions_when_options_is_not_a_list() -> None:
    """When options is present but not a list, the alias branch --
    widened from "options" not in item to "options is not a list" -- still
    rescues actions into options, and the usual blank-option filter still
    runs on what it rescued. This is the test that would catch the pop
    running before the alias branch: with the pop moved earlier, actions
    would already be gone by the time the alias branch runs, options would
    stay "auto", and this assertion would fail even though every case in
    test_normalize_strips_actions_alias_from_output would still pass (that
    test only checks that actions is gone, not that its data went
    anywhere)."""
    normalized = _normalize_ask_user_interactions(
        [
            {
                "type": "select_one",
                "field": "f",
                "options": "auto",
                "actions": [
                    {"label": "   ", "value": "   "},
                    {"label": "B", "value": "b"},
                ],
            }
        ]
    )
    assert normalized[0]["options"] == [{"label": "B", "value": "b"}]
    assert "actions" not in normalized[0]


def test_normalize_logs_when_options_is_not_a_list(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """options present but neither a list nor rescued by an actions alias
    is a malformed shape the model produced; today it silently renders as
    no available options (the renderer falls back to interaction.options ||
    []), and this warning is the only signal that it happened. Same
    integer-only payload discipline as the other two warnings in this
    same function (the all-options-blank warning and the colliding-field-
    name warning)."""
    with caplog.at_level(
        logging.WARNING, logger="xagent.core.agent.pattern.react.react"
    ):
        normalized = _normalize_ask_user_interactions(
            [{"type": "select_one", "field": "choice", "options": "auto"}]
        )

    assert normalized[0]["options"] == "auto"
    assert "actions" not in normalized[0]

    records = [r for r in caplog.records if "non-list options" in r.getMessage()]
    assert len(records) == 1
    record = records[0]

    baseline_attrs = set(
        logging.LogRecord("n", logging.WARNING, "p", 1, "m", None, None).__dict__
    )
    # Same three attributes excluded for the same reasons as the other two
    # warning tests in this module: "taskName" (3.12+ LogRecord attribute),
    # "message" and "asctime" (added by a Formatter.format() pass, not by
    # this call's own extra= payload).
    extra_keys = (
        set(record.__dict__) - baseline_attrs - {"taskName", "message", "asctime"}
    )
    assert extra_keys == {"interaction_index"}
    assert all(isinstance(record.__dict__[k], int) for k in extra_keys)
    assert record.args is None or all(isinstance(a, int) for a in record.args)


@pytest.mark.asyncio
async def test_pause_for_tool_results_deduplicates_normalized_fields() -> None:
    """_pause_for_tool_results runs its own field-name dedup after
    calling the normalizer for each tool. Two tools whose fields now
    normalize (via the BOM fix) to the same string still end up with
    distinct field names in the published interactions -- narrowing the
    normalizer's output domain does not break the existing dedup loop."""
    pattern = ReActPattern(max_iterations=2)
    runtime = PatternRuntime(execution_id="exec-1")
    context = ExecutionContext()
    context.add_user_message("Ask")

    tool_call_a = {"id": "call_a", "name": "tool_a"}
    tool_call_b = {"id": "call_b", "name": "tool_b"}
    result_a = {
        "status": "waiting_for_user",
        "message": "Pick one",
        "message_type": "question",
        "interactions": [{"type": "select_one", "field": "\ufeffchoice"}],
    }
    result_b = {
        "status": "waiting_for_user",
        "message": "Pick another",
        "message_type": "question",
        "interactions": [{"type": "select_one", "field": "choice"}],
    }

    outcome = await pattern._pause_for_tool_results(
        waiting_pairs=[(tool_call_a, result_a), (tool_call_b, result_b)],
        context=context,
        runtime=runtime,
    )

    assert outcome["status"] == "waiting_for_user"
    assert [item["field"] for item in outcome["interactions"]] == [
        "choice",
        "choice_2",
    ]


@pytest.mark.asyncio
async def test_react_pattern_resume_waiting_without_user_response_stays_waiting() -> (
    None
):
    llm = FakeLLM(
        responses=[
            {
                "content": "Need input.",
                "tool_calls": [
                    {
                        "id": "call_question",
                        "function": {
                            "name": "send_message",
                            "arguments": '{"message":"Choose A or B","message_type":"question","expect_response":true}',
                        },
                    }
                ],
            }
        ]
    )
    pattern = ReActPattern(max_iterations=2)
    context = ExecutionContext()
    context.add_user_message("Ask")

    first = await pattern.run(context=context, tools=[], llm=llm)

    assert first["status"] == "waiting_for_user"

    resumed_pattern = ReActPattern(max_iterations=2)
    resumed_pattern.load_state(pattern.get_state())
    resumed_llm = FakeLLM([{"content": "Should not run"}])

    resumed = await resumed_pattern.run(context=context, tools=[], llm=resumed_llm)

    assert resumed["status"] == "waiting_for_user"
    assert resumed["message"] == "Choose A or B"
    assert resumed_llm.calls == []


@pytest.mark.asyncio
async def test_react_pattern_resume_waiting_after_user_response_continues() -> None:
    llm = FakeLLM(
        responses=[
            {
                "content": "Need input.",
                "tool_calls": [
                    {
                        "id": "call_question",
                        "function": {
                            "name": "send_message",
                            "arguments": '{"message":"Choose A or B","message_type":"question","expect_response":true}',
                        },
                    }
                ],
            }
        ]
    )
    pattern = ReActPattern(max_iterations=2)
    context = ExecutionContext()
    context.add_user_message("Ask")

    first = await pattern.run(context=context, tools=[], llm=llm)

    assert first["status"] == "waiting_for_user"
    context.add_user_message("B")

    resumed_pattern = ReActPattern(max_iterations=2)
    resumed_pattern.load_state(pattern.get_state())
    resumed_llm = FakeLLM([{"content": "Continuing with B."}])

    resumed = await resumed_pattern.run(context=context, tools=[], llm=resumed_llm)

    assert resumed["success"] is True
    assert resumed["output"] == "Continuing with B."
    assert len(resumed_llm.calls) == 1
    assert context.messages[-2].content == "B"
    resumed_messages = resumed_llm.calls[0]["messages"]
    assert resumed_messages[-1]["role"] == "user"
    assert "answer to a pending agent question" in resumed_messages[-1]["content"]
    assert "Pending question: Choose A or B" in resumed_messages[-1]["content"]
    assert "User answer: B" in resumed_messages[-1]["content"]


@pytest.mark.asyncio
async def test_react_pattern_replans_after_waiting_control_tool() -> None:
    llm = FakeLLM(
        responses=[
            {
                "content": "Need input, then calculate.",
                "tool_calls": [
                    {
                        "id": "call_question",
                        "function": {
                            "name": "send_message",
                            "arguments": '{"message":"Choose A or B","message_type":"question","expect_response":true}',
                        },
                    },
                    {
                        "id": "call_calc",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"5+5"}',
                        },
                    },
                ],
            }
        ]
    )
    pattern = ReActPattern(max_iterations=4)
    tool = FakeTool()
    context = ExecutionContext()
    context.add_user_message("Ask, then calculate")

    first = await pattern.run(context=context, tools=[tool], llm=llm)

    assert first["status"] == "waiting_for_user"
    assert pattern.pending_tool_calls == []
    assert pattern.tool_ledger["call_calc"].status == "cancelled"
    assert tool.calls == []

    context.add_user_message("B")
    resumed_pattern = ReActPattern(max_iterations=4)
    resumed_pattern.load_state(pattern.get_state())
    resumed_llm = FakeLLM(
        [
            {
                "tool_calls": [
                    {
                        "id": "call_calc_replanned",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"5+5"}',
                        },
                    }
                ]
            },
            {"content": "The result is 10.", "done": True},
        ]
    )

    resumed = await resumed_pattern.run(
        context=context,
        tools=[tool],
        llm=resumed_llm,
    )

    assert resumed["success"] is True
    assert tool.calls == [{"expression": "5+5"}]
    assert context.get_messages_by_role("tool")[-1].tool_call_id == (
        "call_calc_replanned"
    )
    resumed_messages = resumed_llm.calls[0]["messages"]
    cancelled_tool_result = next(
        message
        for message in resumed_messages
        if message.get("role") == "tool" and message.get("tool_call_id") == "call_calc"
    )
    assert "cancelled" in cancelled_tool_result["content"]


@pytest.mark.asyncio
async def test_react_pattern_resume_binds_original_task_to_store_memory() -> None:
    llm = FakeLLM(
        responses=[
            {
                "tool_calls": [
                    {
                        "id": "call_question",
                        "function": {
                            "name": "send_message",
                            "arguments": '{"message":"Choose A or B","message_type":"question","expect_response":true}',
                        },
                    }
                ],
            }
        ]
    )
    pattern = ReActPattern(max_iterations=3)
    context = ExecutionContext()
    context.add_user_message("Ask, then calculate")

    first = await pattern.run(context=context, tools=[], llm=llm)

    assert first["status"] == "waiting_for_user"
    context.add_user_message("B")
    resumed_pattern = ReActPattern(max_iterations=3)
    resumed_pattern.load_state(pattern.get_state())
    resumed_llm = FakeLLM(
        responses=[
            {
                "content": "Storing the preference.",
                "tool_calls": [
                    {
                        "id": "call_mem",
                        "function": {
                            "name": "store_memory",
                            "arguments": (
                                '{"content":"User chose option B.",'
                                '"kind":"user_preference"}'
                            ),
                        },
                    }
                ],
                "done": False,
            },
            {"content": "Continuing with B.", "done": True},
        ]
    )
    memory_store = MemoryToolStore()

    resumed = await resumed_pattern.run(
        context=context,
        tools=[],
        llm=resumed_llm,
        memory_store=memory_store,
    )

    assert resumed["success"] is True
    assert len(memory_store.added) == 1
    # The store_memory tool is bound to the original task, not the resume turn.
    assert memory_store.added[0].metadata["task"] == "Ask, then calculate"


@pytest.mark.asyncio
async def test_react_pattern_tool_errors_are_written_as_observations() -> None:
    llm = FakeLLM(
        responses=[
            {
                "content": "Use missing tool.",
                "tool_calls": [
                    {
                        "id": "call_missing",
                        "function": {
                            "name": "missing",
                            "arguments": '{"value":1}',
                        },
                    }
                ],
            },
            {"content": "Recovered after missing tool."},
        ]
    )
    pattern = ReActPattern(max_iterations=3)
    context = ExecutionContext()
    context.add_user_message("Recover")

    result = await pattern.run(context=context, tools=[], llm=llm)

    assert result["success"] is True
    assert result["response"] == "Recovered after missing tool."
    tool_message = context.get_messages_by_role("tool")[0]
    assert "Tool missing returned" in tool_message.content
    assert tool_message.metadata["raw_result"]["success"] is False
    assert "Tool not found" in tool_message.metadata["raw_result"]["error"]
    assert pattern.tool_ledger["call_missing"].status == "failed"


@pytest.mark.asyncio
async def test_react_pattern_tool_exception_is_written_as_observation() -> None:
    llm = FakeLLM(
        responses=[
            {
                "content": "Use broken tool.",
                "tool_calls": [
                    {
                        "id": "call_broken",
                        "function": {
                            "name": "broken",
                            "arguments": '{"value":2}',
                        },
                    }
                ],
            },
            {"content": "Recovered after broken tool."},
        ]
    )
    pattern = ReActPattern(max_iterations=3)
    context = ExecutionContext()
    context.add_user_message("Recover")

    result = await pattern.run(context=context, tools=[BrokenTool()], llm=llm)

    assert result["success"] is True
    assert result["response"] == "Recovered after broken tool."
    tool_message = context.get_messages_by_role("tool")[0]
    assert tool_message.metadata["raw_result"]["success"] is False
    assert "broken with" in tool_message.metadata["raw_result"]["error"]
    assert pattern.tool_ledger["call_broken"].status == "failed"


@pytest.mark.asyncio
async def test_react_pattern_failed_tool_result_emits_tool_error_trace() -> None:
    tracer = TraceEventRecorder()
    llm = FakeLLM(
        responses=[
            {
                "content": "Use failing tool.",
                "tool_calls": [
                    {
                        "id": "call_failed_result",
                        "function": {
                            "name": "failing_result",
                            "arguments": '{"value":3}',
                        },
                    }
                ],
            },
            {"content": "Recovered after failed tool result."},
        ]
    )
    pattern = ReActPattern(max_iterations=3)
    context = ExecutionContext(execution_id="failed-result")
    context.add_user_message("Recover")
    runtime = PatternRuntime(tracer=tracer)

    result = await pattern.run(
        context=context,
        tools=[FailingResultTool()],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    tool_message = context.get_messages_by_role("tool")[0]
    assert tool_message.metadata["raw_result"]["success"] is False
    assert pattern.tool_ledger["call_failed_result"].status == "failed"
    event_types = {event["event_type"] for event in tracer.events}
    assert "action_error_tool" in event_types
    assert "task_start_react" in event_types
    step_ids = {
        event["step_id"]
        for event in tracer.events
        if event["event_type"] in {"task_start_react", "action_start_tool"}
    }
    assert len(step_ids) == 1
    react_step_id = step_ids.pop()
    assert react_step_id is not None
    assert react_step_id.startswith("react_")
    assert react_step_id != "failed-result"


@pytest.mark.asyncio
async def test_react_pattern_status_error_tool_result_is_failed() -> None:
    tracer = TraceEventRecorder()
    llm = FakeLLM(
        responses=[
            {
                "content": "Use failing tool.",
                "tool_calls": [
                    {
                        "id": "call_status_error_result",
                        "function": {
                            "name": "status_error_result",
                            "arguments": '{"value":3}',
                        },
                    }
                ],
            },
            {"content": "Recovered after status error tool result."},
        ]
    )
    pattern = ReActPattern(max_iterations=3)
    context = ExecutionContext(execution_id="status-error-result")
    context.add_user_message("Recover")
    runtime = PatternRuntime(tracer=tracer)

    result = await pattern.run(
        context=context,
        tools=[StatusErrorResultTool()],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    tool_message = context.get_messages_by_role("tool")[0]
    assert tool_message.metadata["raw_result"]["status"] == "error"
    assert pattern.tool_ledger["call_status_error_result"].status == "failed"
    event_types = {event["event_type"] for event in tracer.events}
    assert "action_error_tool" in event_types


@pytest.mark.asyncio
async def test_react_pattern_mcp_is_error_result_is_failed_without_failure_code() -> (
    None
):
    tracer = TraceEventRecorder()
    llm = FakeLLM(
        responses=[
            {
                "content": "Use failing MCP tool.",
                "tool_calls": [
                    {
                        "id": "call_is_error_result",
                        "function": {
                            "name": "is_error_result",
                            "arguments": '{"value":3}',
                        },
                    }
                ],
            },
            {"content": "Recovered after MCP tool failure."},
        ]
    )
    pattern = ReActPattern(max_iterations=3, finalize_after_tool_result=True)
    context = ExecutionContext(execution_id="is-error-result")
    context.add_user_message("Recover")
    runtime = PatternRuntime(tracer=tracer)

    result = await pattern.run(
        context=context,
        tools=[IsErrorResultTool()],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    tool_message = context.get_messages_by_role("tool")[0]
    assert tool_message.metadata["raw_result"]["is_error"] is True
    assert pattern.tool_ledger["call_is_error_result"].status == "failed"
    assert llm.calls[1]["tools"] is not None
    failure_events = [
        event for event in tracer.events if event["event_type"] == "action_error_tool"
    ]
    assert len(failure_events) == 1
    assert "failure_code" not in failure_events[0]["data"]


@pytest.mark.asyncio
async def test_react_pattern_classified_failure_preserves_failure_code() -> None:
    """A classified delegated failure's ``failure_code`` survives end to end.

    ``AgentTool`` returns a classified failure dict (see
    ``_classified_failure`` in agent_tool.py) carrying a ``failure_code``.
    A real ``ReActPattern`` run must preserve it on all three parent-level
    surfaces: the tool ledger record, the tool message's ``raw_result``
    metadata, and the ``action_error_tool`` trace event's data.
    """
    tracer = TraceEventRecorder()
    llm = FakeLLM(
        responses=[
            {
                "content": "Use classified failure tool.",
                "tool_calls": [
                    {
                        "id": "call_classified",
                        "function": {
                            "name": "classified_failure_result",
                            "arguments": '{"value":3}',
                        },
                    }
                ],
            },
            {"content": "Recovered after classified failure."},
        ]
    )
    pattern = ReActPattern(max_iterations=3, finalize_after_tool_result=True)
    context = ExecutionContext(execution_id="classified-failure-result")
    context.add_user_message("Recover")
    runtime = PatternRuntime(tracer=tracer)

    result = await pattern.run(
        context=context,
        tools=[ClassifiedFailureResultTool()],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert pattern.tool_ledger["call_classified"].status == "failed"
    assert (
        pattern.tool_ledger["call_classified"].result["failure_code"]
        == "unsupported_nested_interaction"
    )

    tool_message = context.get_messages_by_role("tool")[0]
    assert (
        tool_message.metadata["raw_result"]["failure_code"]
        == "unsupported_nested_interaction"
    )

    failure_events = [
        event for event in tracer.events if event["event_type"] == "action_error_tool"
    ]
    assert len(failure_events) == 1
    assert failure_events[0]["data"]["failure_code"] == "unsupported_nested_interaction"


@pytest.mark.asyncio
async def test_react_pattern_generates_new_step_id_per_run() -> None:
    tracer = TraceEventRecorder()
    runtime = PatternRuntime(tracer=tracer)
    context = ExecutionContext(execution_id="multi-react")
    context.add_user_message("First")

    first = ReActPattern(max_iterations=1)
    first_result = await first.run(
        context=context,
        tools=[],
        llm=FakeLLM([{"content": "First done."}]),
        runtime=runtime,
    )

    context.add_user_message("Second")
    second = ReActPattern(max_iterations=1)
    second_result = await second.run(
        context=context,
        tools=[],
        llm=FakeLLM([{"content": "Second done."}]),
        runtime=runtime,
    )

    assert first_result["success"] is True
    assert second_result["success"] is True
    react_start_step_ids = [
        event["step_id"]
        for event in tracer.events
        if event["event_type"] == "task_start_react"
    ]
    assert len(react_start_step_ids) == 2
    assert all(step_id.startswith("react_") for step_id in react_start_step_ids)
    assert react_start_step_ids[0] != react_start_step_ids[1]


@pytest.mark.asyncio
async def test_react_pattern_traces_context_compaction() -> None:
    tracer = TraceEventRecorder()
    runtime = PatternRuntime(tracer=tracer)
    context = ExecutionContext(execution_id="compact-react")
    context.compact_config.threshold = 1
    for index in range(3):
        context.add_user_message(f"message {index}")

    result = await ReActPattern(max_iterations=1).run(
        context=context,
        tools=[],
        llm=FakeLLM([{"content": "done"}]),
        runtime=runtime,
    )

    assert result["success"] is True
    compact_events = [
        event for event in tracer.events if event["event_type"].endswith("_compact")
    ]
    assert [event["event_type"] for event in compact_events] == [
        "action_start_compact",
        "action_end_compact",
    ]
    assert compact_events[0]["step_id"].startswith("react_")
    assert compact_events[1]["data"]["success"] is True
    assert compact_events[1]["data"]["compact_type"] == "execution_context"


@pytest.mark.asyncio
async def test_react_pattern_uses_compact_llm_for_context_compaction() -> None:
    tracer = TraceEventRecorder()
    runtime = PatternRuntime(tracer=tracer)
    context = ExecutionContext(execution_id="compact-react-llm")
    context.compact_config.threshold = 1
    context.add_user_message("current request")
    context.add_assistant_message(
        "",
        tool_calls=[
            {"id": "call-1", "type": "function", "function": {"name": "read_file"}},
        ],
    )
    context.add_tool_result("read_file", {"output": "x" * 200}, tool_call_id="call-1")
    llm = FakeLLM([{"content": "done"}])
    compact_llm = FakeLLM(
        [
            {
                "content": "summarized tool result",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        ]
    )

    result = await ReActPattern(max_iterations=1).run(
        context=context,
        tools=[],
        llm=llm,
        compact_llm=compact_llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert len(compact_llm.calls) == 1
    assert compact_llm.calls[0]["max_tokens"] == 256
    assert "Preserve the language" in compact_llm.calls[0]["messages"][0]["content"]
    assert len(llm.calls) == 1
    assert any(
        "summarized tool result" in message["content"]
        for message in llm.calls[0]["messages"]
    )
    compact_end = next(
        event for event in tracer.events if event["event_type"] == "action_end_compact"
    )
    assert compact_end["data"]["strategy"] == "llm_summary"
    compact_llm_events = [
        event
        for event in tracer.events
        if event["event_type"] in {"action_start_llm", "action_end_llm"}
        and event["data"].get("purpose") == "context_compaction"
    ]
    assert [event["event_type"] for event in compact_llm_events] == [
        "action_start_llm",
        "action_end_llm",
    ]
    assert compact_llm_events[1]["data"]["input_tokens"] == 10
    assert context.get_total_token_usage()["total"] == 15


@pytest.mark.asyncio
async def test_react_pattern_emits_runtime_checkpoints() -> None:
    llm = FakeLLM(
        responses=[
            {
                "content": "I should calculate this first.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"3+3"}',
                        },
                    }
                ],
                "done": False,
            },
            {"content": "The result is 6.", "done": True},
        ]
    )
    pattern = ReActPattern(max_iterations=3)
    context = ExecutionContext()
    context.add_user_message("Calculate 3+3")
    runtime = PatternRuntime()

    result = await pattern.run(
        context=context,
        tools=[FakeTool()],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert [checkpoint["label"] for checkpoint in runtime.checkpoints] == [
        "before_llm",
        "after_llm",
        "before_tool",
        "after_tool",
        "before_llm",
        "after_llm",
        "final",
    ]
    after_llm = runtime.checkpoints[1]
    assert after_llm["pattern_state"]["pending_tool_calls"] == [
        {"id": "call_1", "name": "calculator", "args": {"expression": "3+3"}}
    ]
    assert after_llm["context"]["messages"][1]["tool_calls"][0]["id"] == "call_1"


@pytest.mark.asyncio
async def test_react_pattern_runtime_emits_tracer_task_and_tool_events() -> None:
    tracer = FakeTracer()
    llm = FakeLLM(
        responses=[
            {
                "content": "Need a tool.",
                "tool_calls": [
                    {
                        "id": "call_trace",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"3*3"}',
                        },
                    }
                ],
                "done": False,
            },
            {"content": "The result is 9.", "done": True},
        ]
    )
    pattern = ReActPattern(max_iterations=3)
    context = ExecutionContext()
    context.add_user_message("Calculate 3*3")
    runtime = PatternRuntime(tracer=tracer)

    result = await pattern.run(
        context=context,
        tools=[FakeTool()],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert [event for event, _ in tracer.events] == [
        "start_trace",
        "start_span",
        "finish_span",
        "finish_trace",
    ]


@pytest.mark.asyncio
async def test_react_pattern_pause_cancels_active_llm_call() -> None:
    llm = BlockingLLM()
    runtime = PatternRuntime()
    pattern = ReActPattern(max_iterations=3)
    context = ExecutionContext()
    context.add_user_message("Wait on model")

    task = asyncio.create_task(
        pattern.run(
            context=context,
            tools=[],
            llm=llm,
            runtime=runtime,
        )
    )
    await asyncio.wait_for(llm.started.wait(), timeout=1)

    runtime.request_interrupt("paused by test")
    result = await asyncio.wait_for(task, timeout=1)

    assert llm.cancelled is True
    assert result["success"] is False
    assert result["status"] == "interrupted"
    assert result["interrupt_reason"] == "paused by test"
    assert runtime.last_checkpoint["label"] == "interrupted"
    assert runtime.last_checkpoint["metadata"] == {
        "safe_point": "during_llm",
        "reason": "paused by test",
    }


@pytest.mark.asyncio
async def test_react_pattern_pause_during_tool_protocol_retry_is_during_llm() -> None:
    llm = BlockingProtocolRetryLLM()
    runtime = PatternRuntime()
    pattern = ReActPattern(max_iterations=3)
    context = ExecutionContext()
    context.add_user_message("Retry the model")

    task = asyncio.create_task(
        pattern.run(
            context=context,
            tools=[FakeTool()],
            llm=llm,
            runtime=runtime,
        )
    )
    await asyncio.wait_for(llm.retry_started.wait(), timeout=1)

    runtime.request_interrupt("paused during retry")
    result = await asyncio.wait_for(task, timeout=1)

    assert llm.cancelled is True
    assert result["success"] is False
    assert result["status"] == "interrupted"
    assert result["interrupt_reason"] == "paused during retry"
    assert runtime.last_checkpoint["label"] == "interrupted"
    assert runtime.last_checkpoint["metadata"] == {
        "safe_point": "during_llm",
        "reason": "paused during retry",
    }


@pytest.mark.asyncio
async def test_react_pattern_interrupts_at_tool_boundary() -> None:
    llm = FakeLLM(
        responses=[
            {
                "content": "I should calculate this first.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"4+4"}',
                        },
                    }
                ],
                "done": False,
            },
        ]
    )
    runtime = PatternRuntime()
    runtime.interrupt_checker = lambda: len(runtime.checkpoints) >= 2
    pattern = ReActPattern(max_iterations=3)
    tool = FakeTool()
    context = ExecutionContext()
    context.add_user_message("Calculate 4+4")

    result = await pattern.run(
        context=context,
        tools=[tool],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is False
    assert result["status"] == "interrupted"
    assert tool.calls == []
    assert runtime.last_checkpoint["label"] == "interrupted"
    assert pattern.pending_tool_calls == [
        {"id": "call_1", "name": "calculator", "args": {"expression": "4+4"}}
    ]


@pytest.mark.asyncio
async def test_react_pattern_resumes_pending_tool_call_from_checkpoint() -> None:
    first_llm = FakeLLM(
        responses=[
            {
                "content": "I should calculate this first.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"4+4"}',
                        },
                    }
                ],
                "done": False,
            },
        ]
    )
    first_runtime = PatternRuntime()
    first_runtime.interrupt_checker = lambda: len(first_runtime.checkpoints) >= 2
    first_pattern = ReActPattern(max_iterations=3)
    first_context = ExecutionContext()
    first_context.add_user_message("Calculate 4+4")

    interrupted = await first_pattern.run(
        context=first_context,
        tools=[FakeTool()],
        llm=first_llm,
        runtime=first_runtime,
    )
    checkpoint = first_runtime.last_checkpoint

    assert interrupted["status"] == "interrupted"
    assert checkpoint is not None

    restored_context = ExecutionContext.from_dict(checkpoint["context"])
    restored_pattern = ReActPattern(max_iterations=3)
    restored_pattern.load_state(checkpoint["pattern_state"])
    restored_tool = FakeTool()
    restored_runtime = PatternRuntime()

    result = await restored_pattern.run(
        context=restored_context,
        tools=[restored_tool],
        llm=FakeLLM([{"content": "The result is 8.", "done": True}]),
        runtime=restored_runtime,
    )

    assert result["success"] is True
    assert result["response"] == "The result is 8."
    assert restored_tool.calls == [{"expression": "4+4"}]
    assert [message.role for message in restored_context.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


def test_react_pattern_state_roundtrip() -> None:
    pattern = ReActPattern(
        max_iterations=5,
        repeated_tool_decision_after_consecutive_work_tool_calls=7,
    )
    pattern.status = "acting"
    pattern.current_iteration = 2
    pattern.task_text = "Original task"
    pattern.pending_tool_calls = [{"id": "call_1", "name": "calculator", "args": {}}]
    pattern._record_tool_call(
        {"id": "call_1", "name": "calculator", "args": {"expression": "1+1"}},
        status="completed",
        result={"result": 2},
    )

    restored = ReActPattern()
    restored.load_state(pattern.get_state())

    assert restored.status == "acting"
    assert restored.current_iteration == 2
    assert restored.max_iterations == 5
    assert restored.repeated_tool_decision_after_consecutive_work_tool_calls == 7
    assert restored.task_text == "Original task"
    assert restored.reasoning_mode == ReActReasoningMode.TOOL_CALLING
    assert restored.tool_ledger["call_1"].result == {"result": 2}


def test_react_pattern_state_roundtrip_preserves_disabled_decision_thresholds() -> None:
    pattern = ReActPattern(
        repeated_tool_decision_after_consecutive_tool_calls=None,
        repeated_tool_decision_after_consecutive_work_tool_calls=None,
    )

    restored = ReActPattern()
    restored.load_state(pattern.get_state())

    assert restored.repeated_tool_decision_after_consecutive_tool_calls is None
    assert restored.repeated_tool_decision_after_consecutive_work_tool_calls is None


def test_tool_call_record_from_dict_handles_null_args() -> None:
    record = ToolCallRecord.from_dict(
        {
            "tool_call_id": "call_1",
            "tool_name": "calculator",
            "args": None,
        }
    )

    assert record.args == {}
    assert record.args_hash == ""
    assert record.status == "pending"


def test_args_hash_is_stable_sha256_digest() -> None:
    pattern = ReActPattern()

    digest = pattern._args_hash({"b": 2, "a": 1})

    assert len(digest) == 64
    assert all(char in "0123456789abcdef" for char in digest)
    # Key order must not affect the digest.
    assert digest == pattern._args_hash({"a": 1, "b": 2})
    assert digest != pattern._args_hash({"a": 1, "b": 3})


def test_args_hash_survives_circular_reference_args() -> None:
    pattern = ReActPattern()
    circular: dict = {"query": "x"}
    circular["self"] = circular

    digest = pattern._args_hash(circular)

    assert len(digest) == 64
    assert all(char in "0123456789abcdef" for char in digest)


@pytest.mark.asyncio
async def test_react_pattern_reasoning_action_mode_is_explicit_placeholder() -> None:
    pattern = ReActPattern(reasoning_mode=ReActReasoningMode.REASONING_ACTION)
    context = ExecutionContext()
    context.add_user_message("Think")

    result = await pattern.run(context=context, tools=[], llm=FakeLLM([]))

    assert result["success"] is False
    assert result["status"] == "failed"
    assert "reserved for a future implementation" in result["error"]
    assert result["reasoning_mode"] == ReActReasoningMode.REASONING_ACTION.value
    assert result["error_type"] == "not_implemented"


@pytest.mark.asyncio
async def test_react_pattern_runtime_injected_from_runner_style_traces_events() -> None:
    tracer = FakeTracer()
    llm = FakeLLM(
        responses=[
            {
                "content": "Need a tool.",
                "tool_calls": [
                    {
                        "id": "call_trace",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"3*3"}',
                        },
                    }
                ],
                "done": False,
            },
            {"content": "The result is 9.", "done": True},
        ]
    )
    pattern = ReActPattern(max_iterations=3)
    context = ExecutionContext()
    context.add_user_message("Calculate 3*3")
    runtime = PatternRuntime(tracer=tracer)

    result = await pattern.run(
        context=context,
        tools=[FakeTool()],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert [event for event, _ in tracer.events] == [
        "start_trace",
        "start_span",
        "finish_span",
        "finish_trace",
    ]


class MemoryToolStore:
    """Fake memory store with both search (retrieval/dedup) and add (store)."""

    def __init__(self) -> None:
        self.searches: list[dict[str, Any]] = []
        self.added: list[Any] = []

    def search(self, **kwargs: Any) -> list[Any]:
        self.searches.append(kwargs)
        return []

    def add(self, note: Any) -> Any:
        self.added.append(note)
        return SimpleNamespace(success=True, memory_id=f"mem-{len(self.added)}")


def _tool_names_from_llm_call(call: dict[str, Any]) -> list[str]:
    return [tool["function"]["name"] for tool in list(call.get("tools") or [])]


def test_find_tool_accepts_a_compatible_historical_name() -> None:
    class RenamedTool:
        name = "agent_research_assistant__a42"

        @staticmethod
        def matches_name(name: str) -> bool:
            return name in {"agent_42", "agent_old_name__a42"}

    tool = RenamedTool()
    pattern = ReActPattern()

    assert pattern._find_tool("agent_42", [tool]) is tool
    assert pattern._find_tool("agent_old_name__a42", [tool]) is tool


@pytest.mark.asyncio
async def test_react_pattern_exposes_store_memory_tool_with_memory_store() -> None:
    llm = FakeLLM(
        responses=[
            {
                "content": "Worth remembering.",
                "tool_calls": [
                    {
                        "id": "call_mem",
                        "function": {
                            "name": "store_memory",
                            "arguments": (
                                '{"content":"User prefers reports in Chinese.",'
                                '"kind":"user_preference"}'
                            ),
                        },
                    }
                ],
                "done": False,
            },
            {"content": "Done.", "done": True},
        ]
    )
    memory_store = MemoryToolStore()
    pattern = ReActPattern(max_iterations=3)
    context = ExecutionContext()
    context.add_user_message("Write the weekly report")

    result = await pattern.run(
        context=context,
        tools=[],
        llm=llm,
        memory_store=memory_store,
    )

    assert result["success"] is True
    tool_names = _tool_names_from_llm_call(llm.calls[0])
    assert "store_memory" in tool_names
    assert "search_memory" in tool_names
    assert "update_memory" in tool_names
    assert "delete_memory" in tool_names
    assert len(memory_store.added) == 1
    assert memory_store.added[0].content == "User prefers reports in Chinese."
    # No end-of-run memory-evaluation LLM call: both fake responses are
    # consumed by the ReAct loop itself.
    assert llm.responses == []
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_react_pattern_without_memory_store_has_no_store_memory_tool() -> None:
    llm = FakeLLM(responses=[{"content": "Done.", "done": True}])
    pattern = ReActPattern(max_iterations=1, tool_choice="none")
    context = ExecutionContext()
    context.add_user_message("Do the thing")

    result = await pattern.run(context=context, tools=[], llm=llm)

    assert result["success"] is True
    assert "store_memory" not in _tool_names_from_llm_call(llm.calls[0])


@pytest.mark.asyncio
async def test_react_pattern_skips_store_memory_tool_in_single_call_mode() -> None:
    llm = FakeLLM(responses=[{"content": "Done.", "done": True}])
    memory_store = MemoryToolStore()
    pattern = ReActPattern(max_iterations=2, finalize_after_tool_result=True)
    context = ExecutionContext()
    context.add_user_message("Quick question")

    result = await pattern.run(
        context=context,
        tools=[],
        llm=llm,
        memory_store=memory_store,
    )

    assert result["success"] is True
    assert "store_memory" not in _tool_names_from_llm_call(llm.calls[0])
    assert memory_store.added == []


@pytest.mark.asyncio
async def test_tool_result_user_interaction_fails_closed_when_disabled() -> None:
    class WaitingTool:
        metadata = SimpleNamespace(
            name="approval_gate",
            description="Request approval.",
        )

        def args_type(self) -> type[BaseModel]:
            return CalculatorArgs

        async def run_json_async(self, args: dict[str, Any]) -> dict[str, Any]:
            return {
                "success": False,
                "status": "waiting_for_user",
                "message": "Approve this action?",
            }

    pattern = ReActPattern(max_iterations=2, user_interaction_enabled=False)
    runtime = PatternRuntime(execution_id="unattended-task")
    context = ExecutionContext(execution_id="unattended-task")
    context.add_user_message("Run unattended.")

    result = await pattern.run(
        context=context,
        tools=[WaitingTool()],
        llm=FakeLLM(
            [
                {
                    "tool_calls": [
                        {
                            "id": "wait-call",
                            "function": {
                                "name": "approval_gate",
                                "arguments": '{"expression":"2+2"}',
                            },
                        }
                    ]
                }
            ]
        ),
        runtime=runtime,
    )

    assert result["success"] is False
    assert result["status"] == "failed"
    assert "interaction is disabled" in result["error"]
    assert runtime.outbound_messages == []
    assert pattern.status == "failed"
    assert pattern.waiting_for_user_request is None


@pytest.mark.asyncio
async def test_tool_result_user_interaction_disabled_preserves_the_tool_message() -> (
    None
):
    """The generic 'interaction is disabled' framing alone throws away the
    only actionable part of the failure - e.g. UnavailableMCPTool naming the
    specific app that needs reconnecting. A context that can't honor an
    interactive pause must still relay what the tool was actually asking,
    the same way a plain error result always could pre-pause-support."""

    class UnavailableConnectorTool:
        metadata = SimpleNamespace(
            name="mcp_gmail",
            description="Send email.",
        )

        def args_type(self) -> type[BaseModel]:
            return CalculatorArgs

        async def run_json_async(self, args: dict[str, Any]) -> dict[str, Any]:
            return {
                "success": False,
                "status": "waiting_for_user",
                "message": "I need access to Gmail to continue.",
                "interactions": [
                    {"type": "connect_apps", "field": "connect_apps", "apps": ["Gmail"]}
                ],
            }

    pattern = ReActPattern(max_iterations=2, user_interaction_enabled=False)
    runtime = PatternRuntime(execution_id="unattended-task-2")
    context = ExecutionContext(execution_id="unattended-task-2")
    context.add_user_message("Run unattended.")

    result = await pattern.run(
        context=context,
        tools=[UnavailableConnectorTool()],
        llm=FakeLLM(
            [
                {
                    "tool_calls": [
                        {
                            "id": "wait-call",
                            "function": {"name": "mcp_gmail", "arguments": "{}"},
                        }
                    ]
                }
            ]
        ),
        runtime=runtime,
    )

    assert result["success"] is False
    assert result["status"] == "failed"
    assert "interaction is disabled" in result["error"]
    assert "I need access to Gmail to continue." in result["error"]


@pytest.mark.asyncio
async def test_tool_result_user_interaction_disabled_caps_an_unbounded_message() -> (
    None
):
    """A misbehaving or malicious tool could return an arbitrarily long
    message - it must be capped, not flow uncapped into this failure text."""

    class UnboundedMessageTool:
        metadata = SimpleNamespace(name="mcp_gmail", description="Send email.")

        def args_type(self) -> type[BaseModel]:
            return CalculatorArgs

        async def run_json_async(self, args: dict[str, Any]) -> dict[str, Any]:
            return {
                "success": False,
                "status": "waiting_for_user",
                "message": "x" * 5000,
            }

    pattern = ReActPattern(max_iterations=2, user_interaction_enabled=False)
    runtime = PatternRuntime(execution_id="unattended-task-3")
    context = ExecutionContext(execution_id="unattended-task-3")
    context.add_user_message("Run unattended.")

    result = await pattern.run(
        context=context,
        tools=[UnboundedMessageTool()],
        llm=FakeLLM(
            [
                {
                    "tool_calls": [
                        {
                            "id": "wait-call",
                            "function": {"name": "mcp_gmail", "arguments": "{}"},
                        }
                    ]
                }
            ]
        ),
        runtime=runtime,
    )

    assert "x" * 2000 in result["error"]
    assert "x" * 2001 not in result["error"]


@pytest.mark.asyncio
async def test_tool_result_can_pause_and_resume_with_user_response() -> None:
    class ResumableTool:
        def __init__(self) -> None:
            self.metadata = SimpleNamespace(
                name="approval_gate",
                description="Run an action after the user responds.",
            )
            self.resume_calls: list[dict[str, str]] = []
            self.user_response: str | None = None

        def args_type(self) -> type[BaseModel]:
            return CalculatorArgs

        def resume_user_interaction(
            self,
            *,
            interaction_id: str,
            response: str,
        ) -> None:
            self.resume_calls.append(
                {"interaction_id": interaction_id, "response": response}
            )
            self.user_response = response

        async def run_json_async(self, args: dict[str, Any]) -> Any:
            if self.user_response is None:
                return {
                    "success": False,
                    "status": "waiting_for_user",
                    "interaction_id": "interaction-1",
                    "message": "Should the action continue?",
                    "message_type": "confirmation",
                    "interactions": [
                        {
                            "type": "select_one",
                            "field": "decision",
                            "label": "Decision",
                            "options": [
                                {"label": "Continue", "value": "continue"},
                                {"label": "Stop", "value": "stop"},
                            ],
                        }
                    ],
                }
            response = self.user_response
            self.user_response = None
            return {
                "success": True,
                "expression": args["expression"],
                "user_response": response,
            }

    class MutationTool:
        def __init__(self) -> None:
            self.metadata = SimpleNamespace(
                name="mutation",
                description="Mutate state.",
            )
            self.calls: list[dict[str, Any]] = []

        def args_type(self) -> type[BaseModel]:
            return CalculatorArgs

        async def run_json_async(self, args: dict[str, Any]) -> Any:
            self.calls.append(args)
            return {"success": True}

    first_tool = ResumableTool()
    first_mutation = MutationTool()
    context = ExecutionContext(execution_id="interaction-task")
    context.add_user_message("Run the gated action.")
    tracer = TraceEventRecorder()
    runtime = PatternRuntime(execution_id="interaction-task", tracer=tracer)
    pattern = ReActPattern(max_iterations=4)
    waiting = await pattern.run(
        context=context,
        tools=[first_tool, first_mutation],
        llm=FakeLLM(
            [
                {
                    "tool_calls": [
                        {
                            "id": "wait-call",
                            "function": {
                                "name": "approval_gate",
                                "arguments": '{"expression":"2+2"}',
                            },
                        },
                        {
                            "id": "stale-mutation",
                            "function": {
                                "name": "mutation",
                                "arguments": '{"expression":"99"}',
                            },
                        },
                    ]
                }
            ]
        ),
        runtime=runtime,
    )

    assert waiting["status"] == "waiting_for_user"
    assert waiting["message"] == "Should the action continue?"
    assert pattern.tool_ledger["wait-call"].status == "waiting_for_user"
    assert pattern.tool_ledger["stale-mutation"].status == "cancelled"
    assert pattern.pending_tool_calls == []
    assert first_mutation.calls == []
    assert runtime.outbound_messages[0]["expect_response"] is True
    assert any(
        event["event_type"] == "action_end_tool"
        and event["data"]["status"] == "waiting_for_user"
        for event in tracer.events
    )
    assert not any(
        event["event_type"] == "action_error_tool" for event in tracer.events
    )

    context.add_user_message("Continue")
    resumed_tool = ResumableTool()
    resumed_mutation = MutationTool()
    resumed_pattern = ReActPattern(max_iterations=4)
    resumed_pattern.load_state(pattern.get_state())
    resumed = await resumed_pattern.run(
        context=context,
        tools=[resumed_tool, resumed_mutation],
        llm=FakeLLM(
            [
                {
                    "tool_calls": [
                        {
                            "id": "resume-call",
                            "function": {
                                "name": "approval_gate",
                                "arguments": '{"expression":"2+2"}',
                            },
                        }
                    ]
                },
                {
                    "tool_calls": [
                        {
                            "id": "final-call",
                            "function": {
                                "name": "final_answer",
                                "arguments": (
                                    '{"response_language":"English",'
                                    '"answer":"The action completed.",'
                                    '"outcome":"completed"}'
                                ),
                            },
                        }
                    ]
                },
            ]
        ),
    )

    assert resumed["success"] is True
    assert resumed["completion_outcome"] == "completed"
    assert resumed_tool.resume_calls == [
        {"interaction_id": "interaction-1", "response": "Continue"}
    ]
    assert resumed_mutation.calls == []
    assert resumed_pattern.pending_tool_interaction_responses == []
    response_metadata = next(
        message.metadata
        for message in context.messages
        if message.role == "user" and message.content == "Continue"
    )
    assert response_metadata["response_to_waiting_for_user"]["question"] == (
        "Should the action continue?"
    )


@pytest.mark.parametrize("waiting_request", [None, [], "malformed"])
def test_tool_interaction_response_queue_ignores_malformed_requests(
    waiting_request: Any,
) -> None:
    pattern = ReActPattern()

    pattern._queue_tool_interaction_responses(
        waiting_request=waiting_request,
        response="Continue",
        tools=[],
    )

    assert pattern.pending_tool_interaction_responses == []


@pytest.mark.asyncio
async def test_callback_less_tool_interaction_resumes_by_replanning() -> None:
    class CallbackLessTool:
        def __init__(self) -> None:
            self.metadata = SimpleNamespace(
                name="clarification_gate",
                description="Ask for clarification without retaining server state.",
            )

        def args_type(self) -> type[BaseModel]:
            return CalculatorArgs

        async def run_json_async(self, args: dict[str, Any]) -> dict[str, Any]:
            return {
                "success": False,
                "status": "waiting_for_user",
                "interaction_id": "interaction-1",
                "message": "Which value should be used?",
                "message_type": "question",
                "interactions": [
                    {
                        "type": "text_input",
                        "field": "value",
                        "label": "Value",
                    }
                ],
            }

    tool = CallbackLessTool()
    context = ExecutionContext(execution_id="callback-less-interaction")
    context.add_user_message("Use the requested value.")
    pattern = ReActPattern(max_iterations=2)
    waiting = await pattern.run(
        context=context,
        tools=[tool],
        llm=FakeLLM(
            [
                {
                    "tool_calls": [
                        {
                            "id": "wait-call",
                            "function": {
                                "name": "clarification_gate",
                                "arguments": '{"expression":"2+2"}',
                            },
                        }
                    ]
                }
            ]
        ),
    )

    assert waiting["status"] == "waiting_for_user"

    context.add_user_message("Use 42")
    resumed_pattern = ReActPattern(max_iterations=2)
    resumed_pattern.load_state(pattern.get_state())
    resumed = await resumed_pattern.run(
        context=context,
        tools=[tool],
        llm=FakeLLM(
            [
                {
                    "tool_calls": [
                        {
                            "id": "final-call",
                            "function": {
                                "name": "final_answer",
                                "arguments": (
                                    '{"response_language":"English",'
                                    '"answer":"Using 42.",'
                                    '"outcome":"completed"}'
                                ),
                            },
                        }
                    ]
                }
            ]
        ),
    )

    assert resumed["success"] is True
    assert resumed["completion_outcome"] == "completed"
    assert resumed_pattern.pending_tool_interaction_responses == []
    response_message = next(
        message
        for message in context.messages
        if message.role == "user" and message.content == "Use 42"
    )
    waiting_metadata = response_message.metadata["response_to_waiting_for_user"]
    assert waiting_metadata["requests"][0]["tool_name"] == "clarification_gate"


@pytest.mark.asyncio
async def test_pending_interaction_delivery_is_exact_and_retryable() -> None:
    class ResumableTool:
        def __init__(self) -> None:
            self.metadata = SimpleNamespace(
                name="approval_gate",
                description="Resume exact interactions.",
            )
            self.calls: list[dict[str, str]] = []
            self.fail_interaction_id = "interaction-2"

        def resume_user_interaction(
            self,
            *,
            interaction_id: str,
            response: str,
        ) -> None:
            self.calls.append({"interaction_id": interaction_id, "response": response})
            if interaction_id == self.fail_interaction_id:
                raise RuntimeError("delivery failed")

    pending = [
        {
            "tool_name": "approval_gate",
            "tool_call_id": "call-1",
            "interaction_id": "interaction-1",
            "response": "Approve first",
        },
        {
            "tool_name": "approval_gate",
            "tool_call_id": "call-2",
            "interaction_id": "interaction-2",
            "response": "Reject second",
        },
    ]
    pattern = ReActPattern()
    pattern.pending_tool_interaction_responses = list(pending)
    tool = ResumableTool()
    context = ExecutionContext(execution_id="interaction-delivery")
    runtime = PatternRuntime(execution_id="interaction-delivery")

    with pytest.raises(RuntimeError, match="delivery failed"):
        await pattern._deliver_pending_tool_interaction_responses(
            tools=[tool],
            context=context,
            runtime=runtime,
        )

    assert tool.calls == [
        {"interaction_id": "interaction-1", "response": "Approve first"},
        {"interaction_id": "interaction-2", "response": "Reject second"},
    ]
    assert pattern.pending_tool_interaction_responses == [pending[1]]

    tool.fail_interaction_id = ""
    await pattern._deliver_pending_tool_interaction_responses(
        tools=[tool],
        context=context,
        runtime=runtime,
    )

    assert tool.calls[-1] == {
        "interaction_id": "interaction-2",
        "response": "Reject second",
    }
    assert pattern.pending_tool_interaction_responses == []


@pytest.mark.asyncio
async def test_pending_interaction_without_resume_callback_is_skipped() -> None:
    class CallbackLessTool:
        metadata = SimpleNamespace(
            name="clarification_gate",
            description="No resume callback.",
        )

    pending = {
        "tool_name": "clarification_gate",
        "tool_call_id": "call-1",
        "interaction_id": "interaction-1",
        "response": "Use 42",
    }
    pattern = ReActPattern()
    pattern.pending_tool_interaction_responses = [pending]
    context = ExecutionContext(execution_id="callback-less-delivery")
    runtime = PatternRuntime(execution_id="callback-less-delivery")

    await pattern._deliver_pending_tool_interaction_responses(
        tools=[CallbackLessTool()],
        context=context,
        runtime=runtime,
    )

    assert pattern.pending_tool_interaction_responses == []


@pytest.mark.asyncio
async def test_concurrent_tool_interactions_pause_in_one_deterministic_message() -> (
    None
):
    class ConcurrentWaitingTool:
        def __init__(self, name: str, message: str) -> None:
            self.metadata = SimpleNamespace(
                name=name,
                description=f"Wait for {name} input.",
                concurrency_safe=True,
            )
            self.message = message

        def args_type(self) -> type[BaseModel]:
            return CalculatorArgs

        async def run_json_async(self, args: dict[str, Any]) -> Any:
            return {
                "success": False,
                "status": "waiting_for_user",
                "interaction_id": f"{self.metadata.name}-interaction",
                "message": self.message,
                "interactions": [
                    {
                        "type": "text",
                        "field": "decision",
                        "label": self.metadata.name,
                    }
                ],
            }

    class MutationTool:
        def __init__(self) -> None:
            self.metadata = SimpleNamespace(
                name="mutation",
                description="Mutate state.",
                concurrency_safe=False,
            )
            self.calls: list[dict[str, Any]] = []

        def args_type(self) -> type[BaseModel]:
            return CalculatorArgs

        async def run_json_async(self, args: dict[str, Any]) -> Any:
            self.calls.append(args)
            return {"success": True}

    mutation_tool = MutationTool()
    tools = [
        ConcurrentWaitingTool("first_gate", "Answer the first question."),
        ConcurrentWaitingTool("second_gate", "Answer the second question."),
        mutation_tool,
    ]
    pattern = ReActPattern(
        max_iterations=2,
        tool_parallel_enabled=True,
        tool_max_concurrency=2,
    )
    runtime = PatternRuntime(execution_id="concurrent-interactions")
    context = ExecutionContext(execution_id="concurrent-interactions")
    context.add_user_message("Run both checks.")

    result = await pattern.run(
        context=context,
        tools=tools,
        llm=FakeLLM(
            [
                {
                    "tool_calls": [
                        {
                            "id": "first-call",
                            "function": {
                                "name": "first_gate",
                                "arguments": '{"expression":"1"}',
                            },
                        },
                        {
                            "id": "second-call",
                            "function": {
                                "name": "second_gate",
                                "arguments": '{"expression":"2"}',
                            },
                        },
                        {
                            "id": "stale-mutation",
                            "function": {
                                "name": "mutation",
                                "arguments": '{"expression":"3"}',
                            },
                        },
                    ]
                }
            ]
        ),
        runtime=runtime,
    )

    assert result["status"] == "waiting_for_user"
    assert result["message"].startswith("Multiple tools need your input")
    assert [item["field"] for item in result["interactions"]] == [
        "decision",
        "decision_2",
    ]
    assert [
        request["interactions"][0]["field"]
        for request in pattern.waiting_for_user_request["requests"]
    ] == ["decision", "decision_2"]
    assert len(runtime.outbound_messages) == 1
    assert pattern.tool_ledger["first-call"].status == "waiting_for_user"
    assert pattern.tool_ledger["second-call"].status == "waiting_for_user"
    assert pattern.tool_ledger["stale-mutation"].status == "cancelled"
    assert pattern.pending_tool_calls == []
    assert mutation_tool.calls == []


@pytest.mark.asyncio
async def test_final_answer_preserves_semantic_completion_outcome() -> None:
    pattern = ReActPattern(max_iterations=1)
    context = ExecutionContext()
    context.add_user_message("Complete everything if possible.")

    result = await pattern.run(
        context=context,
        tools=[],
        llm=FakeLLM(
            [
                {
                    "tool_calls": [
                        {
                            "id": "final-partial",
                            "function": {
                                "name": "final_answer",
                                "arguments": (
                                    '{"response_language":"English",'
                                    '"answer":"One item remains.",'
                                    '"outcome":"partial"}'
                                ),
                            },
                        }
                    ]
                }
            ]
        ),
    )

    assert result["success"] is True
    assert result["status"] == "completed"
    assert result["completion_outcome"] == "partial"


class StreamingEmptyFinalAnswerLLM:
    """Calls ``final_answer`` with an unusable answer, then optionally recovers.

    Models the two observed shapes of xorbitsai/xagent#1312: an arguments object
    that omits ``answer`` entirely (truncated output), and an arguments payload
    that fails JSON parsing.
    """

    RECOVERED_ARGUMENTS = (
        '{"response_language":"English","answer":"The result is 4.",'
        '"outcome":"completed"}'
    )

    def __init__(
        self,
        *,
        recover: bool = True,
        broken_arguments: str = '{"response_language":"English","outcome":"completed"}',
        preamble: str = "",
        trailing_work_tool: bool = False,
    ) -> None:
        self.recover = recover
        self.broken_arguments = broken_arguments
        self.preamble = preamble
        self.trailing_work_tool = trailing_work_tool
        self.stream_calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        raise AssertionError("empty final answer path should stay streaming")

    async def stream_chat(
        self, messages: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> Any:
        if messages is not None:
            kwargs["messages"] = messages
        self.stream_calls.append(kwargs)
        call_index = len(self.stream_calls) - 1
        arguments = (
            self.RECOVERED_ARGUMENTS
            if call_index > 0 and self.recover
            else self.broken_arguments
        )
        if self.preamble:
            yield StreamChunk(type=ChunkType.TOKEN, delta=self.preamble)
        tool_calls = [
            {
                "id": f"call_final_{call_index}",
                "function": {"name": "final_answer", "arguments": arguments},
            }
        ]
        if call_index == 0 and self.trailing_work_tool:
            tool_calls.append(
                {
                    "id": "call_work_0",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression":"2+2"}',
                    },
                }
            )
        yield StreamChunk(
            type=ChunkType.TOOL_CALL,
            tool_calls=tool_calls,
        )
        yield StreamChunk(type=ChunkType.END)


def _react_empty_final_answer_fixture() -> tuple[
    ReActPattern,
    ExecutionContext,
    PatternRuntime,
    OutboundCollector,
    TraceEventRecorder,
]:
    pattern = ReActPattern(max_iterations=3)
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="task-1")
    context.add_user_message("What is 2+2?")
    outbound = OutboundCollector()
    tracer = TraceEventRecorder()
    runtime = PatternRuntime(
        execution_id="task-1",
        tracer=tracer,
        outbound_message_handler=outbound,
    )
    return pattern, context, runtime, outbound, tracer


@pytest.mark.asyncio
async def test_react_retries_final_answer_that_omits_the_answer_field() -> None:
    llm = StreamingEmptyFinalAnswerLLM()
    pattern, context, runtime, outbound, tracer = _react_empty_final_answer_fixture()

    result = await pattern.run(
        context=context,
        tools=[FakeTool()],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert result["response"] == "The result is 4."
    assert len(llm.stream_calls) == 2

    retry_starts = [
        event
        for event in tracer.events
        if event["event_type"] == "action_start_llm"
        and event["data"].get("phase") == "empty_final_answer_recovery"
    ]
    assert len(retry_starts) == 1
    assert retry_starts[0]["data"]["recovery_reason"] == "empty_final_answer"

    # The discarded turn must not leak a partial stream to the user.
    assert [event["type"] for event in outbound.events] == [
        "final_answer_start",
        "final_answer_delta",
        "final_answer_end",
    ]
    assert outbound.events[-1]["content"] == "The result is 4."


@pytest.mark.asyncio
async def test_react_strips_empty_final_answer_with_trailing_work_call() -> None:
    """An empty-answer final_answer bundled with a work tool is stripped like
    any other bundled final_answer: the work tool executes instead of being
    discarded along with the empty answer (I-11, streaming path)."""

    llm = StreamingEmptyFinalAnswerLLM(trailing_work_tool=True)
    pattern, context, runtime, _outbound, _tracer = _react_empty_final_answer_fixture()
    tool = FakeTool()

    result = await pattern.run(
        context=context,
        tools=[tool],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert result["response"] == "The result is 4."
    assert tool.calls == [{"expression": "2+2"}]
    assert len(llm.stream_calls) == 2

    assistant_tool_call_ids = {
        tool_call["id"]
        for message in context.messages
        for tool_call in (message.tool_calls or [])
    }
    tool_result_ids = {
        message.tool_call_id for message in context.messages if message.role == "tool"
    }
    assert "call_final_0" not in assistant_tool_call_ids
    assert "call_final_0" not in pattern.tool_ledger
    assert assistant_tool_call_ids <= tool_result_ids


@pytest.mark.asyncio
async def test_react_fails_when_final_answer_stays_empty_after_retry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = StreamingEmptyFinalAnswerLLM(recover=False)
    pattern, context, runtime, outbound, tracer = _react_empty_final_answer_fixture()

    with caplog.at_level(
        logging.WARNING, logger="xagent.core.agent.pattern.react.react"
    ):
        result = await pattern.run(
            context=context,
            tools=[FakeTool()],
            llm=llm,
            runtime=runtime,
        )

    # The core regression: an empty answer must never finalize as success with
    # the "No output provided" placeholder.
    assert result["success"] is False
    assert result["status"] == "invalid_tool_protocol"
    assert "output" not in result
    assert len(llm.stream_calls) == 2
    assert outbound.events == []

    discarded = [
        event["data"]["phase"]
        for event in tracer.events
        if event["event_type"] == "action_end_llm"
        and event["data"].get("success") is False
    ]
    assert discarded == [
        "discarded_invalid_tool_protocol",
        "discarded_invalid_tool_protocol_retry",
    ]

    # Neither condition was logged before this fix, leaving nothing to debug from.
    messages = [record.getMessage() for record in caplog.records]
    assert any("final_answer carried no answer text" in m for m in messages)
    assert any(
        "invalid tool protocol after retry" in m and "failing the run" in m
        for m in messages
    )


@pytest.mark.asyncio
async def test_react_retries_final_answer_with_malformed_arguments() -> None:
    llm = StreamingEmptyFinalAnswerLLM(
        broken_arguments='{"response_language":"English","answer":"The res',
    )
    pattern, context, runtime, outbound, _tracer = _react_empty_final_answer_fixture()

    result = await pattern.run(
        context=context,
        tools=[FakeTool()],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert result["response"] == "The result is 4."
    assert len(llm.stream_calls) == 2
    # Truncated arguments can stream a partial answer before the call is
    # rejected. That partial is closed with an error and superseded by a fresh
    # stream, matching how the pattern already handles a protocol retry - the
    # point is that the run ends with a real answer instead of nothing.
    assert [event["type"] for event in outbound.events] == [
        "final_answer_start",
        "final_answer_delta",
        "final_answer_error",
        "final_answer_start",
        "final_answer_delta",
        "final_answer_end",
    ]
    assert outbound.events[-1]["content"] == "The result is 4."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "blank_answer",
    [
        pytest.param({}, id="missing"),
        pytest.param({"answer": ""}, id="empty-string"),
        pytest.param({"answer": "   \n"}, id="whitespace"),
        # ``str(None)`` is the literal "None", which would sail past a naive
        # ``str(args.get("answer", "")).strip()`` check and be finalized as the
        # user-facing answer.
        pytest.param({"answer": None}, id="none"),
    ],
)
async def test_react_re_requests_final_answer_for_resumed_empty_pending_call(
    blank_answer: dict[str, Any],
) -> None:
    """A pending call restored from a checkpoint bypasses response normalization."""

    llm = StreamingEmptyFinalAnswerLLM()
    pattern, context, runtime, _outbound, _tracer = _react_empty_final_answer_fixture()
    pattern.status = "acting"
    pattern.pending_tool_calls = [
        {
            "id": "call_resumed",
            "name": "final_answer",
            "args": {
                "response_language": "English",
                "outcome": "completed",
                **blank_answer,
            },
        }
    ]

    result = await pattern.run(
        context=context,
        tools=[FakeTool()],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert result["response"] == "The result is 4."
    assert pattern.tool_ledger["call_resumed"].status == "failed"
    assert pattern.pending_tool_calls == []
    # The next turn is forced back to final_answer only.
    assert [
        tool["function"]["name"] for tool in llm.stream_calls[0].get("tools") or []
    ] == ["final_answer"]


def test_final_answer_text_treats_none_as_absent() -> None:
    """One coercion for both the protocol check and finalization.

    ``str(None)`` is the literal "None"; if the two sites coerce independently
    they drift, and a null answer gets finalized as the string "None".
    """

    pattern = ReActPattern()

    assert pattern._final_answer_text({"answer": None}) == ""
    assert pattern._final_answer_text({}) == ""
    assert pattern._final_answer_text(None) == ""
    assert pattern._final_answer_text({"answer": ""}) == ""
    assert pattern._final_answer_text({"answer": "  hi  "}) == "  hi  "
    assert pattern._final_answer_text({"answer": 0}) == "0"


def test_coerce_arguments_drops_unusable_control_tool_payloads() -> None:
    pattern = ReActPattern()

    # Work tools keep the opaque passthrough so existing behavior is unchanged.
    assert pattern._coerce_arguments("not json", tool_name="calculator") == {
        "input": "not json"
    }
    assert pattern._coerce_arguments("[1, 2]", tool_name="calculator") == {
        "input": [1, 2]
    }

    # Blank payloads have nothing to preserve, whether the raw string is blank
    # or the JSON literal of a blank string.
    assert pattern._coerce_arguments("", tool_name="calculator") == {}
    assert pattern._coerce_arguments('""', tool_name="calculator") == {}
    assert pattern._coerce_arguments('"   "', tool_name="calculator") == {}

    # Control tools must not smuggle a malformed payload through as ``input``,
    # which would silently strip ``answer`` and finalize with nothing to show.
    assert pattern._coerce_arguments("not json", tool_name="final_answer") == {}
    assert pattern._coerce_arguments('"just a string"', tool_name="final_answer") == {}
    assert pattern._coerce_arguments("not json", tool_name="send_message") == {}

    # Well-formed payloads are untouched.
    assert pattern._coerce_arguments('{"answer":"hi"}', tool_name="final_answer") == {
        "answer": "hi"
    }


def test_empty_final_answer_call_detects_blank_answers() -> None:
    pattern = ReActPattern()

    def normalized(args: Any) -> dict[str, Any]:
        return {"tool_calls": [{"id": "c1", "name": "final_answer", "args": args}]}

    assert pattern._empty_final_answer_call(normalized({})) is not None
    assert pattern._empty_final_answer_call(normalized({"answer": ""})) is not None
    assert pattern._empty_final_answer_call(normalized({"answer": "  \n"})) is not None
    assert pattern._empty_final_answer_call(normalized({"answer": None})) is not None
    assert pattern._empty_final_answer_call(normalized({"answer": "done"})) is None
    # Coercion must match ``_handle_control_tool``, which stringifies the value,
    # so a scalar answer is not mistaken for a missing one.
    assert pattern._empty_final_answer_call(normalized({"answer": 0})) is None
    assert pattern._empty_final_answer_call({"tool_calls": []}) is None
    assert (
        pattern._empty_final_answer_call(
            {"tool_calls": [{"id": "c1", "name": "calculator", "args": {}}]}
        )
        is None
    )


@pytest.mark.asyncio
async def test_react_discards_rest_of_resumed_batch_after_empty_final_answer() -> None:
    """A rejected resume batch cancels its still-pending sibling calls.

    On resume the whole batch's assistant envelope is already recorded in
    history before execution starts, so every queued call must end with a
    result row; the only safe outcome for siblings behind a rejected empty
    ``final_answer`` is cancellation. A fresh-turn ``[final_answer(""),
    work_tool]`` batch instead strips the ``final_answer`` before anything
    is recorded and runs the work tool, so the two paths diverge on purpose.
    """

    llm = StreamingEmptyFinalAnswerLLM()
    pattern, context, runtime, _outbound, _tracer = _react_empty_final_answer_fixture()
    tool = FakeTool()
    pattern.status = "acting"
    pattern.pending_tool_calls = [
        {
            "id": "call_resumed",
            "name": "final_answer",
            "args": {"response_language": "English", "outcome": "completed"},
        },
        {"id": "call_work", "name": "calculator", "args": {"expression": "2+2"}},
    ]

    result = await pattern.run(
        context=context,
        tools=[tool],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert result["response"] == "The result is 4."
    # The side-effecting tool never ran.
    assert tool.calls == []
    # I2: the abandoned call still got exactly one result.
    assert pattern.tool_ledger["call_work"].status == "cancelled"
    assert pattern.tool_ledger["call_resumed"].status == "failed"
    assert pattern.pending_tool_calls == []


@pytest.mark.asyncio
async def test_react_finalizes_a_scalar_zero_answer() -> None:
    """``answer=0`` is an answer, not an absent one.

    The empty check coerces before testing, so a falsy scalar must survive the
    guard and reach the user rather than being rejected as blank.
    """

    llm = StreamingEmptyFinalAnswerLLM(
        broken_arguments='{"response_language":"English","answer":0,"outcome":"completed"}',
    )
    pattern, context, runtime, _outbound, _tracer = _react_empty_final_answer_fixture()

    result = await pattern.run(
        context=context,
        tools=[FakeTool()],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert result["response"] == "0"
    # Accepted on the first turn: no repair retry was spent.
    assert len(llm.stream_calls) == 1


@pytest.mark.asyncio
async def test_react_marks_the_abandoned_run_when_the_answer_stayed_empty() -> None:
    """The abandoned result distinguishes "never answered" from other violations.

    ``invalid_tool_protocol`` also covers provider protocol errors, mixed control
    calls, and a non-``final_answer`` tool on a forced turn. Delegated-child
    classification reads this marker so it does not collapse all of them into
    "never produced an answer" and discard the child's own diagnostic.
    """

    llm = StreamingEmptyFinalAnswerLLM(recover=False)
    pattern, context, runtime, _outbound, _tracer = _react_empty_final_answer_fixture()

    result = await pattern.run(
        context=context,
        tools=[FakeTool()],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is False
    assert result["status"] == "invalid_tool_protocol"
    assert result["empty_final_answer"] is True
    assert "final_answer without an answer" in result["error"]


@pytest.mark.asyncio
async def test_react_does_not_mark_other_tool_protocol_violations() -> None:
    """A non-empty-answer violation keeps the generic error and no marker."""

    llm = StreamingInvalidToolProtocolFinalAnswerLLM(
        structured_work_tool=True,
        invalid_retry=True,
    )
    pattern = ReActPattern(max_iterations=3, finalize_after_tool_result=True)
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="task-1")
    context.add_user_message("Find an audio clip")
    runtime = PatternRuntime(execution_id="task-1")

    result = await pattern.run(
        context=context,
        tools=[FakeTool()],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is False
    assert result["status"] == "invalid_tool_protocol"
    assert result["empty_final_answer"] is False
    assert "invalid tool protocol response" in result["error"]


@pytest.mark.asyncio
async def test_react_resumed_empty_final_answer_survives_a_state_roundtrip() -> None:
    """The post-rejection state is recoverable through the real mechanism.

    The guard exists for checkpoint resumes, so the rejection it performs has to
    survive ``get_state``/``load_state`` rather than only an in-process mutation.
    """

    pattern = ReActPattern(max_iterations=3)
    pattern.status = "acting"
    pattern.pending_tool_calls = [
        {
            "id": "call_resumed",
            "name": "final_answer",
            "args": {"response_language": "English", "outcome": "completed"},
        }
    ]

    resumed = ReActPattern(max_iterations=3)
    resumed.load_state(pattern.get_state())
    assert resumed.pending_tool_calls == pattern.pending_tool_calls

    llm = StreamingEmptyFinalAnswerLLM()
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="task-1")
    context.add_user_message("What is 2+2?")
    result = await resumed.run(
        context=context,
        tools=[FakeTool()],
        llm=llm,
        runtime=PatternRuntime(execution_id="task-1"),
    )

    assert result["success"] is True
    assert result["response"] == "The result is 4."

    # The forcing the rejection installed is itself serializable, which is what a
    # crash between the rejection and the next turn would depend on.
    carried = ReActPattern()
    carried.load_state(resumed.get_state())
    assert carried.force_final_answer_next is False  # cleared once answered
    assert carried.tool_ledger["call_resumed"].status == "failed"


@pytest.mark.asyncio
async def test_react_resumed_empty_final_answer_abandons_after_forced_retry() -> None:
    """resume -> reject -> forced turn -> empty -> in-turn retry -> abandon."""

    llm = StreamingEmptyFinalAnswerLLM(recover=False)
    pattern, context, runtime, _outbound, _tracer = _react_empty_final_answer_fixture()
    pattern.status = "acting"
    pattern.pending_tool_calls = [
        {
            "id": "call_resumed",
            "name": "final_answer",
            "args": {"response_language": "English", "outcome": "completed"},
        }
    ]

    result = await pattern.run(
        context=context,
        tools=[FakeTool()],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is False
    assert result["status"] == "invalid_tool_protocol"
    assert result["empty_final_answer"] is True
    assert pattern.pending_tool_calls == []
    # The forced turn plus its one in-turn repair attempt, and no more.
    assert len(llm.stream_calls) == 2


@pytest.mark.asyncio
async def test_react_resumed_reverse_batch_order_cannot_undo_an_executed_tool() -> None:
    """Pins the documented limit of the resume guard's batch discard.

    In ``[work_tool, final_answer("")]`` restored onto ``pending_tool_calls``,
    the work tool has already executed by the time the rejection runs, so
    discarding the rest of the batch cannot reach back and undo it - the
    resume guard can only cancel calls still queued, never ones already
    committed.
    """

    llm = StreamingEmptyFinalAnswerLLM()
    tool = FakeTool()
    pattern, context, runtime, _outbound, _tracer = _react_empty_final_answer_fixture()
    pattern.status = "acting"
    pattern.pending_tool_calls = [
        {"id": "call_work", "name": "calculator", "args": {"expression": "2+2"}},
        {
            "id": "call_resumed",
            "name": "final_answer",
            "args": {"response_language": "English", "outcome": "completed"},
        },
    ]

    result = await pattern.run(
        context=context,
        tools=[tool],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    # The work tool ran: the rejection cannot reach back past it.
    assert tool.calls == [{"expression": "2+2"}]
    assert pattern.tool_ledger["call_work"].status == "completed"
    assert pattern.tool_ledger["call_resumed"].status == "failed"


def test_rejected_empty_final_answer_is_recorded_as_a_failure() -> None:
    """The rejection must not read back as a successful tool result."""

    pattern = ReActPattern()
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="task-1")
    context.add_user_message("What is 2+2?")
    tool_call = {"id": "call_1", "name": "final_answer", "args": {"answer": ""}}
    pattern.pending_tool_calls = [tool_call]

    pattern._reject_empty_final_answer(tool_call, context)

    recorded = pattern.tool_ledger["call_1"]
    assert recorded.status == "failed"
    # The recorded result carries the keys ``tool_result_succeeded`` reads, so a
    # consumer cannot read this failure back as a success.
    assert tool_result_succeeded(recorded.result) is False
    # Same shape as the cancelled siblings, so the ledger is uniform.
    assert recorded.result["status"] == "error"
    tool_messages = [m for m in context.messages if getattr(m, "role", None) == "tool"]
    assert "empty answer" in tool_messages[-1].content


def test_reject_empty_final_answer_tolerates_non_string_arg_keys() -> None:
    """Log formatting must not abort the run it is diagnosing.

    ``sorted`` over mixed-type keys raises ``TypeError``, and ``run()`` re-raises
    from its ``except``, so an unsortable payload would fail the run at exactly
    the point this guard exists to recover from.
    """

    pattern = ReActPattern()
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="task-1")
    context.add_user_message("What is 2+2?")
    tool_call = {
        "id": "call_1",
        "name": "final_answer",
        "args": {"answer": "", 1: "numeric", None: "null"},
    }
    pattern.pending_tool_calls = [tool_call]

    pattern._reject_empty_final_answer(tool_call, context)

    assert pattern.force_final_answer_next is True


@pytest.mark.asyncio
async def test_react_discards_the_preamble_with_the_rejected_response() -> None:
    """Pins the documented cost of whole-response rejection.

    The preamble arrived attached to a protocol violation, so it is not a vetted
    answer and is dropped with the response rather than replayed into the retry.
    A model that recovers supersedes it; one that repeats the pattern fails the
    run and the preamble is never shown.
    """

    llm = StreamingEmptyFinalAnswerLLM(preamble="Let me look into that.")
    pattern, context, runtime, outbound, _tracer = _react_empty_final_answer_fixture()

    result = await pattern.run(
        context=context,
        tools=[FakeTool()],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert result["response"] == "The result is 4."
    assert "Let me look into that." not in str(result["response"])
    assert all(
        "Let me look into that." not in str(event.get("content") or event.get("delta"))
        for event in outbound.events
    )


@pytest.mark.parametrize(
    ("tool_names", "unavailable"),
    [
        (["generate_image", "list_image_models"], True),
        (["generate_image", "edit_image"], False),
        (["list_image_models"], False),
        (["web_search"], False),
    ],
)
@pytest.mark.asyncio
async def test_run_flags_missing_image_editing_and_renders_the_correction(
    tool_names: list[str], unavailable: bool
) -> None:
    from xagent.core.agent.context.enrichment import (
        IMAGE_EDIT_UNAVAILABLE_METADATA_KEY,
        SKILL_CONTEXT_METADATA_KEY,
    )

    from .concurrency_harness import FakeTool as NamedFakeTool

    llm = FakeLLM(responses=[{"content": "done", "done": True}])
    pattern = ReActPattern(max_iterations=2)
    context = ExecutionContext(system_prompt="You are helpful.")
    context.metadata[SKILL_CONTEXT_METADATA_KEY] = "Use `edit_image` to refine."
    context.add_user_message("make an ad")

    await pattern.run(
        context=context,
        tools=[NamedFakeTool(name) for name in tool_names],
        llm=llm,
    )

    assert context.metadata[IMAGE_EDIT_UNAVAILABLE_METADATA_KEY] is unavailable
    rendered = llm.calls[0]["messages"][0]["content"]
    assert ("image editing is unavailable here" in rendered) is unavailable
    assert ("attach a reference through images" in rendered) is unavailable


class NoArgToolLLM:
    """Calls a parameterless tool with `arguments: ""`, then finishes.

    Reproduces the provider shape behind xorbitsai/xagent#1501 on the streaming
    path: a blank argument string must reach the tool as an empty call, not kill
    the run.
    """

    def __init__(self) -> None:
        self.stream_calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        raise AssertionError("this path should stay streaming")

    async def stream_chat(
        self, messages: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> Any:
        if messages is not None:
            kwargs["messages"] = messages
        self.stream_calls.append(kwargs)
        if len(self.stream_calls) == 1:
            yield StreamChunk(
                type=ChunkType.TOOL_CALL,
                tool_calls=[
                    {
                        "id": "call_list_0",
                        "function": {"name": "list_models", "arguments": ""},
                    }
                ],
            )
        else:
            yield StreamChunk(
                type=ChunkType.TOOL_CALL,
                tool_calls=[
                    {
                        "id": "call_final_0",
                        "function": {
                            "name": "final_answer",
                            "arguments": (
                                '{"response_language":"English",'
                                '"answer":"gpt-4o","outcome":"completed"}'
                            ),
                        },
                    }
                ],
            )
        yield StreamChunk(type=ChunkType.END)


class NoArgTool:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

        class Metadata:
            name = "list_models"
            description = "List available models."

        self.metadata = Metadata()

    def args_type(self) -> type[BaseModel]:
        return EmptyArgs

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        self.calls.append(args)
        return {"models": ["gpt-4o"]}


@pytest.mark.asyncio
async def test_react_runs_a_parameterless_tool_called_with_blank_arguments() -> None:
    """#1501 (a): a no-arg tool sent `""` executes instead of failing the run."""

    llm = NoArgToolLLM()
    pattern, context, runtime, _outbound, _tracer = _react_empty_final_answer_fixture()
    tool = NoArgTool()

    result = await pattern.run(
        context=context,
        tools=[tool, FakeTool()],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert result["response"] == "gpt-4o"
    assert tool.calls == [{}]


@pytest.mark.asyncio
async def test_react_repairs_final_answer_called_with_blank_arguments() -> None:
    """Guards the `_empty_final_answer_call` repair retry, not `_fallback_arguments`.

    `final_answer` is a control tool, so blank arguments are dropped by the
    control-tool branch and never reach the blank-string branch this PR adds.
    What is pinned here is that the resulting empty args still spend the one
    repair retry instead of finalizing the run.
    """

    llm = StreamingEmptyFinalAnswerLLM(broken_arguments="")
    pattern, context, runtime, outbound, tracer = _react_empty_final_answer_fixture()

    result = await pattern.run(
        context=context,
        tools=[FakeTool()],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert result["response"] == "The result is 4."
    assert len(llm.stream_calls) == 2

    retry_starts = [
        event
        for event in tracer.events
        if event["event_type"] == "action_start_llm"
        and event["data"].get("phase") == "empty_final_answer_recovery"
    ]
    assert len(retry_starts) == 1
    assert outbound.events[-1]["content"] == "The result is 4."


class BlankThenRecoverLLM:
    """Calls a required-argument work tool with `""`, then recovers.

    First turn: `calculator` with blank arguments. Second turn: a valid
    `final_answer`. Pins what actually happens to a required-argument tool
    handed `{}` — the tool fails internally and the error feeds back as a tool
    result, not a dead run.
    """

    def __init__(self) -> None:
        self.stream_calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        raise AssertionError("this path should stay streaming")

    async def stream_chat(
        self, messages: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> Any:
        if messages is not None:
            kwargs["messages"] = messages
        self.stream_calls.append(kwargs)
        if len(self.stream_calls) == 1:
            yield StreamChunk(
                type=ChunkType.TOOL_CALL,
                tool_calls=[
                    {
                        "id": "call_calc_0",
                        "function": {"name": "calculator", "arguments": ""},
                    }
                ],
            )
        else:
            yield StreamChunk(
                type=ChunkType.TOOL_CALL,
                tool_calls=[
                    {
                        "id": "call_final_0",
                        "function": {
                            "name": "final_answer",
                            "arguments": (
                                '{"response_language":"English",'
                                '"answer":"The result is 4.","outcome":"completed"}'
                            ),
                        },
                    }
                ],
            )
        yield StreamChunk(type=ChunkType.END)


@pytest.mark.asyncio
async def test_react_survives_required_argument_tool_called_with_blank_arguments() -> (
    None
):
    """#1501 (b): a required-argument work tool sent `""` receives `{}`, fails
    inside the tool, and the error feeds back to the model instead of killing
    the run."""

    llm = BlankThenRecoverLLM()
    pattern, context, runtime, _outbound, _tracer = _react_empty_final_answer_fixture()
    tool = FakeTool()

    result = await pattern.run(
        context=context,
        tools=[tool],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert result["response"] == "The result is 4."
    assert len(llm.stream_calls) == 2

    # Discriminating assertion for react.py's blank-string fallback branch: the
    # old code delivered {"input": ""} here, which would fail this equality.
    assert tool.calls == [{}]

    tool_results = [m for m in context.messages if m.role == "tool"]
    assert len(tool_results) >= 1
    first = tool_results[0]
    assert first.tool_call_id == "call_calc_0"
    assert "'success': False" in first.content


@pytest.mark.asyncio
async def test_blank_streaming_arguments_flow_from_adapter_into_react(mocker) -> None:
    """The seam: a real `OpenAICompatibleLLM` stream carrying `arguments: ""`
    drives a real ReAct loop and the parameterless tool executes."""

    def sdk_chunk(tool_calls=None, finish_reason=None):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=tool_calls),
                    finish_reason=finish_reason,
                )
            ]
        )

    def sdk_tool_call(name, arguments):
        return SimpleNamespace(
            index=0,
            id=f"call_{name}",
            type="function",
            function=SimpleNamespace(name=name, arguments=arguments),
        )

    async def first_stream():
        yield sdk_chunk([sdk_tool_call("list_models", "")])
        yield sdk_chunk(finish_reason="tool_calls")

    async def second_stream():
        yield sdk_chunk(
            [
                sdk_tool_call(
                    "final_answer",
                    '{"response_language":"English",'
                    '"answer":"gpt-4o","outcome":"completed"}',
                )
            ]
        )
        yield sdk_chunk(finish_reason="tool_calls")

    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = [first_stream(), second_stream()]
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=mock_client,
    )
    from xagent.core.model.chat.basic.openai import OpenAILLM

    llm = OpenAILLM(model_name="gpt-4o-mini", base_url=None, api_key="test-key")
    pattern, context, runtime, _outbound, _tracer = _react_empty_final_answer_fixture()
    tool = NoArgTool()

    result = await pattern.run(
        context=context,
        tools=[tool, FakeTool()],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert result["response"] == "gpt-4o"
    assert tool.calls == [{}]


def _react_clock_prompt(timezone_name: str | None) -> str:
    context = ExecutionContext(
        created_at=datetime(2026, 8, 24, 22, 3, 37, tzinfo=timezone.utc)
    )
    if timezone_name is not None:
        context.metadata[CLOCK_TIMEZONE_METADATA_KEY] = timezone_name
    context.add_user_message("how many shifts do we have on tomorrow?")

    messages = ReActPattern()._messages_for_llm(context, has_tools=True)
    return messages[0]["content"]


# Covers the whole fallback instruction through the get_current_time pointer, so
# reverting any part of it fails rather than only a change to the prefix.
UTC_DATE_INSTRUCTION = (
    "Turn-start date (UTC): 2026-08-24. "
    "For recent, latest, current, or time-sensitive requests, use this "
    "date when forming search queries and judging source relevance. If the "
    "exact current time matters or the turn may have crossed midnight, call "
    "the get_current_time tool if it is available."
)


def _date_instruction(prompt: str) -> str:
    start = prompt.index("Turn-start date (")
    end = prompt.index("if it is available.", start) + len("if it is available.")
    return prompt[start:end]


def test_tool_call_date_line_keeps_utc_wording_without_a_timezone() -> None:
    assert _date_instruction(_react_clock_prompt(None)) == UTC_DATE_INSTRUCTION


def test_tool_call_date_line_uses_the_caller_timezone() -> None:
    prompt = _react_clock_prompt("Australia/Melbourne")

    assert _date_instruction(prompt) == UTC_DATE_INSTRUCTION.replace(
        "Turn-start date (UTC): 2026-08-24. ",
        "Turn-start date (Australia/Melbourne): 2026-08-25. ",
    )
    # The UTC date is what produced the wrong "tomorrow" in production.
    assert "Turn-start date (UTC)" not in prompt


def test_tool_call_date_line_degrades_to_utc_for_an_unusable_timezone() -> None:
    assert _date_instruction(_react_clock_prompt("Not/AZone")) == UTC_DATE_INSTRUCTION
