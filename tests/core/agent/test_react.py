from __future__ import annotations

import asyncio
import json
import re
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
    ToolCallRecord,
)
from xagent.core.model.chat.basic.router import RouterLLM
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
    async def chat(self, **kwargs: Any) -> Any:
        raise AssertionError("streaming mixed tool path should not call chat()")

    async def stream_chat(self, **kwargs: Any) -> Any:
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
    assert re.search(r"Current date \(UTC\): \d{4}-\d{2}-\d{2}", system_prompt)
    assert "use this date when forming search queries" in system_prompt
    assert "not supported by the conversation" in system_prompt
    assert "available context is insufficient" in system_prompt
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
    assert llm.stream_calls[1]["tool_choice"] == "required"
    assert [event["type"] for event in outbound.events] == [
        "final_answer_start",
        "final_answer_delta",
        "final_answer_end",
    ]


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
    assert [tool["function"]["name"] for tool in retry_tools] == ["final_answer"]
    assert llm.stream_calls[2]["tool_choice"] == "required"
    assert (
        "calling the final_answer control tool exactly once"
        in (llm.stream_calls[2]["messages"][0]["content"])
    )
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
        and event["data"].get("phase") == "tool_protocol_retry"
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
async def test_react_pattern_does_not_stream_mixed_final_answer_candidate() -> None:
    llm = StreamingMixedFinalAnswerAndToolLLM()
    pattern = ReActPattern(max_iterations=1)
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="task-1")
    context.add_user_message("Calculate 2+2")
    outbound = OutboundCollector()
    runtime = PatternRuntime(execution_id="task-1", outbound_message_handler=outbound)

    result = await pattern.run(
        context=context,
        tools=[FakeTool()],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    assert result["response"] == "Candidate"
    assert outbound.events == []


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
    final_answer_schema = next(
        schema
        for schema in llm.calls[0]["tools"]
        if schema["function"]["name"] == "final_answer"
    )["function"]
    assert (
        "same natural language as the current user request"
        in final_answer_schema["description"]
    )
    assert "response_language" in final_answer_schema["parameters"]["required"]
    response_language_schema = final_answer_schema["parameters"]["properties"][
        "response_language"
    ]
    assert "Simplified Chinese" in response_language_schema["description"]
    assert "Traditional Chinese" in response_language_schema["description"]
    assert "generic Chinese" in response_language_schema["description"]
    answer_schema = final_answer_schema["parameters"]["properties"]["answer"]
    assert "response_language" in answer_schema["description"]
    assert "tool results, source documents" in answer_schema["description"]


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
                        "id": "call_calc",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"9+1"}',
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
async def test_react_pattern_preserves_pending_calls_after_waiting_control_tool() -> (
    None
):
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
    assert pattern.pending_tool_calls == [
        {"id": "call_calc", "name": "calculator", "args": {"expression": "5+5"}}
    ]

    context.add_user_message("B")
    resumed_pattern = ReActPattern(max_iterations=4)
    resumed_pattern.load_state(pattern.get_state())
    resumed_llm = FakeLLM([{"content": "The result is 10.", "done": True}])

    resumed = await resumed_pattern.run(
        context=context,
        tools=[tool],
        llm=resumed_llm,
    )

    assert resumed["success"] is True
    assert tool.calls == [{"expression": "5+5"}]
    assert context.get_messages_by_role("tool")[-1].tool_call_id == "call_calc"
    resumed_messages = resumed_llm.calls[0]["messages"]
    resumed_tool_result = next(
        message
        for message in resumed_messages
        if message.get("role") == "tool" and message.get("tool_call_id") == "call_calc"
    )
    resumed_tool_envelope_index = resumed_messages.index(resumed_tool_result) - 1
    assert resumed_messages[resumed_tool_envelope_index]["role"] == "assistant"
    assert resumed_messages[resumed_tool_envelope_index]["tool_calls"][0]["id"] == (
        "call_calc"
    )
    assert "Tool calculator returned" in resumed_tool_result["content"]


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
