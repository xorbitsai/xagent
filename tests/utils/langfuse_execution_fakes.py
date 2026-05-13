"""Shared fakes and assertions for Langfuse tracing execution tests."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from pydantic import BaseModel

from xagent.core.agent.trace import TraceHandler
from xagent.core.memory.base import MemoryStore
from xagent.core.memory.core import MemoryNote, MemoryResponse
from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.chat.types import ChunkType, StreamChunk
from xagent.core.tools.adapters.vibe import Tool, ToolMetadata
from xagent.core.tracing.langfuse.handler import LangfuseTraceHandler


class FakeOtelSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}

    def set_attribute(self, key: Any, value: Any) -> None:
        self.attributes[str(key)] = value


class FakeObservation:
    def __init__(
        self,
        span_id: str,
        kwargs: dict[str, Any],
        *,
        client: "FakeLangfuseClient",
        parent_id: Optional[str] = None,
    ) -> None:
        self.trace_id = "trace-1"
        self.id = span_id
        self.parent_id = parent_id
        self.start_kwargs = kwargs
        self.updates: list[dict[str, Any]] = []
        self.trace_updates: list[dict[str, Any]] = []
        self.ended = False
        self.end_count = 0
        self._otel_span = FakeOtelSpan()
        self._client = client

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def update_trace(self, **kwargs: Any) -> None:
        self.trace_updates.append(kwargs)

    def end(self) -> None:
        self.ended = True
        self.end_count += 1

    def start_observation(self, **kwargs: Any) -> "FakeObservation":
        return self._client.start_child_observation(parent=self, **kwargs)


class FakeLangfuseClient:
    def __init__(self) -> None:
        self.observations: list[FakeObservation] = []
        self.flushed = False

    def start_observation(self, **kwargs: Any) -> FakeObservation:
        observation = FakeObservation(
            f"span-{len(self.observations) + 1}", kwargs, client=self
        )
        self.observations.append(observation)
        return observation

    def start_child_observation(
        self, *, parent: FakeObservation, **kwargs: Any
    ) -> FakeObservation:
        observation = FakeObservation(
            f"span-{len(self.observations) + 1}",
            kwargs,
            client=self,
            parent_id=parent.id,
        )
        self.observations.append(observation)
        return observation

    def flush(self) -> None:
        self.flushed = True


class DummyMemoryStore(MemoryStore):
    def add(self, note: MemoryNote) -> MemoryResponse:
        del note
        return MemoryResponse(success=True, memory_id="test-memory-id")

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> list[Any]:
        del query, k, filters, similarity_threshold
        return []

    def get(self, note_id: str) -> MemoryResponse:
        del note_id
        return MemoryResponse(success=False, error="not found")

    def delete(self, note_id: str) -> MemoryResponse:
        del note_id
        return MemoryResponse(success=True)

    def clear(self) -> None:
        return None

    def update(self, note: MemoryNote) -> MemoryResponse:
        del note
        return MemoryResponse(success=True, memory_id="test-memory-id")

    def get_stats(self) -> dict[str, Any]:
        return {}

    def list_all(self, filters: Optional[dict[str, Any]] = None) -> list[Any]:
        del filters
        return []


class CalculatorArgs(BaseModel):
    expression: str


class WeatherArgs(BaseModel):
    city: str = "Singapore"


class CalculatorTool(Tool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(name="calculator", description="Calculate expressions")

    def args_type(self) -> type[BaseModel]:
        return CalculatorArgs

    def return_type(self) -> type[BaseModel]:
        return CalculatorArgs

    def state_type(self) -> None:
        return None

    def is_async(self) -> bool:
        return True

    def return_value_as_string(self, value: Any) -> str:
        return str(value)

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        expression = str(args.get("expression", ""))
        if expression == "2 + 2":
            return {"result": 4}
        return {"result": expression}

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        return {"result": args.get("expression")}

    async def save_state_json(self) -> Mapping[str, Any]:
        return {}

    async def load_state_json(self, state: Mapping[str, Any]) -> None:
        del state


class FailingTool(Tool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(name="failing_tool", description="Always fails")

    def args_type(self) -> type[BaseModel]:
        return CalculatorArgs

    def return_type(self) -> type[BaseModel]:
        return CalculatorArgs

    def state_type(self) -> None:
        return None

    def is_async(self) -> bool:
        return True

    def return_value_as_string(self, value: Any) -> str:
        return str(value)

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        del args
        raise RuntimeError("boom")

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        del args
        raise RuntimeError("boom")

    async def save_state_json(self) -> Mapping[str, Any]:
        return {}

    async def load_state_json(self, state: Mapping[str, Any]) -> None:
        del state


class WeatherTool(Tool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(name="get_weather", description="Get weather")

    def args_type(self) -> type[BaseModel]:
        return WeatherArgs

    def return_type(self) -> type[BaseModel]:
        return WeatherArgs

    def state_type(self) -> None:
        return None

    def is_async(self) -> bool:
        return True

    def return_value_as_string(self, value: Any) -> str:
        return str(value)

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        return {"forecast": "sunny", "city": args.get("city", "Singapore")}

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        return {"forecast": "sunny", "city": args.get("city", "Singapore")}

    async def save_state_json(self) -> Mapping[str, Any]:
        return {}

    async def load_state_json(self, state: Mapping[str, Any]) -> None:
        del state


class DeterministicSingleCallLLM(BaseLLM):
    def __init__(
        self, tool_name: str = "calculator", final_answer: str = "The result is 4"
    ):
        self._tool_name = tool_name
        self._final_answer = final_answer
        self._call_count = 0
        self._abilities = ["chat", "tool_calling"]
        self._model_name = "deterministic-single-call"

    @property
    def abilities(self) -> list[str]:
        return self._abilities

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def supports_thinking_mode(self) -> bool:
        return False

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        del messages, kwargs
        self._call_count += 1
        if self._call_count == 1:
            return {
                "type": "tool_call",
                "tool_calls": [
                    {
                        "function": {
                            "name": self._tool_name,
                            "arguments": json.dumps({"expression": "2 + 2"}),
                        }
                    }
                ],
            }
        return {"type": "text", "content": self._final_answer}

    async def stream_chat(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        del messages, kwargs
        yield StreamChunk(type=ChunkType.END)


class DeterministicReActLLM(BaseLLM):
    def __init__(self) -> None:
        self._call_count = 0
        self._abilities = ["chat", "tool_calling"]
        self._model_name = "deterministic-react"

    @property
    def abilities(self) -> list[str]:
        return self._abilities

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def supports_thinking_mode(self) -> bool:
        return False

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        del messages, kwargs
        self._call_count += 1
        if self._call_count == 1:
            return {
                "type": "tool_call",
                "tool_calls": [
                    {
                        "function": {
                            "name": "calculator",
                            "arguments": json.dumps({"expression": "2 + 2"}),
                        }
                    }
                ],
            }
        if self._call_count == 2:
            return {"content": "The result is 4"}
        return {"content": "The result is 4"}

    async def stream_chat(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        response = await self.chat(messages, **kwargs)
        has_tools = bool(kwargs.get("tools"))
        if has_tools:
            parsed = response if isinstance(response, dict) else json.loads(response)
            if parsed.get("type") == "tool_call":
                yield StreamChunk(
                    type=ChunkType.TOOL_CALL,
                    tool_calls=parsed["tool_calls"],
                )
                yield StreamChunk(type=ChunkType.END, finish_reason="tool_calls")
                return
        content = (
            response.get("content", "") if isinstance(response, dict) else response
        )
        yield StreamChunk(type=ChunkType.TOKEN, content=content, delta=content)
        yield StreamChunk(type=ChunkType.END, finish_reason="stop")


class DeterministicDagLLM(BaseLLM):
    def __init__(self) -> None:
        self._abilities = ["chat", "tool_calling"]
        self._model_name = "deterministic-dag"

    @property
    def abilities(self) -> list[str]:
        return self._abilities

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def supports_thinking_mode(self) -> bool:
        return False

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        content_blob = "\n".join(
            str(message.get("content", "")) for message in messages
        )
        lowered = content_blob.lower()
        response_format = kwargs.get("response_format")
        output_config = kwargs.get("output_config")

        if output_config and "chat" in lowered and "plan" in lowered:
            return '{"type":"plan"}'

        if output_config and "goal achievement analysis and final answer" in lowered:
            return """{
                "achieved": true,
                "reason": "Weather information retrieved successfully",
                "confidence": 0.98,
                "final_answer": "Weather is sunny in Singapore today",
                "memory_insights": {
                    "should_store": false,
                    "reason": "Routine weather lookup with no reusable non-obvious insight",
                    "classification": {
                        "primary_domain": "Information Retrieval",
                        "secondary_domains": ["Weather"],
                        "task_type": "Weather Lookup",
                        "complexity_level": "Simple",
                        "keywords": ["weather", "singapore"]
                    },
                    "execution_insights": "",
                    "failure_analysis": "",
                    "success_factors": "",
                    "learned_patterns": "",
                    "improvement_suggestions": "",
                    "user_preferences": "",
                    "behavioral_patterns": ""
                }
            }"""

        if output_config and "memory_insights" in lowered and "final_answer" in lowered:
            return """{
                "achieved": true,
                "reason": "Weather information retrieved successfully",
                "confidence": 0.98,
                "final_answer": "Weather is sunny in Singapore today",
                "memory_insights": {
                    "should_store": false,
                    "reason": "Routine weather lookup with no reusable non-obvious insight",
                    "classification": {
                        "primary_domain": "Information Retrieval",
                        "secondary_domains": ["Weather"],
                        "task_type": "Weather Lookup",
                        "complexity_level": "Simple",
                        "keywords": ["weather", "singapore"]
                    },
                    "execution_insights": "",
                    "failure_analysis": "",
                    "success_factors": "",
                    "learned_patterns": "",
                    "improvement_suggestions": "",
                    "user_preferences": "",
                    "behavioral_patterns": ""
                }
            }"""

        if response_format == {"type": "json_object"}:
            if "task execution analyzer" in lowered:
                return (
                    '{"success": true, "direct_answer": "Weather is sunny in Singapore today", '
                    '"file_outputs": [], "confidence": "high", '
                    '"reasoning": "The weather information was retrieved successfully."}'
                )
            if "goal achievement analysis and final answer" in lowered:
                return (
                    '{"achieved": true, "reason": "Weather information retrieved successfully", '
                    '"confidence": 0.98, "final_answer": "Weather is sunny in Singapore today", '
                    '"memory_insights": {"should_store": false, "reason": "Routine weather lookup", '
                    '"classification": {"primary_domain": "Information Retrieval", '
                    '"secondary_domains": ["Weather"], "task_type": "Weather Lookup", '
                    '"complexity_level": "Simple", "keywords": ["weather", "singapore"]}, '
                    '"execution_insights": "", "failure_analysis": "", "success_factors": "", '
                    '"learned_patterns": "", "improvement_suggestions": "", '
                    '"user_preferences": "", "behavioral_patterns": ""}}'
                )

        if "create a step-by-step execution plan as a json object" in lowered:
            return """{
                "plan": {
                    "task_name": "Singapore Weather Check",
                    "goal": "Check weather in Singapore",
                    "steps": [
                        {
                            "id": "step1",
                            "name": "get_weather",
                            "description": "Check the weather in Singapore",
                            "tool_names": ["get_weather"],
                            "dependencies": [],
                            "difficulty": "hard",
                            "requires_vision": false
                        }
                    ]
                }
            }"""

        if "tool result from get_weather" in lowered:
            return (
                '{"type":"final_answer","reasoning":"Done",'
                '"answer":"Weather is sunny in Singapore today","success":true,"error":null}'
            )

        if kwargs.get("tools"):
            return '{"type":"tool_call","tool_name":"get_weather","tool_args":{"city":"Singapore"}}'

        if "tool_call" in lowered and "final_answer" in lowered:
            return '{"type":"tool_call","reasoning":"Need weather tool"}'

        if response_format == {"type": "json_object"}:
            return '{"type":"plan"}'

        return (
            '{"type":"final_answer","reasoning":"Done",'
            '"answer":"Weather is sunny in Singapore today","success":true,"error":null}'
        )

    async def stream_chat(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        response = await self.chat(messages, **kwargs)
        if kwargs.get("tools"):
            parsed = json.loads(response)
            if parsed.get("type") == "tool_call":
                yield StreamChunk(
                    type=ChunkType.TOOL_CALL,
                    tool_calls=[
                        {
                            "function": {
                                "name": parsed["tool_name"],
                                "arguments": json.dumps(parsed["tool_args"]),
                            }
                        }
                    ],
                )
                yield StreamChunk(type=ChunkType.END, finish_reason="tool_calls")
                return
        yield StreamChunk(type=ChunkType.TOKEN, content=response, delta=response)
        yield StreamChunk(type=ChunkType.END, finish_reason="stop")


def get_langfuse_handler(handlers: list[TraceHandler]) -> LangfuseTraceHandler:
    for handler in handlers:
        if isinstance(handler, LangfuseTraceHandler):
            return handler
    raise AssertionError("LangfuseTraceHandler not found")


def assert_handler_closed(handler: LangfuseTraceHandler) -> None:
    assert handler._closed is True
    assert handler._action_observations == {}
    assert handler._step_action_observations == {}
    assert handler._task_llm_observations == {}
    assert handler._step_observations == {}


def observations_by_type(
    fake_client: FakeLangfuseClient, as_type: str
) -> list[FakeObservation]:
    return [
        observation
        for observation in fake_client.observations
        if observation.start_kwargs.get("as_type") == as_type
    ]


def observation_names(
    fake_client: FakeLangfuseClient, as_type: Optional[str] = None
) -> list[str]:
    return [
        str(observation.start_kwargs.get("name"))
        for observation in fake_client.observations
        if as_type is None or observation.start_kwargs.get("as_type") == as_type
    ]


def update_data_values(observation: FakeObservation, key: str) -> list[Any]:
    values = []
    for update in observation.updates:
        metadata = update.get("metadata")
        if isinstance(metadata, dict):
            data = metadata.get("data")
            if isinstance(data, dict) and key in data:
                values.append(data[key])
    return values


def find_trace_update(
    observation: FakeObservation, key: str, expected: Any
) -> dict[str, Any]:
    for update in observation.trace_updates:
        if update.get(key) == expected:
            return update
    raise AssertionError(f"Trace update missing {key}={expected!r}")
