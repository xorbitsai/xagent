from __future__ import annotations

import asyncio
import json
import os
import threading
from types import SimpleNamespace
from typing import Any

import boto3
import pytest
from botocore import UNSIGNED
from botocore.stub import Stubber

from xagent.core.agent.context.message import Message
from xagent.core.model.chat.basic.adapter import create_base_llm
from xagent.core.model.chat.basic.bedrock import BedrockLLM
from xagent.core.model.chat.token_context import TokenContextManager
from xagent.core.model.chat.types import ChunkType
from xagent.core.model.model import ChatModelConfig
from xagent.core.model.providers import get_supported_provider_metadata


def _runtime_client():
    return boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )


def _unwrap(llm: Any) -> BedrockLLM:
    return llm._inner  # GenericRetryWrapper preserves the concrete adapter here.


def test_factory_and_metadata_preserve_native_bedrock_settings():
    llm = create_base_llm(
        ChatModelConfig(
            id="bedrock-1",
            model_name="us.anthropic.claude-sonnet-4-6",
            model_provider="bedrock",
            api_key="br-key",
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
            bedrock_region="us-east-1",
            bedrock_auth_mode="api_key",
        )
    )

    concrete = _unwrap(llm)
    assert isinstance(concrete, BedrockLLM)
    assert concrete.model_id == "bedrock-1"
    assert concrete.region_name == "us-east-1"
    assert concrete.endpoint_url == "https://bedrock-runtime.us-east-1.amazonaws.com"
    assert concrete.api_key == "br-key"
    assert concrete.auth_mode == "api_key"
    metadata = next(
        provider
        for provider in get_supported_provider_metadata()
        if provider["id"] == "bedrock"
    )
    assert metadata["category"] == ["llm"]
    assert metadata["credential_fields"][0]["required"] is False


@pytest.mark.asyncio
async def test_converse_translates_system_history_image_and_usage():
    client = _runtime_client()
    stubber = Stubber(client)
    expected = {
        "modelId": "amazon.nova-pro-v1:0",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": "describe"},
                    {
                        "image": {
                            "format": "png",
                            "source": {"bytes": b"image"},
                        }
                    },
                ],
            },
            {"role": "assistant", "content": [{"text": "Earlier reply"}]},
            {"role": "user", "content": [{"text": "continue"}]},
        ],
        "system": [{"text": "Be concise"}],
        "inferenceConfig": {"maxTokens": 321, "temperature": 0.2, "topP": 0.9},
    }
    stubber.add_response(
        "converse",
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "A diagram."}],
                }
            },
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 12,
                "outputTokens": 3,
                "totalTokens": 15,
                "cacheReadInputTokens": 4,
                "cacheWriteInputTokens": 2,
            },
            "metrics": {"latencyMs": 8},
        },
        expected,
    )
    stubber.activate()
    llm = BedrockLLM(
        "amazon.nova-pro-v1:0",
        client=client,
        abilities=["chat", "vision"],
        default_max_tokens=321,
    )
    llm._model_id = "bedrock-config-1"

    with TokenContextManager() as usage_context:
        result = await llm.vision_chat(
            [
                {"role": "system", "content": "Be concise"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
                        },
                    ],
                },
                {"role": "assistant", "content": "Earlier reply"},
                {"role": "user", "content": "continue"},
            ],
            temperature=0.2,
            top_p=0.9,
        )

    assert result["type"] == "text"
    assert result["content"] == "A diagram."
    assert result["finish_reason"] == "end_turn"
    assert result["usage"] == {
        "input_tokens": 12,
        "output_tokens": 3,
        "total_tokens": 15,
        "cached_input_tokens": 4,
        "cache_write_input_tokens": 2,
    }
    usage_details = usage_context.get_usage().details
    assert {detail["model_id"] for detail in usage_details} == {"bedrock-config-1"}
    assert usage_details[0]["cached_tokens"] == 4
    assert usage_details[0]["cache_write_tokens"] == 2
    stubber.assert_no_pending_responses()


@pytest.mark.asyncio
async def test_converse_round_trips_agent_tool_messages_and_multiple_calls():
    client = _runtime_client()
    stubber = Stubber(client)
    expected = {
        "modelId": "model",
        "messages": [
            {"role": "user", "content": [{"text": "Compare weather"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "call-1",
                            "name": "weather",
                            "input": {"city": "Paris"},
                        }
                    },
                    {
                        "toolUse": {
                            "toolUseId": "call-2",
                            "name": "weather",
                            "input": {"city": "Tokyo"},
                        }
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "call-1",
                            "content": [{"json": {"c": 18}}],
                        }
                    },
                    {
                        "toolResult": {
                            "toolUseId": "call-2",
                            "content": [{"json": {"c": 24}}],
                        }
                    },
                ],
            },
        ],
        "inferenceConfig": {"maxTokens": 4096},
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "name": "weather",
                        "description": "Get weather",
                        "strict": True,
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            }
                        },
                    }
                }
            ],
            "toolChoice": {"auto": {}},
        },
    }
    stubber.add_response(
        "converse",
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "Checking both."},
                        {
                            "toolUse": {
                                "toolUseId": "next-1",
                                "name": "weather",
                                "input": {"city": "Sydney"},
                            }
                        },
                        {
                            "toolUse": {
                                "toolUseId": "next-2",
                                "name": "weather",
                                "input": {"city": "London"},
                            }
                        },
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 20, "outputTokens": 8, "totalTokens": 28},
            "metrics": {"latencyMs": 5},
        },
        expected,
    )
    stubber.activate()
    llm = BedrockLLM("model", client=client)
    # Build the replay through the same Message contract AgentContext exposes
    # to runtime adapters, rather than a provider-specific fixture shape.
    messages = [
        Message.role_user("Compare weather").to_dict(),
        Message.role_assistant(
            "",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": '{"city":"Paris"}',
                    },
                },
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": '{"city":"Tokyo"}',
                    },
                },
            ],
        ).to_dict(),
        Message.role_tool('{"c":18}', tool_call_id="call-1").to_dict(),
        Message.role_tool('{"c":24}', tool_call_id="call-2").to_dict(),
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get weather",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]

    result = await llm.chat(messages, tools=tools, tool_choice="auto")

    assert result["type"] == "tool_call"
    assert result["content"] == "Checking both."
    assert result["finish_reason"] == "tool_calls"
    assert [call["id"] for call in result["tool_calls"]] == ["next-1", "next-2"]
    assert json.loads(result["tool_calls"][1]["function"]["arguments"]) == {
        "city": "London"
    }


class _FakeEventStream:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events
        self.closed = False

    def __iter__(self):
        return iter(self.events)

    def close(self) -> None:
        self.closed = True


class _FakeClient:
    def __init__(self, stream: _FakeEventStream) -> None:
        self.stream = stream
        self.request: dict[str, Any] | None = None

    def converse_stream(self, **request: Any) -> dict[str, Any]:
        self.request = request
        return {"stream": self.stream}


class _BlockingEventStream:
    def __init__(self) -> None:
        self.closed = False
        self._released = threading.Event()

    def __iter__(self):
        return self

    def __next__(self):
        self._released.wait(timeout=5)
        raise StopIteration

    def close(self) -> None:
        self.closed = True
        self._released.set()


@pytest.mark.asyncio
async def test_converse_stream_accumulates_fragmented_multi_tool_arguments():
    stream = _FakeEventStream(
        [
            {"messageStart": {"role": "assistant"}},
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"text": "Let me check."},
                }
            },
            {
                "contentBlockStart": {
                    "contentBlockIndex": 1,
                    "start": {"toolUse": {"toolUseId": "a", "name": "weather"}},
                }
            },
            {
                "contentBlockStart": {
                    "contentBlockIndex": 2,
                    "start": {"toolUse": {"toolUseId": "b", "name": "weather"}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 2,
                    "delta": {"toolUse": {"input": '{"city"'}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 1,
                    "delta": {"toolUse": {"input": '{"city":"Par'}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 2,
                    "delta": {"toolUse": {"input": ':"Tokyo"}'}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 1,
                    "delta": {"toolUse": {"input": 'is"}'}},
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 1}},
            {"contentBlockStop": {"contentBlockIndex": 2}},
            {"messageStop": {"stopReason": "tool_use"}},
            {
                "metadata": {
                    "usage": {
                        "inputTokens": 9,
                        "outputTokens": 4,
                        "totalTokens": 13,
                        "cacheReadInputTokens": 3,
                    },
                    "metrics": {"latencyMs": 2},
                }
            },
        ]
    )
    client = _FakeClient(stream)
    llm = BedrockLLM("model", client=client)

    chunks = [
        chunk async for chunk in llm.stream_chat([{"role": "user", "content": "go"}])
    ]

    assert chunks[0].type is ChunkType.TOKEN
    tool_chunks = [chunk for chunk in chunks if chunk.type is ChunkType.TOOL_CALL]
    final_by_id = {
        chunk.tool_calls[0]["id"]: chunk.tool_calls[0] for chunk in tool_chunks
    }
    assert final_by_id["a"]["function"]["arguments"] == '{"city":"Paris"}'
    assert final_by_id["b"]["function"]["arguments"] == '{"city":"Tokyo"}'
    assert all(
        call["function"]["arguments_mode"] == "delta" for call in final_by_id.values()
    )
    usage = next(chunk for chunk in chunks if chunk.type is ChunkType.USAGE)
    assert usage.usage["cached_input_tokens"] == 3
    assert usage.finish_reason == "tool_calls"
    assert chunks[-1].type is ChunkType.END
    assert chunks[-1].finish_reason == "tool_calls"
    assert stream.closed is True


@pytest.mark.asyncio
async def test_stream_cancellation_closes_blocking_botocore_event_stream():
    stream = _BlockingEventStream()
    client = _FakeClient(stream)
    llm = BedrockLLM("model", client=client)
    iterator = llm.stream_chat([{"role": "user", "content": "go"}])
    pending = asyncio.create_task(anext(iterator))
    for _ in range(100):
        if client.request is not None:
            break
        await asyncio.sleep(0.01)
    assert client.request is not None
    pending.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pending

    assert stream.closed is True


def test_explicit_api_keys_are_instance_scoped_and_never_touch_environment(monkeypatch):
    handlers: list[Any] = []
    configs: list[Any] = []

    class Events:
        def register(self, _name: str, handler: Any) -> None:
            handlers.append(handler)

    def fake_client(_service: str, **kwargs: Any):
        configs.append(kwargs["config"])
        return SimpleNamespace(meta=SimpleNamespace(events=Events()))

    monkeypatch.setattr(boto3, "client", fake_client)
    before = dict(os.environ)
    BedrockLLM("m", api_key="first", auth_mode="api_key")._build_client()
    BedrockLLM("m", api_key="second", auth_mode="api_key")._build_client()
    first_request = SimpleNamespace(headers={})
    second_request = SimpleNamespace(headers={})
    handlers[0](first_request)
    handlers[1](second_request)

    assert first_request.headers["Authorization"] == "Bearer first"
    assert second_request.headers["Authorization"] == "Bearer second"
    assert configs[0].signature_version is UNSIGNED
    assert dict(os.environ) == before
    assert "AWS_BEARER_TOKEN_BEDROCK" not in (set(os.environ) - set(before))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {"endpoint_url": "https://bedrock-mantle.us-east-1.api.aws"},
            "does not support Converse",
        ),
        ({"auth_mode": "api_key"}, "requires an api_key"),
        (
            {"api_key": "key", "auth_mode": "aws_credentials"},
            "cannot include an api_key",
        ),
    ],
)
def test_invalid_transport_configuration_fails_closed(kwargs, message):
    with pytest.raises(ValueError, match=message):
        BedrockLLM("model", **kwargs)


@pytest.mark.asyncio
async def test_unsupported_payloads_fail_instead_of_being_dropped():
    llm = BedrockLLM("model", client=object())
    with pytest.raises(ValueError, match="inline data URL"):
        await llm.chat(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/image.png"},
                        }
                    ],
                }
            ]
        )
    with pytest.raises(ValueError, match="JSON response-format"):
        await llm.chat(
            [{"role": "user", "content": "hi"}],
            response_format={"type": "json_object"},
        )
    with pytest.raises(ValueError, match="reasoning support"):
        await llm.chat(
            [{"role": "user", "content": "hi"}],
            thinking={"type": "enabled"},
        )


@pytest.mark.asyncio
async def test_client_error_is_redacted_and_transient_failures_are_retryable():
    client = _runtime_client()
    stubber = Stubber(client)
    stubber.add_client_error(
        "converse",
        service_error_code="ThrottlingException",
        service_message="Authorization: Bearer very-secret api_key=also-secret",
        http_status_code=429,
        expected_params={
            "modelId": "model",
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            "inferenceConfig": {"maxTokens": 4096},
        },
    )
    stubber.activate()
    llm = BedrockLLM("model", client=client)

    with pytest.raises(Exception) as raised:
        await llm.chat([{"role": "user", "content": "hi"}])

    assert type(raised.value).__name__ == "LLMRetryableError"
    assert "very-secret" not in str(raised.value)
    assert "also-secret" not in str(raised.value)
    assert "ThrottlingException" in str(raised.value)
