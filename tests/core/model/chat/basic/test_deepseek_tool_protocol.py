from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from xagent.core.model.chat.basic.deepseek_tool_protocol import (
    adapt_deepseek_stream,
    normalize_deepseek_response,
)
from xagent.core.model.chat.tool_protocol import get_tool_protocol_error
from xagent.core.model.chat.types import ChunkType, StreamChunk


def _tool_schema(
    name: str,
    properties: dict[str, dict[str, str]],
    *,
    additional_properties: bool = True,
) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Call {name}.",
            "parameters": {
                "type": "object",
                "additionalProperties": additional_properties,
                "properties": properties,
            },
        },
    }


FINAL_ANSWER_TOOL = _tool_schema(
    "final_answer",
    {
        "response_language": {"type": "string"},
        "answer": {"type": "string"},
    },
    additional_properties=False,
)
WRITE_FILE_TOOL = _tool_schema(
    "write_file",
    {
        "file_path": {"type": "string"},
        "content": {"type": "string"},
    },
)


def test_deepseek_codec_rejects_serialized_tool_call_content() -> None:
    response = {
        "type": "text",
        "content": (
            "Let me continue.\n\n"
            "<｜｜DSML｜｜tool_calls>\n"
            '<｜｜DSML｜｜invoke name="fetch_web_content">'
        ),
        "raw": {"id": "task-722-shape"},
    }

    normalized = normalize_deepseek_response(
        response,
        tools=[FINAL_ANSWER_TOOL],
    )

    error = get_tool_protocol_error(normalized)
    assert error is not None
    assert error["provider"] == "deepseek"
    assert error["code"] == "serialized_tool_call_content"
    assert normalized["tool_calls"] == []
    assert normalized["content"] == ""


def test_deepseek_codec_rejects_same_line_serialized_tool_call_content() -> None:
    response = {
        "type": "text",
        "content": "Sure: <｜｜DSML｜｜tool_calls>",
    }

    normalized = normalize_deepseek_response(
        response,
        tools=[FINAL_ANSWER_TOOL],
    )

    error = get_tool_protocol_error(normalized)
    assert error is not None
    assert error["code"] == "serialized_tool_call_content"


def test_deepseek_codec_rejects_nested_work_call_in_final_answer() -> None:
    response = {
        "type": "tool_call",
        "tool_calls": [
            {
                "id": "call_malformed_final",
                "type": "function",
                "function": {
                    "name": "final_answer",
                    "arguments": json.dumps(
                        {
                            "response_language": "English",
                            "answer": (
                                "I'll write the script.\n\n"
                                "<｜｜DSML｜｜tool_calls>\n"
                                '<｜｜DSML｜｜invoke name="write_file">\n'
                                '<｜｜DSML｜｜parameter name="file_path" '
                                'string="true">podcast.md'
                            ),
                            "content": "# Podcast script",
                        },
                        ensure_ascii=False,
                    ),
                },
            }
        ],
    }

    normalized = normalize_deepseek_response(
        response,
        tools=[WRITE_FILE_TOOL, FINAL_ANSWER_TOOL],
    )

    error = get_tool_protocol_error(normalized)
    assert error is not None
    assert error["code"] == "nested_serialized_tool_call"


@pytest.mark.parametrize(
    ("tool_call", "tools", "expected_code"),
    [
        (
            {"function": {"arguments": "{}"}},
            [WRITE_FILE_TOOL],
            "malformed_tool_call",
        ),
        (
            {
                "function": {
                    "name": "write_file",
                    "arguments": '{"file_path":',
                }
            },
            [WRITE_FILE_TOOL],
            "malformed_tool_arguments",
        ),
        (
            {
                "function": {
                    "name": "fetch_web_content",
                    "arguments": "{}",
                }
            },
            [WRITE_FILE_TOOL],
            "unavailable_tool_call",
        ),
        (
            {
                "function": {
                    "name": "final_answer",
                    "arguments": json.dumps(
                        {
                            "response_language": "English",
                            "answer": "Done.",
                            "content": "unexpected",
                        }
                    ),
                }
            },
            [FINAL_ANSWER_TOOL],
            "unexpected_tool_arguments",
        ),
    ],
)
def test_deepseek_codec_reports_structured_tool_violations(
    tool_call, tools, expected_code
) -> None:
    normalized = normalize_deepseek_response(
        {"type": "tool_call", "tool_calls": [tool_call]},
        tools=tools,
    )

    error = get_tool_protocol_error(normalized)
    assert error is not None
    assert error["code"] == expected_code


def test_deepseek_codec_keeps_valid_tool_call() -> None:
    response = {
        "type": "tool_call",
        "tool_calls": [
            {
                "id": "call_write",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps(
                        {"file_path": "podcast.md", "content": "script"}
                    ),
                },
            }
        ],
    }

    assert normalize_deepseek_response(response, tools=[WRITE_FILE_TOOL]) is response


def test_deepseek_codec_repairs_complete_malformed_tool_arguments() -> None:
    response = {
        "type": "tool_call",
        "tool_calls": [
            {
                "id": "call_final",
                "type": "function",
                "function": {
                    "name": "final_answer",
                    "arguments": (
                        "{'response_language': 'English', 'answer': 'Done.',}"
                    ),
                },
            }
        ],
    }

    normalized = normalize_deepseek_response(
        response,
        tools=[FINAL_ANSWER_TOOL],
    )

    assert normalized is response
    repaired = json.loads(response["tool_calls"][0]["function"]["arguments"])
    assert repaired == {
        "response_language": "English",
        "answer": "Done.",
    }


def test_deepseek_codec_repairs_brace_inside_single_quoted_string() -> None:
    response = {
        "type": "tool_call",
        "tool_calls": [
            {
                "id": "call_write",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": (
                        "{'file_path':'artifact.txt','content':'Use {value',}"
                    ),
                },
            }
        ],
    }

    normalized = normalize_deepseek_response(
        response,
        tools=[WRITE_FILE_TOOL],
    )

    assert normalized is response
    repaired = json.loads(response["tool_calls"][0]["function"]["arguments"])
    assert repaired == {
        "file_path": "artifact.txt",
        "content": "Use {value",
    }


def test_deepseek_codec_marks_non_object_repair_as_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "xagent.core.model.chat.basic.deepseek_tool_protocol.repair_json_loads",
        lambda *_args, **_kwargs: [],
    )
    normalized = normalize_deepseek_response(
        {
            "type": "tool_call",
            "tool_calls": [
                {
                    "function": {
                        "name": "write_file",
                        "arguments": "{broken}",
                    }
                }
            ],
        },
        tools=[WRITE_FILE_TOOL],
    )

    error = get_tool_protocol_error(normalized)
    assert error is not None
    assert error["code"] == "malformed_tool_arguments"
    assert error["details"]["repair_status"] == "failed_non_dict"


def test_deepseek_codec_accepts_empty_object_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "xagent.core.model.chat.basic.deepseek_tool_protocol.repair_json_loads",
        lambda *_args, **_kwargs: {},
    )
    response = {
        "type": "tool_call",
        "tool_calls": [
            {
                "function": {
                    "name": "ping",
                    "arguments": "{broken}",
                }
            }
        ],
    }

    normalized = normalize_deepseek_response(
        response,
        tools=[_tool_schema("ping", {}, additional_properties=False)],
    )

    assert normalized is response
    assert response["tool_calls"][0]["function"]["arguments"] == "{}"


def test_deepseek_codec_keeps_original_arguments_when_repair_is_unsafe() -> None:
    original_arguments = '{"file_path":'

    normalized = normalize_deepseek_response(
        {
            "type": "tool_call",
            "tool_calls": [
                {
                    "function": {
                        "name": "write_file",
                        "arguments": original_arguments,
                    }
                }
            ],
        },
        tools=[WRITE_FILE_TOOL],
    )

    error = get_tool_protocol_error(normalized)
    assert error is not None
    assert error["code"] == "malformed_tool_arguments"
    assert error["details"] == {
        "original_arguments_preview": original_arguments,
        "original_arguments_length": len(original_arguments),
        "original_arguments_truncated": False,
        "json_error": "Expecting value: line 1 column 14 (char 13)",
        "repair_status": "skipped_incomplete",
    }


def test_deepseek_codec_bounds_original_argument_diagnostics() -> None:
    original_arguments = '{"answer":"' + ("x" * 5000)

    normalized = normalize_deepseek_response(
        {
            "type": "tool_call",
            "tool_calls": [
                {
                    "function": {
                        "name": "final_answer",
                        "arguments": original_arguments,
                    }
                }
            ],
        },
        tools=[FINAL_ANSWER_TOOL],
    )

    error = get_tool_protocol_error(normalized)
    assert error is not None
    details = error["details"]
    assert details["original_arguments_length"] == len(original_arguments)
    assert len(details["original_arguments_preview"]) == 4096
    assert details["original_arguments_truncated"] is True


def test_deepseek_codec_is_inactive_without_requested_tools() -> None:
    response = {
        "type": "text",
        "content": "A DSML tool_calls marker can be discussed as ordinary text.",
    }

    assert normalize_deepseek_response(response, tools=None) is response


@pytest.mark.asyncio
async def test_deepseek_stream_suppresses_serialized_protocol_tokens() -> None:
    async def source() -> AsyncIterator[StreamChunk]:
        yield StreamChunk(
            type=ChunkType.TOKEN,
            delta="Let me continue.\n\n<｜｜D",
        )
        yield StreamChunk(
            type=ChunkType.TOKEN,
            delta="SML｜｜tool_calls>\n<｜｜DSML｜｜invoke",
        )
        yield StreamChunk(type=ChunkType.END, finish_reason="stop")

    chunks = [
        chunk
        async for chunk in adapt_deepseek_stream(
            source(),
            tools=[FINAL_ANSWER_TOOL],
        )
    ]

    streamed_text = "".join(chunk.delta for chunk in chunks if chunk.is_token())
    assert streamed_text == "Let me continue.\n\n"
    assert "DSML" not in streamed_text
    protocol_errors = [chunk for chunk in chunks if chunk.is_protocol_error()]
    assert len(protocol_errors) == 1
    assert protocol_errors[0].protocol_error["code"] == ("serialized_tool_call_content")


@pytest.mark.asyncio
async def test_deepseek_stream_suppresses_same_line_split_marker() -> None:
    async def source() -> AsyncIterator[StreamChunk]:
        yield StreamChunk(
            type=ChunkType.TOKEN,
            delta="Sure: <｜",
        )
        yield StreamChunk(
            type=ChunkType.TOKEN,
            delta="｜DSML｜｜tool_calls>",
        )
        yield StreamChunk(type=ChunkType.END, finish_reason="stop")

    chunks = [
        chunk
        async for chunk in adapt_deepseek_stream(
            source(),
            tools=[FINAL_ANSWER_TOOL],
        )
    ]

    streamed_text = "".join(chunk.delta for chunk in chunks if chunk.is_token())
    assert streamed_text == "Sure: "
    assert len([chunk for chunk in chunks if chunk.is_protocol_error()]) == 1


@pytest.mark.asyncio
async def test_deepseek_stream_replaces_nested_final_answer_with_protocol_error() -> (
    None
):
    malformed_arguments = json.dumps(
        {
            "response_language": "English",
            "answer": (
                "Drafting now.\n\n"
                "<｜｜DSML｜｜tool_calls>\n"
                '<｜｜DSML｜｜invoke name="write_file">'
            ),
            "content": "script",
        },
        ensure_ascii=False,
    )

    async def source() -> AsyncIterator[StreamChunk]:
        yield StreamChunk(
            type=ChunkType.TOOL_CALL,
            tool_calls=[
                {
                    "id": "call_malformed_final",
                    "type": "function",
                    "function": {
                        "name": "final_answer",
                        "arguments": malformed_arguments,
                    },
                }
            ],
        )
        yield StreamChunk(type=ChunkType.END, finish_reason="tool_calls")

    chunks = [
        chunk
        async for chunk in adapt_deepseek_stream(
            source(),
            tools=[WRITE_FILE_TOOL, FINAL_ANSWER_TOOL],
        )
    ]

    assert not any(chunk.is_tool_call() for chunk in chunks)
    protocol_errors = [chunk for chunk in chunks if chunk.is_protocol_error()]
    assert len(protocol_errors) == 1
    assert protocol_errors[0].protocol_error["code"] == ("nested_serialized_tool_call")


@pytest.mark.asyncio
async def test_deepseek_stream_forwards_accumulated_valid_tool_calls() -> None:
    async def source() -> AsyncIterator[StreamChunk]:
        yield StreamChunk(
            type=ChunkType.TOOL_CALL,
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_write",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": '{"file_path":',
                    },
                }
            ],
        )
        yield StreamChunk(
            type=ChunkType.TOOL_CALL,
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_write",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": ('{"file_path":"podcast.md","content":"script"}'),
                    },
                }
            ],
            finish_reason="tool_calls",
        )

    chunks = [
        chunk
        async for chunk in adapt_deepseek_stream(
            source(),
            tools=[WRITE_FILE_TOOL],
        )
    ]

    tool_chunks = [chunk for chunk in chunks if chunk.is_tool_call()]
    assert len(tool_chunks) == 2
    assert tool_chunks[-1].tool_calls[0]["function"]["arguments"] == (
        '{"file_path":"podcast.md","content":"script"}'
    )


@pytest.mark.asyncio
async def test_deepseek_stream_repairs_complete_malformed_tool_arguments() -> None:
    async def source() -> AsyncIterator[StreamChunk]:
        yield StreamChunk(
            type=ChunkType.TOOL_CALL,
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_final",
                    "type": "function",
                    "function": {
                        "name": "final_answer",
                        "arguments": (
                            "{'response_language': 'English', 'answer': 'Done.',}"
                        ),
                    },
                }
            ],
            finish_reason="tool_calls",
        )

    chunks = [
        chunk
        async for chunk in adapt_deepseek_stream(
            source(),
            tools=[FINAL_ANSWER_TOOL],
        )
    ]

    assert not any(chunk.is_protocol_error() for chunk in chunks)
    tool_chunks = [chunk for chunk in chunks if chunk.is_tool_call()]
    assert len(tool_chunks) == 1
    repaired = json.loads(tool_chunks[0].tool_calls[0]["function"]["arguments"])
    assert repaired == {
        "response_language": "English",
        "answer": "Done.",
    }


@pytest.mark.asyncio
async def test_deepseek_stream_records_truncated_original_arguments() -> None:
    original_arguments = '{"response_language":"English","answer":'

    async def source() -> AsyncIterator[StreamChunk]:
        yield StreamChunk(
            type=ChunkType.TOOL_CALL,
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_final",
                    "type": "function",
                    "function": {
                        "name": "final_answer",
                        "arguments": original_arguments,
                    },
                }
            ],
            finish_reason="tool_calls",
        )

    chunks = [
        chunk
        async for chunk in adapt_deepseek_stream(
            source(),
            tools=[FINAL_ANSWER_TOOL],
        )
    ]

    protocol_errors = [chunk for chunk in chunks if chunk.is_protocol_error()]
    assert len(protocol_errors) == 1
    details = protocol_errors[0].protocol_error["details"]
    assert details["original_arguments_preview"] == original_arguments
    assert details["original_arguments_length"] == len(original_arguments)
    assert details["repair_status"] == "skipped_incomplete"


@pytest.mark.asyncio
async def test_deepseek_stream_withholds_split_nested_marker_from_tool_args() -> None:
    async def source() -> AsyncIterator[StreamChunk]:
        yield StreamChunk(
            type=ChunkType.TOOL_CALL,
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_final",
                    "type": "function",
                    "function": {
                        "name": "final_answer",
                        "arguments": (
                            '{"response_language":"English","answer":"Draft: <｜'
                        ),
                    },
                }
            ],
        )
        yield StreamChunk(
            type=ChunkType.TOOL_CALL,
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_final",
                    "type": "function",
                    "function": {
                        "name": "final_answer",
                        "arguments": (
                            '{"response_language":"English",'
                            '"answer":"Draft: <｜｜DSML｜｜tool_calls>"}'
                        ),
                    },
                }
            ],
            finish_reason="tool_calls",
        )

    chunks = [
        chunk
        async for chunk in adapt_deepseek_stream(
            source(),
            tools=[FINAL_ANSWER_TOOL],
        )
    ]

    forwarded_arguments = [
        chunk.tool_calls[0]["function"]["arguments"]
        for chunk in chunks
        if chunk.is_tool_call()
    ]
    assert forwarded_arguments == ['{"response_language":"English","answer":"Draft: ']
    assert all("DSML" not in arguments for arguments in forwarded_arguments)
    protocol_errors = [chunk for chunk in chunks if chunk.is_protocol_error()]
    assert len(protocol_errors) == 1
    assert protocol_errors[0].protocol_error["code"] == ("nested_serialized_tool_call")
