from __future__ import annotations

import json

from tests.e2e.scripted_llm import load_scripted_responses


def test_load_scripted_responses_converts_enveloped_entries(tmp_path):
    responses_path = tmp_path / "responses.json"
    responses_path.write_text(
        json.dumps(
            [
                {
                    "kind": "json_text",
                    "content": {
                        "type": "tool_call",
                        "reasoning": "Use a tool.",
                    },
                },
                {
                    "kind": "tool_call",
                    "tool_calls": [
                        {
                            "id": "call_write",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": {
                                    "file_path": "result.txt",
                                    "content": "hello\n",
                                },
                            },
                        }
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )

    assert load_scripted_responses(responses_path) == [
        {
            "content": '{"type": "tool_call", "reasoning": "Use a tool."}',
            "done": False,
        },
        {
            "type": "tool_call",
            "tool_calls": [
                {
                    "id": "call_write",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": '{"file_path": "result.txt", "content": "hello\\n"}',
                    },
                }
            ],
        },
    ]
