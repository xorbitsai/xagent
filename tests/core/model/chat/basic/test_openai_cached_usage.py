"""Streamed usage must carry the provider's cached-token count to the trace.

The trace's ``llm_call_end`` reads ``cached_input_tokens`` from the usage the
stream hands back. The adapter recorded the count in the token context but left
it out of that usage, so the trace never showed a cache hit.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from xagent.core.agent.runtime import PatternRuntime
from xagent.core.model.chat.basic.openai import OpenAILLM
from xagent.core.model.chat.types import ChunkType


def _usage(**extra: Any) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=1000, completion_tokens=20, total_tokens=1020, **extra
    )


def _parse(usage: SimpleNamespace) -> dict[str, Any]:
    llm = OpenAILLM.__new__(OpenAILLM)
    llm._model_name = "m"  # type: ignore[attr-defined]
    llm._model_id = None  # type: ignore[attr-defined]
    chunk = SimpleNamespace(choices=[], usage=usage)
    (parsed,) = llm._parse_stream_chunks(chunk, {})
    assert parsed.type is ChunkType.USAGE
    return dict(parsed.usage)


@pytest.mark.parametrize(
    "extra",
    [
        {"prompt_tokens_details": SimpleNamespace(cached_tokens=400)},
        {"prompt_cache_hit_tokens": 400},
    ],
)
def test_streamed_usage_carries_cached_tokens(extra: dict[str, Any]) -> None:
    usage = _parse(_usage(**extra))

    assert usage["cached_input_tokens"] == 400
    assert usage["prompt_tokens"] == 1000
    # And the runtime reads it back into the trace field.
    runtime = PatternRuntime(execution_id="cached")
    assert runtime._extract_cached_tokens({"usage": usage}) == 400


def test_streamed_usage_without_a_cache_report_has_no_cached_field() -> None:
    assert "cached_input_tokens" not in _parse(_usage())
