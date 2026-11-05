"""Failed tool invocations are a counter of their own on ``TokenUsage``."""

from __future__ import annotations

from xagent.core.model.chat import token_context
from xagent.core.model.chat.token_context import (
    TokenUsage,
    get_token_usage,
    set_token_usage,
)
from xagent.web.tracking.task_tracker import _copy_usage


def _usage() -> TokenUsage:
    return TokenUsage(
        input_tokens=11,
        output_tokens=7,
        llm_calls=2,
        tool_calls=5,
        tool_errors=3,
        details=[{"type": "input", "tokens": 11, "model": "m"}],
    )


def test_copy_keeps_every_counter() -> None:
    usage = _usage()

    assert _copy_usage(usage).to_dict() == usage.to_dict()


def test_dict_round_trip_keeps_tool_errors() -> None:
    usage = _usage()

    assert usage.to_dict()["tool_errors"] == 3
    assert TokenUsage.from_dict(usage.to_dict()) == usage


def test_from_dict_defaults_missing_tool_errors_to_zero() -> None:
    assert TokenUsage.from_dict({"tool_calls": 2}).tool_errors == 0


def test_merge_adds_tool_errors() -> None:
    usage = _usage()
    usage.merge(TokenUsage(tool_calls=1, tool_errors=2))

    assert usage.tool_calls == 6
    assert usage.tool_errors == 5


def test_add_tool_error_usage_counts_on_the_current_context() -> None:
    usage = set_token_usage(TokenUsage())

    token_context.add_tool_error_usage()
    token_context.add_tool_error_usage(2)

    assert get_token_usage() is usage
    assert usage.tool_errors == 3
    assert usage.tool_calls == 0
