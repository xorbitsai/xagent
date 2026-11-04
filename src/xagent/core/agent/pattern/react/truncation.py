"""Tool calls cut off at the model's output length limit (xorbitsai/xagent#2850).

A model that writes a long deliverable straight into a tool call (a whole
report into ``write_file``) can hit the provider's output cap mid-JSON. The
response then stops with ``finish_reason == "length"`` and the arguments do
not parse. Resending the same request under the same cap fails the same way,
so ReAct retries with an instruction to split the work, and reports the
failure under its own code when the retry is cut off too.

Two response shapes are recognised:

* a provider tool-protocol error (DeepSeek's ``malformed_tool_arguments``)
  that carries the length stop (``length``, or ``max_tokens`` on a streamed
  Anthropic response: non-streamed Claude drops a cut-off call's tool use), and
* a provider that passes arguments through unvalidated: a tool call whose
  arguments string is not valid JSON, under the same length stop.

The length stop is required in both. Without it a malformed call is an
ordinary formatting error, which keeps its existing repair.
"""

from __future__ import annotations

import json
from typing import Any

from ....model.chat.tool_protocol import get_tool_protocol_error

TRUNCATED_TOOL_ARGUMENTS = "truncated_tool_arguments"
TRUNCATED_TOOL_ARGUMENTS_ERROR = (
    "The model's output reached its output length limit while writing a tool "
    "call, so the call could not run, and one repair attempt did not fix it."
)

_SPLIT_WRITE_TOOL = "append_file"
_LENGTH_STOPS = frozenset({"length", "max_tokens"})


def response_finish_reason(response: Any) -> str | None:
    """The provider's stop reason on a response, when it reports one."""
    if not isinstance(response, dict):
        return None
    finish_reason = response.get("finish_reason")
    return finish_reason if isinstance(finish_reason, str) and finish_reason else None


def is_truncated_tool_call(response: Any) -> bool:
    """Whether a response's tool-call arguments were cut off at the length limit."""
    if response_finish_reason(response) not in _LENGTH_STOPS:
        return False
    protocol_error = get_tool_protocol_error(response)
    if protocol_error is not None:
        return protocol_error.get("code") == "malformed_tool_arguments"
    return any(
        _has_unparsable_arguments(tool_call)
        for tool_call in response.get("tool_calls") or []
    )


def truncated_tool_arguments_instruction(
    tool_names: list[str], *, force_final_answer: bool
) -> str:
    """Retry instruction for a turn whose tool call was cut off."""
    lead = (
        "The previous response reached the model's output length limit while "
        "writing tool-call arguments, so the call was cut off and nothing ran. "
        "Resending the same call will be cut off again. "
    )
    if force_final_answer:
        return (
            lead + "This turn only accepts an answer: call final_answer with a "
            "shorter answer that fits in one response."
        )
    split = "Split the work into several smaller tool calls. "
    if _SPLIT_WRITE_TOOL in tool_names:
        # The cut-off tool is not always known (a protocol-error envelope
        # carries no tool calls), so the file advice is conditional in wording.
        split += (
            "If the cut-off call was writing a file, create it with the first "
            f"part, then add each further part with {_SPLIT_WRITE_TOOL} in "
            "later turns. "
        )
    return (
        lead + split + "Keep every tool call's arguments well under the limit. "
        "If the user only needs the answer itself, call final_answer with a "
        "response that fits in one turn."
    )


def _has_unparsable_arguments(tool_call: Any) -> bool:
    if not isinstance(tool_call, dict):
        return False
    function = tool_call.get("function")
    arguments = (
        function.get("arguments")
        if isinstance(function, dict)
        else tool_call.get("arguments")
    )
    if not isinstance(arguments, str) or not arguments.strip():
        return False
    try:
        json.loads(arguments)
    except json.JSONDecodeError:
        return True
    return False
