"""Shared fakes for the forced-answer compaction tests.

Collected here rather than redeclared per file because one of these pieces is
a trap: a compact model with no ``context_window`` makes ``PatternRuntime``
refuse compaction outright, so a test that forgets it passes for the wrong
reason. ``WindowlessCompactingLLM`` is that same fact used deliberately.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from xagent.core.agent import ExecutionContext
from xagent.core.agent.context import ContextManager

OBSERVATION_MARKER = "CLIENT_ROW_MARKER_{index}"


class ScriptedLLM:
    """A chat model that replays queued responses and records its calls."""

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.responses:
            return {"content": "done", "tool_calls": []}
        return self.responses.pop(0)


class CompactingLLM(ScriptedLLM):
    """A compact model that declares a window, so compaction is not refused."""

    context_window = 200_000
    summary = "COMPACTION_SUMMARY_MARKER: the agent listed clients."

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return {"content": self.summary, "tool_calls": []}


class WindowlessCompactingLLM(ScriptedLLM):
    """A compact model with no window, which makes the runtime refuse to compact."""

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return {"content": "unused", "tool_calls": []}


def build_context(
    *,
    observations: int = 6,
    threshold: int | None = None,
) -> ExecutionContext:
    """A context holding ``observations`` tool results, each uniquely marked."""
    context = ContextManager().create_context(
        execution_id="exec-forced-answer",
        system_prompt="You are helpful.",
    )
    context.add_user_message("List every client and their shifts.")
    for index in range(observations):
        context.add_assistant_message(
            "",
            tool_calls=[
                {
                    "id": f"call-{index}",
                    "type": "function",
                    "function": {"name": "list_clients", "arguments": "{}"},
                }
            ],
        )
        context.add_tool_result(
            tool_call_id=f"call-{index}",
            tool_name="list_clients",
            result={
                "success": True,
                "rows": [OBSERVATION_MARKER.format(index=index)] * 40,
            },
        )
    context.compact_config.max_messages = 4
    if threshold is not None:
        context.compact_config.threshold = threshold
    return context


def prompt_text(messages: list[dict[str, Any]]) -> str:
    """Flatten one prompt's messages into a single searchable string."""
    return "\n".join(str(message.get("content", "")) for message in messages)


class _CalculatorArgs(BaseModel):
    expression: str = ""


class CalculatorTool:
    """A minimal tool whose call always succeeds.

    Used for the turns that must go through a real tool call rather than
    ending on ``final_answer`` right away; the result is a fixed value, not
    an evaluated expression, since nothing here depends on the arithmetic.
    """

    def __init__(self) -> None:
        class Metadata:
            name = "calculator"
            description = "Evaluate a simple arithmetic expression."

        self.metadata = Metadata()

    def args_type(self) -> type[BaseModel]:
        return _CalculatorArgs

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        return {"success": True, "result": 2}
