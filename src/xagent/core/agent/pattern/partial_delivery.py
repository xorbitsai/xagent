from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from ...model.chat.tool_protocol import get_tool_protocol_error
from ..runtime import (
    ExecutionInterrupted,
    PatternRuntime,
    prepare_llm_for_context,
    resolved_llm_metadata,
)


async def request_partial_delivery(
    *,
    context: Any,
    llm: Any,
    runtime: PatternRuntime,
    messages: list[dict[str, Any]],
    schema: dict[str, Any],
    parse_response: Callable[[Any], dict[str, Any] | None],
    metadata: dict[str, Any],
    timeout: float = 30.0,
) -> dict[str, Any] | None:
    """One buffered answer-only call. Never execute tools or repair/retry it.

    Patterns own the evidence, attempt checkpoint, and terminal status. Keeping
    this call buffered prevents unaccepted responses from reaching chat output.
    """
    metadata = dict(metadata)
    try:
        async with asyncio.timeout(timeout):
            call_llm = await prepare_llm_for_context(
                llm=llm, messages=messages, context=context
            )
            metadata.update(resolved_llm_metadata(call_llm))
            await runtime.on_llm_start(
                context=context, messages=messages, tools=[schema], metadata=metadata
            )
            response = await runtime.run_llm_call(
                call_llm,
                messages=messages,
                tools=[schema],
                tool_choice={"type": "function", "function": {"name": "final_answer"}},
                max_tokens=2048,
            )
        args = (
            parse_response(response)
            if get_tool_protocol_error(response) is None
            else None
        )
        if args is not None and not Draft202012Validator(
            schema["function"]["parameters"]
        ).is_valid(args):
            args = None
        await runtime.on_llm_end(
            context=context,
            response=response,
            metadata={**metadata, "success": args is not None},
        )
        return args
    except ExecutionInterrupted:
        raise
    except Exception as exc:
        await runtime.on_llm_error(context=context, error=exc, metadata=metadata)
        return None
