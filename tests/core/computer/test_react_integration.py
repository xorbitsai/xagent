from __future__ import annotations

import json
from typing import Any

import pytest

from xagent.core.agent import ExecutionContext, PatternRuntime, ReActPattern
from xagent.core.context_ref import ContextReference


class Resolver:
    async def resolve_image(self, reference: ContextReference) -> str:
        return "data:image/png;base64,c2NyZWVuc2hvdA=="


class VisionLLM:
    abilities = ["chat", "vision"]

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"content": "The settings dialog is visible.", "done": True}


class CapturingTracer:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def trace_event(self, event_type: Any, **kwargs: Any) -> None:
        self.events.append(
            {
                "type": getattr(event_type, "value", str(event_type)),
                "data": kwargs.get("data") or {},
            }
        )


@pytest.mark.asyncio
async def test_existing_react_pattern_materializes_refs_only_at_llm_boundary() -> None:
    reference = ContextReference(
        file_ref={
            "file_id": "image-1",
            "filename": "frame.png",
            "mime_type": "image/png",
        },
        text_fallback="Current browser frame",
    )
    context = ExecutionContext(system_prompt="Inspect the browser.")
    context.add_user_message("What is visible?", context_refs=[reference])
    tracer = CapturingTracer()
    runtime = PatternRuntime(
        tracer=tracer,
        execution_id="computer-react",
        context_ref_resolver=Resolver(),
    )
    llm = VisionLLM()

    result = await ReActPattern(max_iterations=1).run(
        context=context,
        tools=[],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    provider_user_message = next(
        message for message in llm.calls[0]["messages"] if message["role"] == "user"
    )
    assert provider_user_message["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )

    durable_payload = json.dumps(context.to_dict())
    trace_payload = json.dumps(tracer.events)
    assert "image-1" in durable_payload
    assert "base64" not in durable_payload
    assert "image-1" in trace_payload
    assert "base64" not in trace_payload
