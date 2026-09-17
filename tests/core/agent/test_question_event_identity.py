from __future__ import annotations

from typing import Any

import pytest

from xagent.core.agent import PatternRuntime


class OutboundCollector:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def __call__(self, payload: dict[str, Any]) -> None:
        self.events.append(dict(payload))


@pytest.mark.asyncio
async def test_waiting_message_has_one_identity_before_publication() -> None:
    outbound = OutboundCollector()
    runtime = PatternRuntime(
        execution_id="exec-1",
        outbound_message_handler=outbound,
    )

    emitted = await runtime.send_message(
        message="Which region should I use?",
        message_type="question",
        expect_response=True,
    )

    assert isinstance(emitted["event_id"], str)
    assert emitted["event_id"]
    assert outbound.events == [emitted]


@pytest.mark.asyncio
async def test_non_waiting_message_keeps_the_legacy_runtime_payload_shape() -> None:
    outbound = OutboundCollector()
    runtime = PatternRuntime(
        execution_id="exec-1",
        outbound_message_handler=outbound,
    )

    emitted = await runtime.send_message(
        message="Still working",
        message_type="info",
        expect_response=False,
    )

    assert "event_id" not in emitted
    assert outbound.events == [emitted]


@pytest.mark.asyncio
async def test_question_message_has_identity_even_without_response_waiting() -> None:
    outbound = OutboundCollector()
    runtime = PatternRuntime(
        execution_id="exec-1",
        outbound_message_handler=outbound,
    )

    emitted = await runtime.send_message(
        message="Would you like a status summary?",
        message_type="question",
        expect_response=False,
    )

    assert emitted["event_id"]
    assert outbound.events == [emitted]
