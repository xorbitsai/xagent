from __future__ import annotations

from typing import Any

import pytest

from xagent.core.agent import PatternRuntime
from xagent.web.api.websocket import create_stream_event, make_agent_outbound_handler


@pytest.mark.asyncio
async def test_question_identity_is_shared_by_runtime_persistence_and_broadcast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persisted: list[dict[str, Any]] = []
    broadcast: list[dict[str, Any]] = []

    def fake_persist(_task_id: int, event: dict[str, Any]) -> None:
        persisted.append(dict(event))

    async def fake_broadcast(event: dict[str, Any], _task_id: int) -> None:
        broadcast.append(dict(event))

    monkeypatch.setattr(
        "xagent.web.api.websocket._persist_agent_outbound_event",
        fake_persist,
    )
    monkeypatch.setattr(
        "xagent.web.api.websocket.manager.broadcast_to_task",
        fake_broadcast,
    )

    runtime = PatternRuntime(
        execution_id="exec-1",
        outbound_message_handler=make_agent_outbound_handler(365),
    )
    emitted = await runtime.send_message(
        message="Which region should I use?",
        message_type="question",
        expect_response=True,
    )

    assert len(persisted) == len(broadcast) == 1
    assert persisted[0]["event_id"] == emitted["event_id"]
    assert broadcast[0]["event_id"] == emitted["event_id"]
    assert persisted[0]["data"]["event_id"] == emitted["event_id"]
    assert broadcast[0]["data"]["event_id"] == emitted["event_id"]


def test_generic_stream_event_does_not_promote_nested_domain_identity() -> None:
    event = create_stream_event(
        "tool_event",
        365,
        {"event_id": "tool-domain-event-1"},
    )

    assert event["event_id"] != "tool-domain-event-1"
    assert event["data"]["event_id"] == "tool-domain-event-1"
