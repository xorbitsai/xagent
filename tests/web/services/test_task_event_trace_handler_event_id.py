"""Live trace frames carry the identity of the persisted trace row."""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from xagent.core.agent.trace import (
    TraceAction,
    TraceCategory,
    TraceEvent,
    TraceEventType,
    TraceScope,
)
from xagent.web.services import task_event_trace_handler as handler_module
from xagent.web.services.task_event_trace_handler import TaskEventTraceHandler

TASK_ID = 4242


def _ai_message_event() -> TraceEvent:
    return TraceEvent(
        event_type=TraceEventType(
            TraceScope.TASK, TraceAction.UPDATE, TraceCategory.MESSAGE
        ),
        task_id=str(TASK_ID),
        data={"content": "hello"},
    )


@pytest.fixture
def published(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []

    async def fake_publish(frame: dict[str, Any], task_id: int) -> None:
        frames.append(frame)

    async def no_description(self: TaskEventTraceHandler) -> None:
        return None

    monkeypatch.setattr(handler_module, "publish_task_event", fake_publish)
    monkeypatch.setattr(handler_module, "task_has_audience", lambda task_id: True)
    monkeypatch.setattr(TaskEventTraceHandler, "_load_task_description", no_description)
    monkeypatch.setattr(
        TaskEventTraceHandler,
        "_has_prior_user_message_turn",
        lambda self, *args, **kwargs: False,
    )
    return frames


@pytest.mark.asyncio
async def test_non_authoritative_live_frame_uses_persisted_event_id(
    published: list[dict[str, Any]],
) -> None:
    event = _ai_message_event()

    await TaskEventTraceHandler(TASK_ID).handle_event(event)

    assert len(published) == 1
    # The persisted row is written with event_id=str(event.id).
    assert published[0]["event_id"] == str(event.id)


@pytest.mark.asyncio
async def test_authoritative_branch_frame_is_not_rewritten(
    published: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    authoritative_frame = {
        "type": "trace_event",
        "event_id": str(uuid.uuid4()),
        "message_id": "message-from-display-store",
        "event_type": "ai_message",
        "task_id": TASK_ID,
        "data": {"content": "hello"},
    }
    monkeypatch.setattr(
        TaskEventTraceHandler,
        "_load_event_display",
        lambda self, protocol_event_id: dict(authoritative_frame),
    )

    await TaskEventTraceHandler(TASK_ID, authoritative=True).handle_event(
        _ai_message_event()
    )

    assert published == [authoritative_frame]
