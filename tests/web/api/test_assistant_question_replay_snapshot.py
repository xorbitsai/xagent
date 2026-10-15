"""Historical replay must emit one event per assistant question (#2292).

A waiting question is written both as a trace event and as a transcript row
whose text has the rendered interaction list appended, so the two can never
compare equal. Replay used to ship both, and the widget session page -- which
never clears messages on reconnect -- printed the clarification form again on
every reconnect.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from xagent.core.agent.transcript import build_assistant_transcript_content
from xagent.web.api.conversation_logs import _serialize_transcript_with_events
from xagent.web.api.websocket import send_historical_data_as_stream
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task import TraceEvent as DatabaseTraceEvent
from xagent.web.models.user import User
from xagent.web.services.chat_history_service import (
    get_latest_waiting_question,
    load_task_transcript,
    persist_assistant_message,
)
from xagent.web.services.file_reference_output_service import (
    reconcile_assistant_file_references,
)
from xagent.web.services.task_execution import _persist_agent_outbound_event
from xagent.web.services.task_interaction_read import get_pending_interaction_question

INTERACTIONS = [{"type": "confirm", "label": "Proceed?", "default": True}]
QUESTION = "Round 1: please confirm"


def _make_task(username: str):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    user = User(username=username, password_hash="hashed_password", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    task = Task(
        user_id=int(user.id),
        title="Chat task",
        description="Task chat",
        status=TaskStatus.PENDING,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return SessionLocal, db, task


def _question_trace_row(task_id: int, event_id: str, when: datetime):
    return DatabaseTraceEvent(
        task_id=task_id,
        event_id=event_id,
        event_type="agent_message",
        timestamp=when,
        data={
            "event_id": event_id,
            "message": QUESTION,
            "expect_response": True,
            "metadata": {"interactions": INTERACTIONS},
        },
    )


def _question_row(task_id: int, user_id: int, when: datetime, **kwargs):
    return TaskChatMessage(
        task_id=task_id,
        user_id=user_id,
        role="assistant",
        content=build_assistant_transcript_content(QUESTION, INTERACTIONS),
        message_type="question",
        interactions=INTERACTIONS,
        created_at=when,
        **kwargs,
    )


async def _replay(
    monkeypatch, SessionLocal, task_id: int, user_id: int, *, cache=None
) -> list[dict]:
    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    sent: list[dict] = []

    async def send_personal_message(event: dict, websocket: object) -> None:
        sent.append(event)

    monkeypatch.setattr("xagent.web.models.database.get_db", get_test_db)
    monkeypatch.setattr("xagent.web.api.websocket.get_db", get_test_db)
    monkeypatch.setattr(
        "xagent.web.api.websocket.get_session_local", lambda: SessionLocal
    )
    monkeypatch.setattr("xagent.web.models.database._SessionLocal", SessionLocal)
    cache = {} if cache is None else cache
    monkeypatch.setattr(
        "xagent.web.api.websocket.cache_get", lambda *args: cache.get("value")
    )
    monkeypatch.setattr(
        "xagent.web.api.websocket.cache_set",
        lambda key, value, **kwargs: cache.update(value=value),
    )
    monkeypatch.setattr(
        "xagent.web.api.websocket.manager.send_personal_message",
        send_personal_message,
    )

    await send_historical_data_as_stream(
        websocket=object(),
        task_id=task_id,
        user=SimpleNamespace(id=user_id, is_admin=False),
    )
    return sent


def _question_events(sent: list[dict]) -> list[dict]:
    return [
        event
        for event in sent
        if event.get("type") == "trace_event"
        and event.get("event_type") == "agent_message"
        and QUESTION in str(event.get("data", {}).get("message", ""))
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("message_type", ["question", "question_superseded"])
async def test_one_question_replays_once_and_keeps_the_trace_event_identity(
    monkeypatch,
    message_type,
) -> None:
    SessionLocal, db, task = _make_task("replay-one")
    task_id, user_id = int(task.id), int(task.user_id)
    db.close()

    def get_test_db() -> Iterator[Session]:
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    # Go through the real producer so the trace row and the transcript row are
    # written exactly the way production writes them. ``task_execution`` binds
    # ``get_db`` at import time, so the module attribute is the one to patch.
    monkeypatch.setattr("xagent.web.services.task_execution.get_db", get_test_db)
    monkeypatch.setattr("xagent.web.models.database.get_db", get_test_db)
    _persist_agent_outbound_event(
        task_id,
        {
            "event_type": "agent_message",
            "timestamp": datetime(2026, 9, 10, tzinfo=timezone.utc).timestamp(),
            "data": {
                "event_id": "ea62bd91",
                "message": QUESTION,
                "expect_response": True,
                "metadata": {"interactions": INTERACTIONS},
            },
        },
    )

    probe = SessionLocal()
    try:
        row = probe.query(TaskChatMessage).one()
        # Precondition for the whole defect: the two texts differ, so no
        # content comparison could ever have paired them.
        assert row.content != QUESTION
        assert row.source_event_id == "ea62bd91"
        row.message_type = message_type
        probe.commit()
    finally:
        probe.close()

    sent = await _replay(monkeypatch, SessionLocal, task_id, user_id)
    questions = _question_events(sent)

    assert len(questions) == 1
    event = questions[0]
    assert event["event_id"] == "ea62bd91"
    # The surviving copy is the sanitized one: a historical question must not
    # flip the client back into waiting_for_user.
    assert event["data"]["expect_response"] is False
    assert event["data"]["source"] == "chat_history"
    assert event["data"]["metadata"]["interactions"] == INTERACTIONS
    assert event["data"]["message"] == QUESTION
    assert event["data"]["content"] == QUESTION

    # Only the public display changes. Pairing, model context and the internal
    # waiting-question reader keep the complete canonical transcript.
    with SessionLocal() as probe:
        full_text = build_assistant_transcript_content(QUESTION, INTERACTIONS)
        assert get_latest_waiting_question(probe, task_id, allow_superseded=True) == (
            full_text,
            INTERACTIONS,
        )
        assert load_task_transcript(probe, task_id) == [
            {"role": "assistant", "content": full_text}
        ]
        task = probe.get(Task, task_id)
        assert get_pending_interaction_question(probe, task) == (QUESTION, INTERACTIONS)
        rows = probe.query(TaskChatMessage).filter_by(task_id=task_id).all()
        log_messages = _serialize_transcript_with_events(probe, task, rows, [])
        assert log_messages[0]["content"] == QUESTION
        assert log_messages[0]["interactions"] == INTERACTIONS
        assert rows[0].content == full_text

    reconnected = _question_events(
        await _replay(monkeypatch, SessionLocal, task_id, user_id)
    )
    assert len(reconnected) == 1
    assert reconnected[0]["event_id"] == event["event_id"]
    assert reconnected[0]["data"]["message"] == QUESTION


@pytest.mark.asyncio
@pytest.mark.parametrize("message_type", ["question", "question_superseded"])
@pytest.mark.parametrize(
    "interaction",
    [
        {"type": "confirm", "label": "Review [sample](file:missing)"},
        {
            "type": "text_input",
            "label": "Owner",
            "placeholder": "Review [sample](file:missing)",
        },
        {
            "type": "select_one",
            "label": "Sample",
            "options": [{"value": "sample", "label": "[sample](file:missing)"}],
        },
    ],
)
async def test_question_appendix_is_projected_before_file_link_reconciliation(
    monkeypatch, message_type, interaction
) -> None:
    SessionLocal, db, task = _make_task("replay-question-link")
    task_id, user_id = int(task.id), int(task.user_id)
    interactions = [interaction]
    task.status = TaskStatus.WAITING_FOR_USER
    row = persist_assistant_message(
        db,
        task_id,
        user_id,
        QUESTION,
        message_type=message_type,
        interactions=interactions,
    )
    canonical_content = row.content
    assert "file:missing" in canonical_content
    assert (
        reconcile_assistant_file_references(
            db, task_id=task_id, user_id=user_id, content=canonical_content, records=[]
        )
        != canonical_content
    )
    db.close()

    sent = await _replay(monkeypatch, SessionLocal, task_id, user_id)
    questions = _question_events(sent)
    assert len(questions) == 1
    assert questions[0]["data"]["message"] == QUESTION
    assert questions[0]["data"]["metadata"]["interactions"] == interactions
    waiting = [event for event in sent if event["type"] == "task_waiting_for_user"]
    assert len(waiting) == 1
    assert waiting[0]["question"] == QUESTION
    assert waiting[0]["interactions"] == interactions
    with SessionLocal() as probe:
        task = probe.get(Task, task_id)
        rows = probe.query(TaskChatMessage).filter_by(task_id=task_id).all()
        messages = _serialize_transcript_with_events(probe, task, rows, [])
        assert messages[0]["content"] == QUESTION
        assert messages[0]["interactions"] == interactions
        assert rows[0].content == canonical_content


@pytest.mark.asyncio
async def test_non_question_appendix_like_text_is_not_projected(monkeypatch) -> None:
    SessionLocal, db, task = _make_task("replay-answer-appendix")
    task_id, user_id = int(task.id), int(task.user_id)
    row = persist_assistant_message(
        db,
        task_id,
        user_id,
        QUESTION,
        message_type="assistant",
        interactions=INTERACTIONS,
    )
    canonical_content = row.content
    assert canonical_content != QUESTION
    db.close()

    sent = await _replay(monkeypatch, SessionLocal, task_id, user_id)
    answers = _question_events(sent)
    assert len(answers) == 1
    assert answers[0]["data"]["message"] == canonical_content
    assert answers[0]["data"]["metadata"]["interactions"] == INTERACTIONS
    with SessionLocal() as probe:
        task = probe.get(Task, task_id)
        rows = probe.query(TaskChatMessage).filter_by(task_id=task_id).all()
        messages = _serialize_transcript_with_events(probe, task, rows, [])
        assert messages[0]["content"] == canonical_content
        assert messages[0]["interactions"] == INTERACTIONS
        assert rows[0].content == canonical_content


@pytest.mark.asyncio
async def test_a_legacy_row_without_source_event_id_still_replays_once(
    monkeypatch,
) -> None:
    SessionLocal, db, task = _make_task("replay-legacy")
    task_id, user_id = int(task.id), int(task.user_id)
    when = datetime(2026, 9, 10, tzinfo=timezone.utc)
    try:
        db.add(_question_trace_row(task_id, "ea62bd91", when))
        db.add(_question_row(task_id, user_id, when))
        db.commit()
    finally:
        db.close()

    sent = await _replay(monkeypatch, SessionLocal, task_id, user_id)
    questions = _question_events(sent)

    assert len(questions) == 1
    assert questions[0]["event_id"] == "ea62bd91"
    assert questions[0]["data"]["expect_response"] is False
    assert questions[0]["data"]["message"] == QUESTION


@pytest.mark.asyncio
async def test_replay_rebuilds_pre_projection_cache_and_keeps_waiting_text_consistent(
    monkeypatch,
) -> None:
    SessionLocal, db, task = _make_task("replay-display-cache")
    task_id, user_id = int(task.id), int(task.user_id)
    when = datetime(2026, 9, 10, tzinfo=timezone.utc)
    try:
        task.status = TaskStatus.WAITING_FOR_USER
        db.add(_question_trace_row(task_id, "cached-question", when))
        db.add(_question_row(task_id, user_id, when, source_event_id="cached-question"))
        db.commit()
    finally:
        db.close()
    cache = {}
    await _replay(monkeypatch, SessionLocal, task_id, user_id, cache=cache)
    assert cache["value"]["question_display_version"] == 1
    # A pre-deploy cache has all the same row watermarks, but the wrong display.
    cache["value"].pop("question_display_version")
    cache["value"]["events"] = []
    for _ in range(2):  # rebuild, then reuse the rebuilt cache on reconnect
        sent = await _replay(monkeypatch, SessionLocal, task_id, user_id, cache=cache)
        questions = _question_events(sent)
        assert len(questions) == 1
        assert questions[0]["data"]["message"] == QUESTION
        waiting = [e for e in sent if e["type"] == "task_waiting_for_user"]
        assert len(waiting) == 1
        assert waiting[0]["question"] == QUESTION
        assert waiting[0]["interactions"] == INTERACTIONS


@pytest.mark.asyncio
async def test_two_rounds_asking_the_same_question_both_survive(
    monkeypatch,
) -> None:
    # The guard against over-collapsing: identical text across rounds is
    # routine, and #2250 is the record of what collapsing it costs.
    SessionLocal, db, task = _make_task("replay-two-rounds")
    task_id, user_id = int(task.id), int(task.user_id)
    first = datetime(2026, 9, 10, 1, tzinfo=timezone.utc)
    second = datetime(2026, 9, 10, 2, tzinfo=timezone.utc)
    try:
        db.add(_question_trace_row(task_id, "ea62bd91", first))
        db.add(_question_row(task_id, user_id, first, source_event_id="ea62bd91"))
        db.add(_question_trace_row(task_id, "79652fd2", second))
        db.add(_question_row(task_id, user_id, second, source_event_id="79652fd2"))
        db.commit()
    finally:
        db.close()

    sent = await _replay(monkeypatch, SessionLocal, task_id, user_id)
    questions = _question_events(sent)

    assert len(questions) == 2
    assert [event["event_id"] for event in questions] == ["ea62bd91", "79652fd2"]


@pytest.mark.asyncio
async def test_an_assistant_answer_is_not_touched_by_question_pairing(
    monkeypatch,
) -> None:
    # Answers are deduped the other way round (row dropped, trace kept) and
    # their trace events carry streaming identity. Pairing must not reach them.
    SessionLocal, db, task = _make_task("replay-answer")
    task_id, user_id = int(task.id), int(task.user_id)
    when = datetime(2026, 9, 10, tzinfo=timezone.utc)
    try:
        db.add(
            DatabaseTraceEvent(
                task_id=task_id,
                event_id="final-1",
                event_type="ai_message",
                timestamp=when,
                data={"event_id": "final-1", "message": "Final answer"},
            )
        )
        db.add(
            TaskChatMessage(
                task_id=task_id,
                user_id=user_id,
                role="assistant",
                content="Final answer",
                message_type="assistant",
                created_at=when,
            )
        )
        db.commit()
    finally:
        db.close()

    sent = await _replay(monkeypatch, SessionLocal, task_id, user_id)
    answers = [
        event
        for event in sent
        if event.get("type") == "trace_event"
        and event.get("data", {}).get("message") == "Final answer"
    ]

    assert len(answers) == 1
    assert answers[0]["event_id"] == "final-1"
    assert answers[0]["event_type"] == "ai_message"
