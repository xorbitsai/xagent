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
from xagent.web.api.websocket import (
    _QUESTION_DISPLAY_VERSION,
    send_historical_data_as_stream,
)
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

    monkeypatch.setattr("xagent.web.models.database._SessionLocal", SessionLocal)
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
    monkeypatch.setattr("xagent.web.models.database._SessionLocal", SessionLocal)
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
    assert cache["value"]["question_display_version"] == _QUESTION_DISPLAY_VERSION
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


# --- An answer stored as both a trace row and a transcript row -------------
#
# The trace row keeps the model's raw text; the transcript row is stored with
# its file references reconciled. Replay must recognise them as one answer and
# display the reconciled text.

_FAKE_FILE_ID = "123e4567-e89b-12d3-a456-426614174000"


def _output_file(db: Session, user_id: int, task_id: int, name: str, rel: str):
    import uuid

    from xagent.web.models.uploaded_file import UploadedFile

    file_id = str(uuid.uuid4())
    db.add(
        UploadedFile(
            file_id=file_id,
            user_id=user_id,
            task_id=task_id,
            filename=name,
            storage_path=f"/store/{file_id}/{name}",
            workspace_relative_path=rel,
            workspace_category="output",
            storage_status="committed",
        )
    )
    db.commit()
    return file_id


def _answer_frames(sent: list[dict]) -> list[dict]:
    return [
        event
        for event in sent
        if event.get("type") == "trace_event"
        and event.get("event_type") in {"ai_message", "agent_message"}
    ]


def _frame_text(frame: dict) -> str:
    data = frame["data"]
    return data.get("content") or data.get("message")


_ANSWER_CELLS = {
    "plain": ("ai_message", "Your report is ready.", False),
    "valid": ("ai_message", "Your report is ready: [report.pdf](file:{fid})", False),
    "fake": (
        "ai_message",
        f"Your report is ready: [x.pdf](file:{_FAKE_FILE_ID})",
        False,
    ),
    "repairable": (
        "ai_message",
        f"Your report is ready: [report.pdf](file:{_FAKE_FILE_ID})",
        False,
    ),
    "path": (
        "ai_message",
        "Your report is ready: [report.pdf](output/report.pdf)",
        True,
    ),
    "result_card_fake": (
        "agent_message",
        f"Your report is ready: [x.pdf](file:{_FAKE_FILE_ID})",
        False,
    ),
    "streamed_repairable": (
        "ai_message",
        f"Your report is ready: [report.pdf](file:{_FAKE_FILE_ID})",
        False,
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("cell", list(_ANSWER_CELLS))
async def test_answer_stored_as_trace_and_transcript_row_replays_once(
    monkeypatch, cell
) -> None:
    from datetime import timedelta

    from xagent.web.api.websocket import _normalize_file_outputs
    from xagent.web.services.chat_history_service import (
        persist_assistant_message_no_commit,
    )
    from xagent.web.services.task_execution import _rewrite_file_links_to_file_id

    event_type, template, with_file_outputs = _ANSWER_CELLS[cell]
    SessionLocal, db, task = _make_task(f"answer-{cell}")
    task_id, user_id = int(task.id), int(task.user_id)
    when = datetime(2026, 11, 25, tzinfo=timezone.utc)
    file_id = _output_file(db, user_id, task_id, "report.pdf", "output/report.pdf")
    raw = template.replace("{fid}", file_id)
    is_card = event_type == "agent_message"

    data: dict = {"content": raw, "status": "completed"}
    if event_type == "agent_message":
        card = {"title": raw, "items": ["a"], "source": {"name": "Cosmos"}}
        data = {
            "event_id": "answer-1",
            "message": raw,
            "message_type": "result_card",
            "expect_response": False,
            "display": "chat",
            "visible": True,
            "metadata": {"display": "result_card", "result_card": card},
        }
    if cell == "streamed_repairable":
        data["stream_message_id"] = "final_answer_abc"
    db.add(
        DatabaseTraceEvent(
            task_id=task_id,
            event_id="answer-1",
            event_type=event_type,
            timestamp=when,
            data=data,
        )
    )
    path_map: dict = {}
    if with_file_outputs:
        outputs = [
            {
                "file_id": file_id,
                "filename": "report.pdf",
                "relative_path": "output/report.pdf",
            }
        ]
        db.add(
            DatabaseTraceEvent(
                task_id=task_id,
                event_id="completion-1",
                event_type="task_completion",
                timestamp=when + timedelta(milliseconds=1),
                data={
                    "result": {"content": raw, "file_outputs": outputs},
                    "success": True,
                    "file_outputs": outputs,
                },
            )
        )
        _, path_map = _normalize_file_outputs(db, task_id, user_id, outputs)
    # The real finishing path rewrites path links, then lets the store
    # reconcile the text because the caller does not declare it reconciled.
    persist_assistant_message_no_commit(
        db,
        task_id=task_id,
        user_id=user_id,
        content=_rewrite_file_links_to_file_id(raw, path_map),
        message_type="result_card" if is_card else "assistant_response",
        message_metadata=data["metadata"] if is_card else None,
        source_event_id="answer-1" if is_card else None,
    )
    db.commit()
    stored = (
        db.query(TaskChatMessage).filter(TaskChatMessage.task_id == task_id).one()
    ).content
    db.close()

    frames = _answer_frames(await _replay(monkeypatch, SessionLocal, task_id, user_id))

    assert len(frames) == 1
    assert _frame_text(frames[0]) == stored
    assert not [f for f in frames if f["data"].get("source") == "chat_history"]


@pytest.mark.asyncio
async def test_transcript_only_answer_still_replays(monkeypatch) -> None:
    from xagent.web.services.chat_history_service import (
        persist_assistant_message_no_commit,
    )

    SessionLocal, db, task = _make_task("answer-transcript-only")
    task_id, user_id = int(task.id), int(task.user_id)
    persist_assistant_message_no_commit(
        db,
        task_id=task_id,
        user_id=user_id,
        content="Only in the transcript.",
        message_type="assistant_response",
    )
    db.commit()
    db.close()

    frames = _answer_frames(await _replay(monkeypatch, SessionLocal, task_id, user_id))

    assert [_frame_text(frame) for frame in frames] == ["Only in the transcript."]
    assert frames[0]["data"].get("source") == "chat_history"
