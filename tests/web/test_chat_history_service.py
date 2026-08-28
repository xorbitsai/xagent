from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from xagent.core.agent.transcript import build_assistant_transcript_content
from xagent.core.context_ref import CONTEXT_REFS_KEY
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services import chat_history_service
from xagent.web.services.assistant_history_safety import safe_str
from xagent.web.services.chat_history_service import (
    _MAX_HISTORICAL_IMAGE_CONTEXT_REFS,
    DELIVERY_COMPLETED,
    DELIVERY_DISPATCHED,
    DELIVERY_FAILED,
    claim_user_message_delivery,
    get_latest_waiting_question,
    inspect_user_message_delivery,
    load_task_transcript,
    mark_user_message_delivery,
    persist_assistant_message,
    persist_assistant_message_no_commit,
    persist_user_message,
    persist_user_message_no_commit,
)


def _create_db_session():
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return SessionLocal()


def _create_task(db_session):
    user = User(username="tester", password_hash="hashed_password", is_admin=False)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    task = Task(
        user_id=int(user.id),
        title="Chat task",
        description="Task chat",
        status=TaskStatus.PENDING,
    )
    db_session.add(task)
    db_session.commit()
    db_session.refresh(task)
    return task


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, ""), ("", ""), (False, "False"), (0, "0"), ("answer", "answer")],
)
def test_safe_str_only_special_cases_none(value: object | None, expected: str) -> None:
    assert safe_str(value) == expected


def test_load_task_transcript_returns_prior_turns_only():
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)

        first_user = persist_user_message(
            db_session,
            int(task.id),
            int(task.user_id),
            "Summarize the repo",
        )
        assert first_user is not None

        assistant = persist_assistant_message(
            db_session,
            int(task.id),
            int(task.user_id),
            "The main risks are architecture drift and persistence gaps.",
            message_type="assistant_response",
        )
        assert assistant is not None

        second_user = persist_user_message(
            db_session,
            int(task.id),
            int(task.user_id),
            "Expand the persistence gap",
        )
        assert second_user is not None

        transcript = load_task_transcript(
            db_session,
            int(task.id),
            before_message_id=int(second_user.id),
        )

        assert transcript == [
            {"role": "user", "content": "Summarize the repo"},
            {
                "role": "assistant",
                "content": "The main risks are architecture drift and persistence gaps.",
            },
        ]
    finally:
        db_session.close()


def test_persist_assistant_message_without_provenance_fails_closed_on_load() -> None:
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        raw_content = "unproven provider token=secret"

        assistant = persist_assistant_message(
            db_session,
            int(task.id),
            int(task.user_id),
            raw_content,
        )

        assert assistant is not None
        assert assistant.message_type == "assistant_message"
        assert load_task_transcript(db_session, int(task.id)) == [
            {"role": "assistant", "content": "Task execution failed."}
        ]
    finally:
        db_session.close()


def test_load_task_transcript_coerces_nullable_fields_to_empty_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = SimpleNamespace(
        role="assistant",
        content=None,
        message_type=None,
        attachments=None,
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.order_by.return_value.all.return_value = [
        message
    ]

    def project_content(*, content: str, message_type: str) -> str:
        assert content == ""
        assert message_type == ""
        return "safe fallback"

    monkeypatch.setattr(
        chat_history_service,
        "client_safe_assistant_history_content",
        project_content,
    )

    assert load_task_transcript(db, 1) == [
        {"role": "assistant", "content": "safe fallback"}
    ]


@pytest.mark.parametrize(
    ("message_type", "content", "expected_content"),
    [
        ("task_failure", "provider token=secret", "Task execution failed."),
        (
            "chat_response",
            "A legitimate pre-cutover assistant response.",
            "Task execution failed.",
        ),
        (
            "assistant_message",
            "Legacy managed failure token=secret",
            "Task execution failed.",
        ),
        (
            "final_answer",
            "Legacy websocket failure token=secret",
            "Task execution failed.",
        ),
        ("unknown_assistant_type", "unknown token=secret", "Task execution failed."),
        ("assistant", "A known-safe legacy answer.", "A known-safe legacy answer."),
        ("assistant_response", "A known-safe answer.", "A known-safe answer."),
    ],
)
def test_load_task_transcript_projects_assistant_history_for_subsequent_turns(
    message_type: str,
    content: str,
    expected_content: str,
) -> None:
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        assistant = persist_assistant_message(
            db_session,
            int(task.id),
            int(task.user_id),
            content,
            message_type=message_type,
        )
        assert assistant is not None

        transcript = load_task_transcript(db_session, int(task.id))

        assert transcript == [{"role": "assistant", "content": expected_content}]
        assert "secret" not in repr(transcript)
    finally:
        db_session.close()


def test_load_task_transcript_preserves_uploaded_images_as_context_refs():
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        image_message = persist_user_message(
            db_session,
            int(task.id),
            int(task.user_id),
            "What is shown?",
            attachments=[
                {
                    "file_id": "image-id",
                    "name": "diagram.png",
                    "size": 321,
                    "type": "image/png",
                },
                {
                    "file_id": "pdf-id",
                    "name": "notes.pdf",
                    "size": 654,
                    "type": "application/pdf",
                },
            ],
        )
        assert image_message is not None

        transcript = load_task_transcript(db_session, int(task.id))

        assert transcript[0]["content"] == "What is shown?"
        references = transcript[0][CONTEXT_REFS_KEY]
        assert len(references) == 1
        assert references[0]["file_ref"]["file_id"] == "image-id"
        assert references[0]["metadata"] == {"source": "user_upload"}
    finally:
        db_session.close()


def test_load_task_transcript_bounds_historical_images_to_recent_window():
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        total_images = _MAX_HISTORICAL_IMAGE_CONTEXT_REFS + 2
        for index in range(total_images):
            persist_user_message(
                db_session,
                int(task.id),
                int(task.user_id),
                f"Image {index}",
                attachments=[
                    {
                        "file_id": f"image-{index}",
                        "name": f"image-{index}.png",
                        "type": "image/png",
                    }
                ],
            )

        transcript = load_task_transcript(db_session, int(task.id))

        assert len(transcript) == total_images
        assert CONTEXT_REFS_KEY not in transcript[0]
        assert CONTEXT_REFS_KEY not in transcript[1]
        retained_ids = [
            message[CONTEXT_REFS_KEY][0]["file_ref"]["file_id"]
            for message in transcript
            if CONTEXT_REFS_KEY in message
        ]
        assert retained_ids == [f"image-{index}" for index in range(2, total_images)]
    finally:
        db_session.close()


def test_persist_assistant_message_formats_interactions_into_transcript():
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)

        persist_assistant_message(
            db_session,
            int(task.id),
            int(task.user_id),
            "I need one more detail before I continue.",
            message_type="chat_response",
            interactions=[
                {
                    "type": "text_input",
                    "label": "Repository path",
                    "placeholder": "Enter the repository path",
                }
            ],
        )

        stored_message = (
            db_session.query(TaskChatMessage)
            .filter(TaskChatMessage.task_id == int(task.id))
            .first()
        )

        assert stored_message is not None
        assert stored_message.role == "assistant"
        assert "Please answer the following questions:" in stored_message.content
        assert "Repository path: Enter the repository path" in stored_message.content
    finally:
        db_session.close()


def test_persist_assistant_message_repairs_invented_file_id():
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        db_session.add(
            UploadedFile(
                file_id="real-video-id",
                user_id=int(task.user_id),
                task_id=int(task.id),
                filename="generated_video.mp4",
                storage_path="/tmp/real-video-id/generated_video.mp4",
                mime_type="video/mp4",
                file_size=123,
            )
        )
        db_session.commit()

        message = persist_assistant_message(
            db_session,
            int(task.id),
            int(task.user_id),
            "[generated_video.mp4](file:invented-video-id)",
            message_type="final_answer",
        )

        assert message is not None
        assert message.content == "[generated_video.mp4](file:real-video-id)"
    finally:
        db_session.close()


def test_persist_assistant_message_skips_duplicate_reconciliation():
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)

        with patch(
            "xagent.web.services.chat_history_service."
            "reconcile_assistant_file_references",
            side_effect=AssertionError("content was already reconciled"),
        ):
            message = persist_assistant_message(
                db_session,
                int(task.id),
                int(task.user_id),
                "Already canonical",
                message_type="final_answer",
                content_is_reconciled=True,
            )

        assert message is not None
        assert message.content == "Already canonical"
    finally:
        db_session.close()


def test_get_latest_waiting_question_returns_latest_question_only():
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)

        persist_assistant_message(
            db_session,
            int(task.id),
            int(task.user_id),
            "First question",
            message_type="question",
            interactions=[{"type": "text_input", "label": "First"}],
        )
        persist_assistant_message(
            db_session,
            int(task.id),
            int(task.user_id),
            "Regular answer",
            message_type="assistant_message",
        )
        persist_assistant_message(
            db_session,
            int(task.id),
            int(task.user_id),
            "Second question",
            message_type="question",
            interactions=[{"type": "text_input", "label": "Second"}],
        )

        question, interactions = get_latest_waiting_question(db_session, int(task.id))

        assert question is not None
        assert question.startswith("Second question")
        assert interactions == [{"type": "text_input", "label": "Second"}]
    finally:
        db_session.close()


def test_build_assistant_transcript_content_skips_empty_unknown_interactions_header():
    content = build_assistant_transcript_content("Test", [{"type": "unknown_type"}])

    assert content == "Test"


def test_persist_user_message_stores_attachments_for_chip_replay():
    """Uploaded-file metadata must round-trip through ``attachments`` so the
    historical-replay path can render the same chips the user saw live."""
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        attachments = [
            {
                "file_id": "fid-1",
                "name": "Q1 Report.xlsx",
                "size": 12345,
                "type": (
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                ),
            }
        ]
        persist_user_message(
            db_session,
            int(task.id),
            int(task.user_id),
            "Read this for me.",
            attachments=attachments,
            turn_id="turn-attachments",
        )
        row = db_session.query(TaskChatMessage).first()
        assert row is not None
        assert row.turn_id == "turn-attachments"
        assert row.attachments == attachments
    finally:
        db_session.close()


def test_delivery_claim_rejects_same_turn_with_different_attachments() -> None:
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        first = claim_user_message_delivery(
            db_session,
            int(task.id),
            int(task.user_id),
            "Analyze this",
            attachments=[{"file_id": "file-a", "name": "a.pdf"}],
            turn_id="client-turn-files",
        )
        retried = claim_user_message_delivery(
            db_session,
            int(task.id),
            int(task.user_id),
            "Analyze this",
            attachments=[{"file_id": "file-b", "name": "b.pdf"}],
            turn_id="client-turn-files",
        )

        assert first.claimed is True
        assert retried.claimed is False
        assert retried.payload_matches is False
        assert db_session.query(TaskChatMessage).count() == 1
    finally:
        db_session.close()


def test_database_rejects_duplicate_user_turn_claims() -> None:
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        claim_user_message_delivery(
            db_session,
            int(task.id),
            int(task.user_id),
            "First claimant",
            turn_id="atomic-turn",
        )
        db_session.add(
            TaskChatMessage(
                task_id=int(task.id),
                user_id=int(task.user_id),
                role="user",
                content="Racing claimant",
                message_type="user_message",
                turn_id="atomic-turn",
            )
        )
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()
        assert db_session.query(TaskChatMessage).count() == 1
    finally:
        db_session.close()


def test_delivery_claim_recovers_from_unique_constraint_race(monkeypatch) -> None:
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        winner = claim_user_message_delivery(
            db_session,
            int(task.id),
            int(task.user_id),
            "Concurrent guidance",
            turn_id="raced-turn",
        )
        inspection_count = 0

        def hide_winner_on_initial_inspection(*args, **kwargs):
            nonlocal inspection_count
            inspection_count += 1
            if inspection_count == 1:
                return None
            return inspect_user_message_delivery(*args, **kwargs)

        monkeypatch.setattr(
            "xagent.web.services.chat_history_service.inspect_user_message_delivery",
            hide_winner_on_initial_inspection,
        )

        loser = claim_user_message_delivery(
            db_session,
            int(task.id),
            int(task.user_id),
            "Concurrent guidance",
            turn_id="raced-turn",
        )

        assert loser.claimed is False
        assert loser.message.id == winner.message.id
        assert inspection_count == 2
        assert db_session.query(TaskChatMessage).count() == 1
    finally:
        db_session.close()


def test_delivery_claim_surfaces_failed_handoff() -> None:
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        first = claim_user_message_delivery(
            db_session,
            int(task.id),
            int(task.user_id),
            "Apply guidance",
            turn_id="failed-turn",
        )
        first.message.delivery_status = DELIVERY_FAILED
        db_session.commit()

        retried = claim_user_message_delivery(
            db_session,
            int(task.id),
            int(task.user_id),
            "Apply guidance",
            turn_id="failed-turn",
        )
        assert retried.claimed is False
        assert retried.failed is True
    finally:
        db_session.close()


def test_delivery_transition_does_not_regress_a_completed_turn() -> None:
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        claim_user_message_delivery(
            db_session,
            int(task.id),
            int(task.user_id),
            "Apply guidance",
            turn_id="completed-turn",
        )

        assert (
            mark_user_message_delivery(
                db_session,
                task_id=int(task.id),
                turn_id="completed-turn",
                status=DELIVERY_DISPATCHED,
            ).outcome
            == "updated"
        )
        assert (
            mark_user_message_delivery(
                db_session,
                task_id=int(task.id),
                turn_id="completed-turn",
                status=DELIVERY_COMPLETED,
            ).outcome
            == "updated"
        )
        transition = mark_user_message_delivery(
            db_session,
            task_id=int(task.id),
            turn_id="completed-turn",
            status=DELIVERY_DISPATCHED,
        )

        assert transition.outcome == "conflict"
        assert transition.status == DELIVERY_COMPLETED
    finally:
        db_session.close()


def test_persist_assistant_message_no_commit_keeps_turn_id() -> None:
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        row = persist_assistant_message_no_commit(
            db_session,
            int(task.id),
            int(task.user_id),
            "Guidance applied",
            message_type="final_answer",
            turn_id="client-turn-1",
        )
        assert row is not None
        assert db_session.query(TaskChatMessage).count() == 0

        db_session.commit()
        stored = db_session.query(TaskChatMessage).one()
        assert stored.turn_id == "client-turn-1"
    finally:
        db_session.close()


def test_persist_user_message_no_commit_allows_empty_content_with_attachments():
    """User uploaded files without typing — the row should still be staged
    so the chips survive reload."""
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        attachments = [{"file_id": "fid-only", "name": "x.pdf"}]
        msg = persist_user_message_no_commit(
            db_session,
            int(task.id),
            int(task.user_id),
            "",
            attachments=attachments,
        )
        assert msg is not None
        db_session.commit()
        row = db_session.query(TaskChatMessage).first()
        assert row is not None
        assert row.content == ""
        assert row.attachments == attachments

        # Sanity guard: still drops empty rows with no attachments.
        assert (
            persist_user_message_no_commit(
                db_session,
                int(task.id),
                int(task.user_id),
                "   ",
                attachments=None,
            )
            is None
        )
    finally:
        db_session.close()


def test_persist_user_message_preserves_empty_attachments_list_as_empty_list():
    """An explicit empty ``attachments=[]`` (e.g. a SDK caller that always
    sends the key) must round-trip as ``[]`` rather than being coerced to
    ``NULL`` — callers may want to distinguish "no attachments specified"
    from "attachments key was set, just empty"."""
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        # Non-empty content + empty attachments → row persists with [].
        persist_user_message(
            db_session,
            int(task.id),
            int(task.user_id),
            "Just a text message.",
            attachments=[],
        )
        row = db_session.query(TaskChatMessage).first()
        assert row is not None
        assert row.attachments == []  # not None
    finally:
        db_session.close()
