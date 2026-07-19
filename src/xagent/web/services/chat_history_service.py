"""Persistence helpers for task chat transcripts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ...core.agent.transcript import (
    build_assistant_transcript_content,
    normalize_transcript_messages,
)
from ..models.chat_message import TaskChatMessage
from .file_reference_output_service import reconcile_assistant_file_references

logger = logging.getLogger(__name__)

DELIVERY_PENDING = "pending"
DELIVERY_DISPATCHED = "dispatched"
DELIVERY_COMPLETED = "completed"
DELIVERY_FAILED = "failed"


@dataclass(frozen=True)
class UserMessageDeliveryClaim:
    """Result of inspecting or atomically claiming a client turn id."""

    message: TaskChatMessage
    claimed: bool
    payload_matches: bool

    @property
    def failed(self) -> bool:
        return str(self.message.delivery_status) == DELIVERY_FAILED

    @property
    def pending(self) -> bool:
        return str(self.message.delivery_status) == DELIVERY_PENDING


def _attachment_identity(
    attachments: Optional[List[Dict[str, Any]]],
) -> tuple[str, ...]:
    identities: list[str] = []
    for attachment in attachments or []:
        file_id = str(attachment.get("file_id") or "").strip()
        fallback = "\x1f".join(
            str(attachment.get(key) or "") for key in ("name", "size", "type")
        )
        identities.append(file_id or f"legacy:{fallback}")
    return tuple(sorted(identities))


def _delivery_payload_matches(
    message: TaskChatMessage,
    *,
    content: str,
    attachments: Optional[List[Dict[str, Any]]],
) -> bool:
    stored_attachments = (
        message.attachments if isinstance(message.attachments, list) else None
    )
    return str(message.content) == content.strip() and _attachment_identity(
        stored_attachments
    ) == _attachment_identity(attachments)


def inspect_user_message_delivery(
    db: Session,
    task_id: int,
    content: str,
    *,
    attachments: Optional[List[Dict[str, Any]]],
    turn_id: str,
) -> Optional[UserMessageDeliveryClaim]:
    """Return the durable outcome for ``turn_id`` without creating a row."""

    existing = (
        db.query(TaskChatMessage)
        .filter(
            TaskChatMessage.task_id == task_id,
            TaskChatMessage.role == "user",
            TaskChatMessage.turn_id == turn_id,
        )
        .first()
    )
    if existing is None:
        return None
    return UserMessageDeliveryClaim(
        message=existing,
        claimed=False,
        payload_matches=_delivery_payload_matches(
            existing,
            content=content,
            attachments=attachments,
        ),
    )


def claim_user_message_delivery(
    db: Session,
    task_id: int,
    user_id: int,
    content: str,
    *,
    attachments: Optional[List[Dict[str, Any]]] = None,
    turn_id: str,
) -> UserMessageDeliveryClaim:
    """Atomically claim a live-control turn before dispatching it.

    The unique database index is the cross-worker serializer. A concurrent
    loser rolls back its insert and returns the winner's durable row, so only
    the claimant may inject the message into an active runtime.
    """

    existing = inspect_user_message_delivery(
        db,
        task_id,
        content,
        attachments=attachments,
        turn_id=turn_id,
    )
    if existing is not None:
        return existing

    message = TaskChatMessage(
        task_id=task_id,
        user_id=user_id,
        role="user",
        content=content.strip(),
        message_type="user_message",
        interactions=None,
        turn_id=turn_id,
        delivery_status=DELIVERY_PENDING,
        attachments=attachments,
    )
    db.add(message)
    try:
        db.commit()
        db.refresh(message)
        return UserMessageDeliveryClaim(
            message=message,
            claimed=True,
            payload_matches=True,
        )
    except IntegrityError:
        db.rollback()
        raced = inspect_user_message_delivery(
            db,
            task_id,
            content,
            attachments=attachments,
            turn_id=turn_id,
        )
        if raced is None:
            raise
        return raced


def mark_user_message_delivery(
    db: Session,
    *,
    task_id: int,
    turn_id: str,
    status: str,
) -> None:
    """Persist a delivery transition for a claimed user turn."""

    if status not in {
        DELIVERY_PENDING,
        DELIVERY_DISPATCHED,
        DELIVERY_COMPLETED,
        DELIVERY_FAILED,
    }:
        raise ValueError(f"Unknown delivery status: {status}")
    db.query(TaskChatMessage).filter(
        TaskChatMessage.task_id == task_id,
        TaskChatMessage.role == "user",
        TaskChatMessage.turn_id == turn_id,
    ).update({TaskChatMessage.delivery_status: status}, synchronize_session=False)
    db.commit()


def mark_user_message_delivery_sync(
    task_id: int,
    turn_id: str,
    status: str,
) -> None:
    """Update one delivery from synchronous or ``asyncio.to_thread`` callers."""

    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        mark_user_message_delivery(
            db,
            task_id=task_id,
            turn_id=turn_id,
            status=status,
        )


def persist_user_message(
    db: Session,
    task_id: int,
    user_id: int,
    content: str,
    *,
    attachments: Optional[List[Dict[str, Any]]] = None,
    turn_id: Optional[str] = None,
) -> Optional[TaskChatMessage]:
    return _persist_message(
        db=db,
        task_id=task_id,
        user_id=user_id,
        role="user",
        content=content,
        message_type="user_message",
        attachments=attachments,
        turn_id=turn_id,
    )


def persist_user_message_no_commit(
    db: Session,
    task_id: int,
    user_id: int,
    content: str,
    *,
    attachments: Optional[List[Dict[str, Any]]] = None,
    turn_id: Optional[str] = None,
    delivery_status: Optional[str] = None,
) -> Optional[TaskChatMessage]:
    """``persist_user_message`` variant that stages the row but does NOT commit.

    Used by ``TaskTurnOrchestrator.begin_turn`` so the atomic claim
    UPDATE and the message insert land in the same commit — if the
    insert fails, the status flip is rolled back too. Caller is
    responsible for calling ``db.commit()`` (or ``db.rollback()`` on
    failure).

    Returns ``None`` when content is whitespace-only AND no attachments
    are provided. A row with empty content but non-empty attachments is
    still persisted (the user uploaded files but didn't type anything).
    """
    normalized_content = content.strip()
    if not normalized_content and not attachments:
        return None
    message = TaskChatMessage(
        task_id=task_id,
        user_id=user_id,
        role="user",
        content=normalized_content,
        message_type="user_message",
        interactions=None,
        turn_id=turn_id,
        delivery_status=delivery_status,
        # Pass through ``attachments`` directly so an explicit empty list
        # round-trips as ``[]`` rather than being coerced to ``NULL`` —
        # callers may want to distinguish "no attachments specified" from
        # "attachments key was set, just empty".
        attachments=attachments,
    )
    db.add(message)
    return message


def persist_assistant_message(
    db: Session,
    task_id: int,
    user_id: int,
    content: str,
    *,
    message_type: str = "assistant_message",
    interactions: Optional[List[Dict[str, Any]]] = None,
    turn_id: Optional[str] = None,
    content_is_reconciled: bool = False,
) -> Optional[TaskChatMessage]:
    reconciled_content = (
        content
        if content_is_reconciled
        else reconcile_assistant_file_references(
            db,
            task_id=task_id,
            user_id=user_id,
            content=content,
        )
    )
    transcript_content = build_assistant_transcript_content(
        reconciled_content, interactions
    )
    return _persist_message(
        db=db,
        task_id=task_id,
        user_id=user_id,
        role="assistant",
        content=transcript_content,
        message_type=message_type,
        interactions=interactions,
        turn_id=turn_id,
    )


def persist_assistant_message_no_commit(
    db: Session,
    task_id: int,
    user_id: int,
    content: str,
    *,
    message_type: str = "assistant_message",
    interactions: Optional[List[Dict[str, Any]]] = None,
    turn_id: Optional[str] = None,
    content_is_reconciled: bool = False,
) -> Optional[TaskChatMessage]:
    """Stage an assistant transcript row for an atomic caller-owned commit."""

    reconciled_content = (
        content
        if content_is_reconciled
        else reconcile_assistant_file_references(
            db,
            task_id=task_id,
            user_id=user_id,
            content=content,
        )
    )
    transcript_content = build_assistant_transcript_content(
        reconciled_content, interactions
    )
    normalized_content = transcript_content.strip()
    if not normalized_content:
        return None
    message = TaskChatMessage(
        task_id=task_id,
        user_id=user_id,
        role="assistant",
        content=normalized_content,
        message_type=message_type,
        interactions=interactions,
        turn_id=turn_id,
        attachments=None,
    )
    db.add(message)
    return message


def load_task_transcript(
    db: Session,
    task_id: int,
    *,
    before_message_id: Optional[int] = None,
) -> List[Dict[str, str]]:
    if before_message_id is not None:
        # Check if the reference message actually exists
        exists = (
            db.query(TaskChatMessage.id)
            .filter(
                TaskChatMessage.id == before_message_id,
                TaskChatMessage.task_id == task_id,
            )
            .first()
        )
        if not exists:
            logger.warning(
                "Message id: {before_message_id} does not exit, returning empty list."
            )
            return []

    query = db.query(TaskChatMessage).filter(TaskChatMessage.task_id == task_id)
    if before_message_id is not None:
        query = query.filter(TaskChatMessage.id < before_message_id)

    messages = [
        {"role": str(message.role), "content": str(message.content)}
        for message in query.order_by(TaskChatMessage.id.asc()).all()
    ]
    return normalize_transcript_messages(messages)


def get_latest_waiting_question(
    db: Session, task_id: int
) -> tuple[Optional[str], Optional[list[dict[str, Any]]]]:
    """Return the latest persisted ask-user question for a waiting task."""

    latest_question = (
        db.query(TaskChatMessage)
        .filter(
            TaskChatMessage.task_id == task_id,
            TaskChatMessage.role == "assistant",
            TaskChatMessage.message_type == "question",
        )
        .order_by(TaskChatMessage.id.desc())
        .first()
    )
    if not latest_question:
        return None, None

    interactions = latest_question.interactions
    return (
        str(latest_question.content),
        interactions if isinstance(interactions, list) else None,
    )


def _persist_message(
    db: Session,
    task_id: int,
    user_id: int,
    role: str,
    content: str,
    message_type: str,
    interactions: Optional[List[Dict[str, Any]]] = None,
    attachments: Optional[List[Dict[str, Any]]] = None,
    turn_id: Optional[str] = None,
    delivery_status: Optional[str] = None,
) -> Optional[TaskChatMessage]:
    normalized_content = content.strip()
    if not normalized_content and not attachments:
        return None

    message = TaskChatMessage(
        task_id=task_id,
        user_id=user_id,
        role=role,
        content=normalized_content,
        message_type=message_type,
        interactions=interactions,
        turn_id=turn_id,
        delivery_status=delivery_status,
        # Pass through ``attachments`` directly so an explicit empty list
        # round-trips as ``[]`` rather than being coerced to ``NULL``.
        attachments=attachments,
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message
