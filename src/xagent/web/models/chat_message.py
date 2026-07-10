from sqlalchemy import JSON, Column, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


class TaskChatMessage(Base):  # type: ignore
    """Persisted transcript message for a task chat session."""

    __tablename__ = "task_chat_messages"
    __table_args__ = (
        Index(
            "uq_task_chat_messages_task_role_turn_id",
            "task_id",
            "role",
            "turn_id",
            unique=True,
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(
        Integer, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role = Column(String(32), nullable=False)
    content = Column(Text, nullable=False)
    message_type = Column(String(64), nullable=False)
    interactions = Column(JSON, nullable=True)
    # Stable per-user-turn identity shared with user_message trace events. Used
    # by historical replay to reconcile trace rows with transcript rows without
    # collapsing distinct turns that happen to share text/attachments.
    turn_id = Column(String(64), nullable=True, index=True)
    # Durable handoff outcome for retry-safe user-message delivery. Legacy
    # rows remain NULL and are treated as already delivered.
    delivery_status = Column(String(32), nullable=True)
    # Per-message attachments (uploaded files). For user messages this is the
    # list of files the user attached when sending this turn. Stored as JSON so
    # the original file metadata (file_id, name, size, type) is available for
    # historical replay without having to re-derive it from the message body.
    attachments = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    task = relationship("Task", back_populates="chat_messages")
    user = relationship("User", back_populates="chat_messages")

    def __repr__(self) -> str:
        return (
            f"<TaskChatMessage(id={self.id}, task_id={self.task_id}, "
            f"role='{self.role}', message_type='{self.message_type}')>"
        )
