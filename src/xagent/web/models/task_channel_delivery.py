"""Recoverable platform delivery, independent of the task execution lease."""

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Index, Integer, String

from .database import Base


class TaskChannelDelivery(Base):  # type: ignore
    __tablename__ = "task_channel_deliveries"
    __table_args__ = (
        Index(
            "ix_task_channel_delivery_pending", "channel_id", "status", "available_at"
        ),
    )

    command_id = Column(
        Integer,
        ForeignKey("task_execution_commands.id", ondelete="CASCADE"),
        primary_key=True,
    )
    channel_id = Column(
        Integer, ForeignKey("user_channels.id", ondelete="CASCADE"), nullable=False
    )
    destination = Column(JSON, nullable=False)
    status = Column(
        String(16), nullable=False, default="pending", server_default="pending"
    )
    failure_count = Column(Integer, nullable=False, default=0, server_default="0")
    claim_token = Column(String(64), nullable=True)
    available_at = Column(DateTime(timezone=True), nullable=True)
    delivered_at = Column(DateTime(timezone=True), nullable=True)
