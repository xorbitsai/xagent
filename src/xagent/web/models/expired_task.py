"""What is left of a task the retention purge expired (#2565).

Conversation expiry deletes the task row, and with it everything a reader
could have used to tell "expired under the retention policy" apart from
"deleted by its owner" or "never existed". This row is what survives, so that
a caller who could see the live task can be told it expired instead of
getting a bare not-found.

It holds **no conversation content**. Each column is an input some surface's
access predicate reads -- owner, agent, workforce, source, visibility, the MCP
channel-plumbing marker -- so a surface can apply the same predicate to this
row that it applied to the live task, and disclose the expiry to exactly the
callers who could have seen the task and to no one else. Anything that is not
such an input does not belong here.

Only the scheduled retention purge writes it. A user-initiated deletion writes
nothing and keeps answering not-found: the owner asked for the task to be gone.

The primary key is the purged task's own id. PostgreSQL sequences never reuse
an id, and the purge refuses to run on any other dialect, so a tombstone
cannot collide with a later task. Readers still resolve a live ``tasks`` row
first (see ``services/expired_tasks.py``), so a tombstone can never shadow a
task that exists.
"""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class ExpiredTaskTombstone(Base):  # type: ignore
    __tablename__ = "expired_task_tombstones"

    task_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    #: Owner scope. Cascades so an account deletion leaves nothing behind.
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    #: v1 agent-key ownership. SET NULL mirrors what agent deletion does to
    #: the live task (``stage_delete_agent`` nulls ``Task.agent_id``).
    agent_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("agents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    #: v1 workforce-key ownership. Live, that ownership is proven through
    #: ``WorkforceRun.task_id``, which the purge SETs NULL -- so it is copied
    #: here before the delete, or a workforce key could never prove it.
    workforce_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("workforces.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    #: ``Task.source``: v1 serves only ``sdk``; Conversation Logs selects on
    #: the external sources.
    source: Mapped[str | None] = mapped_column(String(20), nullable=True)
    #: ``Task.is_visible``: Conversation Logs' hidden-external scope.
    is_visible: Mapped[bool] = mapped_column(Boolean, nullable=False)
    #: Whether the task carried the MCP runtime-authorization marker. Those
    #: tasks are channel plumbing, not conversations, and Conversation Logs
    #: excludes them; the boolean is kept instead of the ``agent_config``
    #: that carried it.
    is_channel_plumbing: Mapped[bool] = mapped_column(Boolean, nullable=False)
    task_created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    expired_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
