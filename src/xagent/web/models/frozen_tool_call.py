"""The approved-payload record: what a gated MCP call will run if approved.

One row is written when the write gate pauses a call, and consumed exactly
once when the approver answers. Between those two moments the row -- not the
model -- is the authority on what executes. That is the whole mechanism: an
approval that let the model re-derive its arguments would publish something
nobody agreed to.

Why this is not a row in ``task_interaction_requests``. That table is the
generic record of a blocking interaction and would otherwise be the right
home, but staging a row there requires a resolved checkpoint anchor, and the
checkpoint for *this* pause does not exist yet when the gate runs: it is
written by the pattern's pause handler after the tool returns. The two
records also answer different questions -- that one tracks the conversation
with the user, this one holds the bytes to execute -- and are keyed
differently as a result.
"""

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.sql import func

from .database import Base

# The row's lifecycle. ``pending`` is the only state a resume may consume;
# the two terminal states differ only in why the payload will never run, and
# both are kept rather than deleted so an approval that arrives late reads as
# already-settled instead of as an unknown id.
FROZEN_CALL_STATUSES = ("pending", "executed", "voided")


class FrozenToolCall(Base):  # type: ignore
    """One MCP call frozen at gate time, awaiting a decision.

    ``interaction_id`` is the primary key rather than a surrogate: it is
    minted by the host when the gate pauses, carried into the pattern's
    checkpoint, and handed back verbatim to ``resume_user_interaction``. Any
    other key would need a lookup table between the two halves for nothing.

    ``arguments`` is stored as the gate saw it -- already schema-normalized
    and carrying the runtime-bound values the model may not supply -- so a
    resume can execute it without re-deriving anything.
    """

    __tablename__ = "frozen_tool_calls"
    __table_args__ = (
        Index("ix_frozen_tool_calls_task_status", "task_id", "status"),
        Index("ix_frozen_tool_calls_expires_at", "expires_at"),
        CheckConstraint(
            "status IN ('pending','executed','voided')",
            name="ck_frozen_tool_calls_status",
        ),
    )

    interaction_id = Column(String(255), primary_key=True)
    task_id = Column(
        Integer,
        ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=False,
    )
    # The runtime tool name as the model called it, and the server it came
    # from. Recorded for the resume-side identity check: MCP runtime names
    # embed a renameable display name, so a tool renamed while an approval
    # was pending must not have someone else's approval spent on it.
    tool_name = Column(String(255), nullable=False)
    server_name = Column(String(255), nullable=False)
    # What the server's own annotations claimed at gate time. Not consulted
    # on execution -- it is here so an audit can tell whether a call was
    # gated because it declared itself destructive or because it declared
    # nothing at all.
    write_hint = Column(String(32), nullable=False)
    arguments = Column(JSON(none_as_null=True), nullable=False)
    status = Column(String(32), nullable=False, default="pending")
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # A pending row past this instant must not execute. Enforced by the
    # consumer rather than by a sweep: a row nobody answers is harmless
    # where it sits, and deleting it on a timer would race the approval
    # that is arriving at that moment.
    expires_at = Column(DateTime(timezone=True), nullable=False)
    settled_at = Column(DateTime(timezone=True), nullable=True)
