"""Durable control-plane commands for task execution."""

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import deferred
from sqlalchemy.sql import func

from .database import Base

# A command is in exactly one of two status sets, and together they hold every
# status a command is written with.
#
# Claimable: statuses a dispatcher can still claim. The three claimable partial
# indexes below, the dispatcher's presence precheck and the shared-mode claim
# condition take their status set from here. The legacy claim arms name their
# statuses; a test keeps them within this set, so a precheck cannot miss a
# command a claim would take.
#
# Terminal: statuses of a settled command. A retry may move a failed command
# back to pending, which puts it back in the unfinished-task partial index
# below. That index takes its status set from here, as the unfinished-command
# checks in the transport and admission code (``status NOT IN`` the terminal
# set) do; a test keeps their terminal tuples equal to it.
CLAIMABLE_COMMAND_STATUSES = ("pending", "processing")
TERMINAL_COMMAND_STATUSES = ("completed", "failed")

# Rows a dispatcher can still claim. Settled commands leave the partial index,
# so it stays small however large the command history grows.
_CLAIMABLE_STATUS_CLAUSE = text(
    "status IN ({})".format(", ".join(f"'{s}'" for s in CLAIMABLE_COMMAND_STATUSES))
)

# Rows that are not settled. The partial index keeps only these, so it stays
# small however large the command history grows.
_UNFINISHED_STATUS_CLAUSE = text(
    "status NOT IN ({})".format(", ".join(f"'{s}'" for s in TERMINAL_COMMAND_STATUSES))
)


class TaskExecutionCommand(Base):  # type: ignore
    """One idempotent task command, claimed by at most one worker at a time."""

    __tablename__ = "task_execution_commands"
    __table_args__ = (
        UniqueConstraint("task_id", "command_id", name="uq_task_command_identity"),
        Index("ix_task_commands_status_created", "status", "created_at"),
        Index("ix_task_commands_task_order", "task_id", "id"),
        # Serves the dispatcher's reply-first candidate lookup, which filters
        # claimable commands by kind and takes the oldest id.
        Index(
            "ix_task_commands_claimable_kind",
            "kind",
            "id",
            sqlite_where=_CLAIMABLE_STATUS_CLAUSE,
            postgresql_where=_CLAIMABLE_STATUS_CLAUSE,
        ),
        # Serves the dispatcher's oldest-first fallback lookup, which takes
        # the oldest claimable id of any kind.
        Index(
            "ix_task_commands_claimable_id",
            "id",
            sqlite_where=_CLAIMABLE_STATUS_CLAUSE,
            postgresql_where=_CLAIMABLE_STATUS_CLAUSE,
        ),
        # Ordered for the oldest claimable ids of one tenant key and dispatch class.
        Index(
            "ix_task_commands_claimable_fair",
            "fairness_key",
            "dispatch_class",
            "id",
            sqlite_where=_CLAIMABLE_STATUS_CLAUSE,
            postgresql_where=_CLAIMABLE_STATUS_CLAUSE,
        ),
        # Serves three reads: the oldest unfinished id of one task (the dispatcher's
        # lookup of what a running task has queued), the check, in every
        # claimable query, that a task has no unfinished command before the
        # candidate, and the admission code's check that no earlier command of the
        # same task is still waiting (``_behind_own_task``).
        Index(
            "ix_task_commands_unfinished_task",
            "task_id",
            "id",
            sqlite_where=_UNFINISHED_STATUS_CLAUSE,
            postgresql_where=_UNFINISHED_STATUS_CLAUSE,
        ),
        # Ordered for the claimable command of one tenant key and dispatch class
        # with the highest priority and, within it, the lowest id.
        Index(
            "ix_task_commands_claimable_fair_priority",
            "fairness_key",
            "dispatch_class",
            text("priority DESC"),
            "id",
            sqlite_where=_CLAIMABLE_STATUS_CLAUSE,
            postgresql_where=_CLAIMABLE_STATUS_CLAUSE,
        ),
    )

    id = Column(Integer, primary_key=True)
    task_id = Column(
        Integer,
        ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=False,
    )
    actor_user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    # Immutable acceptance-time pseudonym. ``actor_user_id`` remains the live
    # relational join and may be NULLed when the account is deleted.
    actor_subject = Column(String(64), nullable=True)
    # Immutable task-owner correlation captured at command acceptance. Legacy
    # rows stay NULL because a current numeric owner cannot prove historical
    # ownership after account deletion and SQLite id reuse.
    task_owner_user_id = Column(Integer, nullable=True)
    task_owner_subject = Column(String(64), nullable=True)
    command_id = Column(String(64), nullable=False)
    kind = Column(String(32), nullable=False)
    payload = Column(JSON, nullable=False)

    # Server-generated route, bound before the command becomes visible.
    reply_host_id = Column(String(64), nullable=True)
    reply_origin = Column(String(64), nullable=True)

    # The run/worker observed when the command was accepted. Commands aimed at
    # a live run stay with its lease owner; once that lease expires another
    # worker may recover them from the durable inbox.
    target_run_id = Column(String(64), nullable=True)
    # Legacy rows created before this snapshot existed remain NULL: treating
    # an unknown historical version as a real version 0 would create a false
    # run-correlation tuple during recovery.
    target_state_version = Column(Integer, nullable=True)
    target_runner_id = Column(String(255), nullable=True)

    status = Column(
        String(32), nullable=False, default="pending", server_default="pending"
    )
    claimed_by = Column(String(255), nullable=True)
    claim_expires_at = Column(DateTime(timezone=True), nullable=True)
    # Earliest retry time; independent of task ownership and never renewed.
    retry_available_at = Column(DateTime(timezone=True), nullable=True)
    # What the deployment classifier decided for this command at acceptance:
    # "interactive" or "batch". Existing rows and control commands read
    # "batch", meaning not classified. Set by stage_task_command in the
    # acceptance transaction; never updated after it commits.
    dispatch_class = Column(
        String(16), nullable=False, server_default="batch", default="batch"
    )
    # The tenant this command is accounted to. Set in the acceptance
    # transaction and never updated after it commits. Rows that predate the
    # column read the empty string and count as an ordinary tenant.
    fairness_key = Column(String(64), nullable=False, server_default="", default="")
    # The priority this command was accepted with, -10 to 10, higher first. Set
    # by stage_task_command in the acceptance transaction, never updated after
    # it commits; old rows and control commands read 0. It orders commands of one
    # tenant key and class, only while fair-share dispatch is on. Deferred, so a
    # query for commands does not select it.
    priority = deferred(
        Column(SmallInteger, nullable=False, server_default="0", default=0)
    )
    attempt_count = Column(Integer, nullable=False, default=0, server_default="0")
    failure_count = Column(Integer, nullable=False, default=0, server_default="0")
    defer_count = Column(Integer, nullable=False, default=0, server_default="0")
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)
