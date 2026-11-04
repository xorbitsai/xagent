"""Add the task auto-recovery state and event tables.

Revision ID: 20261008_task_auto_recovery
Revises: 20261005_uploaded_file_cleanup_manifest

New tables only: ``tasks`` is untouched and nothing is backfilled, so a task
without a row (including every task already paused before this revision) does
not take part in automatic recovery.
"""

import sqlalchemy as sa
from alembic import op

revision = "20261008_task_auto_recovery"
down_revision = "20261005_uploaded_file_cleanup_manifest"
branch_labels = None
depends_on = None

STATE_TABLE = "task_auto_recovery"
EVENTS_TABLE = "task_recovery_events"


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    # ``tasks`` is metadata-owned in Alembic-only installations; the tables
    # are then created from the models together with it.
    if not inspector.has_table("tasks"):
        return
    if not inspector.has_table(STATE_TABLE):
        op.create_table(
            STATE_TABLE,
            sa.Column(
                "task_id",
                sa.Integer,
                sa.ForeignKey("tasks.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column("run_id", sa.String(64), nullable=False),
            sa.Column("reason", sa.String(32), nullable=False),
            sa.Column("state", sa.String(24), nullable=False),
            sa.Column("state_detail", sa.String(64), nullable=True),
            sa.Column("paused_state_version", sa.Integer, nullable=False),
            sa.Column("interrupted_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("episode_started_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column(
                "no_progress_resumes",
                sa.Integer,
                nullable=False,
                server_default="0",
            ),
            sa.Column("total_resumes", sa.Integer, nullable=False, server_default="0"),
            sa.Column("progress_marker", sa.String(128), nullable=True),
            sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_command_id", sa.String(64), nullable=True),
            sa.Column("last_error", sa.Text, nullable=True),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
        )
        op.create_index(
            "ix_task_auto_recovery_due",
            STATE_TABLE,
            ["state", "next_attempt_at"],
        )
    if not inspector.has_table(EVENTS_TABLE):
        op.create_table(
            EVENTS_TABLE,
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column(
                "task_id",
                sa.Integer,
                sa.ForeignKey("tasks.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("run_id", sa.String(64), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column("event", sa.String(32), nullable=False),
            sa.Column("reason", sa.String(32), nullable=True),
            sa.Column("attempt", sa.Integer, nullable=True),
            sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("detail", sa.JSON, nullable=True),
        )
        op.create_index("ix_task_recovery_events_task_id", EVENTS_TABLE, ["task_id"])


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table(EVENTS_TABLE):
        op.drop_table(EVENTS_TABLE)
    if inspector.has_table(STATE_TABLE):
        op.drop_table(STATE_TABLE)
