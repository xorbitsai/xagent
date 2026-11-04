"""Record the external cleanup a task deletion still owes (#2587).

A new table with no foreign keys, so unlike most task-adjacent revisions it
does not wait for ``tasks`` to exist: its rows outlive their task by design
and reference it by value only.
"""

import sqlalchemy as sa
from alembic import op

revision = "20260925_task_cleanup_obligations"
down_revision = "20260924_merge_admission_google"
branch_labels = None
depends_on = None

TABLE = "task_cleanup_obligations"


def upgrade() -> None:
    if sa.inspect(op.get_bind()).has_table(TABLE):
        return
    op.create_table(
        TABLE,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("task_id", sa.Integer, nullable=False),
        sa.Column("owner_id", sa.Integer, nullable=True),
        sa.Column("resource_kind", sa.String(32), nullable=False),
        sa.Column("resource_key", sa.String(255), nullable=False, server_default=""),
        sa.Column("locator", sa.JSON, nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text, nullable=True),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        # Never reuse an id on SQLite; see the model for why.
        sqlite_autoincrement=True,
    )
    op.create_index("ix_task_cleanup_obligations_task_id", TABLE, ["task_id"])
    op.create_index(
        "ix_task_cleanup_obligations_status_due",
        TABLE,
        ["status", "next_attempt_at"],
    )


def downgrade() -> None:
    if sa.inspect(op.get_bind()).has_table(TABLE):
        op.drop_table(TABLE)
