"""Add single-turn encrypted inputs and immutable command reply routes.

Revision ID: 20260912_shared_task_execution
Revises: 20260911_update_onedrive_description
"""

import sqlalchemy as sa
from alembic import op

revision = "20260912_shared_task_execution"
down_revision = "20260911_update_onedrive_description"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    # Core task tables are metadata-owned and can be absent when Alembic
    # runs on an empty database. Match the existing task migrations.
    if inspector.has_table("tasks") and not inspector.has_table("task_runtime_secrets"):
        op.create_table(
            "task_runtime_secrets",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column(
                "task_id",
                sa.Integer(),
                sa.ForeignKey("tasks.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("turn_id", sa.String(64), nullable=False),
            sa.Column("run_id", sa.String(64), nullable=True),
            sa.Column("owner_subject", sa.String(64), nullable=False),
            sa.Column("ciphertext", sa.Text(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.UniqueConstraint(
                "task_id", "turn_id", name="uq_task_runtime_secret_turn"
            ),
        )
        op.create_index(
            "ix_task_runtime_secrets_task_id", "task_runtime_secrets", ["task_id"]
        )
    if inspector.has_table("task_execution_commands"):
        columns = {
            column["name"]
            for column in inspector.get_columns("task_execution_commands")
        }
        for name in ("reply_host_id", "reply_origin"):
            if name not in columns:
                op.add_column(
                    "task_execution_commands",
                    sa.Column(name, sa.String(64), nullable=True),
                )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("task_execution_commands"):
        columns = {
            column["name"]
            for column in inspector.get_columns("task_execution_commands")
        }
        for name in ("reply_origin", "reply_host_id"):
            if name in columns:
                op.drop_column("task_execution_commands", name)
    if inspector.has_table("task_runtime_secrets"):
        op.drop_index(
            "ix_task_runtime_secrets_task_id", table_name="task_runtime_secrets"
        )
        op.drop_table("task_runtime_secrets")
