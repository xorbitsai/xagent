"""add durable background jobs

Revision ID: 20260521_add_background_jobs
Revises: 20260522_add_task_chat_message_turn_id
Create Date: 2026-05-21 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260521_add_background_jobs"
down_revision: Union[str, tuple[str, str], None] = (
    "20260522_add_task_chat_message_turn_id"
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "background_jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("job_type", sa.String(length=100), nullable=False),
        sa.Column("queue", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("progress", sa.JSON(), nullable=True),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("idempotency_key", sa.String(length=255), nullable=True),
        sa.Column("celery_task_id", sa.String(length=255), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("idempotency_key"),
    )
    op.create_index(
        op.f("ix_background_jobs_celery_task_id"),
        "background_jobs",
        ["celery_task_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_background_jobs_id"),
        "background_jobs",
        ["id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_background_jobs_job_type"),
        "background_jobs",
        ["job_type"],
        unique=False,
    )
    op.create_index(
        op.f("ix_background_jobs_queue"),
        "background_jobs",
        ["queue"],
        unique=False,
    )
    op.create_index(
        op.f("ix_background_jobs_status"),
        "background_jobs",
        ["status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_background_jobs_status"), table_name="background_jobs")
    op.drop_index(op.f("ix_background_jobs_queue"), table_name="background_jobs")
    op.drop_index(op.f("ix_background_jobs_job_type"), table_name="background_jobs")
    op.drop_index(op.f("ix_background_jobs_id"), table_name="background_jobs")
    op.drop_index(
        op.f("ix_background_jobs_celery_task_id"), table_name="background_jobs"
    )
    op.drop_table("background_jobs")
