"""add durable background jobs

Revision ID: 20260521_add_background_jobs
Revises: 20260525_add_task_visibility
Create Date: 2026-05-21 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260521_add_background_jobs"
down_revision: Union[str, tuple[str, str], None] = "20260525_add_task_visibility"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _index_names(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if table_name not in inspector.get_table_names():
        return set()
    return {index["name"] for index in inspector.get_indexes(table_name)}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = inspector.get_table_names()

    if "background_jobs" not in existing_tables:
        constraints = [
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("idempotency_key"),
        ]
        if "users" in existing_tables:
            constraints.append(
                sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE")
            )

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
            *constraints,
        )

    existing_indexes = _index_names("background_jobs")
    if "ix_background_jobs_celery_task_id" not in existing_indexes:
        op.create_index(
            op.f("ix_background_jobs_celery_task_id"),
            "background_jobs",
            ["celery_task_id"],
            unique=False,
        )
    if "ix_background_jobs_id" not in existing_indexes:
        op.create_index(
            op.f("ix_background_jobs_id"),
            "background_jobs",
            ["id"],
            unique=False,
        )
    if "ix_background_jobs_job_type" not in existing_indexes:
        op.create_index(
            op.f("ix_background_jobs_job_type"),
            "background_jobs",
            ["job_type"],
            unique=False,
        )
    if "ix_background_jobs_queue" not in existing_indexes:
        op.create_index(
            op.f("ix_background_jobs_queue"),
            "background_jobs",
            ["queue"],
            unique=False,
        )
    if "ix_background_jobs_status" not in existing_indexes:
        op.create_index(
            op.f("ix_background_jobs_status"),
            "background_jobs",
            ["status"],
            unique=False,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "background_jobs" not in inspector.get_table_names():
        return

    existing_indexes = _index_names("background_jobs")
    if "ix_background_jobs_status" in existing_indexes:
        op.drop_index(op.f("ix_background_jobs_status"), table_name="background_jobs")
    if "ix_background_jobs_queue" in existing_indexes:
        op.drop_index(op.f("ix_background_jobs_queue"), table_name="background_jobs")
    if "ix_background_jobs_job_type" in existing_indexes:
        op.drop_index(op.f("ix_background_jobs_job_type"), table_name="background_jobs")
    if "ix_background_jobs_id" in existing_indexes:
        op.drop_index(op.f("ix_background_jobs_id"), table_name="background_jobs")
    if "ix_background_jobs_celery_task_id" in existing_indexes:
        op.drop_index(
            op.f("ix_background_jobs_celery_task_id"), table_name="background_jobs"
        )
    op.drop_table("background_jobs")
