"""add trace checkpoint blobs

Revision ID: 20260525_add_trace_checkpoint_blobs
Revises: 20260524_add_trace_message_blobs
Create Date: 2026-05-25 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260525_add_trace_checkpoint_blobs"
down_revision: Union[str, None] = "20260524_add_trace_message_blobs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _inspector() -> sa.Inspector:
    from alembic import context

    return sa.inspect(context.get_bind())


def _table_exists(table_name: str) -> bool:
    return table_name in _inspector().get_table_names()


def _index_exists(table_name: str, index_name: str) -> bool:
    if not _table_exists(table_name):
        return False
    return index_name in {idx["name"] for idx in _inspector().get_indexes(table_name)}


def _drop_index_if_exists(index_name: str, table_name: str) -> None:
    if _index_exists(table_name, index_name):
        op.drop_index(op.f(index_name), table_name=table_name)


def _foreign_key_if_table_exists(
    table_name: str,
    local_cols: list[str],
    remote_cols: list[str],
) -> list[sa.ForeignKeyConstraint]:
    if not _table_exists(table_name):
        return []
    return [sa.ForeignKeyConstraint(local_cols, remote_cols)]


def upgrade() -> None:
    if _table_exists("trace_checkpoint_blobs"):
        return

    op.create_table(
        "trace_checkpoint_blobs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("task_id", sa.Integer(), nullable=False),
        sa.Column("execution_id", sa.String(length=255), nullable=False),
        sa.Column("blob_kind", sa.String(length=255), nullable=False),
        sa.Column("blob_hash", sa.String(length=80), nullable=False),
        sa.Column("blob_data", sa.JSON(), nullable=False),
        sa.Column("blob_bytes", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
        *_foreign_key_if_table_exists("tasks", ["task_id"], ["tasks.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "task_id",
            "blob_kind",
            "blob_hash",
            name="uq_trace_checkpoint_blobs_task_kind_hash",
        ),
    )
    op.create_index(
        op.f("ix_trace_checkpoint_blobs_id"),
        "trace_checkpoint_blobs",
        ["id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_trace_checkpoint_blobs_task_id"),
        "trace_checkpoint_blobs",
        ["task_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_trace_checkpoint_blobs_execution_id"),
        "trace_checkpoint_blobs",
        ["execution_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_trace_checkpoint_blobs_blob_kind"),
        "trace_checkpoint_blobs",
        ["blob_kind"],
        unique=False,
    )


def downgrade() -> None:
    if not _table_exists("trace_checkpoint_blobs"):
        return

    _drop_index_if_exists(
        "ix_trace_checkpoint_blobs_blob_kind", "trace_checkpoint_blobs"
    )
    _drop_index_if_exists(
        "ix_trace_checkpoint_blobs_execution_id", "trace_checkpoint_blobs"
    )
    _drop_index_if_exists("ix_trace_checkpoint_blobs_task_id", "trace_checkpoint_blobs")
    _drop_index_if_exists("ix_trace_checkpoint_blobs_id", "trace_checkpoint_blobs")
    op.drop_table("trace_checkpoint_blobs")
