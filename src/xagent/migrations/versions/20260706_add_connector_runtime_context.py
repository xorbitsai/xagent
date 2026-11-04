"""add connector runtime context persistence

Revision ID: 20260706_add_connector_runtime_context
Revises: 20260707_merge_alembic_heads
Create Date: 2026-07-06 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260706_add_connector_runtime_context"
down_revision: Union[str, None] = "20260707_merge_alembic_heads"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _tables() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def _columns(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if table_name not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns(table_name)}


def _indexes(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if table_name not in inspector.get_table_names():
        return set()
    return {index["name"] for index in inspector.get_indexes(table_name)}


def _add_column_if_missing(table_name: str, column: sa.Column) -> None:
    if table_name in _tables() and column.name not in _columns(table_name):
        op.add_column(table_name, column)


def _drop_column_if_present(table_name: str, column_name: str) -> None:
    if table_name in _tables() and column_name in _columns(table_name):
        op.drop_column(table_name, column_name)


def upgrade() -> None:
    _add_column_if_missing(
        "tasks",
        sa.Column("connector_runtime_selected_refs", sa.JSON(), nullable=True),
    )

    for table_name in ("mcp_servers", "custom_apis"):
        _add_column_if_missing(
            table_name,
            sa.Column("runtime_input_schema", sa.JSON(), nullable=True),
        )
        _add_column_if_missing(
            table_name,
            sa.Column("runtime_bindings", sa.JSON(), nullable=True),
        )
        _add_column_if_missing(
            table_name,
            sa.Column(
                "allow_delegated_authorization",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            ),
        )

    if "tasks" in _tables() and "task_connector_runtime_contexts" not in _tables():
        op.create_table(
            "task_connector_runtime_contexts",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column(
                "task_id",
                sa.Integer(),
                sa.ForeignKey("tasks.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("connector_type", sa.String(length=32), nullable=False),
            sa.Column("connector_id", sa.Integer(), nullable=False),
            sa.Column("context", sa.JSON(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.UniqueConstraint(
                "task_id",
                "connector_type",
                "connector_id",
                name="uq_task_connector_runtime_contexts_ref",
            ),
        )

    if (
        "task_connector_runtime_contexts" in _tables()
        and "ix_task_connector_runtime_contexts_task_id"
        not in _indexes("task_connector_runtime_contexts")
    ):
        op.create_index(
            "ix_task_connector_runtime_contexts_task_id",
            "task_connector_runtime_contexts",
            ["task_id"],
        )


def downgrade() -> None:
    if "task_connector_runtime_contexts" in _tables():
        if "ix_task_connector_runtime_contexts_task_id" in _indexes(
            "task_connector_runtime_contexts"
        ):
            op.drop_index(
                "ix_task_connector_runtime_contexts_task_id",
                table_name="task_connector_runtime_contexts",
            )
        op.drop_table("task_connector_runtime_contexts")

    for table_name in ("mcp_servers", "custom_apis"):
        _drop_column_if_present(table_name, "allow_delegated_authorization")
        _drop_column_if_present(table_name, "runtime_bindings")
        _drop_column_if_present(table_name, "runtime_input_schema")

    _drop_column_if_present("tasks", "connector_runtime_selected_refs")
