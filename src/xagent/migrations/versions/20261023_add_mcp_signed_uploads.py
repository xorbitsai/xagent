"""add signed upload settings to MCP servers

``mcp_servers.signed_uploads`` holds a connector's signed-upload settings as
a JSON list.  NULL and an empty list both mean the connector has none.  The
column is only added; existing rows stay NULL.

Revision ID: 20261023_add_mcp_signed_uploads
Revises: 20261022_task_cmd_fair_index
Create Date: 2026-10-23 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261023_add_mcp_signed_uploads"
down_revision: Union[str, None] = "20261022_task_cmd_fair_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _tables() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def _columns(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if table_name not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns(table_name)}


def _add_column_if_missing(table_name: str, column: sa.Column) -> None:
    if table_name in _tables() and column.name not in _columns(table_name):
        op.add_column(table_name, column)


def _drop_column_if_present(table_name: str, column_name: str) -> None:
    if table_name in _tables() and column_name in _columns(table_name):
        op.drop_column(table_name, column_name)


def upgrade() -> None:
    _add_column_if_missing(
        "mcp_servers",
        sa.Column("signed_uploads", sa.JSON(none_as_null=True), nullable=True),
    )


def downgrade() -> None:
    _drop_column_if_present("mcp_servers", "signed_uploads")
