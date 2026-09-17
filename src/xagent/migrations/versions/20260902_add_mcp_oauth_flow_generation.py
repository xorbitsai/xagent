"""bind MCP OAuth flows to an association lifecycle generation

Revision ID: 20260902_oauth_flow_generation
Revises: 20260826_seed_employment_hero_mcp_app
Create Date: 2026-09-02

Existing authorization flows cannot be attributed safely to a specific
``UserMCPServer`` lifecycle, so their new snapshot remains NULL and callbacks
fail closed. Newly-created flows always persist the exact generation captured
before provider I/O.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260902_oauth_flow_generation"
down_revision: Union[str, None] = "20260826_seed_employment_hero_mcp_app"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE_NAME = "mcp_oauth_flow_states"
COLUMN_NAME = "association_lifecycle_generation"


def _table_exists() -> bool:
    return TABLE_NAME in sa.inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    if not _table_exists():
        return
    columns = {
        column["name"] for column in sa.inspect(op.get_bind()).get_columns(TABLE_NAME)
    }
    if COLUMN_NAME not in columns:
        op.add_column(
            TABLE_NAME,
            sa.Column(COLUMN_NAME, sa.Uuid(as_uuid=True), nullable=True),
        )


def downgrade() -> None:
    if not _table_exists():
        return
    columns = {
        column["name"] for column in sa.inspect(op.get_bind()).get_columns(TABLE_NAME)
    }
    if COLUMN_NAME in columns:
        with op.batch_alter_table(TABLE_NAME) as batch_op:
            batch_op.drop_column(COLUMN_NAME)
