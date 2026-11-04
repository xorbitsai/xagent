"""add reusable MCP OAuth dynamic client registration identity

Revision ID: 20260713_mcp_oauth_dcr
Revises: 20260711_task_commands
Create Date: 2026-07-13

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260713_mcp_oauth_dcr"
down_revision: Union[str, None] = "20260711_task_commands"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "mcp_oauth_clients"
COLUMN = "registration_lookup_hash"
INDEX = "ux_mcp_oauth_clients_registration_lookup_hash"


def upgrade() -> None:
    if op.get_context().as_sql:
        op.add_column(TABLE, sa.Column(COLUMN, sa.String(64), nullable=True))
        op.create_index(INDEX, TABLE, [COLUMN], unique=True)
        return
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns(TABLE)}
    if COLUMN not in columns:
        op.add_column(TABLE, sa.Column(COLUMN, sa.String(64), nullable=True))
    indexes = {index["name"] for index in inspector.get_indexes(TABLE)}
    if INDEX not in indexes:
        op.create_index(INDEX, TABLE, [COLUMN], unique=True)


def downgrade() -> None:
    if op.get_context().as_sql:
        op.drop_index(INDEX, table_name=TABLE)
        op.drop_column(TABLE, COLUMN)
        return
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names():
        return
    indexes = {index["name"] for index in inspector.get_indexes(TABLE)}
    if INDEX in indexes:
        op.drop_index(INDEX, table_name=TABLE)
    columns = {column["name"] for column in inspector.get_columns(TABLE)}
    if COLUMN in columns:
        op.drop_column(TABLE, COLUMN)
