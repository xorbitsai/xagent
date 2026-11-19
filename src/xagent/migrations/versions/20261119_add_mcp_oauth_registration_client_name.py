"""add the client_name sent in MCP OAuth dynamic client registration

Revision ID: 20261119_mcp_oauth_client_name
Revises: 20261117_chat_message_metadata
Create Date: 2026-11-19

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261119_mcp_oauth_client_name"
down_revision: Union[str, None] = "20261117_chat_message_metadata"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "mcp_oauth_clients"
COLUMN = "registration_client_name"


def upgrade() -> None:
    if op.get_context().as_sql:
        op.add_column(TABLE, sa.Column(COLUMN, sa.String(255), nullable=True))
        return
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns(TABLE)}
    if COLUMN not in columns:
        op.add_column(TABLE, sa.Column(COLUMN, sa.String(255), nullable=True))


def downgrade() -> None:
    if op.get_context().as_sql:
        op.drop_column(TABLE, COLUMN)
        return
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns(TABLE)}
    if COLUMN in columns:
        op.drop_column(TABLE, COLUMN)
