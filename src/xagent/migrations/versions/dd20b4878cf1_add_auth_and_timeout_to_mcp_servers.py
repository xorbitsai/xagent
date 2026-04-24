"""add auth and timeout to mcp_servers

Revision ID: dd20b4878cf1
Revises: f1427c3a7261
Create Date: 2026-04-24 14:29:40.991078

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "dd20b4878cf1"
down_revision: Union[str, None] = "f1427c3a7261"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("mcp_servers", sa.Column("timeout", sa.Integer(), nullable=True))
    op.add_column("mcp_servers", sa.Column("auth", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("mcp_servers", "auth")
    op.drop_column("mcp_servers", "timeout")
