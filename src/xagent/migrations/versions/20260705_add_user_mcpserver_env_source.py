"""add env_source to user_mcpservers

Records which env layer a user picked for an MCP server: "own" | "shared" |
"platform". NULL keeps the legacy fallback (global < shared < user).

Revision ID: 20260705_add_user_mcpserver_env_source
Revises: 20260703_seed_google_maps_mcp_app
Create Date: 2026-07-05 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260705_add_user_mcpserver_env_source"
down_revision: Union[str, None] = "20260703_seed_google_maps_mcp_app"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _columns(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if table_name not in inspector.get_table_names():
        return set()
    return {col["name"] for col in inspector.get_columns(table_name)}


def upgrade() -> None:
    columns = _columns("user_mcpservers")
    if not columns or "env_source" in columns:
        return
    op.add_column(
        "user_mcpservers",
        sa.Column("env_source", sa.String(length=16), nullable=True),
    )


def downgrade() -> None:
    if "env_source" in _columns("user_mcpservers"):
        op.drop_column("user_mcpservers", "env_source")
