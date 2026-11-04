"""add MCP concurrency config

Revision ID: 20260624_add_mcp_concurrency_config
Revises: 20260624_backfill_sdk_task_visibility
Create Date: 2026-06-24 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260624_add_mcp_concurrency_config"
down_revision: Union[str, tuple[str, str], None] = (
    "20260624_backfill_sdk_task_visibility"
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "mcp_servers" not in inspector.get_table_names():
        return

    existing_columns = {
        column["name"] for column in inspector.get_columns("mcp_servers")
    }
    bool_false = sa.text("false") if bind.dialect.name == "postgresql" else sa.text("0")

    if "concurrency_safe" not in existing_columns:
        op.add_column(
            "mcp_servers",
            sa.Column(
                "concurrency_safe",
                sa.Boolean(),
                nullable=False,
                server_default=bool_false,
            ),
        )
    if "concurrent_tools" not in existing_columns:
        op.add_column(
            "mcp_servers", sa.Column("concurrent_tools", sa.JSON(), nullable=True)
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "mcp_servers" not in inspector.get_table_names():
        return

    existing_columns = {
        column["name"] for column in inspector.get_columns("mcp_servers")
    }
    if "concurrent_tools" in existing_columns:
        op.drop_column("mcp_servers", "concurrent_tools")
    if "concurrency_safe" in existing_columns:
        op.drop_column("mcp_servers", "concurrency_safe")
