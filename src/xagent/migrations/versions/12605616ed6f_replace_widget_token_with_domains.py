"""replace_widget_token_with_domains

Revision ID: 12605616ed6f
Revises: 20260403_add_user_tool_configs
Create Date: 2026-04-07 14:18:28.704254

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "12605616ed6f"
down_revision: Union[str, None] = "20260403_add_user_tool_configs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = sa.inspect(bind)

    # Check if agents table exists
    tables = inspector.get_table_names()
    if "agents" not in tables:
        return

    existing_columns = [col["name"] for col in inspector.get_columns("agents")]

    if "widget_enabled" not in existing_columns:
        op.add_column(
            "agents",
            sa.Column(
                "widget_enabled", sa.Boolean(), server_default="1", nullable=False
            ),
        )
    if "allowed_domains" not in existing_columns:
        op.add_column("agents", sa.Column("allowed_domains", sa.JSON(), nullable=True))


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = sa.inspect(bind)

    tables = inspector.get_table_names()
    if "agents" not in tables:
        return

    existing_columns = [col["name"] for col in inspector.get_columns("agents")]

    if "allowed_domains" in existing_columns:
        op.drop_column("agents", "allowed_domains")
    if "widget_enabled" in existing_columns:
        op.drop_column("agents", "widget_enabled")
