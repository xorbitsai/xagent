"""add agent widget end-user signing secret

Revision ID: 20260710_add_widget_end_user_secret
Revises: 1c2ae61b5a6d
Create Date: 2026-07-10 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260710_add_widget_end_user_secret"
down_revision: Union[str, tuple[str, str], None] = "1c2ae61b5a6d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "agents" not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns("agents")}
    if "widget_end_user_secret" not in existing_columns:
        op.add_column(
            "agents",
            sa.Column("widget_end_user_secret", sa.String(length=255), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "agents" not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns("agents")}
    if "widget_end_user_secret" in existing_columns:
        op.drop_column("agents", "widget_end_user_secret")
