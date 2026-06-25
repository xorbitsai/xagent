"""Add profile fields to users

Revision ID: bda204eb2630
Revises: 20260624_add_mcp_concurrency_config
Create Date: 2026-06-25 17:50:21.774271

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "bda204eb2630"
down_revision: Union[str, None] = "20260624_add_mcp_concurrency_config"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from alembic import context
    from sqlalchemy import inspect

    bind = context.get_bind()
    inspector = inspect(bind)
    existing_columns = {c["name"] for c in inspector.get_columns("users")}

    new_columns = [
        ("first_name", sa.String(100)),
        ("last_name", sa.String(100)),
        ("organization", sa.String(255)),
        ("country", sa.String(100)),
        ("phone", sa.String(50)),
    ]

    for col_name, col_type in new_columns:
        if col_name not in existing_columns:
            op.add_column("users", sa.Column(col_name, col_type, nullable=True))


def downgrade() -> None:
    for col_name in ["phone", "country", "organization", "last_name", "first_name"]:
        op.drop_column("users", col_name)
