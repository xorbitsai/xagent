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
    from sqlalchemy.engine.reflection import Inspector

    bind = op.get_bind()
    inspector = Inspector.from_engine(bind)

    if "users" not in inspector.get_table_names():
        return

    existing_columns = {c["name"] for c in inspector.get_columns("users")}

    new_columns = [
        ("first_name", sa.String(100)),
        ("last_name", sa.String(100)),
        ("organization", sa.String(255)),
        ("country", sa.String(100)),
        ("phone", sa.String(50)),
    ]

    with op.batch_alter_table("users") as batch_op:
        for col_name, col_type in new_columns:
            if col_name not in existing_columns:
                batch_op.add_column(sa.Column(col_name, col_type, nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("users") as batch_op:
        for col_name in ["phone", "country", "organization", "last_name", "first_name"]:
            batch_op.drop_column(col_name)
