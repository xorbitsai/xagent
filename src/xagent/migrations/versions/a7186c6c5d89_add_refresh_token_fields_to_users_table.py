"""Add refresh_token fields to users table

Revision ID: a7186c6c5d89
Revises: b74d4cf2f479
Create Date: 2025-11-14 11:56:11.622532

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "a7186c6c5d89"
down_revision: Union[str, None] = "b74d4cf2f479"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if users table exists
    tables = inspector.get_table_names()
    if "users" not in tables:
        # Table doesn't exist yet, will be created by SQLAlchemy or a later migration
        return

    # Check which columns already exist
    existing_columns = [col["name"] for col in inspector.get_columns("users")]

    # Add refresh token fields to users table
    if "refresh_token" not in existing_columns:
        op.add_column(
            "users", sa.Column("refresh_token", sa.String(length=255), nullable=True)
        )
    if "refresh_token_expires_at" not in existing_columns:
        op.add_column(
            "users",
            sa.Column(
                "refresh_token_expires_at", sa.DateTime(timezone=True), nullable=True
            ),
        )


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if users table exists
    tables = inspector.get_table_names()
    if "users" not in tables:
        # Table doesn't exist yet, will be created by SQLAlchemy or a later migration
        return

    # Check which columns exist before dropping
    existing_columns = [col["name"] for col in inspector.get_columns("users")]

    # Remove refresh token fields from users table
    if "refresh_token_expires_at" in existing_columns:
        op.drop_column("users", "refresh_token_expires_at")
    if "refresh_token" in existing_columns:
        op.drop_column("users", "refresh_token")
