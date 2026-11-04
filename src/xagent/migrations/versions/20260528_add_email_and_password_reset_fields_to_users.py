"""add email and password reset fields to users

Revision ID: 20260528_add_email_and_password_reset_fields_to_users
Revises: 20260526_seed_builtin_microsoft_graph_mcp_apps
Create Date: 2026-05-28 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "20260528_add_email_and_password_reset_fields_to_users"
down_revision: Union[str, None] = "20260526_seed_builtin_microsoft_graph_mcp_apps"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _get_existing_indexes(inspector: Inspector, table_name: str) -> set[str]:
    return {index["name"] for index in inspector.get_indexes(table_name)}


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    if "users" not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns("users")}
    existing_indexes = _get_existing_indexes(inspector, "users")

    if "email" not in existing_columns:
        op.add_column("users", sa.Column("email", sa.String(length=255), nullable=True))
    if "password_reset_token_hash" not in existing_columns:
        op.add_column(
            "users",
            sa.Column("password_reset_token_hash", sa.String(length=64), nullable=True),
        )
    if "password_reset_expires_at" not in existing_columns:
        op.add_column(
            "users",
            sa.Column(
                "password_reset_expires_at", sa.DateTime(timezone=True), nullable=True
            ),
        )

    if "ix_users_email" not in existing_indexes:
        op.create_index("ix_users_email", "users", ["email"], unique=True)
    if "ix_users_password_reset_token_hash" not in existing_indexes:
        op.create_index(
            "ix_users_password_reset_token_hash",
            "users",
            ["password_reset_token_hash"],
            unique=False,
        )


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    if "users" not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns("users")}
    existing_indexes = _get_existing_indexes(inspector, "users")

    if "ix_users_password_reset_token_hash" in existing_indexes:
        op.drop_index("ix_users_password_reset_token_hash", table_name="users")
    if "ix_users_email" in existing_indexes:
        op.drop_index("ix_users_email", table_name="users")

    if "password_reset_expires_at" in existing_columns:
        op.drop_column("users", "password_reset_expires_at")
    if "password_reset_token_hash" in existing_columns:
        op.drop_column("users", "password_reset_token_hash")
    if "email" in existing_columns:
        op.drop_column("users", "email")
