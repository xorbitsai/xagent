"""Add custom_apis and user_custom_apis tables

Revision ID: 654adb358ecd
Revises: 20260410_add_filename_index
Create Date: 2026-04-17 19:02:01.310411

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "654adb358ecd"
down_revision: Union[str, None] = "20260410_add_filename_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if table already exists
    existing_tables = inspector.get_table_names()

    if "custom_apis" not in existing_tables:
        op.create_table(
            "custom_apis",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("name", sa.String(length=100), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("url", sa.String(length=500), nullable=True),
            sa.Column("method", sa.String(length=20), nullable=True),
            sa.Column("headers", sa.JSON(), nullable=True),
            sa.Column("env", sa.JSON(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("(CURRENT_TIMESTAMP)"),
                nullable=True,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("(CURRENT_TIMESTAMP)"),
                nullable=True,
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("name"),
        )
        op.create_index(op.f("ix_custom_apis_id"), "custom_apis", ["id"], unique=False)

    if "user_custom_apis" not in existing_tables:
        foreign_keys = [
            sa.ForeignKeyConstraint(
                ["custom_api_id"], ["custom_apis.id"], ondelete="CASCADE"
            )
        ]

        if "users" in existing_tables:
            foreign_keys.append(
                sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE")
            )

        op.create_table(
            "user_custom_apis",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("custom_api_id", sa.Integer(), nullable=False),
            sa.Column("is_owner", sa.Boolean(), nullable=False),
            sa.Column("can_edit", sa.Boolean(), nullable=False),
            sa.Column("can_delete", sa.Boolean(), nullable=False),
            sa.Column("is_shared", sa.Boolean(), nullable=False),
            sa.Column("is_active", sa.Boolean(), nullable=False),
            sa.Column("is_default", sa.Boolean(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("(CURRENT_TIMESTAMP)"),
                nullable=True,
            ),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            *foreign_keys,
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("user_id", "custom_api_id", name="uq_user_custom_apis"),
        )
        op.create_index(
            op.f("ix_user_custom_apis_id"), "user_custom_apis", ["id"], unique=False
        )


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)
    existing_tables = inspector.get_table_names()

    if "user_custom_apis" in existing_tables:
        op.drop_index(op.f("ix_user_custom_apis_id"), table_name="user_custom_apis")
        op.drop_table("user_custom_apis")

    if "custom_apis" in existing_tables:
        op.drop_index(op.f("ix_custom_apis_id"), table_name="custom_apis")
        op.drop_table("custom_apis")
