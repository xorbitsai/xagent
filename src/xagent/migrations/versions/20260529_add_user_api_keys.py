"""add user personal api keys

Revision ID: 20260529_add_user_api_keys
Revises: 20260529_merge_email_reset_and_agent_origin_heads
Create Date: 2026-05-29 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260529_add_user_api_keys"
down_revision: Union[str, tuple[str, str], None] = (
    "20260529_merge_email_reset_and_agent_origin_heads"
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _index_names(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if table_name not in inspector.get_table_names():
        return set()
    return {index["name"] for index in inspector.get_indexes(table_name)}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = inspector.get_table_names()

    if "user_api_keys" not in existing_tables:
        constraints = [sa.PrimaryKeyConstraint("id")]
        if "users" in existing_tables:
            constraints.append(
                sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE")
            )

        op.create_table(
            "user_api_keys",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("key_prefix", sa.String(length=12), nullable=False),
            sa.Column("key_hash", sa.String(length=128), nullable=False),
            sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=False,
            ),
            *constraints,
        )

    existing_indexes = _index_names("user_api_keys")
    if "ix_user_api_keys_id" not in existing_indexes:
        op.create_index(op.f("ix_user_api_keys_id"), "user_api_keys", ["id"])
    if "ix_user_api_keys_user_id" not in existing_indexes:
        op.create_index(op.f("ix_user_api_keys_user_id"), "user_api_keys", ["user_id"])
    if "ix_user_api_keys_key_prefix" not in existing_indexes:
        op.create_index(
            op.f("ix_user_api_keys_key_prefix"),
            "user_api_keys",
            ["key_prefix"],
            unique=True,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "user_api_keys" not in inspector.get_table_names():
        return

    existing_indexes = _index_names("user_api_keys")
    if "ix_user_api_keys_key_prefix" in existing_indexes:
        op.drop_index(op.f("ix_user_api_keys_key_prefix"), table_name="user_api_keys")
    if "ix_user_api_keys_user_id" in existing_indexes:
        op.drop_index(op.f("ix_user_api_keys_user_id"), table_name="user_api_keys")
    if "ix_user_api_keys_id" in existing_indexes:
        op.drop_index(op.f("ix_user_api_keys_id"), table_name="user_api_keys")
    op.drop_table("user_api_keys")
