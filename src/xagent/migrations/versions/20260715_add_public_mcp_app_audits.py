"""add durable public MCP app audit records

Revision ID: 20260715_add_public_mcp_app_audits
Revises: 20260715_normalize_builtin_mcp_launch
Create Date: 2026-07-15

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260715_add_public_mcp_app_audits"
down_revision: Union[str, None] = "20260715_normalize_builtin_mcp_launch"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


TABLE = "public_mcp_app_audits"
INDEXES: tuple[tuple[str, str], ...] = (
    ("ix_public_mcp_app_audits_actor_user_id", "actor_user_id"),
    ("ix_public_mcp_app_audits_action", "action"),
    ("ix_public_mcp_app_audits_app_id", "app_id"),
    ("ix_public_mcp_app_audits_request_id", "request_id"),
    ("ix_public_mcp_app_audits_created_at", "created_at"),
)


def upgrade() -> None:
    if not op.get_context().as_sql:
        inspector = sa.inspect(op.get_bind())
        tables = set(inspector.get_table_names())
        if TABLE in tables:
            return
        # Core application tables are created by SQLAlchemy after Alembic on
        # a fresh installation.  Skipping here lets Base.metadata.create_all()
        # create this table together with users and retain the declared FK.
        if "users" not in tables:
            return

    op.create_table(
        TABLE,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "actor_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("action", sa.String(16), nullable=False),
        sa.Column("app_id", sa.String(100), nullable=False),
        sa.Column("before_values", sa.JSON(), nullable=True),
        sa.Column("after_values", sa.JSON(), nullable=True),
        sa.Column("request_id", sa.String(128), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    for index_name, column_name in INDEXES:
        op.create_index(index_name, TABLE, [column_name], unique=False)


def downgrade() -> None:
    existing_indexes: set[str] | None = None
    if not op.get_context().as_sql:
        inspector = sa.inspect(op.get_bind())
        if TABLE not in inspector.get_table_names():
            return
        existing_indexes = {
            index["name"] for index in inspector.get_indexes(TABLE) if index["name"]
        }

    for index_name, _column_name in reversed(INDEXES):
        if existing_indexes is None or index_name in existing_indexes:
            op.drop_index(index_name, table_name=TABLE)
    op.drop_table(TABLE)
