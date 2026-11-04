"""add gmail watch states

Revision ID: 20260629_add_gmail_watch_states
Revises: 20260629_backfill_external_conversation_logs
Create Date: 2026-06-29 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260629_add_gmail_watch_states"
down_revision: Union[str, None] = "20260629_backfill_external_conversation_logs"
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

    if "gmail_watch_states" not in existing_tables:
        constraints = [
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("oauth_account_id"),
        ]
        if "users" in existing_tables:
            constraints.append(
                sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE")
            )
        if "user_oauth" in existing_tables:
            constraints.append(
                sa.ForeignKeyConstraint(
                    ["oauth_account_id"], ["user_oauth.id"], ondelete="CASCADE"
                )
            )

        op.create_table(
            "gmail_watch_states",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("oauth_account_id", sa.Integer(), nullable=False),
            sa.Column("email", sa.String(length=255), nullable=False),
            sa.Column("history_id", sa.String(length=255), nullable=False),
            sa.Column("watch_expiration", sa.DateTime(timezone=True), nullable=True),
            sa.Column("topic_name", sa.String(length=512), nullable=False),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=True,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=True,
            ),
            *constraints,
        )

    for index_name, columns in (
        ("ix_gmail_watch_states_id", ["id"]),
        ("ix_gmail_watch_states_user_id", ["user_id"]),
        ("ix_gmail_watch_states_oauth_account_id", ["oauth_account_id"]),
        ("ix_gmail_watch_states_email", ["email"]),
        ("ix_gmail_watch_states_watch_expiration", ["watch_expiration"]),
    ):
        if (
            "gmail_watch_states" in inspector.get_table_names()
            and index_name not in _index_names("gmail_watch_states")
        ):
            op.create_index(index_name, "gmail_watch_states", columns, unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "gmail_watch_states" not in inspector.get_table_names():
        return

    for index_name in (
        "ix_gmail_watch_states_watch_expiration",
        "ix_gmail_watch_states_email",
        "ix_gmail_watch_states_oauth_account_id",
        "ix_gmail_watch_states_user_id",
        "ix_gmail_watch_states_id",
    ):
        if index_name in _index_names("gmail_watch_states"):
            op.drop_index(index_name, table_name="gmail_watch_states")
    op.drop_table("gmail_watch_states")
