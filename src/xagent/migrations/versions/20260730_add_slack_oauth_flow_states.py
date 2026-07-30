"""add slack oauth flow states table

Revision ID: 20260730_add_slack_oauth_flow_states
Revises: 20260725_add_uploaded_file_recovery_index
Create Date: 2026-07-30 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260730_add_slack_oauth_flow_states"
down_revision: Union[str, None] = "20260725_add_uploaded_file_recovery_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "slack_oauth_flow_states"


def _table_exists() -> bool:
    inspector = sa.inspect(op.get_bind())
    return TABLE in inspector.get_table_names()


def upgrade() -> None:
    if _table_exists():
        return
    op.create_table(
        TABLE,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("nonce", sa.String(length=64), nullable=False),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_slack_oauth_flow_states_nonce", TABLE, ["nonce"], unique=True)
    op.create_index("ix_slack_oauth_flow_states_user_id", TABLE, ["user_id"])
    op.create_index("ix_slack_oauth_flow_states_expires_at", TABLE, ["expires_at"])


def downgrade() -> None:
    if not _table_exists():
        return
    op.drop_index("ix_slack_oauth_flow_states_expires_at", table_name=TABLE)
    op.drop_index("ix_slack_oauth_flow_states_user_id", table_name=TABLE)
    op.drop_index("ix_slack_oauth_flow_states_nonce", table_name=TABLE)
    op.drop_table(TABLE)
