"""add slack oauth flow states table

Revision ID: 20260730_add_slack_oauth_flow_states
Revises: 20260728_add_facebook_pages_read_user_content_scope
Create Date: 2026-07-30 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260730_add_slack_oauth_flow_states"
down_revision: Union[str, None] = "20260728_add_facebook_pages_read_user_content_scope"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "slack_oauth_flow_states"


def _existing_tables() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    return set(inspector.get_table_names())


def upgrade() -> None:
    existing_tables = _existing_tables()
    if TABLE in existing_tables:
        return
    constraints: list[sa.schema.SchemaItem] = [sa.PrimaryKeyConstraint("id")]
    # The base schema (users etc.) is created outside this migration chain in
    # some environments; only declare the FK when the table is present.
    if "users" in existing_tables:
        constraints.append(
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE")
        )
    op.create_table(
        TABLE,
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("nonce", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        *constraints,
    )
    op.create_index("ix_slack_oauth_flow_states_nonce", TABLE, ["nonce"], unique=True)
    op.create_index("ix_slack_oauth_flow_states_user_id", TABLE, ["user_id"])
    op.create_index("ix_slack_oauth_flow_states_expires_at", TABLE, ["expires_at"])


def downgrade() -> None:
    if TABLE not in _existing_tables():
        return
    op.drop_index("ix_slack_oauth_flow_states_expires_at", table_name=TABLE)
    op.drop_index("ix_slack_oauth_flow_states_user_id", table_name=TABLE)
    op.drop_index("ix_slack_oauth_flow_states_nonce", table_name=TABLE)
    op.drop_table(TABLE)
