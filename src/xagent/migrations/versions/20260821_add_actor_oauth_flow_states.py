"""add minimal actor OAuth flow nonce table

Revision ID: 20260821_actor_oauth_flow_states
Revises: 20260826_seed_deputy_mcp_app
Create Date: 2026-08-21
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260821_actor_oauth_flow_states"
down_revision: Union[str, None] = "20260826_seed_deputy_mcp_app"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "actor_oauth_flow_states"


def upgrade() -> None:
    """Create the nonce-only state consumed by trusted actor callbacks."""
    context = op.get_context()
    if not context.as_sql and sa.inspect(op.get_bind()).has_table(TABLE):
        return

    op.create_table(
        TABLE,
        sa.Column("nonce", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("nonce", name="pk_actor_oauth_flow_states"),
    )


def downgrade() -> None:
    """Remove actor OAuth flow state after entry points have been disabled."""
    context = op.get_context()
    if not context.as_sql and not sa.inspect(op.get_bind()).has_table(TABLE):
        return

    op.drop_table(TABLE)
