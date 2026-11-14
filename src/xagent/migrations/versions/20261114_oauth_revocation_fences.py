"""Fence builtin OAuth grant revocations against concurrent callbacks.

Revision ID: 20261114_oauth_revocation_fences
Revises: 20261110_rename_atlassian_to_confluence
"""

import sqlalchemy as sa
from alembic import op

revision = "20261114_oauth_revocation_fences"
down_revision = "20261110_rename_atlassian_to_confluence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if not sa.inspect(op.get_bind()).has_table("oauth_revocation_fences"):
        op.create_table(
            "oauth_revocation_fences",
            sa.Column("provider", sa.String(50), primary_key=True, nullable=False),
            sa.Column(
                "provider_user_id", sa.String(), primary_key=True, nullable=False
            ),
            sa.Column("revoking_until", sa.DateTime(timezone=True), nullable=True),
        )


def downgrade() -> None:
    op.drop_table("oauth_revocation_fences")
