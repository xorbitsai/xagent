"""add consumed OIDC token table

Revision ID: 20260529_add_oidc_consumed_tokens
Revises: 20260529_add_user_api_keys, 20260526_add_user_identities
Create Date: 2026-05-29 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260529_add_oidc_consumed_tokens"
down_revision: Union[str, tuple[str, str], None] = (
    "20260529_add_user_api_keys",
    "20260526_add_user_identities",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "oidc_consumed_tokens" in inspector.get_table_names():
        return

    op.create_table(
        "oidc_consumed_tokens",
        sa.Column("token_id", sa.String(length=96), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("token_id"),
    )
    op.create_index(
        "ix_oidc_consumed_tokens_expires_at",
        "oidc_consumed_tokens",
        ["expires_at"],
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "oidc_consumed_tokens" not in inspector.get_table_names():
        return

    op.drop_index(
        "ix_oidc_consumed_tokens_expires_at",
        table_name="oidc_consumed_tokens",
    )
    op.drop_table("oidc_consumed_tokens")
