"""add user identities table for OIDC login

Revision ID: 20260526_add_user_identities
Revises: 20260521_merge_alembic_heads
Create Date: 2026-05-26 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260526_add_user_identities"
down_revision: Union[str, None] = "20260526_seed_builtin_microsoft_graph_mcp_apps"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from alembic import context
    from sqlalchemy import inspect

    bind = context.get_bind()
    inspector = inspect(bind)
    existing_tables = inspector.get_table_names()

    if "user_identities" not in existing_tables:
        foreign_keys = []
        if "users" in existing_tables:
            foreign_keys.append(
                sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE")
            )

        op.create_table(
            "user_identities",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False, index=True),
            sa.Column("provider", sa.String(length=50), nullable=False, index=True),
            sa.Column("provider_subject", sa.String(length=255), nullable=False),
            sa.Column("email", sa.String(length=320), nullable=True),
            sa.Column("email_verified", sa.Boolean(), nullable=False),
            sa.Column("display_name", sa.String(length=255), nullable=True),
            sa.Column("picture_url", sa.String(length=1000), nullable=True),
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
            *foreign_keys,
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "provider",
                "provider_subject",
                name="uq_user_identity_provider_subject",
            ),
        )


def downgrade() -> None:
    from alembic import context
    from sqlalchemy import inspect

    bind = context.get_bind()
    inspector = inspect(bind)
    if "user_identities" not in inspector.get_table_names():
        return

    op.drop_table("user_identities")
