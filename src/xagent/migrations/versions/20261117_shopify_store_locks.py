"""Lock one Shopify store's token grants and refreshes against each other.

Revision ID: 20261117_shopify_store_locks
Revises: 20261116_browser_idle_session
"""

import sqlalchemy as sa
from alembic import op

revision = "20261117_shopify_store_locks"
down_revision = "20261116_browser_idle_session"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if not sa.inspect(op.get_bind()).has_table("shopify_store_locks"):
        op.create_table(
            "shopify_store_locks",
            sa.Column("provider", sa.String(50), primary_key=True, nullable=False),
            sa.Column("store", sa.String(255), primary_key=True, nullable=False),
        )


def downgrade() -> None:
    op.drop_table("shopify_store_locks")
