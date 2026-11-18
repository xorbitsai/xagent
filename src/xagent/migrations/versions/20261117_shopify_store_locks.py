"""Lock one Shopify store's token grants and refreshes against each other.

Revision ID: 20261117_shopify_store_locks
Revises: 20261116_shopify_oauth
"""

import sqlalchemy as sa
from alembic import op

revision = "20261117_shopify_store_locks"
down_revision = "20261116_shopify_oauth"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Offline SQL generation cannot inspect: it emits the CREATE TABLE.
    if not op.get_context().as_sql and sa.inspect(op.get_bind()).has_table(
        "shopify_store_locks"
    ):
        return
    op.create_table(
        "shopify_store_locks",
        sa.Column("provider", sa.String(50), primary_key=True, nullable=False),
        sa.Column("store", sa.String(255), primary_key=True, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("shopify_store_locks")
