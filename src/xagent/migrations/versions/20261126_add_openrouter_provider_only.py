"""Persist per-model OpenRouter endpoint allowlists.

Revision ID: 20261126_openrouter_routing
Revises: 20261119_shopify_store_locks
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261126_openrouter_routing"
down_revision: Union[str, None] = "20261119_shopify_store_locks"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if "models" not in inspector.get_table_names():
        return
    if "openrouter_provider_only" not in {
        column["name"] for column in inspector.get_columns("models")
    }:
        op.add_column(
            "models",
            sa.Column("openrouter_provider_only", sa.JSON(), nullable=True),
        )
    columns = {
        column["name"] for column in sa.inspect(op.get_bind()).get_columns("models")
    }
    if "openrouter_zdr" not in columns:
        op.add_column(
            "models", sa.Column("openrouter_zdr", sa.Boolean(), nullable=True)
        )
    if "openrouter_data_collection" not in columns:
        op.add_column(
            "models",
            sa.Column("openrouter_data_collection", sa.String(16), nullable=True),
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if "models" not in inspector.get_table_names():
        return
    if "openrouter_provider_only" in {
        column["name"] for column in inspector.get_columns("models")
    }:
        op.drop_column("models", "openrouter_provider_only")
    columns = {
        column["name"] for column in sa.inspect(op.get_bind()).get_columns("models")
    }
    if "openrouter_zdr" in columns:
        op.drop_column("models", "openrouter_zdr")
    if "openrouter_data_collection" in columns:
        op.drop_column("models", "openrouter_data_collection")
