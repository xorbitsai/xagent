"""Persist per-model OpenRouter endpoint allowlists.

The three nullable columns have no defaults, so existing rows read NULL and
PostgreSQL does not rewrite the table. PostgreSQL DDL uses a transaction-local
five-second lock timeout; Alembic runs each PostgreSQL revision in a separate
transaction. A failed lock acquisition rolls back the whole revision.

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

_LOCK_TIMEOUT_SQL = sa.text("SET LOCAL lock_timeout = '5s'")


def _set_lock_timeout() -> None:
    if op.get_context().dialect.name == "postgresql":
        op.execute(_LOCK_TIMEOUT_SQL)


def _columns() -> set[str] | None:
    inspector = sa.inspect(op.get_bind())
    if "models" not in inspector.get_table_names():
        return None
    return {column["name"] for column in inspector.get_columns("models")}


def upgrade() -> None:
    context = op.get_context()
    if context.as_sql:
        _set_lock_timeout()
        op.add_column(
            "models", sa.Column("openrouter_provider_only", sa.JSON(), nullable=True)
        )
        op.add_column(
            "models", sa.Column("openrouter_zdr", sa.Boolean(), nullable=True)
        )
        op.add_column(
            "models",
            sa.Column("openrouter_data_collection", sa.String(16), nullable=True),
        )
        return

    columns = _columns()
    if columns is None:
        return
    _set_lock_timeout()
    if "openrouter_provider_only" not in columns:
        op.add_column(
            "models",
            sa.Column("openrouter_provider_only", sa.JSON(), nullable=True),
        )
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
    context = op.get_context()
    if context.as_sql:
        _set_lock_timeout()
        op.drop_column("models", "openrouter_provider_only")
        op.drop_column("models", "openrouter_zdr")
        op.drop_column("models", "openrouter_data_collection")
        return

    columns = _columns()
    if columns is None:
        return
    _set_lock_timeout()
    if "openrouter_provider_only" in columns:
        op.drop_column("models", "openrouter_provider_only")
    if "openrouter_zdr" in columns:
        op.drop_column("models", "openrouter_zdr")
    if "openrouter_data_collection" in columns:
        op.drop_column("models", "openrouter_data_collection")
