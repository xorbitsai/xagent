"""add the widget voice switch to agents and deployments

``agents.widget_voice_enabled`` and ``deployments.widget_voice_enabled`` are
non-null booleans with a server default of false, so every existing agent and
workforce deployment reads false: voice stays off until an administrator turns
it on.

Each table and column is checked before it is touched, so the upgrade does
nothing for a missing table and skips a column that already exists.

Revision ID: 20261114_add_widget_voice_enabled
Revises: 20261115_serving_generation
Create Date: 2026-11-14

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261114_add_widget_voice_enabled"
down_revision: Union[str, None] = "20261115_serving_generation"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLES = ("agents", "deployments")
COLUMN = "widget_voice_enabled"


def _columns(table: str) -> set[str] | None:
    """The column names of ``table`` now, or None without it; never cached."""
    inspector = sa.inspect(op.get_bind())
    if table not in inspector.get_table_names():
        return None
    return {column["name"] for column in inspector.get_columns(table)}


def upgrade() -> None:
    for table in TABLES:
        columns = _columns(table)
        if columns is None or COLUMN in columns:
            continue
        op.add_column(
            table,
            sa.Column(COLUMN, sa.Boolean(), nullable=False, server_default=sa.false()),
        )


def downgrade() -> None:
    for table in TABLES:
        if COLUMN in (_columns(table) or set()):
            op.drop_column(table, COLUMN)
