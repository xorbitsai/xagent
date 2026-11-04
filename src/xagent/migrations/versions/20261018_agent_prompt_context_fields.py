"""add prompt_context_fields to agents

The column holds the declared prompt-context fields the unsigned widget entry
may pass: ``[{"name": str, "max_length": int, "description": str | null}]``.
Existing rows stay NULL and read as no declared fields.

Revision ID: 20261018_agent_prompt_context_fields
Revises: 20261019_zendesk_marketplace
Create Date: 2026-10-18 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261018_agent_prompt_context_fields"
down_revision: Union[str, None] = "20261019_zendesk_marketplace"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "agents"
COLUMN = "prompt_context_fields"


def _column_names(bind: sa.engine.Connection) -> set[str]:
    inspector = sa.inspect(bind)
    if TABLE not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns(TABLE)}


def upgrade() -> None:
    columns = _column_names(op.get_bind())
    if not columns or COLUMN in columns:
        return
    with op.batch_alter_table(TABLE) as batch_op:
        batch_op.add_column(sa.Column(COLUMN, sa.JSON(), nullable=True))


def downgrade() -> None:
    columns = _column_names(op.get_bind())
    if not columns or COLUMN not in columns:
        return
    with op.batch_alter_table(TABLE) as batch_op:
        batch_op.drop_column(COLUMN)
