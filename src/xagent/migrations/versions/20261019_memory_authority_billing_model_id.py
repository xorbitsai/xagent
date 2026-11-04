"""Add billing_model_id to the global memory embedding authority.

The model id memory embedding usage is billed under. Nullable, with no
default: an existing authority keeps recording its usage with an empty model
id until an operator sets one. The table holds at most one row.

Revision ID: 20261019_memory_authority_billing
Revises: 20261018_agent_prompt_context_fields
Create Date: 2026-10-19

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

revision: str = "20261019_memory_authority_billing"
down_revision: Union[str, None] = "20261018_agent_prompt_context_fields"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLE = "global_memory_embedding_authority"
_COLUMN = "billing_model_id"


def _columns() -> set[str] | None:
    inspector = Inspector.from_engine(op.get_bind())
    if _TABLE not in inspector.get_table_names():
        return None
    return {column["name"] for column in inspector.get_columns(_TABLE)}


def upgrade() -> None:
    columns = _columns()
    if columns is None or _COLUMN in columns:
        return
    op.add_column(_TABLE, sa.Column(_COLUMN, sa.String(100), nullable=True))


def downgrade() -> None:
    columns = _columns()
    if columns is None or _COLUMN not in columns:
        return
    with op.batch_alter_table(_TABLE) as batch_op:
        batch_op.drop_column(_COLUMN)
