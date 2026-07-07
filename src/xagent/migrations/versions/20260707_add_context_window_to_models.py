"""add context_window field to models

Revision ID: 20260707_add_context_window
Revises: 20260707_merge_alembic_heads
Create Date: 2026-07-07

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "20260707_add_context_window"
down_revision: Union[str, None] = "20260707_merge_alembic_heads"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    tables = inspector.get_table_names()
    if "models" not in tables:
        return

    existing_columns = [col["name"] for col in inspector.get_columns("models")]
    if "context_window" not in existing_columns:
        op.add_column(
            "models", sa.Column("context_window", sa.Integer(), nullable=True)
        )


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    tables = inspector.get_table_names()
    if "models" not in tables:
        return

    existing_columns = [col["name"] for col in inspector.get_columns("models")]
    if "context_window" in existing_columns:
        op.drop_column("models", "context_window")
