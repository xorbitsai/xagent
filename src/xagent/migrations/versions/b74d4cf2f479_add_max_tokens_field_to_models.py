"""add max_tokens field to models

Revision ID: b74d4cf2f479
Revises: 441d4f5d399c
Create Date: 2025-11-03 21:09:44.186547

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "b74d4cf2f479"
down_revision: Union[str, None] = "0260431_extend_alembic_version"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if models table exists
    tables = inspector.get_table_names()
    if "models" not in tables:
        # Table doesn't exist yet, will be created by SQLAlchemy or a later migration
        return

    # Check if column already exists
    existing_columns = [col["name"] for col in inspector.get_columns("models")]
    if "max_tokens" not in existing_columns:
        op.add_column("models", sa.Column("max_tokens", sa.Integer(), nullable=True))


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if models table exists
    tables = inspector.get_table_names()
    if "models" not in tables:
        # Table doesn't exist, nothing to drop
        return

    # Check if column exists before dropping
    existing_columns = [col["name"] for col in inspector.get_columns("models")]
    if "max_tokens" in existing_columns:
        op.drop_column("models", "max_tokens")
