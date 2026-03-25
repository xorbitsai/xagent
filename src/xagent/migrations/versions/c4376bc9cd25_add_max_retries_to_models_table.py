"""Add max_retries to models table

Revision ID: c4376bc9cd25
Revises: 3713a331796f
Create Date: 2026-01-16 13:40:04.536821

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "c4376bc9cd25"
down_revision: Union[str, None] = "3713a331796f"
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
    if "max_retries" not in existing_columns:
        op.add_column("models", sa.Column("max_retries", sa.Integer(), nullable=True))


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if models table exists
    tables = inspector.get_table_names()
    if "models" not in tables:
        # Table doesn't exist yet, will be created by SQLAlchemy or a later migration
        return

    # Check if column exists before dropping
    existing_columns = [col["name"] for col in inspector.get_columns("models")]
    if "max_retries" in existing_columns:
        op.drop_column("models", "max_retries")
