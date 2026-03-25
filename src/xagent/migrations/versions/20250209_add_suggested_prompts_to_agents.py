"""add suggested_prompts to agents table

Revision ID: 20250209_add_suggested_prompts
Revises: 20250209_add_execution_mode
Create Date: 2025-02-09 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "20250209_add_suggested_prompts_to_agents"
down_revision: Union[str, None] = "20250209_add_execution_mode_to_agents"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if agents table exists
    tables = inspector.get_table_names()
    if "agents" not in tables:
        # Table doesn't exist yet, will be created by SQLAlchemy or a later migration
        return

    # Check if column already exists
    existing_columns = [col["name"] for col in inspector.get_columns("agents")]
    if "suggested_prompts" not in existing_columns:
        # Add suggested_prompts column to agents table
        op.add_column(
            "agents", sa.Column("suggested_prompts", sa.JSON(), nullable=True)
        )


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if agents table exists
    tables = inspector.get_table_names()
    if "agents" not in tables:
        # Table doesn't exist yet, will be created by SQLAlchemy or a later migration
        return

    # Check if column exists before dropping
    existing_columns = [col["name"] for col in inspector.get_columns("agents")]
    if "suggested_prompts" in existing_columns:
        # Remove suggested_prompts column from agents table
        op.drop_column("agents", "suggested_prompts")
