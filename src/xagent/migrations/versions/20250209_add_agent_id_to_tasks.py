"""add agent_id to tasks table

Revision ID: 20250209_add_agent_id_to_tasks
Revises: 20250209_add_suggested_prompts
Create Date: 2025-02-09 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "20250209_add_agent_id_to_tasks"
down_revision: Union[str, None] = "20250209_add_suggested_prompts_to_agents"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if tasks table exists
    tables = inspector.get_table_names()
    if "tasks" not in tables:
        # Table doesn't exist yet, will be created by SQLAlchemy or a later migration
        return

    # Check if column already exists
    existing_columns = [col["name"] for col in inspector.get_columns("tasks")]
    if "agent_id" not in existing_columns:
        # Add agent_id column to tasks table
        op.add_column("tasks", sa.Column("agent_id", sa.Integer(), nullable=True))


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)
    dialect_name = bind.dialect.name

    # Check if tasks table exists
    tables = inspector.get_table_names()
    if "tasks" not in tables:
        # Table doesn't exist yet, will be created by SQLAlchemy or a later migration
        return

    # Check if column exists before dropping
    existing_columns = [col["name"] for col in inspector.get_columns("tasks")]
    if "agent_id" in existing_columns:
        # Remove agent_id column
        if dialect_name == "sqlite":
            # SQLite: use batch_alter_table
            with op.batch_alter_table("tasks", recreate="auto") as batch_op:
                batch_op.drop_column("agent_id")
        else:
            # PostgreSQL: drop column directly
            op.drop_column("tasks", "agent_id")
