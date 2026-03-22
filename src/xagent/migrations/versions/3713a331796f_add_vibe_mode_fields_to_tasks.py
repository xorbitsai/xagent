"""add_vibe_mode_fields_to_tasks

Revision ID: 3713a331796f
Revises: a47ef367a4f3
Create Date: 2026-01-05 17:21:52.069669

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "3713a331796f"
down_revision: Union[str, None] = "253dd836197e"
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

    # Check which columns already exist
    existing_columns = [col["name"] for col in inspector.get_columns("tasks")]

    # Add vibe_mode column to tasks table
    if "vibe_mode" not in existing_columns:
        op.add_column(
            "tasks", sa.Column("vibe_mode", sa.String(length=20), nullable=True)
        )
        # Update existing rows to have 'task' as default vibe_mode
        op.execute("UPDATE tasks SET vibe_mode = 'task' WHERE vibe_mode IS NULL")

    # Add process_description column to tasks table
    if "process_description" not in existing_columns:
        op.add_column(
            "tasks", sa.Column("process_description", sa.Text(), nullable=True)
        )

    # Add examples column to tasks table
    if "examples" not in existing_columns:
        op.add_column("tasks", sa.Column("examples", sa.JSON(), nullable=True))


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if tasks table exists
    tables = inspector.get_table_names()
    if "tasks" not in tables:
        # Table doesn't exist yet, will be created by SQLAlchemy or a later migration
        return

    # Check which columns exist before dropping
    existing_columns = [col["name"] for col in inspector.get_columns("tasks")]

    # Remove examples column
    if "examples" in existing_columns:
        op.drop_column("tasks", "examples")

    # Remove process_description column
    if "process_description" in existing_columns:
        op.drop_column("tasks", "process_description")

    # Remove vibe_mode column
    if "vibe_mode" in existing_columns:
        op.drop_column("tasks", "vibe_mode")
