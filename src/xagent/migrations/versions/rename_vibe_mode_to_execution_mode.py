"""rename_vibe_mode_to_execution_mode

Revision ID: 001_rename_vibe_mode_to_execution_mode
Revises: 654adb358ecd
Create Date: 2026-04-20 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "001_rename_vibe_mode_to_execution_mode"
down_revision: Union[str, None] = "654adb358ecd"
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

    # Check which columns exist
    existing_columns = [col["name"] for col in inspector.get_columns("tasks")]

    # Step 1: Add execution_mode column if it doesn't exist
    if "execution_mode" not in existing_columns:
        op.add_column(
            "tasks", sa.Column("execution_mode", sa.String(length=20), nullable=True)
        )

    # Step 2: Migrate data from vibe_mode to execution_mode
    # Only if vibe_mode column exists (it might have been already renamed in some environments)
    if "vibe_mode" in existing_columns:
        # Map old values to new values
        # "task" -> "balanced" (most everyday tasks)
        # "process" -> "think" (complex multi-step tasks)
        # NULL -> "balanced" (default)
        op.execute("""
            UPDATE tasks
            SET execution_mode = CASE
                WHEN vibe_mode = 'task' THEN 'balanced'
                WHEN vibe_mode = 'process' THEN 'think'
                ELSE 'balanced'  -- Default for NULL or unknown values
            END
        """)

        # Step 3: Drop the old vibe_mode column
        op.drop_column("tasks", "vibe_mode")
    else:
        # If vibe_mode column doesn't exist, set default values for execution_mode
        op.execute(
            "UPDATE tasks SET execution_mode = 'balanced' WHERE execution_mode IS NULL"
        )


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if tasks table exists
    tables = inspector.get_table_names()
    if "tasks" not in tables:
        return

    # Check which columns exist
    existing_columns = [col["name"] for col in inspector.get_columns("tasks")]

    # Step 1: Add vibe_mode column back
    if "vibe_mode" not in existing_columns:
        op.add_column(
            "tasks", sa.Column("vibe_mode", sa.String(length=20), nullable=True)
        )

    # Step 2: Migrate data back from execution_mode to vibe_mode
    if "execution_mode" in existing_columns:
        op.execute("""
            UPDATE tasks
            SET vibe_mode = CASE
                WHEN execution_mode = 'balanced' THEN 'task'
                WHEN execution_mode = 'think' THEN 'process'
                WHEN execution_mode = 'flash' THEN 'task'  -- Flash mode didn't exist before, map to task
                ELSE 'task'  -- Default
            END
        """)

        # Step 3: Drop the execution_mode column
        op.drop_column("tasks", "execution_mode")
