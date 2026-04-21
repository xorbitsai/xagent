"""update_agent_execution_mode_to_new_values

Revision ID: 002_update_agent_execution_mode
Revises: 001_rename_vibe_mode_to_execution_mode
Create Date: 2026-04-20 12:30:00.000000

"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "002_update_agent_execution_mode"
down_revision: Union[str, None] = "001_rename_vibe_mode_to_execution_mode"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if agents table exists
    tables = inspector.get_table_names()
    if "agents" not in tables:
        return

    # Check which columns exist
    existing_columns = [col["name"] for col in inspector.get_columns("agents")]

    if "execution_mode" in existing_columns:
        # Update existing agents' execution_mode values
        # Map old values to new values
        # "simple" -> "flash" (though simple wasn't really used)
        # "react" -> "balanced"
        # "graph" -> "think"
        op.execute("""
            UPDATE agents
            SET execution_mode = CASE
                WHEN execution_mode = 'simple' THEN 'flash'
                WHEN execution_mode = 'react' THEN 'balanced'
                WHEN execution_mode = 'graph' THEN 'think'
                WHEN execution_mode IN ('flash', 'balanced', 'think') THEN execution_mode  -- Already new values
                ELSE 'balanced'  -- Default for NULL or unknown values
            END
        """)


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if agents table exists
    tables = inspector.get_table_names()
    if "agents" not in tables:
        return

    # Check which columns exist
    existing_columns = [col["name"] for col in inspector.get_columns("agents")]

    if "execution_mode" in existing_columns:
        # Revert back to old values
        op.execute("""
            UPDATE agents
            SET execution_mode = CASE
                WHEN execution_mode = 'flash' THEN 'simple'
                WHEN execution_mode = 'balanced' THEN 'react'
                WHEN execution_mode = 'think' THEN 'graph'
                ELSE 'react'  -- Default
            END
        """)
