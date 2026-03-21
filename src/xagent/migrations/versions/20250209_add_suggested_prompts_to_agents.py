"""add suggested_prompts to agents table

Revision ID: 20250209_add_suggested_prompts
Revises: 20250209_add_execution_mode
Create Date: 2025-02-09 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20250209_add_suggested_prompts_to_agents"
down_revision: Union[str, None] = "20250209_add_execution_mode_to_agents"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add suggested_prompts column to agents table
    op.add_column("agents", sa.Column("suggested_prompts", sa.JSON(), nullable=True))


def downgrade() -> None:
    # Remove suggested_prompts column from agents table
    op.drop_column("agents", "suggested_prompts")
