"""add task computer runtime kind

Revision ID: 20260725_add_task_computer_runtime_kind
Revises: 20260724_seed_google_ads_mcp_app
Create Date: 2026-07-25 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260725_add_task_computer_runtime_kind"
down_revision: Union[str, None] = "20260724_seed_google_ads_mcp_app"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "tasks",
        sa.Column("computer_runtime_kind", sa.String(length=32), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("tasks", "computer_runtime_kind")
