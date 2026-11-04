"""merge sharepoint mcp app and task input receipts migrations

Revision ID: 91899e1d97d3
Revises: 20260917_seed_sharepoint_mcp_app, 20260919_task_input_receipts
Create Date: 2026-09-20 11:12:27.745786

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "91899e1d97d3"
down_revision: Union[str, None] = (
    "20260917_seed_sharepoint_mcp_app",
    "20260919_task_input_receipts",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
