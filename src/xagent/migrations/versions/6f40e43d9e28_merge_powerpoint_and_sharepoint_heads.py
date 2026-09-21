"""merge PowerPoint and SharePoint MCP app migration heads

Revision ID: 6f40e43d9e28
Revises: 20260917_seed_powerpoint_mcp_app, 91899e1d97d3
Create Date: 2026-09-20 16:00:00.000000

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "6f40e43d9e28"
down_revision: Union[str, Sequence[str], None] = (
    "20260917_seed_powerpoint_mcp_app",
    "91899e1d97d3",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
