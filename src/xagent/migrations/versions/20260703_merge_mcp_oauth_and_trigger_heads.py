"""merge MCP OAuth authorization and trigger provider foundation heads

Revision ID: 20260703_merge_mcp_oauth_and_trigger_heads
Revises: 20260702_add_mcp_oauth_tables, 20260702_add_trigger_provider_foundation
Create Date: 2026-07-03 00:00:00.000000

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "20260703_merge_mcp_oauth_and_trigger_heads"
down_revision: Union[str, tuple[str, str], None] = (
    "20260702_add_mcp_oauth_tables",
    "20260702_add_trigger_provider_foundation",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
