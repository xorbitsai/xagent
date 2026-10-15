"""Merge the OAuth token encryption and Xero MCP application heads."""

from typing import Sequence, Union

revision: str = "20261015_merge_oauth_and_xero_heads"
down_revision: Union[str, Sequence[str], None] = (
    "20261014_encrypt_user_oauth_tokens",
    "20261015_sync_xero_mcp_app",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
