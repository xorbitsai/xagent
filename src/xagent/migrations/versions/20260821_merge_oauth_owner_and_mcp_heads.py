"""Merge the OAuth-owner and Jira/Linear MCP migration heads.

The owner-aware OAuth migration and the Jira/Linear MCP merge revision were
created on parallel branches. This no-op merge revision joins those branches
so Alembic retains one upgrade head after both changes land.

Revision ID: 20260821_merge_oauth_owner_and_mcp_heads
Revises: 20260818_user_oauth_resource_owner, 20260819_merge_jira_and_linear_heads
Create Date: 2026-08-21

"""

from typing import Sequence, Union

revision: str = "20260821_merge_oauth_owner_and_mcp_heads"
down_revision: Union[str, Sequence[str], None] = (
    "20260818_user_oauth_resource_owner",
    "20260819_merge_jira_and_linear_heads",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
