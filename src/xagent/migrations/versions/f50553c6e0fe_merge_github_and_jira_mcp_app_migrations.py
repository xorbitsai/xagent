"""merge github and jira mcp app migrations

Revision ID: f50553c6e0fe
Revises: 20260817_seed_github_mcp_app, 20260818_seed_jira_mcp_app
Create Date: 2026-08-19 17:36:01.846117

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "f50553c6e0fe"
down_revision: Union[str, None] = (
    "20260817_seed_github_mcp_app",
    "20260818_seed_jira_mcp_app",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
