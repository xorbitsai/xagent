"""merge salesforce and jira mcp app migrations

Revision ID: 0b38b8d46e1c
Revises: 20260818_seed_jira_mcp_app, 20260818_seed_salesforce_mcp_app
Create Date: 2026-08-20 13:55:56.143163

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "0b38b8d46e1c"
down_revision: Union[str, Sequence[str], None] = (
    "20260818_seed_jira_mcp_app",
    "20260818_seed_salesforce_mcp_app",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
