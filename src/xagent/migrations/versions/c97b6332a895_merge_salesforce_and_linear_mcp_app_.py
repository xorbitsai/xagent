"""merge salesforce and linear mcp app migrations

Revision ID: c97b6332a895
Revises: 0b38b8d46e1c, 20260819_merge_jira_and_linear_heads
Create Date: 2026-08-21 00:19:07.933437

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "c97b6332a895"
down_revision: Union[str, Sequence[str], None] = (
    "0b38b8d46e1c",
    "20260819_merge_jira_and_linear_heads",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
