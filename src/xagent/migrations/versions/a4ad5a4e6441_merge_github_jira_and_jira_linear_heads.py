"""merge github+jira and jira+linear heads

Revision ID: a4ad5a4e6441
Revises: 20260819_merge_jira_and_linear_heads, f50553c6e0fe
Create Date: 2026-08-21 00:18:57.154161

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "a4ad5a4e6441"
down_revision: Union[str, None] = (
    "20260819_merge_jira_and_linear_heads",
    "f50553c6e0fe",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
