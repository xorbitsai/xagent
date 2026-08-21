"""merge github+jira+linear and jira+posthog heads

Revision ID: b1efe0dbe0af
Revises: 20260820_merge_jira_posthog_heads, a4ad5a4e6441
Create Date: 2026-08-21 15:31:12.421371

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "b1efe0dbe0af"
down_revision: Union[str, None] = (
    "20260820_merge_jira_posthog_heads",
    "a4ad5a4e6441",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
