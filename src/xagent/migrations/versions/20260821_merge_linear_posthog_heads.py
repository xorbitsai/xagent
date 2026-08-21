"""merge linear+jira and jira+posthog seed-head merges

Two independent PRs each added their own merge migration reconciling a
fork off the same jira-adjacent lineage: one joining jira+linear
(20260819_merge_jira_and_linear_heads), the other joining jira+posthog
(20260820_merge_jira_posthog_heads). Landing both leaves alembic with two
heads again, making every ``alembic upgrade`` (and ``init_db``) fail with
"Multiple heads are present". This no-op merge revision joins them back
into a single head.

Revision ID: 20260821_merge_linear_posthog_heads
Revises: 20260819_merge_jira_and_linear_heads, 20260820_merge_jira_posthog_heads
Create Date: 2026-08-21

"""

from typing import Sequence, Union

revision: str = "20260821_merge_linear_posthog_heads"
down_revision: Union[str, Sequence[str], None] = (
    "20260819_merge_jira_and_linear_heads",
    "20260820_merge_jira_posthog_heads",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
