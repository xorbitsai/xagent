"""merge jira and posthog connector seed heads

The Jira and PostHog connector seed migrations both branched off
20260813_trace_json_columns_to_jsonb in separate parallel PRs, leaving
alembic with two heads and making every ``alembic upgrade`` (and
``init_db``) fail with "Multiple heads are present". This no-op merge
revision joins them back into a single head.

Revision ID: 20260820_merge_jira_posthog_heads
Revises: 20260818_seed_jira_mcp_app, 20260818_seed_posthog_mcp_app
Create Date: 2026-08-20

"""

from typing import Sequence, Union

revision: str = "20260820_merge_jira_posthog_heads"
down_revision: Union[str, Sequence[str], None] = (
    "20260818_seed_jira_mcp_app",
    "20260818_seed_posthog_mcp_app",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
