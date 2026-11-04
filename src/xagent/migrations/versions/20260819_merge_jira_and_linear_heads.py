"""merge jira and linear mcp app heads

The Jira (#1447) and Linear MCP connector seed migrations were authored in
parallel PRs and both branched off 20260813_trace_json_columns_to_jsonb,
leaving alembic with two heads and making every ``alembic upgrade`` (and
``init_db``) fail with "Multiple heads are present". This no-op merge
revision joins them back into a single head.

Revision ID: 20260819_merge_jira_and_linear_heads
Revises: 20260818_seed_jira_mcp_app, 20260818_seed_linear_mcp_app
Create Date: 2026-08-19

"""

from typing import Sequence, Union

revision: str = "20260819_merge_jira_and_linear_heads"
down_revision: Union[str, Sequence[str], None] = (
    "20260818_seed_jira_mcp_app",
    "20260818_seed_linear_mcp_app",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
