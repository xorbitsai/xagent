"""Merge admission and Freshdesk heads created by parallel PRs.

Both migrations descend from task_last_activity_at. Join their histories so
upgrading either deployed branch can reach a single head without rewriting it.
"""

revision = "20260923_merge_admission"
down_revision = (
    "20260923_seed_freshdesk_mcp_app",
    "20260923_task_admission",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
