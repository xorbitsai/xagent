"""Join task outcome and identity histories without rewriting either branch."""

revision = "20260930_merge_task_outcome_identity"
down_revision = (
    "20260930_task_completion_outcome",
    "20260930_task_identity",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
