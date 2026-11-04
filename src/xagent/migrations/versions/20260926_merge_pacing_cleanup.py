"""Merge admission pacing and task cleanup obligation histories."""

revision = "20260926_merge_pacing_cleanup"
down_revision = (
    "20260925_merge_admission_pacing",
    "20260925_task_cleanup_obligations",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
