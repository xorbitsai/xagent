"""Merge startup pacing with the current admission migration history."""

revision = "20260925_merge_admission_pacing"
down_revision = (
    "20260923_admission_pacing",
    "20260924_merge_admission_google",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
