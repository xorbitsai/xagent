"""Merge admission and Google Drive visibility migration histories.

This no-op merge preserves both parent histories while restoring a single head.
"""

revision = "20260924_merge_admission_google"
down_revision = (
    "20260923_merge_admission",
    "20260924_hide_google_drive_until_picker",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
