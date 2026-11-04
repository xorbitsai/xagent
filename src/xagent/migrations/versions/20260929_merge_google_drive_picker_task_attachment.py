"""Merge the Google Drive Picker and current task migration branches.

Revision ID: 20260929_merge_google_drive_picker_task_attachment
Revises: 20260924_enable_google_drive_picker, 20260930_merge_task_outcome_identity
"""

from collections.abc import Sequence

revision: str = "20260929_merge_google_drive_picker_task_attachment"
down_revision: tuple[str, str] = (
    "20260924_enable_google_drive_picker",
    "20260930_merge_task_outcome_identity",
)
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
