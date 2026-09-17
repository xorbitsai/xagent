"""merge heads

Revision ID: a138278b9d0b
Revises: 20260901_seed_zendesk_mcp_app, 20260908_update_google_drive_description
Create Date: 2026-09-09 12:10:46.880437

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "a138278b9d0b"
down_revision: Union[str, Sequence[str], None] = (
    "20260901_seed_zendesk_mcp_app",
    "20260908_update_google_drive_description",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
