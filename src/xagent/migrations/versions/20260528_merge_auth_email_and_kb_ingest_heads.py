"""merge auth email reset and kb ingest target heads

Revision ID: 20260528_merge_auth_email_and_kb_ingest_heads
Revises: 20260528_add_email_and_password_reset_fields_to_users, 20260528_add_kb_ingest_targets
Create Date: 2026-05-28 00:00:00.000000

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "20260528_merge_auth_email_and_kb_ingest_heads"
down_revision: Union[str, tuple[str, str], None] = (
    "20260528_add_email_and_password_reset_fields_to_users",
    "20260528_add_kb_ingest_targets",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
