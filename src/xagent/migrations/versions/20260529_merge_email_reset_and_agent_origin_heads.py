"""merge email reset and agent origin heads

Revision ID: 20260529_merge_email_reset_and_agent_origin_heads
Revises: 20260528_merge_auth_email_and_kb_ingest_heads, 20260529_add_agent_origin
Create Date: 2026-05-29 00:00:00.000000

"""

from typing import Sequence, Union

revision: str = "20260529_merge_email_reset_and_agent_origin_heads"
down_revision: Union[str, tuple[str, str], None] = (
    "20260528_merge_auth_email_and_kb_ingest_heads",
    "20260529_add_agent_origin",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
