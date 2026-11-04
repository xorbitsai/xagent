"""merge public mcp visibility and chat message attachments heads

Revision ID: 20260521_merge_alembic_heads
Revises: 20260518_add_chat_message_attachments, 20260519_add_public_mcp_visibility
Create Date: 2026-05-21 00:00:00.000000

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "20260521_merge_alembic_heads"
down_revision: Union[str, tuple[str, str], None] = (
    "20260518_add_chat_message_attachments",
    "20260519_add_public_mcp_visibility",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
