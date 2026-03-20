"""merge migrations

Revision ID: 6ba8bead0889
Revises: 20260317_add_task_chat_messages, c7dfa28cc67a
Create Date: 2026-03-21 01:48:57.683885

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "6ba8bead0889"
down_revision: Union[str, None] = ("20260317_add_task_chat_messages", "c7dfa28cc67a")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
