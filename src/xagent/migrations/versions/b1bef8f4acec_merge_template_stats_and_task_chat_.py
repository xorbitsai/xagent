"""merge template_stats and task_chat_messages branches

Revision ID: b1bef8f4acec
Revises: 20250201_add_template_stats, 6ba8bead0889
Create Date: 2026-03-21 22:05:02.502273

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "b1bef8f4acec"
down_revision: Union[str, None] = ("20250201_add_template_stats", "6ba8bead0889")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
