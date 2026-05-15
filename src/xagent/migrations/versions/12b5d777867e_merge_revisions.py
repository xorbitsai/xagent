"""merge revisions

Revision ID: 12b5d777867e
Revises: 20260514_add_user_template_relations, 20260514_drop_delegate_agent_ids_from_tasks
Create Date: 2026-05-15 13:36:13.482129

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "12b5d777867e"
down_revision: Union[str, None] = (
    "20260514_add_user_template_relations",
    "20260514_drop_delegate_agent_ids_from_tasks",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
