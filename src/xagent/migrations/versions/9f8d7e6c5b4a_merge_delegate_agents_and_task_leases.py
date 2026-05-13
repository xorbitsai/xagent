"""merge delegate agent ids and task lease branches

Revision ID: 9f8d7e6c5b4a
Revises: 20260509_add_delegate_agent_ids_to_tasks, 7f4d2c9a1b58
Create Date: 2026-05-12 00:00:00.000000

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "9f8d7e6c5b4a"
down_revision: Union[str, tuple[str, ...], None] = (
    "20260509_add_delegate_agent_ids_to_tasks",
    "7f4d2c9a1b58",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
