"""merge user template relations and agent v2 migration heads

Revision ID: 20260515_merge_template_relations_and_agent_v2
Revises: 20260514_add_user_template_relations, 9f8d7e6c5b4a
Create Date: 2026-05-15
"""

from typing import Sequence

# revision identifiers, used by Alembic.
revision: str = "20260515_merge_template_relations_and_agent_v2"
down_revision: tuple[str, str] | None = (
    "20260514_add_user_template_relations",
    "9f8d7e6c5b4a",
)
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
