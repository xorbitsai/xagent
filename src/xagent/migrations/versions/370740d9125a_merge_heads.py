"""merge heads

Revision ID: 370740d9125a
Revises: 20260909_actor_mcp_connections, a138278b9d0b
Create Date: 2026-09-09 19:14:27.764331

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "370740d9125a"
down_revision: Union[str, Sequence[str], None] = (
    "20260909_actor_mcp_connections",
    "a138278b9d0b",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
