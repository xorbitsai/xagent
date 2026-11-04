"""merge agent_type and tool_directories branches

Revision ID: 253dd836197e
Revises: be6f77416f06, a1b2c3d4e5f6
Create Date: 2025-12-31 18:27:17.528244

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "253dd836197e"
down_revision: Union[str, tuple[str, ...], Sequence[str], None] = (
    "be6f77416f06",
    "a1b2c3d4e5f6",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
