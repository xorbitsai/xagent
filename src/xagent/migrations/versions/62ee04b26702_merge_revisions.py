"""merge revisions

Revision ID: 62ee04b26702
Revises: 9c4a2d1e8f2b, b1bef8f4acec
Create Date: 2026-03-23 19:37:36.028811

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "62ee04b26702"
down_revision: Union[str, None] = ("9c4a2d1e8f2b", "b1bef8f4acec")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
