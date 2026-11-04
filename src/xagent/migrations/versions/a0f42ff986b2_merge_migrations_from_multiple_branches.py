"""merge migrations from multiple branches

Revision ID: a0f42ff986b2
Revises: 20260225_add_uploaded_files, 805d5a835b7b
Create Date: 2026-03-09 00:02:16.919015

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "a0f42ff986b2"
down_revision: Union[str, None] = ("20260225_add_uploaded_files", "805d5a835b7b")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
