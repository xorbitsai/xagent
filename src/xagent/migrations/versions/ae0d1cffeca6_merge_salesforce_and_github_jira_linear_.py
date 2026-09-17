"""merge salesforce and github/jira/linear mcp app migrations

Revision ID: ae0d1cffeca6
Revises: b1efe0dbe0af, c97b6332a895
Create Date: 2026-08-21 18:01:02.044646

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "ae0d1cffeca6"
down_revision: Union[str, Sequence[str], None] = (
    "b1efe0dbe0af",
    "c97b6332a895",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
