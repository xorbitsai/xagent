"""merge salesforce and user_oauth resource_owner heads

Revision ID: a3b70c638cc3
Revises: 20260818_user_oauth_resource_owner, ae0d1cffeca6
Create Date: 2026-08-24 11:20:40.110386

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "a3b70c638cc3"
down_revision: Union[str, Sequence[str], None] = (
    "20260818_user_oauth_resource_owner",
    "ae0d1cffeca6",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
