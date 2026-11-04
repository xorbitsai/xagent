"""merge connector runtime and context window migrations

Revision ID: 1c2ae61b5a6d
Revises: 20260706_add_connector_runtime_context, 20260707_add_context_window
Create Date: 2026-07-08 14:13:24.289563

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "1c2ae61b5a6d"
down_revision: Union[str, None] = (
    "20260706_add_connector_runtime_context",
    "20260707_add_context_window",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
