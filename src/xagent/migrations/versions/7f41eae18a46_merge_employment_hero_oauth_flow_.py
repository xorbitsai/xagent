"""merge employment hero oauth-flow-generation and myob mcp app migrations

Revision ID: 7f41eae18a46
Revises: 20260902_oauth_flow_generation, 20260903_seed_myob_mcp_app
Create Date: 2026-09-04 12:38:48.483555

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "7f41eae18a46"
down_revision: Union[str, Sequence[str], None] = (
    "20260902_oauth_flow_generation",
    "20260903_seed_myob_mcp_app",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
