"""merge Stripe and OAuth owner heads

Revision ID: 0108d2704fc1
Revises: 20260818_user_oauth_resource_owner, 20260818_seed_stripe_mcp_app
Create Date: 2026-08-22 00:23:13.727048

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "0108d2704fc1"
down_revision: Union[str, Sequence[str], None] = (
    "20260818_user_oauth_resource_owner",
    "20260818_seed_stripe_mcp_app",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
