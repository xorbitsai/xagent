"""add preferences to users table

Revision ID: 20260823_add_preferences_to_users
Revises: a3b70c638cc3
Create Date: 2026-08-23 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260823_add_preferences_to_users"
# Repointed from 20260818_user_oauth_resource_owner (this migration's
# original parent) to a3b70c638cc3, the merge revision that later folded
# that same head together with the Salesforce connector's branch - this
# migration had not landed on main yet when that merge happened, so it's
# safe to just move its parent rather than add a second reconciling
# merge revision. a3b70c638cc3 already descends from
# 20260818_user_oauth_resource_owner, so nothing this migration needs is
# lost.
down_revision: Union[str, None] = "a3b70c638cc3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())

    if "users" not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns("users")}
    if "preferences" not in existing_columns:
        op.add_column("users", sa.Column("preferences", sa.JSON(), nullable=True))


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())

    if "users" not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns("users")}
    if "preferences" in existing_columns:
        op.drop_column("users", "preferences")
