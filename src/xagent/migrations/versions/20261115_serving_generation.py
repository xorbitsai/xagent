"""Add the serving generation record.

One row names the backend generation that may claim new singleton work while
two generations overlap during a release. Expand-only: a new table, nothing else.

Revision ID: 20261115_serving_generation
Revises: 20261114_oauth_revocation_fences
"""

import sqlalchemy as sa
from alembic import op

revision = "20261115_serving_generation"
down_revision = "20261114_oauth_revocation_fences"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Offline SQL generation cannot inspect: it emits the CREATE TABLE.
    if not op.get_context().as_sql and sa.inspect(op.get_bind()).has_table(
        "serving_generation"
    ):
        return
    op.create_table(
        "serving_generation",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("generation", sa.String(length=255), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint("id = 1", name="ck_serving_generation_single_row"),
    )


def downgrade() -> None:
    op.drop_table("serving_generation")
