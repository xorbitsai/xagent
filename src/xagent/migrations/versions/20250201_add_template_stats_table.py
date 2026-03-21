"""add template_stats table

Revision ID: 20250201_add_template_stats
Revises: f79da474c69d
Create Date: 2025-02-01 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "20250201_add_template_stats"
down_revision: Union[str, None] = "aaca07b20ea9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Check if table already exists (compatible with both SQLite and PostgreSQL)
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    existing_tables = inspector.get_table_names()
    if "template_stats" not in existing_tables:
        # Create template_stats table
        op.create_table(
            "template_stats",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("template_id", sa.String(length=200), nullable=False),
            sa.Column("views", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("likes", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("used_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        # Check if index exists before creating
        existing_indexes = (
            [idx["name"] for idx in inspector.get_indexes("template_stats")]
            if "template_stats" in existing_tables
            else []
        )
        if "ix_template_stats_template_id" not in existing_indexes:
            op.create_index(
                "ix_template_stats_template_id",
                "template_stats",
                ["template_id"],
                unique=True,
            )


def downgrade() -> None:
    op.drop_index("ix_template_stats_template_id", table_name="template_stats")
    op.drop_table("template_stats")
