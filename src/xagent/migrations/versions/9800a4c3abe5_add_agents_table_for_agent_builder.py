"""add agents table for agent builder

Revision ID: 9800a4c3abe5
Revises: 20250128_add_token_tracking
Create Date: 2026-01-31 22:43:13.553060

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "9800a4c3abe5"
down_revision: Union[str, None] = "20250128_add_token_tracking"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if table already exists
    existing_tables = inspector.get_table_names()
    if "agents" not in existing_tables:
        # Build table constraints dynamically based on what tables exist
        # This handles the case where migrations are run from empty database
        # and users table hasn't been created yet (will be created by SQLAlchemy)
        foreign_keys = []

        if "users" in existing_tables:
            foreign_keys.append(
                sa.ForeignKeyConstraint(
                    ["user_id"], ["users.id"], name="fk_agents_user_id_users"
                )
            )

        # Create agents table
        op.create_table(
            "agents",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("name", sa.String(length=200), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("instructions", sa.Text(), nullable=True),
            sa.Column("knowledge_bases", sa.JSON(), nullable=True),
            sa.Column("skills", sa.JSON(), nullable=True),
            sa.Column("tool_categories", sa.JSON(), nullable=True),
            sa.Column("logo_url", sa.String(length=500), nullable=True),
            sa.Column(
                "status",
                sa.Enum("draft", "published", "archived", name="agentstatus"),
                nullable=False,
            ),
            sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            *foreign_keys,
            sa.PrimaryKeyConstraint("id"),
        )

    # Check and create index
    existing_indexes = (
        [idx["name"] for idx in inspector.get_indexes("agents")]
        if "agents" in existing_tables
        else []
    )
    if "ix_agents_user_id" not in existing_indexes:
        op.create_index("ix_agents_user_id", "agents", ["user_id"])


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if table exists before dropping
    existing_tables = inspector.get_table_names()
    if "agents" in existing_tables:
        op.drop_table("agents")
