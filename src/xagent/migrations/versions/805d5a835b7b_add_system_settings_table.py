"""add system settings table

Revision ID: 805d5a835b7b
Revises: 20250209_add_agent_id_to_tasks
Create Date: 2026-02-25 22:05:00.965921

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "805d5a835b7b"
down_revision: Union[str, None] = "20250209_add_agent_id_to_tasks"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if table already exists
    existing_tables = inspector.get_table_names()
    if "system_settings" not in existing_tables:
        op.create_table(
            "system_settings",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("key", sa.String(length=128), nullable=False),
            sa.Column("value", sa.Text(), nullable=False),
            sa.Column(
                "created_at", sa.DateTime(timezone=True), server_default=sa.func.now()
            ),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("key"),
        )

    # Check and create indexes
    existing_indexes = (
        [idx["name"] for idx in inspector.get_indexes("system_settings")]
        if "system_settings" in existing_tables
        else []
    )
    if "ix_system_settings_id" not in existing_indexes:
        op.create_index(
            op.f("ix_system_settings_id"), "system_settings", ["id"], unique=False
        )
    if "ix_system_settings_key" not in existing_indexes:
        op.create_index(
            op.f("ix_system_settings_key"), "system_settings", ["key"], unique=False
        )


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if table exists before dropping
    existing_tables = inspector.get_table_names()
    if "system_settings" in existing_tables:
        op.drop_index(op.f("ix_system_settings_key"), table_name="system_settings")
        op.drop_index(op.f("ix_system_settings_id"), table_name="system_settings")
        op.drop_table("system_settings")
