"""extend alembic_version column to support longer revision IDs

Revision ID: 0260431_extend_alembic_version
Revises: 441d4f5d399c
Create Date: 2026-03-31

This migration extends the version_num column in alembic_version table
from varchar(32) to varchar(255) to support longer revision IDs.

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "0260431_extend_alembic_version"
down_revision: Union[str, None] = "441d4f5d399c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)
    dialect_name = bind.dialect.name

    # Check if alembic_version table exists
    existing_tables = inspector.get_table_names()
    if "alembic_version" not in existing_tables:
        return

    # Check current version_num column length
    columns = {col["name"]: col for col in inspector.get_columns("alembic_version")}
    if "version_num" not in columns:
        return

    version_num_col = columns["version_num"]
    current_length = None

    # Get the character length for varchar type
    if hasattr(version_num_col.get("type"), "length"):
        current_length = version_num_col["type"].length

    # Only extend if it's currently varchar(32) or smaller
    if current_length and current_length < 255:
        if dialect_name == "postgresql":
            # PostgreSQL: Use ALTER COLUMN ... TYPE
            op.execute(
                "ALTER TABLE alembic_version ALTER COLUMN version_num TYPE varchar(255)"
            )
        elif dialect_name == "sqlite":
            # SQLite: Need to recreate table
            with op.batch_alter_table("alembic_version") as batch_op:
                batch_op.alter_column(
                    "version_num",
                    existing_type=sa.String(current_length),
                    type_=sa.String(255),
                )


def downgrade() -> None:
    # Revert to varchar(32) - this may fail if longer revision IDs exist
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)
    dialect_name = bind.dialect.name

    # Check if alembic_version table exists
    existing_tables = inspector.get_table_names()
    if "alembic_version" not in existing_tables:
        return

    # Check current version_num column length
    columns = {col["name"]: col for col in inspector.get_columns("alembic_version")}
    if "version_num" not in columns:
        return

    version_num_col = columns["version_num"]
    current_length = None

    if hasattr(version_num_col.get("type"), "length"):
        current_length = version_num_col["type"].length

    # Only revert if it's currently varchar(255)
    if current_length and current_length == 255:
        if dialect_name == "postgresql":
            # First check if any revision IDs are longer than 32 characters
            result = bind.execute(
                sa.text(
                    "SELECT COUNT(*) FROM alembic_version WHERE LENGTH(version_num) > 32"
                )
            )
            count = result.scalar()
            if count and count > 0:
                raise RuntimeError(
                    f"Cannot downgrade: {count} revision IDs are longer than 32 characters"
                )

            op.execute(
                "ALTER TABLE alembic_version ALTER COLUMN version_num TYPE varchar(32)"
            )
        elif dialect_name == "sqlite":
            with op.batch_alter_table("alembic_version") as batch_op:
                batch_op.alter_column(
                    "version_num",
                    existing_type=sa.String(255),
                    type_=sa.String(32),
                )
