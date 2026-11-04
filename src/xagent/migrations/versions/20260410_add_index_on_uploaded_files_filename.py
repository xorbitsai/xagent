"""Add index on uploaded_files.filename for web ingestion URL deduplication.

This migration adds an index on the filename column of the uploaded_files table
to optimize URL deduplication queries during web ingestion. Without this index,
each URL deduplication check requires a full table scan, which becomes a bottleneck
at scale (>1000 pages).

Revision ID: 20260410_add_filename_index
Revises: 12605616ed6f
Create Date: 2026-04-10
"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy.engine.reflection import Inspector

revision: str = "20260410_add_filename_index"
down_revision: Union[str, None] = "12605616ed6f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if table exists
    existing_tables = inspector.get_table_names()
    if "uploaded_files" not in existing_tables:
        return

    # Check if index already exists
    existing_indexes = [idx["name"] for idx in inspector.get_indexes("uploaded_files")]

    # Create index on filename column if it doesn't exist
    if "ix_uploaded_files_filename" not in existing_indexes:
        op.create_index(
            op.f("ix_uploaded_files_filename"),
            "uploaded_files",
            ["filename"],
            unique=False,
        )


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if table and index exist before dropping
    existing_tables = inspector.get_table_names()
    if "uploaded_files" not in existing_tables:
        return

    existing_indexes = [idx["name"] for idx in inspector.get_indexes("uploaded_files")]
    if "ix_uploaded_files_filename" in existing_indexes:
        op.drop_index(op.f("ix_uploaded_files_filename"), table_name="uploaded_files")
