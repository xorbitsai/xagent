"""Preserve claimed upload identity for late KB writers.

Revision ID: 20261005_uploaded_file_cleanup_fences
Revises: 20261002_hubspot_remote_mcp
"""

import sqlalchemy as sa
from alembic import op

revision = "20261005_uploaded_file_cleanup_fences"
down_revision = "20261002_hubspot_remote_mcp"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_context().as_sql:
        raise RuntimeError("Cleanup fences require an online migration")
    if not sa.inspect(op.get_bind()).has_table("uploaded_file_cleanup_fences"):
        op.create_table(
            "uploaded_file_cleanup_fences",
            sa.Column("file_id", sa.String(36), primary_key=True, nullable=False),
            sa.Column(
                "claimed_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
        )


def downgrade() -> None:
    if op.get_context().as_sql:
        raise RuntimeError("Cleanup fences require an online migration")
    table = sa.table("uploaded_file_cleanup_fences", sa.column("file_id"))
    if op.get_bind().execute(sa.select(table.c.file_id).limit(1)).first() is not None:
        raise RuntimeError(
            "Retain cleanup fences while reclaimed file identities exist"
        )
    op.drop_table("uploaded_file_cleanup_fences")
