"""Retain the resource snapshot and progress of upload compensation.

Revision ID: 20261005_uploaded_file_cleanup_manifest
Revises: 20261005_uploaded_file_cleanup_fences
"""

import sqlalchemy as sa
from alembic import op

revision = "20261005_uploaded_file_cleanup_manifest"
down_revision = "20261005_uploaded_file_cleanup_fences"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_context().as_sql:
        raise RuntimeError("Upload cleanup manifests require an online migration")
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("uploaded_files"):
        return
    columns = inspector.get_columns("uploaded_files")
    if not any(column["name"] == "cleanup_manifest" for column in columns):
        op.add_column(
            "uploaded_files", sa.Column("cleanup_manifest", sa.JSON(none_as_null=True))
        )


def downgrade() -> None:
    if op.get_context().as_sql:
        raise RuntimeError("Upload cleanup manifests require an online migration")
    if not sa.inspect(op.get_bind()).has_table("uploaded_files"):
        return
    uploads = sa.table("uploaded_files", sa.column("cleanup_manifest"))
    if (
        op.get_bind()
        .execute(
            sa.select(uploads.c.cleanup_manifest)
            .where(uploads.c.cleanup_manifest.isnot(None))
            .limit(1)
        )
        .first()
        is not None
    ):
        raise RuntimeError(
            "Finish pending upload cleanup before removing its manifests"
        )
    with op.batch_alter_table("uploaded_files") as batch:
        batch.drop_column("cleanup_manifest")
