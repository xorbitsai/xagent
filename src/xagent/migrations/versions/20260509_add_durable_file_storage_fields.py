from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

revision: str = "20260509_add_durable_file_storage_fields"
down_revision: Union[str, None] = "fab71cf4b1ad"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)
    if "uploaded_files" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("uploaded_files")}
    new_columns = [
        (
            "storage_backend",
            sa.Column("storage_backend", sa.String(length=64), nullable=True),
        ),
        (
            "storage_key",
            sa.Column("storage_key", sa.String(length=2048), nullable=True),
        ),
        (
            "storage_uri",
            sa.Column("storage_uri", sa.String(length=4096), nullable=True),
        ),
        ("checksum", sa.Column("checksum", sa.String(length=128), nullable=True)),
        ("etag", sa.Column("etag", sa.String(length=255), nullable=True)),
        (
            "workspace_relative_path",
            sa.Column("workspace_relative_path", sa.String(length=2048), nullable=True),
        ),
        (
            "workspace_category",
            sa.Column("workspace_category", sa.String(length=64), nullable=True),
        ),
        (
            "storage_status",
            sa.Column(
                "storage_status",
                sa.String(length=32),
                nullable=False,
                server_default="legacy",
            ),
        ),
    ]
    for name, column in new_columns:
        if name not in columns:
            op.add_column("uploaded_files", column)

    indexes = {index["name"] for index in inspector.get_indexes("uploaded_files")}
    if "ix_uploaded_files_storage_identity" not in indexes:
        op.create_index(
            "ix_uploaded_files_storage_identity",
            "uploaded_files",
            ["storage_backend", "storage_key"],
            unique=True,
        )


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)
    if "uploaded_files" not in inspector.get_table_names():
        return

    indexes = {index["name"] for index in inspector.get_indexes("uploaded_files")}
    if "ix_uploaded_files_storage_identity" in indexes:
        op.drop_index("ix_uploaded_files_storage_identity", table_name="uploaded_files")

    columns = {column["name"] for column in inspector.get_columns("uploaded_files")}
    for name in [
        "storage_status",
        "workspace_category",
        "workspace_relative_path",
        "etag",
        "checksum",
        "storage_uri",
        "storage_key",
        "storage_backend",
    ]:
        if name in columns:
            op.drop_column("uploaded_files", name)
