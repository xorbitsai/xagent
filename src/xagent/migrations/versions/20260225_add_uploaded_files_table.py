from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

revision: str = "20260225_add_uploaded_files"
down_revision: Union[str, None] = "20250209_add_agent_id_to_tasks"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if table already exists
    existing_tables = inspector.get_table_names()
    if "uploaded_files" not in existing_tables:
        # Build foreign key constraints dynamically based on what tables exist
        # This handles running migrations from empty database
        foreign_keys = []

        # Only add task_id FK if tasks table exists
        if "tasks" in existing_tables:
            foreign_keys.append(
                sa.ForeignKeyConstraint(["task_id"], ["tasks.id"], ondelete="SET NULL")
            )

        # Only add user_id FK if users table exists
        if "users" in existing_tables:
            foreign_keys.append(
                sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE")
            )

        op.create_table(
            "uploaded_files",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("file_id", sa.String(length=36), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("task_id", sa.Integer(), nullable=True),
            sa.Column("filename", sa.String(length=512), nullable=False),
            sa.Column("storage_path", sa.String(length=2048), nullable=False),
            sa.Column("mime_type", sa.String(length=255), nullable=True),
            sa.Column("file_size", sa.Integer(), nullable=False, server_default="0"),
            sa.Column(
                "created_at", sa.DateTime(timezone=True), server_default=sa.func.now()
            ),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            *foreign_keys,
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("file_id"),
            sa.UniqueConstraint("storage_path"),
        )

    # Check and create indexes
    existing_indexes = (
        [idx["name"] for idx in inspector.get_indexes("uploaded_files")]
        if "uploaded_files" in existing_tables
        else []
    )
    if "ix_uploaded_files_id" not in existing_indexes:
        op.create_index(
            op.f("ix_uploaded_files_id"), "uploaded_files", ["id"], unique=False
        )
    if "ix_uploaded_files_file_id" not in existing_indexes:
        op.create_index(
            op.f("ix_uploaded_files_file_id"),
            "uploaded_files",
            ["file_id"],
            unique=False,
        )


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if table exists before dropping
    existing_tables = inspector.get_table_names()
    if "uploaded_files" in existing_tables:
        op.drop_index(op.f("ix_uploaded_files_file_id"), table_name="uploaded_files")
        op.drop_index(op.f("ix_uploaded_files_id"), table_name="uploaded_files")
        op.drop_table("uploaded_files")
