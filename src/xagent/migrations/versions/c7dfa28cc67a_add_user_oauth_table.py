"""add user_oauth table

Revision ID: c7dfa28cc67a
Revises: 44a6d3a54c35
Create Date: 2026-03-13 12:53:59.677483

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "c7dfa28cc67a"
down_revision: Union[str, None] = "44a6d3a54c35"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if table already exists
    existing_tables = inspector.get_table_names()
    if "user_oauth" not in existing_tables:
        # Build foreign key constraints dynamically based on what tables exist
        # This handles running migrations from empty database
        foreign_keys = []

        # Only add user_id FK if users table exists
        if "users" in existing_tables:
            foreign_keys.append(
                sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE")
            )

        op.create_table(
            "user_oauth",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("provider", sa.String(length=50), nullable=False),
            sa.Column("access_token", sa.String(), nullable=False),
            sa.Column("refresh_token", sa.String(), nullable=True),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("token_type", sa.String(length=50), nullable=True),
            sa.Column("scope", sa.String(), nullable=True),
            sa.Column("provider_user_id", sa.String(), nullable=True),
            sa.Column("email", sa.String(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("(CURRENT_TIMESTAMP)"),
                nullable=True,
            ),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            *foreign_keys,
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "user_id",
                "provider",
                "provider_user_id",
                name="uq_user_provider_account",
            ),
        )

    # Check and create index
    existing_indexes = (
        [idx["name"] for idx in inspector.get_indexes("user_oauth")]
        if "user_oauth" in existing_tables
        else []
    )
    if "ix_user_oauth_id" not in existing_indexes:
        op.create_index(op.f("ix_user_oauth_id"), "user_oauth", ["id"], unique=False)

    # Check if uploaded_files table exists before altering
    dialect_name = bind.dialect.name
    if "uploaded_files" in existing_tables:
        # Check if the index/constraint already exists
        existing_indexes = [
            idx["name"] for idx in inspector.get_indexes("uploaded_files")
        ]
        existing_constraints = [
            cons["name"] for cons in inspector.get_unique_constraints("uploaded_files")
        ]

        if dialect_name == "sqlite":
            # SQLite workaround for altering columns and constraints
            # Check if users table exists before attempting to alter uploaded_files
            # (uploaded_files may have FK to users, and batch_alter_table tries to reflect FKs)
            if "users" in existing_tables:
                with op.batch_alter_table("uploaded_files", schema=None) as batch_op:
                    batch_op.alter_column(
                        "id",
                        existing_type=sa.INTEGER(),
                        nullable=False,
                        autoincrement=True,
                    )
                    if "ix_uploaded_files_file_id" in existing_indexes:
                        batch_op.drop_index("ix_uploaded_files_file_id")
                    if (
                        "ix_uploaded_files_file_id" not in existing_indexes or True
                    ):  # Always recreate to ensure unique=True
                        batch_op.create_index(
                            batch_op.f("ix_uploaded_files_file_id"),
                            ["file_id"],
                            unique=True,
                        )
                    if "uq_uploaded_files_storage_path" not in existing_constraints:
                        batch_op.create_unique_constraint(
                            batch_op.f("uq_uploaded_files_storage_path"),
                            ["storage_path"],
                        )
        else:
            # For PostgreSQL and other databases, use native operations
            if "uq_uploaded_files_storage_path" not in existing_constraints:
                op.create_unique_constraint(
                    "uq_uploaded_files_storage_path", "uploaded_files", ["storage_path"]
                )
            # For PostgreSQL, we need to drop and recreate the index to change uniqueness
            if "ix_uploaded_files_file_id" in existing_indexes:
                op.drop_index("ix_uploaded_files_file_id", table_name="uploaded_files")
            op.create_index(
                "ix_uploaded_files_file_id", "uploaded_files", ["file_id"], unique=True
            )


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    existing_tables = inspector.get_table_names()

    # Check if uploaded_files table exists before reverting changes
    if "uploaded_files" in existing_tables:
        dialect_name = bind.dialect.name
        existing_constraints = [
            cons["name"] for cons in inspector.get_unique_constraints("uploaded_files")
        ]
        existing_indexes = [
            idx["name"] for idx in inspector.get_indexes("uploaded_files")
        ]

        if dialect_name == "sqlite":
            # SQLite workaround for reverting changes
            # Skip batch_alter_table if users table doesn't exist (FK reflection fails)
            if "users" in existing_tables:
                with op.batch_alter_table("uploaded_files", schema=None) as batch_op:
                    if "uq_uploaded_files_storage_path" in existing_constraints:
                        batch_op.drop_constraint(
                            batch_op.f("uq_uploaded_files_storage_path"), type_="unique"
                        )
                    if "ix_uploaded_files_file_id" in existing_indexes:
                        batch_op.drop_index(batch_op.f("ix_uploaded_files_file_id"))
                    batch_op.create_index(
                        "ix_uploaded_files_file_id", ["file_id"], unique=False
                    )
                    batch_op.alter_column(
                        "id",
                        existing_type=sa.INTEGER(),
                        nullable=True,
                        autoincrement=True,
                    )
        else:
            # For PostgreSQL and other databases, use native operations
            if "uq_uploaded_files_storage_path" in existing_constraints:
                op.drop_constraint(
                    "uq_uploaded_files_storage_path", "uploaded_files", type_="unique"
                )
            if "ix_uploaded_files_file_id" in existing_indexes:
                op.drop_index("ix_uploaded_files_file_id", table_name="uploaded_files")
            op.create_index(
                "ix_uploaded_files_file_id", "uploaded_files", ["file_id"], unique=False
            )

    # Check if user_oauth table exists before dropping
    if "user_oauth" in existing_tables:
        op.drop_index(op.f("ix_user_oauth_id"), table_name="user_oauth")
        op.drop_table("user_oauth")
