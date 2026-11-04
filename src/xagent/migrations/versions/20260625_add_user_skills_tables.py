"""add user_skills and user_skill_files tables

Revision ID: 20260625_add_user_skills_tables
Revises: 20260702_add_mcp_oauth_tables, 20260702_add_trigger_provider_foundation
Create Date: 2026-06-25 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

revision: str = "20260625_add_user_skills_tables"
down_revision: Union[str, Sequence[str], None] = (
    "20260702_add_mcp_oauth_tables",
    "20260702_add_trigger_provider_foundation",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)
    existing_tables = inspector.get_table_names()

    if "user_skills" not in existing_tables and "users" in existing_tables:
        op.create_table(
            "user_skills",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("name", sa.String(length=100), nullable=False),
            sa.Column(
                "origin", sa.String(length=50), nullable=False, server_default="custom"
            ),
            sa.Column("clawhub_slug", sa.String(length=128), nullable=True),
            sa.Column("clawhub_version", sa.String(length=64), nullable=True),
            sa.Column("skill_metadata", sa.JSON(), nullable=True),
            sa.Column("created_by_user_id", sa.Integer(), nullable=True),
            sa.Column("updated_by_user_id", sa.Integer(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=True,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=True,
            ),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["created_by_user_id"], ["users.id"]),
            sa.ForeignKeyConstraint(["updated_by_user_id"], ["users.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("user_id", "name", name="uq_user_skill_name"),
        )
        op.create_index("ix_user_skills_id", "user_skills", ["id"], unique=False)
        op.create_index(
            "ix_user_skills_user_id", "user_skills", ["user_id"], unique=False
        )

    if "user_skill_files" not in existing_tables and "user_skills" in existing_tables:
        op.create_table(
            "user_skill_files",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("skill_id", sa.Integer(), nullable=False),
            sa.Column("path", sa.String(length=500), nullable=False),
            sa.Column("content", sa.LargeBinary(), nullable=False),
            sa.Column("size_bytes", sa.Integer(), nullable=False),
            sa.Column("sha256", sa.String(length=64), nullable=False),
            sa.Column("media_type", sa.String(length=100), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=True,
            ),
            sa.ForeignKeyConstraint(
                ["skill_id"], ["user_skills.id"], ondelete="CASCADE"
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("skill_id", "path", name="uq_user_skill_file_path"),
        )
        op.create_index(
            "ix_user_skill_files_id", "user_skill_files", ["id"], unique=False
        )
        op.create_index(
            "ix_user_skill_files_skill_id",
            "user_skill_files",
            ["skill_id"],
            unique=False,
        )


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)
    existing_tables = inspector.get_table_names()

    if "user_skill_files" in existing_tables:
        op.drop_index("ix_user_skill_files_skill_id", table_name="user_skill_files")
        op.drop_index("ix_user_skill_files_id", table_name="user_skill_files")
        op.drop_table("user_skill_files")

    if "user_skills" in existing_tables:
        op.drop_index("ix_user_skills_user_id", table_name="user_skills")
        op.drop_index("ix_user_skills_id", table_name="user_skills")
        op.drop_table("user_skills")
