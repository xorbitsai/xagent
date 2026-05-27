"""add kb background ingest target generations

Revision ID: 20260528_add_kb_ingest_targets
Revises: 20260521_add_background_jobs, 20260526_seed_builtin_microsoft_graph_mcp_apps
Create Date: 2026-05-28 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260528_add_kb_ingest_targets"
down_revision: Union[str, tuple[str, str], None] = (
    "20260521_add_background_jobs",
    "20260526_seed_builtin_microsoft_graph_mcp_apps",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _index_names(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if table_name not in inspector.get_table_names():
        return set()
    return {index["name"] for index in inspector.get_indexes(table_name)}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = inspector.get_table_names()

    if "kb_ingest_targets" not in existing_tables:
        constraints = [
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "user_id",
                "collection",
                "target_path",
                name="uq_kb_ingest_targets_user_collection_path",
            ),
        ]
        if "users" in existing_tables:
            constraints.append(
                sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE")
            )
        if "background_jobs" in existing_tables:
            constraints.append(
                sa.ForeignKeyConstraint(
                    ["latest_job_id"], ["background_jobs.id"], ondelete="SET NULL"
                )
            )

        op.create_table(
            "kb_ingest_targets",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("collection", sa.String(length=255), nullable=False),
            sa.Column("target_path", sa.String(length=2048), nullable=False),
            sa.Column("file_id", sa.String(length=36), nullable=False),
            sa.Column("latest_generation_id", sa.String(length=36), nullable=False),
            sa.Column("latest_job_id", sa.String(length=36), nullable=True),
            sa.Column("latest_file_sha256", sa.String(length=64), nullable=False),
            sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
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
            *constraints,
        )

    existing_indexes = _index_names("kb_ingest_targets")
    if "ix_kb_ingest_targets_id" not in existing_indexes:
        op.create_index(
            op.f("ix_kb_ingest_targets_id"),
            "kb_ingest_targets",
            ["id"],
            unique=False,
        )
    if "ix_kb_ingest_targets_user_collection" not in existing_indexes:
        op.create_index(
            "ix_kb_ingest_targets_user_collection",
            "kb_ingest_targets",
            ["user_id", "collection"],
            unique=False,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "kb_ingest_targets" not in inspector.get_table_names():
        return

    existing_indexes = _index_names("kb_ingest_targets")
    if "ix_kb_ingest_targets_user_collection" in existing_indexes:
        op.drop_index(
            "ix_kb_ingest_targets_user_collection",
            table_name="kb_ingest_targets",
        )
    if "ix_kb_ingest_targets_id" in existing_indexes:
        op.drop_index(op.f("ix_kb_ingest_targets_id"), table_name="kb_ingest_targets")

    op.drop_table("kb_ingest_targets")
