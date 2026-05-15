"""Add user template relations table

Revision ID: 20260514_add_user_template_relations
Revises: 9f8d7e6c5b4a
Create Date: 2026-05-14
"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260514_add_user_template_relations"
down_revision: str | None = "9f8d7e6c5b4a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


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

    if "user_template_relations" not in existing_tables:
        constraints = [
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "user_id",
                "template_id",
                "relation_type",
                name="uq_user_template_relation",
            ),
        ]
        if "users" in existing_tables:
            constraints.append(
                sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE")
            )

        op.create_table(
            "user_template_relations",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("template_id", sa.String(length=200), nullable=False),
            sa.Column("relation_type", sa.String(length=50), nullable=False),
            sa.Column(
                "is_active",
                sa.Boolean(),
                nullable=False,
                server_default=sa.true(),
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            *constraints,
        )

    existing_indexes = _index_names("user_template_relations")
    if "ix_user_template_relations_user_id" not in existing_indexes:
        op.create_index(
            op.f("ix_user_template_relations_user_id"),
            "user_template_relations",
            ["user_id"],
            unique=False,
        )
    if "ix_user_template_relations_template_id" not in existing_indexes:
        op.create_index(
            op.f("ix_user_template_relations_template_id"),
            "user_template_relations",
            ["template_id"],
            unique=False,
        )
    if "ix_user_template_relations_relation_type" not in existing_indexes:
        op.create_index(
            op.f("ix_user_template_relations_relation_type"),
            "user_template_relations",
            ["relation_type"],
            unique=False,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "user_template_relations" not in inspector.get_table_names():
        return

    existing_indexes = _index_names("user_template_relations")
    if "ix_user_template_relations_relation_type" in existing_indexes:
        op.drop_index(
            op.f("ix_user_template_relations_relation_type"),
            table_name="user_template_relations",
        )
    if "ix_user_template_relations_template_id" in existing_indexes:
        op.drop_index(
            op.f("ix_user_template_relations_template_id"),
            table_name="user_template_relations",
        )
    if "ix_user_template_relations_user_id" in existing_indexes:
        op.drop_index(
            op.f("ix_user_template_relations_user_id"),
            table_name="user_template_relations",
        )

    op.drop_table("user_template_relations")
