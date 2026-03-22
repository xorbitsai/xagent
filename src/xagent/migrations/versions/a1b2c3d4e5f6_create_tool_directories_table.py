"""create tool_directories table

Revision ID: a1b2c3d4e5f6
Revises: 222f2073c886
Create Date: 2025-12-10 00:00:00.000000

"""

from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine.reflection import Inspector

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "be6f77416f06"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if table already exists
    existing_tables = inspector.get_table_names()
    if "tool_directories" in existing_tables:
        return  # Table already exists, skip migration

    # Get database dialect
    dialect_name = bind.dialect.name

    # Create tool_directories table with dialect-specific types
    exclude_patterns_type: Any
    datetime_type: Any
    bool_true: Any
    bool_false: Any
    timestamp_default: Any

    if dialect_name == "postgresql":
        exclude_patterns_type = postgresql.JSON(astext_type=sa.Text())
        datetime_type = sa.DateTime(timezone=True)
        bool_true = sa.text("true")
        bool_false = sa.text("false")
        timestamp_default = sa.text("now()")
    else:  # sqlite and others
        exclude_patterns_type = sa.Text()
        datetime_type = sa.DateTime()
        bool_true = sa.text("1")
        bool_false = sa.text("0")
        timestamp_default = sa.text("CURRENT_TIMESTAMP")

    op.create_table(
        "tool_directories",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False, unique=True),
        sa.Column("path", sa.String(length=500), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=bool_true),
        sa.Column("recursive", sa.Boolean(), nullable=False, server_default=bool_true),
        sa.Column("exclude_patterns", exclude_patterns_type, nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_valid", sa.Boolean(), nullable=False, server_default=bool_false),
        sa.Column("last_validated_at", datetime_type, nullable=True),
        sa.Column("validation_error", sa.Text(), nullable=True),
        sa.Column(
            "tool_count", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "created_at", datetime_type, server_default=timestamp_default, nullable=True
        ),
        sa.Column(
            "updated_at", datetime_type, server_default=timestamp_default, nullable=True
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes (we just created the table, so no need to check if they exist)
    op.create_index(
        op.f("ix_tool_directories_id"), "tool_directories", ["id"], unique=False
    )
    op.create_index(
        op.f("ix_tool_directories_name"), "tool_directories", ["name"], unique=True
    )
    op.create_index(
        op.f("ix_tool_directories_enabled"),
        "tool_directories",
        ["enabled"],
        unique=False,
    )


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if table exists before dropping
    existing_tables = inspector.get_table_names()
    if "tool_directories" in existing_tables:
        op.drop_index(
            op.f("ix_tool_directories_enabled"), table_name="tool_directories"
        )
        op.drop_index(op.f("ix_tool_directories_name"), table_name="tool_directories")
        op.drop_index(op.f("ix_tool_directories_id"), table_name="tool_directories")
        op.drop_table("tool_directories")
