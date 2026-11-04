from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260403_add_user_tool_configs"
down_revision: str | None = "594413e35640"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = inspector.get_table_names()

    if "tool_configs" in existing_tables:
        tool_config_columns = {
            column["name"] for column in inspector.get_columns("tool_configs")
        }
        if "requires_configuration" not in tool_config_columns:
            op.add_column(
                "tool_configs",
                sa.Column(
                    "requires_configuration",
                    sa.Boolean(),
                    nullable=False,
                    server_default=sa.false(),
                ),
            )

    if "user_tool_configs" not in existing_tables:
        if "users" in existing_tables:
            op.create_table(
                "user_tool_configs",
                sa.Column("id", sa.Integer(), nullable=False),
                sa.Column("user_id", sa.Integer(), nullable=False),
                sa.Column("tool_name", sa.String(length=100), nullable=False),
                sa.Column("config", sa.JSON(), nullable=True),
                sa.Column(
                    "created_at",
                    sa.DateTime(timezone=True),
                    server_default=sa.func.now(),
                ),
                sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
                sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
                sa.PrimaryKeyConstraint("id"),
                sa.UniqueConstraint("user_id", "tool_name", name="uq_user_tool_config"),
            )
        else:
            op.create_table(
                "user_tool_configs",
                sa.Column("id", sa.Integer(), nullable=False),
                sa.Column("user_id", sa.Integer(), nullable=False),
                sa.Column("tool_name", sa.String(length=100), nullable=False),
                sa.Column("config", sa.JSON(), nullable=True),
                sa.Column(
                    "created_at",
                    sa.DateTime(timezone=True),
                    server_default=sa.func.now(),
                ),
                sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
                sa.PrimaryKeyConstraint("id"),
                sa.UniqueConstraint("user_id", "tool_name", name="uq_user_tool_config"),
            )

    refreshed_tables = inspector.get_table_names()
    existing_indexes = (
        [idx["name"] for idx in inspector.get_indexes("user_tool_configs")]
        if "user_tool_configs" in refreshed_tables
        else []
    )
    if "ix_user_tool_configs_id" not in existing_indexes:
        op.create_index(
            op.f("ix_user_tool_configs_id"),
            "user_tool_configs",
            ["id"],
            unique=False,
        )
    if "ix_user_tool_configs_user_id" not in existing_indexes:
        op.create_index(
            op.f("ix_user_tool_configs_user_id"),
            "user_tool_configs",
            ["user_id"],
            unique=False,
        )
    if "ix_user_tool_configs_tool_name" not in existing_indexes:
        op.create_index(
            op.f("ix_user_tool_configs_tool_name"),
            "user_tool_configs",
            ["tool_name"],
            unique=False,
        )


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = inspector.get_table_names()

    if "user_tool_configs" in existing_tables:
        existing_indexes = [
            idx["name"] for idx in inspector.get_indexes("user_tool_configs")
        ]
        if "ix_user_tool_configs_tool_name" in existing_indexes:
            op.drop_index(
                op.f("ix_user_tool_configs_tool_name"), table_name="user_tool_configs"
            )
        if "ix_user_tool_configs_user_id" in existing_indexes:
            op.drop_index(
                op.f("ix_user_tool_configs_user_id"), table_name="user_tool_configs"
            )
        if "ix_user_tool_configs_id" in existing_indexes:
            op.drop_index(
                op.f("ix_user_tool_configs_id"), table_name="user_tool_configs"
            )
        op.drop_table("user_tool_configs")

    if "tool_configs" in existing_tables:
        tool_config_columns = {
            column["name"] for column in inspector.get_columns("tool_configs")
        }
        if "requires_configuration" in tool_config_columns:
            op.drop_column("tool_configs", "requires_configuration")
